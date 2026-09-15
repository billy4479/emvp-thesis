//! Backend-agnostic orchestration for answering several encrypted matrices.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use rayon::prelude::*;

use crate::dispatch::{AnswerBackend, select_cpu_backend};
use crate::{
    AnswerMatrix, EmvpParams, EncryptedMatrix, EncryptedQuery, ProtocolError, answer_batch,
};

static NEXT_ENGINE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_MATRIX_ID: AtomicU64 = AtomicU64::new(1);

/// An answer runner and the resources shared by its prepared matrices.
pub struct AnswerEngine<const MODULUS: u32> {
    id: u64,
}

/// A matrix validated and retained for repeated answer operations.
#[derive(Clone)]
pub struct PreparedMatrix<const MODULUS: u32> {
    inner: Arc<PreparedMatrixInner<MODULUS>>,
}

struct PreparedMatrixInner<const MODULUS: u32> {
    id: u64,
    engine_id: u64,
    params: EmvpParams,
    matrix: EncryptedMatrix<MODULUS>,
}

/// One entry in a multi-matrix answer operation.
pub struct AnswerJob<'a, const MODULUS: u32> {
    /// The prepared matrix to evaluate.
    pub matrix: &'a PreparedMatrix<MODULUS>,
    /// The encrypted queries to answer, in output order.
    pub queries: &'a [EncryptedQuery<MODULUS>],
}

/// Why a multi-matrix operation was rejected.
#[derive(Debug)]
#[non_exhaustive]
pub enum AnswerEngineError {
    /// The operation contained no jobs.
    EmptyJobs,
    /// One job contained no queries.
    EmptyQueries { entry: usize },
    /// A prepared matrix occurred more than once.
    RepeatedMatrix { entry: usize },
    /// A matrix was prepared by another engine.
    ForeignMatrix { entry: usize },
    /// Protocol validation or computation failed for one entry.
    Protocol { entry: usize, source: ProtocolError },
}

impl fmt::Display for AnswerEngineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyJobs => formatter.write_str("multi-matrix answer request is empty"),
            Self::EmptyQueries { entry } => {
                write!(formatter, "answer entry {entry} has no queries")
            }
            Self::RepeatedMatrix { entry } => {
                write!(formatter, "answer entry {entry} repeats a prepared matrix")
            }
            Self::ForeignMatrix { entry } => {
                write!(formatter, "answer entry {entry} belongs to another engine")
            }
            Self::Protocol { entry, source } => write!(formatter, "answer entry {entry}: {source}"),
        }
    }
}

impl std::error::Error for AnswerEngineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Protocol { source, .. } => Some(source),
            _ => None,
        }
    }
}

/// Execution statistics for one input entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnswerEntryReport {
    /// The CPU tier selected for this entry.
    pub backend: AnswerBackend,
    /// Number of queries in the entry.
    pub queries: usize,
    /// Estimated field multiplications.
    pub multiplications: usize,
    /// Number of query field words.
    pub query_words: usize,
    /// Number of answer field words.
    pub answer_words: usize,
    /// Number of GPU segments; zero on a CPU engine.
    pub gpu_segments: usize,
}

/// Wall-clock and shape statistics for one multi-matrix operation.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AnswerReport {
    /// Reports in input order.
    pub entries: Vec<AnswerEntryReport>,
    /// Entries evaluated on the CPU.
    pub cpu_entries: usize,
    /// Entries evaluated on the GPU.
    pub gpu_entries: usize,
    /// Sum of all estimated field multiplications.
    pub multiplications: usize,
    /// Time spent validating and planning.
    pub planning: Duration,
    /// Time spent evaluating CPU entries.
    pub cpu_compute: Duration,
    /// End-to-end wall time.
    pub total: Duration,
}

impl<const MODULUS: u32> AnswerEngine<MODULUS> {
    /// Creates an engine that evaluates every matrix on the CPU.
    #[must_use]
    pub fn cpu() -> Self {
        Self {
            id: NEXT_ENGINE_ID.fetch_add(1, Ordering::Relaxed),
        }
    }

    /// Validates and retains an encrypted matrix for repeated evaluation.
    ///
    /// # Errors
    ///
    /// Returns a protocol error when the parameters and matrix disagree.
    pub fn prepare(
        &self,
        params: EmvpParams,
        matrix: EncryptedMatrix<MODULUS>,
    ) -> Result<PreparedMatrix<MODULUS>, ProtocolError> {
        params.validate_dimensions()?;
        let n = params.n()?;
        if matrix.columns() != n {
            return Err(ProtocolError::LengthMismatch {
                name: "encrypted matrix columns",
                expected: n,
                actual: matrix.columns(),
            });
        }
        if matrix.rows() == 0 {
            return Err(ProtocolError::LengthMismatch {
                name: "matrix rows",
                expected: 1,
                actual: 0,
            });
        }
        let expected = matrix
            .rows()
            .checked_mul(n)
            .ok_or(ProtocolError::DimensionOverflow)?;
        if matrix.values().len() != expected {
            return Err(ProtocolError::LengthMismatch {
                name: "encrypted matrix values",
                expected,
                actual: matrix.values().len(),
            });
        }
        Ok(PreparedMatrix {
            inner: Arc::new(PreparedMatrixInner {
                id: NEXT_MATRIX_ID.fetch_add(1, Ordering::Relaxed),
                engine_id: self.id,
                params,
                matrix,
            }),
        })
    }

    /// Answers all jobs and returns answer groups in input order.
    ///
    /// # Errors
    ///
    /// Returns an indexed validation or protocol failure without returning
    /// partial results.
    pub fn answer_many(
        &self,
        jobs: &[AnswerJob<'_, MODULUS>],
    ) -> Result<Vec<Vec<AnswerMatrix<MODULUS>>>, AnswerEngineError> {
        self.answer_many_with_report(jobs)
            .map(|(answers, _)| answers)
    }

    /// Answers all jobs and returns answers plus execution statistics.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::answer_many`].
    pub fn answer_many_with_report(
        &self,
        jobs: &[AnswerJob<'_, MODULUS>],
    ) -> Result<(Vec<Vec<AnswerMatrix<MODULUS>>>, AnswerReport), AnswerEngineError> {
        let total_start = Instant::now();
        let planning_start = Instant::now();
        if jobs.is_empty() {
            return Err(AnswerEngineError::EmptyJobs);
        }
        let mut seen = HashSet::with_capacity(jobs.len());
        let threads = rayon::current_num_threads();
        let mut entries = Vec::with_capacity(jobs.len());
        for (entry, job) in jobs.iter().enumerate() {
            if job.matrix.inner.engine_id != self.id {
                return Err(AnswerEngineError::ForeignMatrix { entry });
            }
            if !seen.insert(job.matrix.inner.id) {
                return Err(AnswerEngineError::RepeatedMatrix { entry });
            }
            if job.queries.is_empty() {
                return Err(AnswerEngineError::EmptyQueries { entry });
            }
            let inner = &job.matrix.inner;
            let n = inner.matrix.columns();
            for query in job.queries {
                if query.values().len() != n {
                    return Err(AnswerEngineError::Protocol {
                        entry,
                        source: ProtocolError::LengthMismatch {
                            name: "encrypted query",
                            expected: n,
                            actual: query.values().len(),
                        },
                    });
                }
                if query.instance_id() != inner.matrix.instance_id() {
                    return Err(AnswerEngineError::Protocol {
                        entry,
                        source: ProtocolError::InstanceMismatch {
                            name: "encrypted query",
                            expected: inner.matrix.instance_id(),
                            actual: query.instance_id(),
                        },
                    });
                }
            }
            let answer_words = inner
                .matrix
                .rows()
                .checked_mul(inner.params.blocks().map_err(|source| {
                    AnswerEngineError::Protocol {
                        entry,
                        source: source.into(),
                    }
                })?)
                .and_then(|words| words.checked_mul(job.queries.len()))
                .ok_or(AnswerEngineError::Protocol {
                    entry,
                    source: ProtocolError::DimensionOverflow,
                })?;
            let multiplications = job
                .queries
                .len()
                .saturating_mul(inner.matrix.rows())
                .saturating_mul(n);
            entries.push(AnswerEntryReport {
                backend: select_cpu_backend(job.queries.len(), inner.matrix.rows(), n, threads),
                queries: job.queries.len(),
                multiplications,
                query_words: job.queries.len().saturating_mul(n),
                answer_words,
                gpu_segments: 0,
            });
        }
        let planning = planning_start.elapsed();
        let cpu_start = Instant::now();
        let computed: Vec<Result<Vec<_>, AnswerEngineError>> = jobs
            .par_iter()
            .enumerate()
            .map(|(entry, job)| {
                let inner = &job.matrix.inner;
                answer_batch(&inner.params, &inner.matrix, job.queries)
                    .map_err(|source| AnswerEngineError::Protocol { entry, source })
            })
            .collect();
        let cpu_compute = cpu_start.elapsed();
        let answers: Result<Vec<_>, _> = computed.into_iter().collect();
        let answers = answers?;
        let multiplications = entries.iter().fold(0_usize, |total, entry| {
            total.saturating_add(entry.multiplications)
        });
        let report = AnswerReport {
            cpu_entries: entries.len(),
            gpu_entries: 0,
            multiplications,
            entries,
            planning,
            cpu_compute,
            total: total_start.elapsed(),
        };
        Ok((answers, report))
    }
}

#[cfg(test)]
mod tests {
    use prime_field_layer::PrimeField;

    use super::*;

    const MODULUS: u32 = 1_073_479_681;
    const PARAMS: EmvpParams = EmvpParams {
        k: 8,
        ell: 8,
        b: 2,
        lambda: 7,
    };

    fn matrix(instance_id: u128, rows: usize) -> EncryptedMatrix<MODULUS> {
        let zero = PrimeField::<MODULUS>::new().element_u32(0);
        EncryptedMatrix::from_parts(instance_id, rows, 16, vec![zero; rows * 16]).unwrap()
    }

    fn query(instance_id: u128, query_id: u64) -> EncryptedQuery<MODULUS> {
        let one = PrimeField::<MODULUS>::new().element_u32(1);
        EncryptedQuery::from_parts(instance_id, query_id, vec![one; 16])
    }

    #[test]
    fn cpu_engine_preserves_entry_and_query_order() {
        let engine = AnswerEngine::cpu();
        let first = engine.prepare(PARAMS, matrix(11, 2)).unwrap();
        let second = engine.prepare(PARAMS, matrix(22, 3)).unwrap();
        let first_queries = [query(11, 7), query(11, 8)];
        let second_queries = [query(22, 9)];
        let jobs = [
            AnswerJob {
                matrix: &first,
                queries: &first_queries,
            },
            AnswerJob {
                matrix: &second,
                queries: &second_queries,
            },
        ];
        let (answers, report) = engine.answer_many_with_report(&jobs).unwrap();
        assert_eq!(answers.len(), 2);
        assert_eq!(
            answers[0]
                .iter()
                .map(AnswerMatrix::query_id)
                .collect::<Vec<_>>(),
            [7, 8]
        );
        assert_eq!(answers[1][0].query_id(), 9);
        assert_eq!(answers[0][0].rows(), 2);
        assert_eq!(answers[1][0].rows(), 3);
        assert_eq!(report.cpu_entries, 2);
        assert_eq!(report.entries.len(), 2);
    }

    #[test]
    fn validation_rejects_empty_repeated_and_foreign_jobs() {
        let engine = AnswerEngine::cpu();
        let foreign_engine = AnswerEngine::cpu();
        let prepared = engine.prepare(PARAMS, matrix(11, 2)).unwrap();
        let foreign = foreign_engine.prepare(PARAMS, matrix(22, 2)).unwrap();
        let queries = [query(11, 1)];
        let foreign_queries = [query(22, 1)];

        assert!(matches!(
            engine.answer_many(&[]),
            Err(AnswerEngineError::EmptyJobs)
        ));
        assert!(matches!(
            engine.answer_many(&[AnswerJob {
                matrix: &prepared,
                queries: &[]
            }]),
            Err(AnswerEngineError::EmptyQueries { entry: 0 })
        ));
        assert!(matches!(
            engine.answer_many(&[
                AnswerJob {
                    matrix: &prepared,
                    queries: &queries
                },
                AnswerJob {
                    matrix: &prepared,
                    queries: &queries
                },
            ]),
            Err(AnswerEngineError::RepeatedMatrix { entry: 1 })
        ));
        assert!(matches!(
            engine.answer_many(&[AnswerJob {
                matrix: &foreign,
                queries: &foreign_queries
            }]),
            Err(AnswerEngineError::ForeignMatrix { entry: 0 })
        ));
    }
}
