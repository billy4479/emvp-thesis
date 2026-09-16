//! Backend-agnostic orchestration for answering several encrypted matrices.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use rayon::prelude::*;

#[cfg(feature = "gpu")]
use crate::dispatch::select_answer_backend;
use crate::dispatch::{AnswerBackend, select_cpu_backend};
#[cfg(feature = "gpu")]
use crate::gpu::packed::{PackedJob, PackedStats, PackedTimings};
#[cfg(feature = "gpu")]
use crate::gpu::{GpuAnswerer, GpuEncryptedMatrix, GpuError};
use crate::{
    AnswerMatrix, EmvpParams, EncryptedMatrix, EncryptedQuery, ProtocolError, answer_batch,
};

static NEXT_ENGINE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_MATRIX_ID: AtomicU64 = AtomicU64::new(1);

/// An answer runner and the resources shared by its prepared matrices.
pub struct AnswerEngine<const MODULUS: u32> {
    id: u64,
    #[cfg(feature = "gpu")]
    gpu: Option<Arc<GpuEngine>>,
}

#[cfg(feature = "gpu")]
struct GpuEngine {
    answerer: GpuAnswerer,
    max_bytes: u64,
    reserved_bytes: AtomicU64,
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
    #[cfg(feature = "gpu")]
    gpu_matrix: Option<GpuEncryptedMatrix<MODULUS>>,
    #[cfg(feature = "gpu")]
    _reservation: Option<Arc<GpuReservation>>,
}

#[cfg(feature = "gpu")]
struct GpuReservation {
    engine: Arc<GpuEngine>,
    bytes: u64,
}

#[cfg(feature = "gpu")]
impl Drop for GpuReservation {
    fn drop(&mut self) {
        self.engine
            .reserved_bytes
            .fetch_sub(self.bytes, Ordering::AcqRel);
    }
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
    /// Device execution failed after all jobs had been validated.
    #[cfg(feature = "gpu")]
    Gpu(GpuError),
    /// An internal execution invariant was violated without producing partial output.
    InternalState(&'static str),
}

/// A failure while validating and retaining a matrix.
#[derive(Debug)]
#[non_exhaustive]
pub enum PrepareMatrixError {
    /// The matrix or its protocol parameters are malformed.
    Protocol(ProtocolError),
    /// The matrix could not be reserved or uploaded to the device.
    #[cfg(feature = "gpu")]
    Gpu(GpuError),
    /// An internal execution invariant was violated without producing a
    /// handle.
    InternalState(&'static str),
}

impl fmt::Display for PrepareMatrixError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Protocol(error) => error.fmt(formatter),
            #[cfg(feature = "gpu")]
            Self::Gpu(error) => error.fmt(formatter),
            Self::InternalState(reason) => write!(formatter, "answer engine state error: {reason}"),
        }
    }
}

impl std::error::Error for PrepareMatrixError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Protocol(error) => Some(error),
            #[cfg(feature = "gpu")]
            Self::Gpu(error) => Some(error),
            Self::InternalState(_) => None,
        }
    }
}

impl From<ProtocolError> for PrepareMatrixError {
    fn from(error: ProtocolError) -> Self {
        Self::Protocol(error)
    }
}

#[cfg(feature = "gpu")]
impl From<GpuError> for PrepareMatrixError {
    fn from(error: GpuError) -> Self {
        Self::Gpu(error)
    }
}

/// A failure while validating, reserving, or uploading a whole matrix set.
///
/// Every failure is all-or-nothing: no handle of the rejected set is
/// returned and none of its residency stays reserved.
#[derive(Debug)]
#[non_exhaustive]
pub enum PrepareBatchError {
    /// The batch contained no matrices.
    Empty,
    /// The matrix at `index`, or its protocol parameters, are malformed.
    /// Nothing was reserved or uploaded.
    Protocol {
        /// The input position of the rejected matrix.
        index: usize,
        /// The validation failure.
        source: ProtocolError,
    },
    /// The whole set does not fit the engine's device residency budget.
    /// Nothing was uploaded.
    #[cfg(feature = "gpu")]
    Budget(GpuError),
    /// The matrix at `index` failed its device upload. The handles already
    /// prepared for earlier entries were discarded together with the whole
    /// set's reservation.
    #[cfg(feature = "gpu")]
    Upload {
        /// The input position of the failed matrix.
        index: usize,
        /// The device failure.
        source: GpuError,
    },
    /// An internal execution invariant was violated without producing
    /// partial output.
    InternalState(&'static str),
}

impl fmt::Display for PrepareBatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("matrix batch is empty"),
            Self::Protocol { index, source } => write!(formatter, "batch matrix {index}: {source}"),
            #[cfg(feature = "gpu")]
            Self::Budget(error) => error.fmt(formatter),
            #[cfg(feature = "gpu")]
            Self::Upload { index, source } => {
                write!(formatter, "batch matrix {index}: {source}")
            }
            Self::InternalState(reason) => write!(formatter, "answer engine state error: {reason}"),
        }
    }
}

impl std::error::Error for PrepareBatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Protocol { source, .. } => Some(source),
            #[cfg(feature = "gpu")]
            Self::Budget(error) | Self::Upload { source: error, .. } => Some(error),
            Self::Empty | Self::InternalState(_) => None,
        }
    }
}

impl From<PrepareBatchError> for PrepareMatrixError {
    fn from(error: PrepareBatchError) -> Self {
        match error {
            PrepareBatchError::Protocol { source, .. } => Self::Protocol(source),
            #[cfg(feature = "gpu")]
            PrepareBatchError::Budget(source) | PrepareBatchError::Upload { source, .. } => {
                Self::Gpu(source)
            }
            PrepareBatchError::Empty => {
                Self::InternalState("an empty batch reached the single-matrix preparation path")
            }
            PrepareBatchError::InternalState(reason) => Self::InternalState(reason),
        }
    }
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
            #[cfg(feature = "gpu")]
            Self::Gpu(error) => error.fmt(formatter),
            Self::InternalState(reason) => write!(formatter, "answer engine state error: {reason}"),
        }
    }
}

impl std::error::Error for AnswerEngineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Protocol { source, .. } => Some(source),
            #[cfg(feature = "gpu")]
            Self::Gpu(error) => Some(error),
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
    /// Queries evaluated on the GPU.
    pub gpu_queries: usize,
    /// Bytes occupied by live query words in GPU packs, including alignment gaps.
    pub gpu_query_bytes: u64,
    /// Bytes occupied by live answer words in GPU packs, including alignment gaps.
    pub gpu_answer_bytes: u64,
    /// Number of packed GPU chunks.
    pub gpu_chunks: usize,
    /// Number of GPU dispatch segments across all chunks.
    pub gpu_segments: usize,
    /// Time spent validating and planning.
    pub planning: Duration,
    /// Time spent evaluating CPU entries.
    pub cpu_compute: Duration,
    /// Time leasing and growing packed device buffers.
    pub gpu_prepare_buffers: Duration,
    /// Time encoding packed query uploads.
    pub gpu_upload: Duration,
    /// Time recording and submitting the single GPU command buffer.
    pub gpu_submit: Duration,
    /// Time waiting for mapped GPU readback.
    pub gpu_wait: Duration,
    /// Time reconstructing GPU answers.
    pub gpu_reconstruct: Duration,
    /// End-to-end wall time.
    pub total: Duration,
}

impl<const MODULUS: u32> AnswerEngine<MODULUS> {
    /// Creates an engine that evaluates every matrix on the CPU.
    #[must_use]
    pub fn cpu() -> Self {
        Self {
            id: NEXT_ENGINE_ID.fetch_add(1, Ordering::Relaxed),
            #[cfg(feature = "gpu")]
            gpu: None,
        }
    }

    /// Creates a GPU-capable engine with a long-lived matrix residency budget.
    /// A machine without an adapter becomes a CPU engine; every other device
    /// initialization failure is returned.
    ///
    /// # Errors
    ///
    /// Returns device-request failures and any adapter error other than
    /// [`GpuError::NoAdapter`].
    #[cfg(feature = "gpu")]
    pub fn new(max_gpu_bytes: u64) -> Result<Self, GpuError> {
        match GpuAnswerer::new() {
            Ok(answerer) => Ok(Self {
                id: NEXT_ENGINE_ID.fetch_add(1, Ordering::Relaxed),
                gpu: Some(Arc::new(GpuEngine {
                    answerer,
                    max_bytes: max_gpu_bytes,
                    reserved_bytes: AtomicU64::new(0),
                })),
            }),
            Err(GpuError::NoAdapter { .. }) => Ok(Self::cpu()),
            Err(error) => Err(error),
        }
    }

    /// Validates and retains an encrypted matrix for repeated evaluation.
    ///
    /// A one-matrix [`Self::prepare_batch`]: the whole set is validated and
    /// its residency reserved before any upload begins.
    ///
    /// # Errors
    ///
    /// Returns a protocol error when the parameters and matrix disagree. A
    /// GPU engine also returns strict budget, capacity, and upload failures.
    pub fn prepare(
        &self,
        params: EmvpParams,
        matrix: EncryptedMatrix<MODULUS>,
    ) -> Result<PreparedMatrix<MODULUS>, PrepareMatrixError> {
        let mut prepared = self
            .prepare_batch_core(vec![(params, matrix)])
            .map_err(PrepareMatrixError::from)?;
        prepared.pop().ok_or(PrepareMatrixError::InternalState(
            "a one-matrix batch produced no handle",
        ))
    }

    /// Validates and retains a whole matrix set atomically.
    ///
    /// Every matrix is validated first, the device residency of the whole
    /// set is reserved with one atomic budget transition, and only then are
    /// the matrices uploaded. Concurrent callers therefore observe each set
    /// as all-or-nothing: a set either fits entirely alongside every set
    /// reserved before it, or it is rejected whole. On any failure no
    /// handle is returned and the entire reservation is released. On
    /// success each returned handle retains the reservation of its own
    /// matrix's bytes, released when that handle's final clone drops, so
    /// dropping one handle never holds another matrix's bytes.
    ///
    /// The returned handles keep the input order.
    ///
    /// # Errors
    ///
    /// Returns [`PrepareBatchError::Empty`] for an empty batch, the input
    /// index of the first malformed matrix as
    /// [`PrepareBatchError::Protocol`], the budget rejection of the whole
    /// set as [`PrepareBatchError::Budget`], or the input index of the
    /// first device upload failure as [`PrepareBatchError::Upload`]. A CPU
    /// engine can only produce the validation failures.
    pub fn prepare_batch<I>(
        &self,
        batch: I,
    ) -> Result<Vec<PreparedMatrix<MODULUS>>, PrepareBatchError>
    where
        I: IntoIterator<Item = (EmvpParams, EncryptedMatrix<MODULUS>)>,
    {
        self.prepare_batch_core(batch.into_iter().collect())
    }

    /// The shared validate, reserve, and upload pipeline behind
    /// [`Self::prepare_batch`] and [`Self::prepare`].
    fn prepare_batch_core(
        &self,
        matrices: Vec<(EmvpParams, EncryptedMatrix<MODULUS>)>,
    ) -> Result<Vec<PreparedMatrix<MODULUS>>, PrepareBatchError> {
        if matrices.is_empty() {
            return Err(PrepareBatchError::Empty);
        }
        // Phase 1: validate the whole set and total its residency before
        // touching the budget or the device.
        let mut total_bytes = 0_u64;
        #[cfg(feature = "gpu")]
        let mut residencies = Vec::with_capacity(matrices.len());
        for (index, (params, matrix)) in matrices.iter().enumerate() {
            let words = validate_matrix_shape(params, matrix)
                .map_err(|source| PrepareBatchError::Protocol { index, source })?;
            let bytes = residency_bytes(words).ok_or(PrepareBatchError::Protocol {
                index,
                source: ProtocolError::DimensionOverflow,
            })?;
            total_bytes = total_bytes
                .checked_add(bytes)
                .ok_or(PrepareBatchError::Protocol {
                    index,
                    source: ProtocolError::DimensionOverflow,
                })?;
            #[cfg(feature = "gpu")]
            residencies.push(bytes);
        }
        // Phase 2: reserve the whole set with one atomic transition before
        // any upload; failure rejects the set whole.
        #[cfg(feature = "gpu")]
        let gpu = self.gpu.as_ref();
        #[cfg(feature = "gpu")]
        if let Some(gpu) = gpu {
            reserve_budget(gpu, total_bytes).map_err(PrepareBatchError::Budget)?;
        }
        #[cfg(feature = "gpu")]
        let mut reservations = gpu.map_or_else(
            || Box::new(std::iter::empty()) as Box<dyn Iterator<Item = Arc<GpuReservation>>>,
            |gpu| {
                // Split the reserved total into one reservation per matrix
                // so every handle releases exactly its own bytes on drop.
                // Reservations not yet attached release the un-uploaded
                // remainder when this pipeline aborts.
                Box::new(residencies.iter().copied().map(move |bytes| {
                    Arc::new(GpuReservation {
                        engine: Arc::clone(gpu),
                        bytes,
                    })
                })) as Box<dyn Iterator<Item = Arc<GpuReservation>>>
            },
        );
        // Phase 3: upload. Any failure drops the handles prepared so far
        // and the reservations not yet attached, which together release
        // exactly the reserved total.
        let mut prepared = Vec::with_capacity(matrices.len());
        #[cfg_attr(
            not(feature = "gpu"),
            expect(
                unused_variables,
                reason = "the input index only names GPU upload failures"
            )
        )]
        for (index, (params, matrix)) in matrices.into_iter().enumerate() {
            #[cfg(feature = "gpu")]
            let (gpu_matrix, reservation) = match (gpu, reservations.next()) {
                (Some(gpu), Some(reservation)) => {
                    let uploaded = gpu
                        .answerer
                        .upload_matrix(&params, &matrix)
                        .map_err(|source| PrepareBatchError::Upload { index, source })?;
                    (Some(uploaded), Some(reservation))
                }
                (None, None) => (None, None),
                _ => {
                    return Err(PrepareBatchError::InternalState(
                        "the reservation count diverged from the matrix count",
                    ));
                }
            };
            prepared.push(PreparedMatrix {
                inner: Arc::new(PreparedMatrixInner {
                    id: NEXT_MATRIX_ID.fetch_add(1, Ordering::Relaxed),
                    engine_id: self.id,
                    params,
                    matrix,
                    #[cfg(feature = "gpu")]
                    gpu_matrix,
                    #[cfg(feature = "gpu")]
                    _reservation: reservation,
                }),
            });
        }
        Ok(prepared)
    }

    /// Whether this engine holds a live compute device, so prepared
    /// matrices stay resident under the configured device budget. `false`
    /// means every matrix is answered on the CPU and no device budget is
    /// active.
    #[must_use]
    #[cfg(feature = "gpu")]
    pub const fn has_device(&self) -> bool {
        self.gpu.is_some()
    }

    /// Whether this engine holds a live compute device, so prepared
    /// matrices stay resident under the configured device budget. `false`
    /// means every matrix is answered on the CPU and no device budget is
    /// active.
    #[must_use]
    #[cfg(not(feature = "gpu"))]
    pub const fn has_device(&self) -> bool {
        false
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
        let threads = rayon::current_num_threads();
        let entries = validate_jobs(self.id, jobs, threads)?;
        #[cfg(feature = "gpu")]
        let mut entries = entries;

        #[cfg(feature = "gpu")]
        let gpu_jobs = collect_gpu_jobs(jobs, &entries);
        #[cfg(feature = "gpu")]
        let packed_plan = if gpu_jobs.is_empty() {
            None
        } else {
            let gpu = self.gpu_answerer()?;
            Some(
                gpu.answerer
                    .plan_packed(&gpu_jobs)
                    .map_err(AnswerEngineError::Gpu)?,
            )
        };
        let planning = planning_start.elapsed();

        #[cfg(feature = "gpu")]
        let flight = if let Some(plan) = packed_plan {
            let flight = self
                .gpu_answerer()?
                .answerer
                .begin_packed(&gpu_jobs, plan)
                .map_err(AnswerEngineError::Gpu)?;
            for (entry, count) in flight.segment_counts() {
                entries[entry].gpu_segments = count;
            }
            Some(flight)
        } else {
            None
        };

        let cpu_start = Instant::now();
        let computed = compute_cpu_jobs(jobs, &entries);
        let cpu_compute = cpu_start.elapsed();
        let mut answers: Vec<Option<Vec<AnswerMatrix<MODULUS>>>> =
            (0..jobs.len()).map(|_| None).collect();
        for result in computed {
            let (entry, entry_answers) = result?;
            answers[entry] = Some(entry_answers);
        }
        #[cfg(feature = "gpu")]
        let (gpu_stats, gpu_timings) = if let Some(flight) = flight {
            let (gpu_answers, stats, timings) = self
                .gpu_answerer()?
                .answerer
                .finish_packed(flight)
                .map_err(AnswerEngineError::Gpu)?;
            for (entry, entry_answers) in gpu_answers {
                answers[entry] = Some(entry_answers);
            }
            (stats, timings)
        } else {
            (PackedStats::default(), PackedTimings::default())
        };
        let answers = collect_answers(answers)?;
        #[cfg(feature = "gpu")]
        let report = build_report(
            entries,
            planning,
            cpu_compute,
            gpu_stats,
            gpu_timings,
            total_start.elapsed(),
        );
        #[cfg(not(feature = "gpu"))]
        let report = build_report(entries, planning, cpu_compute, total_start.elapsed());
        Ok((answers, report))
    }

    #[cfg(feature = "gpu")]
    fn gpu_answerer(&self) -> Result<&GpuEngine, AnswerEngineError> {
        self.gpu.as_deref().ok_or(AnswerEngineError::InternalState(
            "GPU work was selected without a GPU engine",
        ))
    }
}

/// Validates one matrix against its parameters and returns its field-word
/// count `rows * n`.
///
/// This is the shape contract of both the CPU answer path and the device
/// upload path, applied before any budget or device work.
fn validate_matrix_shape<const MODULUS: u32>(
    params: &EmvpParams,
    matrix: &EncryptedMatrix<MODULUS>,
) -> Result<usize, ProtocolError> {
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
    Ok(expected)
}

/// The device residency of `words` field words, one `u32` each.
fn residency_bytes(words: usize) -> Option<u64> {
    u64::try_from(words)
        .ok()
        .and_then(|words| words.checked_mul(4))
}

/// Reserves `bytes` of long-lived matrix residency with one compare-and-
/// swap transition of the engine's budget counter, so concurrent callers
/// each reserve their whole request or nothing of it.
#[cfg(feature = "gpu")]
fn reserve_budget(gpu: &GpuEngine, bytes: u64) -> Result<(), GpuError> {
    let mut current = gpu.reserved_bytes.load(Ordering::Acquire);
    loop {
        let available = gpu.max_bytes.saturating_sub(current);
        let exhausted = || GpuError::BudgetExceeded {
            requested_bytes: bytes,
            available_bytes: available,
        };
        let Some(next) = current.checked_add(bytes) else {
            return Err(exhausted());
        };
        if next > gpu.max_bytes {
            return Err(exhausted());
        }
        match gpu.reserved_bytes.compare_exchange_weak(
            current,
            next,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => return Ok(()),
            Err(observed) => current = observed,
        }
    }
}

fn validate_jobs<const MODULUS: u32>(
    engine_id: u64,
    jobs: &[AnswerJob<'_, MODULUS>],
    threads: usize,
) -> Result<Vec<AnswerEntryReport>, AnswerEngineError> {
    if jobs.is_empty() {
        return Err(AnswerEngineError::EmptyJobs);
    }
    let mut seen = HashSet::with_capacity(jobs.len());
    let mut entries = Vec::with_capacity(jobs.len());
    for (entry, job) in jobs.iter().enumerate() {
        if job.matrix.inner.engine_id != engine_id {
            return Err(AnswerEngineError::ForeignMatrix { entry });
        }
        if !seen.insert(job.matrix.inner.id) {
            return Err(AnswerEngineError::RepeatedMatrix { entry });
        }
        if job.queries.is_empty() {
            return Err(AnswerEngineError::EmptyQueries { entry });
        }
        entries.push(validate_entry(entry, job, threads)?);
    }
    Ok(entries)
}

fn validate_entry<const MODULUS: u32>(
    entry: usize,
    job: &AnswerJob<'_, MODULUS>,
    threads: usize,
) -> Result<AnswerEntryReport, AnswerEngineError> {
    let inner = &job.matrix.inner;
    let n = inner.matrix.columns();
    for query in job.queries {
        validate_query(entry, inner, query, n)?;
    }
    let answer_words = inner
        .matrix
        .rows()
        .checked_mul(
            inner
                .params
                .blocks()
                .map_err(|source| AnswerEngineError::Protocol {
                    entry,
                    source: source.into(),
                })?,
        )
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
    let cpu_backend = select_cpu_backend(job.queries.len(), inner.matrix.rows(), n, threads);
    #[cfg(feature = "gpu")]
    let backend = inner.gpu_matrix.as_ref().map_or(cpu_backend, |_| {
        select_answer_backend(job.queries.len(), inner.matrix.rows(), n, threads)
    });
    #[cfg(not(feature = "gpu"))]
    let backend = cpu_backend;
    Ok(AnswerEntryReport {
        backend,
        queries: job.queries.len(),
        multiplications,
        query_words: job.queries.len().saturating_mul(n),
        answer_words,
        gpu_segments: 0,
    })
}

fn validate_query<const MODULUS: u32>(
    entry: usize,
    inner: &PreparedMatrixInner<MODULUS>,
    query: &EncryptedQuery<MODULUS>,
    n: usize,
) -> Result<(), AnswerEngineError> {
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
    Ok(())
}

#[cfg(feature = "gpu")]
fn collect_gpu_jobs<'a, const MODULUS: u32>(
    jobs: &'a [AnswerJob<'a, MODULUS>],
    entries: &[AnswerEntryReport],
) -> Vec<PackedJob<'a, MODULUS>> {
    jobs.iter()
        .enumerate()
        .filter_map(|(entry, job)| {
            if entries[entry].backend != AnswerBackend::Gpu {
                return None;
            }
            job.matrix
                .inner
                .gpu_matrix
                .as_ref()
                .map(|matrix| PackedJob {
                    entry,
                    matrix,
                    queries: job.queries,
                })
        })
        .collect()
}

#[cfg(feature = "gpu")]
fn compute_cpu_jobs<const MODULUS: u32>(
    jobs: &[AnswerJob<'_, MODULUS>],
    entries: &[AnswerEntryReport],
) -> Vec<Result<(usize, Vec<AnswerMatrix<MODULUS>>), AnswerEngineError>> {
    jobs.par_iter()
        .enumerate()
        .filter(|(entry, _)| entries[*entry].backend != AnswerBackend::Gpu)
        .map(|(entry, job)| {
            let inner = &job.matrix.inner;
            answer_batch(&inner.params, &inner.matrix, job.queries)
                .map(|answers| (entry, answers))
                .map_err(|source| AnswerEngineError::Protocol { entry, source })
        })
        .collect()
}

#[cfg(not(feature = "gpu"))]
fn compute_cpu_jobs<const MODULUS: u32>(
    jobs: &[AnswerJob<'_, MODULUS>],
    _entries: &[AnswerEntryReport],
) -> Vec<Result<(usize, Vec<AnswerMatrix<MODULUS>>), AnswerEngineError>> {
    jobs.par_iter()
        .enumerate()
        .map(|(entry, job)| {
            let inner = &job.matrix.inner;
            answer_batch(&inner.params, &inner.matrix, job.queries)
                .map(|answers| (entry, answers))
                .map_err(|source| AnswerEngineError::Protocol { entry, source })
        })
        .collect()
}

fn collect_answers<const MODULUS: u32>(
    answers: Vec<Option<Vec<AnswerMatrix<MODULUS>>>>,
) -> Result<Vec<Vec<AnswerMatrix<MODULUS>>>, AnswerEngineError> {
    answers
        .into_iter()
        .map(|answer| {
            answer.ok_or(AnswerEngineError::InternalState(
                "a planned job produced no answer",
            ))
        })
        .collect()
}

#[cfg(feature = "gpu")]
fn build_report(
    entries: Vec<AnswerEntryReport>,
    planning: Duration,
    cpu_compute: Duration,
    gpu_stats: PackedStats,
    gpu_timings: PackedTimings,
    total: Duration,
) -> AnswerReport {
    let cpu_entries = entries
        .iter()
        .filter(|entry| entry.backend != AnswerBackend::Gpu)
        .count();
    let gpu_entries = entries.len() - cpu_entries;
    let gpu_queries = entries
        .iter()
        .filter(|entry| entry.backend == AnswerBackend::Gpu)
        .map(|entry| entry.queries)
        .sum();
    AnswerReport {
        multiplications: total_multiplications(&entries),
        entries,
        cpu_entries,
        gpu_entries,
        gpu_queries,
        gpu_query_bytes: gpu_stats.query_bytes,
        gpu_answer_bytes: gpu_stats.answer_bytes,
        gpu_chunks: gpu_stats.chunks,
        gpu_segments: gpu_stats.segments,
        planning,
        cpu_compute,
        gpu_prepare_buffers: gpu_timings.prepare_buffers,
        gpu_upload: gpu_timings.encode_upload_queries,
        gpu_submit: gpu_timings.dispatch_submit,
        gpu_wait: gpu_timings.wait_readback,
        gpu_reconstruct: gpu_timings.reconstruct,
        total,
    }
}

#[cfg(not(feature = "gpu"))]
fn build_report(
    entries: Vec<AnswerEntryReport>,
    planning: Duration,
    cpu_compute: Duration,
    total: Duration,
) -> AnswerReport {
    AnswerReport {
        multiplications: total_multiplications(&entries),
        cpu_entries: entries.len(),
        entries,
        planning,
        cpu_compute,
        total,
        ..AnswerReport::default()
    }
}

fn total_multiplications(entries: &[AnswerEntryReport]) -> usize {
    entries.iter().fold(0, |total, entry| {
        total.saturating_add(entry.multiplications)
    })
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
    #[cfg(feature = "gpu")]
    const GPU_PARAMS: EmvpParams = EmvpParams {
        k: 1,
        ell: 1,
        b: 2,
        lambda: 7,
    };

    fn matrix(instance_id: u128, rows: usize) -> EncryptedMatrix<MODULUS> {
        let zero = PrimeField::<MODULUS>::new().element_u32(0);
        EncryptedMatrix::from_parts(instance_id, rows, 16, vec![zero; rows * 16]).unwrap()
    }

    /// A well-formed matrix whose declared width disagrees with [`PARAMS`],
    /// whose `n` is 16.
    fn narrow_matrix(instance_id: u128) -> EncryptedMatrix<MODULUS> {
        let zero = PrimeField::<MODULUS>::new().element_u32(0);
        EncryptedMatrix::from_parts(instance_id, 2, 15, vec![zero; 2 * 15]).unwrap()
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

    #[test]
    fn batch_prepare_returns_handles_in_input_order() {
        let engine = AnswerEngine::cpu();
        let batch = engine
            .prepare_batch([
                (PARAMS, matrix(11, 2)),
                (PARAMS, matrix(22, 3)),
                (PARAMS, matrix(33, 5)),
            ])
            .unwrap();
        assert_eq!(batch.len(), 3);
        let query_sets: Vec<Vec<EncryptedQuery<MODULUS>>> = vec![
            vec![query(11, 1)],
            vec![query(22, 2), query(22, 3)],
            vec![query(33, 4)],
        ];
        let jobs: Vec<AnswerJob<'_, MODULUS>> = batch
            .iter()
            .zip(query_sets.iter())
            .map(|(matrix, queries)| AnswerJob {
                matrix,
                queries: queries.as_slice(),
            })
            .collect();
        let answers = engine.answer_many(&jobs).unwrap();
        let answer_rows: Vec<usize> = answers
            .iter()
            .map(|group| group.iter().map(AnswerMatrix::rows).sum())
            .collect();
        assert_eq!(answer_rows, [2, 6, 5]);
    }

    #[test]
    fn batch_prepare_rejects_the_whole_set_and_indexes_the_failure() {
        let engine = AnswerEngine::cpu();
        assert!(matches!(
            engine.prepare_batch(Vec::<(EmvpParams, EncryptedMatrix<MODULUS>)>::new()),
            Err(PrepareBatchError::Empty)
        ));
        // A leading malformed matrix is indexed at its input position.
        assert!(matches!(
            engine.prepare_batch([(PARAMS, narrow_matrix(12)), (PARAMS, matrix(11, 2))]),
            Err(PrepareBatchError::Protocol { index: 0, .. })
        ));
        // A later malformed matrix rejects the whole set, including the
        // matrices before it: no handle comes back.
        assert!(matches!(
            engine.prepare_batch([(PARAMS, matrix(11, 2)), (PARAMS, narrow_matrix(12))]),
            Err(PrepareBatchError::Protocol { index: 1, .. })
        ));
        // The rejected set left nothing behind: the engine still works.
        engine.prepare(PARAMS, matrix(13, 2)).unwrap();
    }

    #[test]
    fn single_prepare_shares_the_batch_validation() {
        let engine = AnswerEngine::cpu();
        assert!(matches!(
            engine.prepare(PARAMS, narrow_matrix(12)),
            Err(PrepareMatrixError::Protocol(
                ProtocolError::LengthMismatch { .. }
            ))
        ));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_engine_matches_cpu_and_reports_packed_execution_when_available() {
        let engine = AnswerEngine::new(32 * 1024 * 1024).unwrap();
        if engine.gpu.is_none() {
            return;
        }
        let rows = crate::MIN_GPU_MULTIPLICATIONS / 2;
        let field = PrimeField::<MODULUS>::new();
        let values: Vec<_> = (0..rows * 2)
            .map(|index| field.element_u32((index % 17) as u32))
            .collect();
        let host = EncryptedMatrix::from_parts(91, rows, 2, values).unwrap();
        let expected_query =
            EncryptedQuery::from_parts(91, 7, vec![field.element_u32(3), field.element_u32(5)]);
        let expected =
            answer_batch(&GPU_PARAMS, &host, std::slice::from_ref(&expected_query)).unwrap();
        let prepared = engine.prepare(GPU_PARAMS, host).unwrap();
        let queries = [expected_query];
        let (actual, report) = engine
            .answer_many_with_report(&[AnswerJob {
                matrix: &prepared,
                queries: &queries,
            }])
            .unwrap();
        assert_eq!(actual[0], expected);
        assert_eq!(report.gpu_entries, 1);
        assert_eq!(report.gpu_queries, 1);
        assert_eq!(report.gpu_chunks, 1);
        assert_eq!(report.gpu_segments, 1);
        assert_eq!(report.entries[0].gpu_segments, 1);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_budget_is_held_until_the_final_prepared_clone_drops() {
        let bytes = (2 * PARAMS.n().unwrap() * 4) as u64;
        let engine = AnswerEngine::new(bytes).unwrap();
        if engine.gpu.is_none() {
            return;
        }
        let prepared = engine.prepare(PARAMS, matrix(31, 2)).unwrap();
        let final_clone = prepared.clone();
        drop(prepared);
        assert!(matches!(
            engine.prepare(PARAMS, matrix(32, 2)),
            Err(PrepareMatrixError::Gpu(GpuError::BudgetExceeded { .. }))
        ));
        drop(final_clone);
        engine.prepare(PARAMS, matrix(32, 2)).unwrap();
    }

    /// A GPU engine with the requested budget, or a skipped notice when the
    /// machine has no compute adapter.
    #[cfg(feature = "gpu")]
    fn gpu_engine_with_budget(bytes: u64) -> AnswerEngine<MODULUS> {
        let engine = AnswerEngine::new(bytes).unwrap();
        if engine.gpu.is_none() {
            eprintln!("gpu test skipped: no compute adapter");
        }
        engine
    }

    /// The engine's live matrix residency, readable because the tests are
    /// in this module.
    #[cfg(feature = "gpu")]
    fn reserved_bytes(engine: &AnswerEngine<MODULUS>) -> u64 {
        engine
            .gpu
            .as_ref()
            .map_or(0, |gpu| gpu.reserved_bytes.load(Ordering::Acquire))
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn batch_prepare_reserves_once_and_releases_per_handle() {
        let one = (2 * PARAMS.n().unwrap() * 4) as u64;
        let engine = gpu_engine_with_budget(3 * one);
        if engine.gpu.is_none() {
            return;
        }
        let batch = engine
            .prepare_batch([
                (PARAMS, matrix(41, 2)),
                (PARAMS, matrix(42, 2)),
                (PARAMS, matrix(43, 2)),
            ])
            .unwrap();
        // The whole set was reserved with one transition; nothing partial.
        assert_eq!(reserved_bytes(&engine), 3 * one);
        let another_set = [(PARAMS, matrix(44, 2)), (PARAMS, matrix(45, 2))];
        assert!(matches!(
            engine.prepare_batch(another_set),
            Err(PrepareBatchError::Budget(GpuError::BudgetExceeded {
                requested_bytes,
                available_bytes
            })) if requested_bytes == 2 * one && available_bytes == 0
        ));
        assert_eq!(reserved_bytes(&engine), 3 * one);
        // Dropping one handle releases exactly its own matrix's bytes; the
        // unrelated matrices stay resident.
        let keep = batch[0].clone();
        drop(batch);
        assert_eq!(reserved_bytes(&engine), one);
        // The released room fits exactly one more matrix.
        engine.prepare(PARAMS, matrix(44, 2)).unwrap();
        assert_eq!(reserved_bytes(&engine), 2 * one);
        // The final clone of the first handle releases the rest.
        drop(keep);
        assert_eq!(reserved_bytes(&engine), one);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn a_batch_validation_failure_leaves_no_reservation() {
        let one = (2 * PARAMS.n().unwrap() * 4) as u64;
        let engine = gpu_engine_with_budget(4 * one);
        if engine.gpu.is_none() {
            return;
        }
        assert!(matches!(
            engine.prepare_batch([
                (PARAMS, matrix(41, 2)),
                (PARAMS, narrow_matrix(42)),
                (PARAMS, matrix(43, 2)),
            ]),
            Err(PrepareBatchError::Protocol { index: 1, .. })
        ));
        // Validation precedes the reservation: the whole set was rejected
        // before any byte was booked, so a fresh set still fits.
        assert_eq!(reserved_bytes(&engine), 0);
        let batch = engine
            .prepare_batch([(PARAMS, matrix(41, 2)), (PARAMS, matrix(43, 2))])
            .unwrap();
        assert_eq!(reserved_bytes(&engine), 2 * one);
        drop(batch);
        assert_eq!(reserved_bytes(&engine), 0);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn concurrent_batches_reserve_the_whole_budget_atomically() {
        let one = (2 * PARAMS.n().unwrap() * 4) as u64;
        let engine = Arc::new(gpu_engine_with_budget(2 * one));
        if engine.gpu.is_none() {
            return;
        }
        // Two concurrent sets, each exactly the whole budget: exactly one
        // set wins its single whole-set reservation, the loser is rejected
        // whole, and no half-reserved set survives either outcome.
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let workers: Vec<_> = (0_usize..2)
            .map(|worker| {
                let engine = Arc::clone(&engine);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    engine.prepare_batch([
                        (PARAMS, matrix(u128::try_from(100 + worker).unwrap(), 2)),
                        (PARAMS, matrix(u128::try_from(200 + worker).unwrap(), 2)),
                    ])
                })
            })
            .collect();
        let mut won = None;
        for result in workers.into_iter().map(|worker| worker.join().unwrap()) {
            match result {
                Ok(batch) => {
                    assert!(won.is_none(), "two batches won one budget");
                    won = Some(batch);
                }
                Err(PrepareBatchError::Budget(_)) => {}
                Err(error) => panic!("unexpected batch failure: {error}"),
            }
        }
        // The winner's whole-set residency is exactly the budget, and
        // releasing it releases everything.
        assert_eq!(reserved_bytes(&engine), 2 * one);
        drop(won);
        assert_eq!(reserved_bytes(&engine), 0);
    }
}
