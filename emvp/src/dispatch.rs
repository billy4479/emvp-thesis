//! Backend-selection policy for the server answer phase.
//!
//! One answer batch can run on three backends: a single core, the rayon
//! pool, or the GPU. This module is the single decision point between them:
//! two multiplication-count thresholds split the estimated batch work
//! `queries * rows * n` into three tiers, and (with the `gpu` feature) the
//! `AnswerDispatcher` runner turns that policy into an executable answer
//! server that owns the device hand-off.
//!
//! # Thresholds
//!
//! `MIN_PARALLEL_MULTIPLICATIONS` gates the single-core to rayon tier. It
//! was calibrated once with the benchmark suite when sizing the rayon thread
//! pool; smaller workloads stay serial because scheduling overhead dominates.
//!
//! `MIN_GPU_MULTIPLICATIONS` (with the `gpu` feature) gates the rayon to GPU
//! tier. It was originally calibrated by sweeping both answer paths. The
//! 2026-09-14 calibration ran on a 12-thread AMD Ryzen 5 2600X against an
//! NVIDIA GTX 1060 6GB: the raw crossover sits between 2^18 and 2^19
//! multiplications (the device already edges ahead at 2^19, but only by 4%,
//! well within run-to-run variance), and 2^20 is the smallest power of two
//! at which it wins decisively (22% there, up to 97% at the largest swept
//! shapes). The constant is machine-dependent by nature; recalibrate with
//!
//! ```text
//! cargo bench -p emvp --features gpu --bench dispatch -- \
//!     --save-baseline dispatch-policy-v2 'answer_dispatch_llm_v2/(cpu|gpu)'
//! ```
//!
//! when the server hardware changes. On a weaker integrated GPU the
//! crossover sits higher, so a value calibrated on a discrete card can hand
//! off too early there (an Intel Iris Xe iGPU measured 2^22-2^23).
//!
//! The selection additionally respects a runtime guard at the rayon tier:
//! parallel dispatch needs at least two answer rows per pool thread, so a
//! batch whose grid is too flat for the configured pool stays single-core
//! even above the parallel threshold.
//!
//! # Device shortfalls demote to the actual CPU tier
//!
//! A dispatcher without a usable device (built with `AnswerDispatcher::cpu`
//! or demoted at construction because the matrix exceeded device capacity)
//! cannot run GPU-tier batches. Instead of assuming the rayon tier, the
//! dispatcher recomputes the CPU tier for the same shape under the current
//! pool: one-threaded pools and grids too flat to parallelize demote all
//! the way to `AnswerBackend::SingleCore`. `AnswerDispatcher::backend`
//! reports the same tier `AnswerDispatcher::answer_batch` executes.

#[cfg(feature = "gpu")]
use rayon::current_num_threads;

#[cfg(feature = "gpu")]
use crate::gpu::{GpuAnswerer, GpuEncryptedMatrix, GpuError};
#[cfg(feature = "gpu")]
use crate::protocol::validate_matrix_shape;
#[cfg(feature = "gpu")]
use crate::protocol::{AnswerMatrix, EncryptedMatrix, EncryptedQuery, ProtocolError};

/// Minimum estimated field multiplications before the CPU answer path
/// switches from the serial row loop to rayon.
///
/// Smaller workloads stay single-core. Calibrated once with the benchmark
/// suite when sizing the rayon thread pool.
pub const MIN_PARALLEL_MULTIPLICATIONS: usize = 32 * 1024;

/// Minimum estimated field multiplications before an answer batch is worth
/// dispatching to the GPU.
///
/// Smaller workloads run on the CPU path. Calibrated for a 12-thread AMD
/// Ryzen 5 2600X against an NVIDIA GTX 1060 6GB (2026-09-14 full-suite run);
/// see the [module documentation](self) for the calibration procedure.
#[cfg(feature = "gpu")]
pub const MIN_GPU_MULTIPLICATIONS: usize = 1_048_576;

/// The execution backend selected for one answer batch.
///
/// The variants are ordered by escalation: a batch selected at the `Gpu`
/// tier would also have cleared every lower tier.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Ord, PartialOrd, Hash)]
pub enum AnswerBackend {
    /// The serial row loop on one core.
    SingleCore,
    /// The row grid distributed across the rayon pool.
    Rayon,
    /// The WGSL compute kernel on the device.
    #[cfg(feature = "gpu")]
    Gpu,
}

/// The CPU tier of the policy: is this `(work, grid, threads)` combination
/// worth distributing across the rayon pool?
///
/// `work` is the estimated field-multiplication count, `grid` the number of
/// independent answer rows. Shared by [`select_answer_backend`] and the CPU
/// kernels in [`crate::protocol`] so the policy and the code that executes
/// it cannot drift apart.
pub(crate) const fn is_parallel_work(work: usize, grid: usize, threads: usize) -> bool {
    threads > 1 && grid >= threads.saturating_mul(2) && work >= MIN_PARALLEL_MULTIPLICATIONS
}

/// Selects the CPU tier for one answer batch: rayon when the work clears
/// [`MIN_PARALLEL_MULTIPLICATIONS`] and the grid offers at least two rows
/// per pool thread, single-core otherwise.
///
/// This is the escalation base of [`select_answer_backend`] and the tier the
/// dispatcher falls back to when the raw policy selects the GPU tier but no
/// usable device exists. All dimension products saturate, so every `usize`
/// input is accepted.
pub const fn select_cpu_backend(
    queries: usize,
    rows: usize,
    n: usize,
    rayon_threads: usize,
) -> AnswerBackend {
    let grid = queries.saturating_mul(rows);
    let work = grid.saturating_mul(n);
    if is_parallel_work(work, grid, rayon_threads) {
        AnswerBackend::Rayon
    } else {
        AnswerBackend::SingleCore
    }
}

/// Selects the backend for one answer batch under the dispatch policy.
///
/// `queries * rows * n` estimates the batch's field-multiplication work;
/// work of at least the GPU threshold (with the `gpu` feature) selects the
/// `AnswerBackend::Gpu` tier, and below that the CPU tiers apply: rayon
/// when the work clears [`MIN_PARALLEL_MULTIPLICATIONS`] and the grid offers
/// at least two rows per pool thread, single-core otherwise. All dimension
/// products saturate, so every `usize` input is accepted.
#[must_use]
pub const fn select_answer_backend(
    queries: usize,
    rows: usize,
    n: usize,
    rayon_threads: usize,
) -> AnswerBackend {
    #[cfg(feature = "gpu")]
    {
        let grid = queries.saturating_mul(rows);
        let work = grid.saturating_mul(n);
        if work >= MIN_GPU_MULTIPLICATIONS {
            return AnswerBackend::Gpu;
        }
    }
    select_cpu_backend(queries, rows, n, rayon_threads)
}

/// A failed answer-phase dispatch (with the `gpu` feature).
#[cfg(feature = "gpu")]
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum AnswerDispatchError {
    /// The CPU path rejected the request: malformed parameters, a shape or
    /// identifier mismatch, or dimension overflow.
    Protocol(ProtocolError),
    /// The device path failed. Device failures never fall back to the CPU
    /// silently at call time; only construction-time availability shortfalls
    /// demote a dispatcher to the CPU path.
    Gpu(GpuError),
}

#[cfg(feature = "gpu")]
impl std::fmt::Display for AnswerDispatchError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Protocol(error) => error.fmt(formatter),
            Self::Gpu(error) => error.fmt(formatter),
        }
    }
}

#[cfg(feature = "gpu")]
impl std::error::Error for AnswerDispatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Protocol(error) => Some(error),
            Self::Gpu(error) => Some(error),
        }
    }
}

#[cfg(feature = "gpu")]
impl From<ProtocolError> for AnswerDispatchError {
    fn from(error: ProtocolError) -> Self {
        Self::Protocol(error)
    }
}

#[cfg(feature = "gpu")]
impl From<GpuError> for AnswerDispatchError {
    fn from(error: GpuError) -> Self {
        Self::Gpu(error)
    }
}

/// The server's answer-phase runner: it owns the backend decision for every
/// batch against one encrypted matrix.
///
/// Construct it once per matrix and answer every batch through
/// [`Self::answer_batch`]. When built with a [`GpuAnswerer`], the matrix is
/// uploaded to the device once and every batch that clears
/// [`MIN_GPU_MULTIPLICATIONS`] runs on the WGSL kernel; smaller batches run
/// the CPU [`answer_batch`](crate::answer_batch) path, whose serial and
/// rayon tiers follow the same policy. When built without an answerer, or
/// when the device cannot host the matrix (its buffers are too small), the
/// dispatcher silently runs every batch on the CPU path; the demotion is
/// observable through [`Self::backend`].
///
/// The dispatcher borrows the encrypted matrix, so both backends read the
/// same host copy and no ciphertext is duplicated for the CPU tier.
#[cfg(feature = "gpu")]
pub struct AnswerDispatcher<'a, const MODULUS: u32> {
    params: crate::params::EmvpParams,
    columns: usize,
    matrix: &'a EncryptedMatrix<MODULUS>,
    device: Option<(&'a GpuAnswerer, GpuEncryptedMatrix<MODULUS>)>,
}

#[cfg(feature = "gpu")]
impl<'a, const MODULUS: u32> AnswerDispatcher<'a, MODULUS> {
    /// Builds a dispatcher that runs every batch on the CPU path.
    ///
    /// # Errors
    ///
    /// Returns an error if the parameters are malformed or disagree with the
    /// matrix shape, matching the validation of the GPU-capable
    /// [`Self::new`].
    pub fn cpu(
        params: crate::params::EmvpParams,
        matrix: &'a EncryptedMatrix<MODULUS>,
    ) -> Result<Self, AnswerDispatchError> {
        let columns = validate_matrix_shape(&params, matrix)?;
        Ok(Self {
            params,
            columns,
            matrix,
            device: None,
        })
    }

    /// Builds a dispatcher and arms the GPU path when `answerer` is
    /// provided.
    ///
    /// The armed upload is the one-time matrix transfer; every GPU-tier batch
    /// reuses the device-resident words. A device that cannot host the
    /// matrix ([`GpuError::UploadTooLarge`]) demotes the dispatcher to the
    /// CPU path silently, because that is a hardware capacity shortfall a
    /// smaller deployment can absorb; every other upload failure propagates,
    /// because a shape or identifier mismatch would fail on the CPU path
    /// anyway and is better surfaced here.
    ///
    /// # Errors
    ///
    /// Returns an error if the parameters are malformed or disagree with the
    /// matrix shape, or if the armed upload fails for a reason other than
    /// device capacity.
    pub fn new(
        params: crate::params::EmvpParams,
        matrix: &'a EncryptedMatrix<MODULUS>,
        answerer: Option<&'a GpuAnswerer>,
    ) -> Result<Self, AnswerDispatchError> {
        let columns = validate_matrix_shape(&params, matrix)?;
        let mut device = None;
        if let Some(answerer) = answerer {
            match answerer.upload_matrix(&params, matrix) {
                Ok(gpu_matrix) => device = Some((answerer, gpu_matrix)),
                Err(GpuError::UploadTooLarge { .. }) => {}
                Err(error) => return Err(AnswerDispatchError::Gpu(error)),
            }
        }
        Ok(Self {
            params,
            columns,
            matrix,
            device,
        })
    }

    /// The backend this dispatcher would run for a batch of `queries`
    /// queries under the current rayon pool.
    ///
    /// This is the effective decision, not the raw policy: a GPU-tier batch
    /// on a dispatcher without a usable device is demoted to the actual CPU
    /// tier the CPU path would run for the same shape under the current
    /// pool, which can be [`AnswerBackend::SingleCore`] for a one-threaded
    /// pool or a grid too flat to parallelize.
    #[must_use]
    pub fn backend(&self, queries: usize) -> AnswerBackend {
        match select_answer_backend(
            queries,
            self.matrix.rows(),
            self.columns,
            current_num_threads(),
        ) {
            AnswerBackend::Gpu if self.device.is_none() => select_cpu_backend(
                queries,
                self.matrix.rows(),
                self.columns,
                current_num_threads(),
            ),
            backend => backend,
        }
    }

    /// Whether the dispatcher holds a device-resident copy of the matrix.
    #[must_use]
    pub const fn is_device_backed(&self) -> bool {
        self.device.is_some()
    }

    /// Answers a batch of encrypted queries, dispatching under the policy.
    ///
    /// The GPU tier and the CPU tiers produce bit-identical answers, so the
    /// backend choice is never observable in the output.
    ///
    /// # Errors
    ///
    /// Returns an error if the batch is empty, any query has the wrong
    /// length or a foreign instance identifier, or the selected backend
    /// fails. Validation is all-or-nothing on both paths: an error means no
    /// answer was produced.
    pub fn answer_batch(
        &self,
        queries: &[EncryptedQuery<MODULUS>],
    ) -> Result<Vec<AnswerMatrix<MODULUS>>, AnswerDispatchError> {
        // `backend` only reports the GPU tier when the dispatcher holds a
        // device-resident matrix, so the `Some` arm covers exactly the
        // device batches and every other combination runs the CPU path.
        match (self.backend(queries.len()), self.device.as_ref()) {
            (AnswerBackend::Gpu, Some((answerer, gpu_matrix))) => {
                Ok(answerer.answer_batch(gpu_matrix, queries)?)
            }
            _ => Ok(crate::answer_batch(&self.params, self.matrix, queries)?),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn single_core_below_the_parallel_threshold() {
        // A grid of 64 rows satisfies the two-rows-per-thread guard for any
        // pool up to 32 threads; the multiplication count decides the tier.
        let (queries, rows, threads) = (1, 64, 8);
        let n = MIN_PARALLEL_MULTIPLICATIONS / (queries * rows);
        assert_eq!(queries * rows * n, MIN_PARALLEL_MULTIPLICATIONS);
        assert_eq!(
            select_answer_backend(queries, rows, n, threads),
            AnswerBackend::Rayon
        );
        assert_eq!(
            select_answer_backend(queries, rows, n - 1, threads),
            AnswerBackend::SingleCore
        );
    }

    #[test]
    fn flat_grids_stay_single_core_above_the_parallel_threshold() {
        // Work 4x over the threshold, but the grid guard decides: eight rows
        // per thread parallelize, one row per thread stays serial, and a
        // single-threaded pool never parallelizes.
        let work = 4 * MIN_PARALLEL_MULTIPLICATIONS;
        assert_eq!(
            select_answer_backend(1, 64, work / 64, 8),
            AnswerBackend::Rayon
        );
        assert_eq!(
            select_answer_backend(1, 8, work / 8, 8),
            AnswerBackend::SingleCore
        );
        assert_eq!(
            select_answer_backend(1, 4096, work / 4096, 1),
            AnswerBackend::SingleCore
        );
    }

    #[test]
    fn saturating_dimensions_never_panic() {
        let huge = usize::MAX;
        // Without the gpu feature the CPU tiers cap the escalation.
        #[cfg(feature = "gpu")]
        let expected = AnswerBackend::Gpu;
        #[cfg(not(feature = "gpu"))]
        let expected = AnswerBackend::Rayon;
        assert_eq!(select_answer_backend(huge, huge, huge, 8), expected);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_tier_boundary() {
        let (queries, rows, threads) = (4, 65_536, 8);
        let n = MIN_GPU_MULTIPLICATIONS / (queries * rows);
        assert_eq!(queries * rows * n, MIN_GPU_MULTIPLICATIONS);
        assert_eq!(
            select_answer_backend(queries, rows, n, threads),
            AnswerBackend::Gpu
        );
        assert_eq!(
            select_answer_backend(queries, rows, n - 1, threads),
            AnswerBackend::Rayon
        );
    }
}
