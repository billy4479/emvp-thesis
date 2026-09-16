//! Backend-selection policy for the server answer phase.
//!
//! One answer batch can run on three backends: a single core, the rayon
//! pool, or the GPU. This module is the single decision point between them:
//! two multiplication-count thresholds split the estimated batch work
//! `queries * rows * n` into three tiers, and the server executors
//! ([`crate::answer`] and `crate::engine`) turn that policy into their
//! fixed per-entry plan backends.
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
//! An executor without a usable device (a CPU engine, or a GPU-tier matrix
//! that is not device-resident) cannot run GPU-tier batches. Instead of
//! assuming the rayon tier, planning recomputes the CPU tier for the same
//! shape under the current pool: one-threaded pools and grids too flat to
//! parallelize demote all the way to `AnswerBackend::SingleCore`. The
//! engine's [`crate::engine::AnswerEngine::plan`] reports the same tier its
//! execute step runs, so the backend recorded in a plan is always the
//! backend that produces the answers.

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
/// This is the escalation base of [`select_answer_backend`] and the tier
/// executors fall back to when the raw policy selects the GPU tier but no
/// usable device exists. All dimension products saturate, so every `usize`
/// input is accepted.
#[must_use]
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
