#![expect(
    clippy::unwrap_used,
    reason = "benchmarks use fixed valid parameters and keep setup beside measurement"
)]
#![expect(
    clippy::unnecessary_literal_unwrap,
    reason = "the adapter probe wraps an unexpected error in `Err` so the single `unwrap` path fails the bench loudly; static analysis flags the literal even though the error is dynamic"
)]

//! GPU-versus-CPU benchmarks for the server answer phase at LLM scale,
//! sweeping the query batch size.
//!
//! # Fairness contract
//!
//! CPU and GPU answer the same batch of the same queries against the same
//! encrypted-matrix fixture through the same plan → reserve → execute
//! batched API: one planned batch executed per measured iteration into a
//! case-owned workspace reserved before timing, over the
//! `ell in {4096, 8192}` suites, `rows in {4096, 8192, 16384}` matrix
//! heights, and `batch in {1, 8, 64, 256}` query batches. The batch grid
//! matches the `gpu_online` suite's, so the two suites' element/s figures
//! are directly comparable. The GPU tier plans through
//! [`GpuAnswerer::answer_batch_plan`] and executes through
//! [`GpuAnswerer::execute_answer_batch_into`]; the CPU tier plans through
//! [`AnswerPlan::plan`] on the shared pinned eight-thread pool
//! (`benches/common::benchmark_pool`, the same size the protocol suite
//! pins, which also fixes each plan's serial-or-rayon tier) and executes
//! through [`execute_answer_batch`], so the CPU numbers are comparable
//! with the saved protocol batched-answer baselines. The parameter suites
//! and fixtures are the shared ones from `benches/common`, so GPU and CPU
//! rows here are directly comparable with each other. Fixture construction
//! (derive + encrypt + the maximum query set) and the one-time matrix
//! upload happen before timing; the largest shape uploads a 1 GiB
//! encrypted matrix, which fits the 6 GiB reference card together with its
//! staging buffer. Without a compute adapter the binary prints a notice
//! and benchmarks nothing.
//!
//! Per-phase GPU diagnostics live in the `gpu_phase_sweep` suite, which
//! times every `PhaseTimings` phase across a much finer batch grid.
//!
//! # Case IDs and filters
//!
//! Each `(ell, rows, batch)` shape runs two cases under the
//! `gpu_answer_v3` group: `gpu_answer_v3/gpu/ellE-rowsR-batchB` (device
//! answer path, total wall time) and
//! `gpu_answer_v3/cpu/ellE-rowsR-batchB` (the eight-thread-pool CPU
//! reference). [`Throughput::Elements`] counts `batch * rows * ell`
//! logical elements per iteration, so Criterion's elem/s column rises
//! with batch and compares directly with the `gpu_online` suite.
//!
//! ```text
//! cargo bench -p emvp --features gpu --bench gpu -- 'gpu_answer_v3/(gpu|cpu)'
//! ```

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use emvp::{AnswerPlan, AnswerWorkspace, GpuAnswerer, GpuError, execute_answer_batch, search};
use rayon::ThreadPool;

mod common;

use common::{
    LLM_LAMBDA, LLM_RECORD_LENGTHS, LLM_ROW_COUNTS, benchmark_pool, elements,
    protocol_fixtures_batch,
};

// Query batches per measured iteration; the same grid the `gpu_online`
// suite sweeps, so elem/s figures compare across the two suites.
const BATCHES: [usize; 4] = [1, 8, 64, 256];

fn gpu_benches(c: &mut Criterion) {
    let answerer = match GpuAnswerer::new() {
        Ok(answerer) => answerer,
        Err(GpuError::NoAdapter { reason }) => {
            println!("skipping gpu benches: no compute adapter available ({reason})");
            return;
        }
        Err(error) => {
            let failure = Err::<GpuAnswerer, GpuError>(error);
            failure.unwrap()
        }
    };
    let pool: &'static ThreadPool = benchmark_pool();
    let mut group = c.benchmark_group("gpu_answer_v3");
    // Iterations at the top sizes cost tens of milliseconds; fewer samples
    // keep the run bounded, matching the huge protocol-suite configuration.
    group.sample_size(10);
    for &ell in &LLM_RECORD_LENGTHS {
        let params = search(ell, LLM_LAMBDA).unwrap();
        println!(
            "llm gpu suite at ell = {ell}: k = {}, b = {}, n = {}, lambda = {}",
            params.k,
            params.b,
            params.n().unwrap(),
            params.lambda
        );
        for &rows in &LLM_ROW_COUNTS {
            // One shared fixture for both tiers and every batch: same
            // encrypted matrix, maximum query set sliced per batch.
            let (encrypted, queries, _decoding_keys) =
                protocol_fixtures_batch(params, rows, BATCHES[BATCHES.len() - 1]);
            // One-time upload; the measured GPU iterations reuse the
            // device-resident matrix.
            let gpu_matrix = answerer.upload_matrix(&params, &encrypted).unwrap();
            for &batch in &BATCHES {
                let tag = format!("ell{ell}-rows{rows}-batch{batch}");
                let batch_queries = &queries[..batch];
                group.throughput(elements(batch * rows * ell));
                // Device path: plan once, reserve the case workspace once,
                // then execute into it every iteration. The host matrix
                // plans the arena (the shape arithmetic is identical on
                // both tiers).
                let gpu_shape =
                    answerer.answer_batch_plan(&gpu_matrix, batch_queries).unwrap();
                let host_plan =
                    pool.install(|| AnswerPlan::plan(&params, &encrypted, batch_queries).unwrap());
                assert_eq!(gpu_shape, host_plan.shape(), "tier shapes must agree");
                let mut gpu_workspace = AnswerWorkspace::new();
                gpu_workspace.reserve(&host_plan).unwrap();
                group.bench_function(BenchmarkId::new("gpu", tag.clone()), |b| {
                    b.iter(|| {
                        // The answers view is dropped here; the workspace's
                        // arena stays reserved for the next iteration.
                        let _answers = answerer
                            .execute_answer_batch_into(
                                &gpu_matrix,
                                batch_queries,
                                &mut gpu_workspace,
                            )
                            .unwrap();
                    });
                });
                // CPU reference through the same plan-reserve-execute path,
                // same batch size, same fixture, on the pinned eight-thread
                // pool: planned on the pool (which fixes the serial-or-rayon
                // tier) into a case-owned workspace reserved once.
                let mut cpu_workspace = AnswerWorkspace::new();
                cpu_workspace.reserve(&host_plan).unwrap();
                group.bench_function(BenchmarkId::new("cpu", tag.clone()), |b| {
                    b.iter(|| {
                        // The answers view is dropped here; the workspace's
                        // arena stays reserved for the next iteration.
                        let _answers = pool.install(|| {
                            execute_answer_batch(&host_plan, &mut cpu_workspace).unwrap()
                        });
                    });
                });
            }
            drop(gpu_matrix);
        }
    }
    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(4))
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = gpu_benches
}
criterion_main!(benches);
