#![expect(
    clippy::unwrap_used,
    reason = "benchmarks use fixed valid parameters and keep setup beside measurement"
)]
#![expect(
    clippy::unnecessary_literal_unwrap,
    reason = "the adapter probe wraps an unexpected error in `Err` so the single `unwrap` path fails the bench loudly; static analysis flags the literal even though the error is dynamic"
)]

//! GPU-versus-CPU benchmarks for the server answer phase at LLM scale.
//!
//! # Fairness contract
//!
//! CPU and GPU answer the same batch of the same queries against the same
//! encrypted-matrix fixture through the same plan → reserve → execute
//! batched API: one planned single-query batch executed per measured
//! iteration into a case-owned workspace reserved before timing, over
//! the `ell in {4096, 8192}` suites and `rows in {4096, 8192, 16384}`
//! matrix heights. The GPU tier plans through
//! [`GpuAnswerer::answer_batch_plan`] and executes through
//! [`GpuAnswerer::execute_answer_batch_into`]; the CPU tier plans through
//! [`AnswerPlan::plan`] on the shared pinned eight-thread pool
//! (`benches/common::benchmark_pool`, the same size the protocol suite
//! pins, which also fixes each plan's serial-or-rayon tier) and executes
//! through [`execute_answer_batch`], so the CPU numbers are comparable
//! with the saved protocol batched-answer baselines, but they are a
//! different case from the protocol suite's single-query `answer_into`
//! `answer` cases, so neither replaces the other in the tables. The
//! parameter suites and fixtures are the shared ones from
//! `benches/common`, so GPU and CPU rows here are directly comparable with
//! each other. Fixture construction (derive + encrypt) and the one-time
//! matrix upload happen before timing; the largest shape uploads a 1 GiB
//! encrypted matrix, which fits the 6 GiB reference card together with its
//! staging buffer. Without a compute adapter the binary prints a notice
//! and benchmarks nothing.
//!
//! # Case IDs and filters
//!
//! Each `(ell, rows)` size runs two cases under the `gpu_answer_v2` group:
//! `gpu_answer_v2/gpu/ellE-rowsR` (device answer path, total wall time) and
//! `gpu_answer_v2/cpu/ellE-rowsR` (the eight-thread-pool CPU reference). A
//! host-side [`PhaseTimings`] median|mean breakdown is printed per size;
//! its samples are collected
//! outside Criterion timing, a fixed number per size, so the timed
//! iterations never pay for storing instrumentation data.
//!
//! ```text
//! cargo bench -p emvp --features gpu -- 'gpu_answer_v2/(gpu|cpu)'
//! ```

use std::slice;
use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use emvp::{
    AnswerPlan, AnswerWorkspace, EncryptedQuery, GpuAnswerer, GpuEncryptedMatrix, GpuError,
    PhaseTimings, execute_answer_batch, search,
};
use rayon::ThreadPool;

mod common;

use common::{
    LLM_LAMBDA, LLM_RECORD_LENGTHS, LLM_ROW_COUNTS, MODULUS, benchmark_pool, protocol_fixtures,
};

// Fixed number of instrumented phase-diagnostic calls collected per size,
// strictly outside Criterion timing.
const PHASE_DIAGNOSTIC_SAMPLES: usize = 32;

/// Accessor for one [`PhaseTimings`] phase field.
type PhaseAccessor = fn(&PhaseTimings) -> Duration;

/// Prints the host-side phase breakdown collected across the fixed number
/// of out-of-band instrumented calls.
fn print_phase_breakdown(tag: &str, samples: &[PhaseTimings]) {
    if samples.is_empty() {
        return;
    }
    let phases: [(&str, PhaseAccessor); 6] = [
        ("prepare_buffers", |timings| timings.prepare_buffers),
        ("encode_upload_queries", |timings| {
            timings.encode_upload_queries
        }),
        ("dispatch_submit", |timings| timings.dispatch_submit),
        ("wait_readback", |timings| timings.wait_readback),
        ("reconstruct", |timings| timings.reconstruct),
        ("total", PhaseTimings::total),
    ];
    println!(
        "GPU phase diagnostics for {tag} over {} calls (median | mean):",
        samples.len()
    );
    for (name, accessor) in phases {
        let mut values: Vec<Duration> = samples.iter().map(accessor).collect();
        values.sort();
        let median = values[values.len() / 2];
        let mean =
            values.iter().sum::<Duration>() / u32::try_from(values.len()).unwrap_or(u32::MAX);
        println!("  {name:<24}{median:.3?} | {mean:.3?}");
    }
}

fn collect_phase_diagnostics(
    answerer: &GpuAnswerer,
    gpu_matrix: &GpuEncryptedMatrix<MODULUS>,
    query: &EncryptedQuery<MODULUS>,
    workspace: &mut AnswerWorkspace<MODULUS>,
    tag: &str,
) {
    let queries = slice::from_ref(query);
    let mut samples = Vec::with_capacity(PHASE_DIAGNOSTIC_SAMPLES);
    for _ in 0..PHASE_DIAGNOSTIC_SAMPLES {
        let (_answers, timings) = answerer
            .execute_answer_batch_into(gpu_matrix, queries, workspace)
            .unwrap();
        samples.push(timings);
    }
    print_phase_breakdown(tag, &samples);
}

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
    let mut group = c.benchmark_group("gpu_answer_v2");
    // Iterations at the top size cost tens of milliseconds; fewer samples
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
            let tag = format!("ell{ell}-rows{rows}");
            // One shared fixture for both sides: same encrypted matrix,
            // same single query.
            let (encrypted, query, _decoding_key) = protocol_fixtures(params, rows);
            let queries = slice::from_ref(&query);
            // One-time upload; the measured GPU iterations reuse the
            // device-resident matrix.
            let gpu_matrix = answerer.upload_matrix(&params, &encrypted).unwrap();
            let elements = u64::try_from(rows * params.n().unwrap()).unwrap();
            group.throughput(Throughput::Elements(elements));
            // Device path: plan once, reserve the case workspace once, then
            // execute into it every iteration. The host matrix plans the
            // arena (the shape arithmetic is identical on both tiers).
            let gpu_shape = answerer.answer_batch_plan(&gpu_matrix, queries).unwrap();
            let host_plan =
                pool.install(|| AnswerPlan::plan(&params, &encrypted, queries).unwrap());
            assert_eq!(gpu_shape, host_plan.shape(), "tier shapes must agree");
            let mut gpu_workspace = AnswerWorkspace::new();
            gpu_workspace.reserve(&host_plan).unwrap();
            // Total wall time, device path.
            group.bench_function(BenchmarkId::new("gpu", tag.clone()), |b| {
                b.iter(|| {
                    // The answers view is dropped here; the workspace's
                    // arena stays reserved for the next iteration.
                    let _answers = answerer
                        .execute_answer_batch_into(&gpu_matrix, queries, &mut gpu_workspace)
                        .unwrap();
                });
            });
            // CPU reference through the same plan-reserve-execute path,
            // same batch size, same fixture, on the pinned eight-thread
            // pool: planned on the pool (which fixes the serial-or-rayon
            // tier) into a case-owned workspace reserved once.
            let cpu_plan = pool.install(|| AnswerPlan::plan(&params, &encrypted, queries).unwrap());
            let mut cpu_workspace = AnswerWorkspace::new();
            cpu_workspace.reserve(&cpu_plan).unwrap();
            group.bench_function(BenchmarkId::new("cpu", tag.clone()), |b| {
                b.iter(|| {
                    // The answers view is dropped here; the workspace's
                    // arena stays reserved for the next iteration.
                    let _answers = pool
                        .install(|| execute_answer_batch(&cpu_plan, &mut cpu_workspace).unwrap());
                });
            });
            // Phase diagnostics: a fixed number of instrumented calls
            // collected outside Criterion timing, so no storage or
            // instrumentation ever lands inside a measured iteration.
            collect_phase_diagnostics(&answerer, &gpu_matrix, &query, &mut gpu_workspace, &tag);
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
