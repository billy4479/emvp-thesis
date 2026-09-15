#![expect(
    clippy::unwrap_used,
    reason = "benchmarks use fixed valid parameters and keep setup beside measurement"
)]
#![expect(
    clippy::unnecessary_literal_unwrap,
    reason = "the adapter probe wraps an unexpected error in `Err` so the single `unwrap` path fails the bench loudly; static analysis flags the literal even though the error is dynamic"
)]

//! CPU-versus-GPU crossover sweep for the server answer phase.
//!
//! This suite calibrates the workload size (in field multiplications,
//! `batch * rows * n` for the fixed parameter set) at which the device
//! answer path starts beating the CPU `answer_batch`, so the result can
//! become the hard-coded `MIN_GPU_MULTIPLICATIONS` dispatch threshold.
//! The fixed parameter set, shared fixture construction
//! (`benches/common::protocol_fixtures_batch`), and the shared pinned
//! eight-thread CPU pool (`benches/common::benchmark_pool`) match
//! `benches/gpu.rs` exactly, so CPU and GPU rows are directly comparable
//! with each other here and with the GPU suite; only the shapes differ,
//! sweeping work from 2^16 up to 2^24 estimated field multiplications.
//! Fixture construction (derive + encrypt + queries) and the one-time
//! matrix upload happen before timing; every measured iteration is one
//! full `answer_batch` on either path. Without a compute adapter the
//! binary prints a notice and benchmarks nothing.
//!
//! # Case IDs and filters
//!
//! Each `(batch, rows)` shape runs two cases under the `answer_dispatch_llm_v2`
//! group: `answer_dispatch_llm_v2/cpu/batchB-rowsR` (CPU reference) and
//! `answer_dispatch_llm_v2/gpu/batchB-rowsR` (device answer path). The criterion
//! `--` filter is a regular expression over full case IDs, so one filter
//! selects the whole paired sweep in a single run:
//!
//! ```text
//! cargo bench -p emvp --features gpu --bench dispatch -- \
//!     --save-baseline dispatch-policy-v2 'answer_dispatch_llm_v2/(cpu|gpu)'
//! ```
//!
//! # Calibrated result
//!
//! On the current calibration machine (12 CPU threads, AMD Ryzen 5 2600X,
//! NVIDIA GTX 1060 6GB, Vulkan; 2026-09-14 full-suite run, then the legacy
//! k = 512 parameter set) the raw crossover sat between 2^18 and 2^19
//! estimated field multiplications (the device edged ahead at 2^19 by only
//! 4%), with the device ahead decisively (22%, growing to 97% at the
//! largest shapes) from 2^20 up; that value is hard-coded as
//! `emvp::dispatch::MIN_GPU_MULTIPLICATIONS` (previously 2^24 from an
//! Intel Iris Xe iGPU calibration). The sweep now runs the LLM-scale
//! deployment parameter set (n = 8192) over the same 2^16..2^24 work
//! range. Its fixture and pool semantics differ from legacy saved baselines;
//! establish a fresh `dispatch-policy-v2` baseline on GPU-equipped hardware
//! whenever the answer hardware or pool size changes.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use emvp::{GpuAnswerer, GpuError, answer_batch, search};

mod common;

use common::{LLM_LAMBDA, benchmark_pool, protocol_fixtures_batch};

// LLM-scale record length for the sweep's parameter set: the same
// parameter search and parameter set the `gpu` suite's first suite uses,
// so the two crossover studies share their n = 8192 query width.
const LLM_RECORD_LENGTH: usize = 4096;

// Crossover sweep as (batch, rows) pairs. The work metric
// `batch * rows * n` with n = 8192 spans 2^16 (first shape) through
// 2^24 (last shapes) field multiplications, doubling shape by shape
// around the expected CPU/GPU crossover. Small-row shapes fall under the
// `rows >= 2 * threads` parallel threshold and exercise the serial CPU
// tier the device path must also beat.
const CASES: [(usize, usize); 14] = [
    (1, 8),
    (1, 16),
    (1, 32),
    (1, 64),
    (1, 128),
    (1, 256),
    (1, 512),
    (1, 1024),
    (1, 2048),
    (2, 512),
    (2, 1024),
    (4, 256),
    (4, 512),
    (8, 256),
];

fn dispatch_benches(c: &mut Criterion) {
    let answerer = match GpuAnswerer::new() {
        Ok(answerer) => answerer,
        Err(GpuError::NoAdapter { reason }) => {
            println!("skipping dispatch benches: no compute adapter available ({reason})");
            return;
        }
        Err(error) => {
            let failure = Err::<GpuAnswerer, GpuError>(error);
            failure.unwrap()
        }
    };
    let params = search(LLM_RECORD_LENGTH, LLM_LAMBDA).unwrap();
    println!(
        "dispatch sweep at ell = {LLM_RECORD_LENGTH}: k = {}, b = {}, n = {}, lambda = {}",
        params.k,
        params.b,
        params.n().unwrap(),
        params.lambda
    );
    let pool = benchmark_pool();
    let mut group = c.benchmark_group("answer_dispatch_llm_v2");
    // Iterations at the top sizes cost tens of milliseconds each; few
    // samples and short phases keep the 28-case sweep bounded.
    group.sample_size(10);
    for &(batch, rows) in &CASES {
        let (encrypted, queries, _decoding_keys) = protocol_fixtures_batch(params, rows, batch);
        // One-time upload; the measured GPU iterations reuse the
        // device-resident matrix.
        let gpu_matrix = answerer.upload_matrix(&params, &encrypted).unwrap();
        let elements = u64::try_from(batch * rows * params.n().unwrap()).unwrap();
        group.throughput(Throughput::Elements(elements));
        group.bench_function(
            BenchmarkId::new("gpu", format!("batch{batch}-rows{rows}")),
            |b| {
                b.iter(|| answerer.answer_batch(&gpu_matrix, &queries).unwrap());
            },
        );
        drop(gpu_matrix);
        // CPU reference on the shared pinned eight-thread pool; small-row
        // shapes under the parallel thresholds exercise the serial CPU
        // tier, the rest the parallel tier.
        group.bench_function(
            BenchmarkId::new("cpu", format!("batch{batch}-rows{rows}")),
            |b| {
                b.iter(|| pool.install(|| answer_batch(&params, &encrypted, &queries).unwrap()));
            },
        );
    }
    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(3))
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = dispatch_benches
}
criterion_main!(benches);
