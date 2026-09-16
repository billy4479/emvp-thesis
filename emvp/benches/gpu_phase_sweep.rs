#![expect(
    clippy::unwrap_used,
    reason = "benchmarks use fixed valid parameters and keep setup beside measurement"
)]
#![expect(
    clippy::unnecessary_literal_unwrap,
    reason = "the adapter probe wraps an unexpected error in `Err` so the single `unwrap` path fails the bench loudly; static analysis flags the literal even though the error is dynamic"
)]

//! Per-phase timing sweep of the GPU answer path across query batch sizes.
//!
//! # Why this exists
//!
//! The [`answer_mvp_throughput_v1`](crate) Criterion sweep shows a
//! throughput dip around batches 16-64 (per-query marginal cost rises
//! ~15% over the large-batch plateau) that pure code analysis cannot
//! attribute: one synchronous dispatch per call, identical per-thread
//! work, batch-independent memory traffic, and host phases measured in
//! microseconds. This tool times every [`PhaseTimings`] phase of
//! [`GpuAnswerer::answer_batch_with_timings`] for each batch directly, so
//! the dip can be localized to one phase (`wait_readback` = device
//! execution plus copyback, `reconstruct` = host readback pass, ...).
//!
//! # Order sensitivity
//!
//! Each sweep runs the batch plan three times: ascending, descending,
//! ascending again. If the mid-batch penalty follows run order (warming
//! or throttling history) the descending pass shifts it; if it is
//! structural it stays pinned to the same batch sizes in all passes. A
//! best-effort background sampler polls `nvidia-smi` for SM clocks,
//! temperature, and power so clock behavior can be correlated offline.
//!
//! # Output
//!
//! Comment lines start with `#`; everything else is one CSV row per
//! `(pass, batch)` with the median of each phase in microseconds (min/max
//! for `wait`, the phase that blocks on the device), written to stdout:
//!
//! ```text
//! pass,batch,calls,prepare_us,encode_us,dispatch_us,wait_us,wait_min_us,wait_max_us,reconstruct_us,total_us,wall_us
//! ```
//!
//! ```text
//! cargo bench -p emvp --features gpu --bench gpu_phase_sweep
//! ```

use std::io::Write;
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use emvp::{EncryptedQuery, GpuAnswerer, GpuEncryptedMatrix, GpuError, PhaseTimings, search};

mod common;

use common::{LLM_LAMBDA, MODULUS, protocol_fixtures_batch};

// Same shape as the answer-throughput suite: one 4096 x 4096-equivalent
// attention projection over 4096-element records.
const LLM_RECORD_LENGTH: usize = 4096;
const MATRIX_ROWS: usize = 4096;
const MAX_BATCH: usize = 2048;

// Sweep plan: (batch, timed calls, warmup calls). Non-power batches
// bracket the dip's edges; call counts shrink with per-call cost so one
// pass stays around two minutes.
const PLAN: &[(usize, usize, usize)] = &[
    (1, 24, 3),
    (2, 24, 3),
    (4, 24, 3),
    (8, 24, 3),
    (12, 16, 2),
    (16, 24, 3),
    (24, 16, 2),
    (32, 24, 3),
    (48, 16, 2),
    (64, 24, 3),
    (96, 8, 1),
    (128, 16, 2),
    (192, 6, 1),
    (256, 8, 1),
    (384, 4, 1),
    (512, 6, 1),
    (768, 2, 1),
    (1024, 3, 1),
    (1536, 2, 1),
    (2048, 3, 1),
];

// One median in fractional microseconds. Nanosecond counts stay far below
// f64's 53-bit mantissa for every duration here (the largest is seconds).
#[expect(
    clippy::cast_precision_loss,
    reason = "display-only microsecond rendering of sub-minute durations"
)]
fn median_us(mut samples: Vec<Duration>) -> f64 {
    samples.sort();
    let middle = samples.len() / 2;
    match samples.len() % 2 {
        0 => {
            let (low, high) = (samples[middle - 1], samples[middle]);
            (low.as_nanos() + high.as_nanos()) as f64 / 2_000.0
        }
        _ => samples[middle].as_nanos() as f64 / 1_000.0,
    }
}

#[expect(
    clippy::cast_precision_loss,
    reason = "display-only microsecond rendering of sub-minute durations"
)]
fn min_us(samples: &[Duration]) -> f64 {
    samples.iter().min().unwrap().as_nanos() as f64 / 1_000.0
}

#[expect(
    clippy::cast_precision_loss,
    reason = "display-only microsecond rendering of sub-minute durations"
)]
fn max_us(samples: &[Duration]) -> f64 {
    samples.iter().max().unwrap().as_nanos() as f64 / 1_000.0
}

/// Polls `nvidia-smi` in a loop, printing one comment line per sample.
///
/// Best effort: if the binary is missing or one call fails, the sampler
/// exits without disturbing the measurement.
fn spawn_clock_sampler(stop: Arc<AtomicBool>, started: Instant) -> std::thread::JoinHandle<()> {
    std::thread::spawn(move || {
        let query = [
            "clocks.sm",
            "clocks.mem",
            "temperature.gpu",
            "power.draw",
            "pstate",
        ]
        .join(",");
        while !stop.load(Ordering::Relaxed) {
            std::thread::sleep(Duration::from_millis(250));
            if stop.load(Ordering::Relaxed) {
                break;
            }
            let Ok(output) = std::process::Command::new("nvidia-smi")
                .arg(format!("--query-gpu={query}"))
                .args(["--format=csv,noheader"])
                .output()
            else {
                return;
            };
            let reading = String::from_utf8_lossy(&output.stdout).trim().to_string();
            if reading.is_empty() {
                return;
            }
            println!("#clock,{:.3}s,{reading}", started.elapsed().as_secs_f64());
            let _flush = std::io::stdout().flush();
        }
    })
}

/// Times one batch: `warmups` untuned calls, then `calls` timed calls.
fn measure_batch(
    answerer: &GpuAnswerer,
    gpu_matrix: &GpuEncryptedMatrix<MODULUS>,
    queries: &[EncryptedQuery<MODULUS>],
    batch: usize,
    calls: usize,
    warmups: usize,
) -> Vec<(PhaseTimings, Duration)> {
    for _ in 0..warmups {
        let (answers, _timings) = answerer
            .answer_batch_with_timings(gpu_matrix, &queries[..batch])
            .unwrap();
        assert_eq!(answers.len(), batch, "one answer per query");
    }
    let mut samples = Vec::with_capacity(calls);
    for _ in 0..calls {
        let wall_start = Instant::now();
        let (answers, timings) = answerer
            .answer_batch_with_timings(gpu_matrix, &queries[..batch])
            .unwrap();
        let wall = wall_start.elapsed();
        assert_eq!(answers.len(), batch, "one answer per query");
        drop(answers);
        samples.push((timings, wall));
    }
    samples
}

fn print_row(pass: &str, batch: usize, calls: usize, samples: &[(PhaseTimings, Duration)]) {
    let prepare = median_us(samples.iter().map(|(t, _)| t.prepare_buffers).collect());
    let encode = median_us(
        samples
            .iter()
            .map(|(t, _)| t.encode_upload_queries)
            .collect(),
    );
    let dispatch = median_us(samples.iter().map(|(t, _)| t.dispatch_submit).collect());
    let waits: Vec<Duration> = samples.iter().map(|(t, _)| t.wait_readback).collect();
    let wait = median_us(waits.clone());
    let reconstruct = median_us(samples.iter().map(|(t, _)| t.reconstruct).collect());
    let total = median_us(samples.iter().map(|(t, _)| t.total()).collect());
    let wall = median_us(samples.iter().map(|(_, w)| *w).collect());
    println!(
        "{pass},{batch},{calls},{prepare:.1},{encode:.1},{dispatch:.1},{wait:.1},{:.1},{:.1},{reconstruct:.1},{total:.1},{wall:.1}",
        min_us(&waits),
        max_us(&waits),
    );
    let _flush = std::io::stdout().flush();
}

fn run_pass(
    pass: &str,
    answerer: &GpuAnswerer,
    gpu_matrix: &GpuEncryptedMatrix<MODULUS>,
    queries: &[EncryptedQuery<MODULUS>],
    descending: bool,
) {
    let plan: Vec<&(usize, usize, usize)> = if descending {
        PLAN.iter().rev().collect()
    } else {
        PLAN.iter().collect()
    };
    for &(batch, calls, warmups) in plan {
        let samples = measure_batch(answerer, gpu_matrix, queries, batch, calls, warmups);
        print_row(pass, batch, calls, &samples);
    }
}

fn main() {
    let started = Instant::now();
    let answerer = match GpuAnswerer::new() {
        Ok(answerer) => answerer,
        Err(GpuError::NoAdapter { reason }) => {
            println!("skipping phase sweep: no compute adapter available ({reason})");
            return;
        }
        Err(error) => {
            let failure = Err::<GpuAnswerer, GpuError>(error);
            failure.unwrap()
        }
    };
    let params = search(LLM_RECORD_LENGTH, LLM_LAMBDA).unwrap();
    let n = params.n().unwrap();
    println!(
        "#gpu phase sweep at ell = {LLM_RECORD_LENGTH}, rows = {MATRIX_ROWS}: \
         k = {}, b = {}, n = {n}, lambda = {}, matrix {} MiB, max batch {MAX_BATCH}",
        params.k,
        params.b,
        params.lambda,
        MATRIX_ROWS * n * 4 / (1024 * 1024),
    );
    let (encrypted, queries, _decoding_keys) =
        protocol_fixtures_batch(params, MATRIX_ROWS, MAX_BATCH);
    let gpu_matrix = answerer.upload_matrix(&params, &encrypted).unwrap();

    println!(
        "#pass,batch,calls,prepare_us,encode_us,dispatch_us,wait_us,wait_min_us,wait_max_us,reconstruct_us,total_us,wall_us"
    );
    let stop = Arc::new(AtomicBool::new(false));
    let sampler = spawn_clock_sampler(Arc::clone(&stop), started);
    run_pass("asc1", &answerer, &gpu_matrix, &queries, false);
    run_pass("desc", &answerer, &gpu_matrix, &queries, true);
    run_pass("asc2", &answerer, &gpu_matrix, &queries, false);
    stop.store(true, Ordering::Relaxed);
    let _joined = sampler.join();
    println!("#done in {:.1}s", started.elapsed().as_secs_f64());
}
