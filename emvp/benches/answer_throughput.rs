#![expect(
    clippy::unwrap_used,
    reason = "benchmarks use fixed valid parameters and keep setup beside measurement"
)]
#![expect(
    clippy::unnecessary_literal_unwrap,
    reason = "the engine and adapter probes wrap unexpected errors in `Err` so the single `unwrap` paths fail the bench loudly; static analysis flags the literals even though the errors are dynamic"
)]

//! Sustained matrix-vector-product throughput of the answer engine against
//! one fixed LLM-scale matrix.
//!
//! # Fairness contract and metric
//!
//! One fixed encrypted matrix (4096 rows over 4096-element records, the
//! plaintext-equivalent of one 7B-class attention projection) is prepared
//! on one [`AnswerEngine`] with the same `search(4096, 128)` protocol
//! parameters the `gpu` and `dispatch` suites use, so the codeword width
//! `n = 2k` is shared across the three suites. The preparation, the
//! maximum query set of 2048 encrypted queries, and the device upload all
//! happen strictly outside timing. Every measured iteration is exactly one
//! [`AnswerEngine::answer_many`] call with exactly one [`AnswerJob`]
//! holding `queries[..batch]`, sweeping the batch size over the powers of
//! two 1..=2048: the many-products-against-one-matrix serving pattern.
//!
//! Every case clears [`MIN_GPU_MULTIPLICATIONS`] on its own (a single
//! query against this matrix already costs `rows * n` estimated field
//! multiplications), so the fixed dispatch policy sends every case to the
//! device; this is asserted from the engine report before timing,
//! together with the batch answer count. [`Throughput::Elements`] counts
//! the batch, so Criterion's element/s figure is matrix-vector products
//! per second, directly.
//!
//! Per case the gate call's packed byte accounting is printed: the packed
//! query buffer, the packed output buffer, and the transient packed
//! scratch, which is the output buffer plus its equal-sized readback
//! staging copy, roughly `query + 2 * output` per call. The engine leases
//! these buffers per call and keeps their capacity warm across
//! iterations.
//!
//! Under `--features bench-quick` only the Criterion timing shortens:
//! the shapes and the batch sweep are pinned and cannot shrink. Without a
//! compute adapter the binary prints a notice and benchmarks nothing.
//!
//! # Case IDs and filters
//!
//! Each batch size runs one case under the `answer_mvp_throughput_v1`
//! group: `answer_mvp_throughput_v1/batch/B` for `B` in `1..=2048` powers
//! of two. Full sweep with a saved baseline:
//!
//! ```text
//! cargo bench -p emvp --features gpu --bench answer_throughput -- \
//!     --save-baseline answer-mvp-throughput-v1 'answer_mvp_throughput_v1'
//! ```
//!
//! Only the large cases (batch 256..=2048), also saving a baseline; the
//! `--` filter is a regular expression over the full case IDs:
//!
//! ```text
//! cargo bench -p emvp --features gpu --bench answer_throughput -- \
//!     --save-baseline answer-mvp-throughput-v1-large \
//!     'answer_mvp_throughput_v1/batch/(256|512|1024|2048)$'
//! ```
//!
//! For the quick-feedback configuration add `--features gpu,bench-quick`
//! (timing only; shapes and batches stay fixed).

use std::time::Duration;

use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main, measurement::WallTime,
};
use emvp::{
    AnswerBackend, AnswerEngine, AnswerJob, EncryptedQuery, GpuAnswerer, GpuError,
    MIN_GPU_MULTIPLICATIONS, PreparedMatrix, search,
};

mod common;

use common::{LLM_LAMBDA, MODULUS, elements, protocol_fixtures_batch};

// Engine device residency budget. The fixed matrix occupies
// `rows * n * 4` bytes, about 128 MiB, far below this bound.
const ENGINE_RESIDENCY_BYTES: u64 = 1 << 30;

// LLM-scale record length and matrix height: one 4096 x 4096 attention
// projection, with the parameter search shared by the `gpu` and `dispatch`
// suites' ell = 4096 cases.
const LLM_RECORD_LENGTH: usize = 4096;
const MATRIX_ROWS: usize = 4096;

// Query-batch sweep: powers of two from one query through 2048 queries.
// Each value is the matrix-vector product count of one measured iteration.
const BATCHES: [usize; 12] = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048];
const MAX_BATCH: usize = 2048;

/// Mebibyte rendering of a byte count for the per-case size report.
#[expect(
    clippy::cast_precision_loss,
    reason = "a display-only MiB approximation of large byte counts"
)]
fn mib(bytes: u64) -> f64 {
    bytes as f64 / (1024.0 * 1024.0)
}

/// Everything every case needs: the engine, its one prepared matrix, and
/// the maximum query set, prepared entirely outside timing.
struct SuiteContext {
    engine: AnswerEngine<MODULUS>,
    prepared: PreparedMatrix<MODULUS>,
    /// The maximum query set, generated once; each case slices it.
    queries: Vec<EncryptedQuery<MODULUS>>,
    n: usize,
}

/// Builds the engine, the fixture, and the query set.
///
/// Returns `None` when the machine has no compute adapter, after printing
/// the exact skip reason.
fn setup() -> Option<SuiteContext> {
    let engine = match AnswerEngine::<MODULUS>::new(ENGINE_RESIDENCY_BYTES) {
        Ok(engine) => engine,
        Err(error) => {
            let failure = Err::<AnswerEngine<MODULUS>, GpuError>(error);
            failure.unwrap()
        }
    };
    if !engine.has_device() {
        // `AnswerEngine::new` demotes only a missing adapter to a CPU
        // engine, so re-probe purely to report the exact reason.
        match GpuAnswerer::new() {
            Err(GpuError::NoAdapter { reason }) => {
                println!(
                    "skipping answer-throughput benches: no compute adapter available ({reason})"
                );
            }
            Err(error) => {
                let failure = Err::<GpuAnswerer, GpuError>(error);
                failure.unwrap();
            }
            Ok(_) => {
                let failure =
                    Err::<(), &str>("answer engine demoted to CPU although an adapter exists");
                failure.unwrap();
            }
        }
        return None;
    }
    let params = search(LLM_RECORD_LENGTH, LLM_LAMBDA).unwrap();
    let n = params.n().unwrap();
    println!(
        "answer throughput suite at ell = {LLM_RECORD_LENGTH}, rows = {MATRIX_ROWS}: \
         k = {}, b = {}, n = {n}, lambda = {}, matrix residency {} MiB, \
         max batch {MAX_BATCH}",
        params.k,
        params.b,
        params.lambda,
        MATRIX_ROWS * n * 4 / (1024 * 1024),
    );
    // The benchmark's premise: one query against the fixed matrix alone
    // clears the GPU tier of the fixed dispatch policy.
    assert!(
        MATRIX_ROWS * n >= MIN_GPU_MULTIPLICATIONS,
        "the single-query case must clear the GPU dispatch threshold on its own"
    );
    let (encrypted, queries, _decoding_keys) =
        protocol_fixtures_batch(params, MATRIX_ROWS, MAX_BATCH);
    // One-time prepare and upload; the measured iterations reuse the
    // device-resident matrix.
    let prepared = engine.prepare(params, encrypted).unwrap();
    Some(SuiteContext {
        engine,
        prepared,
        queries,
        n,
    })
}

/// Times one batch size: exactly one [`AnswerJob`] per iteration, after
/// the report gate has passed.
fn bench_case(group: &mut BenchmarkGroup<'_, WallTime>, context: &SuiteContext, batch: usize) {
    let n = context.n;
    let jobs = [AnswerJob {
        matrix: &context.prepared,
        queries: &context.queries[..batch],
    }];
    // Report gate, strictly before timing: the engine must select the GPU
    // tier for the case and output one answer per query.
    let (answers, report) = context.engine.answer_many_with_report(&jobs).unwrap();
    assert_eq!(answers.len(), 1);
    assert_eq!(
        answers[0].len(),
        batch,
        "the engine must output one answer per query"
    );
    assert_eq!(
        report.gpu_entries, 1,
        "the fixed dispatch policy must select the GPU tier for the case"
    );
    assert_eq!(
        report.cpu_entries, 0,
        "no case may fall back to the CPU tier"
    );
    assert!(
        report
            .entries
            .iter()
            .all(|entry| entry.backend == AnswerBackend::Gpu),
        "every entry must run on the device path"
    );
    assert_eq!(
        report.multiplications,
        batch * MATRIX_ROWS * n,
        "engine work accounting diverges from the batch estimate"
    );
    drop(answers);
    // Packed per-case byte accounting from the gate call (includes
    // alignment gaps): the packed query buffer, the packed output buffer,
    // and the transient scratch = output plus its equal-sized readback
    // staging copy.
    let query_bytes = report.gpu_query_bytes;
    let output_bytes = report.gpu_answer_bytes;
    let transient_bytes = query_bytes + output_bytes.saturating_mul(2);
    println!(
        "case batch{batch}: query {} bytes ({:.1} MiB), output {} bytes ({:.1} MiB), \
         transient {} bytes ({:.1} MiB)",
        query_bytes,
        mib(query_bytes),
        output_bytes,
        mib(output_bytes),
        transient_bytes,
        mib(transient_bytes),
    );
    // Elements counts matrix-vector products, so element/s is MVP/s.
    group.throughput(elements(batch));
    group.bench_function(BenchmarkId::new("batch", batch), |b| {
        b.iter(|| context.engine.answer_many(&jobs).unwrap());
    });
}

fn answer_throughput_benches(c: &mut Criterion) {
    let Some(context) = setup() else {
        return;
    };
    let mut group = c.benchmark_group("answer_mvp_throughput_v1");
    for &batch in &BATCHES {
        bench_case(&mut group, &context, batch);
    }
    group.finish();
}

fn criterion_config() -> Criterion {
    bench_common::criterion_tuned(10, Duration::from_secs(1), Duration::from_secs(3))
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = answer_throughput_benches
}
criterion_main!(benches);
