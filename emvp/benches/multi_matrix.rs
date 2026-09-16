#![expect(
    clippy::unwrap_used,
    reason = "benchmarks use fixed valid parameters and keep setup beside measurement"
)]
#![expect(
    clippy::unnecessary_literal_unwrap,
    reason = "the engine and adapter probes wrap unexpected errors in `Err` so the single `unwrap` paths fail the bench loudly; static analysis flags the literals even though the errors are dynamic"
)]

//! Packed multi-matrix answering versus the pre-engine per-matrix loop.
//!
//! # Fairness contract
//!
//! Both sides answer the same heterogeneous matrix set with the same
//! per-matrix query batches on the same device. The baseline is the
//! server's answer loop before commit 086a519: one
//! [`GpuAnswerer::answer_batch`] per stored matrix, so one submit/wait
//! round trip per matrix per iteration. The packed side is one
//! [`AnswerEngine::answer_many`] call over the whole job set: one
//! planning pass, every query batch packed into shared device buffers,
//! and exactly one submit and one wait. The shapes are heterogeneous:
//! matrix heights grow with the matrix index, query batch lengths cycle
//! `1..=3`, and every matrix is an independent protocol instance with its
//! own instance identifier (distinct derivation domains). Every entry
//! clears [`MIN_GPU_MULTIPLICATIONS`] on its own, so the fixed dispatch
//! policy selects the GPU tier for each entry of each case; this is
//! asserted from the engine report before timing.
//!
//! Fixture construction (derive + encrypt + queries) and both one-time
//! upload paths (the baseline answerer's uploads and the engine's
//! `prepare_batch`) happen strictly before any timed iteration. Before
//! any measurement the packed answers are asserted bit-identical to the
//! per-matrix answers, the report must select [`AnswerBackend::Gpu`] for
//! every entry, and its multiplication accounting must equal the local
//! estimate. Throughput is the estimated field-multiplication work
//! `queries * rows * n` summed over the case's entries, matching the
//! dispatch suite's metric.
//!
//! Under `--features bench-quick` only the Criterion timing shortens: the
//! shapes are pinned by the GPU dispatch threshold and cannot shrink.
//! Without a compute adapter the binary prints a notice and benchmarks
//! nothing.
//!
//! # Case IDs and filters
//!
//! Each matrix count runs two cases under the `answer_engine_multi_v1`
//! group: `answer_engine_multi_v1/per-matrix/countC` (the repeated
//! `answer_batch` baseline) and `answer_engine_multi_v1/packed/countC`
//! (one `answer_many` call):
//!
//! ```text
//! cargo bench -p emvp --features gpu --bench multi_matrix -- \
//!     --save-baseline multi-engine-v1 'answer_engine_multi_v1/(packed|per-matrix)'
//! ```
//!
//! For the quick-feedback configuration add `--features gpu,bench-quick`.

use std::time::Duration;

use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use emvp::{
    AnswerBackend, AnswerEngine, AnswerJob, AnswerMatrix, EmvpParams, EncryptedMatrix,
    EncryptedQuery, GpuAnswerer, GpuEncryptedMatrix, GpuError, MIN_GPU_MULTIPLICATIONS,
    PreparedMatrix, encrypt, query, search,
};

mod common;

use common::{CONTEXT_TOEPLITZ, LLM_LAMBDA, MODULUS, derive_with, field_values, toeplitz_block};

// Engine device residency budget. The largest case holds `8 * 640` matrix
// rows of `n = 8192` words, about 109 MiB, far below this bound.
const ENGINE_RESIDENCY_BYTES: u64 = 1 << 30;

// Matrix counts swept by the suite; each case answers the first `count`
// fixtures of the shared fixture set.
const MATRIX_COUNTS: [usize; 3] = [2, 4, 8];

// LLM-scale record length for the suite's parameter set, the same search
// the `gpu` and `dispatch` suites run, so the three suites share their
// `n = 8192` query width.
const LLM_RECORD_LENGTH: usize = 4096;

// Heterogeneous matrix heights: `ROWS_BASE + ROWS_STEP * index` keeps the
// eight fixtures distinct while the smallest (192 rows) still clears the
// GPU dispatch threshold with a single query.
const ROWS_BASE: usize = 192;
const ROWS_STEP: usize = 64;

// Distinct derivation domain per fixture: distinct instance identifier, so
// every matrix is an independent protocol instance. The suite never
// exceeds eight fixtures, far below the `u8` bound.
const fn domain_for(index: usize) -> u8 {
    0x50 + index as u8
}

const fn rows_for(index: usize) -> usize {
    ROWS_BASE + index * ROWS_STEP
}

// Query batch lengths cycle `1..=3` so the packed planner sees mixed
// batch sizes within one case.
const fn queries_for(index: usize) -> usize {
    1 + index % 3
}

/// Estimated field multiplications `queries * rows * n` summed over the
/// first `count` fixtures, the throughput unit of every case.
fn case_multiplications(n: usize, count: usize) -> usize {
    (0..count)
        .map(|index| queries_for(index) * rows_for(index) * n)
        .sum()
}

/// One heterogeneous fixture set: encrypted matrices and their matching
/// query batches, in fixture order.
struct MultiFixtures {
    matrices: Vec<EncryptedMatrix<MODULUS>>,
    query_sets: Vec<Vec<EncryptedQuery<MODULUS>>>,
}

/// Builds `count` independent protocol instances from the shared seeded
/// builders, each with its own derivation domain, height, and query batch
/// length.
fn build_fixtures(params: EmvpParams, count: usize) -> MultiFixtures {
    let n = params.n().unwrap();
    let mut matrices = Vec::with_capacity(count);
    let mut query_sets = Vec::with_capacity(count);
    for index in 0..count {
        let rows = rows_for(index);
        let domain = domain_for(index);
        let mut state = derive_with(params, rows, domain, CONTEXT_TOEPLITZ, toeplitz_block);
        let matrix = field_values(rows * params.ell, domain ^ 0x01);
        let record = field_values(params.ell, domain ^ 0x02);
        let encrypted = encrypt(&mut state, &matrix).unwrap();
        let mut queries = Vec::with_capacity(queries_for(index));
        for _ in 0..queries_for(index) {
            let (encrypted_query, _decoding_key) = query(&mut state, &record).unwrap();
            queries.push(encrypted_query);
        }
        // The benchmark's premise: each entry alone clears the GPU tier of
        // the fixed dispatch policy.
        assert!(
            queries.len() * rows * n >= MIN_GPU_MULTIPLICATIONS,
            "fixture {index} must clear the GPU dispatch threshold on its own"
        );
        query_sets.push(queries);
        matrices.push(encrypted);
    }
    MultiFixtures {
        matrices,
        query_sets,
    }
}

/// Everything the per-count cases need, prepared entirely outside timing.
struct SuiteContext {
    engine: AnswerEngine<MODULUS>,
    answerer: GpuAnswerer,
    n: usize,
    fixtures: MultiFixtures,
    /// Baseline-side device-resident matrices, in fixture order.
    gpu_matrices: Vec<GpuEncryptedMatrix<MODULUS>>,
    /// Engine-side prepared handles, in fixture order.
    prepared: Vec<PreparedMatrix<MODULUS>>,
}

/// Builds the engine, the fixtures, and both upload paths.
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
                println!("skipping multi-matrix benches: no compute adapter available ({reason})");
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
    let answerer = match GpuAnswerer::new() {
        Ok(answerer) => answerer,
        Err(error) => {
            let failure = Err::<GpuAnswerer, GpuError>(error);
            failure.unwrap()
        }
    };
    let params = search(LLM_RECORD_LENGTH, LLM_LAMBDA).unwrap();
    let n = params.n().unwrap();
    println!(
        "multi-matrix suite at ell = {LLM_RECORD_LENGTH}: k = {}, b = {}, n = {n}, lambda = {}",
        params.k, params.b, params.lambda
    );
    let max_count = MATRIX_COUNTS.into_iter().max().unwrap();
    let fixtures = build_fixtures(params, max_count);
    // One-time uploads: the baseline answerer keeps its own device copies
    // and the engine prepares and uploads its own set; both happen
    // strictly before any timed iteration.
    let gpu_matrices: Vec<GpuEncryptedMatrix<MODULUS>> = fixtures
        .matrices
        .iter()
        .map(|matrix| answerer.upload_matrix(&params, matrix).unwrap())
        .collect();
    let prepared = engine
        .prepare_batch(
            fixtures
                .matrices
                .iter()
                .map(|matrix| (params, matrix.clone())),
        )
        .unwrap();
    Some(SuiteContext {
        engine,
        answerer,
        n,
        fixtures,
        gpu_matrices,
        prepared,
    })
}

/// Times one matrix count: the per-matrix baseline and the packed engine,
/// after the equivalence gate has passed.
fn bench_case(group: &mut BenchmarkGroup<'_, WallTime>, context: &SuiteContext, count: usize) {
    let tag = format!("count{count}");
    let n = context.n;
    let jobs: Vec<AnswerJob<MODULUS>> = context.prepared[..count]
        .iter()
        .zip(context.fixtures.query_sets[..count].iter())
        .map(|(matrix, queries)| AnswerJob { matrix, queries })
        .collect();
    let multiplications = case_multiplications(n, count);
    // Equivalence gate, strictly before timing: the packed engine must
    // reproduce the per-matrix answers bit for bit, select the GPU tier
    // for every entry under the fixed policy, and agree with the local
    // work estimate.
    let baseline_answers: Vec<Vec<AnswerMatrix<MODULUS>>> = context.gpu_matrices[..count]
        .iter()
        .zip(context.fixtures.query_sets[..count].iter())
        .map(|(matrix, queries)| context.answerer.answer_batch(matrix, queries).unwrap())
        .collect();
    let (packed_answers, report) = context.engine.answer_many_with_report(&jobs).unwrap();
    assert_eq!(packed_answers.len(), count);
    for (entry, (packed, per_matrix)) in packed_answers.iter().zip(&baseline_answers).enumerate() {
        assert_eq!(
            packed, per_matrix,
            "packed answers diverge from the per-matrix answers at entry {entry}"
        );
    }
    assert_eq!(
        report.gpu_entries, count,
        "the fixed dispatch policy must select the GPU tier for every entry"
    );
    assert!(
        report
            .entries
            .iter()
            .all(|entry| entry.backend == AnswerBackend::Gpu),
        "every entry must run on the device path"
    );
    assert_eq!(
        report.multiplications, multiplications,
        "engine work accounting diverges from the local estimate"
    );
    drop(packed_answers);
    drop(baseline_answers);
    println!(
        "multi-matrix case {tag}: {count} matrices, rows {}..{}, {multiplications} field multiplications per iteration",
        rows_for(0),
        rows_for(count - 1),
    );
    group.throughput(Throughput::Elements(
        u64::try_from(multiplications).unwrap(),
    ));
    // Baseline: the old server loop, one answer_batch (one submit/wait)
    // per matrix.
    group.bench_function(BenchmarkId::new("per-matrix", tag.clone()), |b| {
        b.iter(|| {
            context.gpu_matrices[..count]
                .iter()
                .zip(context.fixtures.query_sets[..count].iter())
                .map(|(matrix, queries)| context.answerer.answer_batch(matrix, queries).unwrap())
                .collect::<Vec<_>>()
        });
    });
    // Packed: one answer_many call for the whole set (one submit, one
    // wait).
    group.bench_function(BenchmarkId::new("packed", tag), |b| {
        b.iter(|| context.engine.answer_many(&jobs).unwrap());
    });
}

fn multi_matrix_benches(c: &mut Criterion) {
    let Some(context) = setup() else {
        return;
    };
    let mut group = c.benchmark_group("answer_engine_multi_v1");
    for &count in &MATRIX_COUNTS {
        bench_case(&mut group, &context, count);
    }
    group.finish();
}

fn criterion_config() -> Criterion {
    bench_common::criterion_tuned(10, Duration::from_secs(1), Duration::from_secs(3))
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = multi_matrix_benches
}
criterion_main!(benches);
