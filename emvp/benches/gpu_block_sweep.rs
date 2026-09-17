#![expect(
    clippy::unwrap_used,
    reason = "benchmarks use fixed valid parameters and keep setup beside measurement"
)]
#![expect(
    clippy::unnecessary_literal_unwrap,
    reason = "the adapter probe wraps an unexpected error in `Err` so the single `unwrap` path fails the bench loudly; static analysis flags the literal even though the error is dynamic"
)]

//! GPU answer-phase throughput across protocol block sizes at a fixed
//! query batch.
//!
//! # Why this exists
//!
//! The production parameter search ([`emvp::search`]) pins one block size
//! `b` per record length (the largest divisor of `n = 2k` that validates),
//! so the other GPU suites always measure the same decomposition. This
//! suite instead sweeps block sizes so answer-phase throughput can be
//! attributed to the decomposition itself: the kernel's fold structure
//! over `s = n / b` blocks of size `b`, the permuted matrix layout, and
//! the exact-vs-REDC final-fold path (`fold_fast_path` flips with `b`).
//! GPU only: the CPU-versus-GPU comparison is the `gpu` suite's job, and
//! the CPU reference would only repeat it.
//!
//! # Case shape
//!
//! One fixed protocol shape for every case: `ell = k = 4096` (n = 8192,
//! the 7B-class parameter set the production search emits),
//! `rows = 4096` (one attention projection), `batch = 64` queries per
//! measured iteration. The sweep runs `b in {8, 16, ..., 256}` — every
//! block size that validates at this rank, up to the `b = 256` the
//! production search picks — under the `gpu_block_sweep_v1` group,
//! `gpu_block_sweep_v1/bB-sS`. Because the rank (hence `n`, the matrix
//! width, and the total Montgomery-product work `batch * rows * n`) is
//! constant across the sweep, the only varying work is the decomposition
//! itself, and elem/s isolates its cost cleanly.
//!
//! The batch cannot rescue the tiny end: the server answer for one query
//! is `rows * s` words (`s = n / b` partial block products folded on the
//! host), so at `batch = 64` the answer buffer of `b = 4` alone needs
//! `2^29` words — one word past the device's per-buffer binding limit
//! (and `b = 2` needs twice that). Those sizes are excluded; the skip
//! itself is the deployment argument against tiny blocks.
//!
//! Fixtures (derive + encrypt + the query batch) and the one-time matrix
//! upload happen before timing; every measured iteration is one
//! [`GpuAnswerer::execute_answer_batch_into`] into the case's
//! once-reserved workspace, the steady-state plan-reserve-execute cycle.
//! [`Throughput::Elements`] counts `batch * rows * ell` logical elements,
//! identical for every case, so elem/s compares directly across the sweep
//! and with the other suites.
//!
//! # Correctness pins
//!
//! Once per case, outside Criterion timing: the answer of the first query
//! must decode to the plaintext `matrix · record`, checked word-for-word
//! against the CPU `dot_product` per row. This matters here more than in
//! the other suites because every off-search block size is a path the
//! production configuration never exercises. The pin call doubles as the
//! workspace's warm-up.
//!
//! Without a compute adapter the binary prints a notice and benchmarks
//! nothing.
//!
//! ```text
//! cargo bench -p emvp --features gpu --bench gpu_block_sweep
//! ```

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use emvp::{
    AnswerPlan, AnswerWorkspace, EmvpParams, GpuAnswerer, GpuEncryptedMatrix, GpuError,
    decode_into, encrypt, query_batch,
};
use prime_field_layer::arithmetic_kernels::dot_product;
use prime_field_layer::{FieldElement, PrimeField};

use emvp_bench_common as common;

use common::{
    CONTEXT_TOEPLITZ, LLM_LAMBDA, MODULUS, derive_with, elements, field_values, toeplitz_block,
};

// The record length and matrix height, identical for every case: the
// 7B-class deployment shape (one attention projection).
const ELL: usize = 4096;
const ROWS: usize = 4096;
// The query batch, identical for every case: large enough for the tiled
// kernel regime the other suites' throughput cases run in.
const BATCH: usize = 64;

// Block sizes to sweep, ascending: every size that validates at the
// sweep's fixed rank `k = ell`, up to the production `b = 256`. Smaller
// sizes cannot serve `batch = 64` within the device's per-buffer binding
// limit and are deliberately absent.
const BLOCK_SIZES: [usize; 6] = [8, 16, 32, 64, 128, 256];

/// The parameter set for one block size at the sweep's fixed rank.
fn params_for_block(b: usize) -> EmvpParams {
    let params = EmvpParams {
        k: ELL,
        ell: ELL,
        b,
        lambda: LLM_LAMBDA,
    };
    assert!(
        params.validate().is_ok(),
        "block size {b} does not validate at k = {ELL}, lambda = {LLM_LAMBDA}"
    );
    params
}

/// One case's protocol instance and device state, built entirely outside
/// timing. The answer plan is not stored: the suite loop plans and
/// reserves the workspace per case, exactly like the other suites.
struct CaseFixtures {
    gpu_matrix: GpuEncryptedMatrix<MODULUS>,
    encrypted: emvp::EncryptedMatrix<MODULUS>,
    queries: Vec<emvp::EncryptedQuery<MODULUS>>,
    decoding_keys: Vec<emvp::DecodingKey<MODULUS>>,
    /// Plaintext record the queries encrypt; the correctness pin compares
    /// decoded answers against `matrix · record`.
    record: Vec<FieldElement<MODULUS>>,
    matrix: Vec<FieldElement<MODULUS>>,
}

/// Builds one case's fixtures: derive, encrypt, upload, query batch, plan.
fn build_fixtures(answerer: &GpuAnswerer, params: EmvpParams) -> CaseFixtures {
    let mut state = derive_with(params, ROWS, 0x06, CONTEXT_TOEPLITZ, toeplitz_block);
    let matrix = field_values(ROWS * params.ell, 0x07);
    let record = field_values(params.ell, 0x08);
    let encrypted = encrypt(&mut state, &matrix).unwrap();
    let gpu_matrix = answerer.upload_matrix(&params, &encrypted).unwrap();
    let record_slices: Vec<&[FieldElement<MODULUS>]> = vec![record.as_slice(); BATCH];
    let pairs = query_batch(&mut state, &record_slices).unwrap();
    let (queries, decoding_keys): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
    CaseFixtures {
        gpu_matrix,
        encrypted,
        queries,
        decoding_keys,
        record,
        matrix,
    }
}

/// One-time correctness pin: the first query's answer must decode to the
/// plaintext matrix-vector product, row for row. Runs outside timing and
/// doubles as the case workspace's warm-up.
fn check_decode_parity(
    answerer: &GpuAnswerer,
    fixtures: &CaseFixtures,
    workspace: &mut AnswerWorkspace<MODULUS>,
) {
    let CaseFixtures {
        gpu_matrix,
        encrypted: _,
        queries,
        decoding_keys,
        record,
        matrix,
    } = fixtures;
    let (answers, _timings) = answerer
        .execute_answer_batch_into(gpu_matrix, queries, workspace)
        .unwrap();
    let mut decoded = vec![PrimeField::<MODULUS>::new().element_u32(0); ROWS];
    if let Some((answer, key)) = answers
        .iter(queries)
        .unwrap()
        .zip(decoding_keys.iter())
        .next()
    {
        decode_into(&answer, key, &mut decoded).unwrap();
    }
    for (row, decoded_value) in decoded.iter().enumerate() {
        let expected = dot_product(&matrix[row * ELL..(row + 1) * ELL], record).unwrap();
        assert_eq!(
            decoded_value.to_raw(),
            expected.to_raw(),
            "block-size case: decoded row {row} diverges from plaintext matvec"
        );
    }
}

fn gpu_block_sweep_benches(c: &mut Criterion) {
    let answerer = match GpuAnswerer::new() {
        Ok(answerer) => answerer,
        Err(GpuError::NoAdapter { reason }) => {
            println!("skipping gpu block sweep: no compute adapter available ({reason})");
            return;
        }
        Err(error) => {
            let failure = Err::<GpuAnswerer, GpuError>(error);
            failure.unwrap()
        }
    };
    println!(
        "gpu block sweep at ell = k = {ELL}, rows = {ROWS}, batch = {BATCH}, \
         lambda = {LLM_LAMBDA}: one case per block size at fixed rank"
    );
    let mut group = c.benchmark_group("gpu_block_sweep_v1");
    group.sample_size(10);
    for &b in &BLOCK_SIZES {
        let params = params_for_block(b);
        let n = params.n().unwrap();
        let s = params.blocks().unwrap();
        let tag = format!("b{b}-s{s}");
        println!(
            "case {tag}: k = {}, b = {b}, n = {n}, s = {s}, batch = {BATCH}, \
             lambda = {}",
            params.k, params.lambda
        );
        let fixtures = build_fixtures(&answerer, params);
        let gpu_shape = answerer
            .answer_batch_plan(&fixtures.gpu_matrix, &fixtures.queries)
            .unwrap();
        let host_plan = AnswerPlan::plan(&params, &fixtures.encrypted, &fixtures.queries).unwrap();
        assert_eq!(gpu_shape, host_plan.shape(), "tier shapes must agree");
        let mut workspace = AnswerWorkspace::<MODULUS>::new();
        workspace.reserve(&host_plan).unwrap();
        check_decode_parity(&answerer, &fixtures, &mut workspace);

        group.throughput(elements(BATCH * ROWS * ELL));
        group.bench_function(BenchmarkId::new("b", tag), |bench| {
            bench.iter(|| {
                // The answers view is dropped here; the workspace's arena
                // stays reserved for the next iteration.
                let (answers, _timings) = answerer
                    .execute_answer_batch_into(
                        &fixtures.gpu_matrix,
                        &fixtures.queries,
                        &mut workspace,
                    )
                    .unwrap();
                std::hint::black_box(&answers);
            });
        });
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
    targets = gpu_block_sweep_benches
}
criterion_main!(benches);
