#![expect(
    clippy::unwrap_used,
    reason = "benchmarks use fixed valid parameters and keep setup beside measurements"
)]

//! Fair single-core comparison of the EMVP online phase against the
//! plaintext matrix-vector product.
//!
//! Fairness contract, enforced end to end:
//!
//! - Single core: every measured closure runs inside a one-thread rayon
//!   pool, so every internal parallel decision in the library (the answer
//!   and decode dispatch policy and the mask evaluation inside `query`)
//!   deterministically selects its serial path; the pool also keeps the
//!   plaintext product off any helper threads.
//! - No GPU: the suite never enables the `gpu` feature and only measures
//!   the CPU protocol path.
//! - Same sizes: every case of one shape multiplies the same logical
//!   `rows x ell` matrix, and every case reports
//!   `Throughput::Elements(rows * ell)`, so criterion's elem/s column and
//!   the plaintext-to-protocol ratio compare directly across cases.
//! - Real protocol costs: `query` keeps its per-call allocations and PRF
//!   expansion, and the `total` case runs the full online round trip
//!   `query -> answer -> decode` per iteration, including the fresh answer
//!   buffer a real client receives from the network. The `total` time is
//!   the number to divide into the `plaintext` time for the protocol's
//!   overhead factor.
//!
//! Cases per shape, under the `online` group: `{toeplitz,raa,ring}/query`
//! and `{toeplitz,raa,ring}/total` for the three mask constructions, plus
//! the mask-independent `answer`, `decode`, and `plaintext` cases. Fixtures
//! are shared with the `protocol` suite (`mod common`), so numbers refer to
//! identical seeded artifacts.

use std::hint::black_box;

use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main, measurement::WallTime,
};
use emvp::{
    AnswerMatrix, DerivedState, EmvpParams, EncryptedMatrix, answer_into, decode_into, encrypt,
    query, search,
};
use prime_field_layer::arithmetic_kernels::dot_product;
use prime_field_layer::{FieldElement, PrimeField};
use rayon::ThreadPool;
use trapdoor_matrices::TdmMask;

mod common;

use common::{
    BlockBuilder, LLM_LAMBDA, LLM_RECORD_LENGTHS, LLM_ROW_COUNTS, MODULUS, MaskSuite, PARAMS,
    SUITE_RAA, SUITE_RING, SUITE_TOEPLITZ, bench_parameter, derive_with, elements, field_values,
    protocol_fixtures, raa_block, ring_block, serial_pool, suite_group, toeplitz_block,
};

// Quick-mode row counts for the legacy parameter set: they keep both sides
// of the n-row mask-block boundary (rows = n = 1024) plus the one-row tail
// block at 1025 in every case.
const QUICK_ROW_COUNTS: [usize; 3] = [128, 1024, 1025];

// A derived client state and its encrypted matrix, sharing one instance:
// everything the full online round trip needs.
struct OnlineFixtures<M: TdmMask<MODULUS>> {
    state: DerivedState<MODULUS, M>,
    encrypted: EncryptedMatrix<MODULUS>,
    record: Vec<FieldElement<MODULUS>>,
}

#[must_use]
fn online_fixtures<M: TdmMask<MODULUS>>(
    params: EmvpParams,
    rows: usize,
    suite: MaskSuite,
    build_block: BlockBuilder<M>,
) -> OnlineFixtures<M> {
    let mut state = derive_with(params, rows, 0x06, suite.context, build_block);
    let matrix = field_values(rows * params.ell, 0x07);
    let encrypted = encrypt(&mut state, &matrix).unwrap();
    let record = field_values(params.ell, 0x08);
    OnlineFixtures {
        state,
        encrypted,
        record,
    }
}

// One query generation per iteration. The stream keeps advancing across
// iterations, mirroring repeated queries with fresh randomness.
fn bench_query_case<M: TdmMask<MODULUS>>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    tag: &str,
    suite: MaskSuite,
    params: EmvpParams,
    rows: usize,
    build_block: BlockBuilder<M>,
    pool: &ThreadPool,
) {
    let mut state = derive_with(params, rows, 0x03, suite.context, build_block);
    let record = field_values(params.ell, 0x04);
    group.throughput(elements(rows * params.ell));
    group.bench_function(
        BenchmarkId::new(format!("{}/query", suite.label), bench_parameter(tag, rows)),
        |b| {
            b.iter(|| {
                pool.install(|| {
                    let (encrypted_query, decoding_key) =
                        query(black_box(&mut state), black_box(&record)).unwrap();
                    black_box((encrypted_query, decoding_key))
                })
            });
        },
    );
}

// One full online round trip per iteration: query generation, the server's
// answer, and the client's decode. The answer buffer is allocated per
// iteration because a real client receives the answer as fresh data over
// the transport; query artifacts and the decoded output follow the protocol
// API as written.
fn bench_total_case<M: TdmMask<MODULUS>>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    tag: &str,
    suite: MaskSuite,
    params: EmvpParams,
    rows: usize,
    build_block: BlockBuilder<M>,
    pool: &ThreadPool,
) {
    let OnlineFixtures {
        mut state,
        encrypted,
        record,
    } = online_fixtures(params, rows, suite, build_block);
    let zero = PrimeField::<MODULUS>::new().element_u32(0);
    let blocks = params.blocks().unwrap();
    let mut decoded = vec![zero; rows];
    group.throughput(elements(rows * params.ell));
    group.bench_function(
        BenchmarkId::new(format!("{}/total", suite.label), bench_parameter(tag, rows)),
        |b| {
            b.iter(|| {
                pool.install(|| {
                    let (encrypted_query, decoding_key) =
                        query(black_box(&mut state), black_box(&record)).unwrap();
                    let mut answer = vec![zero; rows * blocks];
                    answer_into(
                        black_box(&params),
                        black_box(&encrypted),
                        &encrypted_query,
                        black_box(&mut answer),
                    )
                    .unwrap();
                    let answer = AnswerMatrix::from_parts(
                        encrypted.instance_id(),
                        encrypted_query.query_id(),
                        answer,
                        rows,
                        blocks,
                    );
                    decode_into(
                        black_box(&answer),
                        black_box(&decoding_key),
                        black_box(&mut decoded),
                    )
                    .unwrap();
                });
            });
        },
    );
}

// The server's online work alone: `answer_into` into a reused output
// buffer, matching a server that serves many queries against one encrypted
// matrix.
fn bench_answer_case(
    group: &mut BenchmarkGroup<'_, WallTime>,
    tag: &str,
    params: EmvpParams,
    rows: usize,
    pool: &ThreadPool,
) {
    let (encrypted, encrypted_query, _decoding_key) = protocol_fixtures(params, rows);
    let zero = PrimeField::<MODULUS>::new().element_u32(0);
    let mut output = vec![zero; rows * params.blocks().unwrap()];
    group.throughput(elements(rows * params.ell));
    group.bench_function(
        BenchmarkId::new("answer", bench_parameter(tag, rows)),
        |b| {
            b.iter(|| {
                pool.install(|| {
                    answer_into(
                        black_box(&params),
                        black_box(&encrypted),
                        black_box(&encrypted_query),
                        black_box(&mut output),
                    )
                })
                .unwrap();
                black_box(&output);
            });
        },
    );
}

// The client's decryption work alone: decoding one precomputed answer into
// a reused output buffer.
fn bench_decode_case(
    group: &mut BenchmarkGroup<'_, WallTime>,
    tag: &str,
    params: EmvpParams,
    rows: usize,
    pool: &ThreadPool,
) {
    let (encrypted, encrypted_query, decoding_key) = protocol_fixtures(params, rows);
    let zero = PrimeField::<MODULUS>::new().element_u32(0);
    let blocks = params.blocks().unwrap();
    let mut answer_values = vec![zero; rows * blocks];
    answer_into(&params, &encrypted, &encrypted_query, &mut answer_values).unwrap();
    let answer_matrix = AnswerMatrix::from_parts(
        encrypted.instance_id(),
        encrypted_query.query_id(),
        answer_values,
        rows,
        blocks,
    );
    let mut output = vec![zero; rows];
    group.throughput(elements(rows * params.ell));
    group.bench_function(
        BenchmarkId::new("decode", bench_parameter(tag, rows)),
        |b| {
            b.iter(|| {
                pool.install(|| {
                    decode_into(
                        black_box(&answer_matrix),
                        black_box(&decoding_key),
                        black_box(&mut output),
                    )
                })
                .unwrap();
                black_box(&output);
            });
        },
    );
}

// The plaintext reference: one row-at-a-time Montgomery-form dot product,
// the kernel a plaintext deployment would use, on a single core.
fn bench_plaintext_case(
    group: &mut BenchmarkGroup<'_, WallTime>,
    tag: &str,
    params: EmvpParams,
    rows: usize,
    pool: &ThreadPool,
) {
    let matrix = field_values(rows * params.ell, 0x0a);
    let record = field_values(params.ell, 0x0b);
    let zero = PrimeField::<MODULUS>::new().element_u32(0);
    let mut output = vec![zero; rows];
    group.throughput(elements(rows * params.ell));
    group.bench_function(
        BenchmarkId::new("plaintext", bench_parameter(tag, rows)),
        |b| {
            b.iter(|| {
                pool.install(|| {
                    for (slot, matrix_row) in output.iter_mut().zip(matrix.chunks_exact(params.ell))
                    {
                        *slot = dot_product(matrix_row, &record).unwrap();
                    }
                });
                black_box(&output);
            });
        },
    );
}

// All cases for one parameter set and row count. Query generation and the
// full round trip run for every mask construction; the answer, decode, and
// plaintext cases are mask-independent and run once.
fn run_suite(criterion: &mut Criterion, tag: &str, params: EmvpParams, row_counts: &[usize]) {
    let huge = !tag.is_empty();
    let mut group = suite_group(criterion, "online", huge);
    let pool = serial_pool();
    for &rows in row_counts {
        bench_query_case(
            &mut group,
            tag,
            SUITE_TOEPLITZ,
            params,
            rows,
            toeplitz_block,
            pool,
        );
        bench_total_case(
            &mut group,
            tag,
            SUITE_TOEPLITZ,
            params,
            rows,
            toeplitz_block,
            pool,
        );
        bench_query_case(&mut group, tag, SUITE_RAA, params, rows, raa_block, pool);
        bench_total_case(&mut group, tag, SUITE_RAA, params, rows, raa_block, pool);
        bench_query_case(&mut group, tag, SUITE_RING, params, rows, ring_block, pool);
        bench_total_case(&mut group, tag, SUITE_RING, params, rows, ring_block, pool);
        bench_answer_case(&mut group, tag, params, rows, pool);
        bench_decode_case(&mut group, tag, params, rows, pool);
        bench_plaintext_case(&mut group, tag, params, rows, pool);
    }
    group.finish();
}

fn online_benches(criterion: &mut Criterion) {
    if bench_common::is_quick() {
        run_suite(criterion, "", PARAMS, &QUICK_ROW_COUNTS);
        return;
    }
    for &ell in &LLM_RECORD_LENGTHS {
        let params = search(ell, LLM_LAMBDA).unwrap();
        println!(
            "online suite at ell = {ell}: k = {}, b = {}, n = {}, lambda = {}",
            params.k,
            params.b,
            params.n().unwrap(),
            params.lambda
        );
        run_suite(criterion, &format!("ell{ell}"), params, &LLM_ROW_COUNTS);
    }
}

fn criterion_config() -> Criterion {
    bench_common::criterion_default()
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = online_benches
}
criterion_main!(benches);
