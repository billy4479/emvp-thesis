#![expect(
    clippy::unwrap_used,
    reason = "benchmarks use fixed valid parameters and keep setup beside measurements"
)]

use std::hint::black_box;

use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main,
    measurement::WallTime,
};
use emvp::{
    AnswerPlan, AnswerRef, AnswerWorkspace, EmvpParams, SecretKey, answer_into, decode_into,
    encrypt, execute_answer_batch, query, query_batch, search,
};
use prime_field_layer::{FieldElement, PrimeField};
use rayon::prelude::*;
use trapdoor_matrices::TdmMask;

use emvp_bench_common as common;

use common::{
    BlockBuilder, LLM_LAMBDA, LLM_RECORD_LENGTHS, LLM_ROW_COUNTS, MODULUS, MaskSuite, PARAMS,
    SUITE_RAA, SUITE_RING, SUITE_TOEPLITZ, bench_parameter, derive_with, elements, field_values,
    protocol_fixtures, protocol_fixtures_batch, raa_block, ring_block, seeded_rng, suite_group,
    toeplitz_block,
};

// Quick-mode row counts, picked to keep both sides of the n-row mask-block
// boundary (rows = n = 1024) in every phase while staying fast.
const QUICK_DERIVE_ROW_COUNTS: [usize; 2] = [128, 1025];
const QUICK_CLIENT_ROW_COUNTS: [usize; 3] = [128, 1024, 1025];
const QUICK_SERVER_ROW_COUNTS: [usize; 2] = [128, 1024];
// Cases around the serial-to-parallel answer crossover; calibration-only
// because that boundary depends on the machine's pool size.
const ANSWER_CALIBRATION_ROW_COUNTS: [usize; 3] = [16, 31, 32];
// Batch size of the answer_batch cases.
const ANSWER_BATCH: usize = 4;
// Batch size of the query_batch cases.
const QUERY_BATCH: usize = 4;

// Row counts per phase for one parameter set.
struct SuiteRows {
    derive: &'static [usize],
    client: &'static [usize],
    server: &'static [usize],
}

const QUICK_ROWS: SuiteRows = SuiteRows {
    derive: &QUICK_DERIVE_ROW_COUNTS,
    client: &QUICK_CLIENT_ROW_COUNTS,
    server: &QUICK_SERVER_ROW_COUNTS,
};

fn bench_derive_for<M: TdmMask<MODULUS>>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    tag: &str,
    suite: MaskSuite,
    params: EmvpParams,
    rows: usize,
    build_block: BlockBuilder<M>,
) {
    group.bench_function(
        BenchmarkId::new(suite.label, bench_parameter(tag, rows)),
        |b| {
            b.iter(|| {
                let mut rng = seeded_rng(0xf2, rows);
                black_box(
                    SecretKey::<MODULUS>::new(params, [0x72; 32])
                        .unwrap()
                        .derive(suite.context, rows, &mut rng, |stream, index| {
                            build_block(params, stream, index)
                        })
                        .unwrap(),
                )
            });
        },
    );
}

fn bench_encrypt_for<M: TdmMask<MODULUS>>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    tag: &str,
    suite: MaskSuite,
    params: EmvpParams,
    rows: usize,
    build_block: BlockBuilder<M>,
) {
    let matrix = field_values(rows * params.ell, 0x02);
    group.throughput(elements(rows * params.ell));
    group.bench_function(
        BenchmarkId::new(suite.label, bench_parameter(tag, rows)),
        |b| {
            b.iter_batched(
                || derive_with(params, rows, 0x01, suite.context, build_block),
                |mut state| black_box(encrypt(black_box(&mut state), black_box(&matrix)).unwrap()),
                BatchSize::SmallInput,
            );
        },
    );
}

// One query per iteration on rayon's global pool: the mask evaluation
// inside `query` sizes its parallel decision against the machine's pool.
fn bench_query_for<M: TdmMask<MODULUS>>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    tag: &str,
    suite: MaskSuite,
    params: EmvpParams,
    rows: usize,
    build_block: BlockBuilder<M>,
) {
    let mut state = derive_with(params, rows, 0x03, suite.context, build_block);
    let record = field_values(params.ell, 0x04);
    // The stream keeps advancing across iterations, mirroring repeated
    // queries with fresh randomness.
    group.bench_function(
        BenchmarkId::new(suite.label, bench_parameter(tag, rows)),
        |b| {
            b.iter(|| {
                let (encrypted_query, decoding_key) =
                    query(black_box(&mut state), black_box(&record)).unwrap();
                black_box((encrypted_query, decoding_key))
            });
        },
    );
}

// One query batch per iteration, either through `query_batch` or, as the
// pre-batch reference, through a sequential `query` loop holding the same
// derived state. The counter keeps advancing across iterations, so both
// variants generate fresh randomness each time.
fn bench_query_batch_for<M: TdmMask<MODULUS>>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    tag: &str,
    suite: MaskSuite,
    params: EmvpParams,
    rows: usize,
    batch: usize,
    build_block: BlockBuilder<M>,
) {
    let mut state = derive_with(params, rows, 0x03, suite.context, build_block);
    let record = field_values(params.ell, 0x04);
    let queries: Vec<&[FieldElement<MODULUS>]> = (0..batch).map(|_| record.as_slice()).collect();
    group.throughput(elements(batch * params.ell));
    group.bench_function(
        BenchmarkId::new(format!("batch{batch}"), bench_parameter(tag, rows)),
        |b| {
            b.iter(|| black_box(query_batch(black_box(&mut state), &queries).unwrap()));
        },
    );
    group.bench_function(
        BenchmarkId::new(format!("serial{batch}"), bench_parameter(tag, rows)),
        |b| {
            b.iter(|| {
                for _ in 0..batch {
                    let artifacts = query(black_box(&mut state), black_box(&record)).unwrap();
                    black_box(artifacts);
                }
            });
        },
    );
}

fn bench_plaintext(
    group: &mut BenchmarkGroup<'_, WallTime>,
    tag: &str,
    params: EmvpParams,
    rows: usize,
) {
    let matrix = field_values(rows * params.ell, 0x0a);
    let query = field_values(params.ell, 0x0b);
    let zero = PrimeField::<MODULUS>::new().element_u32(0);
    let mut output = vec![zero; rows];
    group.throughput(elements(rows * params.ell));
    group.bench_function(
        BenchmarkId::from_parameter(bench_parameter(tag, rows)),
        |b| {
            b.iter(|| {
                output
                    .par_iter_mut()
                    .zip(matrix.par_chunks(params.ell))
                    .for_each(|(slot, matrix_row)| {
                        let mut accumulator = zero;
                        for (&coefficient, &value) in matrix_row.iter().zip(&query) {
                            accumulator += coefficient * value;
                        }
                        *slot = accumulator;
                    });
                black_box(&output);
            });
        },
    );
}

fn bench_answer(
    group: &mut BenchmarkGroup<'_, WallTime>,
    tag: &str,
    params: EmvpParams,
    rows: usize,
) {
    let (encrypted, encrypted_query, _decoding_key) = protocol_fixtures(params, rows);
    let zero = PrimeField::<MODULUS>::new().element_u32(0);
    let mut output = vec![zero; rows * params.blocks().unwrap()];
    group.throughput(elements(rows * params.n().unwrap()));
    group.bench_function(
        BenchmarkId::from_parameter(bench_parameter(tag, rows)),
        |b| {
            b.iter(|| {
                answer_into(
                    black_box(&params),
                    black_box(&encrypted),
                    black_box(&encrypted_query),
                    black_box(&mut output),
                )
                .unwrap();
                black_box(&output);
            });
        },
    );
}

// The batched server answer answers `batch` queries against one encrypted
// matrix through the plan-reserve-execute path: the plan fixes the CPU tier
// from a snapshot of the machine's global pool, the workspace is reserved
// once per case, and every iteration executes the plan into the reused
// workspace, parallelizing the flattened (query, row) grid internally
// without allocating.
fn bench_answer_batch(
    group: &mut BenchmarkGroup<'_, WallTime>,
    tag: &str,
    params: EmvpParams,
    rows: usize,
    batch: usize,
) {
    let (encrypted, queries, _decoding_keys) = protocol_fixtures_batch(params, rows, batch);
    let plan = AnswerPlan::plan(&params, &encrypted, &queries).unwrap();
    let mut workspace = AnswerWorkspace::new();
    workspace.reserve(&plan).unwrap();
    group.throughput(elements(batch * rows * params.n().unwrap()));
    group.bench_function(
        BenchmarkId::new(format!("batch{batch}"), bench_parameter(tag, rows)),
        |b| {
            b.iter(|| {
                execute_answer_batch(black_box(&plan), &mut workspace).unwrap();
            });
        },
    );
}

fn bench_decode(
    group: &mut BenchmarkGroup<'_, WallTime>,
    tag: &str,
    params: EmvpParams,
    rows: usize,
) {
    let (encrypted, encrypted_query, decoding_key) = protocol_fixtures(params, rows);
    let zero = PrimeField::<MODULUS>::new().element_u32(0);
    let blocks = params.blocks().unwrap();
    let mut answer_values = vec![zero; rows * blocks];
    answer_into(&params, &encrypted, &encrypted_query, &mut answer_values).unwrap();
    let answer_ref = AnswerRef::new(
        encrypted.instance_id(),
        encrypted_query.query_id(),
        &answer_values,
        rows,
        blocks,
    )
    .unwrap();
    let mut output = vec![zero; rows];
    group.throughput(elements(rows * blocks));
    group.bench_function(
        BenchmarkId::from_parameter(bench_parameter(tag, rows)),
        |b| {
            b.iter(|| {
                decode_into(
                    black_box(&answer_ref),
                    black_box(&decoding_key),
                    black_box(&mut output),
                )
                .unwrap();
                black_box(&output);
            });
        },
    );
}

// Query phase for one parameter set: single queries for every block
// construction plus batched toeplitz cases in both parallel and sequential
// loop variants.
fn run_query_suite(criterion: &mut Criterion, tag: &str, params: EmvpParams, counts: &SuiteRows) {
    let huge = !tag.is_empty();
    let mut query_group = suite_group(criterion, "query", huge);
    for &rows in counts.client {
        bench_query_for(
            &mut query_group,
            tag,
            SUITE_TOEPLITZ,
            params,
            rows,
            toeplitz_block,
        );
        bench_query_for(&mut query_group, tag, SUITE_RAA, params, rows, raa_block);
        bench_query_for(&mut query_group, tag, SUITE_RING, params, rows, ring_block);
    }
    for &rows in counts.client {
        bench_query_batch_for(
            &mut query_group,
            tag,
            SUITE_TOEPLITZ,
            params,
            rows,
            QUERY_BATCH,
            toeplitz_block,
        );
    }
    query_group.finish();
}

// All phases for one parameter set. The empty tag marks the legacy
// quick-mode parameter set and unlocks its calibration cases.
fn run_suite(criterion: &mut Criterion, tag: &str, params: EmvpParams, counts: &SuiteRows) {
    let huge = !tag.is_empty();

    {
        let mut derive_group = suite_group(criterion, "derive", huge);
        for &rows in counts.derive {
            bench_derive_for(
                &mut derive_group,
                tag,
                SUITE_TOEPLITZ,
                params,
                rows,
                toeplitz_block,
            );
            bench_derive_for(&mut derive_group, tag, SUITE_RAA, params, rows, raa_block);
            bench_derive_for(&mut derive_group, tag, SUITE_RING, params, rows, ring_block);
        }
        derive_group.finish();
    }

    {
        let mut encrypt_group = suite_group(criterion, "encrypt", huge);
        for &rows in counts.client {
            bench_encrypt_for(
                &mut encrypt_group,
                tag,
                SUITE_TOEPLITZ,
                params,
                rows,
                toeplitz_block,
            );
            bench_encrypt_for(&mut encrypt_group, tag, SUITE_RAA, params, rows, raa_block);
            bench_encrypt_for(
                &mut encrypt_group,
                tag,
                SUITE_RING,
                params,
                rows,
                ring_block,
            );
        }
        encrypt_group.finish();
    }

    run_query_suite(criterion, tag, params, counts);

    {
        let mut answer_group = suite_group(criterion, "answer", huge);
        for &rows in counts.server {
            bench_answer(&mut answer_group, tag, params, rows);
        }
        for &rows in counts.server {
            bench_answer_batch(&mut answer_group, tag, params, rows, ANSWER_BATCH);
        }
        if !huge && bench_common::calibration_enabled() {
            for &rows in &ANSWER_CALIBRATION_ROW_COUNTS {
                bench_answer(&mut answer_group, tag, params, rows);
            }
        }
        answer_group.finish();
    }

    {
        let mut decode_group = suite_group(criterion, "decode", huge);
        for &rows in counts.server {
            bench_decode(&mut decode_group, tag, params, rows);
        }
        decode_group.finish();
    }

    {
        let mut plaintext_group = suite_group(criterion, "plaintext", huge);
        for &rows in counts.server {
            bench_plaintext(&mut plaintext_group, tag, params, rows);
        }
        plaintext_group.finish();
    }
}

// LLM-scale cases whose fixtures hold the plaintext and encrypted
// `rows x ell` matrices, so the largest one (16384 x 8192) peaks above a
// gigabyte of RAM. This is the default suite: it measures exactly the
// shapes the trained model will exercise.
fn llm_benches(criterion: &mut Criterion) {
    for &ell in &LLM_RECORD_LENGTHS {
        let params = search(ell, LLM_LAMBDA).unwrap();
        println!(
            "llm suite at ell = {ell}: k = {}, b = {}, n = {}, lambda = {}",
            params.k,
            params.b,
            params.n().unwrap(),
            params.lambda
        );
        let counts = SuiteRows {
            derive: &LLM_ROW_COUNTS,
            client: &LLM_ROW_COUNTS,
            server: &LLM_ROW_COUNTS,
        };
        run_suite(criterion, &format!("ell{ell}"), params, &counts);
    }
}

fn protocol_benches(c: &mut Criterion) {
    if bench_common::is_quick() {
        // Fast feedback suite: the legacy parameter set at a few small row
        // counts, keeping both sides of the mask-block boundary in every
        // phase.
        run_suite(c, "", PARAMS, &QUICK_ROWS);
        return;
    }
    llm_benches(c);
}

fn criterion_config() -> Criterion {
    bench_common::criterion_default()
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = protocol_benches
}
criterion_main!(benches);
