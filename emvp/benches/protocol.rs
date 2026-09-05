#![expect(
    clippy::unwrap_used,
    reason = "benchmarks use fixed valid parameters and keep setup beside measurements"
)]

use std::{hint::black_box, sync::OnceLock, time::Duration};

use bench_common as common;
use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use emvp::{
    AnswerMatrix, DecodingKey, DerivedState, EmvpParams, EncryptedMatrix, EncryptedQuery,
    ProtocolError, SecretKey, TdmMask, answer_into, decode_into, encrypt, query, search,
};
use prime_field_layer::{FieldElement, PrimeField};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use rayon::{ThreadPool, ThreadPoolBuilder, prelude::*};
use trapdoor_matrices::{
    IrreducibleRingLpn, RaaWeightedProduct, SparseMatrix, ToeplitzFastProduct,
};

// NTT-friendly prime: 1_073_479_681 - 1 is divisible by 2^18.
const MODULUS: u32 = 1_073_479_681;

// Realistic parameter set satisfying every `EmvpParams::validate` constraint:
// d = ceil(512 / 15) = 35 so 16^(d-1) * min(16,d) = 2^140 >= 2^128, and
// (n / b + 1) * k = 65 * 512 = 33280 > n + lambda = 1152.
const PARAMS: EmvpParams = EmvpParams {
    k: 512,
    ell: 512,
    b: 16,
    lambda: 128,
};

// Online client cases cross the n-row mask-block boundary at rows = n = 1024.
const CLIENT_ROW_COUNTS: [usize; 5] = [32, 128, 1024, 1025, 2048];
// Server cases extend beyond cache-resident encrypted matrices. The answer
// and decode phases are TDM-agnostic: they consume only the encrypted matrix.
const SERVER_ROW_COUNTS: [usize; 3] = [128, 1024, 8192];
// Cases around the eight-thread answer crossover; calibration-only because
// that boundary was measured once when sizing the rayon thread pool.
const ANSWER_CALIBRATION_ROW_COUNTS: [usize; 3] = [16, 31, 32];
// Covers both sides of the n-row mask-block boundary during derivation.
const DERIVE_ROW_COUNTS: [usize; 4] = [32, 128, 1024, 1025];

// LLM-scale record lengths: the model's hidden dimension, so `ell = 4096`
// matches 7B-class weight matrices and `ell = 8192` 70B-class ones. Each
// suite derives concrete (k, b) from the record length with the same
// parameter search a production deployment would run.
const LLM_RECORD_LENGTHS: [usize; 2] = [4096, 8192];
// LLM-scale matrix heights; 4096 x 4096 is one attention projection and
// 16384 x 8192 approaches a large FFN layer.
const LLM_ROW_COUNTS: [usize; 3] = [4096, 8192, 16384];
const LLM_LAMBDA: u32 = 128;

// Row counts per phase for one parameter set.
struct SuiteRows {
    derive: &'static [usize],
    client: &'static [usize],
    server: &'static [usize],
}

const STANDARD_ROWS: SuiteRows = SuiteRows {
    derive: &DERIVE_ROW_COUNTS,
    client: &CLIENT_ROW_COUNTS,
    server: &SERVER_ROW_COUNTS,
};

// Sparse column weight of the Ring-LPN benchmark blocks.
const TARGET_COLUMN_WEIGHT: usize = 16;

fn benchmark_pool() -> &'static ThreadPool {
    static POOL: OnceLock<ThreadPool> = OnceLock::new();
    POOL.get_or_init(|| ThreadPoolBuilder::new().num_threads(8).build().unwrap())
}

fn seeded_rng(domain: u8, size: usize) -> ChaCha20Rng {
    let mut seed = [domain; 32];
    for (slot, byte) in seed.iter_mut().zip(size.to_le_bytes()) {
        *slot ^= byte;
    }
    ChaCha20Rng::from_seed(seed)
}

fn field_values(length: usize, domain: u8) -> Vec<FieldElement<MODULUS>> {
    let field = PrimeField::<MODULUS>::new();
    let mut values = vec![field.element_u32(0); length];
    field.fill_uniform(&mut seeded_rng(domain, length), &mut values);
    values
}

fn elements(count: usize) -> Throughput {
    Throughput::Elements(u64::try_from(count).unwrap())
}

// Standard-suite benchmark IDs keep their historical shape so saved
// baselines stay comparable; other suites prefix the parameter.
fn bench_parameter(tag: &str, rows: usize) -> String {
    if tag.is_empty() {
        rows.to_string()
    } else {
        format!("{tag}-rows{rows}")
    }
}

fn suite_group<'a>(
    criterion: &'a mut Criterion,
    name: &str,
    huge: bool,
) -> BenchmarkGroup<'a, WallTime> {
    let mut group = criterion.benchmark_group(name);
    if huge {
        // LLM-scale iterations cost seconds; fewer samples keep the suite
        // run time bounded.
        group.sample_size(10);
    }
    group
}

// One `n x n` Toeplitz mask block, matching the codeword length n = 2k.
fn toeplitz_block(
    params: EmvpParams,
    stream: &mut ChaCha20Rng,
    _index: usize,
) -> Result<ToeplitzFastProduct<MODULUS>, ProtocolError> {
    Ok(ToeplitzFastProduct::sample(params.n()?, stream)?)
}

// One `n x n` RAA mask block with three nonzero weights per factor.
fn raa_block(
    params: EmvpParams,
    stream: &mut ChaCha20Rng,
    _index: usize,
) -> Result<RaaWeightedProduct<MODULUS>, ProtocolError> {
    Ok(RaaWeightedProduct::sample_nonzero(params.n()?, 3, stream)?)
}

// One square `n x n` Ring-LPN mask block. The sparse secret keeps a fixed
// column weight, and the monic binomial modulus is benchmark input, not an
// irreducibility claim; the unchecked constructor isolates block sampling
// from an impractical irreducibility search at degree n.
fn ring_block(
    params: EmvpParams,
    stream: &mut ChaCha20Rng,
    _index: usize,
) -> Result<IrreducibleRingLpn<MODULUS>, ProtocolError> {
    let n = params.n()?;
    let field = PrimeField::<MODULUS>::new();
    let multiplier: Vec<u32> = (0..n)
        .map(|_| field.sample_uniform(stream).value())
        .collect();
    let rows = 2 * n;
    let target_weight = TARGET_COLUMN_WEIGHT.min(n);
    let mut offsets = Vec::with_capacity(n + 1);
    let mut row_indices = Vec::with_capacity(n * target_weight);
    let mut values = Vec::with_capacity(n * target_weight);
    offsets.push(0);
    for column in 0..n {
        for entry in 0..target_weight {
            row_indices.push((17 * column + entry) % rows);
            values.push(field.sample_uniform_nonzero(stream));
        }
        offsets.push(row_indices.len());
    }
    let sparse = SparseMatrix::new(rows, n, offsets, row_indices, values)?;
    let mut modulus = vec![0; n + 1];
    modulus[0] = MODULUS - 3;
    modulus[n] = 1;
    Ok(IrreducibleRingLpn::new_unchecked_irreducible(
        n,
        &modulus,
        &multiplier,
        sparse,
    )?)
}

type BlockBuilder<M> = fn(EmvpParams, &mut ChaCha20Rng, usize) -> Result<M, ProtocolError>;

// The expanded long-term secrets for `rows` matrix rows.
fn derive_with<M: TdmMask<MODULUS>>(
    params: EmvpParams,
    rows: usize,
    domain: u8,
    build_block: BlockBuilder<M>,
) -> DerivedState<MODULUS, M> {
    let mut rng = seeded_rng(domain ^ 0x80, rows);
    SecretKey::<MODULUS>::new(params, [domain; 32])
        .unwrap()
        .derive(rows, &mut rng, |stream, index| {
            build_block(params, stream, index)
        })
        .unwrap()
}

fn bench_derive_for<M: TdmMask<MODULUS>>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    tag: &str,
    label: &str,
    params: EmvpParams,
    rows: usize,
    build_block: BlockBuilder<M>,
) {
    group.bench_function(BenchmarkId::new(label, bench_parameter(tag, rows)), |b| {
        b.iter(|| {
            let mut rng = seeded_rng(0xf2, rows);
            black_box(
                SecretKey::<MODULUS>::new(params, [0x72; 32])
                    .unwrap()
                    .derive(rows, &mut rng, |stream, index| {
                        build_block(params, stream, index)
                    })
                    .unwrap(),
            )
        });
    });
}

fn bench_encrypt_for<M: TdmMask<MODULUS>>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    tag: &str,
    label: &str,
    params: EmvpParams,
    rows: usize,
    build_block: BlockBuilder<M>,
) {
    let matrix = field_values(rows * params.ell, 0x02);
    group.throughput(elements(rows * params.ell));
    group.bench_function(BenchmarkId::new(label, bench_parameter(tag, rows)), |b| {
        b.iter_batched(
            || derive_with(params, rows, 0x01, build_block),
            |mut state| black_box(encrypt(black_box(&mut state), black_box(&matrix)).unwrap()),
            BatchSize::SmallInput,
        );
    });
}

fn bench_query_for<M: TdmMask<MODULUS>>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    tag: &str,
    label: &str,
    params: EmvpParams,
    rows: usize,
    build_block: BlockBuilder<M>,
) {
    let mut state = derive_with(params, rows, 0x03, build_block);
    let record = field_values(params.ell, 0x04);
    // The stream keeps advancing across iterations, mirroring repeated
    // queries with fresh randomness.
    group.bench_function(BenchmarkId::new(label, bench_parameter(tag, rows)), |b| {
        b.iter(|| {
            let (encrypted_query, decoding_key) =
                query(black_box(&mut state), black_box(&record)).unwrap();
            black_box((encrypted_query, decoding_key))
        });
    });
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
                benchmark_pool().install(|| {
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
                });
                black_box(&output);
            });
        },
    );
}

// One client run producing the server and client fixtures for the answer
// and decode phases.
fn protocol_fixtures(
    params: EmvpParams,
    rows: usize,
) -> (
    EncryptedMatrix<MODULUS>,
    EncryptedQuery<MODULUS>,
    DecodingKey<MODULUS>,
) {
    let mut state = derive_with(params, rows, 0x06, toeplitz_block);
    let matrix = field_values(rows * params.ell, 0x07);
    let record = field_values(params.ell, 0x08);
    let encrypted = encrypt(&mut state, &matrix).unwrap();
    let (encrypted_query, decoding_key) = query(&mut state, &record).unwrap();
    (encrypted, encrypted_query, decoding_key)
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
                benchmark_pool()
                    .install(|| {
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
    let answer_matrix = AnswerMatrix::from_parts(
        encrypted.instance_id(),
        encrypted_query.query_id(),
        answer_values,
        rows,
        blocks,
    );
    let mut output = vec![zero; rows];
    group.throughput(elements(rows * blocks));
    group.bench_function(
        BenchmarkId::from_parameter(bench_parameter(tag, rows)),
        |b| {
            b.iter(|| {
                decode_into(
                    black_box(&answer_matrix),
                    black_box(&decoding_key),
                    black_box(&mut output),
                )
                .unwrap();
                black_box(&output);
            });
        },
    );
}

// All phases for one parameter set. The empty tag marks the standard
// parameter set and unlocks its calibration cases.
fn run_suite(criterion: &mut Criterion, tag: &str, params: EmvpParams, counts: &SuiteRows) {
    let huge = !tag.is_empty();

    {
        let mut derive_group = suite_group(criterion, "derive", huge);
        for &rows in counts.derive {
            bench_derive_for(
                &mut derive_group,
                tag,
                "toeplitz",
                params,
                rows,
                toeplitz_block,
            );
            bench_derive_for(&mut derive_group, tag, "raa", params, rows, raa_block);
            bench_derive_for(&mut derive_group, tag, "ring", params, rows, ring_block);
        }
        derive_group.finish();
    }

    {
        let mut encrypt_group = suite_group(criterion, "encrypt", huge);
        for &rows in counts.client {
            bench_encrypt_for(
                &mut encrypt_group,
                tag,
                "toeplitz",
                params,
                rows,
                toeplitz_block,
            );
            bench_encrypt_for(&mut encrypt_group, tag, "raa", params, rows, raa_block);
            bench_encrypt_for(&mut encrypt_group, tag, "ring", params, rows, ring_block);
        }
        encrypt_group.finish();
    }

    {
        let mut query_group = suite_group(criterion, "query", huge);
        for &rows in counts.client {
            bench_query_for(
                &mut query_group,
                tag,
                "toeplitz",
                params,
                rows,
                toeplitz_block,
            );
            bench_query_for(&mut query_group, tag, "raa", params, rows, raa_block);
            bench_query_for(&mut query_group, tag, "ring", params, rows, ring_block);
        }
        query_group.finish();
    }

    {
        let mut answer_group = suite_group(criterion, "answer", huge);
        for &rows in counts.server {
            bench_answer(&mut answer_group, tag, params, rows);
        }
        if !huge && common::calibration_enabled() {
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
// gigabyte of RAM. Opt-in via the `bench-huge` feature.
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
    run_suite(c, "", PARAMS, &STANDARD_ROWS);
    if common::skip_unless_huge("llm") {
        return;
    }
    llm_benches(c);
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(4))
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = protocol_benches
}
criterion_main!(benches);
