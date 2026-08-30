#![expect(
    clippy::unwrap_used,
    reason = "benchmarks use fixed valid parameters and keep setup beside measurements"
)]

use std::{hint::black_box, time::Duration};

use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use emvp::{
    DecodingKey, DerivedState, EmvpParams, EncryptedMatrix, EncryptedQuery, ProtocolError,
    SecretKey, answer, decode, encrypt, query,
};
use prime_field_layer::{FieldElement, PrimeField};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use trapdoor_matrices::ToeplitzFastProduct;

// NTT-friendly prime: 998_244_353 - 1 is divisible by 2^23.
const MODULUS: u32 = 998_244_353;

// Realistic parameter set satisfying every `EmvpParams::validate` constraint:
// d = ceil(512 / 15) = 35 so 16^(d-1) * min(16,d) = 2^140 >= 2^128, and
// (n / b + 1) * k = 65 * 512 = 33280 > n + lambda = 1152.
const PARAMS: EmvpParams = EmvpParams {
    k: 512,
    ell: 512,
    b: 16,
    lambda: 128,
};

// Matrix row counts benchmarked for every phase.
const ROW_COUNTS: [usize; 2] = [32, 128];
// Covers both sides of the n-row mask-block boundary during derivation.
const DERIVE_ROW_COUNTS: [usize; 4] = [32, 128, 1024, 1025];

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

// One `n x n` Toeplitz mask block, matching the codeword length n = 2k.
fn toeplitz_block(
    stream: &mut ChaCha20Rng,
    _index: usize,
) -> Result<ToeplitzFastProduct<MODULUS>, ProtocolError> {
    Ok(ToeplitzFastProduct::sample(PARAMS.n().unwrap(), stream)?)
}

// The expanded long-term secrets for `rows` matrix rows.
fn derive_state(rows: usize, domain: u8) -> DerivedState<MODULUS, ToeplitzFastProduct<MODULUS>> {
    SecretKey::<MODULUS>::new(PARAMS, [domain; 32])
        .unwrap()
        .derive(rows, toeplitz_block)
        .unwrap()
}

fn bench_derive(group: &mut BenchmarkGroup<'_, WallTime>, rows: usize) {
    group.bench_function(BenchmarkId::new("toeplitz", rows), |b| {
        b.iter(|| {
            black_box(
                SecretKey::<MODULUS>::new(PARAMS, [0x72; 32])
                    .unwrap()
                    .derive(rows, toeplitz_block)
                    .unwrap(),
            )
        });
    });
}

fn bench_encrypt(group: &mut BenchmarkGroup<'_, WallTime>, rows: usize) {
    let matrix = field_values(rows * PARAMS.ell, 0x02);
    group.throughput(elements(rows * PARAMS.ell));
    group.bench_function(BenchmarkId::new("toeplitz", rows), |b| {
        b.iter_batched(
            || derive_state(rows, 0x01),
            |mut state| black_box(encrypt(black_box(&mut state), black_box(&matrix)).unwrap()),
            BatchSize::SmallInput,
        );
    });
}

fn bench_query(group: &mut BenchmarkGroup<'_, WallTime>, rows: usize) {
    let mut state = derive_state(rows, 0x03);
    let record = field_values(PARAMS.ell, 0x04);
    // The stream keeps advancing across iterations, mirroring repeated
    // queries with fresh randomness.
    let mut rng = seeded_rng(0x05, rows);
    group.throughput(elements(PARAMS.ell));
    group.bench_function(BenchmarkId::new("toeplitz", rows), |b| {
        b.iter(|| {
            let (encrypted_query, decoding_key) = query(
                black_box(&mut state),
                black_box(&record),
                black_box(&mut rng),
            )
            .unwrap();
            black_box((encrypted_query, decoding_key))
        });
    });
}

// One client run producing the server and client fixtures for the answer
// and decode phases.
fn protocol_fixtures(
    rows: usize,
) -> (
    EncryptedMatrix<MODULUS>,
    EncryptedQuery<MODULUS>,
    DecodingKey<MODULUS>,
) {
    let mut state = derive_state(rows, 0x06);
    let matrix = field_values(rows * PARAMS.ell, 0x07);
    let record = field_values(PARAMS.ell, 0x08);
    let mut rng = seeded_rng(0x09, rows);
    let encrypted = encrypt(&mut state, &matrix).unwrap();
    let (encrypted_query, decoding_key) = query(&mut state, &record, &mut rng).unwrap();
    (encrypted, encrypted_query, decoding_key)
}

fn bench_answer(group: &mut BenchmarkGroup<'_, WallTime>, rows: usize) {
    let (encrypted, encrypted_query, _decoding_key) = protocol_fixtures(rows);
    group.throughput(elements(rows * PARAMS.n().unwrap()));
    group.bench_function(BenchmarkId::new("toeplitz", rows), |b| {
        b.iter(|| {
            black_box(
                answer(
                    black_box(&PARAMS),
                    black_box(&encrypted),
                    black_box(&encrypted_query),
                )
                .unwrap(),
            )
        });
    });
}

fn bench_decode(group: &mut BenchmarkGroup<'_, WallTime>, rows: usize) {
    let (encrypted, encrypted_query, decoding_key) = protocol_fixtures(rows);
    let answer_matrix = answer(&PARAMS, &encrypted, &encrypted_query).unwrap();
    group.throughput(elements(rows * PARAMS.blocks().unwrap()));
    group.bench_function(BenchmarkId::new("toeplitz", rows), |b| {
        b.iter(|| black_box(decode(black_box(&answer_matrix), black_box(&decoding_key)).unwrap()));
    });
}

fn protocol_benches(c: &mut Criterion) {
    {
        let mut derive_group = c.benchmark_group("derive");
        for &rows in &DERIVE_ROW_COUNTS {
            bench_derive(&mut derive_group, rows);
        }
        derive_group.finish();
    }

    {
        let mut encrypt_group = c.benchmark_group("encrypt");
        for &rows in &ROW_COUNTS {
            bench_encrypt(&mut encrypt_group, rows);
        }
        encrypt_group.finish();
    }

    {
        let mut query_group = c.benchmark_group("query");
        for &rows in &ROW_COUNTS {
            bench_query(&mut query_group, rows);
        }
        query_group.finish();
    }

    {
        let mut answer_group = c.benchmark_group("answer");
        for &rows in &ROW_COUNTS {
            bench_answer(&mut answer_group, rows);
        }
        answer_group.finish();
    }

    {
        let mut decode_group = c.benchmark_group("decode");
        for &rows in &ROW_COUNTS {
            bench_decode(&mut decode_group, rows);
        }
        decode_group.finish();
    }
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2))
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = protocol_benches
}
criterion_main!(benches);
