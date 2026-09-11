#![expect(
    clippy::unwrap_used,
    reason = "benchmarks use fixed valid parameters and keep setup beside measurements"
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
//! The parameters, fixture construction, and the rayon-global-pool CPU
//! reference match `benches/gpu.rs` exactly; only the shapes differ,
//! sweeping work from 2^16 up to 2^24 estimated field multiplications.
//! Fixture construction (derive + encrypt + queries) and the one-time
//! matrix upload happen before timing; every measured iteration is one
//! full `answer_batch` on either path. Without a compute adapter the
//! binary prints a notice and benchmarks nothing.
//!
//! # Case IDs and filters
//!
//! Each `(batch, rows)` shape runs two cases under the `answer_dispatch`
//! group: `answer_dispatch/cpu/batchB-rowsR` (CPU reference) and
//! `answer_dispatch/gpu/batchB-rowsR` (device answer path). The criterion
//! `--` filter is a regular expression over full case IDs, so one filter
//! selects the whole paired sweep in a single run:
//!
//! ```text
//! cargo bench -p emvp --features gpu --bench dispatch -- \
//!     --save-baseline dispatch-policy 'answer_dispatch/(cpu|gpu)'
//! ```
//!
//! # Calibrated result
//!
//! On the calibration machine (8 CPU threads, Intel Iris Xe iGPU, Mesa
//! ANV Vulkan) the raw crossover sits between 2^22 and 2^23 estimated
//! field multiplications, with the device ahead decisively (20-32%) from
//! 2^24 up; that value is hard-coded as
//! `emvp::dispatch::MIN_GPU_MULTIPLICATIONS`. Re-run this suite against
//! the saved `dispatch-policy` baseline whenever the answer hardware or
//! pool size changes.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use emvp::{
    DerivedState, EmvpParams, EncryptedMatrix, EncryptedQuery, GpuAnswerer, GpuError,
    ProtocolError, SecretKey, answer_batch, encrypt, query,
};
use prime_field_layer::{FieldElement, PrimeField};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use trapdoor_matrices::ToeplitzFastProduct;

// Same NTT-friendly prime as benches/gpu.rs and benches/protocol.rs.
const MODULUS: u32 = 1_073_479_681;

// Same parameter set as the standard protocol suite: k = 512, n = 1024,
// b = 16, s = 64.
const PARAMS: EmvpParams = EmvpParams {
    k: 512,
    ell: 512,
    b: 16,
    lambda: 128,
};

// Crossover sweep as (batch, rows) pairs. The work metric
// `batch * rows * n` with n = 1024 spans 2^16 (first shape) through
// 2^24 (last shapes) field multiplications, doubling shape by shape
// around the expected CPU/GPU crossover.
const CASES: [(usize, usize); 14] = [
    (1, 64),
    (1, 128),
    (1, 256),
    (1, 512),
    (1, 1024),
    (1, 2048),
    (1, 4096),
    (1, 8192),
    (1, 16384),
    (4, 512),
    (4, 1024),
    (4, 2048),
    (4, 4096),
    (8, 2048),
];

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

fn toeplitz_block(
    params: EmvpParams,
    stream: &mut ChaCha20Rng,
    _index: usize,
) -> Result<ToeplitzFastProduct<MODULUS>, ProtocolError> {
    Ok(ToeplitzFastProduct::sample(params.n()?, stream)?)
}

fn derive_with(
    params: EmvpParams,
    rows: usize,
    domain: u8,
) -> DerivedState<MODULUS, ToeplitzFastProduct<MODULUS>> {
    let mut rng = seeded_rng(domain ^ 0x80, rows);
    SecretKey::<MODULUS>::new(params, [domain; 32])
        .unwrap()
        .derive(rows, &mut rng, |stream, index| {
            toeplitz_block(params, stream, index)
        })
        .unwrap()
}

// One client run producing the encrypted matrix plus `count` queries;
// mirrors `benches/gpu.rs::protocol_fixtures_batch` with fresh records.
fn protocol_fixtures_batch(
    params: EmvpParams,
    rows: usize,
    count: usize,
) -> (EncryptedMatrix<MODULUS>, Vec<EncryptedQuery<MODULUS>>) {
    let mut state = derive_with(params, rows, 0x06);
    let matrix = field_values(rows * params.ell, 0x07);
    let encrypted = encrypt(&mut state, &matrix).unwrap();
    let mut queries = Vec::with_capacity(count);
    for _ in 0..count {
        let record = field_values(params.ell, 0x08);
        let (encrypted_query, _decoding_key) = query(&mut state, &record).unwrap();
        queries.push(encrypted_query);
    }
    (encrypted, queries)
}

fn dispatch_benches(c: &mut Criterion) {
    let answerer = match GpuAnswerer::new_sync() {
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
    let mut group = c.benchmark_group("answer_dispatch");
    // Iterations at the top sizes cost tens of milliseconds each; few
    // samples and short phases keep the 28-case sweep bounded.
    group.sample_size(10);
    for &(batch, rows) in &CASES {
        let (encrypted, queries) = protocol_fixtures_batch(PARAMS, rows, batch);
        // One-time upload; the measured GPU iterations reuse the
        // device-resident matrix.
        let gpu_matrix = answerer.upload_matrix_sync(&PARAMS, &encrypted).unwrap();
        let elements = u64::try_from(batch * rows * PARAMS.n().unwrap()).unwrap();
        group.throughput(Throughput::Elements(elements));
        group.bench_function(
            BenchmarkId::new("gpu", format!("batch{batch}-rows{rows}")),
            |b| {
                b.iter(|| answerer.answer_batch_sync(&gpu_matrix, &queries).unwrap());
            },
        );
        drop(gpu_matrix);
        // CPU reference on the rayon global pool; every swept shape is
        // above both parallel-path thresholds (`work >= 2^15` and
        // `rows >= 2 * threads`), matching the deployment pool size.
        group.bench_function(
            BenchmarkId::new("cpu", format!("batch{batch}-rows{rows}")),
            |b| {
                b.iter(|| answer_batch(&PARAMS, &encrypted, &queries).unwrap());
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
