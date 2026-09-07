#![expect(
    clippy::unwrap_used,
    reason = "benchmarks use fixed valid parameters and keep setup beside measurements"
)]
#![expect(
    clippy::unnecessary_literal_unwrap,
    reason = "the adapter probe wraps an unexpected error in `Err` so the single `unwrap` path fails the bench loudly; static analysis flags the literal even though the error is dynamic"
)]

//! GPU-versus-CPU benchmarks for the server answer phase.
//!
//! The cases run on the rayon global pool a deployment would use, over the
//! same field and parameter set as `benches/protocol.rs` so the CPU numbers
//! stay comparable with the saved protocol baselines. Fixture construction
//! (derive + encrypt + queries) and the one-time matrix upload happen before
//! timing; every measured iteration is one full `answer_batch`. Without a
//! compute adapter the binary prints a notice and benchmarks nothing.
//!
//! # Case IDs and filters
//!
//! Each `(batch, rows)` size runs three cases under the `gpu_answer` group:
//! `gpu_answer/gpu/batchB-rowsR` (device answer path),
//! `gpu_answer/cpu/batchB-rowsR` (CPU reference on the same shapes), and
//! `gpu_answer/phase/batchB-rowsR` (the same device path with the
//! host-side [`PhaseTimings`] breakdown printed after the criterion table).
//!
//! The criterion `--` filter is a regular expression over full case IDs, so
//! one filter selects the whole paired suite in a single run:
//!
//! ```text
//! cargo bench -p emvp --features gpu -- 'gpu_answer/(gpu|cpu|phase)'
//! ```
//!
//! Plain `-- gpu` matches only the `gpu_answer/gpu/*` cases because the
//! existing `cpu` and `phase` IDs intentionally do not contain the string
//! "gpu" — the `gpu` and `cpu` IDs are kept verbatim so criterion baselines
//! saved by earlier validation runs still resolve.

use std::time::Duration;

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use emvp::{
    DerivedState, EmvpParams, EncryptedMatrix, EncryptedQuery, GpuAnswerer, GpuError, PhaseTimings,
    ProtocolError, SecretKey, answer_batch, encrypt, query,
};
use prime_field_layer::{FieldElement, PrimeField};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use trapdoor_matrices::ToeplitzFastProduct;

// Same NTT-friendly prime as benches/protocol.rs.
const MODULUS: u32 = 1_073_479_681;

// Same parameter set as the standard protocol suite: k = 512, n = 1024,
// b = 16, s = 64.
const PARAMS: EmvpParams = EmvpParams {
    k: 512,
    ell: 512,
    b: 16,
    lambda: 128,
};

// Batched answer cases as (batch, rows) pairs. The 262144-row encrypted
// matrix is a 1 GiB device upload; outputs and staging stay at or below
// 128 MiB each, so the whole footprint fits the 6 GiB reference card with
// room to spare.
const CASES: [(usize, usize); 3] = [(1, 65_536), (8, 65_536), (1, 262_144)];

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
// mirrors `benches/protocol.rs::protocol_fixtures_batch` with fresh records.
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

/// Accessor for one [`PhaseTimings`] phase field.
type PhaseAccessor = fn(&PhaseTimings) -> Duration;

/// Prints the host-side phase breakdown a `phase` case collected across all
/// of its measured iterations.
fn print_phase_breakdown(batch: usize, rows: usize, samples: &[PhaseTimings]) {
    if samples.is_empty() {
        return;
    }
    let phases: [(&str, PhaseAccessor); 6] = [
        ("prepare_buffers", |timings| timings.prepare_buffers),
        ("encode_upload_queries", |timings| {
            timings.encode_upload_queries
        }),
        ("dispatch_submit", |timings| timings.dispatch_submit),
        ("wait_readback", |timings| timings.wait_readback),
        ("reconstruct", |timings| timings.reconstruct),
        ("total", PhaseTimings::total),
    ];
    println!(
        "phase breakdown gpu_answer/phase/batch{batch}-rows{rows} over {} calls (median | mean):",
        samples.len()
    );
    for (name, accessor) in phases {
        let mut values: Vec<Duration> = samples.iter().map(accessor).collect();
        values.sort();
        let median = values[values.len() / 2];
        let mean =
            values.iter().sum::<Duration>() / u32::try_from(values.len()).unwrap_or(u32::MAX);
        println!("  {name:<24}{median:.3?} | {mean:.3?}");
    }
}

fn gpu_benches(c: &mut Criterion) {
    let answerer = match GpuAnswerer::new_sync() {
        Ok(answerer) => answerer,
        Err(GpuError::NoAdapter { reason }) => {
            println!("skipping gpu benches: no compute adapter available ({reason})");
            return;
        }
        Err(error) => {
            let failure = Err::<GpuAnswerer, GpuError>(error);
            failure.unwrap()
        }
    };
    let mut group = c.benchmark_group("gpu_answer");
    // Iterations at the top size cost seconds; fewer samples keep the run
    // bounded, matching the huge protocol-suite configuration.
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
        // The same device path instrumented with the host-side phase
        // breakdown; the criterion number is the same wall time as the
        // `gpu` case and the block printed afterwards attributes it to the
        // phases of `PhaseTimings`.
        let mut phase_samples: Vec<PhaseTimings> = Vec::new();
        group.bench_function(
            BenchmarkId::new("phase", format!("batch{batch}-rows{rows}")),
            |b| {
                b.iter(|| {
                    let (_answers, timings) = answerer
                        .answer_batch_sync_with_timings(&gpu_matrix, &queries)
                        .unwrap();
                    phase_samples.push(timings);
                });
            },
        );
        print_phase_breakdown(batch, rows, &phase_samples);
        drop(gpu_matrix);
        // CPU reference on the rayon global pool, matching the huge
        // answer_batch cases in benches/protocol.rs.
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
        .measurement_time(Duration::from_secs(4))
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = gpu_benches
}
criterion_main!(benches);
