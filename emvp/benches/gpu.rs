#![expect(
    clippy::unwrap_used,
    reason = "benchmarks use fixed valid parameters and keep setup beside measurement"
)]
#![expect(
    clippy::unnecessary_literal_unwrap,
    reason = "the adapter probe wraps an unexpected error in `Err` so the single `unwrap` path fails the bench loudly; static analysis flags the literal even though the error is dynamic"
)]

//! GPU-versus-CPU benchmarks for the server answer phase at LLM scale.
//!
//! The parameter suites, fixtures, case IDs, and the pinned eight-thread CPU
//! pool mirror `benches/protocol.rs` exactly, so the CPU numbers stay
//! comparable with the saved protocol answer baselines and the GPU numbers
//! slot next to them in the same tables: one single-query `answer_batch` per
//! measured iteration over the same `ell in {4096, 8192}` suites and
//! `rows in {4096, 8192, 16384}` matrix heights. Fixture construction
//! (derive + encrypt) and the one-time matrix upload happen before timing;
//! the largest shape uploads a 512 MiB encrypted matrix, which fits the
//! 6 GiB reference card together with its staging buffer. Without a compute
//! adapter the binary prints a notice and benchmarks nothing.
//!
//! # Case IDs and filters
//!
//! Each `(ell, rows)` size runs three cases under the `gpu_answer` group:
//! `gpu_answer/gpu/ellE-rowsR` (device answer path),
//! `gpu_answer/phase/ellE-rowsR` (the same device path instrumented; prints
//! a host-side [`PhaseTimings`] median|mean breakdown to stdout after the
//! criterion table), and `gpu_answer/cpu/ellE-rowsR` (the eight-thread pool
//! reference matching `benches/protocol.rs`).
//!
//! ```text
//! cargo bench -p emvp --features gpu -- 'gpu_answer/(gpu|cpu|phase)'
//! ```

use std::{slice, sync::OnceLock, time::Duration};

use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use emvp::{
    DerivedState, EmvpParams, EncryptedMatrix, EncryptedQuery, GpuAnswerer, GpuError, PhaseTimings,
    ProtocolError, SecretKey, answer_batch, encrypt, query, search,
};
use prime_field_layer::{FieldElement, PrimeField};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use rayon::{ThreadPool, ThreadPoolBuilder};
use trapdoor_matrices::ToeplitzFastProduct;

// Same NTT-friendly prime as benches/protocol.rs.
const MODULUS: u32 = 1_073_479_681;

// LLM-scale record lengths and matrix heights, matching the default
// protocol suite: each suite derives concrete (k, b) from the record length
// with the same parameter search a production deployment would run.
const LLM_RECORD_LENGTHS: [usize; 2] = [4096, 8192];
const LLM_ROW_COUNTS: [usize; 3] = [4096, 8192, 16384];
const LLM_LAMBDA: u32 = 128;

// The fixed eight-thread pool keeps the CPU reference comparable with the
// saved protocol answer baselines; installed before every measured
// iteration.
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

// One client run producing the encrypted matrix plus one query; mirrors
// `benches/protocol.rs::protocol_fixtures`.
fn protocol_fixtures(
    params: EmvpParams,
    rows: usize,
) -> (EncryptedMatrix<MODULUS>, EncryptedQuery<MODULUS>) {
    let mut state = derive_with(params, rows, 0x06);
    let matrix = field_values(rows * params.ell, 0x07);
    let record = field_values(params.ell, 0x08);
    let encrypted = encrypt(&mut state, &matrix).unwrap();
    let (encrypted_query, _decoding_key) = query(&mut state, &record).unwrap();
    (encrypted, encrypted_query)
}

/// Accessor for one [`PhaseTimings`] phase field.
type PhaseAccessor = fn(&PhaseTimings) -> Duration;

/// Prints the host-side phase breakdown a `phase` case collected across all
/// of its measured iterations.
fn print_phase_breakdown(tag: &str, samples: &[PhaseTimings]) {
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
        "phase breakdown gpu_answer/phase/{tag} over {} calls (median | mean):",
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
    // Iterations at the top size cost tens of milliseconds; fewer samples
    // keep the run bounded, matching the huge protocol-suite configuration.
    group.sample_size(10);
    for &ell in &LLM_RECORD_LENGTHS {
        let params = search(ell, LLM_LAMBDA).unwrap();
        println!(
            "llm gpu suite at ell = {ell}: k = {}, b = {}, n = {}, lambda = {}",
            params.k,
            params.b,
            params.n().unwrap(),
            params.lambda
        );
        for &rows in &LLM_ROW_COUNTS {
            let tag = format!("ell{ell}-rows{rows}");
            let (encrypted, query) = protocol_fixtures(params, rows);
            // One-time upload; the measured GPU iterations reuse the
            // device-resident matrix.
            let gpu_matrix = answerer.upload_matrix_sync(&params, &encrypted).unwrap();
            let elements = u64::try_from(rows * params.n().unwrap()).unwrap();
            group.throughput(Throughput::Elements(elements));
            group.bench_function(BenchmarkId::new("gpu", tag.clone()), |b| {
                b.iter(|| {
                    answerer
                        .answer_batch_sync(&gpu_matrix, slice::from_ref(&query))
                        .unwrap()
                });
            });
            // The same device path instrumented with the host-side phase
            // breakdown; the criterion number is the same wall time as the
            // `gpu` case and the block printed afterwards attributes it to
            // the phases of `PhaseTimings`.
            let mut phase_samples: Vec<PhaseTimings> = Vec::new();
            group.bench_function(BenchmarkId::new("phase", tag.clone()), |b| {
                b.iter(|| {
                    let (_answers, timings) = answerer
                        .answer_batch_sync_with_timings(&gpu_matrix, slice::from_ref(&query))
                        .unwrap();
                    phase_samples.push(timings);
                });
            });
            print_phase_breakdown(&tag, &phase_samples);
            drop(gpu_matrix);
            // CPU reference on the pinned eight-thread pool, matching the
            // answer cases in benches/protocol.rs.
            group.bench_function(BenchmarkId::new("cpu", tag), |b| {
                b.iter(|| {
                    benchmark_pool().install(|| {
                        answer_batch(&params, &encrypted, slice::from_ref(&query)).unwrap()
                    })
                });
            });
        }
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
