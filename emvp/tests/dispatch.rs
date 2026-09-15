#![cfg_attr(
    feature = "gpu",
    expect(
        clippy::unwrap_used,
        reason = "fixed test fixtures establish that protocol steps and dispatch operations must succeed"
    )
)]

//! Dispatch-policy and answer-dispatcher tests. The policy tests always
//! run; the dispatcher tests need the `gpu` feature. The adapter-backed
//! dispatcher cases are marked `#[ignore = "requires a compute adapter"]`,
//! so the default suite skips them and the CPU-only demotion paths are
//! exercised on every machine; when run explicitly with `--ignored` on a
//! GPU-less machine, the adapter acquisition fails the test clearly instead
//! of succeeding silently.

use emvp::{AnswerBackend, MIN_PARALLEL_MULTIPLICATIONS, select_answer_backend};

#[cfg(feature = "gpu")]
use emvp::{
    AnswerDispatchError, AnswerDispatcher, EmvpParams, EncryptedMatrix, EncryptedQuery,
    GpuAnswerer, MIN_GPU_MULTIPLICATIONS, MaskContextId, ProtocolError, SecretKey, answer_batch,
    encrypt, query,
};
#[cfg(feature = "gpu")]
use prime_field_layer::{FieldElement, PrimeField};
#[cfg(feature = "gpu")]
use rand_chacha::ChaCha20Rng;
#[cfg(feature = "gpu")]
use rand_core::SeedableRng;
#[cfg(feature = "gpu")]
use rayon::ThreadPoolBuilder;
#[cfg(feature = "gpu")]
use trapdoor_matrices::ToeplitzFastProduct;

// The deployment field: p = 998244353 is prime and below 2^30.
#[cfg(feature = "gpu")]
const MODULUS: u32 = 998_244_353;

// Tiny deliberately-insecure research parameters; structure is still
// enforced (ell <= k, b >= 2, b divides n = 2k).
#[cfg(feature = "gpu")]
const fn test_params(ell: usize, k: usize, b: usize) -> EmvpParams {
    EmvpParams {
        k,
        ell,
        b,
        lambda: 7,
    }
}

// Stable context identifier of this suite's Toeplitz fixtures, folded with
// the concrete configuration so distinct shapes never share a context.
#[cfg(feature = "gpu")]
const CONTEXT_TOEPLITZ: MaskContextId = MaskContextId::from_u64(0x544f_4550);

#[cfg(feature = "gpu")]
const fn fixture_context(k: usize, ell: usize) -> MaskContextId {
    MaskContextId::new(
        CONTEXT_TOEPLITZ.get() ^ ((k as u64 as u128) << 64) ^ ((ell as u64 as u128) << 32),
    )
}

#[cfg(feature = "gpu")]
fn random_vector<const M: u32>(length: usize, rng: &mut ChaCha20Rng) -> Vec<FieldElement<M>> {
    let field = PrimeField::<M>::new();
    (0..length).map(|_| field.sample_uniform(rng)).collect()
}

#[cfg(feature = "gpu")]
fn toeplitz_block<const M: u32>(
    n: usize,
    stream: &mut ChaCha20Rng,
) -> Result<ToeplitzFastProduct<M>, ProtocolError> {
    Ok(ToeplitzFastProduct::sample(n, stream)?)
}

// One encrypted matrix plus `batch` queries, deterministically seeded.
#[cfg(feature = "gpu")]
struct Fixture<const M: u32> {
    params: EmvpParams,
    encrypted: emvp::EncryptedMatrix<M>,
    queries: Vec<EncryptedQuery<M>>,
}

#[cfg(feature = "gpu")]
fn protocol_fixture<const M: u32>(
    k: usize,
    ell: usize,
    b: usize,
    rows: usize,
    batch: usize,
    key: u8,
) -> Fixture<M> {
    let params = test_params(ell, k, b);
    let n = 2 * k;
    let mut derive_rng = ChaCha20Rng::seed_from_u64(0xd100 + u64::from(key));
    let mut state = SecretKey::<M>::new_insecure(params, [key; 32])
        .unwrap()
        .derive(
            fixture_context(k, ell),
            rows,
            &mut derive_rng,
            |stream, _index| toeplitz_block::<M>(n, stream),
        )
        .unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(0x8f00 + u64::from(key));
    let matrix = random_vector::<M>(rows * ell, &mut rng);
    let encrypted = encrypt(&mut state, &matrix).unwrap();
    let mut queries = Vec::with_capacity(batch);
    for _ in 0..batch {
        let record = random_vector::<M>(ell, &mut rng);
        let (encrypted_query, _decoding_key) = query(&mut state, &record).unwrap();
        queries.push(encrypted_query);
    }
    Fixture {
        params,
        encrypted,
        queries,
    }
}

#[cfg(feature = "gpu")]
fn assert_answers_match_cpu<const M: u32>(fixture: &Fixture<M>, answers: &[emvp::AnswerMatrix<M>]) {
    let cpu = answer_batch(&fixture.params, &fixture.encrypted, &fixture.queries).unwrap();
    assert_eq!(answers.len(), cpu.len());
    for (answer, cpu_answer) in answers.iter().zip(&cpu) {
        assert_eq!(answer, cpu_answer);
        assert_eq!(answer.instance_id(), cpu_answer.instance_id());
        assert_eq!(answer.query_id(), cpu_answer.query_id());
    }
}

#[test]
fn policy_matches_the_executing_cpu_kernels() {
    // The policy's rayon tier must agree with the guard the answer kernels
    // actually apply, including the grid-flatness escape hatch.
    let threads = rayon::current_num_threads();
    let small = (1, 4, 16); // work 64: below the parallel threshold
    let (queries, rows, n) = small;
    let expected = if threads > 1
        && queries * rows >= 2 * threads
        && queries * rows * n >= MIN_PARALLEL_MULTIPLICATIONS
    {
        AnswerBackend::Rayon
    } else {
        AnswerBackend::SingleCore
    };
    assert_eq!(select_answer_backend(queries, rows, n, threads), expected);
}

#[cfg(feature = "gpu")]
mod dispatcher {
    use super::*;

    #[test]
    fn cpu_only_dispatcher_demotes_the_gpu_tier() {
        // 2^24 multiplications, well above the GPU threshold: GPU tier by
        // policy, rayon tier without a device.
        let fixture = protocol_fixture::<MODULUS>(32, 8, 2, 65_536, 4, 0x21);
        let dispatcher = AnswerDispatcher::cpu(fixture.params, &fixture.encrypted).unwrap();
        assert!(!dispatcher.is_device_backed());
        assert_eq!(
            dispatcher.backend(fixture.queries.len()),
            AnswerBackend::Rayon
        );
        let answers = dispatcher.answer_batch(&fixture.queries).unwrap();
        assert_answers_match_cpu(&fixture, &answers);
    }

    // One synthetic encrypted matrix at `(rows, columns)`; only the shape
    // matters for backend selection, so zero words suffice.
    fn shape_fixture(rows: usize, n: usize) -> (EmvpParams, EncryptedMatrix<MODULUS>) {
        let zero = PrimeField::<MODULUS>::new().element_u32(0);
        let params = test_params(8, n / 2, 2);
        let matrix = EncryptedMatrix::from_parts(1, rows, n, vec![zero; rows * n]).unwrap();
        (params, matrix)
    }

    #[test]
    fn flat_grids_demote_the_gpu_tier_to_single_core() {
        // 2^20 multiplications with a one-row grid clear the GPU threshold
        // but cannot parallelize: the demoted CPU tier must be SingleCore,
        // not the rayon tier the old fallback reported.
        let (params, matrix) = shape_fixture(1, MIN_GPU_MULTIPLICATIONS);
        let dispatcher = AnswerDispatcher::cpu(params, &matrix).unwrap();
        assert_eq!(dispatcher.backend(1), AnswerBackend::SingleCore);
    }

    #[test]
    fn gpu_tiers_demote_per_the_pool_guard() {
        // More than 2^20 multiplications per query on a four-thread pool:
        // seven rows offer fewer than two per thread and demote to SingleCore,
        // while eight rows offer exactly two per thread and demote to Rayon. A
        // one-threaded pool demotes even a 2^20-multiplication batched
        // shape all the way to SingleCore.
        let pool = ThreadPoolBuilder::new().num_threads(4).build().unwrap();
        let serial = ThreadPoolBuilder::new().num_threads(1).build().unwrap();
        let n = MIN_GPU_MULTIPLICATIONS / 4;
        let (params_flat, matrix_flat) = shape_fixture(7, n);
        let (params_wide, matrix_wide) = shape_fixture(8, n);
        let (params_batched, matrix_batched) = shape_fixture(64, 256);
        pool.install(|| {
            let flat = AnswerDispatcher::cpu(params_flat, &matrix_flat).unwrap();
            assert_eq!(flat.backend(1), AnswerBackend::SingleCore);
            let wide = AnswerDispatcher::cpu(params_wide, &matrix_wide).unwrap();
            assert_eq!(wide.backend(1), AnswerBackend::Rayon);
        });
        serial.install(|| {
            let batched = AnswerDispatcher::cpu(params_batched, &matrix_batched).unwrap();
            assert_eq!(batched.backend(64), AnswerBackend::SingleCore);
        });
    }

    #[test]
    fn dispatcher_rejects_shape_mismatches_before_answering() {
        let fixture = protocol_fixture::<MODULUS>(8, 8, 2, 5, 2, 0x22);
        let mismatched = test_params(8, 16, 2);
        assert!(matches!(
            AnswerDispatcher::cpu(mismatched, &fixture.encrypted),
            Err(AnswerDispatchError::Protocol(
                ProtocolError::LengthMismatch {
                    name: "encrypted matrix columns",
                    ..
                }
            ))
        ));
        let armed = AnswerDispatcher::cpu(fixture.params, &fixture.encrypted).unwrap();
        assert!(matches!(
            armed.answer_batch(&[]),
            Err(AnswerDispatchError::Protocol(
                ProtocolError::LengthMismatch {
                    name: "queries",
                    ..
                }
            ))
        ));
    }

    #[test]
    #[ignore = "requires a compute adapter"]
    fn armed_dispatcher_selects_the_gpu_tier_at_the_threshold() {
        // A missing adapter must fail this ignored test clearly, which the
        // unwrap's error message does.
        let answerer = GpuAnswerer::new().unwrap();
        // batch * rows * n = 4 * 2^12 * 2^6 = 2^20 = MIN_GPU_MULTIPLICATIONS:
        // the exact threshold must select the device.
        let fixture = protocol_fixture::<MODULUS>(32, 8, 2, 4_096, 4, 0x23);
        let dispatcher =
            AnswerDispatcher::new(fixture.params, &fixture.encrypted, Some(&answerer)).unwrap();
        assert!(dispatcher.is_device_backed());
        assert_eq!(
            dispatcher.backend(fixture.queries.len()),
            AnswerBackend::Gpu
        );
        let answers = dispatcher.answer_batch(&fixture.queries).unwrap();
        assert_answers_match_cpu(&fixture, &answers);

        // Below the threshold the same armed dispatcher falls back to the
        // CPU path, and the answers stay bit-identical.
        let small = protocol_fixture::<MODULUS>(32, 8, 2, 256, 1, 0x24);
        let small_dispatcher =
            AnswerDispatcher::new(small.params, &small.encrypted, Some(&answerer)).unwrap();
        assert_ne!(
            small_dispatcher.backend(small.queries.len()),
            AnswerBackend::Gpu
        );
        let answers = small_dispatcher.answer_batch(&small.queries).unwrap();
        assert_answers_match_cpu(&small, &answers);
    }

    #[test]
    #[ignore = "requires a compute adapter"]
    fn armed_dispatcher_answers_small_batches_across_shapes() {
        let answerer = GpuAnswerer::new().unwrap();
        let shapes = [(8, 8, 2, 5, 3), (16, 16, 4, 17, 2), (12, 7, 6, 9, 5)];
        for (index, &(k, ell, b, rows, batch)) in shapes.iter().enumerate() {
            let fixture = protocol_fixture::<MODULUS>(k, ell, b, rows, batch, 0x25 + index as u8);
            let dispatcher =
                AnswerDispatcher::new(fixture.params, &fixture.encrypted, Some(&answerer)).unwrap();
            assert_eq!(
                dispatcher.backend(fixture.queries.len()),
                AnswerBackend::SingleCore
            );
            let answers = dispatcher.answer_batch(&fixture.queries).unwrap();
            assert_answers_match_cpu(&fixture, &answers);
        }
    }
}
