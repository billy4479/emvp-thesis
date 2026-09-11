#![cfg_attr(
    feature = "gpu",
    expect(
        clippy::unwrap_used,
        reason = "fixed test fixtures establish that protocol steps and dispatch operations must succeed"
    )
)]

//! Dispatch-policy and answer-dispatcher tests. The policy tests always
//! run; the dispatcher tests need the `gpu` feature. Only the adapter-backed
//! cases probe for a compute device and skip with a notice when none is
//! available, so the CPU-only demotion paths are exercised on every machine.

use emvp::{AnswerBackend, MIN_PARALLEL_MULTIPLICATIONS, select_answer_backend};

#[cfg(feature = "gpu")]
use emvp::{
    AnswerDispatchError, AnswerDispatcher, EmvpParams, EncryptedQuery, GpuAnswerer, GpuError,
    ProtocolError, SecretKey, answer_batch, encrypt, query,
};
#[cfg(feature = "gpu")]
use prime_field_layer::{FieldElement, PrimeField};
#[cfg(feature = "gpu")]
use rand_chacha::ChaCha20Rng;
#[cfg(feature = "gpu")]
use rand_core::SeedableRng;
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
        .derive(rows, &mut derive_rng, |stream, _index| {
            toeplitz_block::<M>(n, stream)
        })
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
fn assert_answers_match_cpu<const M: u32>(
    fixture: &Fixture<M>,
    answers: &[emvp::AnswerMatrix<M>],
) {
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

    /// Probes for a compute adapter; `None` means "skip this test".
    fn gpu_answerer() -> Result<Option<GpuAnswerer>, GpuError> {
        match GpuAnswerer::new_sync() {
            Ok(answerer) => Ok(Some(answerer)),
            Err(GpuError::NoAdapter { reason }) => {
                println!("skipping adapter test: no compute adapter ({reason})");
                Ok(None)
            }
            Err(error) => Err(error),
        }
    }

    #[test]
    fn cpu_only_dispatcher_demotes_the_gpu_tier() {
        // 2^24 multiplications exactly: GPU tier by policy, rayon tier
        // without a device.
        let fixture = protocol_fixture::<MODULUS>(32, 8, 2, 65_536, 4, 0x21);
        let dispatcher = AnswerDispatcher::cpu(fixture.params, &fixture.encrypted).unwrap();
        assert!(!dispatcher.is_device_backed());
        assert_eq!(dispatcher.backend(fixture.queries.len()), AnswerBackend::Rayon);
        let answers = dispatcher.answer_batch(&fixture.queries).unwrap();
        assert_answers_match_cpu(&fixture, &answers);
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
    fn armed_dispatcher_selects_the_gpu_tier_at_the_threshold() {
        let Ok(Some(answerer)) = gpu_answerer() else {
            return;
        };
        // batch * rows * n = 4 * 2^16 * 2^6 = 2^24 = MIN_GPU_MULTIPLICATIONS:
        // the exact threshold must select the device.
        let fixture = protocol_fixture::<MODULUS>(32, 8, 2, 65_536, 4, 0x23);
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
    fn armed_dispatcher_answers_small_batches_across_shapes() {
        let Ok(Some(answerer)) = gpu_answerer() else {
            return;
        };
        let shapes = [
            (8, 8, 2, 5, 3),
            (16, 16, 4, 17, 2),
            (12, 7, 6, 9, 5),
        ];
        for (index, &(k, ell, b, rows, batch)) in shapes.iter().enumerate() {
            let fixture = protocol_fixture::<MODULUS>(k, ell, b, rows, batch, 0x25 + index as u8);
            let dispatcher =
                AnswerDispatcher::new(fixture.params, &fixture.encrypted, Some(&answerer))
                    .unwrap();
            assert_eq!(
                dispatcher.backend(fixture.queries.len()),
                AnswerBackend::SingleCore
            );
            let answers = dispatcher.answer_batch(&fixture.queries).unwrap();
            assert_answers_match_cpu(&fixture, &answers);
        }
    }
}
