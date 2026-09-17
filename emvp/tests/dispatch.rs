#![cfg_attr(
    feature = "gpu",
    expect(
        clippy::unwrap_used,
        reason = "fixed test fixtures establish that protocol steps and engine operations must succeed"
    )
)]

//! Dispatch-policy and engine backend-selection tests. The policy tests
//! always run; the engine demotion tests need the `gpu` feature, where a
//! GPU tier exists to demote from. The adapter-backed engine case is
//! marked `#[ignore = "requires a compute adapter"]`, so the default suite
//! skips it and the CPU-only demotion paths are exercised on every
//! machine; when run explicitly with `--ignored` on a GPU-less machine, the
//! adapter acquisition fails the test clearly instead of succeeding
//! silently.

use emvp::{AnswerBackend, MIN_PARALLEL_MULTIPLICATIONS, select_answer_backend};

#[cfg(feature = "gpu")]
use emvp::{
    AnswerEngine, AnswerJob, EmvpParams, EncryptedMatrix, EncryptedQuery, EngineWorkspace,
    MIN_GPU_MULTIPLICATIONS, MaskContextId, PrepareMatrixError, PreparedMatrix, ProtocolError,
    SecretKey,
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
    encrypted: EncryptedMatrix<M>,
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
    let encrypted = emvp::encrypt(&mut state, &matrix).unwrap();
    let mut queries = Vec::with_capacity(batch);
    for _ in 0..batch {
        let record = random_vector::<M>(ell, &mut rng);
        let (encrypted_query, _decoding_key) = emvp::query(&mut state, &record).unwrap();
        queries.push(encrypted_query);
    }
    Fixture {
        params,
        encrypted,
        queries,
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
mod engine_backends {
    use super::*;

    // One synthetic encrypted matrix at `(rows, columns)`; only the shape
    // matters for backend selection, so zero words suffice.
    fn shape_fixture(rows: usize, n: usize) -> (EmvpParams, EncryptedMatrix<MODULUS>) {
        let zero = PrimeField::<MODULUS>::new().element_u32(0);
        let params = test_params(8, n / 2, 2);
        let matrix = EncryptedMatrix::from_parts(1, rows, n, vec![zero; rows * n]).unwrap();
        (params, matrix)
    }

    /// Plans one entry of `queries` zero-filled synthetic queries against
    /// `prepared` on `engine` and returns the recorded backend, which the
    /// callers pin against their expected tier.
    fn planned_backend(
        engine: &AnswerEngine<MODULUS>,
        params: EmvpParams,
        instance_id: u128,
        prepared: &PreparedMatrix<MODULUS>,
        queries: usize,
    ) -> AnswerBackend {
        let zero = PrimeField::<MODULUS>::new().element_u32(0);
        let synthetic: Vec<EncryptedQuery<MODULUS>> = (0..queries)
            .map(|index| {
                EncryptedQuery::from_parts(
                    instance_id,
                    index as u64,
                    vec![zero; params.n().unwrap()],
                )
            })
            .collect();
        let jobs = [AnswerJob {
            matrix: prepared,
            queries: &synthetic,
        }];
        let plan = engine.plan(&jobs).unwrap();
        plan.entry(0).unwrap().backend()
    }

    #[test]
    fn a_cpu_engine_demotes_gpu_tier_work_to_the_cpu_tier() {
        // 2^24 multiplications, well above the GPU threshold: GPU tier by
        // the raw policy, rayon tier when planned on a CPU engine.
        let fixture = protocol_fixture::<MODULUS>(32, 8, 2, 65_536, 4, 0x21);
        let engine = AnswerEngine::cpu();
        assert!(!engine.has_device());
        let prepared = engine
            .prepare(fixture.params, fixture.encrypted.clone())
            .unwrap();
        assert_eq!(
            planned_backend(
                &engine,
                fixture.params,
                fixture.encrypted.instance_id(),
                &prepared,
                fixture.queries.len()
            ),
            AnswerBackend::Rayon
        );

        // The demoted run still produces exactly the per-query reference
        // answers.
        let jobs = [AnswerJob {
            matrix: &prepared,
            queries: &fixture.queries,
        }];
        let plan = engine.plan(&jobs).unwrap();
        let mut workspace = EngineWorkspace::new();
        workspace.reserve(&plan).unwrap();
        let (answers, report) = engine.execute(&plan, &mut workspace).unwrap();
        assert_eq!(report.cpu_entries, 1);
        assert_eq!(report.gpu_entries, 0);
        let zero = PrimeField::<MODULUS>::new().element_u32(0);
        let blocks = fixture.params.blocks().unwrap();
        let entry = answers.entry_answers(0).unwrap();
        for (index, query) in fixture.queries.iter().enumerate() {
            let mut expected = vec![zero; fixture.encrypted.rows() * blocks];
            emvp::answer_into(&fixture.params, &fixture.encrypted, query, &mut expected).unwrap();
            let answer = entry.answer(&fixture.queries, index).unwrap();
            assert_eq!(answer.values(), expected.as_slice());
            assert_eq!(answer.instance_id(), fixture.encrypted.instance_id());
            assert_eq!(answer.query_id(), query.query_id());
        }
    }

    #[test]
    fn flat_grids_demote_to_single_core_in_engine_plans() {
        // 2^20 multiplications with a one-row grid clear the GPU threshold
        // but cannot parallelize: the demoted CPU tier must be SingleCore,
        // not the rayon tier the raw policy's fallback would suggest.
        let (params, matrix) = shape_fixture(1, MIN_GPU_MULTIPLICATIONS);
        let engine = AnswerEngine::cpu();
        let instance_id = matrix.instance_id();
        let prepared = engine.prepare(params, matrix).unwrap();
        assert_eq!(
            planned_backend(&engine, params, instance_id, &prepared, 1),
            AnswerBackend::SingleCore
        );
    }

    #[test]
    fn gpu_tier_shapes_demote_per_the_pool_guard() {
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
            let engine = AnswerEngine::cpu();
            let flat = engine.prepare(params_flat, matrix_flat).unwrap();
            assert_eq!(
                planned_backend(&engine, params_flat, 1, &flat, 1),
                AnswerBackend::SingleCore
            );
            let wide = engine.prepare(params_wide, matrix_wide).unwrap();
            assert_eq!(
                planned_backend(&engine, params_wide, 1, &wide, 1),
                AnswerBackend::Rayon
            );
        });
        serial.install(|| {
            let engine = AnswerEngine::cpu();
            let batched = engine.prepare(params_batched, matrix_batched).unwrap();
            assert_eq!(
                planned_backend(&engine, params_batched, 1, &batched, 64),
                AnswerBackend::SingleCore
            );
        });
    }

    #[test]
    fn engine_rejects_shape_mismatches_before_preparing() {
        // The engine's preparation-time validation rejects a parameter set
        // whose codeword length disagrees with the matrix, exactly as the
        // old dispatch construction did, before any handle exists.
        let fixture = protocol_fixture::<MODULUS>(8, 8, 2, 5, 2, 0x22);
        let engine = AnswerEngine::cpu();
        let mismatched = test_params(8, 16, 2);
        assert!(matches!(
            engine.prepare(mismatched, fixture.encrypted),
            Err(PrepareMatrixError::Protocol(
                ProtocolError::LengthMismatch {
                    name: "encrypted matrix columns",
                    ..
                }
            ))
        ));
    }

    /// A GPU engine plans the device tier for a device-resident batch at
    /// exactly `MIN_GPU_MULTIPLICATIONS`, and a CPU tier below it.
    /// Requires a compute adapter; the acquisition failure fails the test
    /// clearly.
    #[test]
    #[ignore = "requires a compute adapter"]
    fn a_gpu_engine_plans_the_gpu_tier_at_the_threshold() {
        let fixture = protocol_fixture::<MODULUS>(32, 8, 2, 4_096, 4, 0x23);
        // batch * rows * n = 4 * 2^12 * 2^6 = 2^20 = MIN_GPU_MULTIPLICATIONS:
        // the exact threshold selects the device by the raw policy.
        assert_eq!(
            select_answer_backend(
                fixture.queries.len(),
                fixture.encrypted.rows(),
                fixture.params.n().unwrap(),
                rayon::current_num_threads()
            ),
            AnswerBackend::Gpu
        );
        let engine = emvp::AnswerEngine::new(64 * 1024 * 1024).unwrap();
        assert!(engine.has_device(), "this test requires a compute adapter");
        let instance_id = fixture.encrypted.instance_id();
        let prepared = engine.prepare(fixture.params, fixture.encrypted).unwrap();
        let backend = planned_backend(
            &engine,
            fixture.params,
            instance_id,
            &prepared,
            fixture.queries.len(),
        );
        assert_eq!(backend, AnswerBackend::Gpu);

        // Below the threshold the same shape plans on a CPU tier.
        let small = protocol_fixture::<MODULUS>(32, 8, 2, 256, 1, 0x24);
        let small_instance_id = small.encrypted.instance_id();
        let small_prepared = engine.prepare(small.params, small.encrypted).unwrap();
        assert_ne!(
            planned_backend(
                &engine,
                small.params,
                small_instance_id,
                &small_prepared,
                small.queries.len()
            ),
            AnswerBackend::Gpu
        );
    }
}
