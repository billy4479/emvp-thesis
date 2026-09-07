#![cfg(feature = "gpu")]
#![expect(
    clippy::unwrap_used,
    reason = "fixed test fixtures establish that protocol steps and GPU operations must succeed"
)]
#![expect(
    clippy::panic_in_result_fn,
    reason = "the Result return exists solely for the ?-based adapter probe; assertions remain the intended failure mechanism for the comparison itself"
)]

//! GPU answer-path tests. Every test first probes for a compute adapter and
//! returns early with a notice when none is available, so the suite passes
//! on GPU-less machines; on the reference machine it proves the WGSL kernel
//! bit-identical to the CPU `answer_batch`.

use std::time::Duration;

use emvp::{
    DerivedState, EmvpParams, EncryptedMatrix, EncryptedQuery, GpuAnswerer, GpuError,
    ProtocolError, SecretKey, answer_batch, encrypt, query,
};
use prime_field_layer::{FieldElement, PrimeField};
use proptest::{prelude::*, test_runner::TestCaseError};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use trapdoor_matrices::ToeplitzFastProduct;

// The deployment field: p = 998244353 is prime and below 2^30, so the WGSL
// bound arguments hold with room to spare.
const MODULUS: u32 = 998_244_353;
// A second NTT-friendly field proving the per-modulus pipeline constants
// (MODULUS, NEG_INV) are truly derived from the type parameter and cached
// per modulus.
const ALT_MODULUS: u32 = 1_073_479_681;

// Tiny deliberately-insecure research parameters: lambda = 7 bypasses the
// attack-cost validation, but the structural constraints (ell <= k, b >= 2,
// b divides n = 2k) are still enforced.
const fn test_params(ell: usize, k: usize, b: usize) -> EmvpParams {
    EmvpParams {
        k,
        ell,
        b,
        lambda: 7,
    }
}

fn random_vector<const M: u32>(length: usize, rng: &mut ChaCha20Rng) -> Vec<FieldElement<M>> {
    let field = PrimeField::<M>::new();
    (0..length).map(|_| field.sample_uniform(rng)).collect()
}

fn toeplitz_block<const M: u32>(
    n: usize,
    stream: &mut ChaCha20Rng,
    _index: usize,
) -> Result<ToeplitzFastProduct<M>, ProtocolError> {
    Ok(ToeplitzFastProduct::sample(n, stream)?)
}

fn derive_toeplitz<const M: u32>(
    k: usize,
    b: usize,
    ell: usize,
    rows: usize,
    key: u8,
) -> DerivedState<M, ToeplitzFastProduct<M>> {
    let n = 2 * k;
    let mut rng = ChaCha20Rng::seed_from_u64(0xd000 + u64::from(key));
    SecretKey::<M>::new_insecure(test_params(ell, k, b), [key; 32])
        .unwrap()
        .derive(rows, &mut rng, |stream, index| {
            toeplitz_block::<M>(n, stream, index)
        })
        .unwrap()
}

struct BatchFixture<const M: u32> {
    params: EmvpParams,
    encrypted: EncryptedMatrix<M>,
    queries: Vec<EncryptedQuery<M>>,
}

// One encrypted matrix plus `batch` queries, each answering a fresh random
// record, all derived from deterministic seeds so failures reproduce.
fn protocol_fixture<const M: u32>(
    k: usize,
    ell: usize,
    b: usize,
    rows: usize,
    batch: usize,
    key: u8,
) -> BatchFixture<M> {
    let params = test_params(ell, k, b);
    let mut rng = ChaCha20Rng::seed_from_u64(0x8f00 + u64::from(key));
    let mut state = derive_toeplitz::<M>(k, b, ell, rows, key);
    let matrix = random_vector::<M>(rows * ell, &mut rng);
    let encrypted = encrypt(&mut state, &matrix).unwrap();
    let mut queries = Vec::with_capacity(batch);
    for _ in 0..batch {
        let record = random_vector::<M>(ell, &mut rng);
        let (encrypted_query, _decoding_key) = query(&mut state, &record).unwrap();
        queries.push(encrypted_query);
    }
    BatchFixture {
        params,
        encrypted,
        queries,
    }
}

/// Probes for a compute adapter. `Ok(None)` means "skip this test"; any
/// adapter failure other than "no adapter available" is a real error and
/// propagates so the harness fails the test.
fn gpu_answerer() -> Result<Option<GpuAnswerer>, GpuError> {
    match GpuAnswerer::new_sync() {
        Ok(answerer) => Ok(Some(answerer)),
        Err(GpuError::NoAdapter { reason }) => {
            println!("skipping GPU test: no compute adapter available ({reason})");
            Ok(None)
        }
        Err(error) => Err(error),
    }
}

fn assert_gpu_matches_cpu<const M: u32>(answerer: &GpuAnswerer, fixture: &BatchFixture<M>) {
    let cpu = answer_batch(&fixture.params, &fixture.encrypted, &fixture.queries).unwrap();
    let gpu_matrix = answerer
        .upload_matrix_sync(&fixture.params, &fixture.encrypted)
        .unwrap();
    let gpu = answerer
        .answer_batch_sync(&gpu_matrix, &fixture.queries)
        .unwrap();
    assert_eq!(gpu.len(), cpu.len());
    for (gpu_answer, cpu_answer) in gpu.iter().zip(&cpu) {
        assert_eq!(gpu_answer, cpu_answer);
        assert_eq!(gpu_answer.instance_id(), cpu_answer.instance_id());
        assert_eq!(gpu_answer.query_id(), cpu_answer.query_id());
    }
}

#[test]
fn gpu_answer_matches_cpu_across_shapes_and_fields() -> Result<(), GpuError> {
    let Some(answerer) = gpu_answerer()? else {
        return Ok(());
    };
    // (k, ell, b, rows, batch) with b dividing n = 2k. The shapes cover
    // s = 1 (b = n), a single row, a single query, row counts that are not
    // multiples of any tile, and the b = 2 long-block-count extreme.
    let shapes = [
        (8, 8, 2, 5, 3),
        (8, 5, 16, 1, 1),
        (16, 16, 4, 17, 2),
        (12, 7, 6, 9, 5),
        (63, 32, 2, 3, 2),
    ];
    for (index, &(k, ell, b, rows, batch)) in shapes.iter().enumerate() {
        let fixture = protocol_fixture::<MODULUS>(k, ell, b, rows, batch, 0x10 + index as u8);
        assert_gpu_matches_cpu(&answerer, &fixture);
        let alt_fixture =
            protocol_fixture::<ALT_MODULUS>(k, ell, b, rows, batch, 0x30 + index as u8);
        assert_gpu_matches_cpu(&answerer, &alt_fixture);
    }
    Ok(())
}

#[test]
fn gpu_answer_golden_words_match_the_reference_path() -> Result<(), GpuError> {
    let Some(answerer) = gpu_answerer()? else {
        return Ok(());
    };
    let fixture = protocol_fixture::<MODULUS>(8, 8, 2, 4, 2, 0x51);
    // Golden raw Montgomery words derived from the CPU reference path on a
    // fixed fixture; the GPU readback must reproduce them word for word.
    let golden: Vec<Vec<u32>> = answer_batch(&fixture.params, &fixture.encrypted, &fixture.queries)
        .unwrap()
        .iter()
        .map(|answer| {
            answer
                .values()
                .iter()
                .map(|value| value.to_raw())
                .collect::<Vec<_>>()
        })
        .collect();
    // The canary only guards against a silently wrong GPU path if the golden
    // words are not trivially degenerate.
    assert!(
        golden
            .iter()
            .all(|words| words.iter().any(|&word| word != 0))
    );

    let gpu_matrix = answerer
        .upload_matrix_sync(&fixture.params, &fixture.encrypted)
        .unwrap();
    let gpu = answerer
        .answer_batch_sync(&gpu_matrix, &fixture.queries)
        .unwrap();
    for (answer, expected_words) in gpu.iter().zip(&golden) {
        let words: Vec<u32> = answer.values().iter().map(|value| value.to_raw()).collect();
        assert_eq!(words, *expected_words);
    }
    Ok(())
}

#[test]
fn gpu_answer_reuses_scratch_buffers_across_shapes() -> Result<(), GpuError> {
    let Some(answerer) = gpu_answerer()? else {
        return Ok(());
    };
    // First shape allocates the leased scratch set.
    assert_gpu_matches_cpu(&answerer, &protocol_fixture::<MODULUS>(8, 8, 2, 5, 2, 0x60));
    // More rows and a bigger batch grow both reused capacities in place:
    // the answer buffers are replaced, the set is leased again afterwards.
    assert_gpu_matches_cpu(
        &answerer,
        &protocol_fixture::<MODULUS>(8, 8, 2, 37, 4, 0x61),
    );
    // A smaller batch afterwards must reuse the grown buffers untouched and
    // stay bit-identical to the CPU path; stale words from the previous,
    // larger batch beyond the live slice must not leak into the answers.
    assert_gpu_matches_cpu(&answerer, &protocol_fixture::<MODULUS>(8, 8, 2, 9, 1, 0x62));
    // A second matrix instance against the same pooled buffers.
    assert_gpu_matches_cpu(
        &answerer,
        &protocol_fixture::<MODULUS>(8, 8, 2, 12, 3, 0x63),
    );
    Ok(())
}

#[test]
fn gpu_answer_timings_report_every_phase() -> Result<(), GpuError> {
    let Some(answerer) = gpu_answerer()? else {
        return Ok(());
    };
    let fixture = protocol_fixture::<MODULUS>(8, 8, 2, 4, 2, 0x64);
    let gpu_matrix = answerer
        .upload_matrix_sync(&fixture.params, &fixture.encrypted)
        .unwrap();
    let (gpu, timings) = answerer
        .answer_batch_sync_with_timings(&gpu_matrix, &fixture.queries)
        .unwrap();
    let cpu = answer_batch(&fixture.params, &fixture.encrypted, &fixture.queries).unwrap();
    assert_eq!(gpu.len(), cpu.len());
    for (gpu_answer, cpu_answer) in gpu.iter().zip(&cpu) {
        assert_eq!(gpu_answer, cpu_answer);
    }
    // A blocking call pays the wait phase at minimum, and the reported
    // total must reflect real elapsed host time.
    assert!(timings.wait_readback > Duration::ZERO);
    assert_eq!(
        timings.total(),
        timings.prepare_buffers
            + timings.encode_upload_queries
            + timings.dispatch_submit
            + timings.wait_readback
            + timings.reconstruct
    );
    Ok(())
}

#[test]
fn gpu_answer_validation_mirrors_the_cpu_path() -> Result<(), GpuError> {
    let Some(answerer) = gpu_answerer()? else {
        return Ok(());
    };
    let fixture = protocol_fixture::<MODULUS>(8, 8, 2, 4, 2, 0x52);
    let gpu_matrix = answerer
        .upload_matrix_sync(&fixture.params, &fixture.encrypted)
        .unwrap();

    assert!(matches!(
        answerer.answer_batch_sync(&gpu_matrix, &[]),
        Err(GpuError::LengthMismatch {
            name: "queries",
            expected: 1,
            actual: 0,
        })
    ));

    let n = fixture.params.n().unwrap();
    let truncated = EncryptedQuery::from_parts(
        fixture.encrypted.instance_id(),
        fixture.queries[0].query_id(),
        fixture.queries[0].values()[..n - 1].to_vec(),
    );
    let mixed = [fixture.queries[0].clone(), truncated];
    assert!(matches!(
        answerer.answer_batch_sync(&gpu_matrix, &mixed),
        Err(GpuError::LengthMismatch {
            name: "encrypted query",
            ..
        })
    ));

    let other = protocol_fixture::<MODULUS>(8, 8, 2, 4, 1, 0x53);
    let foreign = [other.queries.into_iter().next().unwrap()];
    assert!(matches!(
        answerer.answer_batch_sync(&gpu_matrix, &foreign),
        Err(GpuError::InstanceMismatch { .. })
    ));

    // Parameters whose codeword length disagrees with the uploaded matrix
    // are rejected before any transfer.
    let mismatched_params = test_params(8, 16, 2);
    assert!(matches!(
        answerer.upload_matrix_sync(&mismatched_params, &fixture.encrypted),
        Err(GpuError::LengthMismatch {
            name: "encrypted matrix columns",
            ..
        })
    ));
    Ok(())
}

fn small_answer_strategy() -> impl Strategy<Value = (EmvpParams, usize, usize)> {
    (4_usize..=63).prop_flat_map(|k| {
        (1_usize..=k, 2_usize..=(2 * k))
            .prop_filter_map("b must divide n = 2k", move |(ell, b)| {
                if (2 * k).is_multiple_of(b) {
                    Some(EmvpParams {
                        k,
                        ell,
                        b,
                        lambda: 7,
                    })
                } else {
                    None
                }
            })
            .prop_flat_map(|params| (Just(params), 1_usize..=8_usize, 1_usize..=5_usize))
    })
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn gpu_answer_matches_cpu_on_random_small_params(
        (params, rows, batch) in small_answer_strategy(),
        key in 0_u8..=63_u8,
    ) {
        // The proptest sugar wraps the body in a `Result`-returning closure,
        // so the no-adapter skip and unexpected probe failures return from
        // that closure instead of using the usual `?`-on-`Result` test form.
        let answerer = match gpu_answerer() {
            Ok(Some(answerer)) => answerer,
            Ok(None) => {
                println!("skipping GPU test: no compute adapter available");
                return Ok(());
            }
            Err(error) => return Err(TestCaseError::fail(error.to_string())),
        };
        let mut rng = ChaCha20Rng::seed_from_u64(0x9a00 + u64::from(key));
        let mut state = derive_toeplitz::<MODULUS>(params.k, params.b, params.ell, rows, key);
        let matrix = random_vector::<MODULUS>(rows * params.ell, &mut rng);
        let encrypted = encrypt(&mut state, &matrix).unwrap();
        let mut queries = Vec::with_capacity(batch);
        for _ in 0..batch {
            let record = random_vector::<MODULUS>(params.ell, &mut rng);
            let (encrypted_query, _decoding_key) = query(&mut state, &record).unwrap();
            queries.push(encrypted_query);
        }
        let cpu = answer_batch(&params, &encrypted, &queries).unwrap();
        let gpu_matrix = answerer.upload_matrix_sync(&params, &encrypted).unwrap();
        let gpu = answerer.answer_batch_sync(&gpu_matrix, &queries).unwrap();
        prop_assert_eq!(gpu.len(), cpu.len());
        for (gpu_answer, cpu_answer) in gpu.iter().zip(&cpu) {
            prop_assert_eq!(gpu_answer, cpu_answer);
        }
    }
}
