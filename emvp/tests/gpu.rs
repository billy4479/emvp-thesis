#![cfg(feature = "gpu")]
#![expect(
    clippy::unwrap_used,
    reason = "fixed test fixtures establish that protocol steps and GPU operations must succeed"
)]
#![expect(
    clippy::panic_in_result_fn,
    reason = "the Result return exists solely for the ?-based adapter acquisition, which fails clearly without a device; assertions remain the intended failure mechanism for the comparisons themselves"
)]

//! GPU answer-path tests.
//!
//! Every test in this file requires a compute adapter, so each is marked
//! `#[ignore = "requires a compute adapter"]`: the default `cargo test`
//! run skips them, while the device-independent shader parse-and-validate
//! and host-reconstruction checks live in `emvp/src/gpu`'s unit tests.
//! Running this suite explicitly with `--ignored` on a machine with a
//! compute adapter proves the WGSL kernel bit-identical to the CPU
//! `answer_batch`; on a GPU-less machine the adapter acquisition fails the
//! test clearly instead of silently succeeding.

use std::time::Duration;

use emvp::{
    AnswerPlan, AnswerWorkspace, DecodingKey, DerivedState, EmvpParams, EncryptedMatrix,
    EncryptedQuery, GpuAnswerer, GpuError, MaskContextId, ProtocolError, SecretKey, answer_batch,
    decode_into, encrypt, query,
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
// The widest odd prime the WGSL arithmetic supports: p = 2^31 - 1 is the
// extreme of the kernel's `2 < MODULUS < 2^31` range. It divides
// `2^32 - 2`, so the Montgomery conversion constant is R = 2^32 ≡ 2 (mod p)
// and the hand-derived golden words are trivially transparent:
// word(a) = 2a mod p.
const WIDE_MODULUS: u32 = 2_147_483_647;

// Stable context identifier of this suite's Toeplitz fixtures, folded with
// the concrete configuration so distinct shapes never share a context.
const CONTEXT_TOEPLITZ: MaskContextId = MaskContextId::from_u64(0x544f_4550);

const fn fixture_context(k: usize, ell: usize) -> MaskContextId {
    MaskContextId::new(
        CONTEXT_TOEPLITZ.get() ^ ((k as u64 as u128) << 64) ^ ((ell as u64 as u128) << 32),
    )
}

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
        .derive(fixture_context(k, ell), rows, &mut rng, |stream, index| {
            toeplitz_block::<M>(n, stream, index)
        })
        .unwrap()
}

struct BatchFixture<const M: u32> {
    params: EmvpParams,
    encrypted: EncryptedMatrix<M>,
    queries: Vec<EncryptedQuery<M>>,
    decoding_keys: Vec<DecodingKey<M>>,
}

// One encrypted matrix plus `batch` queries with their decoding keys, each
// answering a fresh random record, all derived from deterministic seeds so
// failures reproduce.
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
    let mut decoding_keys = Vec::with_capacity(batch);
    for _ in 0..batch {
        let record = random_vector::<M>(ell, &mut rng);
        let (encrypted_query, decoding_key) = query(&mut state, &record).unwrap();
        queries.push(encrypted_query);
        decoding_keys.push(decoding_key);
    }
    BatchFixture {
        params,
        encrypted,
        queries,
        decoding_keys,
    }
}

/// Builds the fully transparent two-query fixture: the encrypted matrix and
/// the encrypted queries are constructed directly from their parts with
/// small hand-picked values — no trapdoor, no PRF, no client-side
/// derivation — so the kernel's expected output is derivable by hand.
///
/// Shape: `n = 2` (`k = 1`), `b = 2`, `s = 1`; matrix rows `[1, 2]` and
/// `[3, 4]`; queries `[5, 6]` and `[7, 8]`. The kernel answers output
/// `(query q, row r)` with the field sum
/// `M[r][0] * Q[q][0] + M[r][1] * Q[q][1]`, so the expected sums are 17,
/// 39, 23, and 53.
fn transparent_fixture<const M: u32>() -> (EmvpParams, EncryptedMatrix<M>, Vec<EncryptedQuery<M>>) {
    let field = PrimeField::<M>::new();
    // ell = 1 <= k = 1, and b = 2 divides n = 2: the minimal valid shape.
    let params = test_params(1, 1, 2);
    let encrypted = EncryptedMatrix::from_parts(
        0xA11,
        2,
        2,
        vec![
            field.element_u32(1),
            field.element_u32(2),
            field.element_u32(3),
            field.element_u32(4),
        ],
    )
    .unwrap();
    let queries = vec![
        EncryptedQuery::from_parts(0xA11, 0, vec![field.element_u32(5), field.element_u32(6)]),
        EncryptedQuery::from_parts(0xA11, 1, vec![field.element_u32(7), field.element_u32(8)]),
    ];
    (params, encrypted, queries)
}

/// Acquires the answerer. Every caller is `#[ignore]`d, so when a test runs
/// (explicitly with `--ignored`) a missing adapter fails it clearly instead
/// of succeeding silently.
fn gpu_answerer() -> Result<GpuAnswerer, GpuError> {
    GpuAnswerer::new()
}

fn assert_gpu_matches_cpu<const M: u32>(answerer: &GpuAnswerer, fixture: &BatchFixture<M>) {
    let cpu = answer_batch(&fixture.params, &fixture.encrypted, &fixture.queries).unwrap();
    let gpu_matrix = answerer
        .upload_matrix(&fixture.params, &fixture.encrypted)
        .unwrap();
    let gpu = answerer
        .answer_batch(&gpu_matrix, &fixture.queries)
        .unwrap();
    assert_eq!(gpu.len(), cpu.len());
    for (gpu_answer, cpu_answer) in gpu.iter().zip(&cpu) {
        assert_eq!(gpu_answer, cpu_answer);
        assert_eq!(gpu_answer.instance_id(), cpu_answer.instance_id());
        assert_eq!(gpu_answer.query_id(), cpu_answer.query_id());
    }
}

#[test]
#[ignore = "requires a compute adapter"]
fn gpu_answer_matches_cpu_across_shapes_and_fields() -> Result<(), GpuError> {
    let answerer = gpu_answerer()?;
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
#[ignore = "requires a compute adapter"]
fn gpu_answer_matches_hardcoded_golden_words() -> Result<(), GpuError> {
    // Expected raw Montgomery words, derived by hand from the transparent
    // fixture: the kernel answers (q, r) with the field sum
    // `M[r][0] * Q[q][0] + M[r][1] * Q[q][1]` and returns its canonical
    // Montgomery word `sum * 2^32 mod p` = `sum * 301989884 mod p`:
    //   (q0, r0): 1*5 + 2*6 = 17  -> 17 * 301989884 mod p = 142606263
    //   (q0, r1): 3*5 + 4*6 = 39  -> 39 * 301989884 mod p = 796917593
    //   (q1, r0): 1*7 + 2*8 = 23  -> 23 * 301989884 mod p = 956301214
    //   (q1, r1): 3*7 + 4*8 = 53  -> 53 * 301989884 mod p = 33554204
    // The expectations are fixed constants: nothing recomputes them from
    // the CPU answer path at runtime.
    const GOLDEN_WORDS: [&[u32]; 2] = [&[142_606_263, 796_917_593], &[956_301_214, 33_554_204]];
    const GOLDEN_SUMS: [&[u32]; 2] = [&[17, 39], &[23, 53]];

    let answerer = gpu_answerer()?;
    let (params, encrypted, queries) = transparent_fixture::<MODULUS>();

    let gpu_matrix = answerer.upload_matrix(&params, &encrypted)?;
    let gpu = answerer.answer_batch(&gpu_matrix, &queries)?;
    let field = PrimeField::<MODULUS>::new();
    for (index, answer) in gpu.iter().enumerate() {
        let words: Vec<u32> = answer.values().iter().map(|value| value.to_raw()).collect();
        assert_eq!(words, GOLDEN_WORDS[index], "query {index} words");
        for (value, &sum) in answer.values().iter().zip(GOLDEN_SUMS[index]) {
            assert_eq!(*value, field.element_u32(sum), "query {index} sum {sum}");
        }
    }
    Ok(())
}

#[test]
#[ignore = "requires a compute adapter"]
fn gpu_answer_handles_near_modulus_operands() -> Result<(), GpuError> {
    let answerer = gpu_answerer()?;
    // Raw words within six of the modulus exercise the kernel's REDC and
    // fold bounds at the extreme of the canonical residue range: every
    // product here is of words near p, yet each partial product
    // (p - a)(p - c) ≡ a * c (mod p) keeps the 96-bit accumulator exact.
    let word = |delta: u32| -> FieldElement<MODULUS> {
        FieldElement::<MODULUS>::try_from_raw(MODULUS - delta).unwrap()
    };
    let params = test_params(1, 1, 2);
    // rows = [p-1, p-2] and [p-5, p-6]; query = [p-3, p-4]:
    //   row0: (p-1)(p-3) + (p-2)(p-4) ≡ 1*3 + 2*4 = 11 (mod p)
    //   row1: (p-5)(p-3) + (p-6)(p-4) ≡ 5*3 + 6*4 = 39 (mod p)
    // Expected canonical Montgomery words:
    //   11 * 301989884 mod p = 327155665
    //   39 * 301989884 mod p = 796917593
    let encrypted =
        EncryptedMatrix::from_parts(0xBEE, 2, 2, vec![word(1), word(2), word(5), word(6)]).unwrap();
    let queries = [EncryptedQuery::from_parts(0xBEE, 0, vec![word(3), word(4)])];

    let gpu_matrix = answerer.upload_matrix(&params, &encrypted)?;
    let gpu = answerer.answer_batch(&gpu_matrix, &queries)?;
    let field = PrimeField::<MODULUS>::new();
    let words: Vec<u32> = gpu[0].values().iter().map(|value| value.to_raw()).collect();
    assert_eq!(words, [327_155_665, 796_917_593]);
    for (value, sum) in gpu[0].values().iter().zip([11_u32, 39]) {
        assert_eq!(*value, field.element_u32(sum));
    }
    Ok(())
}

#[test]
#[ignore = "requires a compute adapter"]
fn gpu_answer_matches_hand_derived_words_on_the_widest_supported_odd_prime() -> Result<(), GpuError>
{
    // The transparent fixture over the widest supported odd prime
    // p = 2^31 - 1, the extreme of the kernel's modulus range. Because
    // p divides 2^32 - 2, the Montgomery constant is R = 2^32 ≡ 2 (mod p),
    // so every expected raw word is trivially hand-derivable:
    //   17 -> 34, 39 -> 78, 23 -> 46, 53 -> 106.
    const GOLDEN_WORDS: [&[u32]; 2] = [&[34, 78], &[46, 106]];
    const GOLDEN_SUMS: [&[u32]; 2] = [&[17, 39], &[23, 53]];

    let answerer = gpu_answerer()?;
    let (params, encrypted, queries) = transparent_fixture::<WIDE_MODULUS>();
    let gpu_matrix = answerer.upload_matrix(&params, &encrypted)?;
    let gpu = answerer.answer_batch(&gpu_matrix, &queries)?;
    let field = PrimeField::<WIDE_MODULUS>::new();
    for (index, answer) in gpu.iter().enumerate() {
        let words: Vec<u32> = answer.values().iter().map(|value| value.to_raw()).collect();
        assert_eq!(words, GOLDEN_WORDS[index], "query {index} words");
        for (value, &sum) in answer.values().iter().zip(GOLDEN_SUMS[index]) {
            assert_eq!(*value, field.element_u32(sum), "query {index} sum {sum}");
        }
    }
    // The same transparent answers must equal the CPU reference path over
    // this modulus, pinning the per-modulus pipeline constants (NEG_INV,
    // R2) derivation end to end.
    let cpu = answer_batch(&params, &encrypted, &queries)?;
    assert_eq!(gpu, cpu);
    Ok(())
}

#[test]
#[ignore = "requires a compute adapter"]
fn gpu_answer_followed_by_decode_matches_the_cpu_reference() -> Result<(), GpuError> {
    let answerer = gpu_answerer()?;
    let fixture = protocol_fixture::<MODULUS>(8, 8, 2, 5, 3, 0x55);
    let gpu_matrix = answerer.upload_matrix(&fixture.params, &fixture.encrypted)?;
    let gpu = answerer.answer_batch(&gpu_matrix, &fixture.queries)?;
    let cpu = answer_batch(&fixture.params, &fixture.encrypted, &fixture.queries)?;
    let zero = PrimeField::<MODULUS>::new().element_u32(0);
    for ((gpu_answer, cpu_answer), key) in gpu.iter().zip(&cpu).zip(&fixture.decoding_keys) {
        let mut gpu_decoded = vec![zero; fixture.encrypted.rows()];
        decode_into(gpu_answer, key, &mut gpu_decoded)?;
        let mut cpu_decoded = vec![zero; fixture.encrypted.rows()];
        decode_into(cpu_answer, key, &mut cpu_decoded)?;
        assert_eq!(gpu_decoded, cpu_decoded);
    }
    Ok(())
}

#[test]
#[ignore = "requires a compute adapter"]
fn gpu_answer_reuses_scratch_buffers_across_shapes() -> Result<(), GpuError> {
    let answerer = gpu_answerer()?;
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
#[ignore = "requires a compute adapter"]
fn gpu_answer_timings_report_every_phase() -> Result<(), GpuError> {
    let answerer = gpu_answerer()?;
    let fixture = protocol_fixture::<MODULUS>(8, 8, 2, 4, 2, 0x64);
    let gpu_matrix = answerer.upload_matrix(&fixture.params, &fixture.encrypted)?;
    let (gpu, timings) = answerer.answer_batch_with_timings(&gpu_matrix, &fixture.queries)?;
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
#[ignore = "requires a compute adapter"]
fn gpu_answer_validation_mirrors_the_cpu_path() -> Result<(), GpuError> {
    let answerer = gpu_answerer()?;
    let fixture = protocol_fixture::<MODULUS>(8, 8, 2, 4, 2, 0x52);
    let gpu_matrix = answerer
        .upload_matrix(&fixture.params, &fixture.encrypted)
        .unwrap();

    assert!(matches!(
        answerer.answer_batch(&gpu_matrix, &[]),
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
        answerer.answer_batch(&gpu_matrix, &mixed),
        Err(GpuError::LengthMismatch {
            name: "encrypted query",
            ..
        })
    ));

    let other = protocol_fixture::<MODULUS>(8, 8, 2, 4, 1, 0x53);
    let foreign = [other.queries.into_iter().next().unwrap()];
    assert!(matches!(
        answerer.answer_batch(&gpu_matrix, &foreign),
        Err(GpuError::InstanceMismatch { .. })
    ));

    // Parameters whose codeword length disagrees with the uploaded matrix
    // are rejected before any transfer.
    let mismatched_params = test_params(8, 16, 2);
    assert!(matches!(
        answerer.upload_matrix(&mismatched_params, &fixture.encrypted),
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
    #[ignore = "requires a compute adapter"]
    fn gpu_answer_matches_cpu_on_random_small_params(
        (params, rows, batch) in small_answer_strategy(),
        key in 0_u8..=63_u8,
    ) {
        // The proptest sugar wraps the body in a `Result`-returning closure,
        // so the adapter acquisition failure — which must fail the ignored
        // test clearly — returns from that closure instead of using the
        // usual `?`-on-`Result` test form.
        let answerer = match gpu_answerer() {
            Ok(answerer) => answerer,
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
        let gpu_matrix = answerer.upload_matrix(&params, &encrypted).unwrap();
        let gpu = answerer.answer_batch(&gpu_matrix, &queries).unwrap();
        prop_assert_eq!(gpu.len(), cpu.len());
        for (gpu_answer, cpu_answer) in gpu.iter().zip(&cpu) {
            prop_assert_eq!(gpu_answer, cpu_answer);
        }
    }
}

#[test]
#[ignore = "requires a compute adapter"]
fn execute_into_reuses_the_workspace_across_calls() -> Result<(), GpuError> {
    let answerer = gpu_answerer()?;
    let (params, encrypted, queries) = transparent_fixture::<MODULUS>();
    // The arena is reserved through the shared CPU plan type: the shape
    // arithmetic is identical on both tiers.
    let plan = AnswerPlan::plan(&params, &encrypted, &queries).unwrap();
    let mut workspace = AnswerWorkspace::new();
    workspace.reserve(&plan).unwrap();

    let gpu_matrix = answerer.upload_matrix(&params, &encrypted)?;
    let (first_ptr, first_shape, first_arena) = {
        let (first, _) =
            answerer.execute_answer_batch_into(&gpu_matrix, &queries, &mut workspace)?;
        (first.arena().as_ptr(), first.shape(), first.arena().to_vec())
    };
    let (second, _) =
        answerer.execute_answer_batch_into(&gpu_matrix, &queries, &mut workspace)?;

    // Identical answers over the identical storage prove the workspace was
    // reused: the second execute neither reallocated nor grew the arena.
    assert_eq!(first_ptr, second.arena().as_ptr());
    assert_eq!(first_shape, second.shape());
    assert_eq!(first_arena, second.arena());
    let cpu = answer_batch(&params, &encrypted, &queries).unwrap();
    for (index, cpu_answer) in cpu.iter().enumerate() {
        let view = second.answer(&queries, index).unwrap();
        assert_eq!(view.values(), cpu_answer.values());
    }
    Ok(())
}
