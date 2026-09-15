#![expect(
    clippy::unwrap_used,
    reason = "fixed test fixtures establish that protocol steps must succeed"
)]

use emvp::{
    AnswerMatrix, DecodingKey, DerivedState, EmvpParams, EncryptedMatrix, EncryptedQuery,
    MaskContextId, ProtocolError, SecretKey, TdmMask, answer_batch, answer_into, decode_into,
    encrypt, purpose, query, query_batch, query_with_scratch, search,
};
use prime_field_layer::{FieldElement, PrimeField};
use proptest::prelude::*;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use rayon::{ThreadPool, ThreadPoolBuilder};
use trapdoor_matrices::{
    DenseMatrix, IrreducibleRingLpn, RaaWeightedProduct, TdmError, ToeplitzFastProduct,
};

// NTT-friendly prime: 1_073_479_681 - 1 is divisible by 2^18.
const MODULUS: u32 = 1_073_479_681;

// Tiny deliberately-insecure research parameters: k = 8, ell = 8, b = 2,
// so n = 16 and s = 8. `new_insecure` re-validates structural dimensions
// but skips the concrete attack-cost validation.
const fn test_params(ell: usize) -> EmvpParams {
    EmvpParams {
        k: 8,
        ell,
        b: 2,
        lambda: 7,
    }
}

// Stable mask-suite context identifiers. Every derive/restore call names
// the context of its mask construction and configuration: the suite base
// separates Toeplitz, RAA, Ring-LPN, and the deliberately failing mask, and
// each configuration variant of a suite carries its own value. A real
// deployment fixes one tag per suite and changes it whenever the builder or
// its configuration changes.
const CONTEXT_TOEPLITZ: MaskContextId = MaskContextId::from_u64(0x0100);
const CONTEXT_TOEPLITZ_PADDED: MaskContextId = MaskContextId::from_u64(0x0101);
const CONTEXT_TOEPLITZ_MULTI_BLOCK: MaskContextId = MaskContextId::from_u64(0x0102);
const CONTEXT_TOEPLITZ_BLOCK_4: MaskContextId = MaskContextId::from_u64(0x0104);
const CONTEXT_TOEPLITZ_SINGLE_BLOCK: MaskContextId = MaskContextId::from_u64(0x0105);
const CONTEXT_TOEPLITZ_ODD_RANK: MaskContextId = MaskContextId::from_u64(0x0106);
const CONTEXT_TOEPLITZ_ALT: MaskContextId = MaskContextId::from_u64(0x0107);
const CONTEXT_RAA: MaskContextId = MaskContextId::from_u64(0x0200);
const CONTEXT_RING: MaskContextId = MaskContextId::from_u64(0x0300);
const CONTEXT_FAILING: MaskContextId = MaskContextId::from_u64(0x0400);
const CONTEXT_SEARCHED: MaskContextId = MaskContextId::from_u64(0x0500);

const fn field() -> PrimeField<MODULUS> {
    PrimeField::<MODULUS>::new()
}

fn random_vector(length: usize, rng: &mut ChaCha20Rng) -> Vec<FieldElement<MODULUS>> {
    let field = field();
    (0..length).map(|_| field.sample_uniform(rng)).collect()
}

fn toeplitz_block(
    stream: &mut ChaCha20Rng,
    _index: usize,
) -> Result<ToeplitzFastProduct<MODULUS>, ProtocolError> {
    Ok(ToeplitzFastProduct::sample(16, stream)?)
}

fn raa_block(
    stream: &mut ChaCha20Rng,
    _index: usize,
) -> Result<RaaWeightedProduct<MODULUS>, ProtocolError> {
    Ok(RaaWeightedProduct::sample_nonzero(16, 2, stream)?)
}

// A square `n x n` Ring-LPN mask block for the test parameters, where
// `n = 2k = 16`. The sparse secret `E` has a fixed small column weight,
// built by the shared deterministic test/benchmark builder.
fn ring_block(
    stream: &mut ChaCha20Rng,
    _index: usize,
) -> Result<IrreducibleRingLpn<MODULUS>, ProtocolError> {
    Ok(trapdoor_matrices::testing::ring_block::<MODULUS, _>(
        16, 4, stream,
    )?)
}

// A mask block that succeeds at every construction step but always fails
// evaluation. It defines the "practical deliberately failing mask" used to
// pin that a query failure after reservation consumes the reserved
// identifier, so a retry cannot reuse the query randomness.
struct FailingApplyMask {
    inner: ToeplitzFastProduct<MODULUS>,
}

impl TdmMask<MODULUS> for FailingApplyMask {
    type Scratch = <ToeplitzFastProduct<MODULUS> as TdmMask<MODULUS>>::Scratch;

    fn dims(&self) -> (usize, usize) {
        self.inner.dims()
    }

    fn scratch(&self) -> Self::Scratch {
        self.inner.scratch()
    }

    fn apply(
        &self,
        _input: &[FieldElement<MODULUS>],
        _output: &mut [FieldElement<MODULUS>],
        _scratch: &mut Self::Scratch,
    ) -> Result<(), TdmError> {
        Err(TdmError::ZeroDimension("deliberately failing test mask"))
    }

    fn materialize(&self) -> Result<DenseMatrix<MODULUS>, TdmError> {
        self.inner.materialize()
    }
}

fn derive_toeplitz(
    rows: usize,
    ell: usize,
    key: u8,
) -> DerivedState<MODULUS, ToeplitzFastProduct<MODULUS>> {
    let mut rng = ChaCha20Rng::seed_from_u64(0xd000 + u64::from(key));
    SecretKey::<MODULUS>::new_insecure(test_params(ell), [key; 32])
        .unwrap()
        .derive(CONTEXT_TOEPLITZ, rows, &mut rng, toeplitz_block)
        .unwrap()
}

fn answer_matrix(
    params: &EmvpParams,
    encrypted: &EncryptedMatrix<MODULUS>,
    query: &EncryptedQuery<MODULUS>,
) -> AnswerMatrix<MODULUS> {
    let zero = field().element_u32(0);
    let mut values = vec![zero; encrypted.rows() * params.blocks().unwrap()];
    answer_into(params, encrypted, query, &mut values).unwrap();
    AnswerMatrix::from_parts(
        encrypted.instance_id(),
        query.query_id(),
        values,
        encrypted.rows(),
        params.blocks().unwrap(),
    )
}

fn decode_answer(
    answer: &AnswerMatrix<MODULUS>,
    key: &DecodingKey<MODULUS>,
) -> Vec<FieldElement<MODULUS>> {
    let mut output = vec![field().element_u32(0); answer.rows()];
    decode_into(answer, key, &mut output).unwrap();
    output
}

fn run_protocol<M: TdmMask<MODULUS>>(
    state: &mut DerivedState<MODULUS, M>,
    matrix: &[FieldElement<MODULUS>],
    q: &[FieldElement<MODULUS>],
) -> Vec<FieldElement<MODULUS>> {
    let encrypted = encrypt(state, matrix).unwrap();
    let (encrypted_query, decoding_key) = query(state, q).unwrap();
    let answer = answer_matrix(&state.params(), &encrypted, &encrypted_query);
    decode_answer(&answer, &decoding_key)
}

fn naive_matrix_vector(
    matrix: &[FieldElement<MODULUS>],
    q: &[FieldElement<MODULUS>],
    rows: usize,
    ell: usize,
) -> Vec<FieldElement<MODULUS>> {
    let field = field();
    (0..rows)
        .map(|row| {
            let mut accumulator = field.element_u32(0);
            for column in 0..ell {
                accumulator += matrix[row * ell + column] * q[column];
            }
            accumulator
        })
        .collect()
}

#[test]
fn end_to_end_matches_plaintext_product() {
    let mut rng = ChaCha20Rng::seed_from_u64(1);
    let (rows, ell) = (5_usize, 8_usize);
    let mut state = derive_toeplitz(rows, ell, 2);
    let matrix = random_vector(rows * ell, &mut rng);
    let q = random_vector(ell, &mut rng);
    let decoded = run_protocol(&mut state, &matrix, &q);
    let expected = naive_matrix_vector(&matrix, &q, rows, ell);
    assert_eq!(decoded, expected);
}

#[test]
fn end_to_end_works_with_raa_mask() {
    let mut rng = ChaCha20Rng::seed_from_u64(3);
    let (rows, ell) = (7_usize, 8_usize);
    let mut state = SecretKey::<MODULUS>::new_insecure(test_params(ell), [4; 32])
        .unwrap()
        .derive(CONTEXT_RAA, rows, &mut rng, raa_block)
        .unwrap();
    let matrix = random_vector(rows * ell, &mut rng);
    let q = random_vector(ell, &mut rng);
    let decoded = run_protocol(&mut state, &matrix, &q);
    let expected = naive_matrix_vector(&matrix, &q, rows, ell);
    assert_eq!(decoded, expected);
}

#[test]
fn end_to_end_works_with_ring_lpn_mask() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x7400);
    let (rows, ell) = (7_usize, 8_usize);
    let mut state = SecretKey::<MODULUS>::new_insecure(test_params(ell), [0x74; 32])
        .unwrap()
        .derive(CONTEXT_RING, rows, &mut rng, ring_block)
        .unwrap();
    let matrix = random_vector(rows * ell, &mut rng);
    let q = random_vector(ell, &mut rng);
    let decoded = run_protocol(&mut state, &matrix, &q);
    let expected = naive_matrix_vector(&matrix, &q, rows, ell);
    assert_eq!(decoded, expected);
}

#[test]
fn end_to_end_works_with_blocks_of_four() {
    // b = 4 over n = 16 stacks s = 4 blocks, the block-count/b-size middle
    // ground between the b = 2 default and the s = 1 extreme below.
    let mut rng = ChaCha20Rng::seed_from_u64(0x9100);
    let (rows, ell) = (5_usize, 8_usize);
    let params = EmvpParams {
        k: 8,
        ell,
        b: 4,
        lambda: 7,
    };
    let mut state = SecretKey::<MODULUS>::new_insecure(params, [0x91; 32])
        .unwrap()
        .derive(CONTEXT_TOEPLITZ_BLOCK_4, rows, &mut rng, toeplitz_block)
        .unwrap();
    let matrix = random_vector(rows * ell, &mut rng);
    let q = random_vector(ell, &mut rng);
    assert_eq!(state.params().blocks().unwrap(), 4);
    let decoded = run_protocol(&mut state, &matrix, &q);
    let expected = naive_matrix_vector(&matrix, &q, rows, ell);
    assert_eq!(decoded, expected);
}

#[test]
fn end_to_end_works_with_a_single_query_block() {
    // b = n = 16 collapses the query to one scaled block, s = 1: the
    // supported single-block shape.
    let mut rng = ChaCha20Rng::seed_from_u64(0x9200);
    let (rows, ell) = (5_usize, 8_usize);
    let params = EmvpParams {
        k: 8,
        ell,
        b: 16,
        lambda: 7,
    };
    let mut state = SecretKey::<MODULUS>::new_insecure(params, [0x92; 32])
        .unwrap()
        .derive(
            CONTEXT_TOEPLITZ_SINGLE_BLOCK,
            rows,
            &mut rng,
            toeplitz_block,
        )
        .unwrap();
    let matrix = random_vector(rows * ell, &mut rng);
    let q = random_vector(ell, &mut rng);
    assert_eq!(state.params().blocks().unwrap(), 1);
    let decoded = run_protocol(&mut state, &matrix, &q);
    let expected = naive_matrix_vector(&matrix, &q, rows, ell);
    assert_eq!(decoded, expected);
}

#[test]
fn end_to_end_works_with_a_non_power_of_two_rank() {
    // k = 12 gives n = 24 and b = 6 gives s = 4: neither n nor b is a power
    // of two, exercising the structural path beyond power-of-two shapes.
    let mut rng = ChaCha20Rng::seed_from_u64(0x9300);
    let (rows, ell) = (5_usize, 7_usize);
    let params = EmvpParams {
        k: 12,
        ell,
        b: 6,
        lambda: 7,
    };
    let mut state = SecretKey::<MODULUS>::new_insecure(params, [0x93; 32])
        .unwrap()
        .derive(
            CONTEXT_TOEPLITZ_ODD_RANK,
            rows,
            &mut rng,
            |stream, _index| Ok(ToeplitzFastProduct::sample(24, stream)?),
        )
        .unwrap();
    let matrix = random_vector(rows * ell, &mut rng);
    let q = random_vector(ell, &mut rng);
    assert_eq!(state.params().n().unwrap(), 24);
    let decoded = run_protocol(&mut state, &matrix, &q);
    let expected = naive_matrix_vector(&matrix, &q, rows, ell);
    assert_eq!(decoded, expected);
}

#[test]
fn end_to_end_works_with_a_searched_parameter_set() {
    // The full validation a production deployment would run: the parameter
    // search at a modest record length and security level, then
    // `SecretKey::new` (not `new_insecure`) so the concrete attack-cost
    // constraints must pass too.
    let params = search(16, 8).unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(0x9400);
    let (rows, ell) = (2 * 32 + 5, params.ell);
    let mut state = SecretKey::<MODULUS>::new(params, [0x94; 32])
        .unwrap()
        .derive(CONTEXT_SEARCHED, rows, &mut rng, |stream, _index| {
            Ok(ToeplitzFastProduct::sample(params.n().unwrap(), stream)?)
        })
        .unwrap();
    let matrix = random_vector(rows * ell, &mut rng);
    let q = random_vector(ell, &mut rng);
    let decoded = run_protocol(&mut state, &matrix, &q);
    let expected = naive_matrix_vector(&matrix, &q, rows, ell);
    assert_eq!(decoded, expected);
}

#[test]
fn matrix_mask_can_only_be_used_once() {
    let mut rng = ChaCha20Rng::seed_from_u64(5);
    let (rows, ell) = (5_usize, 8_usize);
    let mut state = derive_toeplitz(rows, ell, 6);
    let matrix = random_vector(rows * ell, &mut rng);
    let first = encrypt(&mut state, &matrix).unwrap();
    assert_eq!(first.rows(), rows);
    assert_eq!(
        encrypt(&mut state, &matrix),
        Err(ProtocolError::AlreadyEncrypted)
    );
}

#[test]
fn instance_id_domain_separates_reconstructed_keys() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x6100);
    let (rows, ell) = (5_usize, 8_usize);
    let matrix = random_vector(rows * ell, &mut rng);
    let mut first_rng = ChaCha20Rng::seed_from_u64(100);
    let mut first = SecretKey::<MODULUS>::new_insecure(test_params(ell), [6; 32])
        .unwrap()
        .derive(CONTEXT_TOEPLITZ, rows, &mut first_rng, toeplitz_block)
        .unwrap();
    let mut second_rng = ChaCha20Rng::seed_from_u64(101);
    let mut second = SecretKey::<MODULUS>::new_insecure(test_params(ell), [6; 32])
        .unwrap()
        .derive(CONTEXT_TOEPLITZ, rows, &mut second_rng, toeplitz_block)
        .unwrap();

    let first_ciphertext = encrypt(&mut first, &matrix).unwrap();
    let second_ciphertext = encrypt(&mut second, &matrix).unwrap();
    assert_ne!(first_ciphertext.values(), second_ciphertext.values());
    assert_ne!(
        first_ciphertext.instance_id(),
        second_ciphertext.instance_id()
    );
}

#[test]
fn query_randomness_is_fresh() {
    let mut rng = ChaCha20Rng::seed_from_u64(7);
    let (rows, ell) = (5_usize, 8_usize);
    let mut state = derive_toeplitz(rows, ell, 8);
    let matrix = random_vector(rows * ell, &mut rng);
    let q = random_vector(ell, &mut rng);
    let encrypted = encrypt(&mut state, &matrix).unwrap();
    let (first_query, first_key) = query(&mut state, &q).unwrap();
    let (second_query, _second_key) = query(&mut state, &q).unwrap();
    assert_ne!(first_query.values(), second_query.values());
    let first_answer = answer_matrix(&state.params(), &encrypted, &first_query);
    let decoded = decode_answer(&first_answer, &first_key);
    assert_eq!(decoded, naive_matrix_vector(&matrix, &q, rows, ell));
}

#[test]
fn query_counter_can_be_resumed_without_reusing_randomness() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x7100);
    let (rows, ell) = (5_usize, 8_usize);
    let q = random_vector(ell, &mut rng);
    let mut state = derive_toeplitz(rows, ell, 8);
    let (first_query, _) = query(&mut state, &q).unwrap();
    let (second_query, _) = query(&mut state, &q).unwrap();
    assert_eq!(state.next_query_index(), 2);

    let mut resumed = SecretKey::<MODULUS>::new_insecure(test_params(ell), [8; 32])
        .unwrap()
        .restore(
            CONTEXT_TOEPLITZ,
            state.instance_nonce(),
            2,
            rows,
            toeplitz_block,
        )
        .unwrap();
    let (resumed_query, _) = query(&mut resumed, &q).unwrap();

    assert_ne!(first_query, second_query);
    assert_ne!(resumed_query, first_query);
    assert_ne!(resumed_query, second_query);
    assert_eq!(resumed_query.query_id(), 2);
    assert_eq!(
        encrypt(&mut resumed, &vec![field().element_u32(0); rows * ell]),
        Err(ProtocolError::AlreadyEncrypted)
    );
}

#[test]
fn reserved_query_ids_support_independent_scratch() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x7110);
    let (rows, ell) = (5_usize, 8_usize);
    let q = random_vector(ell, &mut rng);
    let mut state = derive_toeplitz(rows, ell, 8);
    let mut reservations = state.reserve_query_ids(2).unwrap();
    let first_reservation = reservations.next().unwrap();
    let second_reservation = reservations.next().unwrap();
    assert!(reservations.next().is_none());
    let mut first_scratch = state.query_scratch();
    let mut second_scratch = state.query_scratch();

    let (first, second) = rayon::join(
        || query_with_scratch(&state, first_reservation, &q, &mut first_scratch).unwrap(),
        || query_with_scratch(&state, second_reservation, &q, &mut second_scratch).unwrap(),
    );

    assert_eq!(first.0.query_id(), 0);
    assert_eq!(second.0.query_id(), 1);
    assert_ne!(first.0, second.0);
    assert_eq!(state.next_query_index(), 2);
}

#[test]
fn decode_rejects_a_key_for_another_query() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x7200);
    let (rows, ell) = (5_usize, 8_usize);
    let mut state = derive_toeplitz(rows, ell, 9);
    let matrix = random_vector(rows * ell, &mut rng);
    let q = random_vector(ell, &mut rng);
    let encrypted = encrypt(&mut state, &matrix).unwrap();
    let (first_query, _) = query(&mut state, &q).unwrap();
    let (_, second_key) = query(&mut state, &q).unwrap();
    let first_answer = answer_matrix(&state.params(), &encrypted, &first_query);

    assert!(matches!(
        decode_into(&first_answer, &second_key, &mut []),
        Err(ProtocolError::QueryMismatch { .. })
    ));
}

#[test]
fn caller_owned_buffers_are_fully_overwritten() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x7300);
    let (rows, ell) = (5_usize, 8_usize);
    let mut state = derive_toeplitz(rows, ell, 10);
    let matrix = random_vector(rows * ell, &mut rng);
    let q = random_vector(ell, &mut rng);
    let encrypted = encrypt(&mut state, &matrix).unwrap();
    let (encrypted_query, decoding_key) = query(&mut state, &q).unwrap();

    // Dirty the output buffers to prove both `_into` variants overwrite
    // every slot instead of accumulating into them.
    let garbage = field().element_u32(1);
    let mut answer_values = vec![garbage; rows * state.params().blocks().unwrap()];
    answer_into(
        &state.params(),
        &encrypted,
        &encrypted_query,
        &mut answer_values,
    )
    .unwrap();

    let answer = AnswerMatrix::from_parts(
        encrypted.instance_id(),
        encrypted_query.query_id(),
        answer_values,
        rows,
        state.params().blocks().unwrap(),
    );
    let mut decoded = vec![garbage; rows];
    decode_into(&answer, &decoding_key, &mut decoded).unwrap();
    assert_eq!(decoded, naive_matrix_vector(&matrix, &q, rows, ell));
}

#[test]
fn shorter_records_are_padded() {
    let mut rng = ChaCha20Rng::seed_from_u64(9);
    let (rows, ell) = (4_usize, 5_usize);
    let mut state = SecretKey::<MODULUS>::new_insecure(test_params(ell), [10; 32])
        .unwrap()
        .derive(CONTEXT_TOEPLITZ_PADDED, rows, &mut rng, toeplitz_block)
        .unwrap();
    let matrix = random_vector(rows * ell, &mut rng);
    let q = random_vector(ell, &mut rng);
    let decoded = run_protocol(&mut state, &matrix, &q);
    let expected = naive_matrix_vector(&matrix, &q, rows, ell);
    assert_eq!(decoded, expected);
}

#[test]
fn taller_matrices_stack_extra_mask_blocks() {
    let mut rng = ChaCha20Rng::seed_from_u64(11);
    let (rows, ell) = (2 * 16 + 3, 8_usize);
    let mut state = SecretKey::<MODULUS>::new_insecure(test_params(ell), [12; 32])
        .unwrap()
        .derive(CONTEXT_TOEPLITZ_MULTI_BLOCK, rows, &mut rng, toeplitz_block)
        .unwrap();
    assert_eq!(state.mask().block_count(), 3);
    let matrix = random_vector(rows * ell, &mut rng);
    let q = random_vector(ell, &mut rng);
    let decoded = run_protocol(&mut state, &matrix, &q);
    let expected = naive_matrix_vector(&matrix, &q, rows, ell);
    assert_eq!(decoded, expected);
}

#[test]
fn wrong_instance_is_rejected() {
    let mut rng = ChaCha20Rng::seed_from_u64(13);
    let (rows, ell) = (5_usize, 8_usize);
    let mut state = derive_toeplitz(rows, ell, 14);
    let nonce = state.instance_nonce();
    let mut other = SecretKey::<MODULUS>::new_insecure(test_params(ell), [15; 32])
        .unwrap()
        .restore(CONTEXT_TOEPLITZ, nonce, 0, rows, toeplitz_block)
        .unwrap();
    let matrix = random_vector(rows * ell, &mut rng);
    let q = random_vector(ell, &mut rng);
    let encrypted = encrypt(&mut state, &matrix).unwrap();
    let (query_wrong, _key_wrong) = query(&mut other, &q).unwrap();
    assert!(matches!(
        answer_into(&state.params(), &encrypted, &query_wrong, &mut []),
        Err(ProtocolError::InstanceMismatch { .. })
    ));
}

#[test]
fn reconstruction_context_changes_isolate_instances() {
    // Under the same root key and the same persisted nonce, changing the
    // mask context identifier, the row count, or any parameter must each
    // yield an independent instance; restoring the exact context must
    // reproduce the original state bit for bit; and artifacts must not mix
    // across instances.
    let mut rng = ChaCha20Rng::seed_from_u64(0x9000);
    let (rows, ell) = (5_usize, 8_usize);
    let q = random_vector(ell, &mut rng);
    let nonce = 0x1234_5678_9abc_def0_u128;

    let restore_state = |context: MaskContextId, rows: usize, ell: usize| {
        SecretKey::<MODULUS>::new_insecure(test_params(ell), [0x90; 32])
            .unwrap()
            .restore(context, nonce, 0, rows, toeplitz_block)
            .unwrap()
    };

    let mut baseline = restore_state(CONTEXT_TOEPLITZ, rows, ell);
    let (baseline_query, _baseline_key) = query(&mut baseline, &q).unwrap();

    // Exact restore: the same context, rows, params, key, and nonce
    // reproduce the instance and its query artifacts bit for bit.
    let mut exact = restore_state(CONTEXT_TOEPLITZ, rows, ell);
    assert_eq!(baseline.instance_id(), exact.instance_id());
    let (exact_query, _exact_key) = query(&mut exact, &q).unwrap();
    assert_eq!(baseline_query, exact_query);

    // Each changed context field derives an independent instance.
    let other_context = restore_state(CONTEXT_TOEPLITZ_ALT, rows, ell);
    let other_rows = restore_state(CONTEXT_TOEPLITZ, rows + 1, ell);
    let other_params = restore_state(CONTEXT_TOEPLITZ, rows, ell - 3);
    assert_ne!(baseline.instance_id(), other_context.instance_id());
    assert_ne!(baseline.instance_id(), other_rows.instance_id());
    assert_ne!(baseline.instance_id(), other_params.instance_id());

    // Artifact mixing is rejected: a query of the baseline instance does
    // not answer against a matrix carrying the other-context instance id.
    // (Restored states cannot encrypt, so the foreign matrix is rebuilt
    // from the other instance's public identifier.)
    let zero = field().element_u32(0);
    let n = baseline.params().n().unwrap();
    let other_encrypted =
        EncryptedMatrix::from_parts(other_context.instance_id(), rows, n, vec![zero; rows * n])
            .unwrap();
    assert_ne!(baseline.instance_id(), other_encrypted.instance_id());
    assert!(matches!(
        answer_into(
            &baseline.params(),
            &other_encrypted,
            &baseline_query,
            &mut []
        ),
        Err(ProtocolError::InstanceMismatch { .. })
    ));
}

#[test]
fn a_failed_query_still_consumes_its_reserved_identifier() {
    // `query` reserves the identifier before the mask evaluation, so a
    // mask failure after that point must consume the identifier: retrying
    // continues from the next index instead of reusing the randomness.
    let mut rng = ChaCha20Rng::seed_from_u64(0x9600);
    let (rows, ell) = (5_usize, 8_usize);
    let mut state = SecretKey::<MODULUS>::new_insecure(test_params(ell), [0x96; 32])
        .unwrap()
        .derive(CONTEXT_FAILING, rows, &mut rng, |stream, _index| {
            Ok(FailingApplyMask {
                inner: ToeplitzFastProduct::sample(16, stream)?,
            })
        })
        .unwrap();
    let q = random_vector(ell, &mut rng);
    assert_eq!(
        query(&mut state, &q),
        Err(ProtocolError::Mask(TdmError::ZeroDimension(
            "deliberately failing test mask"
        )))
    );
    assert_eq!(
        state.next_query_index(),
        1,
        "the post-reservation failure must consume the identifier"
    );
    let reservation = state.reserve_query_ids(1).unwrap().next().unwrap();
    assert_eq!(reservation.query_id(), 1, "the next id must not be reused");
}

// Independent naive reimplementation of the encryption pipeline, in plain
// u64 modular arithmetic. The dense mask is taken as input because the
// caller needs the same materialization for its own naive query path.
fn naive_encrypt_gathered(
    state: &DerivedState<MODULUS, ToeplitzFastProduct<MODULUS>>,
    dense_mask: &DenseMatrix<MODULUS>,
    matrix: &[FieldElement<MODULUS>],
    rows: usize,
    ell: usize,
) -> EncryptedMatrix<MODULUS> {
    let params = state.params();
    let code_dim = params.k;
    let width = params.n().unwrap();
    let multiplier: Vec<u32> = state
        .code()
        .multiplier()
        .iter()
        .map(|&element| element.value())
        .collect();
    let permutation: Vec<usize> = state.permutation_indices().to_vec();
    let modulus_u64 = u64::from(MODULUS);
    let mut encrypted_values = Vec::new();
    for row in 0..rows {
        let record: Vec<u32> = matrix[row * ell..(row + 1) * ell]
            .iter()
            .map(|&element| element.value())
            .collect();
        // Left half: -M_g m; systematic half: m padded with zeros.
        let mut encoded = Vec::with_capacity(width);
        for i in 0..code_dim {
            let mut accumulator = 0_u64;
            for j in 0..code_dim {
                accumulator +=
                    u64::from(multiplier[(i + code_dim - j) % code_dim]) * u64::from(record[j]);
            }
            accumulator %= modulus_u64;
            encoded.push((modulus_u64 - accumulator) % modulus_u64);
        }
        for value in record
            .iter()
            .copied()
            .chain(std::iter::repeat(0_u32))
            .take(code_dim)
        {
            encoded.push(u64::from(value));
        }
        for (slot, &masked) in encoded
            .iter_mut()
            .zip(&dense_mask.values()[row * width..(row + 1) * width])
        {
            *slot = (*slot + u64::from(masked.value())) % modulus_u64;
        }
        // Permute columns through the gather.
        let permuted: Vec<u64> = permutation.iter().map(|&index| encoded[index]).collect();
        encrypted_values.extend(
            permuted
                .iter()
                .map(|&value| field().element_u32(value as u32)),
        );
    }
    EncryptedMatrix::from_parts(state.instance_id(), rows, width, encrypted_values).unwrap()
}

#[test]
fn permutation_direction_is_pinned_by_naive_oracle() {
    let mut rng = ChaCha20Rng::seed_from_u64(17);
    let (rows, ell) = (5_usize, 8_usize);
    let mut state = derive_toeplitz(rows, ell, 18);
    let params = state.params();
    let code_dim = params.k;
    let width = params.n().unwrap();
    let block_len = params.block_size();
    let matrix = random_vector(rows * ell, &mut rng);
    let query_vector = random_vector(ell, &mut rng);
    let dense_mask = state.mask().materialize().unwrap();
    let permutation: Vec<usize> = state.permutation_indices().to_vec();
    let encrypted_naive = naive_encrypt_gathered(&state, &dense_mask, &matrix, rows, ell);

    // Naive query, correct scatter direction.
    let (_encrypted_query, decoding_key) = query(&mut state, &query_vector).unwrap();
    // Replicate the protocol's exact derivation of the query codeword
    // stream: the bound instance PRF over the same key, context, row count,
    // and nonce that the derived state used.
    let mut query_rng = SecretKey::<MODULUS>::new_insecure(test_params(ell), [18; 32])
        .unwrap()
        .bound_instance_prf(CONTEXT_TOEPLITZ, rows, state.instance_nonce())
        .stream(purpose::QUERY_CODEWORD, 0)
        .unwrap();
    let codeword: Vec<u32> = {
        let mut raw = vec![field().element_u32(0); width];
        let mut scratch = state.code().scratch();
        state
            .code()
            .sample_codeword(&mut query_rng, &mut raw, &mut scratch)
            .unwrap();
        raw.iter().map(|&element| element.value()).collect()
    };
    let mut q_tilde = codeword;
    for (slot, &coefficient) in q_tilde[code_dim..].iter_mut().zip(&query_vector) {
        *slot = (*slot + coefficient.value()) % MODULUS;
    }
    let alphas: Vec<u32> = decoding_key
        .p_prime()
        .iter()
        .map(|&p| p.inv().unwrap().value())
        .collect();
    let modulus_u64 = u64::from(MODULUS);

    let naive_answer = |gather: bool| {
        let mut q_pi = vec![0_u32; width];
        if gather {
            for (position, &index) in permutation.iter().enumerate() {
                q_pi[position] = q_tilde[index];
            }
        } else {
            for (position, &index) in permutation.iter().enumerate() {
                q_pi[index] = q_tilde[position];
            }
        }
        let mut q_hat = q_pi.clone();
        for (block, &alpha) in alphas.iter().enumerate() {
            for slot in &mut q_hat[block * block_len..(block + 1) * block_len] {
                *slot = (u64::from(*slot) * u64::from(alpha) % modulus_u64) as u32;
            }
        }
        let q_hat_elements: Vec<FieldElement<MODULUS>> = q_hat
            .iter()
            .map(|&value| field().element_u32(value))
            .collect();
        let answer = answer_matrix(
            &params,
            &encrypted_naive,
            &EncryptedQuery::from_parts(state.instance_id(), 0, q_hat_elements),
        );
        // Mask share r' = R q_tilde on the UNPERMUTED query.
        let q_tilde_elements: Vec<FieldElement<MODULUS>> = q_tilde
            .iter()
            .map(|&value| field().element_u32(value))
            .collect();
        let mut r_prime = vec![field().element_u32(0); rows];
        dense_mask.apply(&q_tilde_elements, &mut r_prime).unwrap();
        let key = DecodingKey::from_parts(
            state.instance_id(),
            0,
            decoding_key.p_prime().to_vec(),
            r_prime,
        );
        decode_answer(&answer, &key)
    };

    let correct = naive_answer(true); // gather, matching the matrix columns
    let expected = naive_matrix_vector(&matrix, &query_vector, rows, ell);
    assert_eq!(correct, expected, "scatter direction must decode correctly");

    let flipped = naive_answer(false); // scatter, the wrong direction
    assert_ne!(
        flipped, expected,
        "the gather direction must break correctness"
    );
}

#[test]
fn length_errors_are_rejected_before_mutation() {
    let mut rng = ChaCha20Rng::seed_from_u64(19);
    let (rows, ell) = (5_usize, 8_usize);
    let mut state = derive_toeplitz(rows, ell, 20);
    let n = state.params().n().unwrap();
    let q = random_vector(ell, &mut rng);

    let short_matrix = random_vector(rows * ell - 1, &mut rng);
    assert!(matches!(
        encrypt(&mut state, &short_matrix),
        Err(ProtocolError::LengthMismatch { .. })
    ));

    let short_query = random_vector(ell - 1, &mut rng);
    assert!(matches!(
        query(&mut state, &short_query),
        Err(ProtocolError::LengthMismatch { .. })
    ));

    let encrypted = encrypt(&mut state, &random_vector(rows * ell, &mut rng)).unwrap();
    let (encrypted_query, decoding_key) = query(&mut state, &q).unwrap();
    let answer_matrix = answer_matrix(&state.params(), &encrypted, &encrypted_query);

    let truncated_query = EncryptedQuery::from_parts(
        encrypted_query.instance_id(),
        encrypted_query.query_id(),
        encrypted_query.values()[..n - 1].to_vec(),
    );
    assert!(matches!(
        answer_into(&state.params(), &encrypted, &truncated_query, &mut []),
        Err(ProtocolError::LengthMismatch { .. })
    ));

    let short_key = DecodingKey::from_parts(
        decoding_key.instance_id(),
        decoding_key.query_id(),
        decoding_key.p_prime()[..state.params().blocks().unwrap() - 1].to_vec(),
        decoding_key.r_prime().to_vec(),
    );
    let sentinel = AnswerMatrix::from_parts(
        answer_matrix.instance_id(),
        answer_matrix.query_id(),
        answer_matrix.values().to_vec(),
        rows,
        state.params().blocks().unwrap(),
    );
    assert!(matches!(
        decode_into(&sentinel, &short_key, &mut []),
        Err(ProtocolError::LengthMismatch { .. })
    ));
}

#[test]
fn malformed_answer_is_validated_before_output_allocation() {
    let zero = field().element_u32(0);
    let answer = AnswerMatrix::from_parts(1, 0, Vec::new(), usize::MAX, 2);
    let key = DecodingKey::from_parts(1, 0, vec![zero; 2], Vec::new());
    assert!(matches!(
        decode_into(&answer, &key, &mut []),
        Err(ProtocolError::LengthMismatch { .. })
    ));
}

#[test]
fn derive_rejects_a_mask_with_the_wrong_protocol_width() {
    let key = SecretKey::<MODULUS>::new_insecure(test_params(8), [21; 32]).unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(21);
    let result = key.derive(CONTEXT_TOEPLITZ, 1, &mut rng, |stream, _index| {
        Ok(ToeplitzFastProduct::sample(8, stream)?)
    });
    assert!(matches!(
        result,
        Err(ProtocolError::LengthMismatch {
            name: "mask columns",
            expected: 16,
            actual: 8,
        })
    ));
}

#[test]
fn binary_field_is_rejected() {
    assert!(matches!(
        SecretKey::<2>::new_insecure(test_params(8), [22; 32]),
        Err(ProtocolError::UnsupportedField { modulus: 2 })
    ));
}

#[test]
fn answer_rejects_malformed_block_dimensions() {
    let zero = field().element_u32(0);
    let matrix = EncryptedMatrix::from_parts(1, 1, 16, vec![zero; 16]).unwrap();
    let encrypted_query = EncryptedQuery::from_parts(1, 0, vec![zero; 16]);

    for b in [0, 3] {
        let malformed = EmvpParams {
            k: 8,
            ell: 8,
            b,
            lambda: 7,
        };
        assert!(matches!(
            answer_into(&malformed, &matrix, &encrypted_query, &mut []),
            Err(ProtocolError::Params(_))
        ));
    }
}

#[test]
fn decoding_key_debug_is_redacted() {
    let field = field();
    let key = DecodingKey::from_parts(
        1,
        0,
        vec![field.element_u32(123_456)],
        vec![field.element_u32(654_321)],
    );
    let rendered = format!("{key:?}");
    assert!(!rendered.contains("p_prime"));
    assert!(!rendered.contains("r_prime"));
    assert!(!rendered.contains("FieldElement"));
    assert!(rendered.contains("blocks: 1"));
    assert!(rendered.contains("rows: 1"));
}

fn pool(threads: usize) -> ThreadPool {
    ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap()
}

#[test]
fn parallel_encrypt_matches_the_serial_row_loop_and_naive_oracle() {
    // rows * n = 2048 * 16 = 32768 estimated multiplications clear the
    // crate's parallel-work threshold and 2048 >= 2 * 4 rows satisfies the
    // thread guard, so the four-thread run forces the parallel branch while
    // the single-thread run takes the serial reference path. Derivation is
    // deterministic in the key, so both runs hold identical long-term
    // secrets.
    let (rows, ell) = (2048_usize, 8_usize);
    let mut rng = ChaCha20Rng::seed_from_u64(0x8100);
    let matrix = random_vector(rows * ell, &mut rng);

    let parallel =
        pool(4).install(|| encrypt(&mut derive_toeplitz(rows, ell, 0x81), &matrix).unwrap());
    let serial =
        pool(1).install(|| encrypt(&mut derive_toeplitz(rows, ell, 0x81), &matrix).unwrap());
    assert_eq!(parallel, serial);

    let state = derive_toeplitz(rows, ell, 0x81);
    let dense_mask = state.mask().materialize().unwrap();
    let expected = naive_encrypt_gathered(&state, &dense_mask, &matrix, rows, ell);
    assert_eq!(parallel, expected);
}

#[test]
fn answer_batch_rejects_malformed_batches_before_answering() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x8200);
    let (rows, ell) = (5_usize, 8_usize);
    let mut state = derive_toeplitz(rows, ell, 0x82);
    let matrix = random_vector(rows * ell, &mut rng);
    let q = random_vector(ell, &mut rng);
    let encrypted = encrypt(&mut state, &matrix).unwrap();
    let (good_query, _) = query(&mut state, &q).unwrap();

    assert!(matches!(
        answer_batch(&state.params(), &encrypted, &[]),
        Err(ProtocolError::LengthMismatch {
            name: "queries",
            expected: 1,
            actual: 0,
        })
    ));

    let n = state.params().n().unwrap();
    let truncated = EncryptedQuery::from_parts(
        encrypted.instance_id(),
        good_query.query_id(),
        vec![field().element_u32(0); n - 1],
    );
    assert!(matches!(
        answer_batch(
            &state.params(),
            &encrypted,
            &[good_query.clone(), truncated]
        ),
        Err(ProtocolError::LengthMismatch { .. })
    ));

    let mut other = SecretKey::<MODULUS>::new_insecure(test_params(ell), [0x83; 32])
        .unwrap()
        .restore(
            CONTEXT_TOEPLITZ,
            state.instance_nonce(),
            0,
            rows,
            toeplitz_block,
        )
        .unwrap();
    let (foreign_query, _) = query(&mut other, &q).unwrap();
    assert!(matches!(
        answer_batch(&state.params(), &encrypted, &[good_query, foreign_query]),
        Err(ProtocolError::InstanceMismatch { .. })
    ));
}

#[test]
fn parallel_derive_matches_the_serial_block_loop() {
    // rows = 2 * n + 1 stacks three mask blocks and clears the two-block
    // parallel guard, so the four-thread run constructs blocks across
    // rayon workers while the single-thread run takes the serial loop.
    // Derivation is deterministic in the key and nonce, so both runs must
    // hold identical long-term secrets and produce identical artifacts.
    let (rows, ell) = (2 * 16 + 1, 8_usize);
    let mut rng = ChaCha20Rng::seed_from_u64(0x8600);
    let matrix = random_vector(rows * ell, &mut rng);
    let q = random_vector(ell, &mut rng);

    let artifacts_with = |threads: usize| {
        pool(threads).install(|| {
            let mut state = derive_toeplitz(rows, ell, 0x86);
            let instance_id = state.instance_id();
            let encrypted = encrypt(&mut state, &matrix).unwrap();
            let (encrypted_query, decoding_key) = query(&mut state, &q).unwrap();
            (instance_id, encrypted, encrypted_query, decoding_key)
        })
    };
    let (parallel_id, parallel_encrypted, parallel_query, parallel_key) = artifacts_with(4);
    let (serial_id, serial_encrypted, serial_query, serial_key) = artifacts_with(1);
    assert_eq!(parallel_id, serial_id);
    assert_eq!(parallel_encrypted, serial_encrypted);
    assert_eq!(parallel_query, serial_query);
    assert_eq!(parallel_key, serial_key);
}

#[test]
fn parallel_answer_batch_matches_the_serial_loop() {
    // 16 queries * 128 rows * n = 16 = 32768 estimated multiplications
    // clear the crate's parallel-work threshold and the 2048-row grid
    // satisfies the 2 * 4-threads guard, so the four-thread run forces the
    // parallel branch while the single-thread run takes the serial path.
    let (rows, ell, batch) = (128_usize, 8_usize, 16_usize);
    let params = test_params(ell);
    let mut rng = ChaCha20Rng::seed_from_u64(0x8400);
    let matrix = random_vector(rows * ell, &mut rng);
    let q = random_vector(ell, &mut rng);
    let zero = field().element_u32(0);
    let mut state = derive_toeplitz(rows, ell, 0x85);
    let encrypted = encrypt(&mut state, &matrix).unwrap();
    let blocks = params.blocks().unwrap();

    let mut queries = Vec::new();
    let mut expected = Vec::new();
    for _ in 0..batch {
        let (encrypted_query, _key) = query(&mut state, &q).unwrap();
        let mut answer_values = vec![zero; rows * blocks];
        answer_into(&params, &encrypted, &encrypted_query, &mut answer_values).unwrap();
        queries.push(encrypted_query);
        expected.extend_from_slice(&answer_values);
    }

    let flatten = |answers: Vec<AnswerMatrix<MODULUS>>| {
        answers
            .iter()
            .flat_map(|answer| answer.values().iter().copied())
            .collect::<Vec<_>>()
    };
    let parallel =
        flatten(pool(4).install(|| answer_batch(&params, &encrypted, &queries).unwrap()));
    let serial = flatten(pool(1).install(|| answer_batch(&params, &encrypted, &queries).unwrap()));
    assert_eq!(parallel, expected);
    assert_eq!(parallel, serial);
}

#[test]
fn parallel_decode_matches_the_serial_row_loop() {
    // 4096 rows * s = 8 = 32768 multiplications clear the crate's
    // parallel-work threshold and 4096 >= 2 * 4 rows satisfies the thread
    // guard, so the four-thread run forces the parallel branch while the
    // single-thread run takes the serial reference path. Both runs decode
    // the same answer with the same key.
    let (rows, ell) = (4096_usize, 8_usize);
    let mut rng = ChaCha20Rng::seed_from_u64(0x8700);
    let matrix = random_vector(rows * ell, &mut rng);
    let q = random_vector(ell, &mut rng);
    let mut state = derive_toeplitz(rows, ell, 0x87);
    let encrypted = encrypt(&mut state, &matrix).unwrap();
    let (encrypted_query, decoding_key) = query(&mut state, &q).unwrap();
    let answer = answer_matrix(&state.params(), &encrypted, &encrypted_query);

    let decode_with = |threads: usize| {
        let mut output = vec![field().element_u32(0); rows];
        pool(threads)
            .install(|| decode_into(&answer, &decoding_key, &mut output))
            .unwrap();
        output
    };
    let parallel = decode_with(4);
    let serial = decode_with(1);
    assert_eq!(parallel, serial);
    assert_eq!(parallel, naive_matrix_vector(&matrix, &q, rows, ell));
}

#[test]
fn query_batch_rejects_malformed_batches_without_consuming_ids() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x8800);
    let (rows, ell) = (5_usize, 8_usize);
    let mut state = derive_toeplitz(rows, ell, 0x88);

    assert!(matches!(
        query_batch(&mut state, &[]),
        Err(ProtocolError::LengthMismatch {
            name: "queries",
            expected: 1,
            actual: 0,
        })
    ));

    let short = random_vector(ell - 1, &mut rng);
    let good = random_vector(ell, &mut rng);
    assert_eq!(
        query_batch(&mut state, &[&good, &short]),
        Err(ProtocolError::LengthMismatch {
            name: "query vector",
            expected: ell,
            actual: ell - 1,
        })
    );
    assert_eq!(
        state.next_query_index(),
        0,
        "no identifiers may be consumed"
    );

    let batch = query_batch(&mut state, &[&good, &good]).unwrap();
    assert_eq!(batch.len(), 2);
    assert_eq!(state.next_query_index(), 2);
    assert_eq!(batch[0].0.query_id(), 0);
    assert_eq!(batch[1].0.query_id(), 1);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(16))]

    #[test]
    fn query_batch_matches_sequential_single_queries(
        ell in 1_usize..=8_usize,
        rows in 1_usize..=24_usize,
        batch in 1_usize..=4_usize,
    ) {
        let params = test_params(ell);
        let mut rng = ChaCha20Rng::seed_from_u64(0x8900);
        let q = random_vector(ell, &mut rng);
        let mut state = derive_toeplitz(rows, ell, 0x8a);

        let mut serial = Vec::new();
        for _ in 0..batch {
            serial.push(query(&mut state, &q).unwrap());
        }
        assert_eq!(state.next_query_index(), batch as u64);

        // A fresh state over the same key, context, and nonce assigns the
        // same identifier range, so batched artifacts must equal the serial
        // ones bit for bit.
        let mut batched_state =
            SecretKey::<MODULUS>::new_insecure(params, [0x8a; 32])
                .unwrap()
                .restore(
                    CONTEXT_TOEPLITZ,
                    state.instance_nonce(),
                    0,
                    rows,
                    toeplitz_block,
                )
                .unwrap();
        let queries: Vec<&[FieldElement<MODULUS>]> =
            (0..batch).map(|_| q.as_slice()).collect();
        let batched = query_batch(&mut batched_state, &queries).unwrap();

        prop_assert_eq!(batched.len(), batch);
        for (batched_artifacts, serial_artifacts) in batched.into_iter().zip(serial) {
            prop_assert_eq!(batched_artifacts.0, serial_artifacts.0);
            prop_assert_eq!(batched_artifacts.1, serial_artifacts.1);
        }
    }

    #[test]
    fn answer_batch_matches_sequential_single_query_answers(
        ell in 1_usize..=8_usize,
        rows in 1_usize..=24_usize,
        batch in 1_usize..=4_usize,
    ) {
        let params = test_params(ell);
        let mut rng = ChaCha20Rng::seed_from_u64(0x8300);
        let mut state = derive_toeplitz(rows, ell, 0x84);
        let matrix = random_vector(rows * ell, &mut rng);
        let q = random_vector(ell, &mut rng);
        let encrypted = encrypt(&mut state, &matrix).unwrap();
        let blocks = params.blocks().unwrap();
        let zero = field().element_u32(0);

        let mut queries = Vec::new();
        let mut expected = Vec::new();
        for _ in 0..batch {
            let (encrypted_query, _key) = query(&mut state, &q).unwrap();
            let mut answer_values = vec![zero; rows * blocks];
            answer_into(&params, &encrypted, &encrypted_query, &mut answer_values).unwrap();
            queries.push(encrypted_query);
            expected.extend_from_slice(&answer_values);
        }

        let batched = answer_batch(&params, &encrypted, &queries).unwrap();
        prop_assert_eq!(batched.len(), batch);
        for (answer, answer_values) in batched.iter().zip(expected.chunks(rows * blocks)) {
            prop_assert_eq!(answer.values(), answer_values);
            prop_assert_eq!(answer.rows(), rows);
            prop_assert_eq!(answer.blocks(), blocks);
            prop_assert_eq!(answer.instance_id(), encrypted.instance_id());
        }
    }
}
