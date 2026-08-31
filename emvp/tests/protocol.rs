#![expect(
    clippy::unwrap_used,
    reason = "fixed test fixtures establish that protocol steps must succeed"
)]

use emvp::{
    AnswerMatrix, DecodingKey, DerivedState, EmvpParams, EncryptedMatrix, EncryptedQuery, Prf,
    ProtocolError, SecretKey, TdmMask, answer, answer_into, decode, decode_into, encrypt, purpose,
    query, query_with_scratch,
};
use prime_field_layer::{FieldElement, PrimeField};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use trapdoor_matrices::{DenseMatrix, RaaWeightedProduct, ToeplitzFastProduct};

// NTT-friendly prime: 998_244_353 - 1 is divisible by 2^23.
const MODULUS: u32 = 998_244_353;

// Tiny deliberately-insecure research parameters: k = 8, ell = 8, b = 2,
// so n = 16 and s = 8. The protocol layer does not re-validate params.
const fn test_params(ell: usize) -> EmvpParams {
    EmvpParams {
        k: 8,
        ell,
        b: 2,
        lambda: 7,
    }
}

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

fn derive_toeplitz(
    rows: usize,
    ell: usize,
    key: u8,
) -> DerivedState<MODULUS, ToeplitzFastProduct<MODULUS>> {
    let mut rng = ChaCha20Rng::seed_from_u64(0xd000 + u64::from(key));
    SecretKey::<MODULUS>::new_insecure(test_params(ell), [key; 32])
        .unwrap()
        .derive(rows, &mut rng, toeplitz_block)
        .unwrap()
}

fn run_protocol<M: TdmMask<MODULUS>>(
    state: &mut DerivedState<MODULUS, M>,
    matrix: &[FieldElement<MODULUS>],
    q: &[FieldElement<MODULUS>],
) -> Vec<FieldElement<MODULUS>> {
    let encrypted = encrypt(state, matrix).unwrap();
    let (encrypted_query, decoding_key) = query(state, q).unwrap();
    let answer_matrix = answer(&state.params(), &encrypted, &encrypted_query).unwrap();
    decode(&answer_matrix, &decoding_key).unwrap()
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
        .derive(rows, &mut rng, raa_block)
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
        .derive(rows, &mut first_rng, toeplitz_block)
        .unwrap();
    let mut second_rng = ChaCha20Rng::seed_from_u64(101);
    let mut second = SecretKey::<MODULUS>::new_insecure(test_params(ell), [6; 32])
        .unwrap()
        .derive(rows, &mut second_rng, toeplitz_block)
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
    let first_answer = answer(&state.params(), &encrypted, &first_query).unwrap();
    let decoded = decode(&first_answer, &first_key).unwrap();
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
        .restore(state.instance_nonce(), 2, rows, toeplitz_block)
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
    let first_answer = answer(&state.params(), &encrypted, &first_query).unwrap();

    assert!(matches!(
        decode(&first_answer, &second_key),
        Err(ProtocolError::QueryMismatch { .. })
    ));
}

#[test]
fn caller_owned_answer_and_decode_buffers_match_allocating_wrappers() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x7300);
    let (rows, ell) = (5_usize, 8_usize);
    let mut state = derive_toeplitz(rows, ell, 10);
    let matrix = random_vector(rows * ell, &mut rng);
    let q = random_vector(ell, &mut rng);
    let encrypted = encrypt(&mut state, &matrix).unwrap();
    let (encrypted_query, decoding_key) = query(&mut state, &q).unwrap();
    let expected_answer = answer(&state.params(), &encrypted, &encrypted_query).unwrap();

    let zero = field().element_u32(0);
    let mut answer_values = vec![zero; expected_answer.values().len()];
    answer_into(
        &state.params(),
        &encrypted,
        &encrypted_query,
        &mut answer_values,
    )
    .unwrap();
    assert_eq!(answer_values, expected_answer.values());

    let mut decoded = vec![zero; rows];
    decode_into(&expected_answer, &decoding_key, &mut decoded).unwrap();
    assert_eq!(decoded, naive_matrix_vector(&matrix, &q, rows, ell));
}

#[test]
fn shorter_records_are_padded() {
    let mut rng = ChaCha20Rng::seed_from_u64(9);
    let (rows, ell) = (4_usize, 5_usize);
    let mut state = derive_toeplitz(rows, ell, 10);
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
    let mut state = derive_toeplitz(rows, ell, 12);
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
        .restore(nonce, 0, rows, toeplitz_block)
        .unwrap();
    let matrix = random_vector(rows * ell, &mut rng);
    let q = random_vector(ell, &mut rng);
    let encrypted = encrypt(&mut state, &matrix).unwrap();
    let (query_wrong, _key_wrong) = query(&mut other, &q).unwrap();
    assert!(matches!(
        answer(&state.params(), &encrypted, &query_wrong),
        Err(ProtocolError::InstanceMismatch { .. })
    ));
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
    let mut query_rng = Prf::new([18; 32])
        .derive_context(state.instance_nonce())
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
        let answer_matrix = answer(
            &params,
            &encrypted_naive,
            &EncryptedQuery::from_parts(state.instance_id(), 0, q_hat_elements),
        )
        .unwrap();
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
        decode(&answer_matrix, &key).unwrap()
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
    let answer_matrix = answer(&state.params(), &encrypted, &encrypted_query).unwrap();

    let truncated_query = EncryptedQuery::from_parts(
        encrypted_query.instance_id(),
        encrypted_query.query_id(),
        encrypted_query.values()[..n - 1].to_vec(),
    );
    assert!(matches!(
        answer(&state.params(), &encrypted, &truncated_query),
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
        decode(&sentinel, &short_key),
        Err(ProtocolError::LengthMismatch { .. })
    ));
}

#[test]
fn malformed_answer_is_validated_before_output_allocation() {
    let zero = field().element_u32(0);
    let answer = AnswerMatrix::from_parts(1, 0, Vec::new(), usize::MAX, 2);
    let key = DecodingKey::from_parts(1, 0, vec![zero; 2], Vec::new());
    assert!(matches!(
        decode(&answer, &key),
        Err(ProtocolError::LengthMismatch { .. })
    ));
}

#[test]
fn derive_rejects_a_mask_with_the_wrong_protocol_width() {
    let key = SecretKey::<MODULUS>::new_insecure(test_params(8), [21; 32]).unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(21);
    let result = key.derive(1, &mut rng, |stream, _index| {
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
            answer(&malformed, &matrix, &encrypted_query),
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
