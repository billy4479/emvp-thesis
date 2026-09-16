//! Tests for the borrowed protocol view types (`emvp::view`).
//!
//! The views must behave exactly like the owned protocol artifacts they
//! mirror: constructors reject the same malformed shapes and lengths, the
//! `From`/`as_ref` conversions preserve every accessor value, and the pure
//! consumers that were genericized over the view traits (`decode_into` and
//! the shape validators) produce identical results over borrowed and owned
//! answers.

#![expect(
    clippy::unwrap_used,
    reason = "fixed test fixtures establish that protocol steps must succeed"
)]

use emvp::{
    AnswerMatrix, AnswerRef, DecodingKey, DerivedState, EmvpParams, EncryptedMatrixRef,
    EncryptedQueryRef, MaskContextId, ProtocolError, SecretKey, answer_into, decode_into, encrypt,
    query,
};
use prime_field_layer::{FieldElement, PrimeField};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use trapdoor_matrices::ToeplitzFastProduct;

// NTT-friendly prime, matching emvp/tests/protocol.rs.
const MODULUS: u32 = 1_073_479_681;

// Tiny deliberately-insecure research parameters: k = 8, ell = 8, b = 2,
// so n = 16 and s = 8.
const fn test_params(ell: usize) -> EmvpParams {
    EmvpParams {
        k: 8,
        ell,
        b: 2,
        lambda: 7,
    }
}

const CONTEXT_VIEW: MaskContextId = MaskContextId::from_u64(0x0600);

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

fn derive_toeplitz(rows: usize, ell: usize) -> DerivedState<MODULUS, ToeplitzFastProduct<MODULUS>> {
    let mut rng = ChaCha20Rng::seed_from_u64(0xe000);
    SecretKey::<MODULUS>::new_insecure(test_params(ell), [0xe0; 32])
        .unwrap()
        .derive(CONTEXT_VIEW, rows, &mut rng, toeplitz_block)
        .unwrap()
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
fn encrypted_matrix_ref_rejects_malformed_parts() {
    let values = random_vector(6, &mut ChaCha20Rng::seed_from_u64(1));
    // A well-formed view constructs and exposes its accessors.
    let view = EncryptedMatrixRef::new(1, 2, 3, &values).unwrap();
    assert_eq!(view.rows(), 2);
    assert_eq!(view.columns(), 3);
    assert_eq!(view.instance_id(), 1);
    assert_eq!(view.values(), &values);

    assert!(matches!(
        EncryptedMatrixRef::new(1, 0, 3, &values),
        Err(ProtocolError::LengthMismatch {
            name: "matrix rows",
            ..
        })
    ));
    assert!(matches!(
        EncryptedMatrixRef::new(1, 2, 0, &values),
        Err(ProtocolError::LengthMismatch {
            name: "matrix columns",
            ..
        })
    ));
    assert!(matches!(
        EncryptedMatrixRef::new(1, 2, 2, &values),
        Err(ProtocolError::LengthMismatch {
            name: "matrix values",
            ..
        })
    ));
    assert!(matches!(
        EncryptedMatrixRef::new(1, usize::MAX, 2, &values),
        Err(ProtocolError::DimensionOverflow)
    ));
}

#[test]
fn answer_ref_rejects_malformed_parts() {
    let values = random_vector(6, &mut ChaCha20Rng::seed_from_u64(2));
    // A well-formed view constructs and exposes its accessors.
    let view = AnswerRef::new(1, 7, &values, 2, 3).unwrap();
    assert_eq!(view.rows(), 2);
    assert_eq!(view.blocks(), 3);
    assert_eq!(view.instance_id(), 1);
    assert_eq!(view.query_id(), 7);
    assert_eq!(view.values(), &values);

    assert!(matches!(
        AnswerRef::new(1, 7, &values, 0, 3),
        Err(ProtocolError::LengthMismatch {
            name: "answer rows",
            ..
        })
    ));
    assert!(matches!(
        AnswerRef::new(1, 7, &values, 2, 0),
        Err(ProtocolError::LengthMismatch {
            name: "answer blocks",
            ..
        })
    ));
    assert!(matches!(
        AnswerRef::new(1, 7, &values, 3, 3),
        Err(ProtocolError::LengthMismatch {
            name: "answer matrix",
            ..
        })
    ));
    assert!(matches!(
        AnswerRef::new(1, 7, &values, usize::MAX, 3),
        Err(ProtocolError::DimensionOverflow)
    ));
}

#[test]
fn owned_types_round_trip_through_views() {
    let mut rng = ChaCha20Rng::seed_from_u64(3);
    let (rows, ell) = (5_usize, 8_usize);
    let mut state = derive_toeplitz(rows, ell);
    let matrix = random_vector(rows * ell, &mut rng);
    let q = random_vector(ell, &mut rng);
    let encrypted = encrypt(&mut state, &matrix).unwrap();
    let (encrypted_query, decoding_key) = query(&mut state, &q).unwrap();

    let blocks = state.params().blocks().unwrap();
    let zero = field().element_u32(0);
    let mut answer_values = vec![zero; rows * blocks];
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
        blocks,
    );

    // `as_ref` and `From` must agree and preserve every accessor value of
    // the owned artifacts.
    let matrix_ref = encrypted.as_ref();
    let matrix_from: EncryptedMatrixRef<'_, MODULUS> = (&encrypted).into();
    assert_eq!(matrix_ref, matrix_from);
    assert_eq!(matrix_ref.rows(), encrypted.rows());
    assert_eq!(matrix_ref.columns(), encrypted.columns());
    assert_eq!(matrix_ref.instance_id(), encrypted.instance_id());
    assert_eq!(matrix_ref.values(), encrypted.values());

    let query_ref: EncryptedQueryRef<'_, MODULUS> = encrypted_query.as_ref();
    let query_from: EncryptedQueryRef<'_, MODULUS> = (&encrypted_query).into();
    assert_eq!(query_ref, query_from);
    assert_eq!(query_ref.instance_id(), encrypted_query.instance_id());
    assert_eq!(query_ref.query_id(), encrypted_query.query_id());
    assert_eq!(query_ref.values(), encrypted_query.values());
    // The query view constructor accepts the coordinates unchanged.
    assert_eq!(
        EncryptedQueryRef::new(
            encrypted_query.instance_id(),
            encrypted_query.query_id(),
            encrypted_query.values(),
        )
        .unwrap(),
        query_ref
    );

    let answer_ref = answer.as_ref();
    let answer_from: AnswerRef<'_, MODULUS> = (&answer).into();
    assert_eq!(answer_ref, answer_from);
    assert_eq!(answer_ref.rows(), answer.rows());
    assert_eq!(answer_ref.blocks(), answer.blocks());
    assert_eq!(answer_ref.instance_id(), answer.instance_id());
    assert_eq!(answer_ref.query_id(), answer.query_id());
    assert_eq!(answer_ref.values(), answer.values());

    // And the views still round-trip the decoding key's identifiers.
    assert_eq!(answer_ref.instance_id(), decoding_key.instance_id());
    assert_eq!(answer_ref.query_id(), decoding_key.query_id());
}

#[test]
fn decode_through_a_ref_matches_the_owned_matrix() {
    let mut rng = ChaCha20Rng::seed_from_u64(4);
    let (rows, ell) = (5_usize, 8_usize);
    let mut state = derive_toeplitz(rows, ell);
    let matrix = random_vector(rows * ell, &mut rng);
    let q = random_vector(ell, &mut rng);
    let encrypted = encrypt(&mut state, &matrix).unwrap();
    let (encrypted_query, decoding_key) = query(&mut state, &q).unwrap();

    let blocks = state.params().blocks().unwrap();
    let zero = field().element_u32(0);
    let mut values = vec![zero; rows * blocks];
    answer_into(&state.params(), &encrypted, &encrypted_query, &mut values).unwrap();

    // The same answer buffer, decoded through the owned matrix and through
    // a validated borrowed view, must produce identical products.
    let owned = AnswerMatrix::from_parts(
        encrypted.instance_id(),
        encrypted_query.query_id(),
        values.clone(),
        rows,
        blocks,
    );
    let borrowed = AnswerRef::new(
        encrypted.instance_id(),
        encrypted_query.query_id(),
        &values,
        rows,
        blocks,
    )
    .unwrap();

    let mut from_owned = vec![zero; rows];
    decode_into(&owned, &decoding_key, &mut from_owned).unwrap();
    let mut from_ref = vec![zero; rows];
    decode_into(&borrowed, &decoding_key, &mut from_ref).unwrap();

    assert_eq!(from_owned, from_ref);
    assert_eq!(from_ref, naive_matrix_vector(&matrix, &q, rows, ell));
}

#[test]
fn decode_through_a_ref_rejects_mismatched_keys() {
    let zero = field().element_u32(0);
    let values = random_vector(6, &mut ChaCha20Rng::seed_from_u64(5));
    let borrowed = AnswerRef::new(1, 7, &values, 2, 3).unwrap();

    let other_instance = DecodingKey::from_parts(2, 7, vec![zero; 3], vec![zero; 2]);
    assert!(matches!(
        decode_into(&borrowed, &other_instance, &mut []),
        Err(ProtocolError::InstanceMismatch { .. })
    ));

    let other_query = DecodingKey::from_parts(1, 8, vec![zero; 3], vec![zero; 2]);
    assert!(matches!(
        decode_into(&borrowed, &other_query, &mut []),
        Err(ProtocolError::QueryMismatch { .. })
    ));

    let short_key = DecodingKey::from_parts(1, 7, vec![zero; 2], vec![zero; 2]);
    assert!(matches!(
        decode_into(&borrowed, &short_key, &mut []),
        Err(ProtocolError::LengthMismatch { .. })
    ));
}
