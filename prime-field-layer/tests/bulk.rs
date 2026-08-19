#![expect(
    clippy::unwrap_used,
    reason = "test inputs establish that these operations must succeed"
)]

use prime_field_layer::{FieldError, PrimeField};

const LENGTHS: [usize; 18] = [
    0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 129,
];

fn inputs(length: usize, modulus: u32) -> (Vec<u32>, Vec<u32>) {
    let lhs = (0..length)
        .map(|index| ((index as u64 * 1_103_515_245 + 12_345) % u64::from(modulus)) as u32)
        .collect();
    let rhs = (0..length)
        .map(|index| ((index as u64 * 2_654_435_761 + 97) % u64::from(modulus)) as u32)
        .collect();
    (lhs, rhs)
}

fn check_elementwise<const MODULUS: u32>() {
    let field = PrimeField::<MODULUS>::new();
    for length in LENGTHS {
        let (lhs, rhs) = inputs(length, MODULUS);

        let mut actual = lhs.clone();
        field.add_assign(&mut actual, &rhs).unwrap();
        let expected: Vec<_> = lhs
            .iter()
            .zip(&rhs)
            .map(|(&lhs, &rhs)| field.add(lhs, rhs))
            .collect();
        assert_eq!(actual, expected, "add, modulus {MODULUS}, length {length}");

        let mut actual = lhs.clone();
        field.sub_assign(&mut actual, &rhs).unwrap();
        let expected: Vec<_> = lhs
            .iter()
            .zip(&rhs)
            .map(|(&lhs, &rhs)| field.sub(lhs, rhs))
            .collect();
        assert_eq!(actual, expected, "sub, modulus {MODULUS}, length {length}");

        let mut actual = lhs.clone();
        field.mul_assign(&mut actual, &rhs).unwrap();
        let expected: Vec<_> = lhs
            .iter()
            .zip(&rhs)
            .map(|(&lhs, &rhs)| field.mul(lhs, rhs))
            .collect();
        assert_eq!(actual, expected, "mul, modulus {MODULUS}, length {length}");
    }
}

#[test]
fn elementwise_kernels_match_scalar_operations() {
    check_elementwise::<17>();
    check_elementwise::<65_537>();
    check_elementwise::<4_294_967_291>();
}

fn check_unary_and_scalar<const MODULUS: u32>() {
    let field = PrimeField::<MODULUS>::new();
    for length in LENGTHS {
        let (values, _) = inputs(length, MODULUS);

        let mut actual = values.clone();
        field.neg_assign(&mut actual);
        let expected: Vec<_> = values.iter().map(|&value| field.neg(value)).collect();
        assert_eq!(actual, expected, "neg, modulus {MODULUS}, length {length}");

        for scalar in [0, 1, MODULUS / 2, MODULUS - 1] {
            let mut actual = values.clone();
            field.scalar_mul_assign(&mut actual, scalar);
            let expected: Vec<_> = values
                .iter()
                .map(|&value| field.mul(value, scalar))
                .collect();
            assert_eq!(
                actual, expected,
                "scalar mul, modulus {MODULUS}, scalar {scalar}, length {length}"
            );
        }
    }
}

#[test]
fn unary_and_scalar_kernels_match_scalar_operations() {
    check_unary_and_scalar::<17>();
    check_unary_and_scalar::<65_537>();
    check_unary_and_scalar::<4_294_967_291>();
}

fn check_dot<const MODULUS: u32>() {
    let field = PrimeField::<MODULUS>::new();
    for length in LENGTHS {
        let (lhs, rhs) = inputs(length, MODULUS);
        let expected = lhs
            .iter()
            .zip(&rhs)
            .fold(0, |sum, (&lhs, &rhs)| field.add(sum, field.mul(lhs, rhs)));
        assert_eq!(field.dot(&lhs, &rhs).unwrap(), expected);
    }
}

#[test]
fn dot_product_matches_a_reduced_oracle() {
    check_dot::<17>();
    check_dot::<65_537>();
    check_dot::<4_294_967_291>();
}

fn check_montgomery_bulk<const MODULUS: u32>() {
    let field = PrimeField::<MODULUS>::new();
    for length in LENGTHS {
        let (lhs, rhs) = inputs(length, MODULUS);
        let lhs: Vec<_> = lhs
            .iter()
            .map(|&value| field.element(u64::from(value)))
            .collect();
        let rhs: Vec<_> = rhs
            .iter()
            .map(|&value| field.element(u64::from(value)))
            .collect();

        let mut actual = lhs.clone();
        field.mul_elements_assign(&mut actual, &rhs).unwrap();
        let expected: Vec<_> = lhs.iter().zip(&rhs).map(|(&lhs, &rhs)| lhs * rhs).collect();
        assert_eq!(actual, expected);

        let mut actual = lhs.clone();
        let scalar = field.element(u64::from(MODULUS - 1));
        field.scalar_mul_elements_assign(&mut actual, scalar);
        let expected: Vec<_> = lhs.iter().map(|&value| value * scalar).collect();
        assert_eq!(actual, expected);
    }
}

#[test]
fn montgomery_kernels_match_element_operations() {
    check_montgomery_bulk::<17>();
    check_montgomery_bulk::<65_537>();
    check_montgomery_bulk::<4_294_967_291>();
}

fn check_montgomery_boundaries<const MODULUS: u32>() {
    let field = PrimeField::<MODULUS>::new();
    let boundaries = [0, 1, MODULUS / 2, MODULUS - 2, MODULUS - 1];
    let lhs: Vec<_> = (0..67)
        .map(|index| field.element(u64::from(boundaries[index % boundaries.len()])))
        .collect();
    let rhs: Vec<_> = (0..67)
        .map(|index| field.element(u64::from(boundaries[(index * 3 + 1) % boundaries.len()])))
        .collect();
    let expected: Vec<_> = lhs.iter().zip(&rhs).map(|(&lhs, &rhs)| lhs * rhs).collect();
    let mut actual = lhs;

    field.mul_elements_assign(&mut actual, &rhs).unwrap();
    assert_eq!(actual, expected);
}

#[test]
fn montgomery_simd_handles_boundary_values_and_tails() {
    check_montgomery_boundaries::<17>();
    check_montgomery_boundaries::<65_537>();
    check_montgomery_boundaries::<998_244_353>();
    check_montgomery_boundaries::<4_294_967_291>();
}

fn check_batch_inverse<const MODULUS: u32>() {
    let field = PrimeField::<MODULUS>::new();
    let mut values: Vec<_> = (1_u64..128)
        .map(|value| value % u64::from(MODULUS))
        .map(|value| value.max(1) as u32)
        .collect();
    let expected: Vec<_> = values
        .iter()
        .map(|&value| field.inv(value).unwrap())
        .collect();

    field.batch_inv_assign(&mut values).unwrap();
    assert_eq!(values, expected);
}

#[test]
fn batch_inverse_matches_scalar_inversion() {
    check_batch_inverse::<17>();
    check_batch_inverse::<65_537>();
    check_batch_inverse::<998_244_353>();
}

#[test]
fn binary_kernels_reject_mismatched_lengths() {
    let field = PrimeField::<65_537>::new();
    let rhs = [1, 2, 3];

    assert_eq!(
        field.add_assign(&mut [1, 2], &rhs),
        Err(FieldError::LengthMismatch)
    );
    assert_eq!(
        field.sub_assign(&mut [1, 2], &rhs),
        Err(FieldError::LengthMismatch)
    );
    assert_eq!(
        field.mul_assign(&mut [1, 2], &rhs),
        Err(FieldError::LengthMismatch)
    );
    field.dot(&[1, 2], &rhs).unwrap_err();
}
