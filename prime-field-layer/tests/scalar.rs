#![expect(
    clippy::unreachable,
    clippy::unwrap_used,
    reason = "generated cases use a closed modulus set and valid nonzero operands"
)]

use prime_field_layer::{FieldError, PrimeField};
use proptest::prelude::*;

const fn oracle_pow(mut base: u32, mut exponent: u64, modulus: u32) -> u32 {
    let mut result = 1 % modulus;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = ((result as u64 * base as u64) % modulus as u64) as u32;
        }
        base = ((base as u64 * base as u64) % modulus as u64) as u32;
        exponent >>= 1;
    }
    result
}

fn boundary_values(modulus: u32) -> Vec<u32> {
    let mut values = vec![0, 1 % modulus, modulus.saturating_sub(2), modulus - 1];
    values.sort_unstable();
    values.dedup();
    values
}

fn check_pair<const MODULUS: u32>(field: PrimeField<MODULUS>, lhs: u32, rhs: u32) {
    let modulus = u64::from(MODULUS);
    assert_eq!(
        field.add_canonical(lhs, rhs),
        ((u64::from(lhs) + u64::from(rhs)) % modulus) as u32
    );
    assert_eq!(
        field.sub_canonical(lhs, rhs),
        ((u64::from(lhs) + modulus - u64::from(rhs)) % modulus) as u32
    );
    assert_eq!(
        field.mul(lhs, rhs),
        (u64::from(lhs) * u64::from(rhs) % modulus) as u32
    );
}

fn check_boundaries<const MODULUS: u32>() {
    let field = PrimeField::<MODULUS>::new();
    let values = boundary_values(MODULUS);
    for &lhs in &values {
        assert_eq!(
            field.neg_canonical(lhs),
            if lhs == 0 { 0 } else { MODULUS - lhs }
        );
        assert_eq!(field.square(lhs), field.mul(lhs, lhs));
        for &rhs in &values {
            check_pair(field, lhs, rhs);
        }
    }
}

#[test]
fn arithmetic_matches_oracle_at_boundaries() {
    check_boundaries::<2>();
    check_boundaries::<3>();
    check_boundaries::<17>();
    check_boundaries::<257>();
    check_boundaries::<65_537>();
    check_boundaries::<2_147_483_647>();
    check_boundaries::<4_294_967_291>();
}

fn check_exhaustive<const MODULUS: u32>() {
    let field = PrimeField::<MODULUS>::new();
    for lhs in 0..MODULUS {
        for rhs in 0..MODULUS {
            check_pair(field, lhs, rhs);
        }
    }
}

#[test]
fn arithmetic_is_exhaustive_for_tiny_fields() {
    check_exhaustive::<2>();
    check_exhaustive::<3>();
    check_exhaustive::<5>();
    check_exhaustive::<17>();
}

fn check_reduction<const MODULUS: u32>() {
    let field = PrimeField::<MODULUS>::new();
    let values = [
        0,
        1,
        u64::from(MODULUS) - 1,
        u64::from(MODULUS),
        u64::from(MODULUS) + 1,
        u64::from(u32::MAX),
        u64::MAX,
    ];
    for value in values {
        assert_eq!(field.reduce_u64(value), (value % u64::from(MODULUS)) as u32);
    }
}

#[test]
fn reduction_handles_full_u64_range() {
    check_reduction::<2>();
    check_reduction::<17>();
    check_reduction::<65_537>();
    check_reduction::<1_073_479_681>();
    check_reduction::<4_294_967_291>();
}

fn check_exponentiation<const MODULUS: u32>() {
    let field = PrimeField::<MODULUS>::new();
    let exponents = [
        0,
        1,
        2,
        3,
        15,
        16,
        31,
        32,
        1_000,
        u64::from(u32::MAX),
        u64::MAX,
    ];
    for base in boundary_values(MODULUS) {
        for exponent in exponents {
            assert_eq!(
                field.pow(base, exponent),
                oracle_pow(base, exponent, MODULUS)
            );
        }
    }
}

#[test]
fn exponentiation_matches_oracle() {
    check_exponentiation::<2>();
    check_exponentiation::<17>();
    check_exponentiation::<65_537>();
    check_exponentiation::<1_073_479_681>();
    check_exponentiation::<4_294_967_291>();
}

fn check_all_inverses<const MODULUS: u32>() {
    let field = PrimeField::<MODULUS>::new();
    assert_eq!(field.inv(0), Err(FieldError::DivisionByZero));
    for value in 1..MODULUS {
        let inverse = field.inv(value).unwrap();
        assert!(inverse < MODULUS);
        assert_eq!(field.mul(value, inverse), 1);
    }
}

fn check_boundary_inverses<const MODULUS: u32>() {
    let field = PrimeField::<MODULUS>::new();
    for value in boundary_values(MODULUS)
        .into_iter()
        .filter(|&value| value != 0)
    {
        assert_eq!(field.mul(value, field.inv(value).unwrap()), 1);
    }
}

#[test]
fn inversion_is_multiplicative_and_rejects_zero() {
    check_all_inverses::<2>();
    check_all_inverses::<3>();
    check_all_inverses::<5>();
    check_all_inverses::<17>();
    check_all_inverses::<257>();
    check_boundary_inverses::<65_537>();
    check_boundary_inverses::<1_073_479_681>();
    check_boundary_inverses::<4_294_967_291>();
}

fn check_noncanonical_inverses<const MODULUS: u32>() {
    let field = PrimeField::<MODULUS>::new();
    let largest_multiple = u32::MAX / MODULUS * MODULUS;
    for zero in [MODULUS, largest_multiple] {
        assert_eq!(field.inv(zero), Err(FieldError::DivisionByZero));
    }

    for value in [MODULUS + 1, u32::MAX] {
        if !value.is_multiple_of(MODULUS) {
            let inverse = field.inv(value).unwrap();
            assert_eq!(field.mul(value, inverse), 1);
        }
    }
}

#[test]
fn inversion_reduces_full_width_inputs_before_the_zero_check() {
    check_noncanonical_inverses::<2>();
    check_noncanonical_inverses::<17>();
    check_noncanonical_inverses::<65_537>();
    check_noncanonical_inverses::<1_073_479_681>();
    check_noncanonical_inverses::<4_294_967_291>();
}

#[test]
fn full_width_raw_operations_documented_as_reducing_match_the_oracle() {
    let field = PrimeField::<17>::new();
    for value in [17, 18, u32::MAX - 1, u32::MAX] {
        assert_eq!(field.square(value), oracle_pow(value % 17, 2, 17));
        assert_eq!(field.pow(value, 31), oracle_pow(value % 17, 31, 17));
        assert_eq!(field.mul(value, u32::MAX), {
            (u64::from(value) * u64::from(u32::MAX) % 17) as u32
        });
    }
}

fn check_random<const MODULUS: u32>(lhs: u32, rhs: u32, exponent: u32) {
    let field = PrimeField::<MODULUS>::new();
    let lhs = (u64::from(lhs) % u64::from(MODULUS)) as u32;
    let rhs = (u64::from(rhs) % u64::from(MODULUS)) as u32;
    let expected_add = ((u64::from(lhs) + u64::from(rhs)) % u64::from(MODULUS)) as u32;
    let expected_sub =
        ((u64::from(lhs) + u64::from(MODULUS) - u64::from(rhs)) % u64::from(MODULUS)) as u32;
    let expected_mul = (u64::from(lhs) * u64::from(rhs) % u64::from(MODULUS)) as u32;

    assert_eq!(field.add_canonical(lhs, rhs), expected_add);
    assert_eq!(field.sub_canonical(lhs, rhs), expected_sub);
    assert_eq!(field.mul(lhs, rhs), expected_mul);
    assert_eq!(field.square(lhs), field.mul(lhs, lhs));
    assert_eq!(
        field.pow(lhs, u64::from(exponent)),
        oracle_pow(lhs, u64::from(exponent), MODULUS)
    );
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn random_operations_match_widened_oracle(
        modulus in prop::sample::select(vec![3, 17, 257, 65_537, 1_073_479_681, 4_294_967_291u32]),
        lhs in any::<u32>(),
        rhs in any::<u32>(),
        exponent in any::<u32>(),
    ) {
        match modulus {
            3 => check_random::<3>(lhs, rhs, exponent),
            17 => check_random::<17>(lhs, rhs, exponent),
            257 => check_random::<257>(lhs, rhs, exponent),
            65_537 => check_random::<65_537>(lhs, rhs, exponent),
            1_073_479_681 => check_random::<1_073_479_681>(lhs, rhs, exponent),
            4_294_967_291 => check_random::<4_294_967_291>(lhs, rhs, exponent),
            _ => unreachable!(),
        }
    }
}
