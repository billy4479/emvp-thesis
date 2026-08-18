use prime_field_layer::PrimeField;
use proptest::prelude::*;

fn oracle_pow(mut base: u32, mut exponent: u64, modulus: u32) -> u32 {
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

fn check_pair(field: &PrimeField, lhs: u32, rhs: u32) {
    let modulus = field.modulus() as u64;
    assert_eq!(
        field.add(lhs, rhs),
        ((lhs as u64 + rhs as u64) % modulus) as u32
    );
    assert_eq!(
        field.sub(lhs, rhs),
        ((lhs as u64 + modulus - rhs as u64) % modulus) as u32
    );
    assert_eq!(
        field.mul(lhs, rhs),
        (lhs as u64 * rhs as u64 % modulus) as u32
    );
}

#[test]
fn arithmetic_matches_oracle_at_boundaries() {
    for modulus in [2, 3, 17, 257, 65_537, 2_147_483_647, 4_294_967_291] {
        let field = PrimeField::new(modulus).unwrap();
        let values = boundary_values(modulus);
        for &lhs in &values {
            assert_eq!(field.neg(lhs), if lhs == 0 { 0 } else { modulus - lhs });
            assert_eq!(field.square(lhs), field.mul(lhs, lhs));
            for &rhs in &values {
                check_pair(&field, lhs, rhs);
            }
        }
    }
}

#[test]
fn arithmetic_is_exhaustive_for_tiny_fields() {
    for modulus in [2, 3, 5, 17] {
        let field = PrimeField::new(modulus).unwrap();
        for lhs in 0..modulus {
            for rhs in 0..modulus {
                check_pair(&field, lhs, rhs);
            }
        }
    }
}

#[test]
fn reduction_handles_full_u64_range() {
    for modulus in [2, 17, 65_537, 998_244_353, 4_294_967_291] {
        let field = PrimeField::new(modulus).unwrap();
        let values = [
            0,
            1,
            modulus as u64 - 1,
            modulus as u64,
            modulus as u64 + 1,
            u32::MAX as u64,
            u64::MAX,
        ];
        for value in values {
            assert_eq!(field.reduce_u64(value), (value % modulus as u64) as u32);
        }
    }
}

#[test]
fn exponentiation_matches_oracle() {
    let exponents = [0, 1, 2, 3, 31, 32, 1_000, u32::MAX as u64, u64::MAX];
    for modulus in [2, 17, 65_537, 998_244_353, 4_294_967_291] {
        let field = PrimeField::new(modulus).unwrap();
        for base in boundary_values(modulus) {
            for exponent in exponents {
                assert_eq!(
                    field.pow(base, exponent),
                    oracle_pow(base, exponent, modulus)
                );
            }
        }
    }
}

#[test]
fn inversion_is_multiplicative_and_rejects_zero() {
    for modulus in [2, 3, 5, 17, 257] {
        let field = PrimeField::new(modulus).unwrap();
        assert!(field.inv(0).is_err());
        for value in 1..modulus {
            let inverse = field.inv(value).unwrap();
            assert!(inverse < modulus);
            assert_eq!(field.mul(value, inverse), 1);
        }
    }

    for modulus in [65_537, 998_244_353, 4_294_967_291] {
        let field = PrimeField::new(modulus).unwrap();
        for value in boundary_values(modulus)
            .into_iter()
            .filter(|&value| value != 0)
        {
            assert_eq!(field.mul(value, field.inv(value).unwrap()), 1);
        }
    }
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(512))]

    #[test]
    fn random_operations_match_widened_oracle(
        modulus in prop::sample::select(vec![3, 17, 257, 65_537, 998_244_353, 4_294_967_291]),
        lhs in any::<u32>(),
        rhs in any::<u32>(),
        exponent in any::<u32>(),
    ) {
        let field = PrimeField::new(modulus).unwrap();
        let lhs = (lhs as u64 % modulus as u64) as u32;
        let rhs = (rhs as u64 % modulus as u64) as u32;
        let expected_add = ((lhs as u64 + rhs as u64) % modulus as u64) as u32;
        let expected_sub = ((lhs as u64 + modulus as u64 - rhs as u64) % modulus as u64) as u32;
        let expected_mul = (lhs as u64 * rhs as u64 % modulus as u64) as u32;

        prop_assert_eq!(field.add(lhs, rhs), expected_add);
        prop_assert_eq!(field.sub(lhs, rhs), expected_sub);
        prop_assert_eq!(field.mul(lhs, rhs), expected_mul);
        prop_assert_eq!(field.square(lhs), field.mul(lhs, lhs));
        prop_assert_eq!(field.pow(lhs, exponent as u64), oracle_pow(lhs, exponent as u64, modulus));
    }
}
