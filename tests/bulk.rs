use prime_field_layer::PrimeField;

const LENGTHS: [usize; 18] = [
    0, 1, 2, 3, 4, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 129,
];

fn inputs(length: usize, modulus: u32) -> (Vec<u32>, Vec<u32>) {
    let lhs = (0..length)
        .map(|index| ((index as u64 * 1_103_515_245 + 12_345) % modulus as u64) as u32)
        .collect();
    let rhs = (0..length)
        .map(|index| ((index as u64 * 2_654_435_761 + 97) % modulus as u64) as u32)
        .collect();
    (lhs, rhs)
}

#[test]
fn elementwise_kernels_match_scalar_operations() {
    for modulus in [17, 65_537, 4_294_967_291] {
        let field = PrimeField::new(modulus).unwrap();
        for length in LENGTHS {
            let (lhs, rhs) = inputs(length, modulus);

            let mut actual = lhs.clone();
            field.add_assign(&mut actual, &rhs).unwrap();
            let expected: Vec<_> = lhs
                .iter()
                .zip(&rhs)
                .map(|(&lhs, &rhs)| field.add(lhs, rhs))
                .collect();
            assert_eq!(actual, expected, "add, modulus {modulus}, length {length}");

            let mut actual = lhs.clone();
            field.sub_assign(&mut actual, &rhs).unwrap();
            let expected: Vec<_> = lhs
                .iter()
                .zip(&rhs)
                .map(|(&lhs, &rhs)| field.sub(lhs, rhs))
                .collect();
            assert_eq!(actual, expected, "sub, modulus {modulus}, length {length}");

            let mut actual = lhs.clone();
            field.mul_assign(&mut actual, &rhs).unwrap();
            let expected: Vec<_> = lhs
                .iter()
                .zip(&rhs)
                .map(|(&lhs, &rhs)| field.mul(lhs, rhs))
                .collect();
            assert_eq!(actual, expected, "mul, modulus {modulus}, length {length}");
        }
    }
}

#[test]
fn unary_and_scalar_kernels_match_scalar_operations() {
    for modulus in [17, 65_537, 4_294_967_291] {
        let field = PrimeField::new(modulus).unwrap();
        for length in LENGTHS {
            let (values, _) = inputs(length, modulus);

            let mut actual = values.clone();
            field.neg_assign(&mut actual);
            let expected: Vec<_> = values.iter().map(|&value| field.neg(value)).collect();
            assert_eq!(actual, expected, "neg, modulus {modulus}, length {length}");

            for scalar in [0, 1, modulus / 2, modulus - 1] {
                let mut actual = values.clone();
                field.scalar_mul_assign(&mut actual, scalar);
                let expected: Vec<_> = values
                    .iter()
                    .map(|&value| field.mul(value, scalar))
                    .collect();
                assert_eq!(
                    actual, expected,
                    "scalar mul, modulus {modulus}, scalar {scalar}, length {length}"
                );
            }
        }
    }
}

#[test]
fn dot_product_matches_a_reduced_oracle() {
    for modulus in [17, 65_537, 4_294_967_291] {
        let field = PrimeField::new(modulus).unwrap();
        for length in LENGTHS {
            let (lhs, rhs) = inputs(length, modulus);
            let expected = lhs
                .iter()
                .zip(&rhs)
                .fold(0, |sum, (&lhs, &rhs)| field.add(sum, field.mul(lhs, rhs)));
            assert_eq!(field.dot(&lhs, &rhs).unwrap(), expected);
        }
    }
}

#[test]
fn binary_kernels_reject_mismatched_lengths() {
    let field = PrimeField::new(65_537).unwrap();
    let rhs = [1, 2, 3];

    assert!(field.add_assign(&mut [1, 2], &rhs).is_err());
    assert!(field.sub_assign(&mut [1, 2], &rhs).is_err());
    assert!(field.mul_assign(&mut [1, 2], &rhs).is_err());
    assert!(field.dot(&[1, 2], &rhs).is_err());
}
