use prime_field_layer::{
    ExtensionField, ExtensionFieldError, FieldError, PolynomialAlgorithm, PolynomialReductionPlan,
};

fn canonical(value: u32, modulus: u32) -> u32 {
    (u64::from(value) % u64::from(modulus)) as u32
}

fn oracle_reduce(k: usize, modulus: u32, modulus_polynomial: &[u32], product: &[u32]) -> Vec<u32> {
    let mut values = vec![0; 2 * k - 1];
    for (target, &coefficient) in values.iter_mut().zip(product) {
        *target = canonical(coefficient, modulus);
    }
    for degree in (k..values.len()).rev() {
        let factor = u64::from(values[degree]);
        for index in 0..k {
            let subtract = factor * u64::from(canonical(modulus_polynomial[index], modulus))
                % u64::from(modulus);
            values[degree - k + index] =
                ((u64::from(values[degree - k + index]) + u64::from(modulus) - subtract)
                    % u64::from(modulus)) as u32;
        }
    }
    values.truncate(k);
    values
}

fn oracle_mul(
    k: usize,
    modulus: u32,
    modulus_polynomial: &[u32],
    lhs: &[u32],
    rhs: &[u32],
) -> Vec<u32> {
    let mut product = vec![0; 2 * k - 1];
    for (lhs_index, &lhs) in lhs.iter().enumerate() {
        for (rhs_index, &rhs) in rhs.iter().enumerate() {
            let index = lhs_index + rhs_index;
            product[index] = ((u64::from(product[index])
                + u64::from(canonical(lhs, modulus)) * u64::from(canonical(rhs, modulus)))
                % u64::from(modulus)) as u32;
        }
    }
    oracle_reduce(k, modulus, modulus_polynomial, &product)
}

fn irreducible_binomial(k: usize, modulus: u32) -> Vec<u32> {
    let mut polynomial = vec![0; k + 1];
    polynomial[0] = modulus - 3;
    polynomial[k] = 1;
    polynomial
}

#[test]
fn schoolbook_reduction_matches_independent_long_division() {
    const MODULUS: u32 = 17;
    let k = 4;
    let modulus = [14, 5, 0, 2, 1];
    let product = [u32::MAX, 17, 18, 16, 15, 14, 13];
    let plan = PolynomialReductionPlan::<MODULUS>::new(k, &modulus).unwrap();
    let mut scratch = plan.scratch();
    let mut actual = vec![0; k];
    plan.reduce(&product, &mut actual, &mut scratch).unwrap();
    assert_eq!(actual, oracle_reduce(k, MODULUS, &modulus, &product));
    assert_eq!(plan.algorithm(), PolynomialAlgorithm::Schoolbook);
}

#[test]
fn checked_extension_arithmetic_matches_independent_oracle() {
    const MODULUS: u32 = 17;
    let k = 4;
    let modulus = [14, 0, 0, 0, 1];
    let extension = ExtensionField::<MODULUS>::new(k, &modulus).unwrap();
    let lhs = [u32::MAX, 17, 18, 16];
    let rhs = [16, 15, u32::MAX - 1, 34];
    let mut scratch = extension.scratch();
    let mut product = vec![0; k];
    let mut square = vec![0; k];

    extension
        .mul(&lhs, &rhs, &mut product, &mut scratch)
        .unwrap();
    extension.square(&lhs, &mut square, &mut scratch).unwrap();

    assert_eq!(product, oracle_mul(k, MODULUS, &modulus, &lhs, &rhs));
    assert_eq!(square, oracle_mul(k, MODULUS, &modulus, &lhs, &lhs));

    {
        let mut lhs_copy = lhs;
        extension.add_assign(&mut lhs_copy, &rhs).unwrap();
        assert_eq!(lhs_copy, [16, 15, 0, 16]);
    }

    {
        let mut lhs_copy = lhs;
        extension.sub_assign(&mut lhs_copy, &rhs).unwrap();
        assert_eq!(lhs_copy, [1, 2, 2, 16]);
    }

    assert!(product.iter().all(|&coefficient| coefficient < MODULUS));
    assert!(square.iter().all(|&coefficient| coefficient < MODULUS));
}

#[test]
fn validation_and_reducibility_errors_are_typed() {
    assert!(matches!(
        PolynomialReductionPlan::<17>::new(0, &[1]),
        Err(ExtensionFieldError::ZeroDegree)
    ));
    assert!(matches!(
        PolynomialReductionPlan::<17>::new(4, &[1, 2, 3]),
        Err(ExtensionFieldError::ModulusLength {
            expected: 5,
            actual: 3
        })
    ));
    assert!(matches!(
        PolynomialReductionPlan::<17>::new(2, &[1, 0, 2]),
        Err(ExtensionFieldError::ModulusNotMonic)
    ));
    assert!(matches!(
        ExtensionField::<17>::new(4, &[16, 0, 0, 0, 1]),
        Err(ExtensionFieldError::ReducibleModulus)
    ));

    let canonicalized = PolynomialReductionPlan::<17>::new(2, &[u32::MAX, 34, 18]).unwrap();
    assert_eq!(canonicalized.modulus_polynomial(), &[0, 0, 1]);
    let mut scratch = canonicalized.scratch();
    let mut output = [99; 2];
    assert_eq!(
        canonicalized.reduce(&[1; 4], &mut output, &mut scratch),
        Err(ExtensionFieldError::ProductTooLong {
            maximum: 3,
            actual: 4
        })
    );
    assert_eq!(output, [99; 2]);
}

#[test]
fn runtime_degree_mismatches_are_rejected_without_mutation() {
    let extension = ExtensionField::<17>::new_unchecked_irreducible(2, &[1, 0, 1]).unwrap();
    let other_extension =
        ExtensionField::<17>::new_unchecked_irreducible(3, &[1, 0, 0, 1]).unwrap();
    let mut wrong_scratch = other_extension.scratch();
    let mut output = [99, 99];
    assert_eq!(
        extension.mul(&[1, 2], &[3, 4], &mut output, &mut wrong_scratch),
        Err(ExtensionFieldError::BaseField(FieldError::LengthMismatch))
    );
    assert_eq!(output, [99, 99]);

    let plan = PolynomialReductionPlan::<17>::new(2, &[1, 0, 1]).unwrap();
    let other_plan = PolynomialReductionPlan::<17>::new(3, &[1, 0, 0, 1]).unwrap();
    let mut wrong_scratch = other_plan.scratch();
    assert_eq!(
        plan.reduce(&[1, 2, 3], &mut output, &mut wrong_scratch),
        Err(ExtensionFieldError::BaseField(FieldError::LengthMismatch))
    );
    assert_eq!(output, [99, 99]);
}

#[test]
fn ntt_reduction_multiplication_and_squaring_match_slow_oracle() {
    const MODULUS: u32 = 998_244_353;
    let k: usize = 128;
    let modulus = irreducible_binomial(k, MODULUS);
    let extension = ExtensionField::<MODULUS>::new_unchecked_irreducible(k, &modulus).unwrap();
    assert_eq!(
        extension.algorithm(),
        PolynomialAlgorithm::Ntt {
            transform_length: 256
        }
    );

    let lhs = (0..k)
        .map(|index| ((index as u64 * 2_654_435_761 + u64::from(u32::MAX)) % (1u64 << 32)) as u32)
        .collect::<Box<[u32]>>();
    let rhs = (0..k)
        .map(|index| ((index as u64 * 1_103_515_245 + 12_345) % (1u64 << 32)) as u32)
        .collect::<Box<[u32]>>();
    let mut scratch = extension.scratch();
    let mut product = vec![0; k];
    let mut square = vec![0; k];
    extension
        .mul(&lhs, &rhs, &mut product, &mut scratch)
        .unwrap();
    extension.square(&lhs, &mut square, &mut scratch).unwrap();
    assert_eq!(product, oracle_mul(k, MODULUS, &modulus, &lhs, &rhs));
    assert_eq!(square, oracle_mul(k, MODULUS, &modulus, &lhs, &lhs));

    let reduction = PolynomialReductionPlan::<MODULUS>::new(k, &modulus).unwrap();
    let input: Vec<_> = (0..2 * k - 1)
        .map(|index| (index as u32).wrapping_mul(2_654_435_761).wrapping_add(97))
        .collect();
    let mut reduction_scratch = reduction.scratch();
    let mut reduced = vec![0; k];
    reduction
        .reduce(&input, &mut reduced, &mut reduction_scratch)
        .unwrap();
    assert_eq!(reduced, oracle_reduce(k, MODULUS, &modulus, &input));
}

#[test]
fn threshold_behavior_is_degree_only() {
    let schoolbook_modulus = irreducible_binomial(23, 998_244_353);
    let ntt_modulus = irreducible_binomial(24, 998_244_353);
    assert_eq!(
        PolynomialReductionPlan::<998_244_353>::new(23, &schoolbook_modulus)
            .unwrap()
            .algorithm(),
        PolynomialAlgorithm::Schoolbook
    );
    assert_eq!(
        PolynomialReductionPlan::<998_244_353>::new(24, &ntt_modulus)
            .unwrap()
            .algorithm(),
        PolynomialAlgorithm::Ntt {
            transform_length: 64
        }
    );
}

#[test]
fn caller_scratch_reuses_all_allocations() {
    const MODULUS: u32 = 998_244_353;
    let k: usize = 128;
    let modulus = irreducible_binomial(k, MODULUS);
    let extension = ExtensionField::<MODULUS>::new_unchecked_irreducible(k, &modulus).unwrap();
    let lhs = (0..k).map(|index| index as u32 + 1).collect::<Box<[u32]>>();
    let rhs = (0..k)
        .map(|index| index as u32 * 3 + 7)
        .collect::<Box<[u32]>>();
    let mut output = vec![0; k];
    let mut scratch = extension.scratch();

    extension
        .mul(&lhs, &rhs, &mut output, &mut scratch)
        .unwrap();
    let allocations = allocation_counter::measure(|| {
        for _ in 0..8 {
            extension
                .mul(&lhs, &rhs, &mut output, &mut scratch)
                .unwrap();
            extension.square(&lhs, &mut output, &mut scratch).unwrap();
        }
    });
    assert_eq!(allocations.count_total, 0);
}
