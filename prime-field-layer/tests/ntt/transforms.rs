use prime_field_layer::{FieldError, NegacyclicPlan, NttPlan, PrimeField};

use crate::support::{bit_reverse, check_round_trip, oracle_dft};

#[test]
fn cloned_dynamic_plan_keeps_shared_tables_alive_for_transforms() {
    let plan = NttPlan::<1_073_479_681>::new(4_096).unwrap();
    let clone = plan.clone();
    let input: Vec<_> = (0..plan.len()).map(|index| index as u32).collect();
    let mut values = plan.elements(&input);
    plan.forward(&mut values).unwrap();
    drop(plan);
    clone.inverse(&mut values).unwrap();
    assert_eq!(
        values
            .into_iter()
            .map(prime_field_layer::FieldElement::value)
            .collect::<Vec<_>>(),
        input
    );
}

#[test]
fn round_trips_all_modulus_tiers_and_lengths() {
    for length in [1, 2, 4, 8, 16] {
        check_round_trip::<17>(length);
        check_round_trip::<65_537>(length);
        check_round_trip::<1_073_479_681>(length);
        check_round_trip::<2_013_265_921>(length);
        check_round_trip::<2_281_701_377>(length);
    }
    check_round_trip::<65_537>(256);
    check_round_trip::<1_073_479_681>(1_024);
    check_round_trip::<2_013_265_921>(4_096);
    check_round_trip::<2_281_701_377>(256);
}

#[test]
fn forward_values_match_independent_dft_in_documented_order() {
    let field = PrimeField::<17>::new();
    let input = [0, 1, 2, 3, 16, 15, 8, 9];
    let plan = NttPlan::<17>::new_scalar(input.len()).unwrap();
    let mut actual = plan.elements(&input);
    plan.forward(&mut actual).unwrap();
    let natural = oracle_dft::<17>(&input, field.root_of_unity(input.len()).unwrap());
    let expected: Vec<_> = (0..input.len())
        .map(|index| natural[bit_reverse(index, input.len().trailing_zeros())])
        .collect();
    assert_eq!(
        actual
            .into_iter()
            .map(prime_field_layer::FieldElement::value)
            .collect::<Vec<_>>(),
        expected
    );
}

#[test]
fn incremental_twiddles_match_randomized_wide_prime_dfts() {
    const MODULUS: u32 = 2_281_701_377;
    let field = PrimeField::<MODULUS>::new();
    let plan = NttPlan::<MODULUS>::new_scalar(16).unwrap();
    let root = field.root_of_unity(16).unwrap();
    let mut state = 0x9e37_79b9_7f4a_7c15u64;

    for _ in 0..24 {
        let input: Vec<_> = (0..16)
            .map(|_| {
                state ^= state << 7;
                state ^= state >> 9;
                state ^= state << 8;
                (state % u64::from(MODULUS)) as u32
            })
            .collect();
        let natural = oracle_dft::<MODULUS>(&input, root);
        let expected: Vec<_> = (0..16)
            .map(|index| natural[bit_reverse(index, 4)])
            .collect();
        let mut actual = plan.elements(&input);
        plan.forward(&mut actual).unwrap();
        assert_eq!(
            actual.iter().map(|value| value.value()).collect::<Vec<_>>(),
            expected
        );
        plan.inverse(&mut actual).unwrap();
        assert_eq!(
            actual.iter().map(|value| value.value()).collect::<Vec<_>>(),
            input
        );
    }
}

#[test]
fn invalid_lengths_and_mismatches_are_rejected() {
    for length in [0, 3, 6, 32] {
        assert!(matches!(
            NttPlan::<17>::new(length),
            Err(FieldError::UnsupportedTransformLength(value)) if value == length
        ));
    }
    assert_eq!(
        NegacyclicPlan::<17>::new(16).err(),
        Some(FieldError::UnsupportedTransformLength(32))
    );

    let plan = NttPlan::<17>::new(8).unwrap();
    let mut short = plan.elements(&[1, 2, 3, 4]);
    assert_eq!(plan.forward(&mut short), Err(FieldError::LengthMismatch));
    assert_eq!(
        plan.cyclic_convolution(&[0; 8], &[0; 4]),
        Err(FieldError::LengthMismatch)
    );
    assert_eq!(
        plan.linear_convolution(&[1; 6], &[2; 4]),
        Err(FieldError::PlanTooSmall {
            required: 16,
            available: 8,
        })
    );
}
