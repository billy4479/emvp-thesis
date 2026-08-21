#![expect(
    clippy::unwrap_used,
    reason = "test inputs establish that plans and transforms must succeed"
)]

use prime_field_layer::{
    FieldError, NegacyclicPlan, NttBackend, NttPerformanceWarning, NttPlan, PrimeField,
    StaticNttPlan, linear_convolution,
};
use proptest::prelude::*;

const fn oracle_pow(base: u32, mut exponent: usize, modulus: u32) -> u32 {
    let mut base = base as u64;
    let mut result = 1u64;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = result * base % modulus as u64;
        }
        base = base * base % modulus as u64;
        exponent >>= 1;
    }
    result as u32
}

fn oracle_dft<const MODULUS: u32>(input: &[u32], root: u32) -> Vec<u32> {
    (0..input.len())
        .map(|frequency| {
            input.iter().enumerate().fold(0u64, |sum, (index, &value)| {
                (sum + value as u64 * oracle_pow(root, index * frequency, MODULUS) as u64)
                    % MODULUS as u64
            }) as u32
        })
        .collect()
}

const fn bit_reverse(value: usize, bits: u32) -> usize {
    if bits == 0 {
        0
    } else {
        value.reverse_bits() >> (usize::BITS - bits)
    }
}

fn oracle_linear<const MODULUS: u32>(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    if lhs.is_empty() || rhs.is_empty() {
        return Vec::new();
    }
    let mut result = vec![0u32; lhs.len() + rhs.len() - 1];
    for (lhs_index, &lhs) in lhs.iter().enumerate() {
        for (rhs_index, &rhs) in rhs.iter().enumerate() {
            let index = lhs_index + rhs_index;
            result[index] =
                ((result[index] as u64 + lhs as u64 * rhs as u64) % MODULUS as u64) as u32;
        }
    }
    result
}

fn oracle_cyclic<const MODULUS: u32>(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    let mut result = vec![0u32; lhs.len()];
    for (lhs_index, &lhs_value) in lhs.iter().enumerate() {
        for (rhs_index, &rhs_value) in rhs.iter().enumerate() {
            let index = (lhs_index + rhs_index) % lhs.len();
            result[index] = ((result[index] as u64 + lhs_value as u64 * rhs_value as u64)
                % MODULUS as u64) as u32;
        }
    }
    result
}

fn oracle_negacyclic<const MODULUS: u32>(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    let mut result = vec![0i128; lhs.len()];
    for (lhs_index, &lhs_value) in lhs.iter().enumerate() {
        for (rhs_index, &rhs_value) in rhs.iter().enumerate() {
            let degree = lhs_index + rhs_index;
            let product = lhs_value as i128 * rhs_value as i128;
            if degree < lhs.len() {
                result[degree] += product;
            } else {
                result[degree - lhs.len()] -= product;
            }
        }
    }
    result
        .into_iter()
        .map(|value| value.rem_euclid(MODULUS as i128) as u32)
        .collect()
}

fn check_round_trip<const MODULUS: u32>(length: usize) {
    let plan = NttPlan::<MODULUS>::new(length).unwrap();
    let input: Vec<_> = (0..length)
        .map(|index| ((index as u64 * 2_654_435_761 + 97) % MODULUS as u64) as u32)
        .collect();
    let mut values = plan.elements(&input);
    plan.forward(&mut values).unwrap();
    plan.inverse(&mut values).unwrap();
    assert_eq!(
        values
            .into_iter()
            .map(prime_field_layer::FieldElement::value)
            .collect::<Vec<_>>(),
        input
    );
}

fn check_static_matches_dynamic<const MODULUS: u32, const N: usize>() {
    let dynamic = NttPlan::<MODULUS>::new(N).unwrap();
    let static_plan = StaticNttPlan::<MODULUS, N>::new().unwrap();
    let input: [u32; N] =
        std::array::from_fn(|index| ((index as u64 * 2_654_435_761 + 97) % MODULUS as u64) as u32);
    let mut dynamic_values = dynamic.elements(&input);
    let mut static_values = static_plan.elements(&input);
    dynamic.forward(&mut dynamic_values).unwrap();
    static_plan.forward(&mut static_values);
    assert_eq!(static_values.as_slice(), dynamic_values);
    dynamic.inverse(&mut dynamic_values).unwrap();
    static_plan.inverse(&mut static_values);
    assert_eq!(static_values.as_slice(), dynamic_values);
    assert_eq!(
        static_values.map(prime_field_layer::FieldElement::value),
        input
    );
}

#[test]
fn compile_time_size_plans_match_dynamic_plans() {
    check_static_matches_dynamic::<17, 1>();
    check_static_matches_dynamic::<17, 8>();
    check_static_matches_dynamic::<65_537, 256>();
    check_static_matches_dynamic::<998_244_353, 1_024>();
    check_static_matches_dynamic::<2_013_265_921, 256>();
    check_static_matches_dynamic::<2_281_701_377, 256>();
}

#[test]
fn compile_time_size_pointwise_product_matches_dynamic_plan() {
    const MODULUS: u32 = 998_244_353;
    const N: usize = 256;
    let dynamic = NttPlan::<MODULUS>::new(N).unwrap();
    let static_plan = StaticNttPlan::<MODULUS, N>::new().unwrap();
    let lhs = std::array::from_fn(|index| index as u32 * 31 + 7);
    let rhs = std::array::from_fn(|index| index as u32 * 17 + 11);
    let mut dynamic_lhs = dynamic.elements(&lhs);
    let mut dynamic_rhs = dynamic.elements(&rhs);
    let mut static_lhs = static_plan.elements(&lhs);
    let mut static_rhs = static_plan.elements(&rhs);
    dynamic.forward(&mut dynamic_lhs).unwrap();
    dynamic.forward(&mut dynamic_rhs).unwrap();
    static_plan.forward(&mut static_lhs);
    static_plan.forward(&mut static_rhs);
    dynamic
        .pointwise_mul_assign(&mut dynamic_lhs, &dynamic_rhs)
        .unwrap();
    static_plan.pointwise_mul_assign(&mut static_lhs, &static_rhs);
    assert_eq!(static_lhs.as_slice(), dynamic_lhs);
}

#[test]
fn round_trips_all_modulus_tiers_and_lengths() {
    for length in [1, 2, 4, 8, 16] {
        check_round_trip::<17>(length);
        check_round_trip::<65_537>(length);
        check_round_trip::<998_244_353>(length);
        check_round_trip::<2_013_265_921>(length);
        check_round_trip::<2_281_701_377>(length);
    }
    check_round_trip::<65_537>(256);
    check_round_trip::<998_244_353>(1_024);
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
                (state % MODULUS as u64) as u32
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

fn check_convolutions<const MODULUS: u32>() {
    let boundaries = [0, 1, MODULUS / 2, MODULUS - 2, MODULUS - 1, 7, 11, 3];
    let rhs = [MODULUS - 1, 0, 2, MODULUS / 2, 5, 1, 9, 4];

    let cyclic = NttPlan::<MODULUS>::new(8).unwrap();
    assert_eq!(
        cyclic.cyclic_convolution(&boundaries, &rhs).unwrap(),
        oracle_cyclic::<MODULUS>(&boundaries, &rhs)
    );

    let negacyclic = NegacyclicPlan::<MODULUS>::new(8).unwrap();
    assert_eq!(
        negacyclic.convolution(&boundaries, &rhs).unwrap(),
        oracle_negacyclic::<MODULUS>(&boundaries, &rhs)
    );

    let lhs = &boundaries[..5];
    let rhs = &rhs[..4];
    assert_eq!(
        linear_convolution::<MODULUS>(lhs, rhs).unwrap(),
        oracle_linear::<MODULUS>(lhs, rhs)
    );
}

#[test]
fn convolution_variants_match_independent_oracles() {
    check_convolutions::<17>();
    check_convolutions::<65_537>();
    check_convolutions::<998_244_353>();
    check_convolutions::<2_281_701_377>();
}

#[test]
fn empty_linear_inputs_produce_empty_output() {
    assert_eq!(
        linear_convolution::<17>(&[], &[1, 2]).unwrap(),
        Vec::<u32>::new()
    );
    assert_eq!(
        linear_convolution::<17>(&[1, 2], &[]).unwrap(),
        Vec::<u32>::new()
    );
    assert_eq!(
        linear_convolution::<17>(&[], &[]).unwrap(),
        Vec::<u32>::new()
    );
}

#[test]
fn length_one_convolutions_and_unreduced_inputs_are_supported() {
    let cyclic = NttPlan::<17>::new_scalar(1).unwrap();
    let negacyclic = NegacyclicPlan::<17>::new_scalar(1).unwrap();
    assert_eq!(cyclic.cyclic_convolution(&[16], &[16]).unwrap(), vec![1]);
    assert_eq!(negacyclic.convolution(&[16], &[16]).unwrap(), vec![1]);
    assert_eq!(linear_convolution::<17>(&[16], &[16]).unwrap(), vec![1]);

    let lhs = [u32::MAX, 17, 18, 3_000_000_000];
    let rhs = [u32::MAX - 1, 34, 19];
    assert_eq!(
        linear_convolution::<17>(&lhs, &rhs).unwrap(),
        oracle_linear::<17>(&lhs, &rhs)
    );
    assert_eq!(
        linear_convolution::<2_281_701_377>(&lhs, &rhs).unwrap(),
        oracle_linear::<2_281_701_377>(&lhs, &rhs)
    );
    assert_eq!(
        linear_convolution::<998_244_353>(&lhs, &rhs).unwrap(),
        oracle_linear::<998_244_353>(&lhs, &rhs)
    );
    assert_eq!(
        linear_convolution::<2_013_265_921>(&lhs, &rhs).unwrap(),
        oracle_linear::<2_013_265_921>(&lhs, &rhs)
    );
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

#[test]
fn diagnostics_report_the_actual_modulus_tier() {
    let lazy = NttPlan::<998_244_353>::new_scalar(256).unwrap();
    assert_eq!(lazy.backend(), NttBackend::ScalarShoupLazy);
    assert_eq!(
        lazy.performance_warning(),
        Some(NttPerformanceWarning::ScalarRequested)
    );

    let tight = NttPlan::<2_013_265_921>::new(256).unwrap();
    assert_eq!(tight.backend(), NttBackend::ScalarShoup);
    assert_eq!(
        tight.performance_warning(),
        Some(NttPerformanceWarning::ModulusTooLargeForAvx2)
    );

    let wide = NttPlan::<2_281_701_377>::new(256).unwrap();
    assert_eq!(wide.backend(), NttBackend::ScalarMontgomery);
    assert_eq!(
        wide.performance_warning(),
        Some(NttPerformanceWarning::MontgomeryFallback)
    );

    let below_boundary = NttPlan::<1_053_818_881>::new_scalar(16).unwrap();
    assert_eq!(below_boundary.backend(), NttBackend::ScalarShoupLazy);
    let above_boundary = NttPlan::<1_107_296_257>::new(16).unwrap();
    assert_eq!(above_boundary.backend(), NttBackend::ScalarShoup);

    let short_auto = NttPlan::<998_244_353>::new(256).unwrap();
    #[cfg(target_arch = "x86_64")]
    {
        if std::arch::is_x86_feature_detected!("avx2") {
            assert_eq!(short_auto.backend(), NttBackend::Avx2ShoupLazy);
            assert_eq!(short_auto.performance_warning(), None);
        } else {
            assert_eq!(short_auto.backend(), NttBackend::ScalarShoupLazy);
            assert_eq!(
                short_auto.performance_warning(),
                Some(NttPerformanceWarning::Avx2Unavailable)
            );
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    assert_eq!(
        short_auto.performance_warning(),
        Some(NttPerformanceWarning::Avx2Unavailable)
    );
}

proptest! {
    #[test]
    fn property_linear_matches_naive_998244353(
        lhs in prop::collection::vec(0u32..998_244_353, 0..17),
        rhs in prop::collection::vec(0u32..998_244_353, 0..17),
    ) {
        prop_assert_eq!(
            linear_convolution::<998_244_353>(&lhs, &rhs).unwrap(),
            oracle_linear::<998_244_353>(&lhs, &rhs)
        );
    }

    #[test]
    fn property_cyclic_and_negacyclic_match_naive_65537(
        lhs in prop::collection::vec(0u32..65_537, 16),
        rhs in prop::collection::vec(0u32..65_537, 16),
    ) {
        let cyclic = NttPlan::<65_537>::new(16).unwrap();
        let negacyclic = NegacyclicPlan::<65_537>::new(16).unwrap();
        prop_assert_eq!(
            cyclic.cyclic_convolution(&lhs, &rhs).unwrap(),
            oracle_cyclic::<65_537>(&lhs, &rhs)
        );
        prop_assert_eq!(
            negacyclic.convolution(&lhs, &rhs).unwrap(),
            oracle_negacyclic::<65_537>(&lhs, &rhs)
        );
    }

    #[test]
    fn property_wide_prime_convolutions_match_naive(
        lhs in prop::collection::vec(0u32..2_281_701_377, 0..9),
        rhs in prop::collection::vec(0u32..2_281_701_377, 0..9),
        cyclic_lhs in prop::collection::vec(0u32..2_281_701_377, 8),
        cyclic_rhs in prop::collection::vec(0u32..2_281_701_377, 8),
    ) {
        prop_assert_eq!(
            linear_convolution::<2_281_701_377>(&lhs, &rhs).unwrap(),
            oracle_linear::<2_281_701_377>(&lhs, &rhs)
        );
        let plan = NttPlan::<2_281_701_377>::new_scalar(8).unwrap();
        prop_assert_eq!(
            plan.cyclic_convolution(&cyclic_lhs, &cyclic_rhs).unwrap(),
            oracle_cyclic::<2_281_701_377>(&cyclic_lhs, &cyclic_rhs)
        );
    }
}

#[cfg(target_arch = "x86_64")]
#[unsafe(no_mangle)]
#[inline(never)]
pub extern "C" fn review_wide_montgomery_kernel(lhs: u32, rhs: u32) -> u32 {
    let field = PrimeField::<2_281_701_377>::new();
    (field.element(lhs as u64) * field.element(rhs as u64)).value()
}

#[test]
#[cfg(target_arch = "x86_64")]
fn representative_wide_montgomery_kernel_matches_oracle() {
    for (lhs, rhs) in [
        (0, 0),
        (1, 2_281_701_376),
        (2_281_701_376, 2_281_701_376),
        (u32::MAX, u32::MAX - 1),
    ] {
        assert_eq!(
            review_wide_montgomery_kernel(lhs, rhs),
            (lhs as u64 % 2_281_701_377 * (rhs as u64 % 2_281_701_377) % 2_281_701_377) as u32
        );
    }
}
