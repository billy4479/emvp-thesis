use prime_field_layer::{NegacyclicPlan, NttPlan, linear_convolution};
use proptest::prelude::*;

use crate::support::{oracle_cyclic, oracle_linear, oracle_negacyclic};

proptest! {
    #[test]
    fn property_linear_matches_naive_1073479681(
        lhs in prop::collection::vec(0u32..1_073_479_681, 0..17),
        rhs in prop::collection::vec(0u32..1_073_479_681, 0..17),
    ) {
        prop_assert_eq!(
            linear_convolution::<1_073_479_681>(&lhs, &rhs).unwrap(),
            oracle_linear::<1_073_479_681>(&lhs, &rhs)
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
