use prime_field_layer::{NegacyclicPlan, NttPlan, linear_convolution};

use crate::support::{check_convolutions, oracle_linear};

#[test]
fn convolution_variants_match_independent_oracles() {
    check_convolutions::<17>();
    check_convolutions::<65_537>();
    check_convolutions::<1_073_479_681>();
    check_convolutions::<2_013_265_921>();
    check_convolutions::<2_281_701_377>();
}

#[test]
fn automatic_dispatch_crosses_the_schoolbook_cutoff_in_both_directions() {
    // The free function dispatches to schoolbook while `min(m, n) <= 2` or
    // `m * n <= 5184` and to a fresh NTT otherwise. The shapes below sit
    // exactly on both sides of each rule, including highly rectangular
    // inputs, and every modulus tier's NTT backend (lazy Shoup, reduced
    // Shoup, Montgomery) must reproduce the full-width schoolbook oracle.
    const SCHOOLBOOK_PRODUCT_CUTOFF: usize = 5_184;
    const SHAPES: [(usize, usize, bool); 8] = [
        // `m * n == 5184` exactly, square and rectangular: schoolbook.
        (72, 72, true),
        (3, 1_728, true),
        // One coefficient past the cutoff in each dimension: NTT.
        (72, 73, false),
        (3, 1_729, false),
        // Just below the cutoff and just above it.
        (71, 73, true),
        (65, 80, false),
        // The `min(m, n) <= 2` rule keeps a tall operand schoolbook, while a
        // slightly wider rectangle of comparable size needs the NTT.
        (2, 2_048, true),
        (4, 2_048, false),
    ];

    fn check_shape<const MODULUS: u32>(lhs_length: usize, rhs_length: usize, schoolbook: bool) {
        assert_eq!(
            lhs_length.min(rhs_length) <= 2 || lhs_length * rhs_length <= SCHOOLBOOK_PRODUCT_CUTOFF,
            schoolbook,
            "shape {lhs_length}x{rhs_length} misclassified"
        );
        let lhs: Vec<_> = (0..lhs_length)
            .map(|index| ((index as u64 * 2_654_435_761 + 97) % u64::from(MODULUS)) as u32)
            .collect();
        let rhs: Vec<_> = (0..rhs_length)
            .map(|index| ((index as u64 * 1_103_515_245 + 12_345) % u64::from(MODULUS)) as u32)
            .collect();
        assert_eq!(
            linear_convolution::<MODULUS>(&lhs, &rhs).unwrap(),
            oracle_linear::<MODULUS>(&lhs, &rhs),
            "shape {lhs_length}x{rhs_length}, modulus {MODULUS}"
        );
    }

    for &(lhs_length, rhs_length, schoolbook) in &SHAPES {
        check_shape::<65_537>(lhs_length, rhs_length, schoolbook);
        check_shape::<1_073_479_681>(lhs_length, rhs_length, schoolbook);
        check_shape::<2_013_265_921>(lhs_length, rhs_length, schoolbook);
        check_shape::<2_281_701_377>(lhs_length, rhs_length, schoolbook);
    }
}

#[test]
fn tight_prime_rectangular_linear_convolution_matches_full_width_schoolbook_oracle() {
    const MODULUS: u32 = 2_013_265_921;
    const BOUNDARIES: [u32; 7] = [
        0,
        1,
        MODULUS - 1,
        MODULUS,
        MODULUS + 1,
        u32::MAX - 1,
        u32::MAX,
    ];
    let lhs: Vec<_> = (0..97)
        .map(|index| {
            BOUNDARIES
                .get(index)
                .copied()
                .unwrap_or_else(|| (index as u32).wrapping_mul(2_654_435_761).wrapping_add(97))
        })
        .collect();
    let rhs: Vec<_> = (0..65)
        .map(|index| {
            BOUNDARIES.get(index).copied().unwrap_or_else(|| {
                (index as u32)
                    .wrapping_mul(1_103_515_245)
                    .wrapping_add(12_345)
            })
        })
        .collect();

    assert_eq!(
        linear_convolution::<MODULUS>(&lhs, &rhs).unwrap(),
        oracle_linear::<MODULUS>(&lhs, &rhs)
    );
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
    let cyclic = NttPlan::<17>::new(1).unwrap();
    let negacyclic = NegacyclicPlan::<17>::new(1).unwrap();
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
        linear_convolution::<1_073_479_681>(&lhs, &rhs).unwrap(),
        oracle_linear::<1_073_479_681>(&lhs, &rhs)
    );
    assert_eq!(
        linear_convolution::<2_013_265_921>(&lhs, &rhs).unwrap(),
        oracle_linear::<2_013_265_921>(&lhs, &rhs)
    );
}
