use prime_field_layer::{NegacyclicPlan, NttPlan, linear_convolution};

use crate::support::{check_convolutions, oracle_linear};

#[test]
fn convolution_variants_match_independent_oracles() {
    check_convolutions::<17>();
    check_convolutions::<65_537>();
    check_convolutions::<998_244_353>();
    check_convolutions::<2_013_265_921>();
    check_convolutions::<2_281_701_377>();
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
