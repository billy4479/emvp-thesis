use super::{Stage, Twiddle, halve_interval};
use crate::FieldElement;

#[target_feature(enable = "avx2")]
pub(super) unsafe fn forward<const MODULUS: u32>(
    values: &mut [FieldElement<MODULUS>],
    stages: &[Stage],
) {
    let two_p = MODULUS * 2;
    for stage in stages {
        for (block, &twiddle) in stage.forward.iter().enumerate() {
            let start = block * 2 * stage.distance;
            for index in start..start + stage.distance {
                // SAFETY: construction partitions every stage into in-bounds blocks.
                unsafe {
                    let lhs = halve_interval((*values.as_ptr().add(index)).montgomery(), two_p);
                    let rhs = (*values.as_ptr().add(index + stage.distance)).montgomery();
                    let product = shoup_mul_lazy::<MODULUS>(rhs, twiddle);
                    (*values.as_mut_ptr().add(index)).set_montgomery(lhs + product);
                    (*values.as_mut_ptr().add(index + stage.distance))
                        .set_montgomery(lhs + two_p - product);
                }
            }
        }
    }
    normalize::<MODULUS>(values);
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn inverse<const MODULUS: u32>(
    values: &mut [FieldElement<MODULUS>],
    stages: &[Stage],
) {
    let two_p = MODULUS * 2;
    for stage in stages.iter().rev() {
        for (block, &twiddle) in stage.inverse.iter().enumerate() {
            let start = block * 2 * stage.distance;
            let (lhs_values, rhs_values) =
                values[start..start + 2 * stage.distance].split_at_mut(stage.distance);
            for (lhs_value, rhs_value) in lhs_values.iter_mut().zip(rhs_values) {
                let lhs = lhs_value.montgomery();
                let rhs = rhs_value.montgomery();
                let difference = lhs + two_p - rhs;
                lhs_value.set_montgomery(halve_interval(lhs + rhs, two_p));
                rhs_value.set_montgomery(shoup_mul_lazy::<MODULUS>(difference, twiddle));
            }
        }
    }
    normalize::<MODULUS>(values);
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn forward_reduced<const MODULUS: u32>(
    values: &mut [FieldElement<MODULUS>],
    stages: &[Stage],
) {
    for stage in stages {
        for (block, &twiddle) in stage.forward.iter().enumerate() {
            let start = block * 2 * stage.distance;
            for index in start..start + stage.distance {
                // SAFETY: construction partitions every stage into in-bounds blocks.
                unsafe {
                    let lhs = (*values.as_ptr().add(index)).montgomery();
                    let rhs = (*values.as_ptr().add(index + stage.distance)).montgomery();
                    let product = shoup_mul_reduced::<MODULUS>(rhs, twiddle);
                    (*values.as_mut_ptr().add(index))
                        .set_montgomery(reduce_once(lhs + product, MODULUS));
                    (*values.as_mut_ptr().add(index + stage.distance))
                        .set_montgomery(reduce_once(lhs + MODULUS - product, MODULUS));
                }
            }
        }
    }
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn inverse_reduced<const MODULUS: u32>(
    values: &mut [FieldElement<MODULUS>],
    stages: &[Stage],
) {
    for stage in stages.iter().rev() {
        for (block, &twiddle) in stage.inverse.iter().enumerate() {
            let start = block * 2 * stage.distance;
            let (lhs_values, rhs_values) =
                values[start..start + 2 * stage.distance].split_at_mut(stage.distance);
            for (lhs_value, rhs_value) in lhs_values.iter_mut().zip(rhs_values) {
                let lhs = lhs_value.montgomery();
                let rhs = rhs_value.montgomery();
                lhs_value.set_montgomery(reduce_once(lhs + rhs, MODULUS));
                rhs_value.set_montgomery(shoup_mul_reduced::<MODULUS>(
                    reduce_once(lhs + MODULUS - rhs, MODULUS),
                    twiddle,
                ));
            }
        }
    }
}

#[inline(always)]
fn shoup_mul_lazy<const MODULUS: u32>(value: u32, twiddle: Twiddle) -> u32 {
    let quotient = (u64::from(value) * u64::from(twiddle.shoup)) >> 32;
    (u64::from(value) * u64::from(twiddle.canonical) - quotient * u64::from(MODULUS)) as u32
}

/// Shoup product for canonical inputs when `MODULUS < 2^31`.
///
/// The uncorrected product is in `[0, 2p)`, so it and the butterfly sums fit a
/// `u32` lane. One correction restores `[0, p)` after every multiplication.
#[inline(always)]
fn shoup_mul_reduced<const MODULUS: u32>(value: u32, twiddle: Twiddle) -> u32 {
    reduce_once(shoup_mul_lazy::<MODULUS>(value, twiddle), MODULUS)
}

#[inline(always)]
const fn reduce_once(value: u32, modulus: u32) -> u32 {
    let (reduced, borrow) = value.overflowing_sub(modulus);
    reduced.wrapping_add(modulus & 0u32.wrapping_sub(borrow as u32))
}

#[inline(always)]
fn normalize<const MODULUS: u32>(values: &mut [FieldElement<MODULUS>]) {
    let two_p = MODULUS * 2;
    for value in values {
        let halved = halve_interval(value.montgomery(), two_p);
        value.set_montgomery(reduce_once(halved, MODULUS));
    }
}
