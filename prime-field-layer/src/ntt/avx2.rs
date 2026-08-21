use super::{Stage, Twiddle};
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
                    let lhs = (*values.as_ptr().add(index)).montgomery();
                    let rhs = (*values.as_ptr().add(index + stage.distance)).montgomery();
                    let product = shoup_mul_lazy::<MODULUS>(rhs, twiddle);
                    (*values.as_mut_ptr().add(index))
                        .set_montgomery(reduce_once(lhs + product, two_p));
                    (*values.as_mut_ptr().add(index + stage.distance))
                        .set_montgomery(reduce_once(lhs + two_p - product, two_p));
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
            for index in start..start + stage.distance {
                // SAFETY: construction partitions every stage into in-bounds blocks.
                unsafe {
                    let lhs = (*values.as_ptr().add(index)).montgomery();
                    let rhs = (*values.as_ptr().add(index + stage.distance)).montgomery();
                    (*values.as_mut_ptr().add(index)).set_montgomery(reduce_once(lhs + rhs, two_p));
                    let difference = reduce_once(lhs + two_p - rhs, two_p);
                    (*values.as_mut_ptr().add(index + stage.distance))
                        .set_montgomery(shoup_mul_lazy::<MODULUS>(difference, twiddle));
                }
            }
        }
    }
    normalize::<MODULUS>(values);
}

#[inline(always)]
fn shoup_mul_lazy<const MODULUS: u32>(value: u32, twiddle: Twiddle) -> u32 {
    let quotient = (u64::from(value) * u64::from(twiddle.shoup)) >> 32;
    (u64::from(value) * u64::from(twiddle.canonical) - quotient * u64::from(MODULUS)) as u32
}

#[inline(always)]
const fn reduce_once(value: u32, modulus: u32) -> u32 {
    let (reduced, borrow) = value.overflowing_sub(modulus);
    reduced.wrapping_add(modulus & 0u32.wrapping_sub(borrow as u32))
}

#[inline(always)]
fn normalize<const MODULUS: u32>(values: &mut [FieldElement<MODULUS>]) {
    for value in values {
        value.set_montgomery(reduce_once(value.montgomery(), MODULUS));
    }
}
