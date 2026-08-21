use super::{Stage, Twiddle, halve_interval, stage_twiddle_index};
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
pub(super) unsafe fn forward_static<const MODULUS: u32, const N: usize>(
    values: &mut [FieldElement<MODULUS>; N],
    twiddles: &[Twiddle; N],
) {
    let two_p = MODULUS * 2;
    let mut distance = N / 2;
    while distance != 0 {
        let blocks = N / (2 * distance);
        for block in 0..blocks {
            // SAFETY: the compile-time stage layout has exactly `blocks` entries.
            let twiddle = unsafe { *twiddles.get_unchecked(stage_twiddle_index(blocks, block)) };
            let start = block * 2 * distance;
            for index in start..start + distance {
                // SAFETY: both halves are inside the current static butterfly block.
                unsafe {
                    let lhs = halve_interval((*values.as_ptr().add(index)).montgomery(), two_p);
                    let rhs = (*values.as_ptr().add(index + distance)).montgomery();
                    let product = shoup_mul_lazy::<MODULUS>(rhs, twiddle);
                    (*values.as_mut_ptr().add(index)).set_montgomery(lhs + product);
                    (*values.as_mut_ptr().add(index + distance))
                        .set_montgomery(lhs + two_p - product);
                }
            }
        }
        distance /= 2;
    }
    normalize_static::<MODULUS, N>(values);
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn inverse_static<const MODULUS: u32, const N: usize>(
    values: &mut [FieldElement<MODULUS>; N],
    twiddles: &[Twiddle; N],
) {
    let two_p = MODULUS * 2;
    let mut distance = 1;
    while distance < N {
        let blocks = N / (2 * distance);
        for block in 0..blocks {
            // SAFETY: the compile-time stage layout has exactly `blocks` entries.
            let twiddle = unsafe { *twiddles.get_unchecked(stage_twiddle_index(blocks, block)) };
            let start = block * 2 * distance;
            for index in start..start + distance {
                // SAFETY: both halves are inside the current static butterfly block.
                unsafe {
                    let lhs = (*values.as_ptr().add(index)).montgomery();
                    let rhs = (*values.as_ptr().add(index + distance)).montgomery();
                    let difference = lhs + two_p - rhs;
                    (*values.as_mut_ptr().add(index))
                        .set_montgomery(halve_interval(lhs + rhs, two_p));
                    (*values.as_mut_ptr().add(index + distance)).set_montgomery(shoup_mul_lazy::<
                        MODULUS,
                    >(
                        difference, twiddle,
                    ));
                }
            }
        }
        distance *= 2;
    }
    normalize_static::<MODULUS, N>(values);
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
    let two_p = MODULUS * 2;
    for value in values {
        let halved = halve_interval(value.montgomery(), two_p);
        value.set_montgomery(reduce_once(halved, MODULUS));
    }
}

#[inline(always)]
fn normalize_static<const MODULUS: u32, const N: usize>(values: &mut [FieldElement<MODULUS>; N]) {
    let two_p = MODULUS * 2;
    for value in values {
        let halved = halve_interval(value.montgomery(), two_p);
        value.set_montgomery(reduce_once(halved, MODULUS));
    }
}
