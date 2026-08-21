use super::stage_twiddle_index;
use crate::{FieldElement, ntt::Twiddle};

#[target_feature(enable = "avx2")]
pub(super) unsafe fn forward<const MODULUS: u32, const N: usize>(
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
                    let lhs = (*values.as_ptr().add(index)).montgomery();
                    let rhs = (*values.as_ptr().add(index + distance)).montgomery();
                    let product = shoup_mul_lazy::<MODULUS>(rhs, twiddle);
                    (*values.as_mut_ptr().add(index))
                        .set_montgomery(reduce_once(lhs + product, two_p));
                    (*values.as_mut_ptr().add(index + distance))
                        .set_montgomery(reduce_once(lhs + two_p - product, two_p));
                }
            }
        }
        distance /= 2;
    }
    normalize::<MODULUS, N>(values);
}

#[target_feature(enable = "avx2")]
pub(super) unsafe fn inverse<const MODULUS: u32, const N: usize>(
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
                    (*values.as_mut_ptr().add(index)).set_montgomery(reduce_once(lhs + rhs, two_p));
                    let difference = reduce_once(lhs + two_p - rhs, two_p);
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
    normalize::<MODULUS, N>(values);
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
fn normalize<const MODULUS: u32, const N: usize>(values: &mut [FieldElement<MODULUS>; N]) {
    for value in values {
        value.set_montgomery(reduce_once(value.montgomery(), MODULUS));
    }
}
