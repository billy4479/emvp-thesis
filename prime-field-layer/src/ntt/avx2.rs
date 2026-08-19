use super::{Stage, Twiddle, normalize, shoup_mul_lazy_for};
use crate::FieldElement;

use std::arch::x86_64::{
    __m256i, _mm256_add_epi32, _mm256_and_si256, _mm256_cmpgt_epi32, _mm256_loadu_si256,
    _mm256_mul_epu32, _mm256_mullo_epi32, _mm256_or_si256, _mm256_set1_epi32, _mm256_slli_epi64,
    _mm256_srli_epi64, _mm256_storeu_si256, _mm256_sub_epi32, _mm256_xor_si256,
};

#[target_feature(enable = "avx2")]
pub(super) unsafe fn forward<const MODULUS: u32>(
    values: &mut [FieldElement<MODULUS>],
    stages: &[Stage],
) {
    let two_p = MODULUS * 2;
    for stage in stages {
        for (block, &twiddle) in stage.forward.iter().enumerate() {
            let start = block * 2 * stage.distance;
            let vectorized = stage.distance / 8 * 8;
            let mut offset = 0;
            while offset < vectorized {
                // SAFETY: both vectors are within the current butterfly block.
                unsafe {
                    let lhs = _mm256_loadu_si256(values.as_ptr().add(start + offset).cast());
                    let rhs = _mm256_loadu_si256(
                        values.as_ptr().add(start + stage.distance + offset).cast(),
                    );
                    let product = shoup_mul_vector::<MODULUS>(rhs, twiddle);
                    let sum = reduce_once_vector(_mm256_add_epi32(lhs, product), two_p);
                    let difference = reduce_once_vector(
                        _mm256_sub_epi32(
                            _mm256_add_epi32(lhs, _mm256_set1_epi32(two_p.cast_signed())),
                            product,
                        ),
                        two_p,
                    );
                    _mm256_storeu_si256(values.as_mut_ptr().add(start + offset).cast(), sum);
                    _mm256_storeu_si256(
                        values
                            .as_mut_ptr()
                            .add(start + stage.distance + offset)
                            .cast(),
                        difference,
                    );
                }
                offset += 8;
            }
            for index in start + vectorized..start + stage.distance {
                let lhs = values[index].montgomery();
                let product = shoup_mul_lazy_for::<MODULUS>(
                    values[index + stage.distance].montgomery(),
                    twiddle,
                );
                values[index].set_montgomery(reduce_once_scalar(lhs + product, two_p));
                values[index + stage.distance]
                    .set_montgomery(reduce_once_scalar(lhs + two_p - product, two_p));
            }
        }
    }
    normalize(values);
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
            let vectorized = stage.distance / 8 * 8;
            let mut offset = 0;
            while offset < vectorized {
                // SAFETY: both vectors are within the current butterfly block.
                unsafe {
                    let lhs = _mm256_loadu_si256(values.as_ptr().add(start + offset).cast());
                    let rhs = _mm256_loadu_si256(
                        values.as_ptr().add(start + stage.distance + offset).cast(),
                    );
                    let sum = reduce_once_vector(_mm256_add_epi32(lhs, rhs), two_p);
                    let difference = reduce_once_vector(
                        _mm256_sub_epi32(
                            _mm256_add_epi32(lhs, _mm256_set1_epi32(two_p.cast_signed())),
                            rhs,
                        ),
                        two_p,
                    );
                    let product = shoup_mul_vector::<MODULUS>(difference, twiddle);
                    _mm256_storeu_si256(values.as_mut_ptr().add(start + offset).cast(), sum);
                    _mm256_storeu_si256(
                        values
                            .as_mut_ptr()
                            .add(start + stage.distance + offset)
                            .cast(),
                        product,
                    );
                }
                offset += 8;
            }
            for index in start + vectorized..start + stage.distance {
                let lhs = values[index].montgomery();
                let rhs = values[index + stage.distance].montgomery();
                values[index].set_montgomery(reduce_once_scalar(lhs + rhs, two_p));
                values[index + stage.distance].set_montgomery(shoup_mul_lazy_for::<MODULUS>(
                    reduce_once_scalar(lhs + two_p - rhs, two_p),
                    twiddle,
                ));
            }
        }
    }
    normalize(values);
}

#[target_feature(enable = "avx2")]
unsafe fn shoup_mul_vector<const MODULUS: u32>(value: __m256i, twiddle: Twiddle) -> __m256i {
    let multiplier = _mm256_set1_epi32(twiddle.canonical.cast_signed());
    let shoup = _mm256_set1_epi32(twiddle.shoup.cast_signed());
    let even = _mm256_srli_epi64(_mm256_mul_epu32(value, shoup), 32);
    let odd = _mm256_slli_epi64(
        _mm256_srli_epi64(
            _mm256_mul_epu32(_mm256_srli_epi64(value, 32), _mm256_srli_epi64(shoup, 32)),
            32,
        ),
        32,
    );
    let quotient = _mm256_or_si256(even, odd);
    _mm256_sub_epi32(
        _mm256_mullo_epi32(value, multiplier),
        _mm256_mullo_epi32(quotient, _mm256_set1_epi32(MODULUS.cast_signed())),
    )
}

#[target_feature(enable = "avx2")]
unsafe fn reduce_once_vector(value: __m256i, modulus: u32) -> __m256i {
    let sign = _mm256_set1_epi32(i32::MIN);
    let modulus_vector = _mm256_set1_epi32(modulus.cast_signed());
    let greater_or_equal = _mm256_cmpgt_epi32(
        _mm256_xor_si256(value, sign),
        _mm256_xor_si256(
            _mm256_set1_epi32(modulus.wrapping_sub(1).cast_signed()),
            sign,
        ),
    );
    _mm256_sub_epi32(value, _mm256_and_si256(greater_or_equal, modulus_vector))
}

#[inline(always)]
const fn reduce_once_scalar(value: u32, modulus: u32) -> u32 {
    let (reduced, borrow) = value.overflowing_sub(modulus);
    reduced.wrapping_add(modulus & 0u32.wrapping_sub(borrow as u32))
}
