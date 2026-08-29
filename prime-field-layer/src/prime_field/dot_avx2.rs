#![expect(
    unsafe_code,
    reason = "AVX2 intrinsics require target-feature and pointer safety proofs"
)]
#![expect(
    clippy::cast_ptr_alignment,
    reason = "AVX2 loadu and storeu explicitly support unaligned pointers"
)]

use std::arch::x86_64::{
    __m256i, _mm256_add_epi64, _mm256_and_si256, _mm256_loadu_si256, _mm256_mul_epu32,
    _mm256_set1_epi64x, _mm256_setzero_si256, _mm256_slli_epi64, _mm256_srli_epi64,
    _mm256_storeu_si256,
};

use crate::constant_time::reduce_once_u64;

pub(super) const PSEUDO_MERSENNE_32_MODULUS: u32 = 4_294_967_291;

const SIMD_WIDTH: usize = 8;
const PSEUDO_MERSENNE_CHUNK_ELEMENTS: usize = 1_024;
const PSEUDO_MERSENNE_32_MODULUS_U64: u64 = PSEUDO_MERSENNE_32_MODULUS as u64;

/// Loads eight unaligned `u32` values starting at `offset`.
///
/// # Safety
///
/// AVX2 must be available and `values.add(offset)` must address eight readable
/// `u32` values.
#[target_feature(enable = "avx2")]
unsafe fn load(values: *const u32, offset: usize) -> __m256i {
    // SAFETY: the caller proves that offset begins an in-bounds vector.
    let values = unsafe { values.add(offset) }.cast::<__m256i>();
    // SAFETY: the pointer addresses 32 readable bytes; loadu permits any alignment.
    unsafe { _mm256_loadu_si256(values) }
}

/// Stores four `u64` lanes without requiring alignment.
///
/// # Safety
///
/// AVX2 must be available and `output` must address four writable `u64` values.
#[target_feature(enable = "avx2")]
unsafe fn store(output: *mut u64, value: __m256i) {
    // SAFETY: the caller provides 32 writable bytes; storeu permits any alignment.
    unsafe { _mm256_storeu_si256(output.cast::<__m256i>(), value) };
}

pub(super) fn dot_u64(lhs: &[u32], rhs: &[u32]) -> Option<u64> {
    if lhs.len() != rhs.len() || lhs.len() < SIMD_WIDTH || !std::is_x86_feature_detected!("avx2") {
        return None;
    }

    // SAFETY: runtime detection proves AVX2 availability. The caller checks
    // that the complete dot product fits in u64 before entering this kernel.
    Some(unsafe { dot_u64_avx2(lhs, rhs) })
}

/// Computes an exact dot product known to fit in `u64`.
///
/// # Safety
///
/// AVX2 must be available, the slices must have equal lengths, and their
/// complete dot product must fit in `u64`.
#[target_feature(enable = "avx2")]
unsafe fn dot_u64_avx2(lhs: &[u32], rhs: &[u32]) -> u64 {
    let mut even = _mm256_setzero_si256();
    let mut odd = _mm256_setzero_si256();
    let vectorized_len = lhs.len() / SIMD_WIDTH * SIMD_WIDTH;
    let mut offset = 0;

    // Every lane contains a disjoint sum of nonnegative products. The complete
    // sum bound therefore proves that each lane and horizontal partial fits.
    while offset < vectorized_len {
        // SAFETY: the vectorized prefix leaves eight in-bounds u32 values.
        let lhs_values = unsafe { load(lhs.as_ptr(), offset) };
        // SAFETY: equal lengths leave the same eight values in rhs.
        let rhs_values = unsafe { load(rhs.as_ptr(), offset) };

        even = _mm256_add_epi64(even, _mm256_mul_epu32(lhs_values, rhs_values));
        odd = _mm256_add_epi64(
            odd,
            _mm256_mul_epu32(
                _mm256_srli_epi64::<32>(lhs_values),
                _mm256_srli_epi64::<32>(rhs_values),
            ),
        );
        offset += SIMD_WIDTH;
    }

    let mut lanes = [0u64; 4];
    let combined = _mm256_add_epi64(even, odd);
    // SAFETY: lanes contains exactly 32 writable bytes; storeu permits any alignment.
    unsafe { store(lanes.as_mut_ptr(), combined) };

    let mut sum = lanes.into_iter().sum::<u64>();
    for (&lhs, &rhs) in lhs[vectorized_len..].iter().zip(&rhs[vectorized_len..]) {
        sum += u64::from(lhs) * u64::from(rhs);
    }
    sum
}

pub(super) fn dot_pseudo_mersenne_32(lhs: &[u32], rhs: &[u32]) -> Option<u32> {
    if lhs.len() != rhs.len() || !std::is_x86_feature_detected!("avx2") {
        return None;
    }

    // SAFETY: runtime detection proves AVX2 availability and equal lengths
    // establish the kernel's slice precondition.
    Some(unsafe { dot_pseudo_mersenne_32_avx2(lhs, rhs) })
}

/// Computes a dot product modulo `2^32 - 5` using `2^32 = 5 (mod p)`.
///
/// # Safety
///
/// AVX2 must be available and the slices must have equal lengths.
#[target_feature(enable = "avx2")]
unsafe fn dot_pseudo_mersenne_32_avx2(lhs: &[u32], rhs: &[u32]) -> u32 {
    let low_mask = _mm256_set1_epi64x(i64::from(u32::MAX));
    let vectorized_len = lhs.len() / SIMD_WIDTH * SIMD_WIDTH;
    let mut sum = 0u64;

    for chunk_start in (0..vectorized_len).step_by(PSEUDO_MERSENNE_CHUNK_ELEMENTS) {
        let chunk_end = (chunk_start + PSEUDO_MERSENNE_CHUNK_ELEMENTS).min(vectorized_len);
        let mut even = _mm256_setzero_si256();
        let mut odd = _mm256_setzero_si256();

        for offset in (chunk_start..chunk_end).step_by(SIMD_WIDTH) {
            // SAFETY: each chunk boundary is a multiple of eight and chunk_end
            // does not exceed the vectorized prefix.
            let lhs_values = unsafe { load(lhs.as_ptr(), offset) };
            // SAFETY: equal lengths establish the same bound for rhs.
            let rhs_values = unsafe { load(rhs.as_ptr(), offset) };
            let even_products = _mm256_mul_epu32(lhs_values, rhs_values);
            let odd_products = _mm256_mul_epu32(
                _mm256_srli_epi64::<32>(lhs_values),
                _mm256_srli_epi64::<32>(rhs_values),
            );

            even = _mm256_add_epi64(even, fold_products(even_products, low_mask));
            odd = _mm256_add_epi64(odd, fold_products(odd_products, low_mask));
        }

        let mut even_lanes = [0u64; 4];
        let mut odd_lanes = [0u64; 4];
        // SAFETY: each output array contains exactly 32 writable bytes.
        unsafe { store(even_lanes.as_mut_ptr(), even) };
        // SAFETY: each output array contains exactly 32 writable bytes.
        unsafe { store(odd_lanes.as_mut_ptr(), odd) };

        // A folded product is below 6 * 2^32. Each lane receives at most
        // 128 products, and the sum of all eight lanes remains below 2^45.
        let chunk_sum = even_lanes.into_iter().chain(odd_lanes).sum();
        sum = reduce_pseudo_mersenne_chunk(sum + reduce_pseudo_mersenne_chunk(chunk_sum));
    }

    let mut tail_sum = 0u64;
    for (&lhs, &rhs) in lhs[vectorized_len..].iter().zip(&rhs[vectorized_len..]) {
        let product = u64::from(lhs) * u64::from(rhs);
        tail_sum += (product & u64::from(u32::MAX)) + 5 * (product >> 32);
    }

    reduce_pseudo_mersenne_chunk(sum + reduce_pseudo_mersenne_chunk(tail_sum)) as u32
}

#[target_feature(enable = "avx2")]
fn fold_products(products: __m256i, low_mask: __m256i) -> __m256i {
    let high = _mm256_srli_epi64::<32>(products);
    _mm256_add_epi64(
        _mm256_and_si256(products, low_mask),
        _mm256_add_epi64(high, _mm256_slli_epi64::<2>(high)),
    )
}

/// Reduces the values produced by one 1024-element accumulator chunk.
#[inline(always)]
fn reduce_pseudo_mersenne_chunk(value: u64) -> u64 {
    let folded = (value & u64::from(u32::MAX)) + 5 * (value >> 32);
    reduce_once_u64(folded, PSEUDO_MERSENNE_32_MODULUS_U64)
}
