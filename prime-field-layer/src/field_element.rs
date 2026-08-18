use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::{FieldError, PrimeField};

#[cfg(target_arch = "x86")]
use std::arch::x86::*;
#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// A four-byte field element kept in Montgomery form between API boundaries.
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FieldElement<const MODULUS: u32> {
    montgomery: u32,
}

impl<const MODULUS: u32> FieldElement<MODULUS> {
    /// Constructs an element by reducing `value` into `field`.
    pub fn new(field: &PrimeField<MODULUS>, value: u64) -> Self {
        Self::from_montgomery(PrimeField::<MODULUS>::to_montgomery(
            field.reduce_u64(value),
        ))
    }

    pub fn value(self) -> u32 {
        PrimeField::<MODULUS>::from_montgomery(self.montgomery)
    }

    pub fn field(self) -> PrimeField<MODULUS> {
        PrimeField::<MODULUS>::assume_valid()
    }

    #[inline(always)]
    pub fn square(self) -> Self {
        Self::from_montgomery(PrimeField::<MODULUS>::montgomery_mul(
            self.montgomery,
            self.montgomery,
        ))
    }

    pub fn pow(self, exponent: u64) -> Self {
        Self::from_montgomery(PrimeField::<MODULUS>::pow_montgomery(
            self.montgomery,
            exponent,
        ))
    }

    pub fn inv(self) -> Result<Self, FieldError> {
        if self.montgomery == 0 {
            return Err(FieldError::DivisionByZero);
        }
        Ok(self.pow((MODULUS - 2) as u64))
    }

    #[inline(always)]
    pub(crate) const fn from_montgomery(montgomery: u32) -> Self {
        Self { montgomery }
    }
}

impl<const MODULUS: u32> PrimeField<MODULUS> {
    pub fn element(&self, value: u64) -> FieldElement<MODULUS> {
        FieldElement::new(self, value)
    }

    pub fn add_elements_assign(
        &self,
        lhs: &mut [FieldElement<MODULUS>],
        rhs: &[FieldElement<MODULUS>],
    ) -> Result<(), FieldError> {
        if lhs.len() != rhs.len() {
            return Err(FieldError::LengthMismatch);
        }
        for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
            lhs.montgomery = self.add(lhs.montgomery, rhs.montgomery);
        }
        Ok(())
    }

    pub fn sub_elements_assign(
        &self,
        lhs: &mut [FieldElement<MODULUS>],
        rhs: &[FieldElement<MODULUS>],
    ) -> Result<(), FieldError> {
        if lhs.len() != rhs.len() {
            return Err(FieldError::LengthMismatch);
        }
        for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
            lhs.montgomery = self.sub(lhs.montgomery, rhs.montgomery);
        }
        Ok(())
    }

    pub fn mul_elements_assign(
        &self,
        lhs: &mut [FieldElement<MODULUS>],
        rhs: &[FieldElement<MODULUS>],
    ) -> Result<(), FieldError> {
        if lhs.len() != rhs.len() {
            return Err(FieldError::LengthMismatch);
        }

        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if MODULUS != 2 && lhs.len() >= 8 && std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was detected above and the slices have equal lengths.
            unsafe { mul_elements_assign_avx2(lhs, rhs) };
            return Ok(());
        }

        for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
            lhs.montgomery = Self::montgomery_mul(lhs.montgomery, rhs.montgomery);
        }
        Ok(())
    }

    pub fn scalar_mul_elements_assign(
        &self,
        values: &mut [FieldElement<MODULUS>],
        scalar: FieldElement<MODULUS>,
    ) {
        #[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
        if MODULUS != 2 && values.len() >= 8 && std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was detected above.
            unsafe { scalar_mul_elements_assign_avx2(values, scalar) };
            return;
        }

        for value in values {
            value.montgomery = Self::montgomery_mul(value.montgomery, scalar.montgomery);
        }
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn montgomery_mul_avx2<const MODULUS: u32>(lhs: __m256i, rhs: __m256i) -> __m256i {
    let modulus_32 = _mm256_set1_epi32(MODULUS as i32);
    let modulus_64 = _mm256_set1_epi64x(MODULUS as i64);
    let neg_inv = _mm256_set1_epi32(PrimeField::<MODULUS>::MONTGOMERY_NEG_INV as i32);
    let sign_bit = _mm256_set1_epi64x(i64::MIN);
    let carry_bit = _mm256_set1_epi64x(1i64 << 32);
    let all = _mm256_set1_epi64x(-1);

    let products_even = _mm256_mul_epu32(lhs, rhs);
    let products_odd = _mm256_mul_epu32(_mm256_srli_epi64(lhs, 32), _mm256_srli_epi64(rhs, 32));
    let adjustments = _mm256_mullo_epi32(_mm256_mullo_epi32(lhs, rhs), neg_inv);
    let adjustments_even = _mm256_mul_epu32(adjustments, modulus_32);
    let adjustments_odd = _mm256_mul_epu32(_mm256_srli_epi64(adjustments, 32), modulus_32);

    let reduce = |products: __m256i, adjustments: __m256i| {
        let sums = _mm256_add_epi64(products, adjustments);
        let carries = _mm256_cmpgt_epi64(
            _mm256_xor_si256(products, sign_bit),
            _mm256_xor_si256(sums, sign_bit),
        );
        let reduced = _mm256_or_si256(
            _mm256_srli_epi64(sums, 32),
            _mm256_and_si256(carries, carry_bit),
        );
        let below_modulus = _mm256_cmpgt_epi64(modulus_64, reduced);
        _mm256_sub_epi64(
            reduced,
            _mm256_andnot_si256(below_modulus, _mm256_and_si256(all, modulus_64)),
        )
    };

    let even = reduce(products_even, adjustments_even);
    let odd = reduce(products_odd, adjustments_odd);
    _mm256_or_si256(even, _mm256_slli_epi64(odd, 32))
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn mul_elements_assign_avx2<const MODULUS: u32>(
    lhs: &mut [FieldElement<MODULUS>],
    rhs: &[FieldElement<MODULUS>],
) {
    let vectorized = lhs.len() / 8 * 8;
    let mut index = 0;
    while index < vectorized {
        // SAFETY: each load and store covers eight elements inside the slices.
        unsafe {
            let lhs_vector = _mm256_loadu_si256(lhs.as_ptr().add(index).cast());
            let rhs_vector = _mm256_loadu_si256(rhs.as_ptr().add(index).cast());
            let result = montgomery_mul_avx2::<MODULUS>(lhs_vector, rhs_vector);
            _mm256_storeu_si256(lhs.as_mut_ptr().add(index).cast(), result);
        }
        index += 8;
    }

    for (lhs, rhs) in lhs[vectorized..].iter_mut().zip(&rhs[vectorized..]) {
        lhs.montgomery = PrimeField::<MODULUS>::montgomery_mul(lhs.montgomery, rhs.montgomery);
    }
}

#[cfg(any(target_arch = "x86", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn scalar_mul_elements_assign_avx2<const MODULUS: u32>(
    values: &mut [FieldElement<MODULUS>],
    scalar: FieldElement<MODULUS>,
) {
    let scalar_vector = _mm256_set1_epi32(scalar.montgomery as i32);
    let vectorized = values.len() / 8 * 8;
    let mut index = 0;
    while index < vectorized {
        // SAFETY: each load and store covers eight elements inside the slice.
        unsafe {
            let values_vector = _mm256_loadu_si256(values.as_ptr().add(index).cast());
            let result = montgomery_mul_avx2::<MODULUS>(values_vector, scalar_vector);
            _mm256_storeu_si256(values.as_mut_ptr().add(index).cast(), result);
        }
        index += 8;
    }

    for value in &mut values[vectorized..] {
        value.montgomery =
            PrimeField::<MODULUS>::montgomery_mul(value.montgomery, scalar.montgomery);
    }
}

impl<const MODULUS: u32> Add for FieldElement<MODULUS> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let field = self.field();
        Self::from_montgomery(field.add(self.montgomery, rhs.montgomery))
    }
}

impl<const MODULUS: u32> AddAssign for FieldElement<MODULUS> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const MODULUS: u32> Sub for FieldElement<MODULUS> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        let field = self.field();
        Self::from_montgomery(field.sub(self.montgomery, rhs.montgomery))
    }
}

impl<const MODULUS: u32> SubAssign for FieldElement<MODULUS> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const MODULUS: u32> Mul for FieldElement<MODULUS> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_montgomery(PrimeField::<MODULUS>::montgomery_mul(
            self.montgomery,
            rhs.montgomery,
        ))
    }
}

impl<const MODULUS: u32> MulAssign for FieldElement<MODULUS> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const MODULUS: u32> Neg for FieldElement<MODULUS> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        let field = self.field();
        Self::from_montgomery(field.neg(self.montgomery))
    }
}

impl<const MODULUS: u32> std::fmt::Display for FieldElement<MODULUS> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.value())
    }
}
