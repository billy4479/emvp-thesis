use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::{FieldError, PrimeField};

/// A field element stored as a four-byte Montgomery residue.
///
/// For a canonical value `a`, the private word represents `a * 2^32 mod
/// MODULUS`, not `a` itself. Arithmetic keeps this representation so repeated
/// multiplication, including NTT pointwise multiplication, does not repeatedly
/// convert at API boundaries. Use [`Self::value`] to obtain a canonical `u32`.
/// The modulus is part of the type, and an element stores no field pointer.
/// Hot add, subtract, and Montgomery multiplication kernels are designed without
/// value-dependent branches, but the crate has not been formally audited as a
/// constant-time implementation. Exponentiation has separate caveats documented
/// on [`Self::pow`].
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FieldElement<const MODULUS: u32> {
    montgomery: u32,
}

impl<const MODULUS: u32> FieldElement<MODULUS> {
    /// Reduces `value` modulo `MODULUS` and converts it to Montgomery form.
    ///
    /// This takes constant space and performs one `u64` remainder plus a
    /// Montgomery conversion. For a `u32` input, [`PrimeField::element_u32`]
    /// avoids the separate remainder. `field` establishes the matching
    /// compile-time modulus; it is not retained by the element.
    #[must_use]
    pub fn new(field: &PrimeField<MODULUS>, value: u64) -> Self {
        Self::from_montgomery(PrimeField::<MODULUS>::to_montgomery(
            field.reduce_u64(value),
        ))
    }

    /// Returns this element as its canonical residue in `0..MODULUS`.
    ///
    /// This performs one Montgomery reduction in constant space. Prefer
    /// keeping intermediate values as `FieldElement`s and converting only at
    /// an external boundary or after an inverse NTT.
    #[must_use]
    pub fn value(self) -> u32 {
        PrimeField::<MODULUS>::from_montgomery(self.montgomery)
    }

    /// Returns the zero-sized field value for this element's modulus.
    ///
    /// This is an `O(1)` type-level association and neither inspects nor
    /// converts the element.
    #[must_use]
    pub const fn field(self) -> PrimeField<MODULUS> {
        PrimeField::<MODULUS>::assume_valid()
    }

    /// Squares this element while retaining Montgomery representation.
    ///
    /// This is one Montgomery multiplication, takes constant space, and is
    /// equivalent to `self * self`.
    #[must_use]
    #[inline(always)]
    pub fn square(self) -> Self {
        Self::from_montgomery(PrimeField::<MODULUS>::montgomery_mul(
            self.montgomery,
            self.montgomery,
        ))
    }

    /// Raises this element to `exponent` in the field.
    ///
    /// Binary exponentiation takes `O(log(exponent + 1))` Montgomery
    /// multiplications and allocates nothing. An exponent of zero returns one,
    /// including for a zero base. Control flow depends on the exponent bits, so
    /// this method is not intended to hide a secret exponent.
    #[must_use]
    pub fn pow(self, exponent: u64) -> Self {
        Self::from_montgomery(PrimeField::<MODULUS>::pow_montgomery(
            self.montgomery,
            exponent,
        ))
    }

    /// Returns the multiplicative inverse of this element.
    ///
    /// This uses Fermat exponentiation by the public, compile-time exponent
    /// `MODULUS - 2`, takes `O(log MODULUS)` operations, and allocates nothing.
    /// It returns [`FieldError::DivisionByZero`] for zero.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::DivisionByZero`] when this element is zero.
    pub fn inv(self) -> Result<Self, FieldError> {
        if self.montgomery == 0 {
            return Err(FieldError::DivisionByZero);
        }
        Ok(self.pow(u64::from(MODULUS - 2)))
    }

    #[inline(always)]
    pub(super) const fn from_montgomery(montgomery: u32) -> Self {
        Self { montgomery }
    }

    #[inline(always)]
    pub(super) fn from_u32(value: u32) -> Self {
        Self::from_montgomery(PrimeField::<MODULUS>::to_montgomery(value))
    }

    #[inline(always)]
    pub(super) const fn montgomery(self) -> u32 {
        self.montgomery
    }

    #[inline(always)]
    pub(super) const fn set_montgomery(&mut self, montgomery: u32) {
        self.montgomery = montgomery;
    }
}

impl<const MODULUS: u32> PrimeField<MODULUS> {
    /// Reduces a `u64` and constructs a Montgomery-form [`FieldElement`].
    ///
    /// This is the convenient general constructor and is equivalent to
    /// [`FieldElement::new`]. For `u32` coefficients, especially when preparing
    /// NTT buffers, [`Self::element_u32`] avoids a separate remainder.
    #[must_use]
    pub fn element(&self, value: u64) -> FieldElement<MODULUS> {
        FieldElement::new(self, value)
    }

    /// Reduces any `u32` and constructs a Montgomery-form [`FieldElement`].
    ///
    /// The result represents `value mod MODULUS`, even when `value` is not
    /// canonical. Montgomery reduction accepts the full-width input directly,
    /// avoiding the `u64` remainder performed by [`Self::element`]. This is an
    /// `O(1)`, allocation-free conversion suited to coefficient buffers.
    #[must_use]
    pub fn element_u32(&self, value: u32) -> FieldElement<MODULUS> {
        FieldElement::from_u32(value)
    }

    /// Adds `rhs` element-wise into `lhs` in Montgomery representation.
    ///
    /// This takes `O(n)` time, allocates nothing, and leaves each `lhs[i]` as
    /// `lhs[i] + rhs[i]` in the field. Unlike [`Self::add_assign`], these slices
    /// contain [`FieldElement`] rather than canonical `u32` residues. This loop
    /// is scalar on every target. A length mismatch returns
    /// [`FieldError::LengthMismatch`] before mutating `lhs`.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::LengthMismatch`] when the slices have different lengths.
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

    /// Subtracts `rhs` element-wise from `lhs` in Montgomery representation.
    ///
    /// This takes `O(n)` time, allocates nothing, and leaves each `lhs[i]` as
    /// `lhs[i] - rhs[i]` in the field. Unlike [`Self::sub_assign`], these slices
    /// contain [`FieldElement`] rather than canonical `u32` residues. This loop
    /// is scalar on every target. A length mismatch returns
    /// [`FieldError::LengthMismatch`] before mutating `lhs`.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::LengthMismatch`] when the slices have different lengths.
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

    /// Multiplies `lhs` element-wise by `rhs` in Montgomery representation.
    ///
    /// This is the bulk operation used for NTT pointwise products. It takes
    /// O(n) time and allocates nothing. On x86-64, slices of at least
    /// eight elements use a runtime-detected AVX2 auto-vectorized Montgomery
    /// kernel when `MODULUS != 2`; unsupported targets, shorter slices, and
    /// modulus two use the scalar Montgomery kernel. Dispatch depends on public
    /// target, modulus, and length, while arithmetic kernels are designed
    /// without coefficient-dependent branches; this is not a formal
    /// constant-time audit. A length mismatch returns
    /// [`FieldError::LengthMismatch`] before mutating `lhs`.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::LengthMismatch`] when the slices have different lengths.
    pub fn mul_elements_assign(
        &self,
        lhs: &mut [FieldElement<MODULUS>],
        rhs: &[FieldElement<MODULUS>],
    ) -> Result<(), FieldError> {
        if lhs.len() != rhs.len() {
            return Err(FieldError::LengthMismatch);
        }

        #[cfg(target_arch = "x86_64")]
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

    pub(crate) fn mul_element_arrays_assign<const N: usize>(
        lhs: &mut [FieldElement<MODULUS>; N],
        rhs: &[FieldElement<MODULUS>; N],
    ) {
        #[cfg(target_arch = "x86_64")]
        if MODULUS != 2 && N >= 8 && std::arch::is_x86_feature_detected!("avx2") {
            // SAFETY: AVX2 was detected above and arrays have the same length.
            unsafe { mul_elements_assign_avx2(lhs, rhs) };
            return;
        }

        for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
            lhs.montgomery = Self::montgomery_mul(lhs.montgomery, rhs.montgomery);
        }
    }

    /// Multiplies every element in `values` by `scalar` in place.
    ///
    /// This takes `O(n)` time and allocates nothing while retaining Montgomery
    /// representation. On x86-64, slices of at least eight elements use a
    /// runtime-detected AVX2 auto-vectorized Montgomery kernel when
    /// `MODULUS != 2`; other cases are scalar. The implementation is designed
    /// without coefficient-dependent branches in the arithmetic kernels, but it
    /// has not been formally audited as a constant-time implementation.
    pub fn scalar_mul_elements_assign(
        &self,
        values: &mut [FieldElement<MODULUS>],
        scalar: FieldElement<MODULUS>,
    ) {
        #[cfg(target_arch = "x86_64")]
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

/// Branchless scalar Montgomery multiplication without inline assembly.
///
/// The inline-assembly helpers in `constant_time` are opaque to LLVM and
/// prevent loop vectorization, so this kernel expresses the same REDC with
/// portable wrapping arithmetic. Comparisons materialize masks instead of
/// branches, mirroring the `add`/`sbb` and `sub`/`cmov` sequences of the
/// assembly version while letting LLVM schedule vector code. Callers ensure
/// `MODULUS != 2`. This is a code-generation safeguard, not a claim of a
/// formally verified side-channel implementation.
#[inline(always)]
fn montgomery_mul_vectorized<const MODULUS: u32>(lhs: u32, rhs: u32) -> u32 {
    let product = u64::from(lhs) * u64::from(rhs);
    let adjustment = (product as u32).wrapping_mul(PrimeField::<MODULUS>::MONTGOMERY_NEG_INV);
    let sum = product.wrapping_add(u64::from(adjustment) * u64::from(MODULUS));
    // A wrapped sum means the true sum exceeded 64 bits; the carry belongs in
    // bit 32 of the shifted result, matching `add_with_carry_shr_32`.
    let carry = u64::from(sum < product) << 32;
    let reduced = (sum >> 32) | carry;
    reduced.wrapping_sub(u64::from(reduced >= u64::from(MODULUS)) * u64::from(MODULUS)) as u32
}

/// Multiplies `lhs` element-wise by `rhs` with LLVM-vectorized AVX2 code.
///
/// The loop body is branchless portable Rust; marking only this entry point
/// `avx2` lets LLVM emit `vpmuludq`/`vpmulld` Montgomery reduction without
/// handwritten intrinsics. No `unsafe` operations occur beyond entering the
/// `target_feature` context.
///
/// # Safety
///
/// AVX2 must be supported by the target CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn mul_elements_assign_avx2<const MODULUS: u32>(
    lhs: &mut [FieldElement<MODULUS>],
    rhs: &[FieldElement<MODULUS>],
) {
    for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
        lhs.montgomery = montgomery_mul_vectorized::<MODULUS>(lhs.montgomery, rhs.montgomery);
    }
}

/// Multiplies every element by `scalar` with LLVM-vectorized AVX2 code.
///
/// # Safety
///
/// AVX2 must be supported by the target CPU.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn scalar_mul_elements_assign_avx2<const MODULUS: u32>(
    values: &mut [FieldElement<MODULUS>],
    scalar: FieldElement<MODULUS>,
) {
    for value in values {
        value.montgomery =
            montgomery_mul_vectorized::<MODULUS>(value.montgomery, scalar.montgomery);
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
