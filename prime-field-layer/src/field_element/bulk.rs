use super::FieldElement;
use crate::{FieldError, PrimeField};

impl<const MODULUS: u32> PrimeField<MODULUS> {
    /// Adds `rhs` element-wise into `lhs` in Montgomery representation.
    ///
    /// This takes `O(n)` time, allocates nothing, and leaves each `lhs[i]` as
    /// `lhs[i] + rhs[i]` in the field. Unlike
    /// [`Self::add_assign_canonical`], these slices contain [`FieldElement`]
    /// rather than canonical `u32` residues. This loop is scalar on every
    /// target. A length mismatch returns
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
            lhs.montgomery = self.add_canonical(lhs.montgomery, rhs.montgomery);
        }
        Ok(())
    }

    /// Subtracts `rhs` element-wise from `lhs` in Montgomery representation.
    ///
    /// This takes `O(n)` time, allocates nothing, and leaves each `lhs[i]` as
    /// `lhs[i] - rhs[i]` in the field. Unlike
    /// [`Self::sub_assign_canonical`], these slices contain [`FieldElement`]
    /// rather than canonical `u32` residues. This loop is scalar on every
    /// target. A length mismatch returns
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
            lhs.montgomery = self.sub_canonical(lhs.montgomery, rhs.montgomery);
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
