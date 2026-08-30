use super::FieldElement;
use crate::{FieldError, PrimeField};

impl<const MODULUS: u32> PrimeField<MODULUS> {
    /// Writes the multiplicative inverses of `values` to `output`.
    ///
    /// This uses one field inversion and `3(n - 1)` multiplications. The
    /// output slice doubles as prefix-product storage, so the operation
    /// allocates nothing. Length and zero checks complete before `output` is
    /// mutated.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::LengthMismatch`] if the slices differ in length,
    /// or [`FieldError::DivisionByZero`] if any input is zero.
    pub fn batch_inv_elements(
        &self,
        values: &[FieldElement<MODULUS>],
        output: &mut [FieldElement<MODULUS>],
    ) -> Result<(), FieldError> {
        if values.len() != output.len() {
            return Err(FieldError::LengthMismatch);
        }
        if values.iter().any(|value| value.montgomery == 0) {
            return Err(FieldError::DivisionByZero);
        }
        let Some((&first, rest)) = values.split_first() else {
            return Ok(());
        };

        output[0] = first;
        for (index, &value) in rest.iter().enumerate() {
            output[index + 1] = output[index] * value;
        }

        let mut inverse = output[output.len() - 1].inv()?;
        for index in (1..values.len()).rev() {
            let prefix = output[index - 1];
            output[index] = inverse * prefix;
            inverse *= values[index];
        }
        output[0] = inverse;
        Ok(())
    }

    /// Adds `rhs` element-wise into `lhs` in Montgomery representation.
    ///
    /// This takes `O(n)` time, allocates nothing, and leaves each `lhs[i]` as
    /// `lhs[i] + rhs[i]` in the field. Unlike
    /// [`Self::add_assign_canonical`], these slices contain [`FieldElement`]
    /// rather than canonical `u32` residues. A length mismatch returns
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
    /// rather than canonical `u32` residues. A length mismatch returns
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
    /// O(n) time and allocates nothing. The loop body is branchless portable
    /// Rust over the Montgomery kernel, so compilers may auto-vectorize it on
    /// targets with suitable instruction sets. Dispatch depends on public
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

        for (lhs, rhs) in lhs.iter_mut().zip(rhs) {
            lhs.montgomery = Self::montgomery_mul(lhs.montgomery, rhs.montgomery);
        }
        Ok(())
    }

    /// Multiplies every element in `values` by `scalar` in place.
    ///
    /// This takes `O(n)` time and allocates nothing while retaining Montgomery
    /// representation. The loop body is branchless portable Rust over the
    /// Montgomery kernel, so compilers may auto-vectorize it on targets with
    /// suitable instruction sets. The implementation is designed
    /// without coefficient-dependent branches in the arithmetic kernels, but it
    /// has not been formally audited as a constant-time implementation.
    pub fn scalar_mul_elements_assign(
        &self,
        values: &mut [FieldElement<MODULUS>],
        scalar: FieldElement<MODULUS>,
    ) {
        for value in values {
            value.montgomery = Self::montgomery_mul(value.montgomery, scalar.montgomery);
        }
    }
}
