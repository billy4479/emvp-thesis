use crate::constant_time::reduce_once_u64;

use super::{FieldError, PrimeField};

impl<const MODULUS: u32> PrimeField<MODULUS> {
    #[inline(always)]
    fn reduce_u128(self, value: u128) -> u32 {
        let high = u64::from(self.reduce_u64((value >> 64) as u64));
        let low = u64::from(self.reduce_u64(value as u64));
        self.reduce_u64(high * u64::from(Self::MONTGOMERY_R2) + low)
    }

    /// Adds canonical `rhs` residues element-wise into canonical `lhs` residues.
    ///
    /// Every input must be less than `MODULUS`. Results are canonical and the
    /// operation allocates no memory.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::LengthMismatch`] when the slices have different lengths.
    pub fn add_assign_canonical(&self, lhs: &mut [u32], rhs: &[u32]) -> Result<(), FieldError> {
        if lhs.len() != rhs.len() {
            return Err(FieldError::LengthMismatch);
        }

        for (lhs, &rhs) in lhs.iter_mut().zip(rhs) {
            *lhs = self.add_canonical(*lhs, rhs);
        }
        Ok(())
    }

    /// Subtracts canonical `rhs` residues from canonical `lhs` residues.
    ///
    /// Every input must be less than `MODULUS`. Results are canonical and the
    /// operation allocates no memory.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::LengthMismatch`] when the slices have different lengths.
    pub fn sub_assign_canonical(&self, lhs: &mut [u32], rhs: &[u32]) -> Result<(), FieldError> {
        if lhs.len() != rhs.len() {
            return Err(FieldError::LengthMismatch);
        }

        for (lhs, &rhs) in lhs.iter_mut().zip(rhs) {
            *lhs = self.sub_canonical(*lhs, rhs);
        }
        Ok(())
    }

    /// Multiplies arbitrary `u32` values element-wise and canonicalizes `lhs`.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::LengthMismatch`] when the slices have different lengths.
    pub fn mul_assign(&self, lhs: &mut [u32], rhs: &[u32]) -> Result<(), FieldError> {
        if lhs.len() != rhs.len() {
            return Err(FieldError::LengthMismatch);
        }

        for (lhs, &rhs) in lhs.iter_mut().zip(rhs) {
            *lhs = self.mul(*lhs, rhs);
        }
        Ok(())
    }

    /// Negates canonical residues in place.
    ///
    /// Every input must be less than `MODULUS`. Results are canonical.
    pub fn neg_assign_canonical(&self, values: &mut [u32]) {
        for value in values {
            *value = self.neg_canonical(*value);
        }
    }

    /// Multiplies canonical residues by a canonical scalar in place.
    ///
    /// Every input and `scalar` must be less than `MODULUS`. Results are
    /// canonical.
    pub fn scalar_mul_assign_canonical(&self, values: &mut [u32], scalar: u32) {
        let shoup = (u64::from(scalar) << 32) / Self::MODULUS_U64;
        for value in values {
            let product = u64::from(*value) * u64::from(scalar);
            let quotient = (u64::from(*value) * shoup) >> 32;
            let remainder = product - quotient * Self::MODULUS_U64;
            *value = reduce_once_u64(remainder, Self::MODULUS_U64) as u32;
        }
    }

    /// Computes the dot product of two canonical-residue slices.
    ///
    /// Every input must be less than `MODULUS`. The result is canonical.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::LengthMismatch`] when the slices have different lengths.
    pub fn dot_canonical(&self, lhs: &[u32], rhs: &[u32]) -> Result<u32, FieldError> {
        if lhs.len() != rhs.len() {
            return Err(FieldError::LengthMismatch);
        }

        let max_product = u64::from(MODULUS - 1) * u64::from(MODULUS - 1);
        if lhs.len() as u128 * u128::from(max_product) <= u128::from(u64::MAX) {
            let mut sums = [0u64; 4];
            for (lhs, rhs) in lhs.chunks_exact(4).zip(rhs.chunks_exact(4)) {
                sums[0] += u64::from(lhs[0]) * u64::from(rhs[0]);
                sums[1] += u64::from(lhs[1]) * u64::from(rhs[1]);
                sums[2] += u64::from(lhs[2]) * u64::from(rhs[2]);
                sums[3] += u64::from(lhs[3]) * u64::from(rhs[3]);
            }

            let remainder_start = lhs.len() / 4 * 4;
            for (&lhs, &rhs) in lhs[remainder_start..].iter().zip(&rhs[remainder_start..]) {
                sums[0] += u64::from(lhs) * u64::from(rhs);
            }
            return Ok(self.reduce_u64(sums.into_iter().sum()));
        }

        let mut low = [0u64; 4];
        let mut high = [0u64; 4];
        for (lhs, rhs) in lhs.chunks_exact(4).zip(rhs.chunks_exact(4)) {
            let mut lane = 0;
            while lane < 4 {
                let product = u64::from(lhs[lane]) * u64::from(rhs[lane]);
                let (sum, carry) = low[lane].overflowing_add(product);
                low[lane] = sum;
                high[lane] += u64::from(carry);
                lane += 1;
            }
        }

        let remainder_start = lhs.len() / 4 * 4;
        for (&lhs, &rhs) in lhs[remainder_start..].iter().zip(&rhs[remainder_start..]) {
            let product = u64::from(lhs) * u64::from(rhs);
            let (sum, carry) = low[0].overflowing_add(product);
            low[0] = sum;
            high[0] += u64::from(carry);
        }

        let mut sum = 0u128;
        let mut lane = 0;
        while lane < 4 {
            sum += u128::from(high[lane]) << 64 | u128::from(low[lane]);
            lane += 1;
        }
        Ok(self.reduce_u128(sum))
    }

    /// Inverts all values using one exponentiation and approximately three
    /// multiplications per value.
    ///
    /// Inputs may use any `u32` representation. Successful output is canonical.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::DivisionByZero`] when any value is zero modulo
    /// `MODULUS`. The check completes before any value is changed.
    pub fn batch_inv_assign(&self, values: &mut [u32]) -> Result<(), FieldError> {
        if values.is_empty() {
            return Ok(());
        }
        if MODULUS == 2 {
            if values.iter().any(|value| value & 1 == 0) {
                return Err(FieldError::DivisionByZero);
            }
            // MONTGOMERY_ONE is zero for this modulus, so the Montgomery
            // seeding below cannot represent the identity.
            values.fill(1);
            return Ok(());
        }

        let mut prefixes = Vec::with_capacity(values.len());
        let mut product = Self::MONTGOMERY_ONE;
        for &value in values.iter() {
            let value = Self::to_montgomery(value);
            if value == 0 {
                return Err(FieldError::DivisionByZero);
            }
            product = Self::montgomery_mul(product, value);
            prefixes.push(product);
        }

        let mut inverse = Self::pow_montgomery(product, u64::from(MODULUS - 2));
        for index in (0..values.len()).rev() {
            let previous = if index == 0 {
                Self::MONTGOMERY_ONE
            } else {
                prefixes[index - 1]
            };
            let value = Self::to_montgomery(values[index]);
            values[index] = Self::from_montgomery(Self::montgomery_mul(inverse, previous));
            inverse = Self::montgomery_mul(inverse, value);
        }
        Ok(())
    }
}
