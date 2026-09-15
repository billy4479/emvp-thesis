use crate::constant_time::reduce_once_u64;

use super::{FieldError, PrimeField};

const PSEUDO_MERSENNE_32_MODULUS: u32 = 4_294_967_291;
const PSEUDO_MERSENNE_CHUNK_ELEMENTS: usize = 1_024;
const PSEUDO_MERSENNE_32_MODULUS_U64: u64 = PSEUDO_MERSENNE_32_MODULUS as u64;

/// Computes the exact dot product of two slices into a `u64`.
///
/// The caller proves that the complete dot product fits in `u64`, so no
/// accumulator can overflow. The kernel stays out of line: inlining it into
/// `dot_canonical` lets LLVM see the caller's length guard and it then
/// replaces the clean widening-multiply loop with a much slower
/// carry-tracking vectorization.
#[inline(never)]
fn dot_u64(lhs: &[u32], rhs: &[u32]) -> u64 {
    let mut sum = 0u64;
    for (&lhs, &rhs) in lhs.iter().zip(rhs) {
        sum += u64::from(lhs) * u64::from(rhs);
    }
    sum
}

/// Computes a dot product modulo `2^32 - 5` using `2^32 = 5 (mod p)`.
///
/// The slices must have equal lengths. The low and high halves of each
/// product accumulate separately, which keeps `product >> 32` a pure shift
/// inside the vectorized loop; both halves stay below `2^42` per chunk, so
/// the end-of-chunk fold `low + 5 * high` stays below `2^45` and the chunk
/// reduction keeps every later addition far from overflowing.
fn dot_pseudo_mersenne_32(lhs: &[u32], rhs: &[u32]) -> u32 {
    let mut sum = 0u64;
    // An opaque shift amount stops LLVM from rewriting `product >> 32` into
    // a shuffle-heavy 32x32 `mulhi`; the loop still vectorizes, lowering the
    // shift as `vpsrlvq`.
    let high_shift = std::hint::black_box(32);
    for (lhs_chunk, rhs_chunk) in lhs
        .chunks(PSEUDO_MERSENNE_CHUNK_ELEMENTS)
        .zip(rhs.chunks(PSEUDO_MERSENNE_CHUNK_ELEMENTS))
    {
        let mut low_sum = 0u64;
        let mut high_sum = 0u64;
        for (&lhs, &rhs) in lhs_chunk.iter().zip(rhs_chunk) {
            let product = u64::from(lhs) * u64::from(rhs);
            low_sum += product & u64::from(u32::MAX);
            high_sum += product >> high_shift;
        }
        sum = reduce_pseudo_mersenne_chunk(
            sum + reduce_pseudo_mersenne_chunk(low_sum + 5 * high_sum),
        );
    }
    reduce_pseudo_mersenne_chunk(sum) as u32
}

/// Reduces the values produced by one 1024-element accumulator chunk.
#[inline(always)]
fn reduce_pseudo_mersenne_chunk(value: u64) -> u64 {
    let folded = (value & u64::from(u32::MAX)) + 5 * (value >> 32);
    reduce_once_u64(folded, PSEUDO_MERSENNE_32_MODULUS_U64)
}

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
    /// `MODULUS = 2^32 - 5` folds each product with `2^32 = 5 (mod MODULUS)`
    /// in bounded chunks; the other cases accumulate into an exact wide
    /// accumulator that fits when the length bound proves it. LLVM
    /// auto-vectorizes both loops without handwritten intrinsics.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::LengthMismatch`] when the slices have different lengths.
    pub fn dot_canonical(&self, lhs: &[u32], rhs: &[u32]) -> Result<u32, FieldError> {
        if lhs.len() != rhs.len() {
            return Err(FieldError::LengthMismatch);
        }

        if MODULUS == PSEUDO_MERSENNE_32_MODULUS {
            return Ok(dot_pseudo_mersenne_32(lhs, rhs));
        }

        let max_product = u64::from(MODULUS - 1) * u64::from(MODULUS - 1);
        if lhs.len() as u128 * u128::from(max_product) <= u128::from(u64::MAX) {
            return Ok(self.reduce_u64(dot_u64(lhs, rhs)));
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
    /// Inputs may use any `u32` representation. Successful output is
    /// canonical. The Montgomery images of the inputs are retained in one
    /// scratch `Vec` so the reverse pass does not recompute them; this is the
    /// operation's only allocation.
    ///
    /// The zero scan exits early at the first zero operand, so the running
    /// time of a rejected call reveals the position of the first zero. This
    /// batch inversion is variable-time by design; do not use it when that
    /// timing must be hidden.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::DivisionByZero`] when any value is zero modulo
    /// `MODULUS`. The check completes before any value is changed.
    pub fn batch_inv_assign(&self, values: &mut [u32]) -> Result<(), FieldError> {
        if values.is_empty() {
            return Ok(());
        }

        let mut montgomery = Vec::with_capacity(values.len());
        let mut prefixes = Vec::with_capacity(values.len());
        let mut product = Self::MONTGOMERY_ONE;
        for &value in values.iter() {
            let value = Self::to_montgomery(value);
            if value == 0 {
                return Err(FieldError::DivisionByZero);
            }
            product = Self::montgomery_mul_scalar(product, value);
            prefixes.push(product);
            montgomery.push(value);
        }

        let mut inverse = Self::pow_montgomery(product, u64::from(MODULUS - 2));
        for index in (0..values.len()).rev() {
            let previous = if index == 0 {
                Self::MONTGOMERY_ONE
            } else {
                prefixes[index - 1]
            };
            values[index] = Self::from_montgomery(Self::montgomery_mul_scalar(inverse, previous));
            inverse = Self::montgomery_mul_scalar(inverse, montgomery[index]);
        }
        Ok(())
    }
}
