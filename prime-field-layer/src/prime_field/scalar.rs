use crate::constant_time::reduce_once_u64;

use super::{FieldError, PrimeField};

impl<const MODULUS: u32> PrimeField<MODULUS> {
    /// Adds two canonical residues.
    ///
    /// Both operands must be less than `MODULUS`. The result is canonical.
    #[inline(always)]
    #[must_use]
    pub fn add_canonical(&self, lhs: u32, rhs: u32) -> u32 {
        reduce_once_u64(u64::from(lhs) + u64::from(rhs), Self::MODULUS_U64) as u32
    }

    /// Subtracts one canonical residue from another.
    ///
    /// Both operands must be less than `MODULUS`. The result is canonical.
    #[inline(always)]
    #[must_use]
    pub fn sub_canonical(&self, lhs: u32, rhs: u32) -> u32 {
        reduce_once_u64(
            u64::from(lhs) + Self::MODULUS_U64 - u64::from(rhs),
            Self::MODULUS_U64,
        ) as u32
    }

    /// Negates a canonical residue.
    ///
    /// `value` must be less than `MODULUS`. The result is canonical.
    #[inline(always)]
    #[must_use]
    pub fn neg_canonical(&self, value: u32) -> u32 {
        let negated = MODULUS - value;
        negated & 0u32.wrapping_sub(u32::from(value != 0))
    }

    /// Multiplies two arbitrary `u32` values and returns a canonical residue.
    #[inline(always)]
    #[must_use]
    pub fn mul(&self, lhs: u32, rhs: u32) -> u32 {
        (u64::from(lhs) * u64::from(rhs) % Self::MODULUS_U64) as u32
    }

    /// Squares an arbitrary `u32` value and returns a canonical residue.
    #[inline(always)]
    #[must_use]
    pub fn square(&self, value: u32) -> u32 {
        self.mul(value, value)
    }

    /// Branchless Montgomery multiplication.
    ///
    /// This REDC uses portable wrapping arithmetic and mask selections with
    /// no coefficient-dependent branches, so auto-vectorized loops can
    /// schedule the whole body with vector compares and multiplies.
    /// Latency-sensitive scalar operations use
    /// [`Self::montgomery_mul_scalar`] instead.
    ///
    /// Bounds: callers pass canonical Montgomery operands `lhs`, `rhs` below
    /// `p` (as produced by [`Self::to_montgomery`] and every Montgomery-form
    /// operation), so the product is below `p^2 < R * p` with `R = 2^32 > p`
    /// and the correction `m * p` below `R * p`. The wrapping sum is
    /// therefore below `2 * R * p`; the carry out of bit 63 belongs in
    /// bit 32 of the shifted result, matching an `add`/`sbb` sequence, and
    /// the shifted value is below `2p`, so one masked subtraction restores
    /// `[0, p)`.
    #[inline(always)]
    pub(crate) fn montgomery_mul(lhs: u32, rhs: u32) -> u32 {
        if MODULUS == 2 {
            return lhs & rhs;
        }

        let product = u64::from(lhs) * u64::from(rhs);
        let adjustment = (product as u32).wrapping_mul(Self::MONTGOMERY_NEG_INV);
        let sum = product.wrapping_add(u64::from(adjustment) * u64::from(MODULUS));
        let carry = u64::from(sum < product) << 32;
        let reduced = (sum >> 32) | carry;
        reduced.wrapping_sub(u64::from(reduced >= u64::from(MODULUS)) * u64::from(MODULUS)) as u32
    }

    /// Scalar Montgomery multiplication expressed to favor conditional moves.
    #[inline(always)]
    pub(crate) fn montgomery_mul_scalar(lhs: u32, rhs: u32) -> u32 {
        if MODULUS == 2 {
            return lhs & rhs;
        }

        let product = u64::from(lhs) * u64::from(rhs);
        let adjustment = (product as u32).wrapping_mul(Self::MONTGOMERY_NEG_INV);
        let (sum, carry) = product.overflowing_add(u64::from(adjustment) * u64::from(MODULUS));
        let carry_mask = 0u64.wrapping_sub(u64::from(carry));
        let reduced = (sum >> 32) | (carry_mask & (1u64 << 32));
        reduce_once_u64(reduced, u64::from(MODULUS)) as u32
    }

    #[inline(always)]
    pub(crate) fn to_montgomery(value: u32) -> u32 {
        if MODULUS == 2 {
            return value & 1;
        }
        // REDC accepts this full-width u32 directly: R2 < p and value < R,
        // so value * R2 < R * p. Canonicalization before conversion is not
        // required.
        Self::montgomery_mul_scalar(value, Self::MONTGOMERY_R2)
    }

    #[inline(always)]
    pub(crate) fn from_montgomery(value: u32) -> u32 {
        if MODULUS == 2 {
            return value;
        }
        Self::montgomery_mul_scalar(value, 1)
    }

    pub(crate) fn pow_montgomery(mut base: u32, mut exponent: u64) -> u32 {
        if MODULUS == 2 {
            return if exponent == 0 { 1 } else { base };
        }

        let mut result = Self::MONTGOMERY_ONE;
        while exponent != 0 {
            if exponent & 1 == 1 {
                result = Self::montgomery_mul_scalar(result, base);
            }
            exponent >>= 1;
            if exponent != 0 {
                base = Self::montgomery_mul_scalar(base, base);
            }
        }
        result
    }

    /// Raises an arbitrary `u32` value to `exponent` and returns a canonical residue.
    #[must_use]
    pub fn pow(&self, base: u32, exponent: u64) -> u32 {
        let base = Self::to_montgomery(base);
        Self::from_montgomery(Self::pow_montgomery(base, exponent))
    }

    /// Returns the multiplicative inverse of an arbitrary `u32` representation.
    ///
    /// The result is canonical. Every representable multiple of `MODULUS` is a
    /// representation of zero and is rejected.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::DivisionByZero`] when `value mod MODULUS` is zero.
    pub fn inv(&self, value: u32) -> Result<u32, FieldError> {
        let value = Self::to_montgomery(value);
        if value == 0 {
            return Err(FieldError::DivisionByZero);
        }
        Ok(Self::from_montgomery(Self::pow_montgomery(
            value,
            u64::from(MODULUS - 2),
        )))
    }

    /// Returns an element of exact order `length`.
    ///
    /// Supported lengths are powers of two dividing `MODULUS - 1`; length one
    /// has root one.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::UnsupportedTransformLength`] when `length` is not
    /// a supported power of two.
    pub fn root_of_unity(&self, length: usize) -> Result<u32, FieldError> {
        if !length.is_power_of_two() || length.trailing_zeros() > self.two_adicity() {
            return Err(FieldError::UnsupportedTransformLength(length));
        }

        let mut root = Self::TWO_ADIC_ROOT;
        for _ in length.trailing_zeros()..self.two_adicity() {
            root = self.square(root);
        }
        Ok(root)
    }
}
