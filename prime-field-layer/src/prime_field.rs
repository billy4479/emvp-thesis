use std::fmt;

use crate::constant_time::{add_with_carry_shr_32, reduce_once_u64};

/// Errors caused by invalid field parameters or operands.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FieldError {
    DivisionByZero,
    LengthMismatch,
    UnsupportedTransformLength(usize),
    PlanTooSmall { required: usize, available: usize },
    ConvolutionLengthOverflow,
    Avx2Unavailable,
}

impl fmt::Display for FieldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DivisionByZero => formatter.write_str("zero has no multiplicative inverse"),
            Self::LengthMismatch => formatter.write_str("slice lengths do not match"),
            Self::UnsupportedTransformLength(length) => {
                write!(
                    formatter,
                    "the field does not support transform length {length}"
                )
            }
            Self::PlanTooSmall {
                required,
                available,
            } => write!(
                formatter,
                "convolution requires transform length {required}, but the plan length is {available}"
            ),
            Self::ConvolutionLengthOverflow => formatter
                .write_str("convolution result or required transform length does not fit in usize"),
            Self::Avx2Unavailable => {
                formatter.write_str("AVX2 NTT butterflies are unavailable for this plan")
            }
        }
    }
}

impl std::error::Error for FieldError {}

/// Arithmetic for canonical residues modulo the compile-time prime `MODULUS`.
///
/// Invalid moduli are rejected during compilation:
///
/// ```compile_fail
/// let _ = prime_field_layer::PrimeField::<15>::new();
/// ```
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct PrimeField<const MODULUS: u32> {
    _private: (),
}

impl<const MODULUS: u32> Default for PrimeField<MODULUS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const MODULUS: u32> PrimeField<MODULUS> {
    const VALID_MODULUS: () = assert!(Self::is_prime(MODULUS), "field modulus must be prime");
    const MODULUS_U64: u64 = MODULUS as u64;
    pub(crate) const MONTGOMERY_NEG_INV: u32 = Self::montgomery_neg_inv();
    const MONTGOMERY_R2: u32 = if MODULUS <= 1 {
        0
    } else {
        ((1u128 << 64) % MODULUS as u128) as u32
    };
    const MONTGOMERY_ONE: u32 = if MODULUS <= 1 {
        0
    } else {
        ((1u64 << 32) % MODULUS as u64) as u32
    };
    const TWO_ADIC_ROOT: u32 = Self::calculate_two_adic_root();

    const fn modular_pow(mut base: u64, mut exponent: u32, modulus: u64) -> u64 {
        let mut result = 1;
        while exponent != 0 {
            if exponent & 1 == 1 {
                result = result * base % modulus;
            }
            exponent >>= 1;
            if exponent != 0 {
                base = base * base % modulus;
            }
        }
        result
    }

    const fn is_prime(value: u32) -> bool {
        const SMALL_PRIMES: [u32; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];

        let mut index = 0;
        while index < SMALL_PRIMES.len() {
            let prime = SMALL_PRIMES[index];
            if value.is_multiple_of(prime) {
                return value == prime;
            }
            index += 1;
        }
        if value < 2 {
            return false;
        }

        let shifts = (value - 1).trailing_zeros();
        let odd_part = (value - 1) >> shifts;
        let modulus = value as u64;
        let bases = [2u64, 7, 61];
        let mut base_index = 0;

        while base_index < bases.len() {
            let base = bases[base_index];
            base_index += 1;
            if base >= modulus {
                continue;
            }

            let mut power = Self::modular_pow(base, odd_part, modulus);
            if power == 1 || power == modulus - 1 {
                continue;
            }

            let mut shift = 1;
            let mut passed = false;
            while shift < shifts {
                power = power * power % modulus;
                if power == modulus - 1 {
                    passed = true;
                    break;
                }
                shift += 1;
            }
            if !passed {
                return false;
            }
        }

        true
    }

    const fn montgomery_neg_inv() -> u32 {
        if MODULUS <= 2 {
            return 0;
        }

        let mut inverse = MODULUS;
        let mut iteration = 0;
        while iteration < 5 {
            inverse = inverse.wrapping_mul(2u32.wrapping_sub(MODULUS.wrapping_mul(inverse)));
            iteration += 1;
        }
        inverse.wrapping_neg()
    }

    const fn calculate_two_adic_root() -> u32 {
        if MODULUS == 2 {
            return 1;
        }
        if (MODULUS - 1).trailing_zeros() == 1 {
            return MODULUS - 1;
        }

        let mut non_residue = 2u64;
        while Self::modular_pow(non_residue, (MODULUS - 1) / 2, MODULUS as u64) == 1 {
            non_residue += 1;
        }
        Self::modular_pow(
            non_residue,
            (MODULUS - 1) >> (MODULUS - 1).trailing_zeros(),
            MODULUS as u64,
        ) as u32
    }

    /// Constructs the zero-sized field value.
    ///
    /// Compilation fails when `MODULUS` is not prime.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        let () = Self::VALID_MODULUS;
        Self { _private: () }
    }

    pub(crate) const fn assume_valid() -> Self {
        Self { _private: () }
    }

    #[must_use]
    pub const fn modulus(&self) -> u32 {
        MODULUS
    }

    /// Returns the exponent of two in the factorization of `MODULUS - 1`.
    #[must_use]
    pub const fn two_adicity(&self) -> u32 {
        (MODULUS - 1).trailing_zeros()
    }

    /// Reduces an arbitrary 64-bit integer to a canonical residue.
    #[inline(always)]
    #[must_use]
    pub const fn reduce_u64(&self, value: u64) -> u32 {
        (value % Self::MODULUS_U64) as u32
    }

    #[inline(always)]
    fn reduce_u128(self, value: u128) -> u32 {
        let high = u64::from(self.reduce_u64((value >> 64) as u64));
        let low = u64::from(self.reduce_u64(value as u64));
        self.reduce_u64(high * u64::from(Self::MONTGOMERY_R2) + low)
    }

    #[inline(always)]
    #[must_use]
    pub fn add(&self, lhs: u32, rhs: u32) -> u32 {
        reduce_once_u64(u64::from(lhs) + u64::from(rhs), Self::MODULUS_U64) as u32
    }

    #[inline(always)]
    #[must_use]
    pub fn sub(&self, lhs: u32, rhs: u32) -> u32 {
        reduce_once_u64(
            u64::from(lhs) + Self::MODULUS_U64 - u64::from(rhs),
            Self::MODULUS_U64,
        ) as u32
    }

    #[inline(always)]
    #[must_use]
    pub fn neg(&self, value: u32) -> u32 {
        let negated = MODULUS - value;
        negated & 0u32.wrapping_sub(u32::from(value != 0))
    }

    #[inline(always)]
    #[must_use]
    pub fn mul(&self, lhs: u32, rhs: u32) -> u32 {
        (u64::from(lhs) * u64::from(rhs) % Self::MODULUS_U64) as u32
    }

    #[inline(always)]
    #[must_use]
    pub fn square(&self, value: u32) -> u32 {
        self.mul(value, value)
    }

    #[inline(always)]
    pub(crate) fn montgomery_mul(lhs: u32, rhs: u32) -> u32 {
        if MODULUS == 2 {
            return lhs & rhs;
        }

        let product = u64::from(lhs) * u64::from(rhs);
        let adjustment = (product as u32).wrapping_mul(Self::MONTGOMERY_NEG_INV);
        let reduced = add_with_carry_shr_32(product, u64::from(adjustment) * Self::MODULUS_U64);

        reduce_once_u64(reduced, Self::MODULUS_U64) as u32
    }

    #[inline(always)]
    pub(crate) fn to_montgomery(value: u32) -> u32 {
        if MODULUS == 2 {
            return value & 1;
        }
        // REDC accepts this full-width u32 directly: R2 < p and value < R,
        // so value * R2 < R * p. Canonicalization before conversion is not
        // required.
        Self::montgomery_mul(value, Self::MONTGOMERY_R2)
    }

    #[inline(always)]
    pub(crate) fn from_montgomery(value: u32) -> u32 {
        if MODULUS == 2 {
            return value;
        }
        Self::montgomery_mul(value, 1)
    }

    pub(crate) fn pow_montgomery(mut base: u32, mut exponent: u64) -> u32 {
        if MODULUS == 2 {
            return if exponent == 0 { 1 } else { base };
        }

        let mut result = Self::MONTGOMERY_ONE;
        while exponent != 0 {
            if exponent & 1 == 1 {
                result = Self::montgomery_mul(result, base);
            }
            exponent >>= 1;
            if exponent != 0 {
                base = Self::montgomery_mul(base, base);
            }
        }
        result
    }

    #[must_use]
    pub fn pow(&self, base: u32, exponent: u64) -> u32 {
        let base = Self::to_montgomery(base);
        Self::from_montgomery(Self::pow_montgomery(base, exponent))
    }

    /// Returns the multiplicative inverse of `value`.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::DivisionByZero`] when `value` is zero.
    pub fn inv(&self, value: u32) -> Result<u32, FieldError> {
        if value == 0 {
            return Err(FieldError::DivisionByZero);
        }
        Ok(self.pow(value, u64::from(MODULUS - 2)))
    }

    /// Adds `rhs` element-wise into `lhs` without allocating.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::LengthMismatch`] when the slices have different lengths.
    pub fn add_assign(&self, lhs: &mut [u32], rhs: &[u32]) -> Result<(), FieldError> {
        if lhs.len() != rhs.len() {
            return Err(FieldError::LengthMismatch);
        }

        for (lhs, &rhs) in lhs.iter_mut().zip(rhs) {
            *lhs = self.add(*lhs, rhs);
        }
        Ok(())
    }

    /// Subtracts `rhs` element-wise from `lhs` without allocating.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::LengthMismatch`] when the slices have different lengths.
    pub fn sub_assign(&self, lhs: &mut [u32], rhs: &[u32]) -> Result<(), FieldError> {
        if lhs.len() != rhs.len() {
            return Err(FieldError::LengthMismatch);
        }

        for (lhs, &rhs) in lhs.iter_mut().zip(rhs) {
            *lhs = self.sub(*lhs, rhs);
        }
        Ok(())
    }

    /// Multiplies `lhs` element-wise by `rhs` without allocating.
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

    pub fn neg_assign(&self, values: &mut [u32]) {
        for value in values {
            *value = self.neg(*value);
        }
    }

    pub fn scalar_mul_assign(&self, values: &mut [u32], scalar: u32) {
        let shoup = (u64::from(scalar) << 32) / Self::MODULUS_U64;
        for value in values {
            let product = u64::from(*value) * u64::from(scalar);
            let quotient = (u64::from(*value) * shoup) >> 32;
            let remainder = product - quotient * Self::MODULUS_U64;
            *value = reduce_once_u64(remainder, Self::MODULUS_U64) as u32;
        }
    }

    /// Computes the dot product of two slices.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::LengthMismatch`] when the slices have different lengths.
    pub fn dot(&self, lhs: &[u32], rhs: &[u32]) -> Result<u32, FieldError> {
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
    /// # Errors
    ///
    /// Returns [`FieldError::DivisionByZero`] when any value is zero.
    pub fn batch_inv_assign(&self, values: &mut [u32]) -> Result<(), FieldError> {
        if values.contains(&0) {
            return Err(FieldError::DivisionByZero);
        }
        if values.is_empty() {
            return Ok(());
        }
        if MODULUS == 2 {
            // MONTGOMERY_ONE is zero for this modulus, so the Montgomery
            // seeding below cannot represent the identity.
            values.fill(1);
            return Ok(());
        }

        let mut prefixes = Vec::with_capacity(values.len());
        let mut product = Self::MONTGOMERY_ONE;
        for &value in values.iter() {
            product = Self::montgomery_mul(product, Self::to_montgomery(value));
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
