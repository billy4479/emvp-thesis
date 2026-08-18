use std::fmt;

/// Errors caused by invalid field parameters or operands.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FieldError {
    DivisionByZero,
    LengthMismatch,
    InvalidTransformLength(usize),
}

impl fmt::Display for FieldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DivisionByZero => formatter.write_str("zero has no multiplicative inverse"),
            Self::LengthMismatch => formatter.write_str("slice lengths do not match"),
            Self::InvalidTransformLength(length) => {
                write!(
                    formatter,
                    "the field does not support transform length {length}"
                )
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

        let mut non_residue = 2;
        while Self::modular_pow(non_residue as u64, (MODULUS - 1) / 2, MODULUS as u64) == 1 {
            non_residue += 1;
        }
        Self::modular_pow(
            non_residue as u64,
            (MODULUS - 1) >> (MODULUS - 1).trailing_zeros(),
            MODULUS as u64,
        ) as u32
    }

    /// Constructs the zero-sized field value.
    ///
    /// Compilation fails when `MODULUS` is not prime.
    #[inline(always)]
    pub const fn new() -> Self {
        let () = Self::VALID_MODULUS;
        Self { _private: () }
    }

    pub(crate) const fn assume_valid() -> Self {
        Self { _private: () }
    }

    pub const fn modulus(&self) -> u32 {
        MODULUS
    }

    /// Returns the exponent of two in the factorization of `MODULUS - 1`.
    pub const fn two_adicity(&self) -> u32 {
        (MODULUS - 1).trailing_zeros()
    }

    /// Reduces an arbitrary 64-bit integer to a canonical residue.
    #[inline(always)]
    pub fn reduce_u64(&self, value: u64) -> u32 {
        (value % Self::MODULUS_U64) as u32
    }

    #[inline(always)]
    fn reduce_u128(&self, value: u128) -> u32 {
        let high = self.reduce_u64((value >> 64) as u64) as u64;
        let low = self.reduce_u64(value as u64) as u64;
        self.reduce_u64(high * Self::MONTGOMERY_R2 as u64 + low)
    }

    #[inline(always)]
    pub fn add(&self, lhs: u32, rhs: u32) -> u32 {
        let (sum, carry) = lhs.overflowing_add(rhs);
        let (reduced, borrow) = sum.overflowing_sub(MODULUS);
        if carry || !borrow { reduced } else { sum }
    }

    #[inline(always)]
    pub fn sub(&self, lhs: u32, rhs: u32) -> u32 {
        let (difference, underflow) = lhs.overflowing_sub(rhs);
        difference.wrapping_add(MODULUS & 0u32.wrapping_sub(underflow as u32))
    }

    #[inline(always)]
    pub fn neg(&self, value: u32) -> u32 {
        let negated = MODULUS - value;
        negated & 0u32.wrapping_sub((value != 0) as u32)
    }

    #[inline(always)]
    pub fn mul(&self, lhs: u32, rhs: u32) -> u32 {
        (lhs as u64 * rhs as u64 % Self::MODULUS_U64) as u32
    }

    #[inline(always)]
    pub fn square(&self, value: u32) -> u32 {
        self.mul(value, value)
    }

    #[inline(always)]
    pub(crate) fn montgomery_mul(lhs: u32, rhs: u32) -> u32 {
        if MODULUS == 2 {
            return lhs & rhs;
        }

        let product = lhs as u64 * rhs as u64;
        let adjustment = (product as u32).wrapping_mul(Self::MONTGOMERY_NEG_INV);
        let (sum, overflow) = product.overflowing_add(adjustment as u64 * Self::MODULUS_U64);
        let reduced = (sum >> 32) + ((overflow as u64) << 32);

        if reduced >= Self::MODULUS_U64 {
            (reduced - Self::MODULUS_U64) as u32
        } else {
            reduced as u32
        }
    }

    #[inline(always)]
    pub(crate) fn to_montgomery(value: u32) -> u32 {
        if MODULUS == 2 {
            return value;
        }
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

    pub fn pow(&self, base: u32, exponent: u64) -> u32 {
        let base = Self::to_montgomery(base);
        Self::from_montgomery(Self::pow_montgomery(base, exponent))
    }

    pub fn inv(&self, value: u32) -> Result<u32, FieldError> {
        if value == 0 {
            return Err(FieldError::DivisionByZero);
        }
        Ok(self.pow(value, (MODULUS - 2) as u64))
    }

    /// Adds `rhs` element-wise into `lhs` without allocating.
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
        let shoup = ((scalar as u64) << 32) / Self::MODULUS_U64;
        for value in values {
            let product = *value as u64 * scalar as u64;
            let quotient = (*value as u64 * shoup) >> 32;
            let remainder = product - quotient * Self::MODULUS_U64;
            *value = if remainder >= Self::MODULUS_U64 {
                (remainder - Self::MODULUS_U64) as u32
            } else {
                remainder as u32
            };
        }
    }

    pub fn dot(&self, lhs: &[u32], rhs: &[u32]) -> Result<u32, FieldError> {
        if lhs.len() != rhs.len() {
            return Err(FieldError::LengthMismatch);
        }

        let max_product = (MODULUS - 1) as u64 * (MODULUS - 1) as u64;
        if lhs.len() as u128 * max_product as u128 <= u64::MAX as u128 {
            let mut sums = [0u64; 4];
            for (lhs, rhs) in lhs.chunks_exact(4).zip(rhs.chunks_exact(4)) {
                sums[0] += lhs[0] as u64 * rhs[0] as u64;
                sums[1] += lhs[1] as u64 * rhs[1] as u64;
                sums[2] += lhs[2] as u64 * rhs[2] as u64;
                sums[3] += lhs[3] as u64 * rhs[3] as u64;
            }

            let remainder_start = lhs.len() / 4 * 4;
            for (&lhs, &rhs) in lhs[remainder_start..].iter().zip(&rhs[remainder_start..]) {
                sums[0] += lhs as u64 * rhs as u64;
            }
            return Ok(self.reduce_u64(sums.into_iter().sum()));
        }

        let mut low = [0u64; 4];
        let mut high = [0u64; 4];
        for (lhs, rhs) in lhs.chunks_exact(4).zip(rhs.chunks_exact(4)) {
            let mut lane = 0;
            while lane < 4 {
                let product = lhs[lane] as u64 * rhs[lane] as u64;
                let (sum, carry) = low[lane].overflowing_add(product);
                low[lane] = sum;
                high[lane] += carry as u64;
                lane += 1;
            }
        }

        let remainder_start = lhs.len() / 4 * 4;
        for (&lhs, &rhs) in lhs[remainder_start..].iter().zip(&rhs[remainder_start..]) {
            let product = lhs as u64 * rhs as u64;
            let (sum, carry) = low[0].overflowing_add(product);
            low[0] = sum;
            high[0] += carry as u64;
        }

        let mut sum = 0u128;
        let mut lane = 0;
        while lane < 4 {
            sum += (high[lane] as u128) << 64 | low[lane] as u128;
            lane += 1;
        }
        Ok(self.reduce_u128(sum))
    }

    /// Inverts all values using one exponentiation and approximately three
    /// multiplications per value.
    pub fn batch_inv_assign(&self, values: &mut [u32]) -> Result<(), FieldError> {
        if values.contains(&0) {
            return Err(FieldError::DivisionByZero);
        }
        if values.is_empty() {
            return Ok(());
        }

        let mut prefixes = Vec::with_capacity(values.len());
        let mut product = Self::MONTGOMERY_ONE;
        for &value in values.iter() {
            product = Self::montgomery_mul(product, Self::to_montgomery(value));
            prefixes.push(product);
        }

        let mut inverse = Self::pow_montgomery(product, (MODULUS - 2) as u64);
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
    pub fn root_of_unity(&self, length: usize) -> Result<u32, FieldError> {
        if !length.is_power_of_two() || length.trailing_zeros() > self.two_adicity() {
            return Err(FieldError::InvalidTransformLength(length));
        }

        let mut root = Self::TWO_ADIC_ROOT;
        for _ in length.trailing_zeros()..self.two_adicity() {
            root = self.square(root);
        }
        Ok(root)
    }
}
