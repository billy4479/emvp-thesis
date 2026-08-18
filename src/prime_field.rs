use std::fmt;

/// Errors caused by invalid field parameters or operands.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FieldError {
    InvalidModulus(u32),
    DivisionByZero,
    LengthMismatch,
    InvalidTransformLength(usize),
}

impl fmt::Display for FieldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidModulus(modulus) => write!(formatter, "{modulus} is not prime"),
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

/// Arithmetic context for canonical residues in `[0, modulus)`.
///
/// Implement methods in roughly the order in which they appear. The integration
/// tests and benchmarks use only this public contract.
#[derive(Clone, Debug, PartialEq)]
pub struct PrimeField {
    modulus: u32,
    modulus_u64: u64,
    barrett_mu: u64,
    montgomery_neg_inv: u32,
    montgomery_r2: u32,
    two_adic_root: u32,
}

impl PrimeField {
    fn modular_pow(mut base: u64, mut exponent: u32, modulus: u64) -> u64 {
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

    fn is_prime(value: u32) -> bool {
        const SMALL_PRIMES: [u32; 12] = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37];
        for prime in SMALL_PRIMES {
            if value.is_multiple_of(prime) {
                return value == prime;
            }
        }
        if value < 2 {
            return false;
        }

        let shifts = (value - 1).trailing_zeros();
        let odd_part = (value - 1) >> shifts;
        let modulus = value as u64;

        'witness: for base in [2u64, 7, 61] {
            if base >= modulus {
                continue;
            }

            let mut power = Self::modular_pow(base, odd_part, modulus);
            if power == 1 || power == modulus - 1 {
                continue;
            }

            for _ in 1..shifts {
                power = power * power % modulus;
                if power == modulus - 1 {
                    continue 'witness;
                }
            }
            return false;
        }

        true
    }

    /// Constructs a field after validating that `modulus` is prime.
    pub fn new(modulus: u32) -> Result<Self, FieldError> {
        if !PrimeField::is_prime(modulus) {
            return Err(FieldError::InvalidModulus(modulus));
        }

        let (montgomery_neg_inv, montgomery_r2) = if modulus == 2 {
            (0, 0)
        } else {
            let mut inverse = modulus;
            for _ in 0..5 {
                inverse = inverse.wrapping_mul(2u32.wrapping_sub(modulus.wrapping_mul(inverse)));
            }
            (
                inverse.wrapping_neg(),
                ((1u128 << 64) % modulus as u128) as u32,
            )
        };

        let mut field = PrimeField {
            modulus,
            modulus_u64: modulus as u64,
            barrett_mu: ((1u128 << 64) / modulus as u128) as u64,
            montgomery_neg_inv,
            montgomery_r2,
            two_adic_root: 1,
        };

        if modulus != 2 {
            let mut non_residue = 2;
            while field.pow(non_residue, ((modulus - 1) / 2) as u64) == 1 {
                non_residue += 1;
            }
            field.two_adic_root =
                field.pow(non_residue, ((modulus - 1) >> field.two_adicity()) as u64);
        }

        Ok(field)
    }

    pub fn modulus(&self) -> u32 {
        self.modulus
    }

    /// Returns the exponent of two in the factorization of `modulus - 1`.
    pub fn two_adicity(&self) -> u32 {
        (self.modulus - 1).trailing_zeros()
    }

    /// Reduces an arbitrary 64-bit integer to a canonical residue.
    pub fn reduce_u64(&self, value: u64) -> u32 {
        let q = ((value as u128 * self.barrett_mu as u128) >> 64) as u64;
        let r = value - q * self.modulus_u64;

        let (reduced, underflow) = r.overflowing_sub(self.modulus_u64);

        if underflow { r as u32 } else { reduced as u32 }
    }

    fn reduce_u128(&self, value: u128) -> u32 {
        let high = self.reduce_u64((value >> 64) as u64) as u64;
        let low = self.reduce_u64(value as u64) as u64;
        self.reduce_u64(high * self.montgomery_r2 as u64 + low)
    }

    pub fn add(&self, lhs: u32, rhs: u32) -> u32 {
        let sum = lhs as u64 + rhs as u64;
        if sum >= self.modulus_u64 {
            (sum - self.modulus_u64) as u32
        } else {
            sum as u32
        }
    }

    pub fn sub(&self, lhs: u32, rhs: u32) -> u32 {
        let (diff, overflow) = lhs.overflowing_sub(rhs);

        // If overflow == true then the mask is 0xFFFFFFFF, so we add the modulus normally,
        // if overflow == false then the mask is 0x00000000, so nothing happens.
        let mask = 0u32.wrapping_sub(overflow as u32);

        diff.wrapping_add(self.modulus & mask)
    }

    pub fn neg(&self, value: u32) -> u32 {
        if value == 0 {
            return 0;
        }

        self.modulus - value
    }

    pub fn mul(&self, lhs: u32, rhs: u32) -> u32 {
        self.reduce_u64(lhs as u64 * (rhs as u64))
    }

    pub fn square(&self, value: u32) -> u32 {
        self.mul(value, value)
    }

    fn montgomery_mul(&self, lhs: u32, rhs: u32) -> u32 {
        let product = lhs as u64 * rhs as u64;
        let adjustment = (product as u32).wrapping_mul(self.montgomery_neg_inv);
        let (sum, overflow) = product.overflowing_add(adjustment as u64 * self.modulus_u64);
        let reduced = (sum >> 32) + ((overflow as u64) << 32);

        if reduced >= self.modulus_u64 {
            (reduced - self.modulus_u64) as u32
        } else {
            reduced as u32
        }
    }

    pub fn pow(&self, base: u32, mut exponent: u64) -> u32 {
        if self.modulus == 2 {
            return if exponent == 0 { 1 } else { base };
        }
        if exponent == 0 {
            return 1;
        }

        let mut base = self.montgomery_mul(base, self.montgomery_r2);
        let mut result = self.montgomery_mul(1, self.montgomery_r2);

        while exponent != 0 {
            if exponent & 1 == 1 {
                result = self.montgomery_mul(result, base);
            }
            exponent >>= 1;
            if exponent != 0 {
                base = self.montgomery_mul(base, base);
            }
        }

        self.montgomery_mul(result, 1)
    }

    pub fn inv(&self, value: u32) -> Result<u32, FieldError> {
        if value == 0 {
            return Err(FieldError::DivisionByZero);
        }
        Ok(self.pow(value, (self.modulus - 2) as u64))
    }

    /// Adds `rhs` element-wise into `lhs` without allocating.
    pub fn add_assign(&self, lhs: &mut [u32], rhs: &[u32]) -> Result<(), FieldError> {
        if lhs.len() != rhs.len() {
            return Err(FieldError::LengthMismatch);
        }

        let modulus = self.modulus_u64;
        for (lhs, &rhs) in lhs.iter_mut().zip(rhs) {
            let sum = *lhs as u64 + rhs as u64;
            *lhs = if sum >= modulus {
                (sum - modulus) as u32
            } else {
                sum as u32
            };
        }
        Ok(())
    }

    /// Subtracts `rhs` element-wise from `lhs` without allocating.
    pub fn sub_assign(&self, lhs: &mut [u32], rhs: &[u32]) -> Result<(), FieldError> {
        if lhs.len() != rhs.len() {
            return Err(FieldError::LengthMismatch);
        }

        let modulus = self.modulus;
        for (lhs, &rhs) in lhs.iter_mut().zip(rhs) {
            let (difference, underflow) = lhs.overflowing_sub(rhs);
            *lhs = difference.wrapping_add(modulus & 0u32.wrapping_sub(underflow as u32));
        }
        Ok(())
    }

    /// Multiplies `lhs` element-wise by `rhs` without allocating.
    pub fn mul_assign(&self, lhs: &mut [u32], rhs: &[u32]) -> Result<(), FieldError> {
        if lhs.len() != rhs.len() {
            return Err(FieldError::LengthMismatch);
        }

        for (lhs, &rhs) in lhs.iter_mut().zip(rhs) {
            *lhs = self.reduce_u64(*lhs as u64 * rhs as u64);
        }
        Ok(())
    }

    pub fn neg_assign(&self, values: &mut [u32]) {
        let modulus = self.modulus;
        for value in values {
            let negated = modulus - *value;
            *value = negated & 0u32.wrapping_sub((*value != 0) as u32);
        }
    }

    pub fn scalar_mul_assign(&self, values: &mut [u32], scalar: u32) {
        let shoup = ((scalar as u64) << 32) / self.modulus_u64;
        for value in values {
            let product = *value as u64 * scalar as u64;
            let quotient = (*value as u64 * shoup) >> 32;
            let remainder = product - quotient * self.modulus_u64;
            *value = if remainder >= self.modulus_u64 {
                (remainder - self.modulus_u64) as u32
            } else {
                remainder as u32
            };
        }
    }

    pub fn dot(&self, lhs: &[u32], rhs: &[u32]) -> Result<u32, FieldError> {
        if lhs.len() != rhs.len() {
            return Err(FieldError::LengthMismatch);
        }

        let mut sums = [0u128; 4];
        for (lhs, rhs) in lhs.chunks_exact(4).zip(rhs.chunks_exact(4)) {
            sums[0] += lhs[0] as u128 * rhs[0] as u128;
            sums[1] += lhs[1] as u128 * rhs[1] as u128;
            sums[2] += lhs[2] as u128 * rhs[2] as u128;
            sums[3] += lhs[3] as u128 * rhs[3] as u128;
        }

        let remainder_start = lhs.len() / 4 * 4;
        for (&lhs, &rhs) in lhs[remainder_start..].iter().zip(&rhs[remainder_start..]) {
            sums[0] += lhs as u128 * rhs as u128;
        }

        Ok(self.reduce_u128(sums.into_iter().sum()))
    }

    /// Returns an element of exact order `length`.
    ///
    /// Supported lengths are powers of two dividing `modulus - 1`; length one
    /// has root one.
    pub fn root_of_unity(&self, length: usize) -> Result<u32, FieldError> {
        if !length.is_power_of_two() || length.trailing_zeros() > self.two_adicity() {
            return Err(FieldError::InvalidTransformLength(length));
        }

        let mut root = self.two_adic_root;
        for _ in length.trailing_zeros()..self.two_adicity() {
            root = self.square(root);
        }
        Ok(root)
    }
}
