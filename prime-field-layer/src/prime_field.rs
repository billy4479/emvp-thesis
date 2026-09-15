use std::fmt;

mod bulk;
mod scalar;

/// Errors caused by invalid field parameters or operands.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FieldError {
    DivisionByZero,
    LengthMismatch,
    UnsupportedTransformLength(usize),
    PlanTooSmall {
        required: usize,
        available: usize,
    },
    ConvolutionLengthOverflow,
    /// A raw Montgomery word was not a canonical residue.
    NonCanonicalRaw {
        raw: u32,
        modulus: u32,
    },
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
            Self::NonCanonicalRaw { raw, modulus } => write!(
                formatter,
                "raw Montgomery word {raw} is not a canonical residue below modulus {modulus}"
            ),
        }
    }
}

impl std::error::Error for FieldError {}

/// Arithmetic modulo the compile-time odd prime `MODULUS`.
///
/// The supported invariant is an *odd* prime modulus: every Montgomery and
/// two-adic kernel assumes `MODULUS > 2`, so `F_2` is deliberately not
/// supported and fails to compile exactly like a composite modulus.
///
/// Methods whose names end in `_canonical` require every raw `u32` operand to
/// be less than `MODULUS`. This explicit low-level API avoids a remainder in hot
/// loops. Other raw `u32` methods accept the full `u32` range and reduce as part
/// of their operation. Prefer [`crate::FieldElement`] when values remain in the
/// field across multiple operations; its type guarantees the representation.
///
/// Invalid moduli are rejected during compilation:
///
/// ```compile_fail
/// let _ = prime_field_layer::PrimeField::<15>::new();
/// ```
///
/// The even prime is rejected the same way:
///
/// ```compile_fail
/// let _ = prime_field_layer::PrimeField::<2>::new();
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
    pub(crate) const VALID_MODULUS: () = assert!(
        MODULUS > 2 && Self::is_prime(MODULUS),
        "field modulus must be an odd prime"
    );
    const MODULUS_U64: u64 = MODULUS as u64;
    pub(crate) const MONTGOMERY_NEG_INV: u32 = Self::calculate_montgomery_neg_inv();
    const MONTGOMERY_R2: u32 = ((1u128 << 64) % MODULUS as u128) as u32;
    const MONTGOMERY_ONE: u32 = ((1u64 << 32) % MODULUS as u64) as u32;
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

    const fn calculate_montgomery_neg_inv() -> u32 {
        let mut inverse = MODULUS;
        let mut iteration = 0;
        while iteration < 5 {
            inverse = inverse.wrapping_mul(2u32.wrapping_sub(MODULUS.wrapping_mul(inverse)));
            iteration += 1;
        }
        inverse.wrapping_neg()
    }

    const fn calculate_two_adic_root() -> u32 {
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
    /// Compilation fails when `MODULUS` is not an odd prime.
    #[inline(always)]
    #[must_use]
    pub const fn new() -> Self {
        let () = Self::VALID_MODULUS;
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
    ///
    /// `MODULUS` is a compile-time constant, so the remainder is lowered by
    /// LLVM to a constant-latency multiply-high/shift sequence (or a
    /// branchless `cmov` correction for single reductions), never to the
    /// variable-latency hardware `div` instruction; `--emit asm` builds of
    /// this crate contain zero `div`/`idiv` in the field kernels. Timings
    /// are therefore data-independent for secret inputs.
    #[inline(always)]
    #[must_use]
    pub const fn reduce_u64(&self, value: u64) -> u32 {
        (value % Self::MODULUS_U64) as u32
    }

    /// Reduces an arbitrary 32-bit integer to a canonical residue.
    ///
    /// See [`Self::reduce_u64`] for the constant-latency lowering argument.
    #[inline(always)]
    #[must_use]
    pub const fn reduce_u32(&self, value: u32) -> u32 {
        value % MODULUS
    }

    /// Returns the Montgomery REDC constant `-MODULUS^{-1} mod 2^32`.
    ///
    /// The Montgomery reduction multiplies the low word of the running value
    /// by this constant to cancel the low 32 bits before the division by
    /// `2^32`. This is exposed so external accelerators, such as GPU compute
    /// kernels, can replicate the crate's Montgomery multiplication exactly.
    /// Every supported (odd prime) modulus yields a valid constant.
    #[must_use]
    pub const fn montgomery_neg_inv(&self) -> u32 {
        Self::MONTGOMERY_NEG_INV
    }

    /// Returns the Montgomery conversion constant `2^64 mod MODULUS`.
    ///
    /// Multiplying a canonical residue by this constant under the Montgomery
    /// reduction yields the Montgomery residue `a * 2^32 mod MODULUS`. This
    /// is exposed alongside [`Self::montgomery_neg_inv`] so external
    /// accelerators can convert between canonical and Montgomery words with
    /// the same constants the crate uses.
    #[must_use]
    pub const fn montgomery_r2(&self) -> u32 {
        Self::MONTGOMERY_R2
    }
}
