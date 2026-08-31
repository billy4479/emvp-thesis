//! The cyclic dual code of "Encrypted Matrix-Vector Products from Secret Dual
//! Codes" (IACR ePrint 2025/858), Section 3.1.
//!
//! Fix a block length `k` and the ring `R = F_p[X]/(X^k - 1)`, and identify a
//! vector with its polynomial: index `i` holds the coefficient of `X^i`. For
//! a secret multiplier `g` in `R`, multiplication by `g` acts on column
//! vectors through the circulant matrix `M_g` with
//! `(M_g)[i][j] = g_[(i - j) mod k]`. The secret code has generator
//! `[I_k | M_g]`, so a message `r` encodes to the codeword
//! `c = (r | M_g^T r)` of length `2k`, and the dual has generator
//! `[-M_g^T | I_k]` with `D c = 0` for every codeword `c`.
//!
//! Both protocol directions use only forward cyclic convolution. The
//! transpose satisfies `M_g^T = J M_g J` for the index reversal
//! `J v = reverse(v)`, so `M_g^T r = reverse(conv(g, reverse(r)))`, and a row
//! `m` times the dual gives `m [-M_g^T | I] = (-conv(g, m) | m)`. The cyclic
//! convolution `conv(g, m)`, the product in `R`, uses a direct length-`k` NTT
//! when the field supports one. Other dimensions use a linear convolution of
//! length `2k - 1` whose high half folds back onto the low half.
//!
//! This is experimental cryptography: the construction has no settled
//! security parameters, the multiplier, its cached transform, and scratch
//! buffers are not zeroized on drop, and the implementation has not received
//! a constant-time audit.

use std::fmt;

use prime_field_layer::{FieldElement, FieldError, NttPlan, PrimeField};
use rand_core::CryptoRng;

/// A rejected code construction, encoding, or codeword sampling.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CodeError {
    /// A slice did not have the required length.
    LengthMismatch {
        /// The rejected slice.
        name: &'static str,
        /// The required length.
        expected: usize,
        /// The observed length.
        actual: usize,
    },
    /// A required dimension was zero.
    ZeroDimension(&'static str),
    /// Dimension arithmetic overflowed `usize`.
    DimensionOverflow,
    /// Prime-field or NTT arithmetic failed.
    Field(FieldError),
}

impl fmt::Display for CodeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "{name} length mismatch: expected {expected}, got {actual}"
            ),
            Self::ZeroDimension(name) => write!(formatter, "{name} must be nonzero"),
            Self::DimensionOverflow => formatter.write_str("dimension arithmetic overflowed"),
            Self::Field(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for CodeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Field(error) => Some(error),
            _ => None,
        }
    }
}

impl From<FieldError> for CodeError {
    fn from(error: FieldError) -> Self {
        Self::Field(error)
    }
}

const fn check_len(name: &'static str, expected: usize, actual: usize) -> Result<(), CodeError> {
    if expected == actual {
        Ok(())
    } else {
        Err(CodeError::LengthMismatch {
            name,
            expected,
            actual,
        })
    }
}

/// Reusable storage for [`CyclicDualCode`] evaluation.
///
/// Construct it with [`CyclicDualCode::scratch`]. Its buffers are overwritten
/// on each call. They retain values derived from the secret multiplier and
/// are not zeroized on drop.
pub struct CyclicCodeScratch<const MODULUS: u32> {
    transform: Vec<FieldElement<MODULUS>>,
    operand: Vec<FieldElement<MODULUS>>,
    product: Vec<FieldElement<MODULUS>>,
}

impl<const MODULUS: u32> CyclicCodeScratch<MODULUS> {
    fn new(transform_length: usize, k: usize) -> Self {
        let zero = PrimeField::<MODULUS>::new().element_u32(0);
        Self {
            transform: vec![zero; transform_length],
            operand: vec![zero; k],
            product: vec![zero; k],
        }
    }
}

/// The secret cyclic dual code with generator `[I_k | M_g]`.
///
/// The multiplier `g` and its cached transform are secret and are not
/// zeroized on drop. Evaluation addresses depend only on public dimensions,
/// but this implementation has not received a constant-time audit.
pub struct CyclicDualCode<const MODULUS: u32> {
    k: usize,
    multiplier: Vec<FieldElement<MODULUS>>,
    plan: NttPlan<MODULUS>,
    spectrum: Vec<FieldElement<MODULUS>>,
}

impl<const MODULUS: u32> CyclicDualCode<MODULUS> {
    /// Constructs the code from an explicit multiplier `g` in `F_p^k`.
    ///
    /// # Errors
    ///
    /// Returns an error if `k` is zero, the multiplier length differs from
    /// `k`, dimension arithmetic overflows, or the field does not support the
    /// power-of-two transform length needed to convolve two `k`-coefficient
    /// vectors.
    pub fn new(k: usize, multiplier: Vec<FieldElement<MODULUS>>) -> Result<Self, CodeError> {
        if k == 0 {
            return Err(CodeError::ZeroDimension("code dimension k"));
        }
        check_len("code multiplier", k, multiplier.len())?;
        let linear_length = k
            .checked_mul(2)
            .and_then(|length| length.checked_sub(1))
            .ok_or(CodeError::DimensionOverflow)?;
        let fallback_transform_length = linear_length
            .checked_next_power_of_two()
            .ok_or(CodeError::DimensionOverflow)?;
        let plan = match NttPlan::<MODULUS>::new(k) {
            Ok(plan) => plan,
            Err(FieldError::UnsupportedTransformLength(_)) => {
                NttPlan::<MODULUS>::new(fallback_transform_length)?
            }
            Err(error) => return Err(error.into()),
        };

        let mut spectrum = multiplier.clone();
        spectrum.resize(plan.len(), PrimeField::<MODULUS>::new().element_u32(0));
        plan.forward(&mut spectrum)?;

        Ok(Self {
            k,
            multiplier,
            plan,
            spectrum,
        })
    }

    /// Samples the secret multiplier `g` uniformly from `F_p^k`.
    ///
    /// Rejection sampling gives variable RNG consumption.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::new`].
    pub fn sample<R: CryptoRng + ?Sized>(k: usize, rng: &mut R) -> Result<Self, CodeError> {
        if k == 0 {
            return Err(CodeError::ZeroDimension("code dimension k"));
        }
        k.checked_mul(2).ok_or(CodeError::DimensionOverflow)?;
        let field = PrimeField::<MODULUS>::new();
        let mut multiplier = vec![field.element_u32(0); k];
        field.fill_uniform(rng, &mut multiplier);
        Self::new(k, multiplier)
    }

    /// Returns the block length `k`.
    #[must_use]
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Returns the codeword length `2k`.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.k * 2
    }

    /// Returns the secret multiplier `g`.
    #[must_use]
    pub fn multiplier(&self) -> &[FieldElement<MODULUS>] {
        &self.multiplier
    }

    /// Allocates all buffers needed by [`Self::dual_encode_row`] and
    /// [`Self::sample_codeword`].
    #[must_use]
    pub fn scratch(&self) -> CyclicCodeScratch<MODULUS> {
        CyclicCodeScratch::new(self.plan.len(), self.k)
    }

    /// Writes `row^T D = (-conv(g, row) | row)` into `out`.
    ///
    /// `row` holds the first `row.len() <= k` coefficients of the encoded
    /// message; the missing high coefficients are treated as zero, so the
    /// identity half of `out` is the zero-padded row. All lengths are checked
    /// before `out` is mutated. This performs one NTT convolution that reuses
    /// the plan's tables and the scratch buffers.
    ///
    /// # Errors
    ///
    /// Returns an error before output mutation if the row is longer than
    /// `k`, `out` does not have length `2k`, or the scratch buffers are too
    /// short.
    pub fn dual_encode_row(
        &self,
        row: &[FieldElement<MODULUS>],
        out: &mut [FieldElement<MODULUS>],
        scratch: &mut CyclicCodeScratch<MODULUS>,
    ) -> Result<(), CodeError> {
        if row.len() > self.k {
            return Err(CodeError::LengthMismatch {
                name: "dual-encoding row",
                expected: self.k,
                actual: row.len(),
            });
        }
        check_len("dual encoding output", self.n(), out.len())?;
        self.check_scratch(scratch)?;

        let CyclicCodeScratch {
            transform,
            operand,
            product,
        } = scratch;
        let operand = &mut operand[..self.k];
        operand.fill(PrimeField::<MODULUS>::new().element_u32(0));
        operand[..row.len()].copy_from_slice(row);
        self.cyclic_convolve(operand, &mut product[..self.k], transform)?;

        let (left, right) = out.split_at_mut(self.k);
        for (slot, &convolved) in left.iter_mut().zip(product.iter()) {
            *slot = -convolved;
        }
        right.copy_from_slice(operand);
        Ok(())
    }

    /// Samples `r` uniformly from `F_p^k` and writes the codeword
    /// `c = (r | M_g^T r)` into `out`.
    ///
    /// Rejection sampling gives variable RNG consumption. All lengths are
    /// checked before `out` is mutated.
    ///
    /// # Errors
    ///
    /// Returns an error before output mutation if `out` does not have length
    /// `2k` or the scratch buffers are too short.
    pub fn sample_codeword<R: CryptoRng + ?Sized>(
        &self,
        rng: &mut R,
        out: &mut [FieldElement<MODULUS>],
        scratch: &mut CyclicCodeScratch<MODULUS>,
    ) -> Result<(), CodeError> {
        check_len("codeword output", self.n(), out.len())?;
        self.check_scratch(scratch)?;

        let CyclicCodeScratch {
            transform,
            operand,
            product,
        } = scratch;
        let operand = &mut operand[..self.k];
        PrimeField::<MODULUS>::new().fill_uniform(rng, operand);
        out[..self.k].copy_from_slice(operand);
        operand.reverse();
        self.cyclic_convolve(operand, &mut product[..self.k], transform)?;

        // M_g^T r = J conv(g, J r): reverse the convolution result.
        for (index, slot) in out[self.k..].iter_mut().enumerate() {
            *slot = product[self.k - 1 - index];
        }
        Ok(())
    }

    /// Evaluates `conv(g, input)` through one cached-transform NTT. A direct
    /// length-`k` transform already works modulo `X^k - 1`; the fallback
    /// linear transform folds its high coefficients afterward.
    ///
    /// The transform buffer must have at least the plan's length elements;
    /// the extra tail is ignored. All lengths are checked before any buffer
    /// is written.
    fn cyclic_convolve(
        &self,
        input: &[FieldElement<MODULUS>],
        output: &mut [FieldElement<MODULUS>],
        transform: &mut [FieldElement<MODULUS>],
    ) -> Result<(), CodeError> {
        check_len("cyclic convolution input", self.k, input.len())?;
        check_len("cyclic convolution output", self.k, output.len())?;
        if transform.len() < self.plan.len() {
            return Err(CodeError::LengthMismatch {
                name: "cyclic convolution transform",
                expected: self.plan.len(),
                actual: transform.len(),
            });
        }

        let transform = &mut transform[..self.plan.len()];
        transform.fill(PrimeField::<MODULUS>::new().element_u32(0));
        transform[..self.k].copy_from_slice(input);
        self.plan.forward(transform)?;
        self.plan.pointwise_mul_assign(transform, &self.spectrum)?;
        self.plan.inverse(transform)?;

        output.copy_from_slice(&transform[..self.k]);
        if self.plan.len() != self.k {
            let high = &transform[self.k..2 * self.k - 1];
            for (low, &high) in output.iter_mut().zip(high) {
                *low += high;
            }
        }
        Ok(())
    }

    fn check_scratch(&self, scratch: &CyclicCodeScratch<MODULUS>) -> Result<(), CodeError> {
        if scratch.transform.len() < self.plan.len() {
            return Err(CodeError::LengthMismatch {
                name: "code scratch transform",
                expected: self.plan.len(),
                actual: scratch.transform.len(),
            });
        }
        check_len("code scratch operand", self.k, scratch.operand.len())?;
        check_len("code scratch product", self.k, scratch.product.len())
    }
}
