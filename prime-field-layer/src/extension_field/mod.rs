//! Fixed monic polynomial reduction and extension-field arithmetic.
//!
//! [`ExtensionField`] represents `F_q[X]/(f)` for one fixed monic, irreducible
//! polynomial `f` of degree `K`. The modulus polynomial is used only for ordinary
//! polynomial remainder computation. When `K` is large, an NTT computes
//! zero-padded linear products before reduction; it never changes the quotient
//! to `X^K - 1` or any other NTT-friendly polynomial.
//!
//! Construction canonicalizes the modulus coefficients and precomputes the data
//! needed by repeated reduction. [`ExtensionField::new`] also runs Rabin's
//! deterministic Frobenius/GCD irreducibility test. Use
//! [`ExtensionField::new_unchecked_irreducible`] only when irreducibility was
//! established elsewhere. Both constructors validate the coefficient count and
//! monicity.
//!
//! Multiplication and squaring write into caller-provided arrays and use
//! [`ExtensionFieldScratch`], whose allocation is reusable. Addition and
//! subtraction need no scratch. Every public coefficient input may be any
//! `u32`; every public output and stored modulus coefficient is canonical.

mod irreducibility;
mod reduction;

use std::fmt;

use crate::{FieldElement, FieldError, PrimeField};

pub use reduction::{
    PolynomialAlgorithm, PolynomialReductionPlan, PolynomialReductionScratch,
    SCHOOLBOOK_EXTENSION_DEGREE,
};

use irreducibility::is_irreducible;

/// Errors from fixed polynomial reduction or extension-field construction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExtensionFieldError {
    /// Degree zero does not define a nontrivial polynomial quotient.
    ZeroDegree,
    /// A degree-`K` modulus must contain exactly `K + 1` coefficients.
    ModulusLength { expected: usize, actual: usize },
    /// The degree-`K` coefficient is not one in the base field.
    ModulusNotMonic,
    /// The checked constructor found a nontrivial factor.
    ReducibleModulus,
    /// Reduction input exceeded the maximum `2K - 1` coefficients.
    ProductTooLong { maximum: usize, actual: usize },
    /// The base field could not construct an NTT required by this degree.
    BaseField(FieldError),
}

impl fmt::Display for ExtensionFieldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDegree => formatter.write_str("extension degree must be positive"),
            Self::ModulusLength { expected, actual } => write!(
                formatter,
                "a modulus of the expected degree needs {expected} coefficients, got {actual}"
            ),
            Self::ModulusNotMonic => formatter.write_str("the modulus polynomial must be monic"),
            Self::ReducibleModulus => {
                formatter.write_str("the modulus polynomial is reducible over the base field")
            }
            Self::ProductTooLong { maximum, actual } => write!(
                formatter,
                "reduction accepts at most {maximum} coefficients, got {actual}"
            ),
            Self::BaseField(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ExtensionFieldError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::BaseField(error) => Some(error),
            _ => None,
        }
    }
}

impl From<FieldError> for ExtensionFieldError {
    fn from(error: FieldError) -> Self {
        Self::BaseField(error)
    }
}

/// Arithmetic in `F_q[X]/(f)` for a fixed degree-`K` monic polynomial `f`.
pub struct ExtensionField<const MODULUS: u32, const K: usize> {
    reduction: PolynomialReductionPlan<MODULUS, K>,
}

/// Caller-owned work storage for allocation-free extension multiplication.
pub struct ExtensionFieldScratch<const MODULUS: u32, const K: usize> {
    reduction: PolynomialReductionScratch<MODULUS, K>,
    rhs: Vec<FieldElement<MODULUS>>,
}

impl<const MODULUS: u32, const K: usize> ExtensionField<MODULUS, K> {
    /// Constructs an extension field after deterministic irreducibility testing.
    ///
    /// Rabin's test computes successive `q`-power Frobenius images modulo `f`,
    /// checks a polynomial GCD for every prime divisor of `K`, and verifies
    /// `X^(q^K) = X mod f`. Its current construction-oriented implementation is
    /// intentionally simple and can be expensive at large `K`.
    ///
    /// # Errors
    ///
    /// Returns a validation error for degree zero, a coefficient-count mismatch,
    /// nonmonicity, or reducibility. Large degrees selected for NTT arithmetic
    /// can also return the underlying transform-construction error.
    pub fn new(modulus: &[u32]) -> Result<Self, ExtensionFieldError> {
        let canonical = reduction::validate_modulus::<MODULUS, K>(modulus)?;
        if !is_irreducible::<MODULUS>(&canonical) {
            return Err(ExtensionFieldError::ReducibleModulus);
        }
        Ok(Self {
            reduction: PolynomialReductionPlan::from_canonical(canonical)?,
        })
    }

    /// Constructs an extension field while trusting an external irreducibility proof.
    ///
    /// This skips only irreducibility testing. It still validates degree and
    /// monicity, canonicalizes coefficients, and builds all reduction and NTT
    /// data. Passing a reducible polynomial makes this a quotient ring rather
    /// than a field, so callers must not make field-specific assumptions.
    ///
    /// # Errors
    ///
    /// Returns the same shape, monicity, and NTT errors as [`Self::new`], except
    /// that it cannot return [`ExtensionFieldError::ReducibleModulus`].
    pub fn new_unchecked_irreducible(modulus: &[u32]) -> Result<Self, ExtensionFieldError> {
        Ok(Self {
            reduction: PolynomialReductionPlan::new(modulus)?,
        })
    }

    /// Returns the canonical coefficients of `f`, including its leading one.
    #[must_use]
    pub fn modulus_polynomial(&self) -> &[u32] {
        self.reduction.modulus_polynomial()
    }

    /// Returns the multiplication and reduction kernel selected from `K`.
    #[must_use]
    pub const fn algorithm(&self) -> PolynomialAlgorithm {
        self.reduction.algorithm()
    }

    /// Allocates work storage for repeated multiplication and squaring.
    #[must_use]
    pub fn scratch(&self) -> ExtensionFieldScratch<MODULUS, K> {
        let length = self.reduction.work_len();
        ExtensionFieldScratch {
            reduction: self.reduction.scratch(),
            rhs: vec![self.reduction.zero(); length],
        }
    }

    /// Adds two extension elements coefficient by coefficient.
    #[must_use]
    pub fn add(&self, lhs: &[u32; K], rhs: &[u32; K]) -> [u32; K] {
        let field = PrimeField::<MODULUS>::new();
        std::array::from_fn(|index| {
            field.add_canonical(
                field.reduce_u64(u64::from(lhs[index])),
                field.reduce_u64(u64::from(rhs[index])),
            )
        })
    }

    /// Subtracts two extension elements coefficient by coefficient.
    #[must_use]
    pub fn sub(&self, lhs: &[u32; K], rhs: &[u32; K]) -> [u32; K] {
        let field = PrimeField::<MODULUS>::new();
        std::array::from_fn(|index| {
            field.sub_canonical(
                field.reduce_u64(u64::from(lhs[index])),
                field.reduce_u64(u64::from(rhs[index])),
            )
        })
    }

    /// Multiplies two extension elements without allocating.
    ///
    /// For `K <= 23`, this uses schoolbook multiplication. Larger degrees use a
    /// zero-padded linear NTT of length `next_power_of_two(2K - 1)`, then reduce
    /// the ordinary product modulo the caller's polynomial.
    ///
    /// # Errors
    ///
    /// Returns a base-field length error only if the scratch storage does not
    /// match this degree's fixed transform plan.
    pub fn mul(
        &self,
        lhs: &[u32; K],
        rhs: &[u32; K],
        output: &mut [u32; K],
        scratch: &mut ExtensionFieldScratch<MODULUS, K>,
    ) -> Result<(), ExtensionFieldError> {
        self.prepare_operand(lhs, &mut scratch.reduction.values);
        self.prepare_operand(rhs, &mut scratch.rhs);
        match self.algorithm() {
            PolynomialAlgorithm::Schoolbook => {
                let zero = self.reduction.zero();
                let product = &mut scratch.reduction.values;
                product.fill(zero);
                for (lhs_index, &lhs) in lhs.iter().enumerate() {
                    let lhs = PrimeField::<MODULUS>::new().element_u32(lhs);
                    for (rhs_index, &rhs) in scratch.rhs[..K].iter().enumerate() {
                        product[lhs_index + rhs_index] += lhs * rhs;
                    }
                }
            }
            PolynomialAlgorithm::Ntt { .. } => {
                self.reduction
                    .convolve(&mut scratch.reduction.values, &mut scratch.rhs)?;
            }
        }
        self.reduction
            .reduce_elements(output, &mut scratch.reduction)
    }

    /// Squares one extension element without allocating.
    ///
    /// The schoolbook path computes diagonal products once and doubles each
    /// off-diagonal product. The NTT path needs one forward transform rather than
    /// the two used by general multiplication.
    ///
    /// # Errors
    ///
    /// Returns a base-field length error only if the scratch storage does not
    /// match this degree's fixed transform plan.
    pub fn square(
        &self,
        value: &[u32; K],
        output: &mut [u32; K],
        scratch: &mut ExtensionFieldScratch<MODULUS, K>,
    ) -> Result<(), ExtensionFieldError> {
        self.prepare_operand(value, &mut scratch.rhs);
        let zero = self.reduction.zero();
        let product = &mut scratch.reduction.values;
        product.fill(zero);
        match self.algorithm() {
            PolynomialAlgorithm::Schoolbook => {
                for lhs_index in 0..K {
                    let lhs = scratch.rhs[lhs_index];
                    product[lhs_index * 2] += lhs.square();
                    for rhs_index in lhs_index + 1..K {
                        let cross = lhs * scratch.rhs[rhs_index];
                        product[lhs_index + rhs_index] += cross + cross;
                    }
                }
            }
            PolynomialAlgorithm::Ntt { .. } => {
                product.copy_from_slice(&scratch.rhs);
                self.reduction.square_convolution(product)?;
            }
        }
        self.reduction
            .reduce_elements(output, &mut scratch.reduction)
    }

    fn prepare_operand(&self, input: &[u32; K], output: &mut [FieldElement<MODULUS>]) {
        let field = PrimeField::<MODULUS>::new();
        output.fill(self.reduction.zero());
        for (output, &input) in output.iter_mut().zip(input) {
            *output = field.element_u32(input);
        }
    }
}
