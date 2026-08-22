use crate::{FieldElement, PrimeField};

use super::ExtensionFieldError;
use super::irreducibility::is_irreducible;
use super::reduction::{
    self, DynamicReductionNtt, PolynomialAlgorithm, PolynomialReductionPlan,
    PolynomialReductionScratch, ReductionNtt, StaticPolynomialReductionPlan, StaticReductionNtt,
};

/// Arithmetic in `F_q[X]/(f)` for a fixed degree-`K` monic polynomial `f`.
pub struct ExtensionField<const MODULUS: u32, const K: usize, Ntt = DynamicReductionNtt<MODULUS>> {
    reduction: PolynomialReductionPlan<MODULUS, K, Ntt>,
}

/// An extension field whose NTT length is fixed at compile time.
///
/// `N` must equal `next_power_of_two(2K - 1)`, and `K` must exceed
/// [`crate::SCHOOLBOOK_EXTENSION_DEGREE`]. Construction checks both requirements during
/// constant evaluation. Prefer the default [`ExtensionField`] unless benchmarks
/// for the intended degree show that static specialization is faster.
pub type StaticExtensionField<const MODULUS: u32, const K: usize, const N: usize> =
    ExtensionField<MODULUS, K, StaticReductionNtt<MODULUS, N>>;

/// Caller-owned work storage for allocation-free extension multiplication.
pub struct ExtensionFieldScratch<const MODULUS: u32, const K: usize> {
    reduction: PolynomialReductionScratch<MODULUS, K>,
    rhs: Vec<FieldElement<MODULUS>>,
}

impl<const MODULUS: u32, const K: usize> ExtensionField<MODULUS, K, DynamicReductionNtt<MODULUS>> {
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
            reduction:
                PolynomialReductionPlan::<MODULUS, K, DynamicReductionNtt<MODULUS>>::from_canonical(
                    canonical,
                )?,
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
            reduction: PolynomialReductionPlan::<MODULUS, K, DynamicReductionNtt<MODULUS>>::new(
                modulus,
            )?,
        })
    }
}

impl<const MODULUS: u32, const K: usize, const N: usize>
    ExtensionField<MODULUS, K, StaticReductionNtt<MODULUS, N>>
{
    /// Constructs a statically sized extension field after irreducibility testing.
    ///
    /// `N` must equal `next_power_of_two(2K - 1)`. Invalid static dimensions
    /// fail during constant evaluation.
    ///
    /// # Errors
    ///
    /// Returns a validation error for degree zero, a coefficient-count mismatch,
    /// nonmonicity, or reducibility.
    pub fn new(modulus: &[u32]) -> Result<Self, ExtensionFieldError> {
        let canonical = reduction::validate_modulus::<MODULUS, K>(modulus)?;
        if !is_irreducible::<MODULUS>(&canonical) {
            return Err(ExtensionFieldError::ReducibleModulus);
        }
        Ok(Self {
            reduction: StaticPolynomialReductionPlan::from_canonical(canonical)?,
        })
    }

    /// Constructs a statically sized extension field while trusting an external
    /// irreducibility proof.
    ///
    /// # Errors
    ///
    /// Returns a validation error for degree zero, a coefficient-count mismatch,
    /// or nonmonicity.
    pub fn new_unchecked_irreducible(modulus: &[u32]) -> Result<Self, ExtensionFieldError> {
        Ok(Self {
            reduction: StaticPolynomialReductionPlan::new(modulus)?,
        })
    }
}

impl<const MODULUS: u32, const K: usize, Ntt> ExtensionField<MODULUS, K, Ntt>
where
    Ntt: ReductionNtt<MODULUS>,
{
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
