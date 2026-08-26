use crate::{FieldElement, FieldError, PrimeField};

use super::ExtensionFieldError;
use super::irreducibility::is_irreducible;
use super::reduction::{
    self, PolynomialAlgorithm, PolynomialReductionPlan, PolynomialReductionScratch,
};

/// Arithmetic in `F_q[X]/(f)` for a fixed degree-`K` monic polynomial `f`.
pub struct ExtensionField<const MODULUS: u32> {
    k: usize,
    field: PrimeField<MODULUS>,
    reduction: PolynomialReductionPlan<MODULUS>,
}

/// Caller-owned work storage for allocation-free extension multiplication.
pub struct ExtensionFieldScratch<const MODULUS: u32> {
    k: usize,
    reduction: PolynomialReductionScratch<MODULUS>,
    rhs: Vec<FieldElement<MODULUS>>,
}

impl<const MODULUS: u32> ExtensionField<MODULUS> {
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
    pub fn new(k: usize, modulus: &[u32]) -> Result<Self, ExtensionFieldError> {
        let canonical = reduction::validate_modulus::<MODULUS>(k, modulus)?;
        if !is_irreducible::<MODULUS>(&canonical) {
            return Err(ExtensionFieldError::ReducibleModulus);
        }
        Ok(Self {
            k,
            field: PrimeField::new(),
            reduction: PolynomialReductionPlan::from_canonical(k, canonical)?,
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
    pub fn new_unchecked_irreducible(
        k: usize,
        modulus: &[u32],
    ) -> Result<Self, ExtensionFieldError> {
        Ok(Self {
            k,
            field: PrimeField::new(),
            reduction: PolynomialReductionPlan::new(k, modulus)?,
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
    pub fn scratch(&self) -> ExtensionFieldScratch<MODULUS> {
        let length = self.reduction.work_len();
        ExtensionFieldScratch {
            k: self.k,
            reduction: self.reduction.scratch(),
            rhs: vec![self.reduction.zero(); length],
        }
    }

    /// Adds two extension elements coefficient by coefficient.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::LengthMismatch`] unless both slices have the extension degree.
    pub fn add_assign(&self, lhs: &mut [u32], rhs: &[u32]) -> Result<(), FieldError> {
        if lhs.len() != self.k || rhs.len() != self.k {
            return Err(FieldError::LengthMismatch);
        }
        for (lhs, &rhs) in lhs.iter_mut().zip(rhs) {
            let lhs_canonical = self.field.reduce_u32(*lhs);
            let rhs_canonical = self.field.reduce_u32(rhs);
            *lhs = self.field.add_canonical(lhs_canonical, rhs_canonical);
        }
        Ok(())
    }

    /// Subtracts two extension elements coefficient by coefficient.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::LengthMismatch`] unless both slices have the extension degree.
    pub fn sub_assign(&self, lhs: &mut [u32], rhs: &[u32]) -> Result<(), FieldError> {
        if lhs.len() != self.k || rhs.len() != self.k {
            return Err(FieldError::LengthMismatch);
        }
        for (lhs, &rhs) in lhs.iter_mut().zip(rhs) {
            let lhs_canonical = self.field.reduce_u32(*lhs);
            let rhs_canonical = self.field.reduce_u32(rhs);
            *lhs = self.field.sub_canonical(lhs_canonical, rhs_canonical);
        }
        Ok(())
    }

    /// Multiplies two extension elements without allocating.
    ///
    /// For `K <= 23`, this uses schoolbook multiplication. Larger degrees use a
    /// zero-padded linear NTT of length `next_power_of_two(2K - 1)`, then reduce
    /// the ordinary product modulo the caller's polynomial.
    ///
    /// # Errors
    ///
    /// Returns a base-field length error if an operand, the output, or the
    /// scratch storage does not have this extension's degree.
    pub fn mul(
        &self,
        lhs: &[u32],
        rhs: &[u32],
        output: &mut [u32],
        scratch: &mut ExtensionFieldScratch<MODULUS>,
    ) -> Result<(), ExtensionFieldError> {
        if lhs.len() != self.k
            || rhs.len() != self.k
            || output.len() != self.k
            || scratch.k != self.k
        {
            return Err(ExtensionFieldError::BaseField(FieldError::LengthMismatch));
        }

        self.prepare_operand(lhs, &mut scratch.reduction.values);
        self.prepare_operand(rhs, &mut scratch.rhs);
        match self.algorithm() {
            PolynomialAlgorithm::Schoolbook => {
                let zero = self.reduction.zero();
                let product = &mut scratch.reduction.values;
                product.fill(zero);
                for (lhs_index, &lhs) in lhs.iter().enumerate() {
                    let lhs = PrimeField::<MODULUS>::new().element_u32(lhs);
                    for (rhs_index, &rhs) in scratch.rhs[..self.k].iter().enumerate() {
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
    /// Returns a base-field length error if the input, output, or scratch
    /// storage does not have this extension's degree.
    pub fn square(
        &self,
        value: &[u32],
        output: &mut [u32],
        scratch: &mut ExtensionFieldScratch<MODULUS>,
    ) -> Result<(), ExtensionFieldError> {
        if value.len() != self.k || output.len() != self.k || scratch.k != self.k {
            return Err(ExtensionFieldError::BaseField(FieldError::LengthMismatch));
        }

        self.prepare_operand(value, &mut scratch.rhs);
        let zero = self.reduction.zero();
        let product = &mut scratch.reduction.values;
        product.fill(zero);
        match self.algorithm() {
            PolynomialAlgorithm::Schoolbook => {
                for lhs_index in 0..self.k {
                    let lhs = scratch.rhs[lhs_index];
                    product[lhs_index * 2] += lhs.square();
                    for rhs_index in lhs_index + 1..self.k {
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

    fn prepare_operand(&self, input: &[u32], output: &mut [FieldElement<MODULUS>]) {
        output.fill(self.reduction.zero());
        for (output, &input) in output.iter_mut().zip(input) {
            *output = self.field.element_u32(input);
        }
    }
}
