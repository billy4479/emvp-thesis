mod backend;
mod construction;
mod ntt;

use crate::{FieldElement, PrimeField};

use super::ExtensionFieldError;

#[doc(hidden)]
pub use backend::{DynamicReductionNtt, ReductionNtt, StaticReductionNtt};
use ntt::NttReduction;

/// Extension products at or below this degree use schoolbook code.
///
/// The module's Criterion benchmark places the cached multiply-and-reduce
/// crossover between degrees 23 and 24 on the development AVX2 host. Dispatch
/// depends only on public degree and remains stable across coefficients.
pub const SCHOOLBOOK_EXTENSION_DEGREE: usize = 23;

/// Polynomial multiplication and reduction strategy selected by degree.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum PolynomialAlgorithm {
    Schoolbook,
    Ntt { transform_length: usize },
}

/// Reusable reduction data for one fixed monic degree-`K` polynomial.
///
/// Small degrees retain the negated lower modulus coefficients for monic long
/// division. Large degrees precompute the truncated inverse of the reversed
/// modulus and transform both that inverse and the modulus. Reduction then uses
/// reversed polynomial division and two zero-padded convolutions in
/// `O(K log K)` time.
///
/// Concretely, for `degree(A) < 2K - 1`, reversing the top `K - 1`
/// coefficients of `A` and multiplying by
/// `reverse(f)^(-1) mod X^(K-1)` gives the reversed quotient. Reversing it back
/// and computing `A - quotient * f` gives the degree-below-`K` remainder. Both
/// products are ordinary zero-padded linear products.
pub struct PolynomialReductionPlan<
    const MODULUS: u32,
    const K: usize,
    Ntt = DynamicReductionNtt<MODULUS>,
> {
    field: PrimeField<MODULUS>,
    modulus: Vec<u32>,
    negative_modulus: Vec<FieldElement<MODULUS>>,
    algorithm: PolynomialAlgorithm,
    ntt: Option<NttReduction<MODULUS, Ntt>>,
}

/// Caller-owned work storage for allocation-free repeated reduction.
pub struct PolynomialReductionScratch<const MODULUS: u32, const K: usize> {
    pub(super) values: Vec<FieldElement<MODULUS>>,
    work: Vec<FieldElement<MODULUS>>,
}

/// A polynomial reduction plan whose NTT length is fixed at compile time.
///
/// `N` must equal `next_power_of_two(2K - 1)`, and `K` must exceed
/// [`SCHOOLBOOK_EXTENSION_DEGREE`]. Invalid dimensions fail during constant
/// evaluation:
///
/// ```compile_fail
/// use prime_field_layer::StaticPolynomialReductionPlan;
///
/// // Degree 24 requires transform length 64.
/// let modulus = [1; 25];
/// let _ = StaticPolynomialReductionPlan::<998_244_353, 24, 128>::new(&modulus);
/// ```
pub type StaticPolynomialReductionPlan<const MODULUS: u32, const K: usize, const N: usize> =
    PolynomialReductionPlan<MODULUS, K, StaticReductionNtt<MODULUS, N>>;

impl<const MODULUS: u32, const K: usize, Ntt> PolynomialReductionPlan<MODULUS, K, Ntt>
where
    Ntt: ReductionNtt<MODULUS>,
{
    /// Returns the canonical modulus coefficients, including the leading one.
    #[must_use]
    pub fn modulus_polynomial(&self) -> &[u32] {
        &self.modulus
    }

    /// Returns the selected public-degree dispatch.
    #[must_use]
    pub const fn algorithm(&self) -> PolynomialAlgorithm {
        self.algorithm
    }

    /// Allocates reusable work storage for [`Self::reduce`].
    #[must_use]
    pub fn scratch(&self) -> PolynomialReductionScratch<MODULUS, K> {
        let length = self.work_len();
        PolynomialReductionScratch {
            values: vec![self.zero(); length],
            work: if self.ntt.is_some() {
                vec![self.zero(); length]
            } else {
                Vec::new()
            },
        }
    }

    /// Reduces a polynomial with at most `2K - 1` coefficients modulo `f`.
    ///
    /// Input coefficients are canonicalized. `output` always receives exactly
    /// `K` canonical coefficients. Once `scratch` is constructed, this method
    /// allocates nothing.
    ///
    /// # Errors
    ///
    /// Returns [`ExtensionFieldError::ProductTooLong`] when `product` has more
    /// than `2K - 1` coefficients, or a base-field length error if the supplied
    /// scratch storage does not match this plan.
    pub fn reduce(
        &self,
        product: &[u32],
        output: &mut [u32; K],
        scratch: &mut PolynomialReductionScratch<MODULUS, K>,
    ) -> Result<(), ExtensionFieldError> {
        let maximum = product_len::<K>()?;
        if product.len() > maximum {
            return Err(ExtensionFieldError::ProductTooLong {
                maximum,
                actual: product.len(),
            });
        }
        scratch.values.fill(self.zero());
        for (target, &coefficient) in scratch.values.iter_mut().zip(product) {
            *target = self.field.element_u32(coefficient);
        }
        self.reduce_elements(output, scratch)
    }

    pub(super) fn zero(&self) -> FieldElement<MODULUS> {
        self.field.element_u32(0)
    }

    pub(super) fn work_len(&self) -> usize {
        match self.algorithm {
            PolynomialAlgorithm::Schoolbook => product_len::<K>().unwrap_or(0),
            PolynomialAlgorithm::Ntt { transform_length } => transform_length,
        }
    }

    pub(super) fn reduce_elements(
        &self,
        output: &mut [u32; K],
        scratch: &mut PolynomialReductionScratch<MODULUS, K>,
    ) -> Result<(), ExtensionFieldError> {
        match &self.ntt {
            None => {
                self.reduce_schoolbook(output, &mut scratch.values);
                Ok(())
            }
            Some(ntt) => Self::reduce_ntt(output, scratch, ntt),
        }
    }

    fn reduce_schoolbook(&self, output: &mut [u32; K], values: &mut [FieldElement<MODULUS>]) {
        if K != 0 {
            for degree in (K..values.len()).rev() {
                let high = values[degree];
                for (index, &negative) in self.negative_modulus.iter().enumerate() {
                    values[degree - K + index] += high * negative;
                }
            }
        }
        for (output, value) in output.iter_mut().zip(values.iter()) {
            *output = value.value();
        }
    }
}

pub(super) fn validate_modulus<const MODULUS: u32, const K: usize>(
    modulus: &[u32],
) -> Result<Vec<u32>, ExtensionFieldError> {
    if K == 0 {
        return Err(ExtensionFieldError::ZeroDegree);
    }
    let expected = K.checked_add(1).ok_or(ExtensionFieldError::ModulusLength {
        expected: usize::MAX,
        actual: modulus.len(),
    })?;
    if modulus.len() != expected {
        return Err(ExtensionFieldError::ModulusLength {
            expected,
            actual: modulus.len(),
        });
    }
    let field = PrimeField::<MODULUS>::new();
    let canonical: Vec<_> = modulus
        .iter()
        .map(|&coefficient| field.reduce_u64(u64::from(coefficient)))
        .collect();
    if canonical[K] != 1 {
        return Err(ExtensionFieldError::ModulusNotMonic);
    }
    Ok(canonical)
}

fn product_len<const K: usize>() -> Result<usize, ExtensionFieldError> {
    K.checked_mul(2)
        .and_then(|length| length.checked_sub(1))
        .ok_or(ExtensionFieldError::ZeroDegree)
}

const fn static_transform_len(degree: usize) -> usize {
    (degree * 2 - 1).next_power_of_two()
}
