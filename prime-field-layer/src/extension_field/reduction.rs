use crate::{FieldElement, FieldError, NttPlan, PrimeField, StaticNttPlan};

use super::ExtensionFieldError;

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

#[doc(hidden)]
pub trait ReductionNtt<const MODULUS: u32> {
    fn forward(&self, values: &mut [FieldElement<MODULUS>]) -> Result<(), FieldError>;
    fn inverse(&self, values: &mut [FieldElement<MODULUS>]) -> Result<(), FieldError>;
    fn pointwise_mul_assign(
        &self,
        lhs: &mut [FieldElement<MODULUS>],
        rhs: &[FieldElement<MODULUS>],
    ) -> Result<(), FieldError>;
}

/// Dynamic transform backend used by the default polynomial reduction plan.
#[doc(hidden)]
pub struct DynamicReductionNtt<const MODULUS: u32>(NttPlan<MODULUS>);

impl<const MODULUS: u32> ReductionNtt<MODULUS> for DynamicReductionNtt<MODULUS> {
    fn forward(&self, values: &mut [FieldElement<MODULUS>]) -> Result<(), FieldError> {
        self.0.forward(values)
    }

    fn inverse(&self, values: &mut [FieldElement<MODULUS>]) -> Result<(), FieldError> {
        self.0.inverse(values)
    }

    fn pointwise_mul_assign(
        &self,
        lhs: &mut [FieldElement<MODULUS>],
        rhs: &[FieldElement<MODULUS>],
    ) -> Result<(), FieldError> {
        self.0.pointwise_mul_assign(lhs, rhs)
    }
}

/// Compile-time transform backend used by [`StaticPolynomialReductionPlan`].
#[doc(hidden)]
pub struct StaticReductionNtt<const MODULUS: u32, const N: usize>(StaticNttPlan<MODULUS, N>);

impl<const MODULUS: u32, const N: usize> ReductionNtt<MODULUS> for StaticReductionNtt<MODULUS, N> {
    fn forward(&self, values: &mut [FieldElement<MODULUS>]) -> Result<(), FieldError> {
        let values = values
            .try_into()
            .map_err(|_slice| FieldError::LengthMismatch)?;
        self.0.forward(values);
        Ok(())
    }

    fn inverse(&self, values: &mut [FieldElement<MODULUS>]) -> Result<(), FieldError> {
        let values = values
            .try_into()
            .map_err(|_slice| FieldError::LengthMismatch)?;
        self.0.inverse(values);
        Ok(())
    }

    fn pointwise_mul_assign(
        &self,
        lhs: &mut [FieldElement<MODULUS>],
        rhs: &[FieldElement<MODULUS>],
    ) -> Result<(), FieldError> {
        let lhs = lhs
            .try_into()
            .map_err(|_slice| FieldError::LengthMismatch)?;
        let rhs = rhs
            .try_into()
            .map_err(|_slice| FieldError::LengthMismatch)?;
        self.0.pointwise_mul_assign(lhs, rhs);
        Ok(())
    }
}

struct NttReduction<const MODULUS: u32, Ntt> {
    plan: Ntt,
    reversed_inverse: Vec<FieldElement<MODULUS>>,
    modulus: Vec<FieldElement<MODULUS>>,
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

impl<const MODULUS: u32, const K: usize>
    PolynomialReductionPlan<MODULUS, K, DynamicReductionNtt<MODULUS>>
{
    /// Validates and precomputes reduction for a fixed monic modulus polynomial.
    ///
    /// This does not test irreducibility because polynomial reduction is valid in
    /// any monic quotient ring. [`super::ExtensionField::new`] adds that check.
    ///
    /// # Errors
    ///
    /// Returns a validation error for degree zero, a coefficient-count mismatch,
    /// or nonmonicity. Large degrees can also return an NTT construction error.
    pub fn new(modulus: &[u32]) -> Result<Self, ExtensionFieldError> {
        Self::from_canonical(validate_modulus::<MODULUS, K>(modulus)?)
    }

    pub(super) fn from_canonical(modulus: Vec<u32>) -> Result<Self, ExtensionFieldError> {
        let field = PrimeField::<MODULUS>::new();
        let negative_modulus = modulus[..K]
            .iter()
            .map(|&coefficient| field.element_u32(field.neg_canonical(coefficient)))
            .collect();
        let (algorithm, ntt) = if K <= SCHOOLBOOK_EXTENSION_DEGREE {
            (PolynomialAlgorithm::Schoolbook, None)
        } else {
            let product_length = product_len::<K>()?;
            let transform_length = product_length
                .checked_next_power_of_two()
                .ok_or(crate::FieldError::ConvolutionLengthOverflow)?;
            let plan = DynamicReductionNtt(NttPlan::<MODULUS>::new(transform_length)?);
            let inverse = reversed_inverse::<MODULUS>(&modulus, K.saturating_sub(1));
            let mut reversed_inverse = padded_elements(field, &inverse, transform_length);
            let mut transformed_modulus = padded_elements(field, &modulus, transform_length);
            plan.forward(&mut reversed_inverse)?;
            plan.forward(&mut transformed_modulus)?;
            (
                PolynomialAlgorithm::Ntt { transform_length },
                Some(NttReduction {
                    plan,
                    reversed_inverse,
                    modulus: transformed_modulus,
                }),
            )
        };
        Ok(Self {
            field,
            modulus,
            negative_modulus,
            algorithm,
            ntt,
        })
    }
}

impl<const MODULUS: u32, const K: usize, const N: usize>
    PolynomialReductionPlan<MODULUS, K, StaticReductionNtt<MODULUS, N>>
{
    /// Validates and precomputes reduction using a compile-time NTT length.
    ///
    /// `N` must equal `next_power_of_two(2K - 1)`. Invalid static dimensions
    /// fail during constant evaluation.
    ///
    /// # Errors
    ///
    /// Returns a validation error for degree zero, a coefficient-count mismatch,
    /// or nonmonicity.
    pub fn new(modulus: &[u32]) -> Result<Self, ExtensionFieldError> {
        Self::from_canonical(validate_modulus::<MODULUS, K>(modulus)?)
    }

    pub(super) fn from_canonical(modulus: Vec<u32>) -> Result<Self, ExtensionFieldError> {
        const {
            assert!(
                K > SCHOOLBOOK_EXTENSION_DEGREE,
                "static NTT reduction requires an NTT-dispatched extension degree"
            );
            assert!(
                N == static_transform_len(K),
                "static NTT length must equal next_power_of_two(2K - 1)"
            );
        }
        let field = PrimeField::<MODULUS>::new();
        let negative_modulus = modulus[..K]
            .iter()
            .map(|&coefficient| field.element_u32(field.neg_canonical(coefficient)))
            .collect();
        let plan = StaticReductionNtt(StaticNttPlan::<MODULUS, N>::new()?);
        let inverse = reversed_inverse::<MODULUS>(&modulus, K - 1);
        let mut reversed_inverse = padded_elements(field, &inverse, N);
        let mut transformed_modulus = padded_elements(field, &modulus, N);
        plan.forward(&mut reversed_inverse)?;
        plan.forward(&mut transformed_modulus)?;
        Ok(Self {
            field,
            modulus,
            negative_modulus,
            algorithm: PolynomialAlgorithm::Ntt {
                transform_length: N,
            },
            ntt: Some(NttReduction {
                plan,
                reversed_inverse,
                modulus: transformed_modulus,
            }),
        })
    }
}

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

    pub(super) fn convolve(
        &self,
        lhs: &mut [FieldElement<MODULUS>],
        rhs: &mut [FieldElement<MODULUS>],
    ) -> Result<(), ExtensionFieldError> {
        let Some(ntt) = &self.ntt else {
            return Ok(());
        };
        ntt.plan
            .forward(lhs)
            .and_then(|()| ntt.plan.forward(rhs))
            .and_then(|()| ntt.plan.pointwise_mul_assign(lhs, rhs))
            .and_then(|()| ntt.plan.inverse(lhs))
            .map_err(ExtensionFieldError::from)
    }

    pub(super) fn square_convolution(
        &self,
        values: &mut [FieldElement<MODULUS>],
    ) -> Result<(), ExtensionFieldError> {
        let Some(ntt) = &self.ntt else {
            return Ok(());
        };
        ntt.plan.forward(values)?;
        for value in values.iter_mut() {
            *value = value.square();
        }
        ntt.plan.inverse(values).map_err(ExtensionFieldError::from)
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

    fn reduce_ntt(
        output: &mut [u32; K],
        scratch: &mut PolynomialReductionScratch<MODULUS, K>,
        ntt: &NttReduction<MODULUS, Ntt>,
    ) -> Result<(), ExtensionFieldError> {
        let zero = PrimeField::<MODULUS>::new().element_u32(0);
        scratch.work.fill(zero);
        for index in 0..K - 1 {
            scratch.work[index] = scratch.values[2 * K - 2 - index];
        }
        ntt.plan
            .forward(&mut scratch.work)
            .and_then(|()| {
                ntt.plan
                    .pointwise_mul_assign(&mut scratch.work, &ntt.reversed_inverse)
            })
            .and_then(|()| ntt.plan.inverse(&mut scratch.work))?;

        scratch.work[..K - 1].reverse();
        scratch.work[K - 1..].fill(zero);
        ntt.plan
            .forward(&mut scratch.work)
            .and_then(|()| {
                ntt.plan
                    .pointwise_mul_assign(&mut scratch.work, &ntt.modulus)
            })
            .and_then(|()| ntt.plan.inverse(&mut scratch.work))?;

        for (index, output) in output.iter_mut().enumerate() {
            *output = (scratch.values[index] - scratch.work[index]).value();
        }
        Ok(())
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

fn padded_elements<const MODULUS: u32>(
    field: PrimeField<MODULUS>,
    coefficients: &[u32],
    length: usize,
) -> Vec<FieldElement<MODULUS>> {
    let mut result = vec![field.element_u32(0); length];
    for (target, &coefficient) in result.iter_mut().zip(coefficients) {
        *target = field.element_u32(coefficient);
    }
    result
}

fn reversed_inverse<const MODULUS: u32>(modulus: &[u32], length: usize) -> Vec<u32> {
    let field = PrimeField::<MODULUS>::new();
    let mut inverse = vec![0; length];
    if length == 0 {
        return inverse;
    }
    inverse[0] = 1;
    for degree in 1..length {
        let mut sum = 0;
        for index in 1..=degree.min(modulus.len() - 1) {
            sum = field.add_canonical(
                sum,
                field.mul(modulus[modulus.len() - 1 - index], inverse[degree - index]),
            );
        }
        inverse[degree] = field.neg_canonical(sum);
    }
    inverse
}
