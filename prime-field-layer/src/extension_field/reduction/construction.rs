use crate::{FieldElement, NttPlan, PrimeField, StaticNttPlan};

use super::{
    DynamicReductionNtt, NttReduction, PolynomialAlgorithm, PolynomialReductionPlan, ReductionNtt,
    SCHOOLBOOK_EXTENSION_DEGREE, StaticReductionNtt, product_len, static_transform_len,
    validate_modulus,
};
use crate::extension_field::ExtensionFieldError;

impl<const MODULUS: u32, const K: usize>
    PolynomialReductionPlan<MODULUS, K, DynamicReductionNtt<MODULUS>>
{
    /// Validates and precomputes reduction for a fixed monic modulus polynomial.
    ///
    /// This does not test irreducibility because polynomial reduction is valid in
    /// any monic quotient ring. [`crate::ExtensionField::new`] adds that check.
    ///
    /// # Errors
    ///
    /// Returns a validation error for degree zero, a coefficient-count mismatch,
    /// or nonmonicity. Large degrees can also return an NTT construction error.
    pub fn new(modulus: &[u32]) -> Result<Self, ExtensionFieldError> {
        Self::from_canonical(validate_modulus::<MODULUS, K>(modulus)?)
    }

    pub(in crate::extension_field) fn from_canonical(
        modulus: Vec<u32>,
    ) -> Result<Self, ExtensionFieldError> {
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

    pub(in crate::extension_field) fn from_canonical(
        modulus: Vec<u32>,
    ) -> Result<Self, ExtensionFieldError> {
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
