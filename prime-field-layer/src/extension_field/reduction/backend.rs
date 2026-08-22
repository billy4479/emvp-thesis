use crate::{FieldElement, FieldError, NttPlan, StaticNttPlan};

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
pub struct DynamicReductionNtt<const MODULUS: u32>(pub(super) NttPlan<MODULUS>);

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

/// Compile-time transform backend used by [`super::StaticPolynomialReductionPlan`].
#[doc(hidden)]
pub struct StaticReductionNtt<const MODULUS: u32, const N: usize>(
    pub(super) StaticNttPlan<MODULUS, N>,
);

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
