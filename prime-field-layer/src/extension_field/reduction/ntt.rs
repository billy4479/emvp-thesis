use crate::{FieldElement, PrimeField};

use super::{PolynomialReductionPlan, PolynomialReductionScratch};
use crate::extension_field::ExtensionFieldError;
use crate::ntt::NttPlan;

pub(super) struct NttReduction<const MODULUS: u32> {
    pub(super) plan: NttPlan<MODULUS>,
    pub(super) reversed_inverse: Vec<FieldElement<MODULUS>>,
    pub(super) modulus: Vec<FieldElement<MODULUS>>,
}

impl<const MODULUS: u32> PolynomialReductionPlan<MODULUS> {
    pub(in crate::extension_field) fn convolve(
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

    pub(in crate::extension_field) fn square_convolution(
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

    pub(super) fn reduce_ntt(
        k: usize,
        output: &mut [u32],
        scratch: &mut PolynomialReductionScratch<MODULUS>,
        ntt: &NttReduction<MODULUS>,
    ) -> Result<(), ExtensionFieldError> {
        let zero = PrimeField::<MODULUS>::new().element_u32(0);
        scratch.work.fill(zero);
        for index in 0..k - 1 {
            scratch.work[index] = scratch.values[2 * k - 2 - index];
        }
        ntt.plan
            .forward(&mut scratch.work)
            .and_then(|()| {
                ntt.plan
                    .pointwise_mul_assign(&mut scratch.work, &ntt.reversed_inverse)
            })
            .and_then(|()| ntt.plan.inverse(&mut scratch.work))?;

        scratch.work[..k - 1].reverse();
        scratch.work[k - 1..].fill(zero);
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
