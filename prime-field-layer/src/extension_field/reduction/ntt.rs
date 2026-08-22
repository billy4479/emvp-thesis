use crate::{FieldElement, PrimeField};

use super::{PolynomialReductionPlan, PolynomialReductionScratch, ReductionNtt};
use crate::extension_field::ExtensionFieldError;

pub(super) struct NttReduction<const MODULUS: u32, Ntt> {
    pub(super) plan: Ntt,
    pub(super) reversed_inverse: Vec<FieldElement<MODULUS>>,
    pub(super) modulus: Vec<FieldElement<MODULUS>>,
}

impl<const MODULUS: u32, const K: usize, Ntt> PolynomialReductionPlan<MODULUS, K, Ntt>
where
    Ntt: ReductionNtt<MODULUS>,
{
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
