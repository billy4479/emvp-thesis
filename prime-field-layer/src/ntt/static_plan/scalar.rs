use super::{
    FieldElement, PrimeField, StaticNttPlan, add_mod, halve_interval, normalize, shoup_mul,
    shoup_mul_lazy_for, stage_twiddle_index, sub_mod,
};

impl<const MODULUS: u32, const N: usize> StaticNttPlan<MODULUS, N> {
    #[inline(always)]
    pub(super) fn forward_shoup_lazy(values: &mut [FieldElement<MODULUS>; N]) {
        let two_p = MODULUS * 2;
        let mut distance = N / 2;
        while distance != 0 {
            let blocks = N / (2 * distance);
            for block in 0..blocks {
                // SAFETY: stage_twiddle_index is in `0..N - 1` for every stage.
                let twiddle = unsafe {
                    *Self::FORWARD_TWIDDLES.get_unchecked(stage_twiddle_index(blocks, block))
                };
                let start = block * 2 * distance;
                for index in start..start + distance {
                    // SAFETY: both halves are within this statically sized block.
                    unsafe {
                        let lhs = halve_interval((*values.as_ptr().add(index)).montgomery(), two_p);
                        let rhs = (*values.as_ptr().add(index + distance)).montgomery();
                        let product = shoup_mul_lazy_for::<MODULUS>(rhs, twiddle);
                        (*values.as_mut_ptr().add(index)).set_montgomery(lhs + product);
                        (*values.as_mut_ptr().add(index + distance))
                            .set_montgomery(lhs + two_p - product);
                    }
                }
            }
            distance /= 2;
        }
        normalize(values);
    }

    #[inline(always)]
    pub(super) fn inverse_shoup_lazy(values: &mut [FieldElement<MODULUS>; N]) {
        let two_p = MODULUS * 2;
        let mut distance = 1;
        while distance < N {
            let blocks = N / (2 * distance);
            for block in 0..blocks {
                let twiddle = Self::INVERSE_TWIDDLES[stage_twiddle_index(blocks, block)];
                let start = block * 2 * distance;
                let (lhs_values, rhs_values) =
                    values[start..start + 2 * distance].split_at_mut(distance);
                for (lhs_value, rhs_value) in lhs_values.iter_mut().zip(rhs_values) {
                    let lhs = lhs_value.montgomery();
                    let rhs = rhs_value.montgomery();
                    let difference = lhs + two_p - rhs;
                    lhs_value.set_montgomery(halve_interval(lhs + rhs, two_p));
                    rhs_value.set_montgomery(shoup_mul_lazy_for::<MODULUS>(difference, twiddle));
                }
            }
            distance *= 2;
        }
        normalize(values);
    }

    #[inline(always)]
    pub(super) fn forward_shoup(values: &mut [FieldElement<MODULUS>; N]) {
        let mut distance = N / 2;
        while distance != 0 {
            let blocks = N / (2 * distance);
            for block in 0..blocks {
                let twiddle = Self::FORWARD_TWIDDLES[stage_twiddle_index(blocks, block)];
                let start = block * 2 * distance;
                let (lhs_values, rhs_values) =
                    values[start..start + 2 * distance].split_at_mut(distance);
                for (lhs_value, rhs_value) in lhs_values.iter_mut().zip(rhs_values) {
                    let lhs = lhs_value.montgomery();
                    let product = shoup_mul::<MODULUS>(rhs_value.montgomery(), twiddle);
                    lhs_value.set_montgomery(add_mod::<MODULUS>(lhs, product));
                    rhs_value.set_montgomery(sub_mod::<MODULUS>(lhs, product));
                }
            }
            distance /= 2;
        }
    }

    #[inline(always)]
    pub(super) fn inverse_shoup(values: &mut [FieldElement<MODULUS>; N]) {
        let mut distance = 1;
        while distance < N {
            let blocks = N / (2 * distance);
            for block in 0..blocks {
                let twiddle = Self::INVERSE_TWIDDLES[stage_twiddle_index(blocks, block)];
                let start = block * 2 * distance;
                let (lhs_values, rhs_values) =
                    values[start..start + 2 * distance].split_at_mut(distance);
                for (lhs_value, rhs_value) in lhs_values.iter_mut().zip(rhs_values) {
                    let lhs = lhs_value.montgomery();
                    let rhs = rhs_value.montgomery();
                    lhs_value.set_montgomery(add_mod::<MODULUS>(lhs, rhs));
                    rhs_value.set_montgomery(shoup_mul::<MODULUS>(
                        sub_mod::<MODULUS>(lhs, rhs),
                        twiddle,
                    ));
                }
            }
            distance *= 2;
        }
    }

    #[inline(always)]
    pub(super) fn forward_montgomery(values: &mut [FieldElement<MODULUS>; N]) {
        let mut distance = N / 2;
        while distance != 0 {
            let blocks = N / (2 * distance);
            for block in 0..blocks {
                let twiddle = Self::FORWARD_TWIDDLES[stage_twiddle_index(blocks, block)];
                let start = block * 2 * distance;
                let (lhs_values, rhs_values) =
                    values[start..start + 2 * distance].split_at_mut(distance);
                for (lhs_value, rhs_value) in lhs_values.iter_mut().zip(rhs_values) {
                    let lhs = lhs_value.montgomery();
                    let product = PrimeField::<MODULUS>::montgomery_mul(
                        rhs_value.montgomery(),
                        twiddle.montgomery,
                    );
                    lhs_value.set_montgomery(add_mod::<MODULUS>(lhs, product));
                    rhs_value.set_montgomery(sub_mod::<MODULUS>(lhs, product));
                }
            }
            distance /= 2;
        }
    }

    #[inline(always)]
    pub(super) fn inverse_montgomery(values: &mut [FieldElement<MODULUS>; N]) {
        let mut distance = 1;
        while distance < N {
            let blocks = N / (2 * distance);
            for block in 0..blocks {
                let twiddle = Self::INVERSE_TWIDDLES[stage_twiddle_index(blocks, block)];
                let start = block * 2 * distance;
                let (lhs_values, rhs_values) =
                    values[start..start + 2 * distance].split_at_mut(distance);
                for (lhs_value, rhs_value) in lhs_values.iter_mut().zip(rhs_values) {
                    let lhs = lhs_value.montgomery();
                    let rhs = rhs_value.montgomery();
                    lhs_value.set_montgomery(add_mod::<MODULUS>(lhs, rhs));
                    rhs_value.set_montgomery(PrimeField::<MODULUS>::montgomery_mul(
                        sub_mod::<MODULUS>(lhs, rhs),
                        twiddle.montgomery,
                    ));
                }
            }
            distance *= 2;
        }
    }
}
