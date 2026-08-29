use super::{
    FieldElement, NttPlan, PrimeField, add_mod, halve_interval, normalize, reduce_once, shoup_mul,
    shoup_mul_lazy_for, sub_mod,
};

impl<const MODULUS: u32> NttPlan<MODULUS> {
    // For p < 2^30, forward stages keep lazy residues in [0, 4p) under
    // Harvey's Cooley-Tukey butterfly (J. Symbolic Comput. 60 (2014),
    // section 4, Algorithm 4). For inputs X, Y in [0, 4p) and twiddle w:
    //   X  <- X - 2p when X >= 2p   (the single correction; X now in [0, 2p))
    //   t  = lazy Shoup product of Y (in [0, 2p) by the wide-input bound on
    //                                  `shoup_mul_lazy_for`, valid since
    //                                  Y < 4p <= 2^32)
    //   X' = X + t                    in [0, 4p)
    //   Y' = X + 2p - t               in (0, 4p)
    // The outputs are again in [0, 4p), so the interval is stable across all
    // stages. The scheme needs 4p <= 2^32, i.e. p <= 2^30, which is exactly
    // the tier gate selecting this backend; `normalize` restores [0, p).
    pub(super) fn forward_shoup_lazy(&self, values: &mut [FieldElement<MODULUS>]) {
        let two_p = MODULUS * 2;
        for stage in self.stages.iter() {
            for (block, &twiddle) in stage.forward.iter().enumerate() {
                let start = block * 2 * stage.distance;
                let (lhs_values, rhs_values) =
                    values[start..start + 2 * stage.distance].split_at_mut(stage.distance);
                for (lhs_value, rhs_value) in lhs_values.iter_mut().zip(rhs_values) {
                    let lhs = halve_interval(lhs_value.montgomery(), two_p);
                    let product = shoup_mul_lazy_for::<MODULUS>(rhs_value.montgomery(), twiddle);
                    lhs_value.set_montgomery(lhs + product);
                    rhs_value.set_montgomery(lhs + two_p - product);
                }
            }
        }
        normalize(values);
    }

    // Inverse stages keep lazy residues in [0, 2p) under Harvey's
    // Gentleman-Sande butterfly (J. Symbolic Comput. 60 (2014), section 3,
    // Algorithm 3). For inputs X, Y in [0, 2p) and twiddle w:
    //   S  = X + Y                     in [0, 4p)
    //   S' <- S - 2p when S >= 2p      (the single correction; S' in [0, 2p))
    //   D  = X + 2p - Y                in (0, 4p), no comparison: the planted
    //                                   +2p keeps the signed difference in the
    //                                   unsigned interval
    //   Y' = lazy Shoup product of D   in [0, 2p) by the wide-input bound on
    //                                   `shoup_mul_lazy_for`, valid since
    //                                   D < 4p <= 2^32
    // The outputs are again in [0, 2p), so the interval is stable across all
    // stages, and `normalize` restores [0, p).
    pub(super) fn inverse_shoup_lazy(&self, values: &mut [FieldElement<MODULUS>]) {
        let two_p = MODULUS * 2;
        for stage in self.stages.iter().rev() {
            for (block, &twiddle) in stage.inverse.iter().enumerate() {
                let start = block * 2 * stage.distance;
                let (lhs_values, rhs_values) =
                    values[start..start + 2 * stage.distance].split_at_mut(stage.distance);
                for (lhs_value, rhs_value) in lhs_values.iter_mut().zip(rhs_values) {
                    let lhs = lhs_value.montgomery();
                    let rhs = rhs_value.montgomery();
                    let difference = lhs + two_p - rhs;
                    lhs_value.set_montgomery(halve_interval(lhs + rhs, two_p));
                    rhs_value.set_montgomery(shoup_mul_lazy_for::<MODULUS>(difference, twiddle));
                }
            }
        }
        normalize(values);
    }

    // Reduced Shoup butterflies for the `2^30 <= p < 2^31` tier, mirroring
    // the portable kernel that previously served the AVX2 backend. Every
    // operand and sum fits a `u32`: inputs are canonical, the uncorrected
    // Shoup product lies in `[0, 2p)`, and `lhs + product` and
    // `lhs + p - product` stay below `2p < 2^32`. Each result therefore needs
    // exactly one masked `reduce_once`, avoiding the wide `add_mod` and
    // `sub_mod` helpers whose inline assembly would block loop vectorization.
    pub(super) fn forward_shoup(&self, values: &mut [FieldElement<MODULUS>]) {
        for stage in self.stages.iter() {
            for (block, &twiddle) in stage.forward.iter().enumerate() {
                let start = block * 2 * stage.distance;
                let (lhs_values, rhs_values) =
                    values[start..start + 2 * stage.distance].split_at_mut(stage.distance);
                for (lhs_value, rhs_value) in lhs_values.iter_mut().zip(rhs_values) {
                    let lhs = lhs_value.montgomery();
                    let product = reduce_once(
                        shoup_mul_lazy_for::<MODULUS>(rhs_value.montgomery(), twiddle),
                        MODULUS,
                    );
                    lhs_value.set_montgomery(reduce_once(lhs.wrapping_add(product), MODULUS));
                    rhs_value.set_montgomery(reduce_once(
                        lhs.wrapping_add(MODULUS).wrapping_sub(product),
                        MODULUS,
                    ));
                }
            }
        }
    }

    pub(super) fn inverse_shoup(&self, values: &mut [FieldElement<MODULUS>]) {
        for stage in self.stages.iter().rev() {
            for (block, &twiddle) in stage.inverse.iter().enumerate() {
                let start = block * 2 * stage.distance;
                let (lhs_values, rhs_values) =
                    values[start..start + 2 * stage.distance].split_at_mut(stage.distance);
                for (lhs_value, rhs_value) in lhs_values.iter_mut().zip(rhs_values) {
                    let lhs = lhs_value.montgomery();
                    let rhs = rhs_value.montgomery();
                    lhs_value.set_montgomery(reduce_once(lhs.wrapping_add(rhs), MODULUS));
                    rhs_value.set_montgomery(shoup_mul::<MODULUS>(
                        reduce_once(lhs.wrapping_add(MODULUS).wrapping_sub(rhs), MODULUS),
                        twiddle,
                    ));
                }
            }
        }
    }

    pub(super) fn forward_montgomery(&self, values: &mut [FieldElement<MODULUS>]) {
        for stage in self.stages.iter() {
            for (block, twiddle) in stage.forward.iter().enumerate() {
                let start = block * 2 * stage.distance;
                let (lhs_values, rhs_values) =
                    values[start..start + 2 * stage.distance].split_at_mut(stage.distance);
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
        }
    }

    pub(super) fn inverse_montgomery(&self, values: &mut [FieldElement<MODULUS>]) {
        for stage in self.stages.iter().rev() {
            for (block, twiddle) in stage.inverse.iter().enumerate() {
                let start = block * 2 * stage.distance;
                let (lhs_values, rhs_values) =
                    values[start..start + 2 * stage.distance].split_at_mut(stage.distance);
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
        }
    }
}
