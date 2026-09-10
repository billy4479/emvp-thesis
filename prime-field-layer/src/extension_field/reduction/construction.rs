use crate::{FieldElement, NttPlan, PrimeField};

use super::{
    NttReduction, PolynomialAlgorithm, PolynomialReductionPlan, SCHOOLBOOK_EXTENSION_DEGREE,
    product_len, validate_modulus,
};
use crate::extension_field::ExtensionFieldError;

impl<const MODULUS: u32> PolynomialReductionPlan<MODULUS> {
    /// Validates and precomputes reduction for a fixed monic modulus polynomial.
    ///
    /// This does not test irreducibility because polynomial reduction is valid in
    /// any monic quotient ring. [`crate::ExtensionField::new`] adds that check.
    ///
    /// # Errors
    ///
    /// Returns a validation error for degree zero, a coefficient-count mismatch,
    /// or nonmonicity. Large degrees can also return an NTT construction error.
    pub fn new(k: usize, modulus: &[u32]) -> Result<Self, ExtensionFieldError> {
        Self::from_canonical(k, validate_modulus::<MODULUS>(k, modulus)?)
    }

    pub(in crate::extension_field) fn from_canonical(
        k: usize,
        modulus: Vec<u32>,
    ) -> Result<Self, ExtensionFieldError> {
        let field = PrimeField::<MODULUS>::new();
        // The negated lower coefficients feed only the schoolbook long
        // division; NTT reduction never reads them, so large degrees skip the
        // `K` conversions and the storage entirely.
        let negative_modulus = if k <= SCHOOLBOOK_EXTENSION_DEGREE {
            modulus[..k]
                .iter()
                .map(|&coefficient| field.element_u32(field.neg_canonical(coefficient)))
                .collect()
        } else {
            Vec::new()
        };
        let (algorithm, ntt) = if k <= SCHOOLBOOK_EXTENSION_DEGREE {
            (PolynomialAlgorithm::Schoolbook, None)
        } else {
            let product_length = product_len(k)?;
            let transform_length = product_length
                .checked_next_power_of_two()
                .ok_or(crate::FieldError::ConvolutionLengthOverflow)?;
            let plan = NttPlan::<MODULUS>::new(transform_length)?;
            let inverse = reversed_inverse::<MODULUS>(&plan, &modulus, k.saturating_sub(1))?;
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
            k,
            modulus,
            negative_modulus,
            algorithm,
            ntt,
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

/// Computes `reverse(f)^(-1) mod X^length` with Newton iteration.
///
/// Writing `a = reverse(f)`, monicity gives `a[0] = f[K] = 1`, and the
/// truncated inverse `g` satisfies `a * g = 1 mod X^length`. Newton's
/// iteration for power-series inverses lifts a precision-`t` approximation
/// to precision `2t` via `g_2t = g_t * (2 - a * g_t) mod X^2t`, doubling the
/// number of correct coefficients per step (von zur Gathen & Gerhard,
/// *Modern Computer Algebra*, 3rd edition, chapter 9). Each step evaluates
/// two truncated products as zero-padded linear convolutions with the
/// caller's plan, for five transforms per step; the total is `O(K log K)`
/// field operations instead of the `O(K^2)` schoolbook recurrence, which
/// profiled as the dominant cost of plan construction at the production
/// degrees.
///
/// Truncation is free here: every intermediate convolution stays below the
/// transform length, so no aliasing can fold the dropped high terms back
/// into the kept prefix. The widest products occur at the final step with
/// `t = ceil(length / 2)`: `a * g_t` spreads at most `(K + 1) + t - 1` and
/// `g_t * (2 - a g_t)` at most `t + 2t - 1 = 3t - 1` coefficients, both
/// below `next_power_of_two(2K - 1)` for every `K` that reaches this path.
fn reversed_inverse<const MODULUS: u32>(
    plan: &NttPlan<MODULUS>,
    modulus: &[u32],
    length: usize,
) -> Result<Vec<u32>, ExtensionFieldError> {
    let field = PrimeField::<MODULUS>::new();
    if length == 0 {
        return Ok(Vec::new());
    }
    debug_assert_eq!(
        modulus.last(),
        Some(&1),
        "the truncated inverse requires a monic modulus"
    );
    let transform_length = plan.len();
    let zero = field.element_u32(0);
    // `a_hat` is the full reversed modulus over the Montgomery-domain
    // transform; products against it are automatically truncated mod X^next
    // because the dropped high convolution terms land beyond the prefix the
    // iteration keeps.
    let mut a_hat = vec![zero; transform_length];
    for (slot, &coefficient) in a_hat.iter_mut().zip(modulus.iter().rev()) {
        *slot = field.element_u32(coefficient);
    }
    plan.forward(&mut a_hat)?;

    let mut g = vec![zero; transform_length];
    g[0] = field.element_u32(1);
    let mut high = vec![zero; transform_length];
    let mut work = vec![zero; transform_length];
    let mut precision = 1;
    while precision < length {
        let next = precision.saturating_mul(2).min(length);

        // `high = a * g mod X^next` via one zero-padded convolution. `g` is
        // zero beyond `precision` by the loop invariant, so the copy
        // suffices.
        work.copy_from_slice(&g);
        plan.forward(&mut work)?;
        high.copy_from_slice(&work);
        plan.pointwise_mul_assign(&mut high, &a_hat)?;
        plan.inverse(&mut high)?;

        // `u = 2 - high mod X^next` (the `2` only touches the constant
        // term), keeping `work`'s transform of `g` for the second product.
        for (index, slot) in high.iter_mut().enumerate() {
            if index >= next {
                *slot = zero;
            } else if index == 0 {
                *slot = field.element_u32(2) - *slot;
            } else {
                *slot = -*slot;
            }
        }
        plan.forward(&mut high)?;
        plan.pointwise_mul_assign(&mut high, &work)?;
        plan.inverse(&mut high)?;

        // `g = g * u mod X^next`; the dropped tail may hold convolution
        // terms beyond the kept precision, so it is cleared explicitly.
        g.copy_from_slice(&high);
        g[next..].fill(zero);
        precision = next;
    }

    Ok(g[..length].iter().map(|&value| value.value()).collect())
}
