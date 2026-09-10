//! Dependency-free radix-2 number theoretic transforms and convolution.
//!
//! For a plan of length `N` and the root `omega` returned by
//! [`PrimeField::root_of_unity`], [`NttPlan::forward`] computes
//! `X[k] = sum_j x[j] * omega^(j*k)`. Coefficients enter in natural order, but
//! output slot `i` contains `X[bit_reverse(i)]`. [`NttPlan::inverse`] consumes
//! exactly that bit-reversed frequency order and restores natural-order
//! coefficients. Avoiding explicit permutations is useful in convolution
//! pipelines.
//!
//! Plans retain `O(N)` twiddle storage. Construction takes `O(N + log p)` field
//! operations; each transform and planned convolution takes `O(N log N)` time.
//! Transform and pointwise methods allocate nothing, while convenience
//! convolution methods allocate work and result vectors. Reuse plans when the
//! size and modulus repeat.
//!
//! # Forward and inverse
//!
//! ```
//! use prime_field_layer::NttPlan;
//!
//! let plan = NttPlan::<17>::new(8)?;
//! let input = [1, 2, 3, 4, 5, 6, 7, 8];
//! let mut values = plan.elements(&input);
//! plan.forward(&mut values)?; // Bit-reversed frequency order.
//! plan.inverse(&mut values)?; // Natural coefficient order again.
//! assert_eq!(values.iter().map(|x| x.value()).collect::<Vec<_>>(), input);
//! # Ok::<(), prime_field_layer::FieldError>(())
//! ```
//!
//! # Repeated cyclic products
//!
//! ```
//! use prime_field_layer::NttPlan;
//!
//! let plan = NttPlan::<17>::new(4)?;
//! let first = plan.cyclic_convolution(&[1, 2, 3, 4], &[4, 3, 2, 1])?;
//! let second = plan.cyclic_convolution(&[1, 0, 1, 0], &[1, 1, 1, 1])?;
//! assert_eq!(first, vec![7, 5, 7, 13]);
//! assert_eq!(second, vec![2, 2, 2, 2]);
//! # Ok::<(), prime_field_layer::FieldError>(())
//! ```
//!
//! # Repeated linear products with one fixed operand
//!
//! ```
//! use prime_field_layer::NttPlan;
//!
//! let plan = NttPlan::<17>::new(8)?;
//! let fixed = plan.pretransform_linear_operand(&[1, 2, 3])?;
//! let mut workspace = fixed.workspace();
//! let mut output = [0; 5];
//! fixed.convolve(&[4, 5, 6], &mut output, &mut workspace)?;
//! assert_eq!(output, [4, 13, 11, 10, 1]);
//! # Ok::<(), prime_field_layer::FieldError>(())
//! ```
//!
//! # Negacyclic and one-shot linear convolution
//!
//! ```
//! use prime_field_layer::{NegacyclicPlan, linear_convolution};
//!
//! let plan = NegacyclicPlan::<17>::new(4)?;
//! assert_eq!(plan.convolution(&[1, 2, 3, 4], &[1, 1, 0, 0])?, vec![14, 3, 5, 7]);
//! assert_eq!(linear_convolution::<17>(&[1, 2, 3], &[4, 5])?, vec![4, 13, 5, 15]);
//! # Ok::<(), prime_field_layer::FieldError>(())
//! ```
//!
//! The arithmetic kernels are designed without coefficient-dependent control
//! flow. Dispatch, allocation, errors, and loop counts depend on public modulus,
//! platform, and slice lengths. This is a code-level timing precaution, not a
//! claim of a formally audited constant-time implementation.

use std::{fmt, sync::Arc};

use crate::{FieldElement, FieldError, PrimeField, constant_time::reduce_once_u64};

mod convolution;
mod plan;
mod scalar;
mod support;
#[cfg(test)]
mod tests;

use support::{
    BackendPreference, add_mod, halve_interval, normalize, powers, reduce_once, select_backend,
    shoup_mul, shoup_mul_lazy_for, sub_mod, twiddle_powers,
};

pub use convolution::linear_convolution;

/// Arithmetic implementation selected for an [`NttPlan`].
///
/// Selection occurs once during construction. Inspect this value for
/// diagnostics or benchmarking, not to infer transform ordering or results:
/// all variants implement the same field transform. This describes butterfly
/// selection only; pointwise products and inverse normalization go through
/// [`PrimeField`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NttBackend {
    /// Lazy Harvey Shoup butterflies with one correction per butterfly.
    ///
    /// Forward stages keep residues in `[0, 4p)` and inverse stages in
    /// `[0, 2p)`, so each butterfly needs a single conditional halving and the
    /// lazy products absorb unreduced inputs. Used for `p < 2^30`. Public
    /// transform outputs are normalized to Montgomery residues in `[0, p)`.
    ScalarShoupLazy,
    /// Shoup butterflies reduced to `[0, p)` after each butterfly.
    ///
    /// Used for `2^30 <= p < 2^31`, where the lazy kernel's bounds do not fit.
    ScalarShoup,
    /// Montgomery butterflies for moduli at least `2^31`.
    ///
    /// This general fallback supports wide `u32` primes but does not use the
    /// Shoup transform kernels.
    ScalarMontgomery,
}

/// A typed explanation for a slower-than-ideal transform configuration.
///
/// [`NttPlan::performance_warning`] reports at most one construction-time
/// reason. `None` means the selected plan has no warning.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NttPerformanceWarning {
    /// [`NttPlan::new_scalar`] explicitly requested the portable kernel.
    ScalarRequested,
    /// The modulus is at least `2^31`, requiring Montgomery butterflies.
    MontgomeryFallback,
}

impl fmt::Display for NttPerformanceWarning {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ScalarRequested => formatter.write_str("the scalar backend was requested"),
            Self::MontgomeryFallback => {
                formatter.write_str("the modulus requires scalar Montgomery butterflies")
            }
        }
    }
}

#[derive(Clone, Copy)]
struct Twiddle {
    canonical: u32,
    shoup: u32,
    montgomery: u32,
}

#[derive(Clone)]
struct Stage {
    distance: usize,
    forward: Vec<Twiddle>,
    inverse: Vec<Twiddle>,
}

/// A reusable power-of-two NTT and convolution plan.
///
/// A plan fixes the prime modulus, length, roots, inverse normalization, and
/// selected [`NttBackend`]. Construction allocates and retains `O(N)`
/// stage-ordered twiddles and takes `O(N + log MODULUS)` field operations.
/// Clones share those immutable tables through reference counting. Reusing one
/// avoids setup for subsequent `O(N log N)` transforms and convolutions.
///
/// Transform methods operate in place on exactly `N` [`FieldElement`] values,
/// allocate nothing, and keep both coefficient- and frequency-domain values in
/// Montgomery representation. Use [`Self::elements`] to convert `u32` input and
/// [`FieldElement::value`] after inversion to recover canonical residues.
#[derive(Clone)]
pub struct NttPlan<const MODULUS: u32> {
    field: PrimeField<MODULUS>,
    length: usize,
    inverse_length: FieldElement<MODULUS>,
    stages: Arc<[Stage]>,
    backend: NttBackend,
    warning: Option<NttPerformanceWarning>,
}

/// One natural-order operand transformed for repeated linear convolutions.
///
/// This value borrows the [`NttPlan`] that transformed it. Its private frequency
/// storage therefore remains tied to the field modulus, transform length, root
/// convention, bit-reversed ordering, and backend that subsequent calls use.
/// Construction allocates and transforms one plan-length buffer once.
pub struct PretransformedLinearOperand<'plan, const MODULUS: u32> {
    plan: &'plan NttPlan<MODULUS>,
    values: Vec<FieldElement<MODULUS>>,
    coefficient_length: usize,
}

/// Reusable work storage for [`PretransformedLinearOperand::convolve`].
///
/// Construct this with [`PretransformedLinearOperand::workspace`]. Allocation
/// occurs once at construction; convolution calls only overwrite the retained
/// plan-length buffer.
pub struct LinearConvolutionWorkspace<const MODULUS: u32> {
    values: Vec<FieldElement<MODULUS>>,
}

/// A reusable fixed-size convolution plan modulo `x^N + 1`.
///
/// Construction obtains a primitive `2N`-th root, builds an [`NttPlan`] of
/// length `N`, and retains `N` forward and inverse twist factors. It takes
/// `O(N + log MODULUS)` field operations and `O(N)` storage. Reuse a plan for
/// repeated negacyclic products; each product then takes `O(N log N)` time.
/// Clones share the NTT stage tables through reference counting and clone
/// the twist vectors.
#[derive(Clone)]
pub struct NegacyclicPlan<const MODULUS: u32> {
    ntt: NttPlan<MODULUS>,
    twist: Vec<FieldElement<MODULUS>>,
    inverse_twist: Vec<FieldElement<MODULUS>>,
}
