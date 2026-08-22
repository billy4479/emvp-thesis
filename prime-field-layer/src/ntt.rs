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
//! # Compile-time transform length
//!
//! [`StaticNttPlan`] evaluates roots, twiddles, and stage layout at compile time.
//! Its array API makes transform-size mismatches compile-time type errors:
//!
//! ```
//! use prime_field_layer::StaticNttPlan;
//!
//! let plan = StaticNttPlan::<17, 8>::new()?;
//! let mut values = plan.elements(&[1, 2, 3, 4, 5, 6, 7, 8]);
//! plan.forward(&mut values);
//! plan.inverse(&mut values);
//! assert_eq!(values.map(|value| value.value()), [1, 2, 3, 4, 5, 6, 7, 8]);
//! # Ok::<(), prime_field_layer::FieldError>(())
//! ```
//!
//! The arithmetic kernels are designed without coefficient-dependent control
//! flow. Dispatch, allocation, errors, and loop counts depend on public modulus,
//! platform, and slice lengths. This is a code-level timing precaution, not a
//! claim of a formally audited constant-time implementation.
//!
//! The implementation was independently written; no third-party source was
//! copied. See `NTT_REFERENCES.md` in the crate root for the design literature
//! and provenance statement.

use std::{fmt, sync::Arc};

use crate::{FieldElement, FieldError, PrimeField, constant_time::reduce_once_u64};

#[cfg(target_arch = "x86_64")]
mod avx2;
mod static_plan;

pub use static_plan::StaticNttPlan;

/// Arithmetic and instruction-set implementation selected for an [`NttPlan`].
///
/// Selection occurs once during construction. Inspect this value for
/// diagnostics or benchmarking, not to infer transform ordering or results:
/// all variants implement the same field transform. This describes butterfly
/// dispatch only; pointwise products and inverse normalization dispatch through
/// [`PrimeField`] independently.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NttBackend {
    /// Scalar lazy Harvey Shoup butterflies with one correction per butterfly.
    ///
    /// Forward stages keep residues in `[0, 4p)` and inverse stages in
    /// `[0, 2p)`, so each butterfly needs a single conditional halving and the
    /// lazy products absorb unreduced inputs. Used for `p < 2^30` when scalar
    /// transforms are requested or AVX2 is not selected. Public transform
    /// outputs are normalized to Montgomery residues in `[0, p)`.
    ScalarShoupLazy,
    /// Scalar Shoup butterflies reduced to `[0, p)` after each butterfly.
    ///
    /// Used for `2^30 <= p < 2^31`, where the lazy kernel's bounds do not fit.
    ScalarShoup,
    /// Scalar Montgomery butterflies for moduli at least `2^31`.
    ///
    /// This general fallback supports wide `u32` primes but does not use the
    /// Shoup or AVX2 transform kernels.
    ScalarMontgomery,
    /// LLVM-vectorized AVX2 lazy Harvey Shoup butterflies, one correction per
    /// butterfly, with the same `[0, 4p)` forward and `[0, 2p)` inverse lazy
    /// intervals as [`NttBackend::ScalarShoupLazy`].
    ///
    /// Available only on x86-64 after runtime AVX2 detection, for
    /// `p < 2^30`. Public transform outputs are normalized to `[0, p)`.
    Avx2ShoupLazy,
    /// LLVM-vectorized AVX2 Shoup butterflies reduced to `[0, p)` after each
    /// butterfly.
    ///
    /// Available only on x86-64 after runtime AVX2 detection, for
    /// `2^30 <= p < 2^31`.
    Avx2Shoup,
}

/// A typed explanation for scalar-transform fallback or explicit selection.
///
/// [`NttPlan::performance_warning`] reports at most one construction-time
/// reason. `None` means the selected automatic or forced backend has no warning;
/// it does not guarantee that every operation is vectorized.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum NttPerformanceWarning {
    /// Runtime AVX2 detection failed, or the target architecture has no AVX2
    /// backend.
    Avx2Unavailable,
    /// The transform has fewer than 16 elements, below automatic AVX2 use.
    TransformTooShortForAvx2,
    /// [`NttPlan::new_scalar`] explicitly requested a portable scalar backend.
    ScalarRequested,
    /// The modulus is at least `2^31`, requiring scalar Montgomery butterflies.
    MontgomeryFallback,
}

impl fmt::Display for NttPerformanceWarning {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Avx2Unavailable => formatter.write_str("AVX2 Shoup butterflies are unavailable"),
            Self::TransformTooShortForAvx2 => {
                formatter.write_str("the transform is too short for AVX2 dispatch")
            }
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
/// runtime-selected [`NttBackend`]. Construction allocates and retains `O(N)`
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

impl<const MODULUS: u32> NttPlan<MODULUS> {
    /// Constructs a plan using automatic scalar or AVX2 dispatch.
    ///
    /// `length` must be a nonzero power of two dividing `MODULUS - 1`. Setup
    /// takes `O(length + log MODULUS)` field operations and uses `O(length)`
    /// retained and temporary storage for roots and stage twiddles. Prefer this
    /// constructor for normal use, then reuse the plan across operations.
    ///
    /// On x86-64 with runtime AVX2 support and `MODULUS < 2^31`, automatic
    /// dispatch uses compiler-vectorized AVX2 transforms from length 16. The
    /// lazy kernel covers moduli below `2^30`; larger moduli use reduced Shoup
    /// butterflies. Other targets select the applicable scalar backend.
    /// Use [`Self::new_scalar`] for
    /// portable benchmarking or [`Self::new_avx2`] to require AVX2 butterflies.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::UnsupportedTransformLength`] for an unsupported
    /// length.
    pub fn new(length: usize) -> Result<Self, FieldError> {
        Self::with_backend(length, BackendPreference::Auto)
    }

    /// Constructs a plan that requests portable scalar transform kernels.
    ///
    /// Length requirements, `O(length + log MODULUS)` setup time, and
    /// `O(length)` allocation are the same as for [`Self::new`]. This is useful
    /// for reproducible deployment and backend comparisons; prefer [`Self::new`]
    /// when runtime acceleration is wanted. Modulus bounds still choose among
    /// lazy Shoup, reduced Shoup, and Montgomery scalar arithmetic.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::UnsupportedTransformLength`] for an unsupported
    /// length. [`Self::performance_warning`] normally reports
    /// [`NttPerformanceWarning::ScalarRequested`], except when a wide modulus
    /// produces a more specific fallback warning.
    pub fn new_scalar(length: usize) -> Result<Self, FieldError> {
        Self::with_backend(length, BackendPreference::Scalar)
    }

    /// Constructs a plan that requires AVX2 Shoup butterflies.
    ///
    /// This is intended for testing and platform tuning; [`Self::new`] is
    /// preferable for normal dispatch. It uses AVX2 at every supported length,
    /// including lengths where automatic plans use scalar butterflies. Setup is
    /// `O(length + log MODULUS)` with `O(length)` retained and temporary storage.
    ///
    /// # Errors
    ///
    /// In addition to [`FieldError::UnsupportedTransformLength`], this returns
    /// [`FieldError::Avx2Unavailable`] unless the target is x86-64, AVX2
    /// is detected at runtime, and `MODULUS < 2^31`. No plan is returned with a
    /// scalar fallback.
    pub fn new_avx2(length: usize) -> Result<Self, FieldError> {
        Self::with_backend(length, BackendPreference::Avx2)
    }

    fn with_backend(length: usize, preference: BackendPreference) -> Result<Self, FieldError> {
        let field = PrimeField::<MODULUS>::new();
        let root = field.root_of_unity(length)?;
        let inverse_root = field.inv(root)?;
        let inverse_length = field.element(u64::from(
            field.inv((length as u64 % u64::from(MODULUS)) as u32)?,
        ));
        let (backend, warning) = select_backend::<MODULUS>(length, preference)?;
        let forward_powers = twiddle_powers(field, root, length / 2);
        let inverse_powers = twiddle_powers(field, inverse_root, length / 2);

        let mut stages = Vec::with_capacity(length.trailing_zeros() as usize);
        let mut distance = length / 2;
        while distance != 0 {
            let blocks = length / (2 * distance);
            let bits = blocks.trailing_zeros();
            let mut forward = Vec::with_capacity(blocks);
            let mut inverse = Vec::with_capacity(blocks);
            for block in 0..blocks {
                let reversed = if bits == 0 {
                    0
                } else {
                    block.reverse_bits() >> (usize::BITS - bits)
                };
                let exponent = reversed * distance;
                forward.push(forward_powers[exponent]);
                inverse.push(inverse_powers[exponent]);
            }
            stages.push(Stage {
                distance,
                forward,
                inverse,
            });
            distance /= 2;
        }

        Ok(Self {
            field,
            length,
            inverse_length,
            stages: stages.into(),
            backend,
            warning,
        })
    }

    /// Returns the exact number of elements accepted by transform operations.
    ///
    /// This `O(1)` accessor allocates nothing. The value is always a nonzero
    /// power of two supported by the field.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.length
    }

    /// Returns whether this plan transforms zero elements.
    ///
    /// This always returns `false`: zero-length plans cannot be constructed.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Returns the butterfly backend selected at setup.
    ///
    /// This `O(1)` diagnostic does not perform runtime detection again. All
    /// backends have identical transform semantics and ordering.
    #[must_use]
    pub const fn backend(&self) -> NttBackend {
        self.backend
    }

    /// Returns a diagnostic for scalar selection or fallback, if applicable.
    ///
    /// This `O(1)` accessor allocates nothing.
    #[must_use]
    pub const fn performance_warning(&self) -> Option<NttPerformanceWarning> {
        self.warning
    }

    /// Converts `u32` coefficients to Montgomery-form field elements.
    ///
    /// Every input is reduced modulo `MODULUS`; inputs need not be canonical.
    /// The output preserves input order and length and is not transformed. This
    /// takes `O(values.len())` time and allocates one output vector. Its length
    /// is not required to match the plan until passed to [`Self::forward`],
    /// [`Self::inverse`], or [`Self::pointwise_mul_assign`]. For an existing
    /// buffer, construct elements with [`PrimeField::element_u32`] instead.
    #[must_use]
    pub fn elements(&self, values: &[u32]) -> Vec<FieldElement<MODULUS>> {
        values
            .iter()
            .map(|&value| self.field.element_u32(value))
            .collect()
    }

    /// Transforms one operand for repeated allocation-free linear convolution.
    ///
    /// `operand` is interpreted in natural coefficient order and may contain
    /// unreduced values. It is zero-padded to this plan's length and transformed
    /// once. The returned value borrows this plan so its transform-domain data
    /// cannot be used with a plan having another length, root convention, or
    /// ordering.
    ///
    /// Construct a [`LinearConvolutionWorkspace`] from the returned value, then
    /// reuse both with [`PretransformedLinearOperand::convolve`]. Construction
    /// allocates one plan-length vector and takes `O(N log N)` time.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::ConvolutionLengthOverflow`] if rounding the operand
    /// length to a transform length overflows, and [`FieldError::PlanTooSmall`]
    /// if the padded operand does not fit this plan. An empty operand is valid
    /// and preserves the empty-result semantics of [`Self::linear_convolution`].
    pub fn pretransform_linear_operand(
        &self,
        operand: &[u32],
    ) -> Result<PretransformedLinearOperand<'_, MODULUS>, FieldError> {
        if !operand.is_empty() {
            self.check_convolution_capacity(operand.len())?;
        }

        let mut values = vec![self.field.element_u32(0); self.length];
        for (output, &value) in values.iter_mut().zip(operand) {
            *output = self.field.element_u32(value);
        }
        self.forward(&mut values)?;
        Ok(PretransformedLinearOperand {
            plan: self,
            values,
            coefficient_length: operand.len(),
        })
    }

    /// Applies the forward Cooley-Tukey NTT in place.
    ///
    /// For `omega = PrimeField::root_of_unity(N)`, natural-order input `x[j]`
    /// becomes the DFT `X[k] = sum_j x[j] * omega^(j*k)`, with output slot `i`
    /// holding `X[bit_reverse(i)]`. Values enter and leave as Montgomery-form
    /// [`FieldElement`]s; the output is normalized but remains in the frequency
    /// domain. [`Self::inverse`] consumes this ordering directly.
    ///
    /// The operation takes `O(N log N)` time, allocates nothing, and uses the
    /// backend fixed at plan construction.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::LengthMismatch`] without mutation unless
    /// `values.len()` is
    /// exactly [`Self::len`]. The arithmetic kernels are designed without
    /// coefficient-dependent branches, but are not formally audited for
    /// constant-time behavior.
    pub fn forward(&self, values: &mut [FieldElement<MODULUS>]) -> Result<(), FieldError> {
        self.check_length(values.len())?;
        match self.backend {
            NttBackend::ScalarShoupLazy => {
                self.forward_shoup_lazy(values);
            }
            NttBackend::ScalarShoup => self.forward_shoup(values),
            NttBackend::ScalarMontgomery => self.forward_montgomery(values),
            NttBackend::Avx2ShoupLazy => {
                #[cfg(target_arch = "x86_64")]
                // SAFETY: this backend is selected only after runtime AVX2 detection.
                unsafe {
                    avx2::forward(values, &self.stages);
                }
                #[cfg(not(target_arch = "x86_64"))]
                unreachable!()
            }
            NttBackend::Avx2Shoup => {
                #[cfg(target_arch = "x86_64")]
                // SAFETY: this backend is selected only after runtime AVX2 detection.
                unsafe {
                    avx2::forward_reduced(values, &self.stages);
                }
                #[cfg(not(target_arch = "x86_64"))]
                unreachable!()
            }
        }
        Ok(())
    }

    /// Applies the inverse Gentleman-Sande NTT and normalization in place.
    ///
    /// Input slot `i` must hold frequency `bit_reverse(i)`, as produced by
    /// [`Self::forward`] or a compatible pointwise operation. The method applies
    /// inverse-root butterflies and multiplication by `N^-1`, returning natural
    /// coefficient order. Values remain Montgomery-form [`FieldElement`]s; call
    /// [`FieldElement::value`] for canonical `u32` residues.
    ///
    /// The operation takes `O(N log N)` time and allocates nothing. It uses the
    /// plan's transform backend, then uses the field's independently
    /// runtime-dispatched bulk scalar multiplication for normalization.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::LengthMismatch`] without mutation unless the slice
    /// length is exactly [`Self::len`]. The arithmetic kernels are designed
    /// without coefficient-dependent branches, but are not formally audited.
    pub fn inverse(&self, values: &mut [FieldElement<MODULUS>]) -> Result<(), FieldError> {
        self.check_length(values.len())?;
        match self.backend {
            NttBackend::ScalarShoupLazy => {
                self.inverse_shoup_lazy(values);
            }
            NttBackend::ScalarShoup => self.inverse_shoup(values),
            NttBackend::ScalarMontgomery => self.inverse_montgomery(values),
            NttBackend::Avx2ShoupLazy => {
                #[cfg(target_arch = "x86_64")]
                // SAFETY: this backend is selected only after runtime AVX2 detection.
                unsafe {
                    avx2::inverse(values, &self.stages);
                }
                #[cfg(not(target_arch = "x86_64"))]
                unreachable!()
            }
            NttBackend::Avx2Shoup => {
                #[cfg(target_arch = "x86_64")]
                // SAFETY: this backend is selected only after runtime AVX2 detection.
                unsafe {
                    avx2::inverse_reduced(values, &self.stages);
                }
                #[cfg(not(target_arch = "x86_64"))]
                unreachable!()
            }
        }
        self.field
            .scalar_mul_elements_assign(values, self.inverse_length);
        Ok(())
    }

    /// Multiplies two bit-reversed transform-domain vectors element-wise.
    ///
    /// `lhs` and `rhs` should be outputs of compatible forward transforms with
    /// this modulus, length, root convention, and ordering. Each `lhs[i]` is
    /// replaced by `lhs[i] * rhs[i]`; `rhs` is unchanged and results remain in
    /// Montgomery representation. Apply [`Self::inverse`] to obtain a cyclic
    /// convolution. Prefer [`Self::cyclic_convolution`] for a one-call product.
    ///
    /// This takes `O(N)` time and allocates nothing. It calls
    /// [`PrimeField::mul_elements_assign`], whose runtime AVX2 dispatch is
    /// independent of the transform backend.
    ///
    /// # Errors
    ///
    /// If either length differs from [`Self::len`],
    /// [`FieldError::LengthMismatch`] is returned before mutation.
    pub fn pointwise_mul_assign(
        &self,
        lhs: &mut [FieldElement<MODULUS>],
        rhs: &[FieldElement<MODULUS>],
    ) -> Result<(), FieldError> {
        self.check_length(lhs.len())?;
        self.check_length(rhs.len())?;
        self.field.mul_elements_assign(lhs, rhs)
    }

    /// Computes a linear polynomial convolution using this reusable NTT plan.
    ///
    /// Inputs are natural-order `u32` coefficients and may be unreduced. For
    /// nonempty lengths `m` and `n`, the returned canonical residues have length
    /// `m + n - 1`, with coefficient `k` equal to
    /// `sum_(i+j=k) lhs[i] * rhs[j] mod MODULUS`. Empty input returns an empty
    /// vector. Zero padding prevents cyclic wraparound.
    ///
    /// The method always uses this plan, even for small inputs. It allocates two
    /// plan-length work vectors plus the returned vector and takes
    /// `O(N log N)` time, where `N = self.len()`. Reuse it for repeated products
    /// fitting the same plan; prefer free [`linear_convolution`] for one-shot or
    /// small products because that function chooses the transform size and may
    /// use schoolbook multiplication.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::ConvolutionLengthOverflow`] if the result or rounded
    /// transform length overflows `usize`, and [`FieldError::PlanTooSmall`] if
    /// `self.len()` is smaller than the next power of two covering the result.
    pub fn linear_convolution(&self, lhs: &[u32], rhs: &[u32]) -> Result<Vec<u32>, FieldError> {
        if lhs.is_empty() || rhs.is_empty() {
            return Ok(Vec::new());
        }
        let result_length = convolution_result_length(lhs.len(), rhs.len())?;
        self.check_convolution_capacity(result_length)?;
        let mut lhs_elements = vec![self.field.element_u32(0); self.length];
        let mut rhs_elements = vec![self.field.element_u32(0); self.length];
        for (output, &value) in lhs_elements.iter_mut().zip(lhs) {
            *output = self.field.element_u32(value);
        }
        for (output, &value) in rhs_elements.iter_mut().zip(rhs) {
            *output = self.field.element_u32(value);
        }
        self.convolve_elements(&mut lhs_elements, &mut rhs_elements)?;
        Ok(lhs_elements[..result_length]
            .iter()
            .map(|value| value.value())
            .collect())
    }

    /// Computes a cyclic convolution modulo `x^N - 1`.
    ///
    /// Both inputs are natural-order `u32` coefficient vectors of exactly `N =
    /// self.len()` and may be unreduced. The returned `N` canonical residues are
    /// the product with terms `x^(N+k)` folded into `x^k`. Use this for fixed-size
    /// cyclic products; use [`Self::linear_convolution`] when wraparound is not
    /// wanted.
    ///
    /// Each call allocates two `N`-element work vectors and a returned vector,
    /// and takes `O(N log N)` time. The plan's twiddles and backend selection are
    /// reused.
    ///
    /// # Errors
    ///
    /// [`FieldError::LengthMismatch`] is returned before convolution if
    /// either input is not exactly length `N`.
    pub fn cyclic_convolution(&self, lhs: &[u32], rhs: &[u32]) -> Result<Vec<u32>, FieldError> {
        self.check_length(lhs.len())?;
        self.check_length(rhs.len())?;
        let mut lhs = self.elements(lhs);
        let mut rhs = self.elements(rhs);
        self.convolve_elements(&mut lhs, &mut rhs)?;
        Ok(lhs.into_iter().map(FieldElement::value).collect())
    }

    fn convolve_elements(
        &self,
        lhs: &mut [FieldElement<MODULUS>],
        rhs: &mut [FieldElement<MODULUS>],
    ) -> Result<(), FieldError> {
        self.forward(lhs)?;
        self.forward(rhs)?;
        self.pointwise_mul_assign(lhs, rhs)?;
        self.inverse(lhs)
    }

    const fn check_length(&self, length: usize) -> Result<(), FieldError> {
        if length == self.length {
            Ok(())
        } else {
            Err(FieldError::LengthMismatch)
        }
    }

    fn check_convolution_capacity(&self, result_length: usize) -> Result<(), FieldError> {
        let required = result_length
            .checked_next_power_of_two()
            .ok_or(FieldError::ConvolutionLengthOverflow)?;
        if required > self.length {
            Err(FieldError::PlanTooSmall {
                required,
                available: self.length,
            })
        } else {
            Ok(())
        }
    }

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
    fn forward_shoup_lazy(&self, values: &mut [FieldElement<MODULUS>]) {
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
    fn inverse_shoup_lazy(&self, values: &mut [FieldElement<MODULUS>]) {
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

    fn forward_shoup(&self, values: &mut [FieldElement<MODULUS>]) {
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
                    lhs_value.set_montgomery(add_mod::<MODULUS>(lhs, product));
                    rhs_value.set_montgomery(sub_mod::<MODULUS>(lhs, product));
                }
            }
        }
    }

    fn inverse_shoup(&self, values: &mut [FieldElement<MODULUS>]) {
        for stage in self.stages.iter().rev() {
            for (block, &twiddle) in stage.inverse.iter().enumerate() {
                let start = block * 2 * stage.distance;
                let (lhs_values, rhs_values) =
                    values[start..start + 2 * stage.distance].split_at_mut(stage.distance);
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
        }
    }

    fn forward_montgomery(&self, values: &mut [FieldElement<MODULUS>]) {
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

    fn inverse_montgomery(&self, values: &mut [FieldElement<MODULUS>]) {
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

impl<const MODULUS: u32> PretransformedLinearOperand<'_, MODULUS> {
    /// Returns the number of coefficients in the fixed operand.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.coefficient_length
    }

    /// Returns whether the fixed operand has no coefficients.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.coefficient_length == 0
    }

    /// Returns the plan length retained by this pretransformed operand.
    #[must_use]
    pub const fn transform_len(&self) -> usize {
        self.plan.len()
    }

    /// Allocates work storage suitable for repeated calls to [`Self::convolve`].
    #[must_use]
    pub fn workspace(&self) -> LinearConvolutionWorkspace<MODULUS> {
        LinearConvolutionWorkspace {
            values: vec![self.plan.field.element_u32(0); self.plan.len()],
        }
    }

    /// Returns the linear-convolution output length for an input length.
    ///
    /// Either empty operand produces length zero. For nonempty operands, this
    /// also checks that the product fits the retained transform length.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::ConvolutionLengthOverflow`] if the result or rounded
    /// transform length overflows, and [`FieldError::PlanTooSmall`] if this plan
    /// cannot hold the zero-padded product.
    pub fn output_len(&self, input_length: usize) -> Result<usize, FieldError> {
        if self.is_empty() || input_length == 0 {
            return Ok(0);
        }
        let result_length = convolution_result_length(self.len(), input_length)?;
        self.plan.check_convolution_capacity(result_length)?;
        Ok(result_length)
    }

    /// Convolves an input with this fixed operand into caller-provided output.
    ///
    /// Inputs use natural coefficient order and may be unreduced. For nonempty
    /// lengths `m` and `n`, `output` must have exactly `m + n - 1` elements.
    /// Empty input or an empty fixed operand requires empty output. Zero padding
    /// gives the same linear, non-wrapping semantics as
    /// [`NttPlan::linear_convolution`].
    /// Safe Rust cannot construct overlapping `input` and `output` slices because
    /// they are borrowed shared and mutable, respectively.
    ///
    /// After [`Self::workspace`] has allocated the work vector, this method
    /// allocates nothing. It performs one forward transform, one pointwise
    /// multiplication, and one inverse transform. The fixed transform is reused.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::LengthMismatch`] before mutation if `output` has the
    /// wrong result length or `workspace` was made for another transform length.
    /// Capacity and overflow errors match [`Self::output_len`].
    pub fn convolve(
        &self,
        input: &[u32],
        output: &mut [u32],
        workspace: &mut LinearConvolutionWorkspace<MODULUS>,
    ) -> Result<(), FieldError> {
        let output_length = self.output_len(input.len())?;
        if output.len() != output_length || workspace.values.len() != self.plan.len() {
            return Err(FieldError::LengthMismatch);
        }
        if output_length == 0 {
            return Ok(());
        }

        workspace.values.fill(self.plan.field.element_u32(0));
        for (element, &value) in workspace.values.iter_mut().zip(input) {
            *element = self.plan.field.element_u32(value);
        }
        self.plan.forward(&mut workspace.values)?;
        self.plan
            .pointwise_mul_assign(&mut workspace.values, &self.values)?;
        self.plan.inverse(&mut workspace.values)?;
        for (output, value) in output.iter_mut().zip(&workspace.values) {
            *output = value.value();
        }
        Ok(())
    }
}

#[inline(always)]
pub(super) const fn stage_twiddle_index(blocks: usize, block: usize) -> usize {
    blocks - 1 + block
}

/// A reusable fixed-size convolution plan modulo `x^N + 1`.
///
/// Construction obtains a primitive `2N`-th root, builds an [`NttPlan`] of
/// length `N`, and retains `N` forward and inverse twist factors. It takes
/// `O(N + log MODULUS)` field operations and `O(N)` storage. Reuse a plan for
/// repeated negacyclic products; each product then takes `O(N log N)` time.
/// Cloning also clones the NTT tables and twist vectors.
#[derive(Clone)]
pub struct NegacyclicPlan<const MODULUS: u32> {
    ntt: NttPlan<MODULUS>,
    twist: Vec<FieldElement<MODULUS>>,
    inverse_twist: Vec<FieldElement<MODULUS>>,
}

impl<const MODULUS: u32> NegacyclicPlan<MODULUS> {
    /// Constructs a negacyclic plan with automatic backend dispatch.
    ///
    /// `length` must be a nonzero power of two and `2 * length` must divide
    /// `MODULUS - 1`, so the field contains the required twist root. Setup takes
    /// `O(length + log MODULUS)` time and retains `O(length)` NTT and twist
    /// storage, plus `O(length)` temporary setup storage. Prefer this constructor
    /// for normal use and reuse the resulting plan.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::UnsupportedTransformLength`] if doubling overflows
    /// or either required root is unsupported. Backend selection otherwise
    /// follows [`NttPlan::new`].
    pub fn new(length: usize) -> Result<Self, FieldError> {
        Self::with_scalar_option(length, false)
    }

    /// Constructs a negacyclic plan requesting portable scalar NTT kernels.
    ///
    /// Root requirements, `O(length + log MODULUS)` setup time, and `O(length)`
    /// storage match [`Self::new`]. Use this for reproducible deployment or
    /// backend comparison; prefer [`Self::new`] for automatic acceleration.
    /// Wide modulus tiers still select their applicable scalar arithmetic.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::UnsupportedTransformLength`] when the length or its
    /// doubled twist order is unsupported.
    pub fn new_scalar(length: usize) -> Result<Self, FieldError> {
        Self::with_scalar_option(length, true)
    }

    fn with_scalar_option(length: usize, scalar: bool) -> Result<Self, FieldError> {
        let field = PrimeField::<MODULUS>::new();
        let doubled = length
            .checked_mul(2)
            .ok_or(FieldError::UnsupportedTransformLength(length))?;
        let psi = field.root_of_unity(doubled)?;
        let inverse_psi = field.inv(psi)?;
        let ntt = if scalar {
            NttPlan::new_scalar(length)?
        } else {
            NttPlan::new(length)?
        };
        let twist = powers(field, psi, length);
        let inverse_twist = powers(field, inverse_psi, length);
        Ok(Self {
            ntt,
            twist,
            inverse_twist,
        })
    }

    /// Returns the exact coefficient count accepted by [`Self::convolution`].
    ///
    /// This `O(1)` accessor allocates nothing and always returns a nonzero power
    /// of two.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.ntt.len()
    }

    /// Returns whether this plan accepts zero coefficients.
    ///
    /// This always returns `false`: zero-length plans cannot be constructed.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Returns the backend selected for the underlying NTT.
    ///
    /// Twisting is scalar field-element multiplication; this value describes
    /// the transform and pointwise dispatch inherited from [`NttPlan`]. The
    /// accessor is `O(1)` and performs no new runtime detection.
    #[must_use]
    pub const fn backend(&self) -> NttBackend {
        self.ntt.backend()
    }

    /// Returns the underlying NTT's scalar-selection warning, if any.
    ///
    /// This `O(1)` diagnostic has the same meaning as
    /// [`NttPlan::performance_warning`]. `None` does not imply that the scalar
    /// twist loops are vectorized.
    #[must_use]
    pub const fn performance_warning(&self) -> Option<NttPerformanceWarning> {
        self.ntt.performance_warning()
    }

    /// Computes a negacyclic convolution modulo `x^N + 1`.
    ///
    /// Both inputs are natural-order `u32` coefficient vectors of exactly `N =
    /// self.len()` and may be unreduced. The returned canonical residues have
    /// length `N`; product terms `c * x^(N+k)` contribute `-c` to coefficient
    /// `k`. Prefer [`NttPlan::cyclic_convolution`] for reduction modulo
    /// `x^N - 1`, or [`linear_convolution`] when no wraparound is wanted.
    ///
    /// Each call creates two `N`-element Montgomery work vectors and returns an
    /// allocated `Vec<u32>`. Twisting, two forward transforms, pointwise
    /// multiplication, inverse transformation, and untwisting take
    /// `O(N log N)` time while reusing all plan tables.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::LengthMismatch`] before any convolution work.
    pub fn convolution(&self, lhs: &[u32], rhs: &[u32]) -> Result<Vec<u32>, FieldError> {
        self.ntt.check_length(lhs.len())?;
        self.ntt.check_length(rhs.len())?;
        let mut lhs = self.ntt.elements(lhs);
        let mut rhs = self.ntt.elements(rhs);
        for ((lhs, rhs), &twist) in lhs.iter_mut().zip(&mut rhs).zip(&self.twist) {
            *lhs *= twist;
            *rhs *= twist;
        }
        self.ntt.convolve_elements(&mut lhs, &mut rhs)?;
        for (value, &twist) in lhs.iter_mut().zip(&self.inverse_twist) {
            *value *= twist;
        }
        Ok(lhs.into_iter().map(FieldElement::value).collect())
    }
}

/// Computes a one-shot linear convolution with automatic schoolbook/NTT dispatch.
///
/// Inputs are natural-order `u32` coefficients and are reduced modulo `MODULUS`.
/// If either input is empty, the result is empty. Otherwise, for input lengths
/// `m` and `n`, the result contains `m + n - 1` canonical residues with
/// coefficient `k` equal to `sum_(i+j=k) lhs[i] * rhs[j] mod MODULUS`.
///
/// Inputs whose shorter side has at most two coefficients, or whose `m * n` is
/// at most 5184, use a Montgomery-domain schoolbook kernel in `O(m * n)` time.
/// Larger products construct the smallest power-of-two [`NttPlan`] covering the
/// result and take `O(L log L)` setup-plus-transform time for that call. Both
/// paths allocate converted inputs and the result; the NTT path also allocates
/// plan tables and length-`L` work vectors. Prefer this API for one-shot and
/// small products. Reuse [`NttPlan::linear_convolution`] when many products fit
/// one transform length; that method always uses its existing NTT plan.
///
/// Dispatch depends on public input lengths, modulus, architecture, and runtime
/// AVX2 support, not coefficient values. The arithmetic kernels are designed
/// without coefficient-dependent branches but have not been formally audited
/// as constant-time. Invalid compile-time moduli fail to compile, including on
/// the schoolbook and empty-input paths.
///
/// # Errors
///
/// Returns [`FieldError::ConvolutionLengthOverflow`] if result sizing overflows,
/// or [`FieldError::UnsupportedTransformLength`] if the field cannot support the
/// required NTT length.
///
/// Invalid compile-time moduli are rejected even for the schoolbook path:
///
/// ```compile_fail
/// let _ = prime_field_layer::linear_convolution::<15>(&[1], &[2]);
/// ```
pub fn linear_convolution<const MODULUS: u32>(
    lhs: &[u32],
    rhs: &[u32],
) -> Result<Vec<u32>, FieldError> {
    let field = PrimeField::<MODULUS>::new();
    if lhs.is_empty() || rhs.is_empty() {
        return Ok(Vec::new());
    }
    let result_length = convolution_result_length(lhs.len(), rhs.len())?;
    let use_schoolbook = lhs.len().min(rhs.len()) <= 2
        || lhs
            .len()
            .checked_mul(rhs.len())
            .is_some_and(|products| products <= SCHOOLBOOK_PRODUCT_CUTOFF);
    if use_schoolbook {
        return Ok(schoolbook_linear_convolution::<MODULUS>(
            field,
            lhs,
            rhs,
            result_length,
        ));
    }
    let length = result_length
        .checked_next_power_of_two()
        .ok_or(FieldError::ConvolutionLengthOverflow)?;
    NttPlan::<MODULUS>::new(length)?.linear_convolution(lhs, rhs)
}

#[derive(Clone, Copy)]
enum BackendPreference {
    Auto,
    Scalar,
    Avx2,
}

const SCHOOLBOOK_PRODUCT_CUTOFF: usize = 5_184;

fn select_backend<const MODULUS: u32>(
    length: usize,
    preference: BackendPreference,
) -> Result<(NttBackend, Option<NttPerformanceWarning>), FieldError> {
    if matches!(preference, BackendPreference::Avx2) && MODULUS >= 1 << 31 {
        return Err(FieldError::Avx2Unavailable);
    }
    if MODULUS >= 1 << 31 {
        return Ok((
            NttBackend::ScalarMontgomery,
            Some(NttPerformanceWarning::MontgomeryFallback),
        ));
    }
    let scalar_backend = if MODULUS < 1 << 30 {
        NttBackend::ScalarShoupLazy
    } else {
        NttBackend::ScalarShoup
    };
    let avx2_backend = if MODULUS < 1 << 30 {
        NttBackend::Avx2ShoupLazy
    } else {
        NttBackend::Avx2Shoup
    };
    if matches!(preference, BackendPreference::Scalar) {
        return Ok((scalar_backend, Some(NttPerformanceWarning::ScalarRequested)));
    }
    #[cfg(target_arch = "x86_64")]
    {
        if matches!(preference, BackendPreference::Avx2) {
            return if std::arch::is_x86_feature_detected!("avx2") {
                Ok((avx2_backend, None))
            } else {
                Err(FieldError::Avx2Unavailable)
            };
        }
        if std::arch::is_x86_feature_detected!("avx2") && length >= 16 {
            return Ok((avx2_backend, None));
        }
    }
    #[cfg(not(target_arch = "x86_64"))]
    if matches!(preference, BackendPreference::Avx2) {
        return Err(FieldError::Avx2Unavailable);
    }
    #[cfg(target_arch = "x86_64")]
    let warning = if length < 16 {
        NttPerformanceWarning::TransformTooShortForAvx2
    } else {
        NttPerformanceWarning::Avx2Unavailable
    };
    #[cfg(not(target_arch = "x86_64"))]
    let warning = NttPerformanceWarning::Avx2Unavailable;
    Ok((scalar_backend, Some(warning)))
}

fn schoolbook_linear_convolution<const MODULUS: u32>(
    field: PrimeField<MODULUS>,
    lhs: &[u32],
    rhs: &[u32],
    result_length: usize,
) -> Vec<u32> {
    let lhs: Vec<_> = lhs
        .iter()
        .copied()
        .map(|value| field.element_u32(value))
        .collect();
    let rhs: Vec<_> = rhs
        .iter()
        .copied()
        .map(|value| field.element_u32(value))
        .collect();
    let mut result = vec![field.element_u32(0); result_length];
    for (lhs_index, &lhs) in lhs.iter().enumerate() {
        for (rhs_index, &rhs) in rhs.iter().enumerate() {
            result[lhs_index + rhs_index] += lhs * rhs;
        }
    }
    result.into_iter().map(FieldElement::value).collect()
}

fn convolution_result_length(lhs_length: usize, rhs_length: usize) -> Result<usize, FieldError> {
    lhs_length
        .checked_add(rhs_length)
        .and_then(|length| length.checked_sub(1))
        .ok_or(FieldError::ConvolutionLengthOverflow)
}

fn make_twiddle<const MODULUS: u32>(field: PrimeField<MODULUS>, canonical: u32) -> Twiddle {
    Twiddle {
        canonical,
        shoup: ((u64::from(canonical) << 32) / u64::from(MODULUS)) as u32,
        montgomery: field.element(u64::from(canonical)).montgomery(),
    }
}

fn twiddle_powers<const MODULUS: u32>(
    field: PrimeField<MODULUS>,
    root: u32,
    count: usize,
) -> Vec<Twiddle> {
    let mut canonical = 1;
    (0..count)
        .map(|_| {
            let twiddle = make_twiddle(field, canonical);
            canonical = field.mul(canonical, root);
            twiddle
        })
        .collect()
}

fn powers<const MODULUS: u32>(
    field: PrimeField<MODULUS>,
    base: u32,
    length: usize,
) -> Vec<FieldElement<MODULUS>> {
    let base = field.element(u64::from(base));
    let mut power = field.element(1);
    (0..length)
        .map(|_| {
            let result = power;
            power *= base;
            result
        })
        .collect()
}

#[inline(always)]
fn shoup_mul<const MODULUS: u32>(value: u32, twiddle: Twiddle) -> u32 {
    reduce_once(shoup_mul_lazy_for::<MODULUS>(value, twiddle), MODULUS)
}

/// Lazy Shoup product of any `u32` word with a twiddle constant.
///
/// This is the wide-input form of Shoup's multiplication (Harvey, J. Symbolic
/// Comput. 60 (2014), section 3; also Bradbury et al., ePrint 2021/1396,
/// Theorem 2, for the 32-bit SIMD lanes used here). For a twiddle `w` in
/// `[0, p)` with precomputed `w' = floor(w * 2^32 / p)`, the quotient
/// `q = (z * w') >> 32` and product `t = z * w - q * p` satisfy
/// `0 <= t < 2p` for every input `z` in `[0, 2^32)`, not only reduced inputs:
/// from `w' <= w * 2^32 / p < w' + 1` follows `q <= z * w / p`, hence `t >= 0`,
/// and `q > z * w / p - z / 2^32 - 1`, hence
/// `t < (z / 2^32 + 1) * p < 2p`. The lazy callers have `p < 2^30` and
/// arbitrary `u32` inputs; the reduced callers have `p < 2^31` and inputs
/// below `p`. Thus `z * w` and `q * p` are below `2^62`. The quotient product
/// `z * w'` is below `2^64` for lazy callers and `2^63` for reduced callers,
/// so every intermediate fits `u64`; `t < 2p < 2^32` fits a `u32` lane.
///
/// This bound is what lets the lazy butterflies feed unreduced lazy words
/// straight into the multiplication: a difference planted with a `+2p` offset
/// stays below `4p <= 2^32` for `p < 2^30`, and the quotient estimate in the
/// high half of the product absorbs the planted offset without any
/// conditional correction of the product.
#[inline(always)]
fn shoup_mul_lazy_for<const MODULUS: u32>(value: u32, twiddle: Twiddle) -> u32 {
    let quotient = (u64::from(value) * u64::from(twiddle.shoup)) >> 32;
    (u64::from(value) * u64::from(twiddle.canonical) - quotient * u64::from(MODULUS)) as u32
}

/// Halves the lazy interval `[0, 4p)` to `[0, 2p)` with one branchless minimum.
///
/// For `value` in `[0, 4p)` the wrapping difference `value - two_p` lies below
/// `value` exactly when no borrow occurs, i.e. when `value >= two_p`; when
/// `value < two_p` it wraps far above `value`. The `u32::min` selection
/// therefore returns `value - two_p` on `[2p, 4p)` and `value` unchanged on
/// `[0, 2p)`, with no coefficient-dependent branch. Vector code lowers it to a
/// single `vpminud` beside the subtraction.
#[inline(always)]
fn halve_interval(value: u32, two_p: u32) -> u32 {
    value.min(value.wrapping_sub(two_p))
}

#[inline(always)]
fn reduce_once(value: u32, modulus: u32) -> u32 {
    reduce_once_u64(u64::from(value), u64::from(modulus)) as u32
}

#[inline(always)]
fn add_mod<const MODULUS: u32>(lhs: u32, rhs: u32) -> u32 {
    reduce_once_u64(u64::from(lhs) + u64::from(rhs), u64::from(MODULUS)) as u32
}

#[inline(always)]
fn sub_mod<const MODULUS: u32>(lhs: u32, rhs: u32) -> u32 {
    reduce_once_u64(
        u64::from(lhs) + u64::from(MODULUS) - u64::from(rhs),
        u64::from(MODULUS),
    ) as u32
}

/// Restores canonical `[0, p)` Montgomery words from lazy stage residues.
///
/// Forward lazy stages emit values in `[0, 4p)` and inverse lazy stages values
/// in `[0, 2p)`; halving by `2p` (a no-op for the inverse interval) followed
/// by one `p` correction covers both.
fn normalize<const MODULUS: u32>(values: &mut [FieldElement<MODULUS>]) {
    let two_p = MODULUS * 2;
    for value in values {
        let halved = halve_interval(value.montgomery(), two_p);
        value.set_montgomery(reduce_once(halved, MODULUS));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn forced_avx2_matches_forced_scalar() {
        fn check<const MODULUS: u32>() {
            let scalar = NttPlan::<MODULUS>::new_scalar(256).unwrap();
            let avx2 = NttPlan::<MODULUS>::new_avx2(256).unwrap();
            let boundaries = [0, 1, MODULUS / 2, MODULUS - 2, MODULUS - 1];
            let input: Vec<_> = (0_u64..256)
                .map(|index| {
                    if index < boundaries.len() as u64 {
                        boundaries[index as usize]
                    } else {
                        ((index * 2_654_435_761 + 97) % u64::from(MODULUS)) as u32
                    }
                })
                .collect();
            let mut scalar_values = scalar.elements(&input);
            let mut avx2_values = avx2.elements(&input);
            scalar.forward(&mut scalar_values).unwrap();
            avx2.forward(&mut avx2_values).unwrap();
            assert_eq!(avx2_values, scalar_values);

            let scalar_rhs = scalar_values.clone();
            let avx2_rhs = avx2_values.clone();
            scalar
                .pointwise_mul_assign(&mut scalar_values, &scalar_rhs)
                .unwrap();
            avx2.pointwise_mul_assign(&mut avx2_values, &avx2_rhs)
                .unwrap();
            assert_eq!(avx2_values, scalar_values);
            scalar.inverse(&mut scalar_values).unwrap();
            avx2.inverse(&mut avx2_values).unwrap();
            assert_eq!(avx2_values, scalar_values);
        }

        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        check::<998_244_353>();
        check::<2_013_265_921>();
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_lazy_boundaries_match_scalar_and_normalize_output() {
        const MODULUS: u32 = 998_244_353;

        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let scalar = NttPlan::<MODULUS>::new_scalar(256).unwrap();
        let avx2 = NttPlan::<MODULUS>::new_avx2(256).unwrap();
        // Every endpoint of the [0, 4p) forward lazy interval, plus the
        // interior boundaries at p and 3p.
        let boundaries = [
            0,
            1,
            MODULUS - 1,
            MODULUS,
            2 * MODULUS - 2,
            2 * MODULUS - 1,
            2 * MODULUS,
            3 * MODULUS - 1,
            4 * MODULUS - 2,
            4 * MODULUS - 1,
        ];
        let mut state = 0xd1b5_4a32_d192_ed03u64;
        let mut scalar_values = vec![scalar.field.element(0); 256];
        for (index, value) in scalar_values.iter_mut().enumerate() {
            state ^= state << 7;
            state ^= state >> 9;
            let raw = if index < boundaries.len() {
                boundaries[index]
            } else {
                (state % u64::from(4 * MODULUS)) as u32
            };
            value.set_montgomery(raw);
        }
        let mut avx2_values = scalar_values.clone();
        scalar.forward(&mut scalar_values).unwrap();
        avx2.forward(&mut avx2_values).unwrap();
        assert_eq!(avx2_values, scalar_values);
        assert!(avx2_values.iter().all(|value| value.montgomery() < MODULUS));
        scalar.inverse(&mut scalar_values).unwrap();
        avx2.inverse(&mut avx2_values).unwrap();
        assert_eq!(avx2_values, scalar_values);
        assert!(avx2_values.iter().all(|value| value.montgomery() < MODULUS));
    }

    #[test]
    #[cfg(target_arch = "x86_64")]
    fn avx2_lazy_inverse_boundaries_match_scalar_and_normalize_output() {
        const MODULUS: u32 = 998_244_353;

        if !std::arch::is_x86_feature_detected!("avx2") {
            return;
        }
        let scalar = NttPlan::<MODULUS>::new_scalar(256).unwrap();
        let avx2 = NttPlan::<MODULUS>::new_avx2(256).unwrap();
        // Endpoints of the [0, 2p) inverse lazy interval, seeded directly
        // into the inverse transform without a prior forward pass.
        let boundaries = [0, 1, MODULUS - 1, MODULUS, 2 * MODULUS - 2, 2 * MODULUS - 1];
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut scalar_values = vec![scalar.field.element(0); 256];
        for (index, value) in scalar_values.iter_mut().enumerate() {
            state ^= state << 7;
            state ^= state >> 9;
            state ^= state << 8;
            let raw = if index < boundaries.len() {
                boundaries[index]
            } else {
                (state % u64::from(2 * MODULUS)) as u32
            };
            value.set_montgomery(raw);
        }
        let mut avx2_values = scalar_values.clone();
        scalar.inverse(&mut scalar_values).unwrap();
        avx2.inverse(&mut avx2_values).unwrap();
        assert_eq!(avx2_values, scalar_values);
        assert!(avx2_values.iter().all(|value| value.montgomery() < MODULUS));
    }

    #[test]
    fn transform_outputs_have_canonical_montgomery_words() {
        fn check<const MODULUS: u32>() {
            let plan = NttPlan::<MODULUS>::new_scalar(256).unwrap();
            let input: Vec<_> = (0_u64..256)
                .map(|index| ((index * 1_103_515_245 + 12_345) % u64::from(MODULUS)) as u32)
                .collect();
            let mut values = plan.elements(&input);
            plan.forward(&mut values).unwrap();
            assert!(values.iter().all(|value| value.montgomery() < MODULUS));
            plan.inverse(&mut values).unwrap();
            assert!(values.iter().all(|value| value.montgomery() < MODULUS));
        }

        check::<998_244_353>();
        check::<2_013_265_921>();
        check::<2_281_701_377>();
    }

    #[test]
    fn incremental_twiddle_table_preserves_stage_mapping() {
        let field = PrimeField::<65_537>::new();
        let plan = NttPlan::<65_537>::new_scalar(256).unwrap();
        let root = field.root_of_unity(256).unwrap();
        for stage in plan.stages.iter() {
            let bits = stage.forward.len().trailing_zeros();
            for (block, twiddle) in stage.forward.iter().enumerate() {
                let reversed = if bits == 0 {
                    0
                } else {
                    block.reverse_bits() >> (usize::BITS - bits)
                };
                assert_eq!(
                    twiddle.canonical,
                    field.pow(root, (reversed * stage.distance) as u64)
                );
            }
        }
    }

    #[test]
    fn convolution_length_overflow_has_a_distinct_error() {
        assert_eq!(
            convolution_result_length(usize::MAX, 2),
            Err(FieldError::ConvolutionLengthOverflow)
        );
    }
}
