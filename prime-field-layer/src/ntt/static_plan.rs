use super::{
    BackendPreference, NttBackend, NttPerformanceWarning, Twiddle, add_mod, halve_interval,
    normalize, select_backend, shoup_mul, shoup_mul_lazy_for, stage_twiddle_index, sub_mod,
};
use crate::{FieldElement, FieldError, PrimeField};

mod scalar;
mod tables;

use tables::{pow_mod, root_of_unity, stage_twiddles, to_montgomery_const};

/// An NTT plan whose modulus and transform length are compile-time parameters.
///
/// Twiddles, stage layout, roots, and inverse normalization are evaluated by the
/// compiler for every `(MODULUS, N)` instantiation. The value itself stores only
/// runtime CPU dispatch state and allocates no plan tables. Each instantiated
/// pair whose transform methods are instantiated contributes approximately
/// `24 * N` bytes of forward and inverse tables to the binary before linker
/// deduplication.
///
/// Transform methods accept arrays, so a size mismatch is a type error. Invalid
/// field and transform-size parameters fail during constant evaluation:
///
/// ```compile_fail
/// use prime_field_layer::StaticNttPlan;
///
/// let _ = StaticNttPlan::<17, 3>::new();
/// ```
#[derive(Clone, Copy)]
pub struct StaticNttPlan<const MODULUS: u32, const N: usize> {
    field: PrimeField<MODULUS>,
    backend: NttBackend,
    warning: Option<NttPerformanceWarning>,
}

impl<const MODULUS: u32, const N: usize> StaticNttPlan<MODULUS, N> {
    const FIELD: PrimeField<MODULUS> = {
        assert!(
            N.is_power_of_two() && N.trailing_zeros() <= (MODULUS - 1).trailing_zeros(),
            "NTT length must be a nonzero power of two dividing MODULUS - 1"
        );
        PrimeField::<MODULUS>::new()
    };
    const ROOT: u32 = root_of_unity::<MODULUS, N>();
    const INVERSE_ROOT: u32 = pow_mod(Self::ROOT, (MODULUS - 2) as u64, MODULUS);
    const FORWARD_TWIDDLES: [Twiddle; N] = stage_twiddles::<MODULUS, N>(Self::ROOT);
    const INVERSE_TWIDDLES: [Twiddle; N] = stage_twiddles::<MODULUS, N>(Self::INVERSE_ROOT);
    const INVERSE_LENGTH: FieldElement<MODULUS> =
        FieldElement::from_montgomery(to_montgomery_const::<MODULUS>(pow_mod(
            (N as u64 % MODULUS as u64) as u32,
            (MODULUS - 2) as u64,
            MODULUS,
        )));

    /// Constructs a fixed-size plan using automatic backend dispatch.
    ///
    /// Setup performs CPU feature detection but allocates and computes no
    /// tables. On x86-64, supported Shoup transforms use LLVM-vectorized AVX2
    /// kernels from length 16.
    ///
    /// # Errors
    ///
    /// This constructor has no runtime error for a valid `(MODULUS, N)` pair.
    /// An invalid pair fails during constant evaluation.
    pub fn new() -> Result<Self, FieldError> {
        Self::with_backend(BackendPreference::Auto)
    }

    /// Constructs a fixed-size plan requesting portable scalar kernels.
    ///
    /// # Errors
    ///
    /// This constructor has no runtime error for a valid `(MODULUS, N)` pair.
    /// An invalid pair fails during constant evaluation.
    pub fn new_scalar() -> Result<Self, FieldError> {
        Self::with_backend(BackendPreference::Scalar)
    }

    /// Constructs a fixed-size plan requiring AVX2 transform kernels.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::Avx2Unavailable`] when AVX2 Shoup transforms are
    /// unavailable for the current CPU, target, or modulus.
    pub fn new_avx2() -> Result<Self, FieldError> {
        Self::with_backend(BackendPreference::Avx2)
    }

    fn with_backend(preference: BackendPreference) -> Result<Self, FieldError> {
        let field = Self::FIELD;
        let (backend, warning) = select_backend::<MODULUS>(N, preference)?;
        Ok(Self {
            field,
            backend,
            warning,
        })
    }

    /// Returns the compile-time transform length.
    #[must_use]
    pub const fn len(&self) -> usize {
        N
    }

    /// Returns whether this plan transforms zero elements.
    ///
    /// This is always false because zero is not a valid transform length.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        false
    }

    /// Returns the selected butterfly backend.
    #[must_use]
    pub const fn backend(&self) -> NttBackend {
        self.backend
    }

    /// Returns a scalar-selection or fallback diagnostic, when applicable.
    #[must_use]
    pub const fn performance_warning(&self) -> Option<NttPerformanceWarning> {
        self.warning
    }

    /// Converts exactly `N` canonical or unreduced coefficients to field elements.
    #[must_use]
    pub fn elements(&self, values: &[u32; N]) -> [FieldElement<MODULUS>; N] {
        values.map(|value| self.field.element_u32(value))
    }

    /// Applies the fixed-size forward transform in place.
    ///
    /// Input is in natural coefficient order and output is in bit-reversed
    /// frequency order, matching [`super::NttPlan::forward`]. This allocates
    /// nothing and executes a fixed `O(N log N)` butterfly schedule.
    pub fn forward(&self, values: &mut [FieldElement<MODULUS>; N]) {
        match self.backend {
            NttBackend::ScalarShoupLazy => {
                Self::forward_shoup_lazy(values);
            }
            NttBackend::ScalarShoup => Self::forward_shoup(values),
            NttBackend::ScalarMontgomery => Self::forward_montgomery(values),
            NttBackend::Avx2ShoupLazy => {
                #[cfg(target_arch = "x86_64")]
                // SAFETY: construction selects this backend only after AVX2 detection.
                unsafe {
                    super::avx2::forward_static(values, &Self::FORWARD_TWIDDLES);
                }
                #[cfg(not(target_arch = "x86_64"))]
                unreachable!()
            }
            NttBackend::Avx2Shoup => {
                #[cfg(target_arch = "x86_64")]
                // SAFETY: construction selects this backend only after AVX2 detection.
                unsafe {
                    super::avx2::forward_reduced_static(values, &Self::FORWARD_TWIDDLES);
                }
                #[cfg(not(target_arch = "x86_64"))]
                unreachable!()
            }
        }
    }

    /// Applies the fixed-size inverse transform and normalization in place.
    ///
    /// This consumes bit-reversed frequency order and restores natural
    /// coefficient order. It allocates nothing.
    pub fn inverse(&self, values: &mut [FieldElement<MODULUS>; N]) {
        match self.backend {
            NttBackend::ScalarShoupLazy => {
                Self::inverse_shoup_lazy(values);
            }
            NttBackend::ScalarShoup => Self::inverse_shoup(values),
            NttBackend::ScalarMontgomery => Self::inverse_montgomery(values),
            NttBackend::Avx2ShoupLazy => {
                #[cfg(target_arch = "x86_64")]
                // SAFETY: construction selects this backend only after AVX2 detection.
                unsafe {
                    super::avx2::inverse_static(values, &Self::INVERSE_TWIDDLES);
                }
                #[cfg(not(target_arch = "x86_64"))]
                unreachable!()
            }
            NttBackend::Avx2Shoup => {
                #[cfg(target_arch = "x86_64")]
                // SAFETY: construction selects this backend only after AVX2 detection.
                unsafe {
                    super::avx2::inverse_reduced_static(values, &Self::INVERSE_TWIDDLES);
                }
                #[cfg(not(target_arch = "x86_64"))]
                unreachable!()
            }
        }
        self.field
            .scalar_mul_elements_assign(values, Self::INVERSE_LENGTH);
    }

    /// Multiplies two fixed-size transform-domain arrays element-wise.
    pub fn pointwise_mul_assign(
        &self,
        lhs: &mut [FieldElement<MODULUS>; N],
        rhs: &[FieldElement<MODULUS>; N],
    ) {
        PrimeField::<MODULUS>::mul_element_arrays_assign(lhs, rhs);
    }

    /// Applies two forward transforms, a pointwise product, and one inverse.
    ///
    /// Both arrays are overwritten. On return, `lhs` contains their cyclic
    /// convolution and `rhs` remains in the transform domain.
    pub fn convolve_elements(
        &self,
        lhs: &mut [FieldElement<MODULUS>; N],
        rhs: &mut [FieldElement<MODULUS>; N],
    ) {
        self.forward(lhs);
        self.forward(rhs);
        self.pointwise_mul_assign(lhs, rhs);
        self.inverse(lhs);
    }
}
