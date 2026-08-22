use super::{
    BackendPreference, NttBackend, NttPerformanceWarning, Twiddle, add_mod, halve_interval,
    normalize, select_backend, shoup_mul, shoup_mul_lazy_for, stage_twiddle_index, sub_mod,
};
use crate::{FieldElement, FieldError, PrimeField};

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

    #[inline(always)]
    fn forward_shoup_lazy(values: &mut [FieldElement<MODULUS>; N]) {
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
    fn inverse_shoup_lazy(values: &mut [FieldElement<MODULUS>; N]) {
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
    fn forward_shoup(values: &mut [FieldElement<MODULUS>; N]) {
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
    fn inverse_shoup(values: &mut [FieldElement<MODULUS>; N]) {
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
    fn forward_montgomery(values: &mut [FieldElement<MODULUS>; N]) {
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
    fn inverse_montgomery(values: &mut [FieldElement<MODULUS>; N]) {
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

const fn pow_mod(base: u32, mut exponent: u64, modulus: u32) -> u32 {
    let mut result = 1u64;
    let mut base_u64 = base as u64;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = result * base_u64 % modulus as u64;
        }
        exponent >>= 1;
        if exponent != 0 {
            base_u64 = base_u64 * base_u64 % modulus as u64;
        }
    }
    result as u32
}

const fn two_adic_root<const MODULUS: u32>() -> u32 {
    if MODULUS == 2 {
        return 1;
    }
    if (MODULUS - 1).trailing_zeros() == 1 {
        return MODULUS - 1;
    }
    let mut non_residue = 2;
    while pow_mod(non_residue, ((MODULUS - 1) / 2) as u64, MODULUS) == 1 {
        non_residue += 1;
    }
    pow_mod(
        non_residue,
        ((MODULUS - 1) >> (MODULUS - 1).trailing_zeros()) as u64,
        MODULUS,
    )
}

const fn root_of_unity<const MODULUS: u32, const N: usize>() -> u32 {
    let mut root = two_adic_root::<MODULUS>();
    let mut shifts = N.trailing_zeros();
    while shifts < (MODULUS - 1).trailing_zeros() {
        root = (root as u64 * root as u64 % MODULUS as u64) as u32;
        shifts += 1;
    }
    root
}

const fn to_montgomery_const<const MODULUS: u32>(value: u32) -> u32 {
    ((value as u128 * (1u128 << 32)) % MODULUS as u128) as u32
}

const fn twiddle_const<const MODULUS: u32>(canonical: u32) -> Twiddle {
    Twiddle {
        canonical,
        shoup: ((canonical as u64) << 32).div_euclid(MODULUS as u64) as u32,
        montgomery: to_montgomery_const::<MODULUS>(canonical),
    }
}

const fn stage_twiddles<const MODULUS: u32, const N: usize>(root: u32) -> [Twiddle; N] {
    let one = twiddle_const::<MODULUS>(1);
    let mut powers = [one; N];
    let mut canonical = 1;
    let mut index = 0;
    while index < N {
        powers[index] = twiddle_const::<MODULUS>(canonical);
        canonical = (canonical as u64 * root as u64 % MODULUS as u64) as u32;
        index += 1;
    }

    let mut result = [one; N];
    let mut distance = N / 2;
    while distance != 0 {
        let blocks = N / (2 * distance);
        let bits = blocks.trailing_zeros();
        let mut block = 0;
        while block < blocks {
            let reversed = if bits == 0 {
                0
            } else {
                block.reverse_bits() >> (usize::BITS - bits)
            };
            result[stage_twiddle_index(blocks, block)] = powers[reversed * distance];
            block += 1;
        }
        distance /= 2;
    }
    result
}
