use super::{
    FieldElement, FieldError, NttBackend, NttPerformanceWarning, NttPlan, PrimeField, Stage,
    select_backend, twiddle_powers,
};

impl<const MODULUS: u32> NttPlan<MODULUS> {
    /// Constructs a plan with the arithmetic chosen by the modulus.
    ///
    /// `length` must be a nonzero power of two dividing `MODULUS - 1`. Setup
    /// takes `O(length + log MODULUS)` field operations and uses `O(length)`
    /// retained and temporary storage for roots and stage twiddles. Reuse the
    /// plan across operations.
    ///
    /// Modulus bounds choose among lazy Shoup, reduced Shoup, and Montgomery
    /// butterflies; [`Self::backend`] and [`Self::performance_warning`]
    /// report the tier that was selected.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::UnsupportedTransformLength`] for an unsupported
    /// length.
    pub fn new(length: usize) -> Result<Self, FieldError> {
        let field = PrimeField::<MODULUS>::new();
        let root = field.root_of_unity(length)?;
        let inverse_root = field.inv(root)?;
        let inverse_length = field.element(u64::from(
            field.inv((length as u64 % u64::from(MODULUS)) as u32)?,
        ));
        let (backend, warning) = select_backend::<MODULUS>();
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
    /// All backends have identical transform semantics and ordering.
    #[must_use]
    pub const fn backend(&self) -> NttBackend {
        self.backend
    }

    /// Returns a diagnostic for scalar selection or fallback, if applicable.
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
    /// `values.len()` is exactly [`Self::len`].
    pub fn forward(&self, values: &mut [FieldElement<MODULUS>]) -> Result<(), FieldError> {
        self.check_length(values.len())?;
        match self.backend {
            NttBackend::ScalarShoupLazy => {
                self.forward_shoup_lazy(values);
            }
            NttBackend::ScalarShoup => self.forward_shoup(values),
            NttBackend::ScalarMontgomery => self.forward_montgomery(values),
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
    /// plan's transform kernel, then multiplies by `N^-1` through the field's
    /// bulk scalar multiplication for normalization.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::LengthMismatch`] without mutation unless the slice
    /// length is exactly [`Self::len`].
    pub fn inverse(&self, values: &mut [FieldElement<MODULUS>]) -> Result<(), FieldError> {
        self.check_length(values.len())?;
        match self.backend {
            NttBackend::ScalarShoupLazy => {
                self.inverse_shoup_lazy(values);
            }
            NttBackend::ScalarShoup => self.inverse_shoup(values),
            NttBackend::ScalarMontgomery => self.inverse_montgomery(values),
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
    /// [`PrimeField::mul_elements_assign`], the field's bulk Montgomery kernel.
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

    pub(super) const fn check_length(&self, length: usize) -> Result<(), FieldError> {
        if length == self.length {
            Ok(())
        } else {
            Err(FieldError::LengthMismatch)
        }
    }

    pub(super) fn check_convolution_capacity(
        &self,
        result_length: usize,
    ) -> Result<(), FieldError> {
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
}
