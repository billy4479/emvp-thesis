use super::{
    FieldElement, FieldError, LinearConvolutionWorkspace, NegacyclicPlan, NttBackend,
    NttPerformanceWarning, NttPlan, PretransformedLinearOperand, PrimeField, powers,
};

const SCHOOLBOOK_PRODUCT_CUTOFF: usize = 5_184;

impl<const MODULUS: u32> NttPlan<MODULUS> {
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

    pub(super) fn convolve_elements(
        &self,
        lhs: &mut [FieldElement<MODULUS>],
        rhs: &mut [FieldElement<MODULUS>],
    ) -> Result<(), FieldError> {
        self.forward(lhs)?;
        self.forward(rhs)?;
        self.pointwise_mul_assign(lhs, rhs)?;
        self.inverse(lhs)
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

pub(super) fn convolution_result_length(
    lhs_length: usize,
    rhs_length: usize,
) -> Result<usize, FieldError> {
    lhs_length
        .checked_add(rhs_length)
        .and_then(|length| length.checked_sub(1))
        .ok_or(FieldError::ConvolutionLengthOverflow)
}
