//! Speculative structured-cubic trapdoored matrices built from Toeplitz maps.
//!
//! This construction has no settled security parameters and is not a production
//! cryptographic primitive. Secret diagonals and their cached NTT spectra are
//! not zeroized on drop. Evaluation uses secret-independent addresses and
//! control flow, but the implementation has not received a constant-time audit.

use prime_field_layer::{FieldElement, NttBackend, NttPerformanceWarning, NttPlan, PrimeField};
use rand_core::CryptoRng;

use crate::{DenseMatrix, Permutation, TdmError, error::check_len};

/// A rectangular Toeplitz linear map evaluated through a cached cyclic NTT.
///
/// For `rows = r`, the retained diagonal vector has `r + columns - 1`
/// elements and defines `T[i, j] = diagonals[r - 1 + j - i]`. Both the
/// diagonals and their transformed cyclic embedding are secret and are not
/// zeroized on drop.
pub struct ToeplitzMap<const MODULUS: u32> {
    rows: usize,
    columns: usize,
    diagonals: Vec<FieldElement<MODULUS>>,
    plan: NttPlan<MODULUS>,
    spectrum: Vec<FieldElement<MODULUS>>,
}

impl<const MODULUS: u32> ToeplitzMap<MODULUS> {
    /// Constructs a checked map and caches its transformed cyclic embedding.
    ///
    /// # Errors
    ///
    /// Returns an error if a dimension is zero, dimension arithmetic overflows,
    /// the diagonal count is wrong, or the field does not support the required
    /// power-of-two transform length.
    pub fn new(
        rows: usize,
        columns: usize,
        diagonals: Vec<FieldElement<MODULUS>>,
    ) -> Result<Self, TdmError> {
        if rows == 0 {
            return Err(TdmError::ZeroDimension("Toeplitz rows"));
        }
        if columns == 0 {
            return Err(TdmError::ZeroDimension("Toeplitz columns"));
        }

        let diagonal_count = rows
            .checked_add(columns)
            .and_then(|sum| sum.checked_sub(1))
            .ok_or(TdmError::DimensionOverflow)?;
        check_len("Toeplitz diagonals", diagonal_count, diagonals.len())?;
        let transform_length = diagonal_count
            .checked_next_power_of_two()
            .ok_or(TdmError::DimensionOverflow)?;
        let plan = NttPlan::new(transform_length)?;
        let zero = PrimeField::<MODULUS>::new().element_u32(0);
        let mut spectrum = vec![zero; transform_length];

        for offset in 0..rows {
            spectrum[offset] = diagonals[rows - 1 - offset];
        }
        for offset in 0..columns - 1 {
            spectrum[transform_length - 1 - offset] = diagonals[rows + offset];
        }
        plan.forward(&mut spectrum)?;

        Ok(Self {
            rows,
            columns,
            diagonals,
            plan,
            spectrum,
        })
    }

    /// Returns the output dimension.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the input dimension.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Returns the secret diagonal vector in the constructor's convention.
    #[must_use]
    pub fn diagonals(&self) -> &[FieldElement<MODULUS>] {
        &self.diagonals
    }

    /// Returns the cached transform length.
    #[must_use]
    pub const fn transform_length(&self) -> usize {
        self.plan.len()
    }

    /// Returns the NTT butterfly backend selected during construction.
    #[must_use]
    pub const fn backend(&self) -> NttBackend {
        self.plan.backend()
    }

    /// Returns a scalar-backend or fallback diagnostic, when applicable.
    #[must_use]
    pub const fn performance_warning(&self) -> Option<NttPerformanceWarning> {
        self.plan.performance_warning()
    }

    /// Allocates reusable storage for this map.
    #[must_use]
    pub fn scratch(&self) -> ToeplitzScratch<MODULUS> {
        ToeplitzScratch::new(self.transform_length(), 0)
    }

    /// Applies the map without allocation.
    ///
    /// Addresses and control flow do not depend on the secret diagonals or
    /// input values. This code has not received a constant-time audit.
    ///
    /// # Errors
    ///
    /// Returns before output mutation if an input, output, or transform buffer
    /// has the wrong length.
    pub fn apply(
        &self,
        input: &[FieldElement<MODULUS>],
        output: &mut [FieldElement<MODULUS>],
        scratch: &mut ToeplitzScratch<MODULUS>,
    ) -> Result<(), TdmError> {
        check_len("Toeplitz input", self.columns, input.len())?;
        check_len("Toeplitz output", self.rows, output.len())?;
        if scratch.transform.len() < self.transform_length() {
            return Err(TdmError::LengthMismatch {
                name: "Toeplitz scratch transform",
                expected: self.transform_length(),
                actual: scratch.transform.len(),
            });
        }

        self.apply_with_transform(input, output, &mut scratch.transform)
    }

    fn apply_with_transform(
        &self,
        input: &[FieldElement<MODULUS>],
        output: &mut [FieldElement<MODULUS>],
        transform: &mut [FieldElement<MODULUS>],
    ) -> Result<(), TdmError> {
        let transform = &mut transform[..self.transform_length()];
        let zero = PrimeField::<MODULUS>::new().element_u32(0);
        transform.fill(zero);
        transform[..self.columns].copy_from_slice(input);
        self.plan.forward(transform)?;
        self.plan.pointwise_mul_assign(transform, &self.spectrum)?;
        self.plan.inverse(transform)?;
        output.copy_from_slice(&transform[..self.rows]);
        Ok(())
    }

    fn apply_transpose_with_transform(
        &self,
        input: &[FieldElement<MODULUS>],
        output: &mut [FieldElement<MODULUS>],
        transform: &mut [FieldElement<MODULUS>],
        transpose_spectrum: &[FieldElement<MODULUS>],
    ) -> Result<(), TdmError> {
        check_len("Toeplitz transpose input", self.rows, input.len())?;
        check_len("Toeplitz transpose output", self.columns, output.len())?;
        if transform.len() < self.transform_length() {
            return Err(TdmError::LengthMismatch {
                name: "Toeplitz transpose scratch transform",
                expected: self.transform_length(),
                actual: transform.len(),
            });
        }
        check_len(
            "Toeplitz transpose spectrum",
            self.transform_length(),
            transpose_spectrum.len(),
        )?;

        let transform = &mut transform[..self.transform_length()];
        let zero = PrimeField::<MODULUS>::new().element_u32(0);
        transform.fill(zero);
        transform[..self.rows].copy_from_slice(input);
        self.plan.forward(transform)?;
        self.plan
            .pointwise_mul_assign(transform, transpose_spectrum)?;
        self.plan.inverse(transform)?;
        output.copy_from_slice(&transform[..self.columns]);
        Ok(())
    }

    fn transpose_spectrum(&self) -> Vec<FieldElement<MODULUS>> {
        let length = self.transform_length();
        let zero = PrimeField::<MODULUS>::new().element_u32(0);
        let mut transpose = vec![zero; length];
        let bits = length.trailing_zeros();
        for (target_index, target) in transpose.iter_mut().enumerate() {
            let frequency = target_index.reverse_bits() >> (usize::BITS - bits);
            let source_frequency = if frequency == 0 {
                0
            } else {
                length - frequency
            };
            let source_index = source_frequency.reverse_bits() >> (usize::BITS - bits);
            *target = self.spectrum[source_index];
        }
        transpose
    }
}

/// Reusable allocation for Toeplitz-map and structured-product evaluation.
///
/// Construct it with [`ToeplitzMap::scratch`] or
/// [`ToeplitzFastProduct::scratch`]. Its buffers are overwritten on each call.
pub struct ToeplitzScratch<const MODULUS: u32> {
    transform: Vec<FieldElement<MODULUS>>,
    stage_a: Vec<FieldElement<MODULUS>>,
    stage_b: Vec<FieldElement<MODULUS>>,
}

impl<const MODULUS: u32> ToeplitzScratch<MODULUS> {
    fn new(transform_length: usize, stage_length: usize) -> Self {
        let zero = PrimeField::<MODULUS>::new().element_u32(0);
        Self {
            transform: vec![zero; transform_length],
            stage_a: vec![zero; stage_length],
            stage_b: vec![zero; stage_length],
        }
    }

    /// Returns the number of transform elements retained by this scratch.
    #[must_use]
    pub const fn transform_length(&self) -> usize {
        self.transform.len()
    }

    /// Returns the length of each intermediate product stage.
    #[must_use]
    pub const fn stage_length(&self) -> usize {
        self.stage_a.len()
    }
}

/// The speculative structured-cubic product at expansion factor `c = 2`.
///
/// The product is `S_L Pi_L S Pi_R S_R`, where `S_R` is `2K x K`, `S` is
/// `2K x 2K`, and `S_L` is `K x 2K`. `Pi_R` and `Pi_L` are public gather
/// permutations of length `2K`. This construction has no settled security
/// parameters. Its secret Toeplitz diagonals and cached spectra are not zeroized
/// on drop, and its secret-independent evaluation has not been audited.
pub struct ToeplitzFastProduct<const MODULUS: u32> {
    k: usize,
    s_right: ToeplitzMap<MODULUS>,
    pi_right: Permutation,
    middle: ToeplitzMap<MODULUS>,
    pi_left: Permutation,
    s_left: ToeplitzMap<MODULUS>,
}

struct ToeplitzTransposeSpectra<const MODULUS: u32> {
    right: Vec<FieldElement<MODULUS>>,
    middle: Vec<FieldElement<MODULUS>>,
    left: Vec<FieldElement<MODULUS>>,
}

impl<const MODULUS: u32> ToeplitzFastProduct<MODULUS> {
    /// Constructs a product from explicit maps and public permutations.
    ///
    /// Arguments follow evaluation order: `S_R`, `Pi_R`, `S`, `Pi_L`, `S_L`.
    ///
    /// # Errors
    ///
    /// Returns an error if `K` is zero, `2K` overflows, or any supplied map or
    /// permutation has a shape other than the one required by the construction.
    pub fn new(
        k: usize,
        s_right: ToeplitzMap<MODULUS>,
        pi_right: Permutation,
        middle: ToeplitzMap<MODULUS>,
        pi_left: Permutation,
        s_left: ToeplitzMap<MODULUS>,
    ) -> Result<Self, TdmError> {
        let expanded = expanded_dimension(k)?;
        check_len("S_R rows", expanded, s_right.rows())?;
        check_len("S_R columns", k, s_right.columns())?;
        check_len("Pi_R", expanded, pi_right.len())?;
        check_len("S rows", expanded, middle.rows())?;
        check_len("S columns", expanded, middle.columns())?;
        check_len("Pi_L", expanded, pi_left.len())?;
        check_len("S_L rows", k, s_left.rows())?;
        check_len("S_L columns", expanded, s_left.columns())?;

        Ok(Self {
            k,
            s_right,
            pi_right,
            middle,
            pi_left,
            s_left,
        })
    }

    /// Samples all `10K - 3` secret diagonal elements uniformly and samples two
    /// unbiased public permutations from a caller-owned CSPRNG.
    ///
    /// Rejection sampling gives variable RNG consumption.
    ///
    /// # Errors
    ///
    /// Returns an error for invalid dimensions, unsupported transform lengths,
    /// or a permutation length outside the sampler's range.
    pub fn sample<R: CryptoRng + ?Sized>(k: usize, rng: &mut R) -> Result<Self, TdmError> {
        let expanded = expanded_dimension(k)?;
        let field = PrimeField::<MODULUS>::new();
        let right_count = expanded
            .checked_add(k)
            .and_then(|sum| sum.checked_sub(1))
            .ok_or(TdmError::DimensionOverflow)?;
        let middle_count = expanded
            .checked_add(expanded)
            .and_then(|sum| sum.checked_sub(1))
            .ok_or(TdmError::DimensionOverflow)?;
        let left_count = k
            .checked_add(expanded)
            .and_then(|sum| sum.checked_sub(1))
            .ok_or(TdmError::DimensionOverflow)?;
        let zero = field.element_u32(0);
        let mut right_diagonals = vec![zero; right_count];
        let mut middle_diagonals = vec![zero; middle_count];
        let mut left_diagonals = vec![zero; left_count];
        field.fill_uniform(rng, &mut right_diagonals);
        field.fill_uniform(rng, &mut middle_diagonals);
        field.fill_uniform(rng, &mut left_diagonals);

        let s_right = ToeplitzMap::new(expanded, k, right_diagonals)?;
        let pi_right = Permutation::sample(expanded, rng)?;
        let middle = ToeplitzMap::new(expanded, expanded, middle_diagonals)?;
        let pi_left = Permutation::sample(expanded, rng)?;
        let s_left = ToeplitzMap::new(k, expanded, left_diagonals)?;
        Self::new(k, s_right, pi_right, middle, pi_left, s_left)
    }

    /// Returns `S_R`.
    #[must_use]
    pub const fn s_right(&self) -> &ToeplitzMap<MODULUS> {
        &self.s_right
    }

    /// Returns the public gather permutation `Pi_R`.
    #[must_use]
    pub const fn pi_right(&self) -> &Permutation {
        &self.pi_right
    }

    /// Returns `S`.
    #[must_use]
    pub const fn middle(&self) -> &ToeplitzMap<MODULUS> {
        &self.middle
    }

    /// Returns the public gather permutation `Pi_L`.
    #[must_use]
    pub const fn pi_left(&self) -> &Permutation {
        &self.pi_left
    }

    /// Returns `S_L`.
    #[must_use]
    pub const fn s_left(&self) -> &ToeplitzMap<MODULUS> {
        &self.s_left
    }

    /// Allocates all transform and intermediate storage needed by [`Self::apply`].
    #[must_use]
    pub fn scratch(&self) -> ToeplitzScratch<MODULUS> {
        ToeplitzScratch::new(self.middle.transform_length(), self.middle.rows())
    }

    /// Applies `S_R`, `Pi_R`, `S`, `Pi_L`, and `S_L` without allocation.
    ///
    /// All public lengths are checked before output mutation. Memory addresses
    /// and control flow depend only on public dimensions and permutations, not
    /// on secret diagonals or field values. This has not been audited.
    ///
    /// # Errors
    ///
    /// Returns before output mutation if an input, output, or scratch buffer has
    /// the wrong length.
    pub fn apply(
        &self,
        input: &[FieldElement<MODULUS>],
        output: &mut [FieldElement<MODULUS>],
        scratch: &mut ToeplitzScratch<MODULUS>,
    ) -> Result<(), TdmError> {
        let expanded = self.middle.rows();
        check_len("fast-product input", self.k, input.len())?;
        check_len("fast-product output", self.k, output.len())?;
        check_len(
            "fast-product scratch stage A",
            expanded,
            scratch.stage_a.len(),
        )?;
        check_len(
            "fast-product scratch stage B",
            expanded,
            scratch.stage_b.len(),
        )?;
        if scratch.transform.len() < self.middle.transform_length() {
            return Err(TdmError::LengthMismatch {
                name: "fast-product scratch transform",
                expected: self.middle.transform_length(),
                actual: scratch.transform.len(),
            });
        }

        self.s_right
            .apply_with_transform(input, &mut scratch.stage_a, &mut scratch.transform)?;
        self.pi_right
            .apply(&scratch.stage_a, &mut scratch.stage_b)?;
        self.middle.apply_with_transform(
            &scratch.stage_b,
            &mut scratch.stage_a,
            &mut scratch.transform,
        )?;
        self.pi_left.apply(&scratch.stage_a, &mut scratch.stage_b)?;
        self.s_left
            .apply_with_transform(&scratch.stage_b, output, &mut scratch.transform)
    }

    fn apply_transpose(
        &self,
        input: &[FieldElement<MODULUS>],
        output: &mut [FieldElement<MODULUS>],
        scratch: &mut ToeplitzScratch<MODULUS>,
        spectra: &ToeplitzTransposeSpectra<MODULUS>,
    ) -> Result<(), TdmError> {
        let expanded = self.middle.rows();
        check_len("fast-product transpose input", self.k, input.len())?;
        check_len("fast-product transpose output", self.k, output.len())?;
        check_len(
            "fast-product transpose scratch stage A",
            expanded,
            scratch.stage_a.len(),
        )?;
        check_len(
            "fast-product transpose scratch stage B",
            expanded,
            scratch.stage_b.len(),
        )?;

        self.s_left.apply_transpose_with_transform(
            input,
            &mut scratch.stage_a,
            &mut scratch.transform,
            &spectra.left,
        )?;
        self.pi_left
            .apply_transpose(&scratch.stage_a, &mut scratch.stage_b)?;
        self.middle.apply_transpose_with_transform(
            &scratch.stage_b,
            &mut scratch.stage_a,
            &mut scratch.transform,
            &spectra.middle,
        )?;
        self.pi_right
            .apply_transpose(&scratch.stage_a, &mut scratch.stage_b)?;
        self.s_right.apply_transpose_with_transform(
            &scratch.stage_b,
            output,
            &mut scratch.transform,
            &spectra.right,
        )
    }

    /// Materializes this product as a row-major `K x K` dense matrix.
    ///
    /// # Errors
    ///
    /// Returns an error if matrix-size arithmetic overflows or evaluation fails.
    pub fn materialize(&self) -> Result<DenseMatrix<MODULUS>, TdmError> {
        self.materialize_top_rows(self.k)
    }

    /// Materializes the first `rows` rows of this product.
    ///
    /// This applies the transposed product to `rows` basis vectors, so work
    /// and output storage scale with the requested row count rather than `K`.
    ///
    /// # Errors
    ///
    /// Returns an error if `rows` is zero, exceeds `K`, matrix-size arithmetic
    /// overflows, or transposed evaluation fails.
    pub fn materialize_top_rows(&self, rows: usize) -> Result<DenseMatrix<MODULUS>, TdmError> {
        if rows == 0 {
            return Err(TdmError::ZeroDimension("materialized rows"));
        }
        if rows > self.k {
            return Err(TdmError::LengthMismatch {
                name: "materialized rows",
                expected: self.k,
                actual: rows,
            });
        }
        let field = PrimeField::<MODULUS>::new();
        let zero = field.element_u32(0);
        let one = field.element_u32(1);
        let mut values = Vec::with_capacity(
            rows.checked_mul(self.k)
                .ok_or(TdmError::DimensionOverflow)?,
        );
        let mut input = vec![zero; self.k].into_boxed_slice();
        let mut output = vec![zero; self.k].into_boxed_slice();
        let mut scratch = self.scratch();
        let spectra = ToeplitzTransposeSpectra {
            right: self.s_right.transpose_spectrum(),
            middle: self.middle.transpose_spectrum(),
            left: self.s_left.transpose_spectrum(),
        };

        for row in 0..rows {
            input.fill(zero);
            input[row] = one;
            self.apply_transpose(&input, &mut output, &mut scratch, &spectra)?;
            values.extend_from_slice(&output);
        }
        DenseMatrix::new(rows, self.k, values)
    }
}

fn expanded_dimension(k: usize) -> Result<usize, TdmError> {
    if k == 0 {
        return Err(TdmError::ZeroDimension("K"));
    }
    k.checked_mul(2).ok_or(TdmError::DimensionOverflow)
}
