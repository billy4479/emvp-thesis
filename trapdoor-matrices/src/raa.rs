use prime_field_layer::{FieldElement, PrimeField, weighted_inclusive_scan_assign};
use rand_core::CryptoRng;

use crate::{DenseMatrix, Permutation, TdmError};

/// An experimental RAA-style weighted product.
///
/// This represents
/// `R = D^T Pi_4 S_3 Pi_3 S_2 Pi_2 S_1 Pi_1 D`, where `D` repeats each of
/// the `k` input entries `c` times. The weighted scans `S_i` and permutations
/// act on `n = k * c` entries. This is an ad hoc structured-cubic
/// pseudorandomness construction, not a production cryptographic primitive.
/// The paper's `c >= 3` is guidance rather than a construction requirement,
/// and no parameter set is settled. Secret values are not zeroized on drop.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RaaWeightedProduct<const MODULUS: u32> {
    k: usize,
    c: usize,
    n: usize,
    first_weights: Vec<FieldElement<MODULUS>>,
    second_weights: Vec<FieldElement<MODULUS>>,
    third_weights: Vec<FieldElement<MODULUS>>,
    first_permutation: Permutation,
    second_permutation: Permutation,
    third_permutation: Permutation,
    fourth_permutation: Permutation,
    /// Precomputed `first_permutation[i] / c` gather sources, so the first
    /// stage needs no runtime division (see [`Self::apply`]).
    first_sources: Vec<usize>,
}

/// Reusable two-buffer workspace for [`RaaWeightedProduct`].
///
/// The buffers are not zeroized on drop.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct RaaScratch<const MODULUS: u32> {
    first: Vec<FieldElement<MODULUS>>,
    second: Vec<FieldElement<MODULUS>>,
}

impl<const MODULUS: u32> RaaWeightedProduct<MODULUS> {
    /// Constructs a checked weighted product from explicit weights and
    /// gather-form permutations.
    ///
    /// Both dimensions must be positive. Unlike the paper's parameter
    /// guidance, this API does not require `c >= 3`.
    ///
    /// # Errors
    ///
    /// Returns an error when a dimension is zero, `k * c` overflows, or any
    /// weight or permutation length differs from `k * c`.
    #[expect(
        clippy::too_many_arguments,
        reason = "the construction has seven explicit factors"
    )]
    pub fn new(
        k: usize,
        c: usize,
        first_weights: Vec<FieldElement<MODULUS>>,
        second_weights: Vec<FieldElement<MODULUS>>,
        third_weights: Vec<FieldElement<MODULUS>>,
        first_permutation: Permutation,
        second_permutation: Permutation,
        third_permutation: Permutation,
        fourth_permutation: Permutation,
    ) -> Result<Self, TdmError> {
        if k == 0 {
            return Err(TdmError::ZeroDimension("k"));
        }
        if c == 0 {
            return Err(TdmError::ZeroDimension("c"));
        }
        let n = k.checked_mul(c).ok_or(TdmError::DimensionOverflow)?;

        super::error::check_len("first weights", n, first_weights.len())?;
        super::error::check_len("second weights", n, second_weights.len())?;
        super::error::check_len("third weights", n, third_weights.len())?;
        super::error::check_len("first permutation", n, first_permutation.len())?;
        super::error::check_len("second permutation", n, second_permutation.len())?;
        super::error::check_len("third permutation", n, third_permutation.len())?;
        super::error::check_len("fourth permutation", n, fourth_permutation.len())?;

        // `D` repeats each of the `k` input entries `c` times, so stage one
        // gathers `input[pi[i] / c]`. The division runs once here instead of
        // once per element per apply.
        let first_sources: Vec<usize> = first_permutation
            .indices()
            .iter()
            .map(|&index| index / c)
            .collect();

        Ok(Self {
            k,
            c,
            n,
            first_weights,
            second_weights,
            third_weights,
            first_permutation,
            second_permutation,
            third_permutation,
            fourth_permutation,
            first_sources,
        })
    }

    /// Samples independent uniform nonzero weights and independent uniform
    /// permutations from a caller-owned CSPRNG.
    ///
    /// Sampling uses rejection and therefore has variable running time and RNG
    /// consumption.
    ///
    /// # Errors
    ///
    /// Returns an error when a dimension is zero, `k * c` overflows, or the
    /// permutation sampler cannot represent the resulting length.
    pub fn sample_nonzero<R: CryptoRng + ?Sized>(
        k: usize,
        c: usize,
        rng: &mut R,
    ) -> Result<Self, TdmError> {
        if k == 0 {
            return Err(TdmError::ZeroDimension("k"));
        }
        if c == 0 {
            return Err(TdmError::ZeroDimension("c"));
        }
        let n = k.checked_mul(c).ok_or(TdmError::DimensionOverflow)?;
        let first_permutation = Permutation::sample(n, rng)?;
        let second_permutation = Permutation::sample(n, rng)?;
        let third_permutation = Permutation::sample(n, rng)?;
        let fourth_permutation = Permutation::sample(n, rng)?;

        let field = PrimeField::<MODULUS>::new();
        let zero = field.element_u32(0);
        let mut first_weights = vec![zero; n];
        let mut second_weights = vec![zero; n];
        let mut third_weights = vec![zero; n];
        field.fill_uniform_nonzero(rng, &mut first_weights);
        field.fill_uniform_nonzero(rng, &mut second_weights);
        field.fill_uniform_nonzero(rng, &mut third_weights);

        Self::new(
            k,
            c,
            first_weights,
            second_weights,
            third_weights,
            first_permutation,
            second_permutation,
            third_permutation,
            fourth_permutation,
        )
    }

    /// Returns the input and output dimension `k`.
    #[must_use]
    pub const fn k(&self) -> usize {
        self.k
    }

    /// Returns the repetition factor `c`.
    #[must_use]
    pub const fn c(&self) -> usize {
        self.c
    }

    /// Returns the internal dimension `n = k * c`.
    #[must_use]
    pub const fn n(&self) -> usize {
        self.n
    }

    /// Returns the first scan's weights.
    #[must_use]
    pub fn first_weights(&self) -> &[FieldElement<MODULUS>] {
        &self.first_weights
    }

    /// Returns the second scan's weights.
    #[must_use]
    pub fn second_weights(&self) -> &[FieldElement<MODULUS>] {
        &self.second_weights
    }

    /// Returns the third scan's weights.
    #[must_use]
    pub fn third_weights(&self) -> &[FieldElement<MODULUS>] {
        &self.third_weights
    }

    /// Returns the first gather permutation.
    #[must_use]
    pub const fn first_permutation(&self) -> &Permutation {
        &self.first_permutation
    }

    /// Returns the second gather permutation.
    #[must_use]
    pub const fn second_permutation(&self) -> &Permutation {
        &self.second_permutation
    }

    /// Returns the third gather permutation.
    #[must_use]
    pub const fn third_permutation(&self) -> &Permutation {
        &self.third_permutation
    }

    /// Returns the fourth gather permutation.
    #[must_use]
    pub const fn fourth_permutation(&self) -> &Permutation {
        &self.fourth_permutation
    }

    /// Allocates a reusable workspace sized for this product.
    #[must_use]
    pub fn scratch(&self) -> RaaScratch<MODULUS> {
        RaaScratch::new(self.n)
    }

    /// Applies the structured matrix without allocating.
    ///
    /// All lengths are checked before either scratch buffer or `output` is
    /// changed. With dimensions and permutations treated as public, memory
    /// addresses and control flow do not depend on field values. This code has
    /// not received a formal constant-time audit.
    ///
    /// # Errors
    ///
    /// Returns before mutation if the input, output, or scratch length is
    /// wrong. Arithmetic-kernel errors cannot occur after these checks unless
    /// the kernel's length contract changes.
    pub fn apply(
        &self,
        input: &[FieldElement<MODULUS>],
        output: &mut [FieldElement<MODULUS>],
        scratch: &mut RaaScratch<MODULUS>,
    ) -> Result<(), TdmError> {
        super::error::check_len("input", self.k, input.len())?;
        super::error::check_len("output", self.k, output.len())?;
        super::error::check_len("first scratch buffer", self.n, scratch.first.len())?;
        super::error::check_len("second scratch buffer", self.n, scratch.second.len())?;

        for (result, &index) in scratch.first.iter_mut().zip(&self.first_sources) {
            *result = input[index];
        }
        weighted_inclusive_scan_assign(&mut scratch.first, &self.first_weights)?;

        gather(
            &scratch.first,
            &mut scratch.second,
            &self.second_permutation,
        );
        weighted_inclusive_scan_assign(&mut scratch.second, &self.second_weights)?;

        gather(&scratch.second, &mut scratch.first, &self.third_permutation);
        weighted_inclusive_scan_assign(&mut scratch.first, &self.third_weights)?;

        gather(
            &scratch.first,
            &mut scratch.second,
            &self.fourth_permutation,
        );

        let zero = PrimeField::<MODULUS>::new().element_u32(0);
        for (block, result) in scratch.second.chunks_exact(self.c).zip(output) {
            *result = block.iter().copied().fold(zero, |sum, value| sum + value);
        }
        Ok(())
    }

    /// Materializes the represented `k` by `k` matrix in row-major form.
    ///
    /// # Errors
    ///
    /// Returns an error if the dense matrix size overflows or an internal
    /// checked application fails.
    pub fn materialize(&self) -> Result<DenseMatrix<MODULUS>, TdmError> {
        let field = PrimeField::<MODULUS>::new();
        let zero = field.element_u32(0);
        let one = field.element_u32(1);
        let mut matrix = DenseMatrix::zero(self.k, self.k)?;
        let mut input = vec![zero; self.k];
        let mut output = vec![zero; self.k];
        let mut scratch = self.scratch();

        for column in 0..self.k {
            input[column] = one;
            self.apply(&input, &mut output, &mut scratch)?;
            matrix.set_column(column, &output);
            input[column] = zero;
        }
        Ok(matrix)
    }
}

impl<const MODULUS: u32> RaaScratch<MODULUS> {
    fn new(length: usize) -> Self {
        let zero = PrimeField::<MODULUS>::new().element_u32(0);
        Self {
            first: vec![zero; length],
            second: vec![zero; length],
        }
    }

    /// Returns the length of each reusable buffer.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.first.len()
    }

    /// Returns whether both reusable buffers are empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.first.is_empty() && self.second.is_empty()
    }
}

fn gather<const MODULUS: u32>(
    input: &[FieldElement<MODULUS>],
    output: &mut [FieldElement<MODULUS>],
    permutation: &Permutation,
) {
    for (result, &index) in output.iter_mut().zip(permutation.indices()) {
        *result = input[index];
    }
}
