//! Irreducible-extension Ring-LPN trapdoored matrices.
//!
//! The public map is `H = [I | M_a]`, where the modulus polynomial `f` and
//! multiplier `a` are public. The sparse `2K`-by-`K` matrix `E` is secret, and
//! the materialized matrix is `H E`. No parameter set for this construction is
//! settled; sampling validates the requested parameters against
//! [`crate::assess`] and reports the outcome as warnings while still
//! constructing the instance.
//!
//! Sparse evaluation indexes memory using the secret support of `E`. Its memory
//! access pattern can therefore leak that support. Secret matrices, scratch
//! buffers, and intermediate values are not zeroized on drop.

use prime_field_layer::{ExtensionField, ExtensionFieldScratch, FieldElement, PrimeField};
use rand_core::CryptoRng;

use super::error::check_len;
use super::parameters::{ParameterWarning, assess, automatic_ring_modulus};
use crate::{DenseMatrix, TdmError};

/// Maximum whole-matrix resampling attempts while avoiding empty columns.
///
/// The probability that one Bernoulli matrix has no empty column is
/// `(1 - (1 - weight/(2K))^{2K})^K`. At any sane weight this is
/// indistinguishable from one and the budget never binds. It is sized so
/// that even the least favorable supported pair (`K = 8`, `weight = 1`)
/// still succeeds with probability beyond `2^-40`; larger failures can only
/// happen below the supported weight range.
const MAX_EMPTY_COLUMN_RETRIES: usize = 1024;

/// A matrix in compressed sparse column format.
///
/// Entries in each column retain their supplied order. Duplicate row indices
/// are allowed and contribute independently during multiplication.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SparseMatrix<const MODULUS: u32> {
    rows: usize,
    columns: usize,
    offsets: Vec<usize>,
    row_indices: Vec<usize>,
    values: Vec<FieldElement<MODULUS>>,
}

impl<const MODULUS: u32> SparseMatrix<MODULUS> {
    /// Constructs a checked compressed sparse column matrix.
    ///
    /// `offsets` must have `columns + 1` entries, start at zero, be
    /// nondecreasing, and end at the length of both entry arrays.
    ///
    /// # Errors
    ///
    /// Returns an error for dimension overflow, malformed offsets, unequal
    /// entry-array lengths, or an out-of-range row index.
    pub fn new(
        rows: usize,
        columns: usize,
        offsets: Vec<usize>,
        row_indices: Vec<usize>,
        values: Vec<FieldElement<MODULUS>>,
    ) -> Result<Self, TdmError> {
        let expected_offsets = columns.checked_add(1).ok_or(TdmError::DimensionOverflow)?;
        check_len("sparse offsets", expected_offsets, offsets.len())?;

        if offsets.first() != Some(&0)
            || offsets.windows(2).any(|pair| pair[0] > pair[1])
            || offsets.last() != Some(&row_indices.len())
            || offsets.last() != Some(&values.len())
        {
            return Err(TdmError::InvalidSparseOffsets);
        }

        for (entry, &row) in row_indices.iter().enumerate() {
            if row >= rows {
                return Err(TdmError::SparseRowOutOfBounds { entry, row, rows });
            }
        }

        Ok(Self {
            rows,
            columns,
            offsets,
            row_indices,
            values,
        })
    }

    /// Returns the row count.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the column count.
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Returns the compressed column offsets.
    #[must_use]
    pub fn offsets(&self) -> &[usize] {
        &self.offsets
    }

    /// Returns the row index of each stored entry.
    #[must_use]
    pub fn row_indices(&self) -> &[usize] {
        &self.row_indices
    }

    /// Returns the stored entries in column order.
    #[must_use]
    pub fn values(&self) -> &[FieldElement<MODULUS>] {
        &self.values
    }

    /// Returns the number of stored entries.
    #[must_use]
    pub const fn nnz(&self) -> usize {
        self.values.len()
    }

    fn apply(&self, input: &[FieldElement<MODULUS>], output: &mut [FieldElement<MODULUS>]) {
        output.fill(PrimeField::<MODULUS>::new().element_u32(0));
        for (column, &input_value) in input.iter().enumerate() {
            let start = self.offsets[column];
            let end = self.offsets[column + 1];
            for entry in start..end {
                output[self.row_indices[entry]] += self.values[entry] * input_value;
            }
        }
    }
}

/// Reusable storage for allocation-free Ring-LPN evaluation.
///
/// This storage contains values derived from the secret sparse matrix. It is
/// not zeroized on drop.
pub struct RingLpnScratch<const MODULUS: u32> {
    sparse_product: Vec<FieldElement<MODULUS>>,
    extension: ExtensionFieldScratch<MODULUS>,
    u0: Box<[u32]>,
    u1: Box<[u32]>,
    result: Box<[u32]>,
}

/// The irreducible-extension Ring-LPN map `H = [I | M_a]` with secret `E`.
///
/// The monic irreducible polynomial `f` and multiplier `a` are public. The
/// sparse matrix `E`, with shape `2K`-by-`K`, is secret. No settled parameter
/// set is known for this construction. Evaluation accesses its work buffer at
/// secret-dependent row indices and does not hide the support of `E`. Neither
/// the instance nor its scratch storage is zeroized on drop.
pub struct IrreducibleRingLpn<const MODULUS: u32> {
    k: usize,
    extension: ExtensionField<MODULUS>,
    multiplier: Box<[u32]>,
    sparse_matrix: SparseMatrix<MODULUS>,
}

impl<const MODULUS: u32> IrreducibleRingLpn<MODULUS> {
    /// Constructs an instance and verifies that the public polynomial `f` is
    /// irreducible.
    ///
    /// The public multiplier is canonicalized coefficient by coefficient. The
    /// supplied sparse matrix is retained as the secret `E`.
    ///
    /// # Errors
    ///
    /// Returns an error if `K` is zero, dimensions overflow, `E` is not
    /// `2K`-by-`K`, or extension-field construction fails.
    pub fn new(
        k: usize,
        modulus: &[u32],
        multiplier: &[u32],
        sparse_matrix: SparseMatrix<MODULUS>,
    ) -> Result<Self, TdmError> {
        Self::validate_dimensions(k, &sparse_matrix)?;
        let extension = ExtensionField::<MODULUS>::new(k, modulus)?;
        Ok(Self::from_parts(k, extension, multiplier, sparse_matrix))
    }

    /// Constructs an instance while trusting an external proof that the public
    /// polynomial `f` is irreducible.
    ///
    /// This skips only the irreducibility test. The extension-field constructor
    /// still validates the polynomial's length and monicity.
    ///
    /// # Errors
    ///
    /// Returns an error if `K` is zero, dimensions overflow, `E` is not
    /// `2K`-by-`K`, or extension-ring construction fails.
    pub fn new_unchecked_irreducible(
        k: usize,
        modulus: &[u32],
        multiplier: &[u32],
        sparse_matrix: SparseMatrix<MODULUS>,
    ) -> Result<Self, TdmError> {
        Self::validate_dimensions(k, &sparse_matrix)?;
        let extension = ExtensionField::<MODULUS>::new_unchecked_irreducible(k, modulus)?;
        Ok(Self::from_parts(k, extension, multiplier, sparse_matrix))
    }

    /// Samples an instance with an automatic irreducible modulus.
    ///
    /// The modulus is a binomial `f(X) = X^K - c` whose irreducibility is
    /// proven analytically by the binomial criterion, so construction skips
    /// Rabin's polynomial-time test. Automatic selection supports prime base
    /// fields with two-adicity at least two and power-of-two degrees; use
    /// [`Self::sample_with_modulus`] for anything else.
    ///
    /// Each of the `2K^2` cells of the secret `E` receives one exact
    /// Bernoulli trial with probability `weight / (2K)`; a selected cell is
    /// uniform in `F_q*`. An empty column would make the corresponding
    /// column of `HE` identically zero, so any empty column resamples the
    /// whole matrix. The weight must be positive and at most `K`.
    ///
    /// The returned warnings carry the [`crate::assess`] assessment of
    /// `(K, weight)`; the instance is constructed regardless.
    ///
    /// # Errors
    ///
    /// Returns an error for zero `K`, a zero or oversized weight, an
    /// unsupported field/degree pair, sampling failures, or a failed
    /// extension-ring construction.
    pub fn sample<R: CryptoRng + ?Sized>(
        k: usize,
        weight: usize,
        rng: &mut R,
    ) -> Result<SampledIrreducibleRingLpn<MODULUS>, TdmError> {
        let modulus = automatic_ring_modulus::<MODULUS>(k)?;
        let assessment = assess::<MODULUS>(k, weight);
        let instance = Self::sample_parts(k, weight, &modulus, false, rng)?;
        Ok(SampledIrreducibleRingLpn {
            instance,
            warnings: assessment.warnings,
        })
    }

    /// Samples an instance with a caller-supplied modulus polynomial.
    ///
    /// The supplied `modulus` is canonicalized and verified to be irreducible
    /// with Rabin's deterministic test, which can be expensive at large `K`.
    /// The remaining sampling behavior matches [`Self::sample`], and the
    /// returned warnings carry the same assessment.
    ///
    /// # Errors
    ///
    /// Returns an error for zero `K`, a zero or oversized weight, an invalid
    /// or reducible modulus, sampling failures, or a failed extension-ring
    /// construction.
    pub fn sample_with_modulus<R: CryptoRng + ?Sized>(
        k: usize,
        weight: usize,
        modulus: &[u32],
        rng: &mut R,
    ) -> Result<SampledIrreducibleRingLpn<MODULUS>, TdmError> {
        let assessment = assess::<MODULUS>(k, weight);
        let instance = Self::sample_parts(k, weight, modulus, true, rng)?;
        Ok(SampledIrreducibleRingLpn {
            instance,
            warnings: assessment.warnings,
        })
    }

    fn sample_parts<R: CryptoRng + ?Sized>(
        k: usize,
        weight: usize,
        modulus: &[u32],
        verify_irreducibility: bool,
        rng: &mut R,
    ) -> Result<Self, TdmError> {
        let rows = Self::checked_rows(k)?;
        Self::validate_weight(k, weight)?;
        let extension = if verify_irreducibility {
            ExtensionField::<MODULUS>::new(k, modulus)?
        } else {
            ExtensionField::<MODULUS>::new_unchecked_irreducible(k, modulus)?
        };

        let field = PrimeField::<MODULUS>::new();
        let multiplier: Box<[u32]> = (0..k).map(|_| field.sample_uniform(rng).value()).collect();
        let sparse_matrix = sample_bernoulli_matrix(k, rows, weight, rng)?;
        Ok(Self::from_parts(k, extension, &multiplier, sparse_matrix))
    }

    /// Rejects weights outside the supported regime.
    ///
    /// The Bernoulli rate is `weight / (2K)`, so the weight must be positive,
    /// and weights above `K` would put the noise density above one half,
    /// which is outside the sparse regime this construction models.
    const fn validate_weight(k: usize, weight: usize) -> Result<(), TdmError> {
        if weight == 0 {
            return Err(TdmError::ZeroDimension("column weight"));
        }
        if weight > k {
            return Err(TdmError::WeightExceedsColumns { weight, maximum: k });
        }
        Ok(())
    }

    /// Returns the canonical public multiplier `a`.
    #[must_use]
    pub const fn multiplier(&self) -> &[u32] {
        &self.multiplier
    }

    /// Returns the canonical coefficients of the public modulus polynomial `f`,
    /// including its leading one.
    #[must_use]
    pub fn modulus(&self) -> &[u32] {
        self.extension.modulus_polynomial()
    }

    /// Returns the secret sparse matrix `E`.
    #[must_use]
    pub const fn sparse_matrix(&self) -> &SparseMatrix<MODULUS> {
        &self.sparse_matrix
    }

    /// Returns the number of stored entries in the secret sparse matrix `E`.
    #[must_use]
    pub const fn nnz(&self) -> usize {
        self.sparse_matrix.nnz()
    }

    /// Allocates reusable storage for evaluation.
    #[must_use]
    pub fn scratch(&self) -> RingLpnScratch<MODULUS> {
        let zero = PrimeField::<MODULUS>::new().element_u32(0);
        RingLpnScratch {
            sparse_product: vec![zero; self.sparse_matrix.rows()],
            extension: self.extension.scratch(),
            u0: vec![0; self.k].into_boxed_slice(),
            u1: vec![0; self.k].into_boxed_slice(),
            result: vec![0; self.k].into_boxed_slice(),
        }
    }

    /// Applies `H E` without allocating when `scratch` is reused.
    ///
    /// The output remains unchanged if validation or extension multiplication
    /// fails. Sparse accumulation performs secret-dependent memory accesses.
    ///
    /// # Errors
    ///
    /// Returns an error for an input or output length mismatch, or if the
    /// reusable extension-field multiplication kernel fails.
    pub fn apply(
        &self,
        input: &[FieldElement<MODULUS>],
        output: &mut [FieldElement<MODULUS>],
        scratch: &mut RingLpnScratch<MODULUS>,
    ) -> Result<(), TdmError> {
        check_len("input", self.k, input.len())?;
        check_len("output", self.k, output.len())?;

        self.sparse_matrix.apply(input, &mut scratch.sparse_product);
        for index in 0..self.k {
            scratch.u0[index] = scratch.sparse_product[index].value();
            scratch.u1[index] = scratch.sparse_product[self.k + index].value();
        }
        self.extension.mul(
            &self.multiplier,
            &scratch.u1,
            &mut scratch.result,
            &mut scratch.extension,
        )?;

        let field = PrimeField::<MODULUS>::new();
        for ((result, &u0), output) in scratch.result.iter_mut().zip(&scratch.u0).zip(output) {
            *result = field.add_canonical(*result, u0);
            *output = field.element_u32(*result);
        }
        Ok(())
    }

    /// Materializes `H E` as a row-major dense `K`-by-`K` matrix.
    ///
    /// This allocates a dense matrix and applies the map to each standard basis
    /// vector.
    ///
    /// # Errors
    ///
    /// Returns an error if dense dimensions overflow or evaluation fails.
    pub fn materialize(&self) -> Result<DenseMatrix<MODULUS>, TdmError> {
        let field = PrimeField::<MODULUS>::new();
        let zero = field.element_u32(0);
        let one = field.element_u32(1);
        let mut matrix = DenseMatrix::zero(self.k, self.k)?;
        let mut input = vec![zero; self.k].into_boxed_slice();
        let mut output = vec![zero; self.k].into_boxed_slice();
        let mut scratch = self.scratch();

        for column in 0..self.k {
            input[column] = one;
            self.apply(&input, &mut output, &mut scratch)?;
            matrix.set_column(column, &output);
            input[column] = zero;
        }
        Ok(matrix)
    }

    fn validate_dimensions(
        k: usize,
        sparse_matrix: &SparseMatrix<MODULUS>,
    ) -> Result<(), TdmError> {
        let rows = Self::checked_rows(k)?;
        check_len("sparse matrix rows", rows, sparse_matrix.rows())?;
        check_len("sparse matrix columns", k, sparse_matrix.columns())
    }

    fn checked_rows(k: usize) -> Result<usize, TdmError> {
        if k == 0 {
            return Err(TdmError::ZeroDimension("extension degree"));
        }
        k.checked_mul(2).ok_or(TdmError::DimensionOverflow)
    }

    fn from_parts(
        k: usize,
        extension: ExtensionField<MODULUS>,
        multiplier: &[u32],
        sparse_matrix: SparseMatrix<MODULUS>,
    ) -> Self {
        let field = PrimeField::<MODULUS>::new();
        let multiplier = multiplier
            .iter()
            .map(|value| field.reduce_u32(*value))
            .collect();
        Self {
            k,
            extension,
            multiplier,
            sparse_matrix,
        }
    }
}

/// A sampled instance together with its security-parameter assessment.
///
/// Sampling constructs the instance regardless of the assessment outcome;
/// the warnings describe how the requested `(K, weight)` pair compares with
/// the target security policy. An empty warning slice means every assessment
/// check passed.
pub struct SampledIrreducibleRingLpn<const MODULUS: u32> {
    instance: IrreducibleRingLpn<MODULUS>,
    warnings: Vec<ParameterWarning>,
}

impl<const MODULUS: u32> std::fmt::Debug for SampledIrreducibleRingLpn<MODULUS> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("SampledIrreducibleRingLpn")
            .field("warnings", &self.warnings)
            .finish_non_exhaustive()
    }
}

impl<const MODULUS: u32> SampledIrreducibleRingLpn<MODULUS> {
    /// Returns the sampled instance.
    #[must_use]
    pub const fn instance(&self) -> &IrreducibleRingLpn<MODULUS> {
        &self.instance
    }

    /// Returns the assessment warnings for the sampled parameters.
    #[must_use]
    pub fn warnings(&self) -> &[ParameterWarning] {
        &self.warnings
    }

    /// Consumes the sample and returns the instance alone.
    #[must_use]
    pub fn into_instance(self) -> IrreducibleRingLpn<MODULUS> {
        self.instance
    }
}

/// Samples the secret Bernoulli matrix `E` with no empty column.
///
/// Each cell is nonzero with exact probability `weight / rows`, and selected
/// cells are uniform in `F_q*`. Whole matrices are resampled until no column
/// is empty; see [`MAX_EMPTY_COLUMN_RETRIES`] for the budget.
fn sample_bernoulli_matrix<const MODULUS: u32, R: CryptoRng + ?Sized>(
    k: usize,
    rows: usize,
    weight: usize,
    rng: &mut R,
) -> Result<SparseMatrix<MODULUS>, TdmError> {
    let offsets_capacity = k.checked_add(1).ok_or(TdmError::DimensionOverflow)?;
    let field = PrimeField::<MODULUS>::new();
    for _attempt in 0..MAX_EMPTY_COLUMN_RETRIES {
        let mut offsets = Vec::with_capacity(offsets_capacity);
        let mut row_indices = Vec::new();
        let mut values = Vec::new();
        offsets.push(0);
        for _column in 0..k {
            for row in 0..rows {
                if super::permutation::sample_below(rng, rows)? < weight {
                    row_indices.push(row);
                    values.push(field.sample_uniform_nonzero(rng));
                }
            }
            offsets.push(row_indices.len());
        }
        let matrix = SparseMatrix::new(rows, k, offsets, row_indices, values)?;
        let has_empty_column = matrix.offsets().windows(2).any(|pair| pair[0] == pair[1]);
        if !has_empty_column {
            return Ok(matrix);
        }
    }
    Err(TdmError::SamplingRetryBudgetExhausted {
        retries: MAX_EMPTY_COLUMN_RETRIES,
    })
}
