//! Rectangular masking of encoded matrices by trapdoored matrices.
//!
//! The EMVP protocol (IACR ePrint 2025/858, Section 5) hides an encoded
//! matrix behind a pseudorandom mask `R` in `F_p^{m x n}`: the client stores
//! the trapdoor of `R` so it can evaluate `R v` fast during the online
//! phase, while the offline phase materializes `R` densely. The
//! constructions in the [`trapdoor_matrices`] crate are all square `K x K`,
//! so a rectangular `m x n` mask is assembled from independent square
//! blocks stacked along the rows: block `i` multiplies the full `n`-element
//! input and supplies rows `[i n, (i + 1) n)` of the product. When `m` is
//! not a multiple of `n`, the final block materializes only its top `m mod n`
//! rows when the block implementation supports efficient row prefixes.
//!
//! [`TdmMask`] abstracts one square trapdoored matrix, and
//! [`RowStackMask`] turns any stack of equally sized blocks into a single
//! rectangular mask generic over the block construction.
//!
//! This is experimental cryptography: the underlying constructions have no
//! settled security parameters, secret state is not zeroized on drop, and
//! the implementations have not received a constant-time audit.
//!
//! [`trapdoor_matrices`]: trapdoor_matrices

use std::fmt;

use prime_field_layer::{FieldElement, PrimeField};
use rayon::prelude::*;
use trapdoor_matrices::{
    DenseMatrix, IrreducibleRingLpn, RaaScratch, RaaWeightedProduct, RingLpnScratch, TdmError,
    ToeplitzFastProduct, ToeplitzScratch,
};

/// Minimum estimated field multiplications before mask evaluation or
/// materialization switches to rayon; smaller workloads stay serial. The
/// value mirrors the crossover calibrated for the answer phase in
/// [`crate::protocol`].
const MIN_PARALLEL_MULTIPLICATIONS: usize = 32 * 1024;

/// A rejected mask construction or evaluation.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum MaskError {
    /// A slice did not have the required length.
    LengthMismatch {
        /// The rejected slice.
        name: &'static str,
        /// The required length.
        expected: usize,
        /// The observed length.
        actual: usize,
    },
    /// Dimension arithmetic overflowed `usize`.
    DimensionOverflow,
    /// Trapdoored-matrix construction or arithmetic failed.
    Tdm(TdmError),
}

impl fmt::Display for MaskError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "{name} length mismatch: expected {expected}, got {actual}"
            ),
            Self::DimensionOverflow => formatter.write_str("dimension arithmetic overflowed"),
            Self::Tdm(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for MaskError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Tdm(error) => Some(error),
            _ => None,
        }
    }
}

impl From<TdmError> for MaskError {
    fn from(error: TdmError) -> Self {
        Self::Tdm(error)
    }
}

const fn check_len(name: &'static str, expected: usize, actual: usize) -> Result<(), MaskError> {
    if expected == actual {
        Ok(())
    } else {
        Err(MaskError::LengthMismatch {
            name,
            expected,
            actual,
        })
    }
}

/// One square trapdoored matrix usable as an EMVP mask block.
///
/// The trait is local to this crate, so it can be implemented for the
/// foreign construction types of the [`trapdoor_matrices`] crate as well as
/// for user-supplied blocks. Every block is `rows x columns`; the shipped
/// constructions are square `K x K`.
///
/// # Thread-safety contract
///
/// Implementations must be usable from several threads at once (`Send` plus
/// `Sync`, with a `Send` scratch): large [`RowStackMask`] applications and
/// materializations run across rayon workers, which hold `&self` while
/// moving disjoint output slices and [`Self::Scratch`] buffers between
/// threads. Every shipped construction is a plain-data structure, so the
/// bounds cost nothing for it; user-supplied blocks must uphold the same
/// contract to remain usable as stack blocks.
///
/// # Cryptographic contract
///
/// Protocol implementations must use independently sampled blocks whose
/// materialized matrices are pseudorandom under the intended TDM assumption.
/// [`Self::apply`], [`Self::materialize`], and
/// [`Self::materialize_top_rows`] must all represent the same linear map.
/// The type system cannot verify either requirement for user implementations.
pub trait TdmMask<const MODULUS: u32>: Send + Sync {
    /// Reusable storage for allocation-free evaluation.
    ///
    /// It must be `Send` because parallel evaluation hands one scratch value
    /// to whichever worker owns the matching block.
    type Scratch: Send;

    /// Returns the matrix dimensions `(rows, columns)`.
    #[must_use]
    fn dims(&self) -> (usize, usize);

    /// Allocates all storage needed by [`Self::apply`].
    #[must_use]
    fn scratch(&self) -> Self::Scratch;

    /// Computes `output = self * input`.
    ///
    /// `input` must hold `columns` elements and `output` `rows` elements,
    /// as reported by [`Self::dims`]. All lengths are checked before
    /// `output` is mutated.
    ///
    /// # Errors
    ///
    /// Returns an error if a length mismatches or the trapdoored
    /// evaluation fails.
    fn apply(
        &self,
        input: &[FieldElement<MODULUS>],
        output: &mut [FieldElement<MODULUS>],
        scratch: &mut Self::Scratch,
    ) -> Result<(), MaskError>;

    /// Materializes the matrix in row-major dense form for the offline
    /// phase.
    ///
    /// # Errors
    ///
    /// Returns an error if dense dimensions overflow or evaluation fails.
    fn materialize(&self) -> Result<DenseMatrix<MODULUS>, MaskError>;

    /// Materializes the first `rows` rows in row-major form.
    ///
    /// Implementations may override this to make work proportional to the
    /// requested rows. The default avoids a full dense allocation and uses
    /// structured evaluation once per column. Columns are independent, so
    /// wide masks evaluate them across rayon workers, each owning its own
    /// scratch, input, and output buffers: results land in a column-major
    /// staging buffer of disjoint per-column slices, and one linear pass
    /// transposes them into the row-major output. The result is identical
    /// to the serial column loop.
    ///
    /// # Errors
    ///
    /// Returns an error if `rows` is zero, exceeds the reported row count,
    /// dimensions overflow, or evaluation fails.
    fn materialize_top_rows(&self, rows: usize) -> Result<DenseMatrix<MODULUS>, MaskError> {
        let (full_rows, columns) = self.dims();
        if rows == 0 {
            return Err(MaskError::Tdm(TdmError::ZeroDimension("materialized rows")));
        }
        if rows > full_rows {
            return Err(MaskError::LengthMismatch {
                name: "materialized rows",
                expected: full_rows,
                actual: rows,
            });
        }
        if rows == full_rows {
            return self.materialize();
        }

        let field = PrimeField::<MODULUS>::new();
        let zero = field.element_u32(0);
        let one = field.element_u32(1);
        let length = rows
            .checked_mul(columns)
            .ok_or(MaskError::DimensionOverflow)?;
        let mut staged = vec![zero; length];
        // One structured evaluation applies the full mask: a conservative
        // estimate of `columns * full_rows * columns` multiplications.
        let threads = rayon::current_num_threads();
        let work = columns.saturating_mul(full_rows).saturating_mul(columns);
        if threads > 1
            && columns >= threads.saturating_mul(2)
            && work >= MIN_PARALLEL_MULTIPLICATIONS
        {
            staged
                .par_chunks_mut(rows)
                .enumerate()
                .map_init(
                    || (self.scratch(), vec![zero; columns], vec![zero; full_rows]),
                    |(scratch, input, output), (column, staged_column)| {
                        input.fill(zero);
                        input[column] = one;
                        self.apply(input, output, scratch)?;
                        staged_column.copy_from_slice(&output[..rows]);
                        Ok(())
                    },
                )
                .try_for_each(|result: Result<(), MaskError>| result)?;
        } else {
            let mut input = vec![zero; columns];
            let mut output = vec![zero; full_rows];
            let mut scratch = self.scratch();
            for (column, staged_column) in staged.chunks_mut(rows).enumerate() {
                input.fill(zero);
                input[column] = one;
                self.apply(&input, &mut output, &mut scratch)?;
                staged_column.copy_from_slice(&output[..rows]);
            }
        }

        let mut values = Vec::with_capacity(length);
        for row in 0..rows {
            for column in 0..columns {
                values.push(staged[column * rows + row]);
            }
        }
        Ok(DenseMatrix::new(rows, columns, values)?)
    }
}

impl<const MODULUS: u32> TdmMask<MODULUS> for IrreducibleRingLpn<MODULUS> {
    type Scratch = RingLpnScratch<MODULUS>;

    fn dims(&self) -> (usize, usize) {
        // The public map `H E` is `K x K` and the secret `E` is `2K x K`, so
        // the sparse column count is `K`.
        let k = self.sparse_matrix().columns();
        (k, k)
    }

    fn scratch(&self) -> Self::Scratch {
        self.scratch()
    }

    fn apply(
        &self,
        input: &[FieldElement<MODULUS>],
        output: &mut [FieldElement<MODULUS>],
        scratch: &mut Self::Scratch,
    ) -> Result<(), MaskError> {
        Ok(self.apply(input, output, scratch)?)
    }

    fn materialize(&self) -> Result<DenseMatrix<MODULUS>, MaskError> {
        Ok(self.materialize()?)
    }
}

impl<const MODULUS: u32> TdmMask<MODULUS> for ToeplitzFastProduct<MODULUS> {
    type Scratch = ToeplitzScratch<MODULUS>;

    fn dims(&self) -> (usize, usize) {
        // `S_R` is `2K x K`, so its column count is `K`.
        let k = self.s_right().columns();
        (k, k)
    }

    fn scratch(&self) -> Self::Scratch {
        self.scratch()
    }

    fn apply(
        &self,
        input: &[FieldElement<MODULUS>],
        output: &mut [FieldElement<MODULUS>],
        scratch: &mut Self::Scratch,
    ) -> Result<(), MaskError> {
        Ok(self.apply(input, output, scratch)?)
    }

    fn materialize(&self) -> Result<DenseMatrix<MODULUS>, MaskError> {
        Ok(self.materialize()?)
    }

    fn materialize_top_rows(&self, rows: usize) -> Result<DenseMatrix<MODULUS>, MaskError> {
        Ok(Self::materialize_top_rows(self, rows)?)
    }
}

impl<const MODULUS: u32> TdmMask<MODULUS> for RaaWeightedProduct<MODULUS> {
    type Scratch = RaaScratch<MODULUS>;

    fn dims(&self) -> (usize, usize) {
        (self.k(), self.k())
    }

    fn scratch(&self) -> Self::Scratch {
        self.scratch()
    }

    fn apply(
        &self,
        input: &[FieldElement<MODULUS>],
        output: &mut [FieldElement<MODULUS>],
        scratch: &mut Self::Scratch,
    ) -> Result<(), MaskError> {
        Ok(self.apply(input, output, scratch)?)
    }

    fn materialize(&self) -> Result<DenseMatrix<MODULUS>, MaskError> {
        Ok(self.materialize()?)
    }
}

/// A rectangular `total_rows x n` mask built from independent square blocks.
///
/// The blocks are stacked along the rows: block `i` multiplies the full
/// `n`-element input and supplies rows `[i n, (i + 1) n)` of the product, as
/// in Section 5 of IACR ePrint 2025/858. If `total_rows` is not a multiple
/// of `n`, the last block contributes only its top `total_rows mod n` rows;
/// only its top rows are materialized. All blocks must share the square
/// dimensions `(n, n)`.
///
/// The secret block state is not zeroized on drop.
pub struct RowStackMask<M: TdmMask<MODULUS>, const MODULUS: u32> {
    blocks: Vec<M>,
    total_rows: usize,
    block_rows: usize,
}

impl<M: TdmMask<MODULUS>, const MODULUS: u32> RowStackMask<M, MODULUS> {
    /// Constructs a mask from square blocks stacked along the rows.
    ///
    /// The blocks must be nonempty, square, and equally sized, and
    /// the block count must equal `total_rows.div_ceil(n)` where `n` is the
    /// common block dimension.
    ///
    /// # Errors
    ///
    /// Returns a length error for an empty block list, a nonsquare block,
    /// mismatched block dimensions, or a `total_rows` outside the valid
    /// range, and an overflow error if the row-count arithmetic overflows.
    pub fn new(blocks: Vec<M>, total_rows: usize) -> Result<Self, MaskError> {
        let Some(first) = blocks.first() else {
            return Err(MaskError::LengthMismatch {
                name: "mask blocks",
                expected: 1,
                actual: blocks.len(),
            });
        };
        let (first_rows, first_columns) = first.dims();
        if first_rows != first_columns {
            return Err(MaskError::LengthMismatch {
                name: "first mask block rows",
                expected: first_columns,
                actual: first_rows,
            });
        }
        if first_rows == 0 {
            return Err(MaskError::LengthMismatch {
                name: "mask block rows",
                expected: 1,
                actual: 0,
            });
        }
        let block_rows = first_rows;
        for block in blocks.iter().skip(1) {
            let (rows, columns) = block.dims();
            if rows != block_rows {
                return Err(MaskError::LengthMismatch {
                    name: "mask block rows",
                    expected: block_rows,
                    actual: rows,
                });
            }
            if columns != block_rows {
                return Err(MaskError::LengthMismatch {
                    name: "mask block columns",
                    expected: block_rows,
                    actual: columns,
                });
            }
        }
        let full_rows = blocks
            .len()
            .checked_mul(block_rows)
            .ok_or(MaskError::DimensionOverflow)?;
        if total_rows == 0 {
            return Err(MaskError::LengthMismatch {
                name: "total mask rows",
                expected: 1,
                actual: 0,
            });
        }
        if total_rows > full_rows {
            return Err(MaskError::LengthMismatch {
                name: "total mask rows",
                expected: full_rows,
                actual: total_rows,
            });
        }
        let expected_blocks = total_rows.div_ceil(block_rows);
        if blocks.len() != expected_blocks {
            return Err(MaskError::LengthMismatch {
                name: "mask blocks for total rows",
                expected: expected_blocks,
                actual: blocks.len(),
            });
        }

        Ok(Self {
            blocks,
            total_rows,
            block_rows,
        })
    }

    /// Returns the number of stacked square blocks.
    #[must_use]
    pub const fn block_count(&self) -> usize {
        self.blocks.len()
    }

    /// Returns the stacked blocks, which expose the trapdoors.
    #[must_use]
    pub fn blocks(&self) -> &[M] {
        &self.blocks
    }
}

impl<M: TdmMask<MODULUS>, const MODULUS: u32> TdmMask<MODULUS> for RowStackMask<M, MODULUS> {
    type Scratch = (Vec<M::Scratch>, Vec<FieldElement<MODULUS>>);

    fn dims(&self) -> (usize, usize) {
        (self.total_rows, self.block_rows)
    }

    fn scratch(&self) -> Self::Scratch {
        let block_scratch = self.blocks.iter().map(M::scratch).collect();
        let tail = if self.total_rows.is_multiple_of(self.block_rows) {
            Vec::new()
        } else {
            vec![PrimeField::<MODULUS>::new().element_u32(0); self.block_rows]
        };
        (block_scratch, tail)
    }

    /// Computes `output = self * input`.
    ///
    /// Every block multiplies the full input; block `i` writes rows
    /// `[i n, (i + 1) n)` of the output, and the last block writes only its
    /// top `total_rows - (block_count - 1) * n` rows into the tail. The
    /// full blocks write disjoint output slices, so large stacks evaluate
    /// them across rayon workers while the tail block keeps its
    /// scratch-mediated truncation serial; the output is identical to the
    /// serial block loop. All protocol-level lengths are checked before
    /// `output` is mutated; a failure raised by an individual block
    /// evaluation can leave the rows of earlier blocks in place.
    ///
    /// # Errors
    ///
    /// Returns an error before output mutation if the input, output, or
    /// scratch length mismatches, and propagates any block evaluation
    /// failure.
    fn apply(
        &self,
        input: &[FieldElement<MODULUS>],
        output: &mut [FieldElement<MODULUS>],
        scratch: &mut Self::Scratch,
    ) -> Result<(), MaskError> {
        let block_rows = self.block_rows;
        check_len("mask input", block_rows, input.len())?;
        check_len("mask output", self.total_rows, output.len())?;
        check_len("mask scratch blocks", self.blocks.len(), scratch.0.len())?;
        let expected_tail = if self.total_rows.is_multiple_of(block_rows) {
            0
        } else {
            block_rows
        };
        check_len("mask scratch tail", expected_tail, scratch.1.len())?;

        let last = self.blocks.len() - 1;
        let tail_rows = self.total_rows - last * block_rows;
        // The tail block goes through scratch only when it must truncate;
        // otherwise it is an ordinary full block and joins the others.
        let full_blocks = if tail_rows == block_rows {
            self.blocks.len()
        } else {
            last
        };

        // Every block multiplies the full input: a conservative estimate of
        // `block_count * n * n` multiplications.
        let threads = rayon::current_num_threads();
        let work = self
            .blocks
            .len()
            .saturating_mul(block_rows)
            .saturating_mul(block_rows);
        if threads > 1
            && full_blocks >= threads.saturating_mul(2)
            && work >= MIN_PARALLEL_MULTIPLICATIONS
        {
            let (full_output, _tail_output) = output.split_at_mut(full_blocks * block_rows);
            self.blocks[..full_blocks]
                .par_iter()
                .zip(scratch.0[..full_blocks].par_iter_mut())
                .zip(full_output.par_chunks_mut(block_rows))
                .try_for_each(|((block, block_scratch), block_output)| {
                    block.apply(input, block_output, block_scratch)
                })?;
        } else {
            for (index, block) in self.blocks[..full_blocks].iter().enumerate() {
                block.apply(
                    input,
                    &mut output[index * block_rows..(index + 1) * block_rows],
                    &mut scratch.0[index],
                )?;
            }
        }

        if full_blocks == last {
            self.blocks[last].apply(input, &mut scratch.1, &mut scratch.0[last])?;
            output[last * block_rows..].copy_from_slice(&scratch.1[..tail_rows]);
        }
        Ok(())
    }

    /// Stacks the dense block matrices vertically and truncates the last
    /// block to its top `total_rows mod n` rows.
    ///
    /// The blocks materialize independently into disjoint output slabs, so
    /// large stacks run them across rayon workers; the indexed parallel
    /// collect preserves block order and the result is identical to the
    /// serial loop. The truncated tail block stays serial.
    ///
    /// # Errors
    ///
    /// Returns an error if dense dimensions overflow or a block
    /// materialization fails.
    fn materialize(&self) -> Result<DenseMatrix<MODULUS>, MaskError> {
        let block_rows = self.block_rows;
        let length = self
            .total_rows
            .checked_mul(block_rows)
            .ok_or(MaskError::DimensionOverflow)?;
        let last = self.blocks.len() - 1;
        if last == 0 {
            let matrix = self.blocks[0].materialize_top_rows(self.total_rows)?;
            check_len("materialized tail rows", self.total_rows, matrix.rows())?;
            check_len("materialized tail columns", block_rows, matrix.columns())?;
            return Ok(matrix);
        }

        // One block materialization applies the block to every basis vector:
        // a conservative estimate of `n * n * n` multiplications per block.
        let threads = rayon::current_num_threads();
        let work = last
            .saturating_mul(block_rows)
            .saturating_mul(block_rows)
            .saturating_mul(block_rows);
        let fulls: Vec<DenseMatrix<MODULUS>> = if threads > 1
            && last >= threads.saturating_mul(2)
            && work >= MIN_PARALLEL_MULTIPLICATIONS
        {
            self.blocks[..last]
                .par_iter()
                .map(M::materialize)
                .collect::<Result<Vec<_>, MaskError>>()?
        } else {
            self.blocks[..last]
                .iter()
                .map(M::materialize)
                .collect::<Result<Vec<_>, MaskError>>()?
        };
        let mut values = Vec::with_capacity(length);
        for matrix in &fulls {
            check_len("materialized block rows", block_rows, matrix.rows())?;
            check_len("materialized block columns", block_rows, matrix.columns())?;
            values.extend_from_slice(matrix.values());
        }
        let tail_rows = self.total_rows - last * block_rows;
        let tail_length = tail_rows
            .checked_mul(block_rows)
            .ok_or(MaskError::DimensionOverflow)?;
        let last_matrix = self.blocks[last].materialize_top_rows(tail_rows)?;
        check_len("materialized tail rows", tail_rows, last_matrix.rows())?;
        check_len(
            "materialized tail columns",
            block_rows,
            last_matrix.columns(),
        )?;
        check_len(
            "materialized tail values",
            tail_length,
            last_matrix.values().len(),
        )?;
        values.extend_from_slice(last_matrix.values());

        Ok(DenseMatrix::new(self.total_rows, block_rows, values)?)
    }
}
