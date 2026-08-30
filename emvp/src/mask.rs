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
//! not a multiple of `n`, the final block keeps only its top `m mod n`
//! rows and its surplus rows are discarded.
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
use trapdoor_matrices::{
    DenseMatrix, IrreducibleRingLpn, RaaScratch, RaaWeightedProduct, RingLpnScratch, TdmError,
    ToeplitzFastProduct, ToeplitzScratch,
};

/// A rejected mask construction or evaluation.
#[derive(Clone, Debug, Eq, PartialEq)]
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
pub trait TdmMask<const MODULUS: u32> {
    /// Reusable storage for allocation-free evaluation.
    type Scratch;

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
/// its surplus rows are computed and then discarded. All blocks must share
/// the square dimensions `(n, n)`.
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
    /// `total_rows` must lie in `1..=blocks.len() * n` where `n` is the
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
    type Scratch = Vec<M::Scratch>;

    fn dims(&self) -> (usize, usize) {
        (self.total_rows, self.block_rows)
    }

    fn scratch(&self) -> Self::Scratch {
        self.blocks.iter().map(M::scratch).collect()
    }

    /// Computes `output = self * input`.
    ///
    /// Every block multiplies the full input; block `i` writes rows
    /// `[i n, (i + 1) n)` of the output, and the last block writes only its
    /// top `total_rows - (block_count - 1) * n` rows into the tail. All
    /// protocol-level lengths are checked before `output` is mutated; a
    /// failure raised by an individual block evaluation can leave the rows
    /// of earlier blocks in place.
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
        check_len("mask scratch blocks", self.blocks.len(), scratch.len())?;

        let last = self.blocks.len() - 1;
        for (index, block) in self.blocks[..last].iter().enumerate() {
            block.apply(
                input,
                &mut output[index * block_rows..(index + 1) * block_rows],
                &mut scratch[index],
            )?;
        }

        let tail_rows = self.total_rows - last * block_rows;
        if tail_rows == block_rows {
            self.blocks[last].apply(input, &mut output[last * block_rows..], &mut scratch[last])?;
        } else {
            // The last block computes all `n` rows, so the surplus rows need
            // a temporary buffer before the truncation into the output tail.
            let zero = PrimeField::<MODULUS>::new().element_u32(0);
            let mut buffer = vec![zero; block_rows];
            self.blocks[last].apply(input, &mut buffer, &mut scratch[last])?;
            output[last * block_rows..].copy_from_slice(&buffer[..tail_rows]);
        }
        Ok(())
    }

    /// Stacks the dense block matrices vertically and truncates the last
    /// block to its top `total_rows mod n` rows.
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
        let mut values = Vec::with_capacity(length);

        let last = self.blocks.len() - 1;
        for block in &self.blocks[..last] {
            values.extend_from_slice(block.materialize()?.values());
        }
        let tail_rows = self.total_rows - last * block_rows;
        let tail_length = tail_rows
            .checked_mul(block_rows)
            .ok_or(MaskError::DimensionOverflow)?;
        let last_matrix = self.blocks[last].materialize()?;
        values.extend_from_slice(&last_matrix.values()[..tail_length]);

        Ok(DenseMatrix::new(self.total_rows, block_rows, values)?)
    }
}
