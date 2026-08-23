use prime_field_layer::{FieldElement, PrimeField};

use crate::TdmError;

/// A row-major dense matrix used for materialized TDM output and test oracles.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DenseMatrix<const MODULUS: u32> {
    rows: usize,
    columns: usize,
    values: Vec<FieldElement<MODULUS>>,
}

impl<const MODULUS: u32> DenseMatrix<MODULUS> {
    /// Constructs a checked row-major matrix.
    ///
    /// # Errors
    ///
    /// Returns a length or dimension-overflow error for an invalid buffer.
    pub fn new(
        rows: usize,
        columns: usize,
        values: Vec<FieldElement<MODULUS>>,
    ) -> Result<Self, TdmError> {
        if rows == 0 {
            return Err(TdmError::ZeroDimension("matrix rows"));
        }
        if columns == 0 {
            return Err(TdmError::ZeroDimension("matrix columns"));
        }
        let expected = rows
            .checked_mul(columns)
            .ok_or(TdmError::DimensionOverflow)?;
        super::error::check_len("matrix values", expected, values.len())?;
        Ok(Self {
            rows,
            columns,
            values,
        })
    }

    pub(crate) fn zero(rows: usize, columns: usize) -> Result<Self, TdmError> {
        let length = rows
            .checked_mul(columns)
            .ok_or(TdmError::DimensionOverflow)?;
        let zero = PrimeField::<MODULUS>::new().element_u32(0);
        Ok(Self {
            rows,
            columns,
            values: vec![zero; length],
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

    /// Returns the row-major values.
    #[must_use]
    pub fn values(&self) -> &[FieldElement<MODULUS>] {
        &self.values
    }

    /// Multiplies the dense matrix by a vector.
    ///
    /// # Errors
    ///
    /// Returns before mutation if an input or output length is wrong.
    pub fn apply(
        &self,
        input: &[FieldElement<MODULUS>],
        output: &mut [FieldElement<MODULUS>],
    ) -> Result<(), TdmError> {
        super::error::check_len("input", self.columns, input.len())?;
        super::error::check_len("output", self.rows, output.len())?;
        let zero = PrimeField::<MODULUS>::new().element_u32(0);
        for (row, result) in self
            .values
            .chunks_exact(self.columns)
            .zip(output.iter_mut())
        {
            let mut sum = zero;
            for (&coefficient, &value) in row.iter().zip(input) {
                sum += coefficient * value;
            }
            *result = sum;
        }
        Ok(())
    }

    pub(crate) fn set_column(&mut self, column: usize, values: &[FieldElement<MODULUS>]) {
        for (row, &value) in values.iter().enumerate() {
            self.values[row * self.columns + column] = value;
        }
    }
}
