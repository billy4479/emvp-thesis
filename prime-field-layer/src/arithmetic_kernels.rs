//! Allocation-free arithmetic kernels over Montgomery-form field elements.
//!
//! [`FieldElement`] stores its residue in Montgomery form internally, while
//! [`FieldElement::value`] returns the canonical residue. Accepting only typed
//! elements keeps canonical integers out of these kernels and avoids repeated
//! conversion at their boundaries.
//!
//! The dense loops avoid coefficient-dependent branches where practical and
//! are kept simple so the optimizer can inline and vectorize them. This module
//! has not been formally audited and does not claim constant-time execution.
//! [`sparse_accumulate`] necessarily performs data-dependent memory accesses
//! using the public indices supplied by its entries.

use std::fmt;

use crate::{FieldElement, PrimeField};

/// An error reported by an arithmetic kernel before output mutation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ArithmeticKernelError {
    /// Two corresponding slices have different lengths.
    LengthMismatch { expected: usize, actual: usize },
    /// A batched operation was requested with no lanes.
    ZeroBatchWidth,
    /// The flattened length for a batched operation does not fit in `usize`.
    ShapeOverflow { positions: usize, width: usize },
    /// A sparse entry refers outside the output slice.
    IndexOutOfBounds {
        entry: usize,
        index: usize,
        output_len: usize,
    },
}

impl fmt::Display for ArithmeticKernelError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch { expected, actual } => write!(
                formatter,
                "slice length mismatch: expected {expected}, got {actual}"
            ),
            Self::ZeroBatchWidth => formatter.write_str("batch width must be nonzero"),
            Self::ShapeOverflow { positions, width } => write!(
                formatter,
                "batched shape with {positions} positions and width {width} overflows usize"
            ),
            Self::IndexOutOfBounds {
                entry,
                index,
                output_len,
            } => write!(
                formatter,
                "sparse entry {entry} has index {index}, but output length is {output_len}"
            ),
        }
    }
}

impl std::error::Error for ArithmeticKernelError {}

/// One index and value for [`sparse_accumulate`].
///
/// Repeated indices are allowed and their values are accumulated in entry
/// order. The value remains in Montgomery form throughout accumulation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct IndexedValue<const MODULUS: u32> {
    pub index: usize,
    pub value: FieldElement<MODULUS>,
}

impl<const MODULUS: u32> IndexedValue<MODULUS> {
    #[must_use]
    pub const fn new(index: usize, value: FieldElement<MODULUS>) -> Self {
        Self { index, value }
    }
}

/// Computes the dot product of two Montgomery-form slices.
///
/// The result is a [`FieldElement`] and therefore also remains in Montgomery
/// form internally. Call [`FieldElement::value`] only when canonical output is
/// needed. This takes `O(n)` time and allocates nothing.
///
/// # Errors
///
/// Returns [`ArithmeticKernelError::LengthMismatch`] when the slices have
/// different lengths.
pub fn dot_product<const MODULUS: u32>(
    lhs: &[FieldElement<MODULUS>],
    rhs: &[FieldElement<MODULUS>],
) -> Result<FieldElement<MODULUS>, ArithmeticKernelError> {
    validate_lengths(lhs.len(), rhs.len())?;

    // Four independent accumulators break the serial multiply-then-add
    // dependency chain; `dot_canonical` uses the same lane layout.
    let field = PrimeField::<MODULUS>::new();
    let zero = field.element_u32(0);
    let mut sums = [zero; 4];
    for (lhs, rhs) in lhs.chunks_exact(4).zip(rhs.chunks_exact(4)) {
        sums[0] += lhs[0] * rhs[0];
        sums[1] += lhs[1] * rhs[1];
        sums[2] += lhs[2] * rhs[2];
        sums[3] += lhs[3] * rhs[3];
    }
    let remainder = lhs
        .chunks_exact(4)
        .remainder()
        .iter()
        .zip(rhs.chunks_exact(4).remainder());
    for (&lhs, &rhs) in remainder {
        sums[0] += lhs * rhs;
    }
    Ok(sums.into_iter().fold(zero, |total, sum| total + sum))
}

/// Adds `scalar * input[i]` to each existing `output[i]`.
///
/// Inputs and output stay in Montgomery form. This takes `O(n)` time and
/// allocates nothing.
///
/// # Errors
///
/// Returns [`ArithmeticKernelError::LengthMismatch`] before changing `output`
/// when the slices have different lengths.
pub fn axpy_assign<const MODULUS: u32>(
    output: &mut [FieldElement<MODULUS>],
    scalar: FieldElement<MODULUS>,
    input: &[FieldElement<MODULUS>],
) -> Result<(), ArithmeticKernelError> {
    validate_lengths(output.len(), input.len())?;

    for (output, &input) in output.iter_mut().zip(input) {
        *output += scalar * input;
    }
    Ok(())
}

/// Replaces each value with its weighted inclusive prefix sum.
///
/// After this function returns, `values[i]` is
/// `sum(values[j] * weights[j], j = 0..=i)`, where the values on the right are
/// those supplied on entry. Computation stays in Montgomery form, takes `O(n)`
/// time, and allocates nothing.
///
/// # Errors
///
/// Returns [`ArithmeticKernelError::LengthMismatch`] before changing `values`
/// when the slices have different lengths.
pub fn weighted_inclusive_scan_assign<const MODULUS: u32>(
    values: &mut [FieldElement<MODULUS>],
    weights: &[FieldElement<MODULUS>],
) -> Result<(), ArithmeticKernelError> {
    validate_lengths(values.len(), weights.len())?;

    let mut sum = PrimeField::<MODULUS>::new().element_u32(0);
    for (value, &weight) in values.iter_mut().zip(weights) {
        sum += *value * weight;
        *value = sum;
    }
    Ok(())
}

/// Computes independent weighted inclusive scans in position-major layout.
///
/// `values` is a flattened `[position][lane]` matrix with `width` lanes, and
/// `weights` contains one weight per position. On return, entry `[i][lane]` is
/// `sum(values[j][lane] * weights[j], j = 0..=i)`, using the values supplied on
/// entry. Each transformed row acts as the accumulator for the following row,
/// so the function allocates no memory and requires no extra workspace.
///
/// A positive `width` with empty `values` and `weights` is valid and represents
/// zero positions. Width zero is rejected even when both slices are empty.
///
/// # Errors
///
/// Returns [`ArithmeticKernelError::ZeroBatchWidth`] when `width` is zero,
/// [`ArithmeticKernelError::ShapeOverflow`] when `weights.len() * width` does
/// not fit in `usize`, or [`ArithmeticKernelError::LengthMismatch`] when
/// `values` does not have that flattened length. Validation finishes before
/// changing `values`.
pub fn batched_weighted_inclusive_scan_assign<const MODULUS: u32>(
    values: &mut [FieldElement<MODULUS>],
    weights: &[FieldElement<MODULUS>],
    width: usize,
) -> Result<(), ArithmeticKernelError> {
    validate_batched_shape(values.len(), weights.len(), width)?;
    let Some((&first_weight, remaining_weights)) = weights.split_first() else {
        return Ok(());
    };

    for value in &mut values[..width] {
        *value *= first_weight;
    }

    for (position, &weight) in remaining_weights.iter().enumerate() {
        let row_start = (position + 1) * width;
        let previous_start = row_start - width;
        let (previous_rows, current_and_later) = values.split_at_mut(row_start);
        let previous = &previous_rows[previous_start..row_start];
        let current = &mut current_and_later[..width];
        for (current, &previous) in current.iter_mut().zip(previous) {
            *current = previous + *current * weight;
        }
    }
    Ok(())
}

/// Adds sparse indexed values into an existing output slice.
///
/// Duplicate indices accumulate in entry order. The function takes `O(n)` time
/// in the number of entries and allocates nothing. It validates every index
/// before changing `output`, so an invalid entry cannot leave a partial result.
/// Memory access depends on each entry's index; callers must not use this API
/// when those access locations must remain secret.
///
/// # Errors
///
/// Returns [`ArithmeticKernelError::IndexOutOfBounds`] when an entry's index is
/// not less than `output.len()`.
pub fn sparse_accumulate<const MODULUS: u32>(
    output: &mut [FieldElement<MODULUS>],
    entries: &[IndexedValue<MODULUS>],
) -> Result<(), ArithmeticKernelError> {
    for (entry, indexed) in entries.iter().enumerate() {
        if indexed.index >= output.len() {
            return Err(ArithmeticKernelError::IndexOutOfBounds {
                entry,
                index: indexed.index,
                output_len: output.len(),
            });
        }
    }

    for indexed in entries {
        output[indexed.index] += indexed.value;
    }
    Ok(())
}

#[inline]
const fn validate_lengths(expected: usize, actual: usize) -> Result<(), ArithmeticKernelError> {
    if expected != actual {
        return Err(ArithmeticKernelError::LengthMismatch { expected, actual });
    }
    Ok(())
}

#[inline]
const fn validate_batched_shape(
    actual: usize,
    positions: usize,
    width: usize,
) -> Result<(), ArithmeticKernelError> {
    if width == 0 {
        return Err(ArithmeticKernelError::ZeroBatchWidth);
    }
    let Some(expected) = positions.checked_mul(width) else {
        return Err(ArithmeticKernelError::ShapeOverflow { positions, width });
    };
    validate_lengths(expected, actual)
}
