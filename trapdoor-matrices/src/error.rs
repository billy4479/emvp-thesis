use std::fmt;

use prime_field_layer::{ArithmeticKernelError, ExtensionFieldError, FieldError};

/// An invalid TDM instance, input, or arithmetic operation.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum TdmError {
    /// A slice did not have the required length.
    LengthMismatch {
        name: &'static str,
        expected: usize,
        actual: usize,
    },
    /// A required dimension was zero.
    ZeroDimension(&'static str),
    /// Dimension arithmetic overflowed `usize`.
    DimensionOverflow,
    /// A permutation contained a duplicate or out-of-range index.
    InvalidPermutation { position: usize, index: usize },
    /// A bounded sampler cannot represent the requested range.
    SamplingRangeTooLarge { upper_bound: usize },
    /// A Bernoulli numerator exceeded its nonzero denominator.
    InvalidProbability { numerator: u32, denominator: u32 },
    /// Sparse column offsets were malformed.
    InvalidSparseOffsets,
    /// A sparse row index was outside the matrix.
    SparseRowOutOfBounds {
        entry: usize,
        row: usize,
        rows: usize,
    },
    /// Prime-field or NTT arithmetic failed.
    Field(FieldError),
    /// Extension-field construction or arithmetic failed.
    ExtensionField(ExtensionFieldError),
    /// A reusable arithmetic kernel rejected its inputs.
    Arithmetic(ArithmeticKernelError),
}

impl fmt::Display for TdmError {
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
            Self::ZeroDimension(name) => write!(formatter, "{name} must be nonzero"),
            Self::DimensionOverflow => formatter.write_str("dimension arithmetic overflowed"),
            Self::InvalidPermutation { position, index } => write!(
                formatter,
                "permutation entry {position} contains duplicate or out-of-range index {index}"
            ),
            Self::SamplingRangeTooLarge { upper_bound } => write!(
                formatter,
                "cannot sample uniformly below {upper_bound} from 32-bit words"
            ),
            Self::InvalidProbability {
                numerator,
                denominator,
            } => write!(
                formatter,
                "invalid Bernoulli probability {numerator}/{denominator}"
            ),
            Self::InvalidSparseOffsets => {
                formatter.write_str("sparse column offsets are malformed")
            }
            Self::SparseRowOutOfBounds { entry, row, rows } => write!(
                formatter,
                "sparse entry {entry} uses row {row}, but the matrix has {rows} rows"
            ),
            Self::Field(error) => error.fmt(formatter),
            Self::ExtensionField(error) => error.fmt(formatter),
            Self::Arithmetic(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for TdmError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Field(error) => Some(error),
            Self::ExtensionField(error) => Some(error),
            Self::Arithmetic(error) => Some(error),
            _ => None,
        }
    }
}

impl From<FieldError> for TdmError {
    fn from(error: FieldError) -> Self {
        Self::Field(error)
    }
}

impl From<ExtensionFieldError> for TdmError {
    fn from(error: ExtensionFieldError) -> Self {
        Self::ExtensionField(error)
    }
}

impl From<ArithmeticKernelError> for TdmError {
    fn from(error: ArithmeticKernelError) -> Self {
        Self::Arithmetic(error)
    }
}

pub const fn check_len(name: &'static str, expected: usize, actual: usize) -> Result<(), TdmError> {
    if expected == actual {
        Ok(())
    } else {
        Err(TdmError::LengthMismatch {
            name,
            expected,
            actual,
        })
    }
}
