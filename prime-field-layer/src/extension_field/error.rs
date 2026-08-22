use std::fmt;

use crate::FieldError;

/// Errors from fixed polynomial reduction or extension-field construction.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ExtensionFieldError {
    /// Degree zero does not define a nontrivial polynomial quotient.
    ZeroDegree,
    /// A degree-`K` modulus must contain exactly `K + 1` coefficients.
    ModulusLength { expected: usize, actual: usize },
    /// The degree-`K` coefficient is not one in the base field.
    ModulusNotMonic,
    /// The checked constructor found a nontrivial factor.
    ReducibleModulus,
    /// Reduction input exceeded the maximum `2K - 1` coefficients.
    ProductTooLong { maximum: usize, actual: usize },
    /// The base field could not construct an NTT required by this degree.
    BaseField(FieldError),
}

impl fmt::Display for ExtensionFieldError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroDegree => formatter.write_str("extension degree must be positive"),
            Self::ModulusLength { expected, actual } => write!(
                formatter,
                "a modulus of the expected degree needs {expected} coefficients, got {actual}"
            ),
            Self::ModulusNotMonic => formatter.write_str("the modulus polynomial must be monic"),
            Self::ReducibleModulus => {
                formatter.write_str("the modulus polynomial is reducible over the base field")
            }
            Self::ProductTooLong { maximum, actual } => write!(
                formatter,
                "reduction accepts at most {maximum} coefficients, got {actual}"
            ),
            Self::BaseField(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for ExtensionFieldError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::BaseField(error) => Some(error),
            _ => None,
        }
    }
}

impl From<FieldError> for ExtensionFieldError {
    fn from(error: FieldError) -> Self {
        Self::BaseField(error)
    }
}
