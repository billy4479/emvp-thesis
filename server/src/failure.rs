//! A rejected server request and its wire error code.

use std::fmt;

use emvp::ParamsError;
use emvp_network::ErrorCode;

/// A request the server refused before or during evaluation.
///
/// Every variant maps onto the stable [`ErrorCode`] carried by the `Error`
/// frame, plus human-readable text for the frame's diagnostic field.
#[derive(Debug)]
#[non_exhaustive]
pub enum RequestFailure {
    /// Uploaded protocol parameters were structurally malformed.
    Params(ParamsError),
    /// A matrix's declared width disagreed with its parameters.
    ColumnsMismatch {
        /// The width the parameters imply (`n = 2k`).
        expected: usize,
        /// The width the record declared.
        actual: usize,
    },
    /// An encrypted query's width disagreed with its matrix.
    QueryWidth {
        /// The matrix width `n`.
        expected: usize,
        /// The query's declared width.
        actual: usize,
    },
    /// An artifact belongs to a different matrix instance.
    InstanceMismatch {
        /// The instance identifier of the stored matrix.
        expected: u128,
        /// The instance identifier the artifact carried.
        actual: u128,
    },
    /// The evaluation named a matrix identifier this session never loaded.
    UnknownMatrix {
        /// The rejected identifier.
        id: u64,
    },
    /// One matrix identifier appeared twice in one evaluation request.
    DuplicateMatrix {
        /// The repeated identifier.
        id: u64,
    },
    /// One query identifier appeared twice for a single matrix entry.
    DuplicateQuery {
        /// The matrix entry the repeat occurred in.
        matrix_id: u64,
        /// The repeated query identifier.
        query_id: u64,
    },
    /// A request that requires entries carried none.
    EmptyRequest,
    /// Dimension arithmetic overflowed while sizing a request.
    DimensionOverflow,
}

impl RequestFailure {
    /// The stable wire error code for this failure.
    #[must_use]
    pub const fn code(&self) -> ErrorCode {
        match self {
            Self::Params(_) | Self::DimensionOverflow => ErrorCode::InvalidParameters,
            Self::ColumnsMismatch { .. } | Self::QueryWidth { .. } => ErrorCode::CountMismatch,
            Self::InstanceMismatch { .. } => ErrorCode::InstanceMismatch,
            Self::UnknownMatrix { .. } => ErrorCode::UnknownMatrixId,
            Self::DuplicateMatrix { .. } => ErrorCode::DuplicateMatrixEntry,
            Self::DuplicateQuery { .. } => ErrorCode::DuplicateQueryId,
            Self::EmptyRequest => ErrorCode::EmptyRequest,
        }
    }
}

impl fmt::Display for RequestFailure {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Params(error) => error.fmt(formatter),
            Self::ColumnsMismatch { expected, actual } => {
                write!(
                    formatter,
                    "matrix width mismatch: expected {expected}, got {actual}"
                )
            }
            Self::QueryWidth { expected, actual } => {
                write!(
                    formatter,
                    "query width mismatch: expected {expected}, got {actual}"
                )
            }
            Self::InstanceMismatch { expected, actual } => {
                write!(
                    formatter,
                    "instance mismatch: expected {expected}, got {actual}"
                )
            }
            Self::UnknownMatrix { id } => write!(formatter, "unknown matrix identifier {id}"),
            Self::DuplicateMatrix { id } => {
                write!(
                    formatter,
                    "matrix identifier {id} appears twice in one request"
                )
            }
            Self::DuplicateQuery {
                matrix_id,
                query_id,
            } => write!(
                formatter,
                "query identifier {query_id} appears twice for matrix {matrix_id}"
            ),
            Self::EmptyRequest => formatter.write_str("request carried no entries"),
            Self::DimensionOverflow => formatter.write_str("dimension arithmetic overflowed"),
        }
    }
}

impl std::error::Error for RequestFailure {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Params(error) => Some(error),
            _ => None,
        }
    }
}
