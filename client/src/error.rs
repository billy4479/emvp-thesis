//! A failed client run.

use std::fmt;

use emvp::{ParamsError, ProtocolError};
use emvp_network::{CodecError, ErrorResponse, HandshakeError};

/// A step of the demo run that did not complete.
#[derive(Debug)]
#[non_exhaustive]
pub enum RunError {
    /// The transport failed.
    Io(std::io::Error),
    /// The handshake was rejected or failed.
    Handshake(HandshakeError),
    /// A wire message could not be exchanged.
    Codec(CodecError),
    /// The server answered a request with a structured error.
    Server(ErrorResponse),
    /// Local protocol work (derivation, encryption, queries) failed.
    Protocol(ProtocolError),
    /// Parameter selection failed for the requested vector width.
    Params(ParamsError),
    /// The requested width is too small for the compiled Ring-LPN weight.
    WidthTooSmall {
        /// The code width `n = 2k` the search produced.
        width: usize,
        /// The compiled Ring-LPN secret weight.
        weight: usize,
    },
    /// The row-count list was empty or held a zero.
    Rows,
    /// The vector-width argument was zero.
    Width,
    /// The per-matrix query count was zero.
    Queries,
    /// The server's upload acknowledgment did not match the upload.
    UploadAcknowledge {
        /// The number of uploaded matrices.
        uploaded: usize,
        /// The number of acknowledged matrices.
        acknowledged: usize,
    },
    /// The server's products did not match the evaluation request.
    ProductsShape {
        /// The problem the products response had.
        problem: &'static str,
    },
    /// A decoded product disagreed with the plaintext product.
    Verification {
        /// The matrix whose product disagreed.
        matrix_id: u64,
        /// The query whose product disagreed.
        query_id: u64,
    },
}

impl fmt::Display for RunError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => error.fmt(formatter),
            Self::Handshake(error) => error.fmt(formatter),
            Self::Codec(error) => error.fmt(formatter),
            Self::Server(error) => write!(formatter, "server rejected the request: {error}"),
            Self::Protocol(error) => error.fmt(formatter),
            Self::Params(error) => error.fmt(formatter),
            Self::WidthTooSmall { width, weight } => write!(
                formatter,
                "the searched code width n = {width} is below the compiled Ring-LPN \
                 weight {weight}; choose a larger --width"
            ),
            Self::Rows => formatter.write_str("--rows must name at least one nonzero row count"),
            Self::Width => formatter.write_str("--width must be positive"),
            Self::Queries => formatter.write_str("--queries must be positive"),
            Self::UploadAcknowledge {
                uploaded,
                acknowledged,
            } => write!(
                formatter,
                "the server acknowledged {acknowledged} of {uploaded} uploaded matrices"
            ),
            Self::ProductsShape { problem } => {
                write!(formatter, "the products response is malformed: {problem}")
            }
            Self::Verification {
                matrix_id,
                query_id,
            } => write!(
                formatter,
                "the decoded product for matrix {matrix_id}, query {query_id} does not \
                 match the plaintext product"
            ),
        }
    }
}

impl std::error::Error for RunError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            Self::Handshake(error) => Some(error),
            Self::Codec(error) => Some(error),
            Self::Server(error) => Some(error),
            Self::Protocol(error) => Some(error),
            Self::Params(error) => Some(error),
            _ => None,
        }
    }
}

impl From<std::io::Error> for RunError {
    fn from(error: std::io::Error) -> Self {
        Self::Io(error)
    }
}

impl From<HandshakeError> for RunError {
    fn from(error: HandshakeError) -> Self {
        Self::Handshake(error)
    }
}

impl From<CodecError> for RunError {
    fn from(error: CodecError) -> Self {
        Self::Codec(error)
    }
}

impl From<ProtocolError> for RunError {
    fn from(error: ProtocolError) -> Self {
        Self::Protocol(error)
    }
}

impl From<ParamsError> for RunError {
    fn from(error: ParamsError) -> Self {
        Self::Params(error)
    }
}
