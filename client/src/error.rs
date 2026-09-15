//! A failed client run.

use std::fmt;

use emvp::{ParamsError, PrfError, ProtocolError};
use emvp_network::{CodecError, ErrorResponse, HandshakeError};
use trapdoor_matrices::SecurityWarningKind;

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
    /// Master-seed derivation failed.
    Prf(PrfError),
    /// The operating system refused a fresh master-seed draw.
    Seed(getrandom::Error),
    /// The requested width is too small for the compiled Ring-LPN weight.
    WidthTooSmall {
        /// The code width `n = 2k` the search produced.
        width: usize,
        /// The compiled Ring-LPN secret weight.
        weight: usize,
    },
    /// The requested Ring-LPN ring degree is not a power of two, so no
    /// automatic irreducible modulus can be selected for it.
    RingWidthNotPowerOfTwo {
        /// The requested ring degree (`n = 2k`).
        width: usize,
    },
    /// The requested Ring-LPN configuration was assessed broken and
    /// sampling refuses it.
    InsecureRing {
        /// The requested ring degree (`n = 2k`).
        degree: usize,
        /// The compiled Ring-LPN secret weight.
        weight: usize,
        /// The broken-forcing assessment reasons.
        reasons: Vec<SecurityWarningKind>,
    },
    /// The row-count list was empty or held a zero.
    Rows,
    /// The vector-width argument was zero.
    Width,
    /// The per-matrix query count was zero.
    Queries,
    /// The row count times the record length overflowed.
    RowOverflow {
        /// The requested row count.
        rows: usize,
        /// The record length.
        ell: usize,
    },
    /// The server's upload acknowledgment did not match the upload.
    UploadAcknowledge {
        /// The number of uploaded matrices.
        uploaded: usize,
        /// The number of acknowledged matrices.
        acknowledged: usize,
    },
    /// The server's upload acknowledgment carried invalid identifiers.
    UploadIdentifiers {
        /// The problem the acknowledgment had.
        problem: &'static str,
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
            Self::Prf(error) => error.fmt(formatter),
            Self::Seed(error) => write!(
                formatter,
                "drawing a fresh master seed from the OS failed: {error}"
            ),
            Self::WidthTooSmall { width, weight } => write!(
                formatter,
                "the searched code width n = {width} is below the compiled Ring-LPN \
                 weight {weight}; choose a larger --width"
            ),
            Self::RingWidthNotPowerOfTwo { width } => write!(
                formatter,
                "the requested Ring-LPN ring degree n = {width} is not a power of two, \
                 so no automatic irreducible ring modulus can be selected; choose a \
                 --width whose n = 2k is a power of two"
            ),
            Self::InsecureRing {
                degree,
                weight,
                reasons,
            } => {
                write!(
                    formatter,
                    "the requested Ring-LPN configuration (degree {degree}, weight {weight}) \
                     is assessed broken and sampling was refused: "
                )?;
                for (position, reason) in reasons.iter().enumerate() {
                    if position > 0 {
                        formatter.write_str("; ")?;
                    }
                    write!(formatter, "{reason}")?;
                }
                write!(
                    formatter,
                    "; choose a larger --width (the ring degree must reach the \
                     construction's floor) or a different --mask"
                )
            }
            Self::Rows => formatter.write_str("--rows must name at least one nonzero row count"),
            Self::Width => formatter.write_str("--width must be positive"),
            Self::Queries => formatter.write_str("--queries must be positive"),
            Self::RowOverflow { rows, ell } => write!(
                formatter,
                "rows {rows} times the record length {ell} overflows the address space"
            ),
            Self::UploadAcknowledge {
                uploaded,
                acknowledged,
            } => write!(
                formatter,
                "the server acknowledged {acknowledged} of {uploaded} uploaded matrices"
            ),
            Self::UploadIdentifiers { problem } => {
                write!(
                    formatter,
                    "the upload acknowledgment is malformed: {problem}"
                )
            }
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
            Self::Prf(error) => Some(error),
            // `getrandom::Error` carries no `std::error::Error` source.
            Self::Seed(_) | _ => None,
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

impl From<PrfError> for RunError {
    fn from(error: PrfError) -> Self {
        Self::Prf(error)
    }
}

impl From<getrandom::Error> for RunError {
    fn from(error: getrandom::Error) -> Self {
        Self::Seed(error)
    }
}
