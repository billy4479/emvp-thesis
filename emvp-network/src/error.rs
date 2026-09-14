//! A failed codec or handshake operation.

use std::fmt;

use crate::frame::FrameKind;

/// A stable application-level error code carried by the wire `Error` frame.
///
/// Codes 1-9 are produced by the codec while parsing a frame, codes 10-20 by
/// the server while handling a well-formed request, and code 21 is the
/// catch-all for unexpected internal failures. The numeric values are part
/// of the wire protocol and must not be renumbered.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(u32)]
pub enum ErrorCode {
    /// The frame carried a message kind this binary does not know.
    UnknownFrameKind = 1,
    /// The frame or the stream ended before the declared content arrived.
    TruncatedFrame = 2,
    /// A message finished before the end of its frame payload.
    TrailingFrameBytes = 3,
    /// An encoded field element was not below the protocol modulus.
    NonCanonicalFieldElement = 4,
    /// A wire integer did not fit the platform representation it targets.
    ValueOutOfRange = 5,
    /// Two length or count fields of one record contradicted each other.
    CountMismatch = 6,
    /// Dimension or size arithmetic overflowed.
    DimensionOverflow = 7,
    /// A decode-side buffer allocation was refused.
    AllocationFailed = 8,
    /// Diagnostic text was not valid UTF-8.
    InvalidUtf8 = 9,
    /// Uploaded protocol parameters were structurally malformed.
    InvalidParameters = 10,
    /// An artifact belongs to a different matrix instance than its record.
    InstanceMismatch = 11,
    /// An evaluation named a matrix identifier this session never loaded.
    UnknownMatrixId = 12,
    /// One matrix identifier appeared twice in one evaluation request.
    DuplicateMatrixEntry = 13,
    /// One query identifier appeared twice for a single matrix entry.
    DuplicateQueryId = 14,
    /// A request that requires entries carried none.
    EmptyRequest = 15,
    /// An evaluation arrived before the session loaded a matrix set.
    NotLoaded = 16,
    /// A second matrix-set upload arrived on an already loaded session.
    AlreadyLoaded = 17,
    /// The one permitted matrix-set upload failed.
    UploadFailed = 18,
    /// A GPU operation failed after the session had loaded its matrices.
    GpuFailure = 19,
    /// A frame arrived in the wrong direction for this connection role.
    UnexpectedFrame = 20,
    /// An internal server failure that has no more specific code.
    Internal = 21,
}

impl ErrorCode {
    /// The wire representation of this code.
    #[must_use]
    pub const fn to_u32(self) -> u32 {
        self as u32
    }

    /// The code with the given wire representation, if any.
    #[must_use]
    pub const fn from_u32(code: u32) -> Option<Self> {
        match code {
            1 => Some(Self::UnknownFrameKind),
            2 => Some(Self::TruncatedFrame),
            3 => Some(Self::TrailingFrameBytes),
            4 => Some(Self::NonCanonicalFieldElement),
            5 => Some(Self::ValueOutOfRange),
            6 => Some(Self::CountMismatch),
            7 => Some(Self::DimensionOverflow),
            8 => Some(Self::AllocationFailed),
            9 => Some(Self::InvalidUtf8),
            10 => Some(Self::InvalidParameters),
            11 => Some(Self::InstanceMismatch),
            12 => Some(Self::UnknownMatrixId),
            13 => Some(Self::DuplicateMatrixEntry),
            14 => Some(Self::DuplicateQueryId),
            15 => Some(Self::EmptyRequest),
            16 => Some(Self::NotLoaded),
            17 => Some(Self::AlreadyLoaded),
            18 => Some(Self::UploadFailed),
            19 => Some(Self::GpuFailure),
            20 => Some(Self::UnexpectedFrame),
            21 => Some(Self::Internal),
            _ => None,
        }
    }
}

impl fmt::Display for ErrorCode {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::UnknownFrameKind => "unknown frame kind",
            Self::TruncatedFrame => "truncated frame",
            Self::TrailingFrameBytes => "trailing bytes in frame",
            Self::NonCanonicalFieldElement => "noncanonical field element",
            Self::ValueOutOfRange => "wire value out of range",
            Self::CountMismatch => "inconsistent counts",
            Self::DimensionOverflow => "dimension arithmetic overflowed",
            Self::AllocationFailed => "decode allocation refused",
            Self::InvalidUtf8 => "diagnostic text is not valid UTF-8",
            Self::InvalidParameters => "invalid protocol parameters",
            Self::InstanceMismatch => "artifact instance mismatch",
            Self::UnknownMatrixId => "unknown matrix identifier",
            Self::DuplicateMatrixEntry => "duplicate matrix entry",
            Self::DuplicateQueryId => "duplicate query identifier",
            Self::EmptyRequest => "empty request",
            Self::NotLoaded => "no matrix set loaded",
            Self::AlreadyLoaded => "matrix set already loaded",
            Self::UploadFailed => "matrix set upload failed",
            Self::GpuFailure => "GPU operation failed",
            Self::UnexpectedFrame => "unexpected frame",
            Self::Internal => "internal failure",
        };
        formatter.write_str(text)
    }
}

/// A failed encode, decode, or transport operation.
///
/// Transport failures ([`CodecError::Io`]) leave the connection in an
/// unknown state and are never answered with an `Error` frame; every other
/// variant describes a fully framed message the peer can be told about.
#[derive(Debug)]
#[non_exhaustive]
pub enum CodecError {
    /// The transport failed. The stream state is unknown.
    Io(std::io::Error),
    /// The stream or frame ended before the declared content arrived.
    TruncatedFrame,
    /// A message parsed successfully but its frame still held bytes.
    TrailingFrameBytes {
        /// The number of unparsed bytes left in the frame payload.
        extra: u64,
    },
    /// The frame carried a message kind this binary does not know.
    UnknownFrameKind {
        /// The rejected kind byte.
        kind: u8,
    },
    /// A frame arrived with a valid but wrong message kind for this role.
    UnexpectedFrame {
        /// The kind the reader required.
        expected: FrameKind,
        /// The kind byte the frame carried.
        actual: u8,
    },
    /// The handshake magic did not match [`crate::MAGIC`].
    InvalidMagic,
    /// The handshake status byte was neither accept nor reject.
    InvalidHandshakeStatus {
        /// The rejected status byte.
        status: u8,
    },
    /// An encoded field element was not below the protocol modulus.
    NonCanonicalField {
        /// The rejected integer value.
        value: u32,
    },
    /// A wire integer did not fit its host representation.
    ValueOutOfRange {
        /// The rejected field's name.
        name: &'static str,
        /// The rejected wire value.
        value: u64,
    },
    /// A record declared a dimension that cannot be nonzero-length.
    InvalidDimensions {
        /// The rejected dimension's name.
        name: &'static str,
        /// The rejected wire value.
        value: u64,
    },
    /// Two length or count fields of one record contradicted each other.
    CountMismatch {
        /// The inconsistent field's name.
        name: &'static str,
        /// The value the other fields imply.
        expected: u64,
        /// The value the field declared.
        actual: u64,
    },
    /// Dimension or size arithmetic overflowed while computing a length.
    DimensionOverflow,
    /// A decode-side buffer allocation was refused.
    AllocationFailed,
    /// Diagnostic text was not valid UTF-8.
    InvalidUtf8,
}

impl CodecError {
    /// The application error code that describes this failure in an
    /// `Error` frame.
    ///
    /// Transport failures map to [`ErrorCode::Internal`] because a broken
    /// stream cannot carry a response anyway; callers should never send one.
    #[must_use]
    pub const fn error_code(&self) -> ErrorCode {
        match self {
            Self::Io(_) | Self::InvalidMagic | Self::InvalidHandshakeStatus { .. } => {
                ErrorCode::Internal
            }
            Self::DimensionOverflow => ErrorCode::DimensionOverflow,
            Self::TruncatedFrame => ErrorCode::TruncatedFrame,
            Self::TrailingFrameBytes { .. } => ErrorCode::TrailingFrameBytes,
            Self::UnknownFrameKind { .. } => ErrorCode::UnknownFrameKind,
            Self::UnexpectedFrame { .. } => ErrorCode::UnexpectedFrame,
            Self::NonCanonicalField { .. } => ErrorCode::NonCanonicalFieldElement,
            Self::ValueOutOfRange { .. } => ErrorCode::ValueOutOfRange,
            Self::InvalidDimensions { .. } => ErrorCode::InvalidParameters,
            Self::CountMismatch { .. } => ErrorCode::CountMismatch,
            Self::AllocationFailed => ErrorCode::AllocationFailed,
            Self::InvalidUtf8 => ErrorCode::InvalidUtf8,
        }
    }
}

impl fmt::Display for CodecError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Io(error) => error.fmt(formatter),
            Self::TruncatedFrame => formatter.write_str("frame ended before its content arrived"),
            Self::TrailingFrameBytes { extra } => {
                write!(formatter, "message finished with {extra} frame bytes left")
            }
            Self::UnknownFrameKind { kind } => {
                write!(formatter, "unknown frame kind {kind}")
            }
            Self::UnexpectedFrame { expected, actual } => {
                write!(formatter, "expected a {expected} frame, got kind {actual}")
            }
            Self::InvalidMagic => formatter.write_str("handshake magic did not match"),
            Self::InvalidHandshakeStatus { status } => {
                write!(
                    formatter,
                    "handshake status byte {status} is neither accept nor reject"
                )
            }
            Self::NonCanonicalField { value } => {
                write!(
                    formatter,
                    "encoded field element {value} is not below the protocol modulus"
                )
            }
            Self::ValueOutOfRange { name, value } => {
                write!(
                    formatter,
                    "{name} value {value} does not fit the host representation"
                )
            }
            Self::InvalidDimensions { name, value } => {
                write!(formatter, "{name} must be positive, got {value}")
            }
            Self::CountMismatch {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "{name} count mismatch: expected {expected}, got {actual}"
            ),
            Self::DimensionOverflow => formatter.write_str("dimension arithmetic overflowed"),
            Self::AllocationFailed => formatter.write_str("decode buffer allocation was refused"),
            Self::InvalidUtf8 => formatter.write_str("diagnostic text is not valid UTF-8"),
        }
    }
}

impl std::error::Error for CodecError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(error) => Some(error),
            _ => None,
        }
    }
}

/// A failed version and modulus handshake.
#[derive(Debug)]
#[non_exhaustive]
pub enum HandshakeError {
    /// The handshake exchange failed at the transport or codec level.
    Codec(CodecError),
    /// The server rejected the client's hello.
    Rejected,
    /// The client offered a protocol version this server does not speak.
    UnsupportedVersion {
        /// The rejected version.
        received: u16,
    },
    /// The client offered a field modulus this server does not implement.
    UnsupportedModulus {
        /// The rejected modulus.
        received: u32,
    },
}

impl fmt::Display for HandshakeError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Codec(error) => error.fmt(formatter),
            Self::Rejected => formatter.write_str("the server rejected the handshake"),
            Self::UnsupportedVersion { received } => {
                write!(formatter, "unsupported protocol version {received}")
            }
            Self::UnsupportedModulus { received } => {
                write!(formatter, "unsupported field modulus {received}")
            }
        }
    }
}

impl std::error::Error for HandshakeError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Codec(error) => Some(error),
            _ => None,
        }
    }
}

impl From<CodecError> for HandshakeError {
    fn from(error: CodecError) -> Self {
        Self::Codec(error)
    }
}
