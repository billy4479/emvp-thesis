//! The error and upload-acknowledgment message codecs.
//!
//! Protocol v2 moves the bulk messages (matrix uploads, evaluations,
//! products) to the zero-copy view pipeline in [`crate::v2`]; this module
//! keeps only the surface that stays off the hot path by decision: the
//! structured [`ErrorResponse`] record with its writer and readers, and
//! the `UploadAccepted` acknowledgment pair, which carries only `u64`
//! identifiers and never touches the field-element arena.
//!
//! Full-frame functions read or write the frame header plus the payload
//! and are what a role that only ever produces or consumes whole messages
//! uses. Payload-level functions parse from an already open
//! [`FrameReader`], which is how the server dispatches on the header kind
//! before handing the payload to the message parser.

use std::fmt;
use std::io::{Read, Write};

use crate::error::{CodecError, ErrorCode};
use crate::frame::{
    FrameKind, FrameReader, HEADER_BYTES, checked_size, host_len, read_frame_header, wire_len,
    write_frame_header, write_u32, write_u64,
};

/// The server's structured rejection of a request.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ErrorResponse {
    /// The stable numeric code describing the failure class.
    pub code: u32,
    /// Human-readable diagnostic text.
    pub message: String,
}

impl ErrorResponse {
    /// The numeric code as an [`ErrorCode`], if it is one this binary knows.
    #[must_use]
    pub const fn code(&self) -> Option<ErrorCode> {
        ErrorCode::from_u32(self.code)
    }
}

impl fmt::Display for ErrorResponse {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(formatter, "error {}: {}", self.code, self.message)
    }
}

impl std::error::Error for ErrorResponse {}

/// Writes the accepted upload's matrix identifiers, in upload order.
///
/// Returns the total bytes written, header included.
///
/// # Errors
///
/// Returns [`CodecError::DimensionOverflow`] if the message exceeds the
/// wire length space, and [`CodecError::Io`] if the transport fails.
pub fn write_upload_accepted(
    writer: &mut impl Write,
    identifiers: &[u64],
) -> Result<u64, CodecError> {
    let count = wire_len(identifiers.len())?;
    let ids = count
        .checked_mul(size_of::<u64>() as u64)
        .ok_or(CodecError::DimensionOverflow)?;
    let payload_len = checked_size(8, ids)?;
    write_frame_header(writer, FrameKind::UploadAccepted, payload_len)?;
    write_u64(writer, count)?;
    for &identifier in identifiers {
        write_u64(writer, identifier)?;
    }
    checked_size(payload_len, HEADER_BYTES)
}

/// Writes a structured rejection of a request.
///
/// Returns the total bytes written, header included.
///
/// # Errors
///
/// Returns [`CodecError::DimensionOverflow`] if the message exceeds the
/// wire length space, and [`CodecError::Io`] if the transport fails.
pub fn write_error(
    writer: &mut impl Write,
    code: ErrorCode,
    message: &str,
) -> Result<u64, CodecError> {
    let text_bytes = wire_len(message.len())?;
    let payload_len = checked_size(4 + 8, text_bytes)?;
    write_frame_header(writer, FrameKind::Error, payload_len)?;
    write_u32(writer, code.to_u32())?;
    write_u64(writer, text_bytes)?;
    writer
        .write_all(message.as_bytes())
        .map_err(CodecError::Io)?;
    checked_size(payload_len, HEADER_BYTES)
}

/// Reads the accepted-upload payload from an open frame.
///
/// The server assigns one-based consecutive identifiers, so a decoded
/// identifier of zero is a protocol violation and is rejected here. The
/// client still validates identifier uniqueness itself, because the codec
/// only owns the nonzero invariant, not the per-connection assignment.
///
/// # Errors
///
/// Returns the errors of [`crate::v2::plan_upload_accepted`].
pub fn read_upload_accepted_payload<R: Read>(
    frame: &mut FrameReader<'_, R>,
) -> Result<Vec<u64>, CodecError> {
    let plan = crate::v2::plan_upload_accepted(frame)?;
    let mut workspace = crate::v2::UploadAcceptedWorkspace::new();
    workspace.reserve(&plan)?;
    let identifiers = crate::v2::decode_upload_accepted(frame, &plan, &mut workspace)?;
    Ok(identifiers.to_vec())
}

/// Reads the error payload from an open frame.
///
/// # Errors
///
/// Returns [`CodecError::InvalidUtf8`] for non-UTF-8 diagnostic text, and
/// the framing errors of [`FrameReader`].
pub fn read_error_payload<R: Read>(
    frame: &mut FrameReader<'_, R>,
) -> Result<ErrorResponse, CodecError> {
    let code = frame.read_u32()?;
    let message_len = frame.read_u64()?;
    if message_len > frame.remaining() {
        return Err(CodecError::TruncatedFrame);
    }
    let text_len = host_len("error message", message_len)?;
    let mut text = Vec::new();
    text.try_reserve_exact(text_len)
        .map_err(|_reserve| CodecError::AllocationFailed)?;
    text.resize(text_len, 0);
    frame.read_exact_checked(&mut text)?;
    let message = String::from_utf8(text).map_err(|_utf8| CodecError::InvalidUtf8)?;
    Ok(ErrorResponse { code, message })
}

/// Reads a whole `UploadAccepted` frame.
///
/// Returns the parsed identifiers and the frame's payload length.
///
/// # Errors
///
/// Returns [`CodecError::UnexpectedFrame`] for any other frame kind, and
/// the payload errors of [`read_upload_accepted_payload`].
pub fn read_upload_accepted<R: Read>(reader: &mut R) -> Result<(Vec<u64>, u64), CodecError> {
    let (mut frame, payload_len) = open_frame(reader, FrameKind::UploadAccepted)?;
    let message = read_upload_accepted_payload(&mut frame)?;
    frame.finish()?;
    Ok((message, payload_len))
}

/// Reads a whole `Error` frame.
///
/// Returns the parsed message and the frame's payload length.
///
/// # Errors
///
/// Returns [`CodecError::UnexpectedFrame`] for any other frame kind, and
/// the payload errors of [`read_error_payload`].
pub fn read_error<R: Read>(reader: &mut R) -> Result<(ErrorResponse, u64), CodecError> {
    let (mut frame, payload_len) = open_frame(reader, FrameKind::Error)?;
    let message = read_error_payload(&mut frame)?;
    frame.finish()?;
    Ok((message, payload_len))
}

/// Reads a frame header and opens its payload, requiring one kind.
fn open_frame<R: Read>(
    reader: &mut R,
    expected: FrameKind,
) -> Result<(FrameReader<'_, R>, u64), CodecError> {
    let Some(header) = read_frame_header(reader)? else {
        return Err(CodecError::TruncatedFrame);
    };
    if header.kind != expected {
        return Err(CodecError::UnexpectedFrame {
            expected,
            actual: header.kind.to_u8(),
        });
    }
    Ok((
        FrameReader::new(reader, header.payload_len),
        header.payload_len,
    ))
}
