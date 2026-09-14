//! The protocol messages and their payload codecs.
//!
//! Every function comes in two levels. Full-frame functions read or write
//! the frame header plus the payload and are what a role that only ever
//! produces or consumes whole messages uses. Payload-level functions parse
//! from an already open [`FrameReader`], which is how the server dispatches
//! on the header kind before handing the payload to the message parser.
//!
//! All multi-byte integers are little-endian. Lengths and dimensions
//! travel as `u64`; each message's redundant counts are cross-checked
//! before any element payload is read.

use std::fmt;
use std::io::{Read, Write};

use emvp::{AnswerMatrix, EmvpParams, EncryptedMatrix, EncryptedQuery};

use crate::error::{CodecError, ErrorCode};
use crate::frame::{
    FrameKind, FrameReader, HEADER_BYTES, PROTOCOL_MODULUS, checked_size, field_byte_len, host_len,
    read_frame_header, wire_len, write_field_slice, write_frame_header, write_u32, write_u64,
    write_u128,
};

/// Bytes in one little-endian `u64`.
const U64_BYTES: u64 = size_of::<u64>() as u64;

/// One uploaded encrypted matrix together with the parameters it was
/// encrypted under.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct MatrixUpload {
    /// The protocol parameters the matrix was encrypted under.
    pub params: EmvpParams,
    /// The encrypted matrix with its public instance identifier.
    pub matrix: EncryptedMatrix<PROTOCOL_MODULUS>,
}

/// One evaluation entry: the encrypted queries for a single loaded matrix.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EvaluateEntry {
    /// The server identifier the upload assigned to the matrix.
    pub matrix_id: u64,
    /// The encrypted queries against that matrix, in order.
    pub queries: Vec<EncryptedQuery<PROTOCOL_MODULUS>>,
}

/// One product entry: the encrypted answers for a single requested matrix.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct ProductEntry {
    /// The server identifier of the answered matrix.
    pub matrix_id: u64,
    /// The encrypted answers, in the entry's query order.
    pub answers: Vec<AnswerMatrix<PROTOCOL_MODULUS>>,
}

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

/// Bytes in one matrix record's fixed prefix: four parameter fields, the
/// instance identifier, the two dimensions, and the element count.
const MATRIX_FIXED_BYTES: u64 = 8 + 8 + 8 + 4 + 16 + 8 + 8 + 8;

/// Bytes in one query record's fixed prefix.
const QUERY_FIXED_BYTES: u64 = 16 + 8 + 8;

/// Bytes in one answer record's fixed prefix.
const ANSWER_FIXED_BYTES: u64 = 16 + 8 + 8 + 8 + 8;

/// The payload length of an `UploadMatrices` frame.
fn upload_matrices_payload_len(uploads: &[MatrixUpload]) -> Result<u64, CodecError> {
    let mut total = U64_BYTES;
    for upload in uploads {
        let values = field_byte_len(wire_len(upload.matrix.values().len())?)?;
        total = checked_size(total, MATRIX_FIXED_BYTES)?;
        total = checked_size(total, values)?;
    }
    Ok(total)
}

/// The payload length of an `UploadAccepted` frame.
fn upload_accepted_payload_len(identifiers: &[u64]) -> Result<u64, CodecError> {
    let count = wire_len(identifiers.len())?;
    let ids = count
        .checked_mul(U64_BYTES)
        .ok_or(CodecError::DimensionOverflow)?;
    checked_size(U64_BYTES, ids)
}

/// The payload length of an `Evaluate` frame.
fn evaluate_payload_len(entries: &[EvaluateEntry]) -> Result<u64, CodecError> {
    let mut total = U64_BYTES;
    for entry in entries {
        total = checked_size(total, 8 + 8)?;
        for query in &entry.queries {
            let values = field_byte_len(wire_len(query.values().len())?)?;
            total = checked_size(total, QUERY_FIXED_BYTES)?;
            total = checked_size(total, values)?;
        }
    }
    Ok(total)
}

/// The payload length of a `Products` frame.
fn products_payload_len(entries: &[ProductEntry]) -> Result<u64, CodecError> {
    let mut total = U64_BYTES;
    for entry in entries {
        total = checked_size(total, 8 + 8)?;
        for answer in &entry.answers {
            let values = field_byte_len(wire_len(answer.values().len())?)?;
            total = checked_size(total, ANSWER_FIXED_BYTES)?;
            total = checked_size(total, values)?;
        }
    }
    Ok(total)
}

/// The payload length of an `Error` frame: a `u32` code, a `u64` text
/// length, and the text itself.
fn error_payload_len(message: &str) -> Result<u64, CodecError> {
    checked_size(4 + 8, wire_len(message.len())?)
}

/// Writes the one matrix-set upload of a session.
///
/// Returns the total bytes written, header included.
///
/// # Errors
///
/// Returns [`CodecError::DimensionOverflow`] if the message exceeds the
/// wire length space, and [`CodecError::Io`] if the transport fails.
pub fn write_upload_matrices(
    writer: &mut impl Write,
    uploads: &[MatrixUpload],
) -> Result<u64, CodecError> {
    let payload_len = upload_matrices_payload_len(uploads)?;
    write_frame_header(writer, FrameKind::UploadMatrices, payload_len)?;
    write_u64(writer, wire_len(uploads.len())?)?;
    for upload in uploads {
        write_u64(writer, wire_len(upload.params.k)?)?;
        write_u64(writer, wire_len(upload.params.ell)?)?;
        write_u64(writer, wire_len(upload.params.b)?)?;
        write_u32(writer, upload.params.lambda)?;
        write_u128(writer, upload.matrix.instance_id())?;
        write_u64(writer, wire_len(upload.matrix.rows())?)?;
        write_u64(writer, wire_len(upload.matrix.columns())?)?;
        write_u64(writer, wire_len(upload.matrix.values().len())?)?;
        write_field_slice(writer, upload.matrix.values())?;
    }
    checked_size(payload_len, HEADER_BYTES)
}

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
    let payload_len = upload_accepted_payload_len(identifiers)?;
    write_frame_header(writer, FrameKind::UploadAccepted, payload_len)?;
    write_u64(writer, wire_len(identifiers.len())?)?;
    for &identifier in identifiers {
        write_u64(writer, identifier)?;
    }
    checked_size(payload_len, HEADER_BYTES)
}

/// Writes one ordered evaluation request.
///
/// Returns the total bytes written, header included.
///
/// # Errors
///
/// Returns [`CodecError::DimensionOverflow`] if the message exceeds the
/// wire length space, and [`CodecError::Io`] if the transport fails.
pub fn write_evaluate(
    writer: &mut impl Write,
    entries: &[EvaluateEntry],
) -> Result<u64, CodecError> {
    let payload_len = evaluate_payload_len(entries)?;
    write_frame_header(writer, FrameKind::Evaluate, payload_len)?;
    write_u64(writer, wire_len(entries.len())?)?;
    for entry in entries {
        write_u64(writer, entry.matrix_id)?;
        write_u64(writer, wire_len(entry.queries.len())?)?;
        for query in &entry.queries {
            write_u128(writer, query.instance_id())?;
            write_u64(writer, query.query_id())?;
            write_u64(writer, wire_len(query.values().len())?)?;
            write_field_slice(writer, query.values())?;
        }
    }
    checked_size(payload_len, HEADER_BYTES)
}

/// Writes the encrypted products of one evaluation request.
///
/// Returns the total bytes written, header included.
///
/// # Errors
///
/// Returns [`CodecError::DimensionOverflow`] if the message exceeds the
/// wire length space, and [`CodecError::Io`] if the transport fails.
pub fn write_products(
    writer: &mut impl Write,
    entries: &[ProductEntry],
) -> Result<u64, CodecError> {
    let payload_len = products_payload_len(entries)?;
    write_frame_header(writer, FrameKind::Products, payload_len)?;
    write_u64(writer, wire_len(entries.len())?)?;
    for entry in entries {
        write_u64(writer, entry.matrix_id)?;
        write_u64(writer, wire_len(entry.answers.len())?)?;
        for answer in &entry.answers {
            write_u128(writer, answer.instance_id())?;
            write_u64(writer, answer.query_id())?;
            write_u64(writer, wire_len(answer.rows())?)?;
            write_u64(writer, wire_len(answer.blocks())?)?;
            write_u64(writer, wire_len(answer.values().len())?)?;
            write_field_slice(writer, answer.values())?;
        }
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
    let payload_len = error_payload_len(message)?;
    write_frame_header(writer, FrameKind::Error, payload_len)?;
    write_u32(writer, code.to_u32())?;
    write_u64(writer, wire_len(message.len())?)?;
    writer
        .write_all(message.as_bytes())
        .map_err(CodecError::Io)?;
    checked_size(payload_len, HEADER_BYTES)
}

/// Reads the matrix-set upload payload from an open frame.
///
/// # Errors
///
/// Returns [`CodecError::CountMismatch`] when a record's element count
/// contradicts its dimensions, [`CodecError::ValueOutOfRange`] when a wire
/// dimension does not fit the host, and the framing errors of
/// [`FrameReader`].
pub fn read_upload_matrices_payload<R: Read>(
    frame: &mut FrameReader<'_, R>,
) -> Result<Vec<MatrixUpload>, CodecError> {
    let count = frame.read_u64()?;
    let mut uploads = Vec::new();
    for _ in 0..count {
        let k = frame.read_u64()?;
        let ell = frame.read_u64()?;
        let b = frame.read_u64()?;
        let lambda = frame.read_u32()?;
        let instance_id = frame.read_u128()?;
        let rows = frame.read_u64()?;
        let columns = frame.read_u64()?;
        let value_count = frame.read_u64()?;
        let expected = rows
            .checked_mul(columns)
            .ok_or(CodecError::DimensionOverflow)?;
        if expected != value_count {
            return Err(CodecError::CountMismatch {
                name: "matrix values",
                expected,
                actual: value_count,
            });
        }
        if rows == 0 {
            return Err(CodecError::InvalidDimensions {
                name: "matrix rows",
                value: 0,
            });
        }
        if columns == 0 {
            return Err(CodecError::InvalidDimensions {
                name: "matrix columns",
                value: 0,
            });
        }
        let values = frame.read_field_slice(value_count)?;
        // The dimension and length checks above are exactly the
        // constructor's validation, so this arm is unreachable for any
        // decoded record.
        let Ok(matrix) = EncryptedMatrix::from_parts(
            instance_id,
            host_len("matrix rows", rows)?,
            host_len("matrix columns", columns)?,
            values,
        ) else {
            return Err(CodecError::CountMismatch {
                name: "matrix values",
                expected,
                actual: value_count,
            });
        };
        uploads.push(MatrixUpload {
            params: EmvpParams {
                k: host_len("parameter k", k)?,
                ell: host_len("parameter ell", ell)?,
                b: host_len("parameter b", b)?,
                lambda,
            },
            matrix,
        });
    }
    Ok(uploads)
}

/// Reads the accepted-upload payload from an open frame.
///
/// # Errors
///
/// Returns the framing errors of [`FrameReader`].
pub fn read_upload_accepted_payload<R: Read>(
    frame: &mut FrameReader<'_, R>,
) -> Result<Vec<u64>, CodecError> {
    let count = frame.read_u64()?;
    let mut identifiers = Vec::new();
    for _ in 0..count {
        identifiers.push(frame.read_u64()?);
    }
    Ok(identifiers)
}

/// Reads the evaluation payload from an open frame.
///
/// # Errors
///
/// Returns the framing errors of [`FrameReader`].
pub fn read_evaluate_payload<R: Read>(
    frame: &mut FrameReader<'_, R>,
) -> Result<Vec<EvaluateEntry>, CodecError> {
    let count = frame.read_u64()?;
    let mut entries = Vec::new();
    for _ in 0..count {
        let matrix_id = frame.read_u64()?;
        let query_count = frame.read_u64()?;
        let mut queries = Vec::new();
        for _ in 0..query_count {
            let instance_id = frame.read_u128()?;
            let query_id = frame.read_u64()?;
            let value_count = frame.read_u64()?;
            let values = frame.read_field_slice(value_count)?;
            queries.push(EncryptedQuery::from_parts(instance_id, query_id, values));
        }
        entries.push(EvaluateEntry { matrix_id, queries });
    }
    Ok(entries)
}

/// Reads the products payload from an open frame.
///
/// # Errors
///
/// Returns [`CodecError::CountMismatch`] when an answer's element count
/// contradicts its dimensions, and the framing errors of [`FrameReader`].
pub fn read_products_payload<R: Read>(
    frame: &mut FrameReader<'_, R>,
) -> Result<Vec<ProductEntry>, CodecError> {
    let count = frame.read_u64()?;
    let mut entries = Vec::new();
    for _ in 0..count {
        let matrix_id = frame.read_u64()?;
        let answer_count = frame.read_u64()?;
        let mut answers = Vec::new();
        for _ in 0..answer_count {
            let instance_id = frame.read_u128()?;
            let query_id = frame.read_u64()?;
            let rows = frame.read_u64()?;
            let blocks = frame.read_u64()?;
            let value_count = frame.read_u64()?;
            let expected = rows
                .checked_mul(blocks)
                .ok_or(CodecError::DimensionOverflow)?;
            if expected != value_count {
                return Err(CodecError::CountMismatch {
                    name: "answer values",
                    expected,
                    actual: value_count,
                });
            }
            let values = frame.read_field_slice(value_count)?;
            answers.push(AnswerMatrix::from_parts(
                instance_id,
                query_id,
                values,
                host_len("answer rows", rows)?,
                host_len("answer blocks", blocks)?,
            ));
        }
        entries.push(ProductEntry { matrix_id, answers });
    }
    Ok(entries)
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

/// Reads a whole `UploadMatrices` frame.
///
/// Returns the parsed message and the frame's payload length.
///
/// # Errors
///
/// Returns [`CodecError::UnexpectedFrame`] for any other frame kind, and
/// the payload errors of [`Self::read_upload_matrices_payload`].
pub fn read_upload_matrices<R: Read>(
    reader: &mut R,
) -> Result<(Vec<MatrixUpload>, u64), CodecError> {
    let (mut frame, payload_len) = open_frame(reader, FrameKind::UploadMatrices)?;
    let message = read_upload_matrices_payload(&mut frame)?;
    frame.finish()?;
    Ok((message, payload_len))
}

/// Reads a whole `UploadAccepted` frame.
///
/// Returns the parsed identifiers and the frame's payload length.
///
/// # Errors
///
/// Returns [`CodecError::UnexpectedFrame`] for any other frame kind, and
/// the payload errors of [`Self::read_upload_accepted_payload`].
pub fn read_upload_accepted<R: Read>(reader: &mut R) -> Result<(Vec<u64>, u64), CodecError> {
    let (mut frame, payload_len) = open_frame(reader, FrameKind::UploadAccepted)?;
    let message = read_upload_accepted_payload(&mut frame)?;
    frame.finish()?;
    Ok((message, payload_len))
}

/// Reads a whole `Evaluate` frame.
///
/// Returns the parsed message and the frame's payload length.
///
/// # Errors
///
/// Returns [`CodecError::UnexpectedFrame`] for any other frame kind, and
/// the payload errors of [`Self::read_evaluate_payload`].
pub fn read_evaluate<R: Read>(reader: &mut R) -> Result<(Vec<EvaluateEntry>, u64), CodecError> {
    let (mut frame, payload_len) = open_frame(reader, FrameKind::Evaluate)?;
    let message = read_evaluate_payload(&mut frame)?;
    frame.finish()?;
    Ok((message, payload_len))
}

/// Reads a whole `Products` frame.
///
/// Returns the parsed message and the frame's payload length.
///
/// # Errors
///
/// Returns [`CodecError::UnexpectedFrame`] for any other frame kind, and
/// the payload errors of [`Self::read_products_payload`].
pub fn read_products<R: Read>(reader: &mut R) -> Result<(Vec<ProductEntry>, u64), CodecError> {
    let (mut frame, payload_len) = open_frame(reader, FrameKind::Products)?;
    let message = read_products_payload(&mut frame)?;
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
/// the payload errors of [`Self::read_error_payload`].
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
