//! The protocol messages and their payload codecs.
//!
//! The owned message types ([`MatrixUpload`], [`EvaluateEntry`],
//! [`ProductEntry`], [`ErrorResponse`]) and the `read_*`/`write_*`
//! functions here are the stable convenience API: each one is a thin
//! adapter over the zero-copy pipeline in [`crate::v2`]. The writers
//! build borrowed views over their inputs and stream; the readers run the
//! plan → reserve → decode pipeline with a fresh workspace and convert
//! the decoded views into the owned types. Callers that manage their own
//! workspaces should use [`crate::v2`] directly and never allocate on the
//! hot path.
//!
//! Full-frame functions read or write the frame header plus the payload
//! and are what a role that only ever produces or consumes whole messages
//! uses. Payload-level functions parse from an already open
//! [`FrameReader`], which is how the server dispatches on the header kind
//! before handing the payload to the message parser.

use std::fmt;
use std::io::{Read, Write};

use emvp::{
    AnswerMatrix, AnswerRef, EmvpParams, EncryptedMatrix, EncryptedQuery, EncryptedQueryRef,
};

use crate::error::{CodecError, ErrorCode};
use crate::frame::{
    Field, FrameKind, FrameReader, HEADER_BYTES, PROTOCOL_MODULUS, checked_size, host_len,
    wire_len, write_frame_header, write_u32, write_u64,
};
use crate::v2::{
    EvaluateEntryInput, EvaluateEntryView, EvaluateWorkspace, ProductEntryInput, ProductEntryView,
    ProductsWorkspace, UploadAcceptedWorkspace, UploadMatrixView, UploadWorkspace, decode_evaluate,
    decode_products, decode_upload, decode_upload_accepted, plan_evaluate, plan_products,
    plan_upload, plan_upload_accepted, write_evaluate as write_evaluate_views,
    write_products as write_products_views, write_upload_matrices as write_upload_matrices_views,
};

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
    let views: Vec<UploadMatrixView<'_>> = uploads.iter().map(UploadMatrixView::from).collect();
    write_upload_matrices_views(writer, &views)
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

/// Writes one ordered evaluation request.
///
/// Returns the total bytes written, header included.
///
/// # Errors
///
/// Returns [`CodecError::CountMismatch`] when the queries of one entry
/// disagree on their coordinate count, [`CodecError::DimensionOverflow`]
/// if the message exceeds the wire length space, and [`CodecError::Io`]
/// if the transport fails.
pub fn write_evaluate(
    writer: &mut impl Write,
    entries: &[EvaluateEntry],
) -> Result<u64, CodecError> {
    let query_refs: Vec<Vec<EncryptedQueryRef<'_, PROTOCOL_MODULUS>>> = entries
        .iter()
        .map(|entry| entry.queries.iter().map(EncryptedQueryRef::from).collect())
        .collect();
    let inputs: Vec<EvaluateEntryInput<'_>> = entries
        .iter()
        .zip(&query_refs)
        .map(|(entry, queries)| EvaluateEntryInput {
            matrix_id: entry.matrix_id,
            queries,
        })
        .collect();
    write_evaluate_views(writer, &inputs)
}

/// Writes the encrypted products of one evaluation request.
///
/// Returns the total bytes written, header included.
///
/// # Errors
///
/// Returns [`CodecError::CountMismatch`] when an answer's element count
/// contradicts the entry shape, [`CodecError::DimensionOverflow`] if the
/// message exceeds the wire length space, and [`CodecError::Io`] if the
/// transport fails.
pub fn write_products(
    writer: &mut impl Write,
    entries: &[ProductEntry],
) -> Result<u64, CodecError> {
    let answer_refs: Vec<Vec<AnswerRef<'_, PROTOCOL_MODULUS>>> = entries
        .iter()
        .map(|entry| entry.answers.iter().map(AnswerRef::from).collect())
        .collect();
    let inputs: Vec<ProductEntryInput<'_>> = entries
        .iter()
        .zip(&answer_refs)
        .map(|(entry, answers)| ProductEntryInput {
            matrix_id: entry.matrix_id,
            instance_id: entry.answers.first().map_or(0, AnswerMatrix::instance_id),
            rows: entry.answers.first().map_or(0, AnswerMatrix::rows),
            blocks: entry.answers.first().map_or(0, AnswerMatrix::blocks),
            answers,
        })
        .collect();
    write_products_views(writer, &inputs)
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

/// Reads the matrix-set upload payload from an open frame into freshly
/// allocated owned records.
///
/// This is the allocating convenience form of the [`crate::v2`]
/// pipeline; it reserves a fresh [`UploadWorkspace`], decodes, and copies
/// the borrowed views into owned [`MatrixUpload`] records.
///
/// # Errors
///
/// Returns the errors of [`crate::v2::plan_upload`],
/// [`crate::v2::decode_upload`], and each record's owned conversion.
pub fn read_upload_matrices_payload<R: Read>(
    frame: &mut FrameReader<'_, R>,
) -> Result<Vec<MatrixUpload>, CodecError> {
    let plan = plan_upload(frame)?;
    let mut workspace = UploadWorkspace::new();
    workspace.reserve(&plan)?;
    let views = decode_upload(frame, &plan, &mut workspace)?;
    views.iter().map(owned_upload).collect()
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
    let plan = plan_upload_accepted(frame)?;
    let mut workspace = UploadAcceptedWorkspace::new();
    workspace.reserve(&plan)?;
    let identifiers = decode_upload_accepted(frame, &plan, &mut workspace)?;
    Ok(identifiers.to_vec())
}

/// Reads the evaluation payload from an open frame into freshly allocated
/// owned records.
///
/// This is the allocating convenience form of the [`crate::v2`]
/// pipeline; it reserves a fresh [`EvaluateWorkspace`], decodes, and
/// copies the borrowed views into owned [`EvaluateEntry`] records.
///
/// # Errors
///
/// Returns the errors of [`crate::v2::plan_evaluate`],
/// [`crate::v2::decode_evaluate`], and each record's owned conversion.
pub fn read_evaluate_payload<R: Read>(
    frame: &mut FrameReader<'_, R>,
) -> Result<Vec<EvaluateEntry>, CodecError> {
    let plan = plan_evaluate(frame)?;
    let mut workspace = EvaluateWorkspace::new();
    workspace.reserve(&plan)?;
    let views = decode_evaluate(frame, &plan, &mut workspace)?;
    Ok(views.iter().map(owned_entry).collect())
}

/// Reads the products payload from an open frame into freshly allocated
/// owned records.
///
/// This is the allocating convenience form of the [`crate::v2`]
/// pipeline; it reserves a fresh [`ProductsWorkspace`], decodes, and
/// copies the borrowed views into owned [`ProductEntry`] records.
///
/// # Errors
///
/// Returns the errors of [`crate::v2::plan_products`],
/// [`crate::v2::decode_products`], and each record's owned conversion.
pub fn read_products_payload<R: Read>(
    frame: &mut FrameReader<'_, R>,
) -> Result<Vec<ProductEntry>, CodecError> {
    let plan = plan_products(frame)?;
    let mut workspace = ProductsWorkspace::new();
    workspace.reserve(&plan)?;
    let views = decode_products(frame, &plan, &mut workspace)?;
    Ok(views.iter().map(owned_product).collect())
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
/// the payload errors of [`read_upload_matrices_payload`].
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
/// the payload errors of [`read_upload_accepted_payload`].
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
/// the payload errors of [`read_evaluate_payload`].
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
/// the payload errors of [`read_products_payload`].
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
/// the payload errors of [`read_error_payload`].
pub fn read_error<R: Read>(reader: &mut R) -> Result<(ErrorResponse, u64), CodecError> {
    let (mut frame, payload_len) = open_frame(reader, FrameKind::Error)?;
    let message = read_error_payload(&mut frame)?;
    frame.finish()?;
    Ok((message, payload_len))
}

/// Converts one decoded matrix view into an owned upload record.
fn owned_upload(view: UploadMatrixView<'_>) -> Result<MatrixUpload, CodecError> {
    let matrix = owned_matrix(view.values, view.instance_id, view.rows, view.columns)?;
    Ok(MatrixUpload {
        params: view.params,
        matrix,
    })
}

/// Converts one decoded entry view into an owned evaluation entry.
fn owned_entry(view: EvaluateEntryView<'_>) -> EvaluateEntry {
    let queries: Vec<EncryptedQuery<PROTOCOL_MODULUS>> = view
        .iter()
        .map(|query| {
            EncryptedQuery::from_parts(
                query.instance_id(),
                query.query_id(),
                query.values().to_vec(),
            )
        })
        .collect();
    EvaluateEntry {
        matrix_id: view.matrix_id(),
        queries,
    }
}

/// Converts one decoded product entry view into an owned product entry.
fn owned_product(view: ProductEntryView<'_>) -> ProductEntry {
    let rows = view.rows();
    let blocks = view.blocks();
    let answers: Vec<AnswerMatrix<PROTOCOL_MODULUS>> = view
        .iter()
        .map(|answer| {
            AnswerMatrix::from_parts(
                answer.instance_id(),
                answer.query_id(),
                answer.values().to_vec(),
                rows,
                blocks,
            )
        })
        .collect();
    ProductEntry {
        matrix_id: view.matrix_id(),
        answers,
    }
}

/// Copies a decoded matrix's values into an owned encrypted matrix.
///
/// A validated decode satisfies the shape invariant exactly, so the error
/// arm is unreachable for decoded records.
fn owned_matrix(
    values: &[Field],
    instance_id: u128,
    rows: usize,
    columns: usize,
) -> Result<EncryptedMatrix<PROTOCOL_MODULUS>, CodecError> {
    match EncryptedMatrix::from_parts(instance_id, rows, columns, values.to_vec()) {
        Ok(matrix) => Ok(matrix),
        Err(_unreachable) => {
            let expected = u64::try_from(
                rows.checked_mul(columns)
                    .ok_or(CodecError::DimensionOverflow)?,
            )
            .map_err(|_conversion| CodecError::DimensionOverflow)?;
            let actual =
                u64::try_from(values.len()).map_err(|_conversion| CodecError::DimensionOverflow)?;
            Err(CodecError::CountMismatch {
                name: "matrix values",
                expected,
                actual,
            })
        }
    }
}

/// Reads a frame header and opens its payload, requiring one kind.
fn open_frame<R: Read>(
    reader: &mut R,
    expected: FrameKind,
) -> Result<(FrameReader<'_, R>, u64), CodecError> {
    let Some(header) = crate::frame::read_frame_header(reader)? else {
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
