//! Shared view fixtures and full-frame pipeline helpers for the wire
//! codec test binaries.
//!
//! The fixtures hold plain owned field-element vectors and build the v2
//! borrowed views and encode inputs on demand, so nothing here depends on
//! an owned wire-record type. The `read_*_frame` helpers run the
//! plan → reserve → decode pipeline exactly as a whole-frame consumer
//! would — header kind check, plan, fresh workspace reservation, decode,
//! and the exact-consumption check — and return the populated workspace
//! plus the frame's declared payload length.
//!
//! The encode helpers leak one borrowed-ref slice per entry; tests build
//! small finite fixtures, so the leak keeps the borrowed inputs valid for
//! the whole run without self-referential plumbing.

use std::io::Cursor;

use emvp::{AnswerRef, EmvpParams, EncryptedQueryRef};

use emvp_network::{
    CodecError, EvaluateEntryInput, EvaluateWorkspace, Field, FrameKind, FrameReader,
    PROTOCOL_MODULUS, ProductEntryInput, ProductsWorkspace, UploadMatrixView, UploadWorkspace,
    decode_evaluate, decode_products, decode_upload, plan_evaluate, plan_products, plan_upload,
    read_frame_header, write_evaluate, write_products, write_upload_matrices,
};

/// One uploaded encrypted matrix as owned plain data.
#[derive(Debug)]
pub struct UploadFixture {
    /// The protocol parameters the matrix was encrypted under.
    pub params: EmvpParams,
    /// The public matrix-instance identifier.
    pub instance_id: u128,
    /// The encrypted matrix rows.
    pub rows: usize,
    /// The encrypted matrix columns (`n = 2k`).
    pub columns: usize,
    /// The row-major encrypted entries.
    pub values: Vec<Field>,
}

impl UploadFixture {
    /// The borrowed view the v2 encoder consumes.
    pub fn view(&self) -> UploadMatrixView<'_> {
        UploadMatrixView {
            params: self.params,
            instance_id: self.instance_id,
            rows: self.rows,
            columns: self.columns,
            values: &self.values,
        }
    }
}

/// One encrypted query as owned plain data.
#[derive(Debug)]
pub struct QueryFixture {
    /// The public matrix-instance identifier.
    pub instance_id: u128,
    /// The public query identifier.
    pub query_id: u64,
    /// The encrypted query coordinates.
    pub values: Vec<Field>,
}

impl QueryFixture {
    /// The borrowed query ref the v2 encoder consumes.
    pub fn query_ref(&self) -> EncryptedQueryRef<'_, PROTOCOL_MODULUS> {
        EncryptedQueryRef::new(self.instance_id, self.query_id, &self.values).unwrap()
    }
}

/// One evaluation entry as owned plain data: a matrix identifier plus its
/// queries, which share one coordinate count.
#[derive(Debug)]
pub struct EvaluateFixture {
    /// The server identifier of the targeted matrix.
    pub matrix_id: u64,
    /// The encrypted queries, in order.
    pub queries: Vec<QueryFixture>,
}

/// One encrypted answer as owned plain data.
#[derive(Debug)]
pub struct AnswerFixture {
    /// The public matrix-instance identifier.
    pub instance_id: u128,
    /// The public query identifier.
    pub query_id: u64,
    /// The answer row count.
    pub rows: usize,
    /// The answer block count.
    pub blocks: usize,
    /// The row-major answer entries.
    pub values: Vec<Field>,
}

impl AnswerFixture {
    /// The borrowed answer ref the v2 encoder consumes.
    pub fn answer_ref(&self) -> AnswerRef<'_, PROTOCOL_MODULUS> {
        AnswerRef::new(
            self.instance_id,
            self.query_id,
            &self.values,
            self.rows,
            self.blocks,
        )
        .unwrap()
    }
}

/// One products entry as owned plain data: a matrix identifier plus its
/// answers, which share instance identifier and shape.
#[derive(Debug)]
pub struct ProductFixture {
    /// The server identifier of the answered matrix.
    pub matrix_id: u64,
    /// The encrypted answers, in the entry's query order.
    pub answers: Vec<AnswerFixture>,
}

/// Encodes one upload frame from the fixture views.
pub fn encode_upload(uploads: &[UploadFixture]) -> Vec<u8> {
    let views: Vec<UploadMatrixView<'_>> = uploads.iter().map(UploadFixture::view).collect();
    let mut buffer = Vec::new();
    write_upload_matrices(&mut buffer, &views).unwrap();
    buffer
}

/// The single-entry encode inputs over `entry`.
pub fn evaluate_inputs(entry: &EvaluateFixture) -> [EvaluateEntryInput<'_>; 1] {
    let refs: Vec<EncryptedQueryRef<'_, PROTOCOL_MODULUS>> =
        entry.queries.iter().map(QueryFixture::query_ref).collect();
    [EvaluateEntryInput {
        matrix_id: entry.matrix_id,
        queries: refs.leak(),
    }]
}

/// Encodes one evaluate frame holding all `entries`.
pub fn encode_evaluate(entries: &[EvaluateFixture]) -> Vec<u8> {
    let inputs: Vec<EvaluateEntryInput<'_>> = entries.iter().flat_map(evaluate_inputs).collect();
    let mut buffer = Vec::new();
    write_evaluate(&mut buffer, &inputs).unwrap();
    buffer
}

/// The single-entry encode inputs over `entry`.
pub fn product_inputs(entry: &ProductFixture) -> [ProductEntryInput<'_>; 1] {
    let refs: Vec<AnswerRef<'_, PROTOCOL_MODULUS>> = entry
        .answers
        .iter()
        .map(AnswerFixture::answer_ref)
        .collect();
    let first = entry.answers.first();
    [ProductEntryInput {
        matrix_id: entry.matrix_id,
        instance_id: first.map_or(0, |answer| answer.instance_id),
        rows: first.map_or(0, |answer| answer.rows),
        blocks: first.map_or(0, |answer| answer.blocks),
        answers: refs.leak(),
    }]
}

/// Encodes one products frame holding all `entries`.
pub fn encode_products(entries: &[ProductFixture]) -> Vec<u8> {
    let inputs: Vec<ProductEntryInput<'_>> = entries.iter().flat_map(product_inputs).collect();
    let mut buffer = Vec::new();
    write_products(&mut buffer, &inputs).unwrap();
    buffer
}

/// Reads a frame header and opens its payload, requiring one kind.
pub fn open_frame<R: std::io::Read>(
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
    let payload_len = header.payload_len;
    Ok((FrameReader::new(reader, payload_len), payload_len))
}

/// Runs the whole upload pipeline over `bytes` into a fresh workspace.
pub fn read_upload_frame(bytes: &[u8]) -> Result<(UploadWorkspace, u64), CodecError> {
    let mut cursor = Cursor::new(bytes);
    let (mut frame, payload_len) = open_frame(&mut cursor, FrameKind::UploadMatrices)?;
    let plan = plan_upload(&mut frame)?;
    let mut workspace = UploadWorkspace::new();
    workspace.reserve(&plan)?;
    decode_upload(&mut frame, &plan, &mut workspace)?;
    Ok((workspace, payload_len))
}

/// Runs the whole evaluate pipeline over `bytes` into a fresh workspace.
pub fn read_evaluate_frame(bytes: &[u8]) -> Result<(EvaluateWorkspace, u64), CodecError> {
    let mut cursor = Cursor::new(bytes);
    let (mut frame, payload_len) = open_frame(&mut cursor, FrameKind::Evaluate)?;
    let plan = plan_evaluate(&mut frame)?;
    let mut workspace = EvaluateWorkspace::new();
    workspace.reserve(&plan)?;
    decode_evaluate(&mut frame, &plan, &mut workspace)?;
    Ok((workspace, payload_len))
}

/// Runs the whole products pipeline over `bytes` into a fresh workspace.
pub fn read_products_frame(bytes: &[u8]) -> Result<(ProductsWorkspace, u64), CodecError> {
    let mut cursor = Cursor::new(bytes);
    let (mut frame, payload_len) = open_frame(&mut cursor, FrameKind::Products)?;
    let plan = plan_products(&mut frame)?;
    let mut workspace = ProductsWorkspace::new();
    workspace.reserve(&plan)?;
    decode_products(&mut frame, &plan, &mut workspace)?;
    Ok((workspace, payload_len))
}
