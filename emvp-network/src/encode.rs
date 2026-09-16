//! Protocol v2 writers over borrowed views.
//!
//! Every writer first computes the exact payload length from its borrowed
//! inputs, rejecting shape inconsistencies before any byte is written,
//! then streams the frame through the existing stack-chunk machinery. No
//! writer allocates: descriptors travel first, the contiguous value
//! region second, both straight from the borrowed slices.

use std::io::Write;

use crate::error::CodecError;
use crate::frame::{
    FrameKind, HEADER_BYTES, checked_size, field_byte_len, wire_len, write_field_slice,
    write_frame_header, write_u32, write_u64, write_u128,
};
use crate::plan::{
    EVALUATE_ENTRY_DESCRIPTOR_BYTES, EVALUATE_QUERY_DESCRIPTOR_BYTES, EVALUATE_SUMMARY_BYTES,
    PRODUCTS_ANSWER_DESCRIPTOR_BYTES, PRODUCTS_ENTRY_DESCRIPTOR_BYTES, PRODUCTS_SUMMARY_BYTES,
    UPLOAD_DESCRIPTOR_BYTES, UPLOAD_SUMMARY_BYTES,
};
use crate::views::{EvaluateEntryInput, ProductEntryInput, UploadMatrixView};

/// Writes the one matrix-set upload of a session from borrowed views.
///
/// Returns the total bytes written, header included.
///
/// # Errors
///
/// Returns [`CodecError::DimensionOverflow`] if the message exceeds the
/// wire length space, and [`CodecError::Io`] if the transport fails.
pub fn write_upload_matrices(
    writer: &mut impl Write,
    uploads: &[UploadMatrixView<'_>],
) -> Result<u64, CodecError> {
    let mut payload_len = UPLOAD_SUMMARY_BYTES;
    for upload in uploads {
        payload_len = checked_size(payload_len, UPLOAD_DESCRIPTOR_BYTES)?;
        payload_len = checked_size(payload_len, field_byte_len(wire_len(upload.values.len())?)?)?;
    }
    write_frame_header(writer, FrameKind::UploadMatrices, payload_len)?;
    write_u64(writer, wire_len(uploads.len())?)?;
    for upload in uploads {
        write_u64(writer, wire_len(upload.params.k)?)?;
        write_u64(writer, wire_len(upload.params.ell)?)?;
        write_u64(writer, wire_len(upload.params.b)?)?;
        write_u32(writer, upload.params.lambda)?;
        write_u128(writer, upload.instance_id)?;
        write_u64(writer, wire_len(upload.rows)?)?;
        write_u64(writer, wire_len(upload.columns)?)?;
    }
    for upload in uploads {
        write_field_slice(writer, upload.values)?;
    }
    checked_size(payload_len, HEADER_BYTES)
}

/// Writes one ordered evaluation request from borrowed entry inputs.
///
/// The entry's wire `query_width` is derived from its first query's
/// coordinate count; every query of an entry must share that width.
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
    entries: &[EvaluateEntryInput<'_>],
) -> Result<u64, CodecError> {
    let mut total_query_count = 0_u64;
    let mut value_fields = 0_u64;
    for entry in entries {
        let mut width = None;
        for query in entry.queries {
            let length = wire_len(query.values().len())?;
            match width {
                Some(expected) if expected != length => {
                    return Err(CodecError::CountMismatch {
                        name: "query width",
                        expected,
                        actual: length,
                    });
                }
                Some(_seen) => {}
                None => width = Some(length),
            }
            total_query_count = checked_size(total_query_count, 1)?;
            value_fields = checked_size(value_fields, length)?;
        }
    }
    let entry_table_bytes = wire_len(entries.len())?
        .checked_mul(EVALUATE_ENTRY_DESCRIPTOR_BYTES)
        .ok_or(CodecError::DimensionOverflow)?;
    let query_table_bytes = total_query_count
        .checked_mul(EVALUATE_QUERY_DESCRIPTOR_BYTES)
        .ok_or(CodecError::DimensionOverflow)?;
    let mut payload_len = EVALUATE_SUMMARY_BYTES;
    payload_len = checked_size(payload_len, entry_table_bytes)?;
    payload_len = checked_size(payload_len, query_table_bytes)?;
    payload_len = checked_size(payload_len, field_byte_len(value_fields)?)?;
    write_frame_header(writer, FrameKind::Evaluate, payload_len)?;
    write_u64(writer, wire_len(entries.len())?)?;
    write_u64(writer, total_query_count)?;
    for entry in entries {
        let width = entry
            .queries
            .first()
            .map_or(Ok(0), |query| wire_len(query.values().len()))?;
        write_u64(writer, entry.matrix_id)?;
        write_u64(writer, wire_len(entry.queries.len())?)?;
        write_u64(writer, width)?;
    }
    for entry in entries {
        for query in entry.queries {
            write_u128(writer, query.instance_id())?;
            write_u64(writer, query.query_id())?;
        }
    }
    for entry in entries {
        for query in entry.queries {
            write_field_slice(writer, query.values())?;
        }
    }
    checked_size(payload_len, HEADER_BYTES)
}

/// Writes the encrypted products of one evaluation request from borrowed
/// entry inputs.
///
/// Each entry's answers must all hold exactly `rows * blocks` elements;
/// the entry descriptor's instance identifier comes from
/// [`ProductEntryInput::instance_id`] and each answer descriptor's query
/// identifier from the borrowed ref.
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
    entries: &[ProductEntryInput<'_>],
) -> Result<u64, CodecError> {
    let mut total_answer_count = 0_u64;
    let mut value_fields = 0_u64;
    for entry in entries {
        let answer_words = entry
            .rows
            .checked_mul(entry.blocks)
            .ok_or(CodecError::DimensionOverflow)?;
        let answer_words = wire_len(answer_words)?;
        for answer in entry.answers {
            let length = wire_len(answer.values().len())?;
            if length != answer_words {
                return Err(CodecError::CountMismatch {
                    name: "answer values",
                    expected: answer_words,
                    actual: length,
                });
            }
            total_answer_count = checked_size(total_answer_count, 1)?;
            value_fields = checked_size(value_fields, answer_words)?;
        }
    }
    let entry_table_bytes = wire_len(entries.len())?
        .checked_mul(PRODUCTS_ENTRY_DESCRIPTOR_BYTES)
        .ok_or(CodecError::DimensionOverflow)?;
    let answer_table_bytes = total_answer_count
        .checked_mul(PRODUCTS_ANSWER_DESCRIPTOR_BYTES)
        .ok_or(CodecError::DimensionOverflow)?;
    let mut payload_len = PRODUCTS_SUMMARY_BYTES;
    payload_len = checked_size(payload_len, entry_table_bytes)?;
    payload_len = checked_size(payload_len, answer_table_bytes)?;
    payload_len = checked_size(payload_len, field_byte_len(value_fields)?)?;
    write_frame_header(writer, FrameKind::Products, payload_len)?;
    write_u64(writer, wire_len(entries.len())?)?;
    write_u64(writer, total_answer_count)?;
    for entry in entries {
        write_u64(writer, entry.matrix_id)?;
        write_u128(writer, entry.instance_id)?;
        write_u64(writer, wire_len(entry.rows)?)?;
        write_u64(writer, wire_len(entry.blocks)?)?;
        write_u64(writer, wire_len(entry.answers.len())?)?;
    }
    for entry in entries {
        for answer in entry.answers {
            write_u64(writer, answer.query_id())?;
        }
    }
    for entry in entries {
        for answer in entry.answers {
            write_field_slice(writer, answer.values())?;
        }
    }
    checked_size(payload_len, HEADER_BYTES)
}
