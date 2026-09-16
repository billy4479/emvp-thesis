//! Wire message plans: descriptor-table parsing with every reservation
//! bound derived from the declared payload length.
//!
//! A [`plan_upload`], [`plan_upload_accepted`], [`plan_evaluate`], or
//! [`plan_products`] call reads the fixed summary and the descriptor table
//! of one payload and returns an owned plan: scalar metadata plus the
//! exact number of field elements the value region holds. No step
//! allocates before the descriptor counts have been shown to fit the
//! frame, and the exact byte budget of the whole payload is verified
//! before the plan is returned, so a decode that trusts the plan consumes
//! the value region to the byte.

use std::cmp::Ordering;
use std::io::Read;

use emvp::EmvpParams;

use crate::error::CodecError;
use crate::frame::{FrameReader, checked_size, field_byte_len, host_len};

/// Payload bytes of the `UploadMatrices` list count.
pub const UPLOAD_SUMMARY_BYTES: u64 = 8;

/// Bytes in one upload matrix descriptor: `k`, `ell`, `b` (`u64` each),
/// `lambda` (`u32`), the instance identifier (`u128`), and the two
/// dimensions (`u64` each).
pub const UPLOAD_DESCRIPTOR_BYTES: u64 = 8 + 8 + 8 + 4 + 16 + 8 + 8;

/// Bytes in one upload-accepted identifier descriptor.
pub const ACCEPTED_ID_DESCRIPTOR_BYTES: u64 = 8;

/// Payload bytes of the `Evaluate` summary: the entry count and the total
/// query count.
pub const EVALUATE_SUMMARY_BYTES: u64 = 8 + 8;

/// Bytes in one evaluate entry descriptor: matrix identifier, query count,
/// and query width (`u64` each).
pub const EVALUATE_ENTRY_DESCRIPTOR_BYTES: u64 = 8 + 8 + 8;

/// Bytes in one evaluate query descriptor: the instance identifier
/// (`u128`) and the query identifier (`u64`).
pub const EVALUATE_QUERY_DESCRIPTOR_BYTES: u64 = 16 + 8;

/// Payload bytes of the `Products` summary: the entry count and the total
/// answer count.
pub const PRODUCTS_SUMMARY_BYTES: u64 = 8 + 8;

/// Bytes in one products entry descriptor: matrix identifier (`u64`),
/// instance identifier (`u128`), rows, blocks, and answer count (`u64`
/// each).
pub const PRODUCTS_ENTRY_DESCRIPTOR_BYTES: u64 = 8 + 16 + 8 + 8 + 8;

/// Bytes in one products answer descriptor (the query identifier).
pub const PRODUCTS_ANSWER_DESCRIPTOR_BYTES: u64 = 8;

/// Validated descriptor metadata of one uploaded matrix.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UploadMatrixMeta {
    /// The protocol parameters the matrix was encrypted under.
    pub params: EmvpParams,
    /// The public matrix-instance identifier.
    pub instance_id: u128,
    /// The encrypted matrix rows.
    pub rows: usize,
    /// The encrypted matrix columns (`n = 2k`).
    pub columns: usize,
    /// The index of the matrix's first element in the workspace arena.
    pub value_offset: usize,
}

/// The validated descriptor table of one `UploadMatrices` payload.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct UploadPlan {
    /// One metadata record per uploaded matrix, in wire order.
    pub matrices: Vec<UploadMatrixMeta>,
    /// The total number of field elements in the value region.
    pub value_fields: usize,
}

/// The validated descriptor table of one `UploadAccepted` payload: the
/// identifier list itself, which doubles as the message's descriptor
/// table.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct UploadAcceptedPlan {
    /// The accepted matrix identifiers, in upload order.
    pub identifiers: Vec<u64>,
}

/// Validated descriptor metadata of one evaluate entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EvaluateEntryMeta {
    /// The server identifier of the targeted matrix.
    pub matrix_id: u64,
    /// The index of the entry's first query descriptor.
    pub query_offset: usize,
    /// The number of encrypted queries in the entry.
    pub query_count: usize,
    /// The coordinate count shared by every query of the entry.
    pub query_width: usize,
    /// The index of the entry's first element in the workspace arena.
    pub value_offset: usize,
}

/// Validated descriptor metadata of one encrypted query.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EvaluateQueryMeta {
    /// The public matrix-instance identifier.
    pub instance_id: u128,
    /// The public query identifier.
    pub query_id: u64,
}

/// The validated descriptor tables of one `Evaluate` payload.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct EvaluatePlan {
    /// One metadata record per entry, in wire order.
    pub entries: Vec<EvaluateEntryMeta>,
    /// All query descriptors, in entry order then query order.
    pub queries: Vec<EvaluateQueryMeta>,
    /// The total number of field elements in the value region.
    pub value_fields: usize,
}

/// Validated descriptor metadata of one products entry.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProductEntryMeta {
    /// The server identifier of the answered matrix.
    pub matrix_id: u64,
    /// The public matrix-instance identifier shared by every answer.
    pub instance_id: u128,
    /// The row count of every answer of the entry.
    pub rows: usize,
    /// The block count of every answer of the entry.
    pub blocks: usize,
    /// The index of the entry's first answer descriptor.
    pub answer_offset: usize,
    /// The number of answers in the entry.
    pub answer_count: usize,
    /// The index of the entry's first element in the workspace arena.
    pub value_offset: usize,
}

/// The validated descriptor tables of one `Products` payload.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ProductsPlan {
    /// One metadata record per entry, in wire order.
    pub entries: Vec<ProductEntryMeta>,
    /// The query identifier of every answer, in entry order then answer
    /// order.
    pub query_ids: Vec<u64>,
    /// The total number of field elements in the value region.
    pub value_fields: usize,
}

/// Plans the matrix-set upload payload: reads the list count and the
/// descriptor table, validates the dimensions, and verifies the exact
/// byte budget of the whole payload.
///
/// # Errors
///
/// Returns [`CodecError::TruncatedFrame`] when the descriptor table or the
/// value region cannot fit the declared payload, [`CodecError::AllocationFailed`]
/// when the descriptor metadata cannot be reserved,
/// [`CodecError::InvalidDimensions`] for zero rows or columns,
/// [`CodecError::DimensionOverflow`] when checked arithmetic overflows,
/// [`CodecError::ValueOutOfRange`] when a wire dimension does not fit the
/// host, and the framing errors of [`FrameReader`].
pub fn plan_upload<R: Read>(frame: &mut FrameReader<'_, R>) -> Result<UploadPlan, CodecError> {
    let count = frame.read_u64()?;
    let descriptor_bytes = count
        .checked_mul(UPLOAD_DESCRIPTOR_BYTES)
        .ok_or(CodecError::DimensionOverflow)?;
    if descriptor_bytes > frame.remaining() {
        return Err(CodecError::TruncatedFrame);
    }
    let count = host_len("matrix count", count)?;
    let mut matrices = Vec::new();
    matrices
        .try_reserve_exact(count)
        .map_err(|_reserve| CodecError::AllocationFailed)?;
    let mut value_fields = 0_usize;
    for _ in 0..count {
        let k = frame.read_u64()?;
        let ell = frame.read_u64()?;
        let b = frame.read_u64()?;
        let lambda = frame.read_u32()?;
        let instance_id = frame.read_u128()?;
        let rows = frame.read_u64()?;
        let columns = frame.read_u64()?;
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
        let rows = host_len("matrix rows", rows)?;
        let columns = host_len("matrix columns", columns)?;
        let fields = rows
            .checked_mul(columns)
            .ok_or(CodecError::DimensionOverflow)?;
        let value_offset = value_fields;
        value_fields = value_fields
            .checked_add(fields)
            .ok_or(CodecError::DimensionOverflow)?;
        matrices.push(UploadMatrixMeta {
            params: EmvpParams {
                k: host_len("parameter k", k)?,
                ell: host_len("parameter ell", ell)?,
                b: host_len("parameter b", b)?,
                lambda,
            },
            instance_id,
            rows,
            columns,
            value_offset,
        });
    }
    require_exact_value_region(frame, value_fields)?;
    Ok(UploadPlan {
        matrices,
        value_fields,
    })
}

/// Plans the accepted-upload payload: the identifier list is the message's
/// descriptor table, so it is read and validated here.
///
/// # Errors
///
/// Returns [`CodecError::TruncatedFrame`] or
/// [`CodecError::TrailingFrameBytes`] when the identifier list does not
/// fill the payload exactly, [`CodecError::AllocationFailed`] when the
/// identifier metadata cannot be reserved,
/// [`CodecError::InvalidDimensions`] for a zero identifier, and the
/// framing errors of [`FrameReader`].
pub fn plan_upload_accepted<R: Read>(
    frame: &mut FrameReader<'_, R>,
) -> Result<UploadAcceptedPlan, CodecError> {
    let count = frame.read_u64()?;
    let identifier_bytes = count
        .checked_mul(ACCEPTED_ID_DESCRIPTOR_BYTES)
        .ok_or(CodecError::DimensionOverflow)?;
    match identifier_bytes.cmp(&frame.remaining()) {
        Ordering::Less => {
            return Err(CodecError::TrailingFrameBytes {
                extra: frame.remaining() - identifier_bytes,
            });
        }
        Ordering::Greater => return Err(CodecError::TruncatedFrame),
        Ordering::Equal => {}
    }
    let count = host_len("identifier count", count)?;
    let mut identifiers = Vec::new();
    identifiers
        .try_reserve_exact(count)
        .map_err(|_reserve| CodecError::AllocationFailed)?;
    for _ in 0..count {
        let identifier = frame.read_u64()?;
        if identifier == 0 {
            return Err(CodecError::InvalidDimensions {
                name: "matrix identifier",
                value: 0,
            });
        }
        identifiers.push(identifier);
    }
    Ok(UploadAcceptedPlan { identifiers })
}

/// Plans the evaluation payload: reads the summary, both descriptor
/// tables, cross-checks the per-entry query counts against the summary
/// total, and verifies the exact byte budget of the whole payload.
///
/// # Errors
///
/// Returns [`CodecError::TruncatedFrame`] when a descriptor table or the
/// value region cannot fit the declared payload,
/// [`CodecError::TrailingFrameBytes`] when the value region would leave
/// payload bytes unconsumed, [`CodecError::AllocationFailed`] when the
/// descriptor metadata cannot be reserved, [`CodecError::CountMismatch`]
/// when the per-entry query counts contradict the summary total or an
/// empty entry declares a positive width,
/// [`CodecError::InvalidDimensions`] for a positive-width zero,
/// [`CodecError::DimensionOverflow`] when checked arithmetic overflows,
/// [`CodecError::ValueOutOfRange`] when a wire count does not fit the
/// host, and the framing errors of [`FrameReader`].
pub fn plan_evaluate<R: Read>(frame: &mut FrameReader<'_, R>) -> Result<EvaluatePlan, CodecError> {
    let entry_count = frame.read_u64()?;
    let total_query_count = frame.read_u64()?;
    require_descriptor_fit(
        frame,
        entry_count,
        EVALUATE_ENTRY_DESCRIPTOR_BYTES,
        total_query_count,
        EVALUATE_QUERY_DESCRIPTOR_BYTES,
    )?;
    let entry_count = host_len("entry count", entry_count)?;
    let total_query_count = host_len("total query count", total_query_count)?;
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(entry_count)
        .map_err(|_reserve| CodecError::AllocationFailed)?;
    let mut queries = Vec::new();
    queries
        .try_reserve_exact(total_query_count)
        .map_err(|_reserve| CodecError::AllocationFailed)?;
    let mut summed_query_count = 0_usize;
    let mut value_fields = 0_usize;
    for _ in 0..entry_count {
        let matrix_id = frame.read_u64()?;
        let query_count = frame.read_u64()?;
        let query_width = frame.read_u64()?;
        if query_count == 0 {
            if query_width != 0 {
                return Err(CodecError::CountMismatch {
                    name: "query width",
                    expected: 0,
                    actual: query_width,
                });
            }
        } else if query_width == 0 {
            return Err(CodecError::InvalidDimensions {
                name: "query width",
                value: 0,
            });
        }
        let query_count = host_len("query count", query_count)?;
        let query_width = host_len("query width", query_width)?;
        let query_offset = summed_query_count;
        summed_query_count = summed_query_count
            .checked_add(query_count)
            .ok_or(CodecError::DimensionOverflow)?;
        let entry_fields = query_count
            .checked_mul(query_width)
            .ok_or(CodecError::DimensionOverflow)?;
        let value_offset = value_fields;
        value_fields = value_fields
            .checked_add(entry_fields)
            .ok_or(CodecError::DimensionOverflow)?;
        entries.push(EvaluateEntryMeta {
            matrix_id,
            query_offset,
            query_count,
            query_width,
            value_offset,
        });
    }
    if summed_query_count != total_query_count {
        return Err(CodecError::CountMismatch {
            name: "query count",
            expected: u64::try_from(total_query_count)
                .map_err(|_conversion| CodecError::DimensionOverflow)?,
            actual: u64::try_from(summed_query_count)
                .map_err(|_conversion| CodecError::DimensionOverflow)?,
        });
    }
    for _ in 0..total_query_count {
        let instance_id = frame.read_u128()?;
        let query_id = frame.read_u64()?;
        queries.push(EvaluateQueryMeta {
            instance_id,
            query_id,
        });
    }
    require_exact_value_region(frame, value_fields)?;
    Ok(EvaluatePlan {
        entries,
        queries,
        value_fields,
    })
}

/// Plans the products payload: reads the summary, both descriptor tables,
/// cross-checks the per-entry answer counts against the summary total, and
/// verifies the exact byte budget of the whole payload.
///
/// # Errors
///
/// Returns [`CodecError::TruncatedFrame`] when a descriptor table or the
/// value region cannot fit the declared payload,
/// [`CodecError::TrailingFrameBytes`] when the value region would leave
/// payload bytes unconsumed, [`CodecError::AllocationFailed`] when the
/// descriptor metadata cannot be reserved, [`CodecError::CountMismatch`]
/// when the per-entry answer counts contradict the summary total,
/// [`CodecError::InvalidDimensions`] for zero rows or blocks,
/// [`CodecError::DimensionOverflow`] when checked arithmetic overflows,
/// [`CodecError::ValueOutOfRange`] when a wire count does not fit the
/// host, and the framing errors of [`FrameReader`].
pub fn plan_products<R: Read>(frame: &mut FrameReader<'_, R>) -> Result<ProductsPlan, CodecError> {
    let entry_count = frame.read_u64()?;
    let total_answer_count = frame.read_u64()?;
    require_descriptor_fit(
        frame,
        entry_count,
        PRODUCTS_ENTRY_DESCRIPTOR_BYTES,
        total_answer_count,
        PRODUCTS_ANSWER_DESCRIPTOR_BYTES,
    )?;
    let entry_count = host_len("entry count", entry_count)?;
    let total_answer_count = host_len("total answer count", total_answer_count)?;
    let mut entries = Vec::new();
    entries
        .try_reserve_exact(entry_count)
        .map_err(|_reserve| CodecError::AllocationFailed)?;
    let mut query_ids = Vec::new();
    query_ids
        .try_reserve_exact(total_answer_count)
        .map_err(|_reserve| CodecError::AllocationFailed)?;
    let mut summed_answer_count = 0_usize;
    let mut value_fields = 0_usize;
    for _ in 0..entry_count {
        let matrix_id = frame.read_u64()?;
        let instance_id = frame.read_u128()?;
        let rows = frame.read_u64()?;
        let blocks = frame.read_u64()?;
        let answer_count = frame.read_u64()?;
        if rows == 0 {
            return Err(CodecError::InvalidDimensions {
                name: "answer rows",
                value: 0,
            });
        }
        if blocks == 0 {
            return Err(CodecError::InvalidDimensions {
                name: "answer blocks",
                value: 0,
            });
        }
        let rows = host_len("answer rows", rows)?;
        let blocks = host_len("answer blocks", blocks)?;
        let answer_count = host_len("answer count", answer_count)?;
        let answer_words = rows
            .checked_mul(blocks)
            .ok_or(CodecError::DimensionOverflow)?;
        let answer_offset = summed_answer_count;
        summed_answer_count = summed_answer_count
            .checked_add(answer_count)
            .ok_or(CodecError::DimensionOverflow)?;
        let entry_fields = answer_count
            .checked_mul(answer_words)
            .ok_or(CodecError::DimensionOverflow)?;
        let value_offset = value_fields;
        value_fields = value_fields
            .checked_add(entry_fields)
            .ok_or(CodecError::DimensionOverflow)?;
        entries.push(ProductEntryMeta {
            matrix_id,
            instance_id,
            rows,
            blocks,
            answer_offset,
            answer_count,
            value_offset,
        });
    }
    if summed_answer_count != total_answer_count {
        return Err(CodecError::CountMismatch {
            name: "answer count",
            expected: u64::try_from(total_answer_count)
                .map_err(|_conversion| CodecError::DimensionOverflow)?,
            actual: u64::try_from(summed_answer_count)
                .map_err(|_conversion| CodecError::DimensionOverflow)?,
        });
    }
    for _ in 0..total_answer_count {
        let query_id = frame.read_u64()?;
        query_ids.push(query_id);
    }
    require_exact_value_region(frame, value_fields)?;
    Ok(ProductsPlan {
        entries,
        query_ids,
        value_fields,
    })
}

/// Rejects two descriptor-table counts whose tables cannot both fit the
/// payload bytes left after the summary, before any allocation.
fn require_descriptor_fit<R: Read>(
    frame: &FrameReader<'_, R>,
    primary_count: u64,
    primary_descriptor_bytes: u64,
    secondary_count: u64,
    secondary_descriptor_bytes: u64,
) -> Result<(), CodecError> {
    let primary_bytes = primary_count
        .checked_mul(primary_descriptor_bytes)
        .ok_or(CodecError::DimensionOverflow)?;
    let secondary_bytes = secondary_count
        .checked_mul(secondary_descriptor_bytes)
        .ok_or(CodecError::DimensionOverflow)?;
    let descriptor_bytes = checked_size(primary_bytes, secondary_bytes)?;
    if descriptor_bytes > frame.remaining() {
        return Err(CodecError::TruncatedFrame);
    }
    Ok(())
}

/// Requires the value region implied by `value_fields` to fill the payload
/// bytes exactly: a shortfall is trailing bytes, an excess a truncation.
fn require_exact_value_region<R: Read>(
    frame: &FrameReader<'_, R>,
    value_fields: usize,
) -> Result<(), CodecError> {
    let fields =
        u64::try_from(value_fields).map_err(|_conversion| CodecError::DimensionOverflow)?;
    let value_bytes = field_byte_len(fields)?;
    match value_bytes.cmp(&frame.remaining()) {
        Ordering::Less => Err(CodecError::TrailingFrameBytes {
            extra: frame.remaining() - value_bytes,
        }),
        Ordering::Greater => Err(CodecError::TruncatedFrame),
        Ordering::Equal => Ok(()),
    }
}
