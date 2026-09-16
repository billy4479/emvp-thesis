//! Protocol v2 wire codec: descriptor-table-first layouts decoded through a
//! plan → reserve → decode workspace pipeline.
//!
//! # Layouts
//!
//! Every post-handshake message keeps the frame envelope of
//! [`crate::frame`]: a one-byte [`crate::FrameKind`] and a little-endian `u64`
//! payload length. Inside the payload all counts and dimensions are
//! little-endian `u64`, instance identifiers are little-endian `u128`, and
//! field elements are the canonical four-byte little-endian encoding
//! (noncanonical encodings are rejected, never reduced).
//!
//! - `UploadMatrices`: `u64 matrix_count`; one 60-byte descriptor per
//!   matrix (`u64 k, u64 ell, u64 b, u32 lambda, u128 instance_id,
//!   u64 rows, u64 columns`); then the contiguous value region holding
//!   `rows * columns` elements per matrix in descriptor order. The v1
//!   redundant per-matrix value count is dropped: the value length is
//!   derived from the descriptor dimensions.
//! - `UploadAccepted`: unchanged (`u64 count`, then `count` nonzero
//!   `u64` identifiers).
//! - `Evaluate`: `u64 entry_count; u64 total_query_count`; one 24-byte
//!   entry descriptor per entry (`u64 matrix_id, u64 query_count,
//!   u64 query_width`); `total_query_count` 24-byte query descriptors
//!   (`u128 instance_id, u64 query_id`) in entry order then query order;
//!   then the value region where query `i` of entry `e` holds
//!   `query_width_e` elements.
//! - `Products`: `u64 entry_count; u64 total_answer_count`; one 48-byte
//!   entry descriptor per entry (`u64 matrix_id, u128 instance_id,
//!   u64 rows, u64 blocks, u64 answer_count`); `total_answer_count`
//!   8-byte answer descriptors (`u64 query_id`); then the value region
//!   where every answer of an entry holds `rows * blocks` elements.
//! - `Error`: unchanged.
//!
//! # Pipeline
//!
//! Decoding a message runs in three steps. [`plan_upload`],
//! [`plan_upload_accepted`], [`plan_evaluate`], and [`plan_products`]
//! read only the summary and the descriptor table, deriving every
//! reservation bound from the frame's declared payload length before any
//! allocation: counts that cannot fit the payload are rejected with
//! checked arithmetic, descriptor metadata is reserved with
//! [`std::collections::TryReserveError`]-free `try_reserve_exact`, and the
//! exact byte budget `descriptor_bytes + 4 * value_fields == payload_len`
//! is enforced. The caller then calls [`UploadWorkspace::reserve`],
//! [`EvaluateWorkspace::reserve`], [`ProductsWorkspace::reserve`], or
//! [`UploadAcceptedWorkspace::reserve`], the only step that grows memory.
//! Finally [`decode_upload`], [`decode_upload_accepted`],
//! [`decode_evaluate`], or [`decode_products`] streams the value region
//! into the preallocated workspace through
//! [`crate::FrameReader::read_field_slice_into`] and proves exact consumption
//! with [`crate::FrameReader::finish`], returning borrowed views over the
//! workspace. The decode and encode happy paths perform zero managed
//! allocations once the workspace is reserved.

mod decode;
mod encode;
mod plan;
mod views;
mod workspace;

pub use decode::{decode_evaluate, decode_products, decode_upload, decode_upload_accepted};
pub use encode::{write_evaluate, write_products, write_upload_matrices};
pub use plan::{
    EVALUATE_ENTRY_DESCRIPTOR_BYTES, EVALUATE_QUERY_DESCRIPTOR_BYTES, EVALUATE_SUMMARY_BYTES,
    PRODUCTS_ANSWER_DESCRIPTOR_BYTES, PRODUCTS_ENTRY_DESCRIPTOR_BYTES, PRODUCTS_SUMMARY_BYTES,
    UPLOAD_DESCRIPTOR_BYTES, UPLOAD_SUMMARY_BYTES, EvaluateEntryMeta, EvaluatePlan,
    EvaluateQueryMeta, ProductEntryMeta, ProductsPlan, UploadAcceptedPlan, UploadMatrixMeta,
    UploadPlan, plan_evaluate, plan_products, plan_upload, plan_upload_accepted,
};
pub use views::{
    EvaluateEntryInput, EvaluateEntryView, EvaluateQueryIter, EvaluateViews, ProductAnswerIter,
    ProductEntryInput, ProductEntryView, ProductsViews, UploadAcceptedIds, UploadMatrixView,
    UploadMatrixViewIter, UploadViews,
};
pub use workspace::{
    EvaluateWorkspace, ProductsWorkspace, UploadAcceptedWorkspace, UploadWorkspace,
};
