//! The client-server wire protocol for the EMVP encrypted matrix-vector
//! product service.
//!
//! This crate defines the only boundary the `client` and `server`
//! binaries share: a version and modulus handshake, a length-prefixed
//! frame format, and the five protocol messages that upload encrypted
//! matrices to a server and evaluate encrypted queries against them. The
//! cryptographic protocol itself lives in the [`emvp`] crate; the client
//! keeps every secret, and this crate transports only public artifacts.
//!
//! # Session
//!
//! 1. The client sends a fixed ten-byte hello (magic, protocol version,
//!    field modulus); the server answers with one accept or reject status
//!    byte and closes on rejection. See [`handshake`].
//! 2. The client sends one `UploadMatrices` frame; the server validates
//!    and answers `UploadAccepted` with one connection-local matrix
//!    identifier per uploaded matrix, in upload order.
//! 3. The client sends `Evaluate` frames: ordered entries of a matrix
//!    identifier and its ordered encrypted queries. The server answers
//!    `Products` with one answer per query, in the same order, or `Error`.
//!
//! The connection is strictly sequential: one request, one response.
//! A request that fails validation (malformed frame, rejected parameters,
//! unknown identifiers) is answered with an `Error` frame and leaves the
//! connection usable. State violations are fatal instead: a second upload
//! on a loaded session and an evaluation before the session loaded
//! anything are answered with an `Error` frame and close the connection
//! immediately, without the payload being read. Unknown frame kinds and
//! transport failures also close the connection.
//!
//! # Frame format
//!
//! Every post-handshake message is one frame: a [`FrameKind`] byte, a
//! little-endian `u64` payload length, and exactly that many payload
//! bytes. All multi-byte integers are little-endian; protocol dimensions
//! and counts travel as `u64` and are cross-checked before any element
//! payload is decoded. Field elements use the canonical four-byte
//! little-endian encoding of [`FieldElement::to_canonical_le_bytes`], and
//! decoding rejects integers at or above the modulus instead of reducing
//! them. Payloads are parsed through [`FrameReader`], which is bounded by
//! the declared frame length, so a truncated stream surfaces as an error
//! rather than a silently wrong message or an unbounded allocation.
//!
//! # Protocol v2 wire codec
//!
//! The message payloads are descriptor-table-first: each message carries a
//! fixed summary, a table of per-record descriptors, and one contiguous
//! value region, decoded through a plan → reserve → decode pipeline that
//! derives every reservation bound from the declared payload length before
//! allocating, grows a reusable workspace exactly once, and yields
//! borrowed views over that workspace with zero managed allocations on the
//! decode and encode happy paths. That pipeline is the crate's public
//! surface for the bulk messages, split across [`plan`], [`workspace`],
//! [`decode`], [`encode`], and [`views`]; only the error frame and the
//! `UploadAccepted` acknowledgment keep owned convenience codecs in
//! [`messages`].
//!
//! ## Layouts
//!
//! Every post-handshake message keeps the frame envelope of [`frame`]: a
//! one-byte [`FrameKind`] and a little-endian `u64` payload length. Inside
//! the payload all counts and dimensions are little-endian `u64`, instance
//! identifiers are little-endian `u128`, and field elements are the
//! canonical four-byte little-endian encoding (noncanonical encodings are
//! rejected, never reduced).
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
//! ## Pipeline
//!
//! Decoding a message runs in three steps. [`plan_upload`],
//! [`plan_upload_accepted`], [`plan_evaluate`], and [`plan_products`]
//! read only the summary and the descriptor table, deriving every
//! reservation bound from the frame's declared payload length before any
//! allocation: counts that cannot fit the payload are rejected with
//! checked arithmetic, descriptor metadata is reserved with
//! `try_reserve_exact`, and the exact byte budget
//! `descriptor_bytes + 4 * value_fields == payload_len` is enforced. The
//! caller then calls [`UploadWorkspace::reserve`],
//! [`EvaluateWorkspace::reserve`], [`ProductsWorkspace::reserve`], or
//! [`UploadAcceptedWorkspace::reserve`], the only step that grows memory.
//! Finally [`decode_upload`], [`decode_upload_accepted`],
//! [`decode_evaluate`], or [`decode_products`] streams the value region
//! into the preallocated workspace through
//! [`FrameReader::read_field_slice_into`] and proves exact consumption
//! with [`FrameReader::finish`], returning borrowed views over the
//! workspace.
//!
//! # Experimental
//!
//! This is research plumbing for a POC deployment. The protocol performs
//! no authentication or transport security: instance and query identifiers
//! guard against accidental artifact mixups only, never against a
//! malicious transport.
//!
//! [`FieldElement::to_canonical_le_bytes`]: prime_field_layer::FieldElement::to_canonical_le_bytes

pub mod decode;
pub mod encode;
pub mod error;
pub mod frame;
pub mod handshake;
pub mod messages;
pub mod plan;
pub mod views;
pub mod workspace;

pub use decode::{decode_evaluate, decode_products, decode_upload, decode_upload_accepted};
pub use encode::{write_evaluate, write_products, write_upload_matrices};
pub use error::{CodecError, ErrorCode, HandshakeError};
pub use frame::{
    Field, FrameHeader, FrameKind, FrameReader, HEADER_BYTES, MAGIC, PROTOCOL_MODULUS,
    PROTOCOL_VERSION, read_frame_header, write_field_slice, write_frame_header,
};
pub use handshake::{
    ClientHello, HELLO_BYTES, client_handshake, read_client_hello, read_server_hello,
    server_handshake, write_client_hello, write_server_hello,
};
pub use messages::{
    ErrorResponse, read_error, read_error_payload, read_upload_accepted,
    read_upload_accepted_payload, write_error, write_upload_accepted,
};
pub use plan::{
    EVALUATE_ENTRY_DESCRIPTOR_BYTES, EVALUATE_QUERY_DESCRIPTOR_BYTES, EVALUATE_SUMMARY_BYTES,
    EvaluateEntryMeta, EvaluatePlan, EvaluateQueryMeta, PRODUCTS_ANSWER_DESCRIPTOR_BYTES,
    PRODUCTS_ENTRY_DESCRIPTOR_BYTES, PRODUCTS_SUMMARY_BYTES, ProductsEntryMeta, ProductsPlan,
    UPLOAD_DESCRIPTOR_BYTES, UPLOAD_SUMMARY_BYTES, UploadAcceptedPlan, UploadMatrixMeta,
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
