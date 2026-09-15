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
//! # Experimental
//!
//! This is research plumbing for a POC deployment. The protocol performs
//! no authentication or transport security: instance and query identifiers
//! guard against accidental artifact mixups only, never against a
//! malicious transport.
//!
//! [`FieldElement::to_canonical_le_bytes`]: prime_field_layer::FieldElement::to_canonical_le_bytes

pub mod error;
pub mod frame;
pub mod handshake;
pub mod messages;

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
    ErrorResponse, EvaluateEntry, MatrixUpload, ProductEntry, read_error, read_error_payload,
    read_evaluate, read_evaluate_payload, read_products, read_products_payload,
    read_upload_accepted, read_upload_accepted_payload, read_upload_matrices,
    read_upload_matrices_payload, write_error, write_evaluate, write_products,
    write_upload_accepted, write_upload_matrices,
};
