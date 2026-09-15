#![expect(
    clippy::unwrap_used,
    reason = "fixed test fixtures establish that codec steps must succeed"
)]

//! Wire-protocol codec tests: golden bytes, message round trips, and the
//! rejection of every malformed frame class the parser must detect.

use std::io::Cursor;

use emvp::{AnswerMatrix, EmvpParams, EncryptedMatrix, EncryptedQuery};
use emvp_network::{
    ClientHello, CodecError, ErrorCode, ErrorResponse, EvaluateEntry, FrameKind, HandshakeError,
    MatrixUpload, PROTOCOL_MODULUS, PROTOCOL_VERSION, ProductEntry, read_client_hello, read_error,
    read_evaluate, read_frame_header, read_products, read_server_hello, read_upload_accepted,
    read_upload_matrices, server_handshake, write_client_hello, write_error, write_evaluate,
    write_frame_header, write_products, write_upload_accepted, write_upload_matrices,
};
use prime_field_layer::{FieldElement, PrimeField};

const PARAMS: EmvpParams = EmvpParams {
    k: 8,
    ell: 8,
    b: 2,
    lambda: 7,
};

const fn field() -> PrimeField<PROTOCOL_MODULUS> {
    PrimeField::<PROTOCOL_MODULUS>::new()
}

fn values(count: usize) -> Vec<FieldElement<PROTOCOL_MODULUS>> {
    let field = field();
    (0..count)
        .map(|index| field.element_u32((index % 1000) as u32))
        .collect()
}

fn upload(rows: usize, columns: usize) -> MatrixUpload {
    MatrixUpload {
        params: PARAMS,
        matrix: EncryptedMatrix::from_parts(42, rows, columns, values(rows * columns)).unwrap(),
    }
}

fn evaluate_entry() -> EvaluateEntry {
    EvaluateEntry {
        matrix_id: 7,
        queries: vec![
            EncryptedQuery::from_parts(42, 0, values(16)),
            EncryptedQuery::from_parts(42, 1, values(16)),
        ],
    }
}

fn product_entry() -> ProductEntry {
    ProductEntry {
        matrix_id: 7,
        answers: vec![AnswerMatrix::from_parts(42, 3, values(8), 4, 2)],
    }
}

#[test]
fn client_hello_has_golden_bytes() {
    let mut buffer = Vec::new();
    write_client_hello(&mut buffer, PROTOCOL_VERSION, PROTOCOL_MODULUS).unwrap();
    // "EMVP", version 1 little-endian, 998244353 little-endian.
    assert_eq!(
        buffer,
        vec![0x45, 0x4D, 0x56, 0x50, 0x01, 0x00, 0x01, 0x00, 0x80, 0x3B]
    );
    let hello = read_client_hello(&mut Cursor::new(buffer)).unwrap();
    assert_eq!(
        hello,
        ClientHello {
            version: PROTOCOL_VERSION,
            modulus: PROTOCOL_MODULUS
        }
    );
}

#[test]
fn client_hello_rejects_foreign_magic() {
    let mut buffer = vec![0x45, 0x4D, 0x56, 0x51, 0x01, 0x00, 0x01, 0x00, 0x83, 0x3B];
    assert!(matches!(
        read_client_hello(&mut Cursor::new(buffer.clone())),
        Err(CodecError::InvalidMagic)
    ));
    buffer[3] = 0x50;
    read_client_hello(&mut Cursor::new(buffer)).unwrap();
}

#[test]
fn truncated_hello_is_an_error() {
    let mut buffer = Vec::new();
    write_client_hello(&mut buffer, PROTOCOL_VERSION, PROTOCOL_MODULUS).unwrap();
    buffer.pop();
    assert!(matches!(
        read_client_hello(&mut Cursor::new(buffer)),
        Err(CodecError::TruncatedFrame)
    ));
}

#[test]
fn server_handshake_accepts_the_compiled_protocol() {
    let mut buffer = Vec::new();
    write_client_hello(&mut buffer, PROTOCOL_VERSION, PROTOCOL_MODULUS).unwrap();
    server_handshake(&mut Cursor::new(buffer)).unwrap();
}

#[test]
fn server_handshake_rejects_a_foreign_version_or_modulus() {
    let mut wrong_version = Vec::new();
    write_client_hello(&mut wrong_version, PROTOCOL_VERSION + 1, PROTOCOL_MODULUS).unwrap();
    assert!(matches!(
        server_handshake(&mut Cursor::new(wrong_version)),
        Err(HandshakeError::UnsupportedVersion { received: 2 })
    ));

    let mut wrong_modulus = Vec::new();
    write_client_hello(&mut wrong_modulus, PROTOCOL_VERSION, PROTOCOL_MODULUS - 1).unwrap();
    assert!(matches!(
        server_handshake(&mut Cursor::new(wrong_modulus)),
        Err(HandshakeError::UnsupportedModulus { .. })
    ));
}

#[test]
fn server_rejection_reaches_the_client() {
    // The stream holds a valid hello followed by the server's status byte;
    // the client rewrite of the same hello leaves the status readable.
    let mut rejected = Vec::new();
    write_client_hello(&mut rejected, PROTOCOL_VERSION, PROTOCOL_MODULUS).unwrap();
    rejected.push(0);
    assert!(matches!(
        handshake_as_client(&mut Cursor::new(rejected)),
        Err(HandshakeError::Rejected)
    ));

    let mut accepted = Vec::new();
    write_client_hello(&mut accepted, PROTOCOL_VERSION, PROTOCOL_MODULUS).unwrap();
    accepted.push(1);
    handshake_as_client(&mut Cursor::new(accepted)).unwrap();
}

fn handshake_as_client<S: std::io::Read + std::io::Write>(
    stream: &mut S,
) -> Result<(), HandshakeError> {
    write_client_hello(stream, PROTOCOL_VERSION, PROTOCOL_MODULUS)?;
    if read_server_hello(stream)? {
        Ok(())
    } else {
        Err(HandshakeError::Rejected)
    }
}

#[test]
fn upload_matrices_round_trips() {
    let uploads = vec![upload(3, 4), upload(1, 16)];
    let mut buffer = Vec::new();
    let written = write_upload_matrices(&mut buffer, &uploads).unwrap();
    assert_eq!(written, buffer.len() as u64);

    let (decoded, payload_len) = read_upload_matrices(&mut Cursor::new(&buffer)).unwrap();
    assert_eq!(decoded, uploads);
    assert_eq!(payload_len, buffer.len() as u64 - 9);
}

#[test]
fn upload_accepted_round_trips() {
    let identifiers = vec![1, 2, 3];
    let mut buffer = Vec::new();
    write_upload_accepted(&mut buffer, &identifiers).unwrap();
    let (decoded, _) = read_upload_accepted(&mut Cursor::new(&buffer)).unwrap();
    assert_eq!(decoded, identifiers);
}

#[test]
fn evaluate_round_trips() {
    let entries = vec![evaluate_entry(), {
        let mut second = evaluate_entry();
        second.matrix_id = 9;
        second.queries.clear();
        second
    }];
    let mut buffer = Vec::new();
    let written = write_evaluate(&mut buffer, &entries).unwrap();
    assert_eq!(written, buffer.len() as u64);
    let (decoded, payload_len) = read_evaluate(&mut Cursor::new(&buffer)).unwrap();
    assert_eq!(decoded, entries);
    assert_eq!(payload_len, buffer.len() as u64 - 9);
}

#[test]
fn products_round_trips() {
    let entries = vec![product_entry()];
    let mut buffer = Vec::new();
    write_products(&mut buffer, &entries).unwrap();
    let (decoded, _) = read_products(&mut Cursor::new(&buffer)).unwrap();
    assert_eq!(decoded, entries);
}

#[test]
fn error_round_trips_and_preserves_unknown_codes() {
    let mut buffer = Vec::new();
    let written = write_error(&mut buffer, ErrorCode::UnknownMatrixId, "no matrix 7").unwrap();
    assert_eq!(written, buffer.len() as u64);
    let (decoded, _) = read_error(&mut Cursor::new(&buffer)).unwrap();
    assert_eq!(
        decoded,
        ErrorResponse {
            code: ErrorCode::UnknownMatrixId.to_u32(),
            message: "no matrix 7".into()
        }
    );
    assert_eq!(decoded.code(), Some(ErrorCode::UnknownMatrixId));

    let mut foreign = Vec::new();
    write_error(
        &mut foreign,
        ErrorCode::from_u32(999).unwrap_or(ErrorCode::Internal),
        "x",
    )
    .unwrap();
    // Force an unknown code byte-for-byte: code 999 with a one-byte message.
    foreign[9] = 0xE7;
    foreign[10] = 0x03;
    let (decoded, _) = read_error(&mut Cursor::new(&foreign)).unwrap();
    assert_eq!(decoded.code, 999);
    assert_eq!(decoded.code(), None);
}

#[test]
fn upload_accepted_rejects_zero_identifiers() {
    let mut buffer = Vec::new();
    write_upload_accepted(&mut buffer, &[1, 0, 2]).unwrap();
    assert!(matches!(
        read_upload_accepted(&mut Cursor::new(&buffer)),
        Err(CodecError::InvalidDimensions {
            name: "matrix identifier",
            value: 0
        })
    ));
}

#[test]
fn zero_answer_dimensions_are_rejected() {
    let mut buffer = Vec::new();
    write_products(&mut buffer, &[product_entry()]).unwrap();
    // Per answer: 16 (instance id) + 8 (query id) + 8 (rows) + 8 (blocks)
    // + 8 (value count) after the 9-byte header and three u64 fields.
    let rows_offset = 9 + 8 + 8 + 8 + 16 + 8;
    buffer[rows_offset..rows_offset + 8].copy_from_slice(&0_u64.to_le_bytes());
    buffer[rows_offset + 8..rows_offset + 16].copy_from_slice(&0_u64.to_le_bytes());
    buffer[rows_offset + 16..rows_offset + 24].copy_from_slice(&0_u64.to_le_bytes());
    assert!(matches!(
        read_products(&mut Cursor::new(&buffer)),
        Err(CodecError::InvalidDimensions {
            name: "answer rows",
            value: 0
        })
    ));

    let mut blocks_only = Vec::new();
    write_products(&mut blocks_only, &[product_entry()]).unwrap();
    blocks_only[rows_offset..rows_offset + 8].copy_from_slice(&1_u64.to_le_bytes());
    blocks_only[rows_offset + 8..rows_offset + 16].copy_from_slice(&0_u64.to_le_bytes());
    blocks_only[rows_offset + 16..rows_offset + 24].copy_from_slice(&0_u64.to_le_bytes());
    assert!(matches!(
        read_products(&mut Cursor::new(&blocks_only)),
        Err(CodecError::InvalidDimensions {
            name: "answer blocks",
            value: 0
        })
    ));
}

#[test]
fn empty_message_lists_round_trip() {
    let mut buffer = Vec::new();
    write_upload_matrices(&mut buffer, &[]).unwrap();
    let (decoded, _) = read_upload_matrices(&mut Cursor::new(&buffer)).unwrap();
    assert!(decoded.is_empty());

    let mut accepted = Vec::new();
    write_upload_accepted(&mut accepted, &[]).unwrap();
    assert!(
        read_upload_accepted(&mut Cursor::new(&accepted))
            .unwrap()
            .0
            .is_empty()
    );
}

#[test]
fn clean_eof_is_not_a_frame() {
    let mut empty: &[u8] = &[];
    assert!(read_frame_header(&mut empty).unwrap().is_none());
}

#[test]
fn partial_header_is_truncated() {
    let mut one_byte: &[u8] = &[FrameKind::Evaluate.to_u8()];
    assert!(matches!(
        read_frame_header(&mut one_byte),
        Err(CodecError::TruncatedFrame)
    ));
}

#[test]
fn truncated_payload_is_detected_before_the_next_frame() {
    let mut buffer = Vec::new();
    write_evaluate(&mut buffer, &[evaluate_entry()]).unwrap();
    buffer.pop();
    buffer.pop();
    assert!(matches!(
        read_evaluate(&mut Cursor::new(&buffer)),
        Err(CodecError::TruncatedFrame)
    ));
}

#[test]
fn trailing_payload_bytes_are_detected() {
    let mut buffer = Vec::new();
    write_upload_accepted(&mut buffer, &[1, 2, 3]).unwrap();
    // Claim two extra payload bytes; parsing must stop with them unparsed.
    let declared = u64::from_le_bytes(buffer[1..9].try_into().unwrap());
    buffer[1..9].copy_from_slice(&(declared + 2).to_le_bytes());
    buffer.extend_from_slice(&[0xAA, 0xBB]);
    assert!(matches!(
        read_upload_accepted(&mut Cursor::new(&buffer)),
        Err(CodecError::TrailingFrameBytes { extra: 2 })
    ));
}

#[test]
fn noncanonical_field_elements_are_rejected() {
    let mut buffer = Vec::new();
    write_upload_matrices(&mut buffer, &[upload(1, 4)]).unwrap();
    // The first element's canonical bytes sit at 9 + 8 + 68.
    let first_element = 9 + 8 + 68;
    buffer[first_element..first_element + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(matches!(
        read_upload_matrices(&mut Cursor::new(&buffer)),
        Err(CodecError::NonCanonicalField { value: u32::MAX })
    ));
}

#[test]
fn declared_value_counts_must_match_dimensions() {
    let mut buffer = Vec::new();
    write_upload_matrices(&mut buffer, &[upload(2, 4)]).unwrap();
    // The element count field sits at 9 + 8 (list count) + 60 (record
    // prefix).
    let count_offset = 9 + 8 + 60;
    let declared = u64::from_le_bytes(buffer[count_offset..count_offset + 8].try_into().unwrap());
    assert_eq!(declared, 8);
    buffer[count_offset..count_offset + 8].copy_from_slice(&9_u64.to_le_bytes());
    assert!(matches!(
        read_upload_matrices(&mut Cursor::new(&buffer)),
        Err(CodecError::CountMismatch {
            name: "matrix values",
            expected: 8,
            actual: 9
        })
    ));
}

#[test]
fn zero_dimensions_are_rejected() {
    let mut buffer = Vec::new();
    write_upload_matrices(&mut buffer, &[upload(2, 4)]).unwrap();
    // rows sit after the 28-byte parameter block and the 16-byte instance
    // identifier: 9 + 8 + 28 + 16; the value count trails the columns by
    // another 8 bytes.
    let rows_offset = 9 + 8 + 28 + 16;
    buffer[rows_offset..rows_offset + 8].copy_from_slice(&0_u64.to_le_bytes());
    buffer[rows_offset + 16..rows_offset + 24].copy_from_slice(&0_u64.to_le_bytes());
    assert!(matches!(
        read_upload_matrices(&mut Cursor::new(&buffer)),
        Err(CodecError::InvalidDimensions {
            name: "matrix rows",
            value: 0
        })
    ));
}

#[test]
fn impossible_counts_are_rejected_without_allocation() {
    // A frame claiming a u64::MAX-byte payload with one record whose
    // element count cannot possibly be buffered must fail with an
    // allocation error, not attempt a multi-terabyte reservation.
    let mut buffer = Vec::new();
    write_frame_header(&mut buffer, FrameKind::UploadMatrices, u64::MAX).unwrap();
    buffer.extend_from_slice(&1_u64.to_le_bytes()); // list count
    buffer.extend_from_slice(&u64::try_from(PARAMS.k).unwrap().to_le_bytes());
    buffer.extend_from_slice(&u64::try_from(PARAMS.ell).unwrap().to_le_bytes());
    buffer.extend_from_slice(&u64::try_from(PARAMS.b).unwrap().to_le_bytes());
    buffer.extend_from_slice(&PARAMS.lambda.to_le_bytes());
    buffer.extend_from_slice(&0_u128.to_le_bytes()); // instance id
    buffer.extend_from_slice(&(1_u64 << 33).to_le_bytes()); // rows
    buffer.extend_from_slice(&(1_u64 << 28).to_le_bytes()); // columns
    buffer.extend_from_slice(&(1_u64 << 61).to_le_bytes()); // value count
    let mut cursor = Cursor::new(buffer);
    assert!(matches!(
        read_upload_matrices(&mut cursor),
        Err(CodecError::AllocationFailed)
    ));
}

#[test]
fn element_counts_beyond_the_frame_are_truncations() {
    let mut buffer = Vec::new();
    write_frame_header(&mut buffer, FrameKind::UploadMatrices, 8).unwrap();
    buffer.extend_from_slice(&1_u64.to_le_bytes()); // list count: one record
    let mut cursor = Cursor::new(buffer);
    assert!(matches!(
        read_upload_matrices(&mut cursor),
        Err(CodecError::TruncatedFrame)
    ));
}

#[test]
fn unknown_frame_kinds_are_rejected() {
    let crafted = [0x2A, 0, 0, 0, 0, 0, 0, 0, 0];
    assert!(matches!(
        read_frame_header(&mut Cursor::new(crafted)),
        Err(CodecError::UnknownFrameKind { kind: 0x2A })
    ));
}

#[test]
fn wrong_frame_kinds_are_unexpected() {
    let mut buffer = Vec::new();
    write_upload_accepted(&mut buffer, &[1]).unwrap();
    assert!(matches!(
        read_upload_matrices(&mut Cursor::new(&buffer)),
        Err(CodecError::UnexpectedFrame {
            expected: FrameKind::UploadMatrices,
            actual: 2
        })
    ));
}

#[test]
fn error_messages_must_be_utf8() {
    let mut buffer = Vec::new();
    write_error(&mut buffer, ErrorCode::Internal, "ok").unwrap();
    // The message body starts at 9 + 4 + 8.
    buffer[21] = 0xFF;
    assert!(matches!(
        read_error(&mut Cursor::new(&buffer)),
        Err(CodecError::InvalidUtf8)
    ));
}

#[test]
fn error_codes_map_to_themselves() {
    for code in [
        ErrorCode::UnknownFrameKind,
        ErrorCode::TruncatedFrame,
        ErrorCode::TrailingFrameBytes,
        ErrorCode::NonCanonicalFieldElement,
        ErrorCode::ValueOutOfRange,
        ErrorCode::CountMismatch,
        ErrorCode::DimensionOverflow,
        ErrorCode::AllocationFailed,
        ErrorCode::InvalidUtf8,
        ErrorCode::InvalidParameters,
        ErrorCode::InstanceMismatch,
        ErrorCode::UnknownMatrixId,
        ErrorCode::DuplicateMatrixEntry,
        ErrorCode::DuplicateQueryId,
        ErrorCode::EmptyRequest,
        ErrorCode::NotLoaded,
        ErrorCode::AlreadyLoaded,
        ErrorCode::UploadFailed,
        ErrorCode::GpuFailure,
        ErrorCode::UnexpectedFrame,
        ErrorCode::Internal,
    ] {
        assert_eq!(ErrorCode::from_u32(code.to_u32()), Some(code));
    }
    assert_eq!(ErrorCode::from_u32(0), None);
    assert_eq!(ErrorCode::from_u32(9999), None);
}
