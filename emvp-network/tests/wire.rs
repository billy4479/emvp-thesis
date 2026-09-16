#![expect(
    clippy::unwrap_used,
    reason = "fixed test fixtures establish that codec steps must succeed"
)]

//! Wire-protocol codec tests for protocol v2: golden bytes, round trips
//! through the plan → reserve → decode view pipeline, and the rejection
//! of every malformed frame class the parser must detect — including
//! hostile counts that must be refused before any allocation.

use std::io::Cursor;

use emvp::{AnswerRef, EmvpParams, EncryptedQueryRef};
use emvp_network::{
    ClientHello, CodecError, ErrorCode, ErrorResponse, EvaluateEntryInput, EvaluateWorkspace,
    FrameKind, FrameReader, HandshakeError, PROTOCOL_MODULUS, PROTOCOL_VERSION, ProductEntryInput,
    ProductsWorkspace, UploadAcceptedWorkspace, UploadWorkspace, decode_evaluate, decode_products,
    decode_upload, decode_upload_accepted, plan_evaluate, plan_products, plan_upload,
    plan_upload_accepted, read_client_hello, read_error, read_frame_header, read_server_hello,
    read_upload_accepted, server_handshake, write_client_hello, write_error, write_evaluate,
    write_frame_header, write_products, write_upload_accepted, write_upload_matrices,
};
use prime_field_layer::{FieldElement, PrimeField};

mod common;
use common::{
    AnswerFixture, EvaluateFixture, ProductFixture, QueryFixture, UploadFixture, encode_evaluate,
    encode_products, encode_upload, evaluate_inputs, open_frame, product_inputs,
    read_evaluate_frame, read_products_frame, read_upload_frame,
};

/// Decodes one upload into `workspace`, for the workspace-reuse tests
/// that feed many frames through the same reservation.
fn decode_upload_into(bytes: &[u8], workspace: &mut UploadWorkspace) {
    let mut cursor = Cursor::new(bytes);
    let (mut frame, _) = open_frame(&mut cursor, FrameKind::UploadMatrices).unwrap();
    let plan = plan_upload(&mut frame).unwrap();
    workspace.reserve(&plan).unwrap();
    decode_upload(&mut frame, &plan, workspace).unwrap();
}

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

fn upload(rows: usize, columns: usize) -> UploadFixture {
    UploadFixture {
        params: PARAMS,
        instance_id: 42,
        rows,
        columns,
        values: values(rows * columns),
    }
}

fn evaluate_entry() -> EvaluateFixture {
    EvaluateFixture {
        matrix_id: 7,
        queries: vec![
            QueryFixture {
                instance_id: 42,
                query_id: 0,
                values: values(16),
            },
            QueryFixture {
                instance_id: 42,
                query_id: 1,
                values: values(16),
            },
        ],
    }
}

fn product_entry() -> ProductFixture {
    ProductFixture {
        matrix_id: 7,
        answers: vec![AnswerFixture {
            instance_id: 42,
            query_id: 3,
            rows: 4,
            blocks: 2,
            values: values(8),
        }],
    }
}

fn le32(value: u32) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

fn le64(value: u64) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

fn le128(value: u128) -> Vec<u8> {
    value.to_le_bytes().to_vec()
}

#[test]
fn client_hello_has_golden_bytes() {
    let mut buffer = Vec::new();
    write_client_hello(&mut buffer, PROTOCOL_VERSION, PROTOCOL_MODULUS).unwrap();
    // "EMVP", version 2 little-endian, 998244353 little-endian.
    assert_eq!(
        buffer,
        vec![0x45, 0x4D, 0x56, 0x50, 0x02, 0x00, 0x01, 0x00, 0x80, 0x3B]
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
    let mut buffer = vec![0x45, 0x4D, 0x56, 0x51, 0x02, 0x00, 0x01, 0x00, 0x83, 0x3B];
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
        Err(HandshakeError::UnsupportedVersion { received }) if received == PROTOCOL_VERSION + 1
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
fn upload_matrices_has_golden_v2_bytes() {
    let field = field();
    let upload = UploadFixture {
        params: PARAMS,
        instance_id: 42,
        rows: 2,
        columns: 2,
        values: vec![1, 2, 3, 4]
            .into_iter()
            .map(|value| field.element_u32(value))
            .collect(),
    };
    let mut buffer = Vec::new();
    write_upload_matrices(&mut buffer, &[upload.view()]).unwrap();

    // Header, list count, the 60-byte descriptor
    // {k, ell, b, lambda, instance_id, rows, columns}, then the four
    // canonical value words. No per-record value count exists in v2.
    let mut expected = Vec::new();
    expected.push(FrameKind::UploadMatrices.to_u8());
    expected.extend_from_slice(&84_u64.to_le_bytes());
    expected.extend_from_slice(&1_u64.to_le_bytes());
    expected.extend_from_slice(&le64(u64::try_from(PARAMS.k).unwrap()));
    expected.extend_from_slice(&le64(u64::try_from(PARAMS.ell).unwrap()));
    expected.extend_from_slice(&le64(u64::try_from(PARAMS.b).unwrap()));
    expected.extend_from_slice(&le32(PARAMS.lambda));
    expected.extend_from_slice(&le128(42));
    expected.extend_from_slice(&le64(2));
    expected.extend_from_slice(&le64(2));
    for value in 1..=4 {
        expected.extend_from_slice(&le32(value));
    }
    assert_eq!(buffer, expected);
    assert_eq!(buffer.len(), 9 + 84);

    let (workspace, payload_len) = read_upload_frame(&buffer).unwrap();
    assert_eq!(payload_len, 84);
    let views = workspace.views();
    assert_eq!(views.len(), 1);
    let view = views.get(0).unwrap();
    assert_eq!(view.params, PARAMS);
    assert_eq!(view.instance_id, 42);
    assert_eq!(view.rows, 2);
    assert_eq!(view.columns, 2);
    assert_eq!(
        view.values,
        [1, 2, 3, 4].map(|value| field.element_u32(value))
    );
}

#[test]
fn upload_matrices_round_trips_through_views() {
    let uploads = vec![upload(3, 4), upload(1, 16)];
    let buffer = encode_upload(&uploads);
    let (workspace, payload_len) = read_upload_frame(&buffer).unwrap();
    assert_eq!(payload_len, buffer.len() as u64 - 9);
    let views = workspace.views();
    assert_eq!(views.len(), uploads.len());
    for (index, expected) in uploads.iter().enumerate() {
        let view = views.get(index).unwrap();
        assert_eq!(view.params, expected.params);
        assert_eq!(view.instance_id, expected.instance_id);
        assert_eq!(view.rows, expected.rows);
        assert_eq!(view.columns, expected.columns);
        assert_eq!(view.values, expected.values.as_slice());
        assert_eq!(
            view.matrix().unwrap().values(),
            expected.values.as_slice(),
            "the borrowed matrix ref matches the fixture values"
        );
    }
}

#[test]
fn upload_accepted_has_golden_bytes_and_round_trips() {
    let identifiers = vec![1, 2, 3];
    let mut buffer = Vec::new();
    write_upload_accepted(&mut buffer, &identifiers).unwrap();
    let mut expected = Vec::new();
    expected.push(FrameKind::UploadAccepted.to_u8());
    expected.extend_from_slice(&32_u64.to_le_bytes());
    expected.extend_from_slice(&3_u64.to_le_bytes());
    expected.extend_from_slice(&1_u64.to_le_bytes());
    expected.extend_from_slice(&2_u64.to_le_bytes());
    expected.extend_from_slice(&3_u64.to_le_bytes());
    assert_eq!(buffer, expected);

    let (decoded, _) = read_upload_accepted(&mut Cursor::new(&buffer)).unwrap();
    assert_eq!(decoded, identifiers);
}

#[test]
fn evaluate_has_golden_v2_bytes() {
    let field = field();
    let entry = EvaluateFixture {
        matrix_id: 7,
        queries: vec![
            QueryFixture {
                instance_id: 42,
                query_id: 0,
                values: [10_u32, 11].map(|value| field.element_u32(value)).to_vec(),
            },
            QueryFixture {
                instance_id: 42,
                query_id: 1,
                values: [12_u32, 13].map(|value| field.element_u32(value)).to_vec(),
            },
        ],
    };
    let inputs = evaluate_inputs(&entry);
    let mut buffer = Vec::new();
    write_evaluate(&mut buffer, &inputs).unwrap();

    // Header, {entry count, total query count}, one entry descriptor
    // {matrix_id, query_count, query_width}, two query descriptors
    // {instance_id, query_id}, then four value words.
    let mut expected = Vec::new();
    expected.push(FrameKind::Evaluate.to_u8());
    expected.extend_from_slice(&104_u64.to_le_bytes());
    expected.extend_from_slice(&1_u64.to_le_bytes());
    expected.extend_from_slice(&2_u64.to_le_bytes());
    expected.extend_from_slice(&7_u64.to_le_bytes());
    expected.extend_from_slice(&2_u64.to_le_bytes());
    expected.extend_from_slice(&2_u64.to_le_bytes());
    expected.extend_from_slice(&le128(42));
    expected.extend_from_slice(&0_u64.to_le_bytes());
    expected.extend_from_slice(&le128(42));
    expected.extend_from_slice(&1_u64.to_le_bytes());
    for value in 10..=13 {
        expected.extend_from_slice(&le32(value));
    }
    assert_eq!(buffer, expected);
    assert_eq!(buffer.len(), 9 + 104);

    let (workspace, payload_len) = read_evaluate_frame(&buffer).unwrap();
    assert_eq!(payload_len, 104);
    let views = workspace.views();
    assert_eq!(views.len(), 1);
    let decoded = views.get(0).unwrap();
    assert_eq!(decoded.matrix_id(), 7);
    assert_eq!(decoded.query_width(), 2);
    assert_eq!(decoded.len(), 2);
    for (index, query) in decoded.iter().enumerate() {
        assert_eq!(query.instance_id(), 42);
        assert_eq!(query.query_id(), u64::try_from(index).unwrap());
        assert_eq!(
            query.values(),
            [10 + 2 * index as u32, 11 + 2 * index as u32].map(|value| field.element_u32(value))
        );
    }
}

#[test]
fn evaluate_round_trips_through_views() {
    let entries = vec![evaluate_entry(), {
        let mut second = evaluate_entry();
        second.matrix_id = 9;
        second.queries.clear();
        second
    }];
    let buffer = encode_evaluate(&entries);
    let (workspace, payload_len) = read_evaluate_frame(&buffer).unwrap();
    assert_eq!(payload_len, buffer.len() as u64 - 9);
    let views = workspace.views();
    assert_eq!(views.len(), entries.len());
    let first = views.get(0).unwrap();
    assert_eq!(first.matrix_id(), 7);
    assert_eq!(first.query_width(), 16);
    assert_eq!(first.len(), 2);
    assert_eq!(
        first.query(0).unwrap().values(),
        entries[0].queries[0].values.as_slice()
    );
    assert_eq!(
        first.query(1).unwrap().values(),
        entries[0].queries[1].values.as_slice()
    );
    assert!(first.query(2).is_none());
    let second = views.get(1).unwrap();
    assert_eq!(second.matrix_id(), 9);
    assert_eq!(second.query_width(), 0);
    assert_eq!(second.len(), 0);
    assert!(second.is_empty());
}

#[test]
fn products_has_golden_v2_bytes() {
    let field = field();
    let entry = ProductFixture {
        matrix_id: 7,
        answers: vec![AnswerFixture {
            instance_id: 42,
            query_id: 3,
            rows: 2,
            blocks: 2,
            values: [5_u32, 6, 7, 8]
                .map(|value| field.element_u32(value))
                .to_vec(),
        }],
    };
    let inputs = product_inputs(&entry);
    let mut buffer = Vec::new();
    write_products(&mut buffer, &inputs).unwrap();

    // Header, {entry count, total answer count}, one 48-byte entry
    // descriptor {matrix_id, instance_id, rows, blocks, answer_count},
    // one answer descriptor {query_id}, then four value words.
    let mut expected = Vec::new();
    expected.push(FrameKind::Products.to_u8());
    expected.extend_from_slice(&88_u64.to_le_bytes());
    expected.extend_from_slice(&1_u64.to_le_bytes());
    expected.extend_from_slice(&1_u64.to_le_bytes());
    expected.extend_from_slice(&7_u64.to_le_bytes());
    expected.extend_from_slice(&le128(42));
    expected.extend_from_slice(&2_u64.to_le_bytes());
    expected.extend_from_slice(&2_u64.to_le_bytes());
    expected.extend_from_slice(&1_u64.to_le_bytes());
    expected.extend_from_slice(&3_u64.to_le_bytes());
    for value in 5..=8 {
        expected.extend_from_slice(&le32(value));
    }
    assert_eq!(buffer, expected);
    assert_eq!(buffer.len(), 9 + 88);

    let (workspace, payload_len) = read_products_frame(&buffer).unwrap();
    assert_eq!(payload_len, 88);
    let views = workspace.views();
    assert_eq!(views.len(), 1);
    let decoded = views.get(0).unwrap();
    assert_eq!(decoded.matrix_id(), 7);
    assert_eq!(decoded.instance_id(), 42);
    assert_eq!(decoded.rows(), 2);
    assert_eq!(decoded.blocks(), 2);
    assert_eq!(decoded.len(), 1);
    let answer = decoded.answer(0).unwrap();
    assert_eq!(answer.query_id(), 3);
    assert_eq!(answer.instance_id(), 42);
    assert_eq!(answer.rows(), 2);
    assert_eq!(answer.blocks(), 2);
    assert_eq!(
        answer.values(),
        [5, 6, 7, 8].map(|value| field.element_u32(value))
    );
}

#[test]
fn products_round_trips_through_views() {
    let entries = vec![product_entry(), {
        let mut second = product_entry();
        second.matrix_id = 9;
        second.answers.push(AnswerFixture {
            instance_id: 42,
            query_id: 4,
            rows: 4,
            blocks: 2,
            values: values(8),
        });
        second
    }];
    let buffer = encode_products(&entries);
    let (workspace, _) = read_products_frame(&buffer).unwrap();
    let views = workspace.views();
    assert_eq!(views.len(), entries.len());
    for (index, entry) in views.iter().enumerate() {
        assert_eq!(entry.matrix_id(), entries[index].matrix_id);
        assert_eq!(entry.len(), entries[index].answers.len());
        for (answer_index, answer) in entry.iter().enumerate() {
            let expected = &entries[index].answers[answer_index];
            assert_eq!(answer.query_id(), expected.query_id);
            assert_eq!(answer.instance_id(), expected.instance_id);
            assert_eq!(answer.values(), expected.values.as_slice());
        }
    }
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
    let mut buffer = encode_products(&[product_entry()]);
    // Entry descriptor: matrix_id (8), instance_id (16), then rows at
    // 9 + 8 + 8 + 8 + 16 = 49, blocks at 57, answer_count at 65.
    let rows_offset = 9 + 8 + 8 + 8 + 16;
    buffer[rows_offset..rows_offset + 8].copy_from_slice(&0_u64.to_le_bytes());
    buffer[rows_offset + 8..rows_offset + 16].copy_from_slice(&0_u64.to_le_bytes());
    buffer[rows_offset + 16..rows_offset + 24].copy_from_slice(&0_u64.to_le_bytes());
    assert!(matches!(
        read_products_frame(&buffer),
        Err(CodecError::InvalidDimensions {
            name: "answer rows",
            value: 0
        })
    ));

    let mut blocks_only = encode_products(&[product_entry()]);
    blocks_only[rows_offset..rows_offset + 8].copy_from_slice(&1_u64.to_le_bytes());
    blocks_only[rows_offset + 8..rows_offset + 16].copy_from_slice(&0_u64.to_le_bytes());
    blocks_only[rows_offset + 16..rows_offset + 24].copy_from_slice(&0_u64.to_le_bytes());
    assert!(matches!(
        read_products_frame(&blocks_only),
        Err(CodecError::InvalidDimensions {
            name: "answer blocks",
            value: 0
        })
    ));
}

#[test]
fn zero_matrix_dimensions_are_rejected() {
    let mut buffer = encode_upload(&[upload(2, 4)]);
    // Descriptor: k, ell, b (8 each), lambda (4), instance_id (16), so
    // rows sit at 9 + 8 + 28 + 16 = 61 and columns trail by 8.
    let rows_offset = 9 + 8 + 28 + 16;
    buffer[rows_offset..rows_offset + 8].copy_from_slice(&0_u64.to_le_bytes());
    assert!(matches!(
        read_upload_frame(&buffer),
        Err(CodecError::InvalidDimensions {
            name: "matrix rows",
            value: 0
        })
    ));

    let mut columns_only = encode_upload(&[upload(2, 4)]);
    columns_only[rows_offset..rows_offset + 8].copy_from_slice(&1_u64.to_le_bytes());
    columns_only[rows_offset + 8..rows_offset + 16].copy_from_slice(&0_u64.to_le_bytes());
    assert!(matches!(
        read_upload_frame(&columns_only),
        Err(CodecError::InvalidDimensions {
            name: "matrix columns",
            value: 0
        })
    ));
}

#[test]
fn empty_message_lists_round_trip() {
    let buffer = encode_upload(&[]);
    let (workspace, _) = read_upload_frame(&buffer).unwrap();
    assert!(workspace.views().is_empty());

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
    let mut buffer = encode_evaluate(&[evaluate_entry()]);
    buffer.pop();
    buffer.pop();
    assert!(matches!(
        read_evaluate_frame(&buffer),
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
    let mut buffer = encode_upload(&[upload(1, 4)]);
    // The first element's canonical bytes sit at 9 + 8 + 60: the v2
    // descriptor table replaces v1's fixed prefix.
    let first_element = 9 + 8 + 60;
    buffer[first_element..first_element + 4].copy_from_slice(&u32::MAX.to_le_bytes());
    assert!(matches!(
        read_upload_frame(&buffer),
        Err(CodecError::NonCanonicalField { value: u32::MAX })
    ));
}

#[test]
fn derived_value_lengths_must_match_the_payload_exactly() {
    // Inflating a matrix's rows grows the derived value region beyond the
    // declared payload: a truncation, caught at plan time.
    let mut inflated = encode_upload(&[upload(2, 4)]);
    let rows_offset = 9 + 8 + 28 + 16;
    inflated[rows_offset..rows_offset + 8].copy_from_slice(&3_u64.to_le_bytes());
    assert!(matches!(
        read_upload_frame(&inflated),
        Err(CodecError::TruncatedFrame)
    ));

    // Deflating them leaves payload bytes unconsumed: trailing bytes.
    let mut deflated = encode_upload(&[upload(2, 4)]);
    deflated[rows_offset..rows_offset + 8].copy_from_slice(&1_u64.to_le_bytes());
    assert!(matches!(
        read_upload_frame(&deflated),
        Err(CodecError::TrailingFrameBytes { extra: 16 })
    ));
}

#[test]
fn summary_counts_must_match_descriptor_tables() {
    // Inflating the evaluate summary's total query count contradicts the
    // single per-entry count.
    let mut evaluate = encode_evaluate(&[evaluate_entry()]);
    let total_offset = 9 + 8;
    evaluate[total_offset..total_offset + 8].copy_from_slice(&3_u64.to_le_bytes());
    assert!(matches!(
        read_evaluate_frame(&evaluate),
        Err(CodecError::CountMismatch {
            name: "query count",
            expected: 3,
            actual: 2
        })
    ));

    // The same for products' total answer count.
    let mut products = encode_products(&[product_entry()]);
    products[total_offset..total_offset + 8].copy_from_slice(&2_u64.to_le_bytes());
    assert!(matches!(
        read_products_frame(&products),
        Err(CodecError::CountMismatch {
            name: "answer count",
            expected: 2,
            actual: 1
        })
    ));
}

#[test]
fn descriptor_tables_beyond_the_payload_are_rejected_without_allocation() {
    // One matrix descriptor claimed, one byte short of fitting.
    let mut buffer = Vec::new();
    write_frame_header(&mut buffer, FrameKind::UploadMatrices, 8 + 59).unwrap();
    buffer.extend_from_slice(&1_u64.to_le_bytes());
    buffer.extend_from_slice(&[0_u8; 59]);
    let mut cursor = Cursor::new(&buffer);
    read_frame_header(&mut cursor).unwrap().unwrap();
    let mut frame = FrameReader::new(&mut cursor, 8 + 59);
    let info = allocation_counter::measure(|| {
        assert!(matches!(
            plan_upload(&mut frame),
            Err(CodecError::TruncatedFrame)
        ));
    });
    assert_eq!(info.count_total, 0);

    // Three evaluate entries claimed, no descriptor bytes present.
    let mut evaluate = Vec::new();
    write_frame_header(&mut evaluate, FrameKind::Evaluate, 16).unwrap();
    evaluate.extend_from_slice(&3_u64.to_le_bytes());
    evaluate.extend_from_slice(&0_u64.to_le_bytes());
    let mut cursor = Cursor::new(&evaluate);
    read_frame_header(&mut cursor).unwrap().unwrap();
    let mut frame = FrameReader::new(&mut cursor, 16);
    assert!(matches!(
        plan_evaluate(&mut frame),
        Err(CodecError::TruncatedFrame)
    ));
}

#[test]
fn inflated_counts_are_rejected_without_allocation() {
    // A frame claiming a u64::MAX-byte payload with a record whose derived
    // element count cannot possibly be buffered must fail without
    // attempting a multi-terabyte reservation.
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
    let mut cursor = Cursor::new(&buffer);
    read_frame_header(&mut cursor).unwrap().unwrap();
    let mut frame = FrameReader::new(&mut cursor, u64::MAX);
    let info = allocation_counter::measure(|| {
        assert!(matches!(
            plan_upload(&mut frame),
            Err(CodecError::TrailingFrameBytes { .. })
        ));
    });
    // Only the sanctioned one-record descriptor metadata vector may have
    // been allocated; no value-region reservation was attempted.
    assert!(info.count_total <= 1, "metadata allocations: {info:?}");

    // A descriptor table that fits the declared payload math but whose
    // metadata can never be reserved must fail closed with an allocation
    // refusal. At most one allocation attempt (the refused reservation
    // itself) shows up in the counter, and no memory was retained.
    let count: u64 = 1 << 44;
    let mut hostile = Vec::new();
    write_frame_header(&mut hostile, FrameKind::UploadMatrices, count * 60 + 8).unwrap();
    hostile.extend_from_slice(&count.to_le_bytes());
    let mut cursor = Cursor::new(&hostile);
    read_frame_header(&mut cursor).unwrap().unwrap();
    let mut frame = FrameReader::new(&mut cursor, count * 60 + 8);
    let info = allocation_counter::measure(|| {
        assert!(matches!(
            plan_upload(&mut frame),
            Err(CodecError::AllocationFailed)
        ));
    });
    assert!(info.count_total <= 1, "allocation attempts: {info:?}");
}

#[test]
fn element_counts_beyond_the_frame_are_truncations() {
    let mut buffer = Vec::new();
    write_frame_header(&mut buffer, FrameKind::UploadMatrices, 8).unwrap();
    buffer.extend_from_slice(&1_u64.to_le_bytes()); // list count: one record
    let mut cursor = Cursor::new(&buffer);
    read_frame_header(&mut cursor).unwrap().unwrap();
    let mut frame = FrameReader::new(&mut cursor, 8);
    assert!(matches!(
        plan_upload(&mut frame),
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
        read_upload_frame(&buffer),
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

#[test]
fn read_field_slice_into_round_trips_canonical_words() {
    use emvp_network::write_field_slice;

    for count in [1, 10, 4095, 4096, 4097, 5000] {
        let source = values(count);
        let mut encoded = Vec::new();
        write_field_slice(&mut encoded, &source).unwrap();

        let zero = PrimeField::<PROTOCOL_MODULUS>::new().element_u32(0);
        let mut destination = vec![zero; count];
        let mut cursor = Cursor::new(&encoded);
        let mut frame = FrameReader::new(&mut cursor, encoded.len() as u64);
        frame.read_field_slice_into(&mut destination).unwrap();
        assert_eq!(destination, source);

        // A destination the payload cannot hold truncates.
        destination.push(zero);
        let mut cursor = Cursor::new(&encoded);
        let mut frame = FrameReader::new(&mut cursor, encoded.len() as u64);
        assert!(matches!(
            frame.read_field_slice_into(&mut destination),
            Err(CodecError::TruncatedFrame)
        ));
    }
}

#[test]
fn workspaces_reuse_capacity_across_frames() {
    // Uploads: a two-matrix frame decoded twice, then a one-matrix frame
    // into the same workspace.
    let big = encode_upload(&[upload(3, 4), upload(1, 16)]);
    let small = encode_upload(&[upload(1, 16)]);
    let mut workspace = UploadWorkspace::new();
    decode_upload_into(&big, &mut workspace);
    let grown_capacity = workspace.capacity_fields();
    assert!(grown_capacity >= 28);

    decode_upload_into(&big, &mut workspace);
    assert_eq!(workspace.capacity_fields(), grown_capacity);
    let views = workspace.views();
    assert_eq!(views.len(), 2);
    assert_eq!(views.get(0).unwrap().values, values(12));
    assert_eq!(views.get(1).unwrap().values, values(16));

    decode_upload_into(&small, &mut workspace);
    assert_eq!(workspace.capacity_fields(), grown_capacity);
    let views = workspace.views();
    assert_eq!(views.len(), 1);
    assert_eq!(views.get(0).unwrap().values, values(16));

    // Evaluates: two queries, then one.
    let two_queries = evaluate_entry();
    let mut one_query = evaluate_entry();
    one_query.queries.pop();
    let big = encode_evaluate(&[two_queries]);
    let small = encode_evaluate(&[one_query]);
    let mut workspace = EvaluateWorkspace::new();
    let decode = |bytes: &[u8], workspace: &mut EvaluateWorkspace| {
        let mut cursor = Cursor::new(bytes);
        let (mut frame, _) = open_frame(&mut cursor, FrameKind::Evaluate).unwrap();
        let plan = plan_evaluate(&mut frame).unwrap();
        workspace.reserve(&plan).unwrap();
        decode_evaluate(&mut frame, &plan, workspace).unwrap();
    };
    decode(&big, &mut workspace);
    let grown_capacity = workspace.capacity_fields();
    decode(&big, &mut workspace);
    assert_eq!(workspace.capacity_fields(), grown_capacity);
    assert_eq!(workspace.views().get(0).unwrap().len(), 2);
    assert_eq!(
        workspace.views().get(0).unwrap().query(0).unwrap().values(),
        values(16)
    );
    decode(&small, &mut workspace);
    assert_eq!(workspace.capacity_fields(), grown_capacity);
    assert_eq!(workspace.views().get(0).unwrap().len(), 1);
    assert_eq!(
        workspace.views().get(0).unwrap().query(0).unwrap().values(),
        values(16)
    );

    // Products: one answer, then two.
    let one_answer = product_entry();
    let mut two_answers = product_entry();
    two_answers.answers.push(AnswerFixture {
        instance_id: 42,
        query_id: 4,
        rows: 4,
        blocks: 2,
        values: values(8),
    });
    let small = encode_products(&[one_answer]);
    let big = encode_products(&[two_answers]);
    let mut workspace = ProductsWorkspace::new();
    let decode = |bytes: &[u8], workspace: &mut ProductsWorkspace| {
        let mut cursor = Cursor::new(bytes);
        let (mut frame, _) = open_frame(&mut cursor, FrameKind::Products).unwrap();
        let plan = plan_products(&mut frame).unwrap();
        workspace.reserve(&plan).unwrap();
        decode_products(&mut frame, &plan, workspace).unwrap();
    };
    decode(&big, &mut workspace);
    let grown_capacity = workspace.capacity_fields();
    decode(&big, &mut workspace);
    assert_eq!(workspace.capacity_fields(), grown_capacity);
    assert_eq!(workspace.views().get(0).unwrap().len(), 2);
    assert_eq!(
        workspace
            .views()
            .get(0)
            .unwrap()
            .answer(1)
            .unwrap()
            .values(),
        values(8)
    );
    decode(&small, &mut workspace);
    assert_eq!(workspace.capacity_fields(), grown_capacity);
    assert_eq!(workspace.views().get(0).unwrap().len(), 1);
}

#[test]
fn second_upload_cycle_is_allocation_free() {
    let upload = upload(3, 4);
    let buffer = encode_upload(std::slice::from_ref(&upload));
    let payload_len = buffer.len() as u64 - 9;

    // First cycle: grows the workspace and primes the leaked sink.
    let mut workspace = UploadWorkspace::new();
    {
        let mut cursor = Cursor::new(buffer.as_slice());
        read_frame_header(&mut cursor).unwrap().unwrap();
        let mut frame = FrameReader::new(&mut cursor, payload_len);
        let plan = plan_upload(&mut frame).unwrap();
        workspace.reserve(&plan).unwrap();
        decode_upload(&mut frame, &plan, &mut workspace).unwrap();
    }
    let sink: &'static mut Vec<u8> = Box::leak(Box::new(Vec::with_capacity(buffer.len())));

    // Second cycle: plan metadata is the only sanctioned allocation and
    // stays outside the measurement; reserve, decode, and encode into the
    // preallocated sink must all be allocation-free.
    let mut cursor = Cursor::new(buffer.as_slice());
    read_frame_header(&mut cursor).unwrap().unwrap();
    let mut frame = FrameReader::new(&mut cursor, payload_len);
    let plan = plan_upload(&mut frame).unwrap();
    let info = allocation_counter::measure(|| {
        workspace.reserve(&plan).unwrap();
        let views = decode_upload(&mut frame, &plan, &mut workspace).unwrap();
        let view = views.get(0).unwrap();
        write_upload_matrices(sink, &[view]).unwrap();
    });
    assert_eq!(info.count_total, 0);
    assert_eq!(sink.len(), buffer.len());
    sink.clear();
}

#[test]
fn second_evaluate_cycle_is_allocation_free() {
    let entry = evaluate_entry();
    let mut buffer = Vec::new();
    {
        let inputs = evaluate_inputs(&entry);
        write_evaluate(&mut buffer, &inputs).unwrap();
    }
    let payload_len = buffer.len() as u64 - 9;

    let mut workspace = EvaluateWorkspace::new();
    {
        let mut cursor = Cursor::new(buffer.as_slice());
        read_frame_header(&mut cursor).unwrap().unwrap();
        let mut frame = FrameReader::new(&mut cursor, payload_len);
        let plan = plan_evaluate(&mut frame).unwrap();
        workspace.reserve(&plan).unwrap();
        decode_evaluate(&mut frame, &plan, &mut workspace).unwrap();
    }
    let sink: &'static mut Vec<u8> = Box::leak(Box::new(Vec::with_capacity(buffer.len())));

    let mut cursor = Cursor::new(buffer.as_slice());
    read_frame_header(&mut cursor).unwrap().unwrap();
    let mut frame = FrameReader::new(&mut cursor, payload_len);
    let plan = plan_evaluate(&mut frame).unwrap();
    let info = allocation_counter::measure(|| {
        workspace.reserve(&plan).unwrap();
        let views = decode_evaluate(&mut frame, &plan, &mut workspace).unwrap();
        let decoded = views.get(0).unwrap();
        let queries: [EncryptedQueryRef<'_, PROTOCOL_MODULUS>; 2] =
            [decoded.query(0).unwrap(), decoded.query(1).unwrap()];
        let inputs = [EvaluateEntryInput {
            matrix_id: decoded.matrix_id(),
            queries: &queries,
        }];
        write_evaluate(sink, &inputs).unwrap();
    });
    assert_eq!(info.count_total, 0);
    assert_eq!(sink.len(), buffer.len());
    sink.clear();
}

#[test]
fn second_products_cycle_is_allocation_free() {
    let entry = product_entry();
    let mut buffer = Vec::new();
    {
        let inputs = product_inputs(&entry);
        write_products(&mut buffer, &inputs).unwrap();
    }
    let payload_len = buffer.len() as u64 - 9;

    let mut workspace = ProductsWorkspace::new();
    {
        let mut cursor = Cursor::new(buffer.as_slice());
        read_frame_header(&mut cursor).unwrap().unwrap();
        let mut frame = FrameReader::new(&mut cursor, payload_len);
        let plan = plan_products(&mut frame).unwrap();
        workspace.reserve(&plan).unwrap();
        decode_products(&mut frame, &plan, &mut workspace).unwrap();
    }
    let sink: &'static mut Vec<u8> = Box::leak(Box::new(Vec::with_capacity(buffer.len())));

    let mut cursor = Cursor::new(buffer.as_slice());
    read_frame_header(&mut cursor).unwrap().unwrap();
    let mut frame = FrameReader::new(&mut cursor, payload_len);
    let plan = plan_products(&mut frame).unwrap();
    let info = allocation_counter::measure(|| {
        workspace.reserve(&plan).unwrap();
        let views = decode_products(&mut frame, &plan, &mut workspace).unwrap();
        let decoded = views.get(0).unwrap();
        let answers: [AnswerRef<'_, PROTOCOL_MODULUS>; 1] = [decoded.answer(0).unwrap()];
        let inputs = [ProductEntryInput {
            matrix_id: decoded.matrix_id(),
            instance_id: decoded.instance_id(),
            rows: decoded.rows(),
            blocks: decoded.blocks(),
            answers: &answers,
        }];
        write_products(sink, &inputs).unwrap();
    });
    assert_eq!(info.count_total, 0);
    assert_eq!(sink.len(), buffer.len());
    sink.clear();
}

#[test]
fn upload_accepted_workspace_round_trips() {
    let mut buffer = Vec::new();
    write_upload_accepted(&mut buffer, &[5, 6]).unwrap();
    let mut cursor = Cursor::new(&buffer);
    let header = read_frame_header(&mut cursor).unwrap().unwrap();
    assert_eq!(header.kind, FrameKind::UploadAccepted);
    let mut frame = FrameReader::new(&mut cursor, header.payload_len);
    let plan = plan_upload_accepted(&mut frame).unwrap();
    let mut workspace = UploadAcceptedWorkspace::new();
    workspace.reserve(&plan).unwrap();
    let identifiers = decode_upload_accepted(&mut frame, &plan, &mut workspace);
    assert_eq!(identifiers.unwrap(), &[5, 6]);
    assert_eq!(workspace.identifiers(), &[5, 6]);
}
