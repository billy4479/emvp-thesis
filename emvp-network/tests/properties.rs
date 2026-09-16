#![expect(
    clippy::unwrap_used,
    reason = "property fixtures must encode successfully before a property can be checked"
)]

//! Deterministic, bounded property tests for the wire codecs.
//!
//! Every property uses small generated messages (a couple of records and
//! at most a few dozen field elements), so the exhaustive prefix walks
//! over truncation boundaries stay cheap. The runner is pinned to a fixed
//! RNG seed so repeated runs exercise the same cases.

use std::io::Cursor;

use proptest::prelude::*;
use proptest::test_runner::{Config, RngSeed};

use emvp::{AnswerMatrix, EmvpParams, EncryptedMatrix, EncryptedQuery};
use emvp_network::{
    CodecError, ErrorCode, EvaluateEntry, FrameKind, HEADER_BYTES, MatrixUpload, PROTOCOL_MODULUS,
    ProductEntry, read_error, read_evaluate, read_frame_header, read_products,
    read_upload_accepted, read_upload_matrices, write_error, write_evaluate, write_products,
    write_upload_accepted, write_upload_matrices,
};
use prime_field_layer::{FieldElement, PrimeField};

/// Fixed seed so repeated runs exercise identical cases.
const SEED: u64 = 0x5EED_5EED_5EED_5EED;

/// Every stable error code, for round-trip selection.
const ALL_ERROR_CODES: [ErrorCode; 21] = [
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
];

fn proptest_config() -> Config {
    Config {
        cases: 64,
        rng_seed: RngSeed::Fixed(SEED),
        ..Config::default()
    }
}

proptest! {
    #![proptest_config(proptest_config())]

    #[test]
    fn upload_matrices_round_trip(uploads in uploads_strategy()) {
        let mut buffer = Vec::new();
        let written = write_upload_matrices(&mut buffer, &uploads).unwrap();
        let (decoded, payload_len) = read_upload_matrices(&mut Cursor::new(&buffer)).unwrap();
        prop_assert_eq!(decoded, uploads);
        prop_assert_eq!(payload_len, written - HEADER_BYTES);
    }

    #[test]
    fn upload_accepted_round_trip(identifiers in accepted_identifiers_strategy()) {
        let mut buffer = Vec::new();
        write_upload_accepted(&mut buffer, &identifiers).unwrap();
        let (decoded, _) = read_upload_accepted(&mut Cursor::new(&buffer)).unwrap();
        prop_assert_eq!(decoded, identifiers);
    }

    #[test]
    fn evaluate_round_trip(entries in evaluate_entries_strategy()) {
        let mut buffer = Vec::new();
        let written = write_evaluate(&mut buffer, &entries).unwrap();
        let (decoded, payload_len) = read_evaluate(&mut Cursor::new(&buffer)).unwrap();
        prop_assert_eq!(decoded, entries);
        prop_assert_eq!(payload_len, written - HEADER_BYTES);
    }

    #[test]
    fn products_round_trip(entries in products_strategy()) {
        let mut buffer = Vec::new();
        let written = write_products(&mut buffer, &entries).unwrap();
        let (decoded, payload_len) = read_products(&mut Cursor::new(&buffer)).unwrap();
        prop_assert_eq!(decoded, entries);
        prop_assert_eq!(payload_len, written - HEADER_BYTES);
    }

    #[test]
    fn error_round_trip(code in error_code_strategy(), message in diagnostic_strategy()) {
        let mut buffer = Vec::new();
        write_error(&mut buffer, code, &message).unwrap();
        let (decoded, _) = read_error(&mut Cursor::new(&buffer)).unwrap();
        prop_assert_eq!(decoded.code, code.to_u32());
        prop_assert_eq!(decoded.message, message);
    }

    #[test]
    fn truncated_upload_frames_always_fail(uploads in uploads_strategy()) {
        let mut buffer = Vec::new();
        write_upload_matrices(&mut buffer, &uploads).unwrap();
        let mut read = |bytes: &[u8]| read_upload_matrices(&mut Cursor::new(bytes)).map(|_| ());
        truncated_prefixes_always_fail(&buffer, &mut read);
    }

    #[test]
    fn truncated_evaluate_frames_always_fail(entries in evaluate_entries_strategy()) {
        let mut buffer = Vec::new();
        write_evaluate(&mut buffer, &entries).unwrap();
        let mut read = |bytes: &[u8]| read_evaluate(&mut Cursor::new(bytes)).map(|_| ());
        truncated_prefixes_always_fail(&buffer, &mut read);
    }

    #[test]
    fn truncated_products_frames_always_fail(entries in products_strategy()) {
        let mut buffer = Vec::new();
        write_products(&mut buffer, &entries).unwrap();
        let mut read = |bytes: &[u8]| read_products(&mut Cursor::new(bytes)).map(|_| ());
        truncated_prefixes_always_fail(&buffer, &mut read);
    }

    #[test]
    fn truncated_error_frames_always_fail(code in error_code_strategy(), message in diagnostic_strategy()) {
        let mut buffer = Vec::new();
        write_error(&mut buffer, code, &message).unwrap();
        let mut read = |bytes: &[u8]| read_error(&mut Cursor::new(bytes)).map(|_| ());
        truncated_prefixes_always_fail(&buffer, &mut read);
    }

    #[test]
    fn truncated_multi_frame_sequences_fail_after_the_prefix(
        entries in evaluate_entries_strategy(),
        products in products_strategy(),
    ) {
        let mut stream = Vec::new();
        write_evaluate(&mut stream, &entries).unwrap();
        let prefix = stream.len();
        write_products(&mut stream, &products).unwrap();
        // The first frame always decodes; every cut inside the second
        // frame must fail its read.
        for cut in prefix..stream.len() {
            let truncated = stream[..cut].to_vec();
            let mut cursor = Cursor::new(&truncated);
            read_evaluate(&mut cursor).unwrap();
            assert!(
                read_products(&mut cursor).is_err(),
                "a products frame cut at byte {cut} of {} decoded",
                stream.len()
            );
        }
    }

    #[test]
    fn short_multi_frame_sequences_round_trip(
        uploads in uploads_strategy(),
        identifiers in accepted_identifiers_strategy(),
        entries in evaluate_entries_strategy(),
        products in products_strategy(),
        code in error_code_strategy(),
        message in diagnostic_strategy(),
    ) {
        let mut stream = Vec::new();
        write_upload_matrices(&mut stream, &uploads).unwrap();
        write_upload_accepted(&mut stream, &identifiers).unwrap();
        write_evaluate(&mut stream, &entries).unwrap();
        write_products(&mut stream, &products).unwrap();
        write_error(&mut stream, code, &message).unwrap();

        let mut cursor = Cursor::new(&stream);
        let (decoded_uploads, _) = read_upload_matrices(&mut cursor).unwrap();
        prop_assert_eq!(decoded_uploads, uploads);
        let (decoded_ids, _) = read_upload_accepted(&mut cursor).unwrap();
        prop_assert_eq!(decoded_ids, identifiers);
        let (decoded_entries, _) = read_evaluate(&mut cursor).unwrap();
        prop_assert_eq!(decoded_entries, entries);
        let (decoded_products, _) = read_products(&mut cursor).unwrap();
        prop_assert_eq!(decoded_products, products);
        let (decoded_error, _) = read_error(&mut cursor).unwrap();
        prop_assert_eq!(decoded_error.code, code.to_u32());
        prop_assert_eq!(decoded_error.message, message);
        assert_eq!(cursor.position(), stream.len() as u64);
    }

    #[test]
    fn inflated_query_counts_contradict_the_summary(
        matrix_id in any::<u64>(),
        claimed in 2_u64..=5_u64,
    ) {
        let mut buffer = Vec::new();
        write_evaluate(&mut buffer, &[one_query_entry(matrix_id)]).unwrap();
        // The entry's query count trails the header, the list count, the
        // total query count, and the matrix identifier. The inflated count
        // still fits the descriptor table, so the summary cross-check is
        // what rejects it.
        let query_count_offset = HEADER_BYTES as usize + 8 + 8 + 8;
        buffer[query_count_offset..query_count_offset + 8].copy_from_slice(&claimed.to_le_bytes());
        prop_assert!(
            matches!(
                read_evaluate(&mut Cursor::new(&buffer)),
                Err(CodecError::CountMismatch {
                    name: "query count",
                    expected: 1,
                    actual
                }) if actual == claimed
            ),
            "an inflated query count must contradict the summary total"
        );
    }

    #[test]
    fn inflated_answer_counts_contradict_the_summary(
        matrix_id in any::<u64>(),
        claimed in 2_u64..=5_u64,
    ) {
        let mut buffer = Vec::new();
        write_products(&mut buffer, &[one_answer_entry(matrix_id)]).unwrap();
        // The entry's answer count trails the header, the list count, the
        // total answer count, the matrix identifier, the instance
        // identifier, rows, and blocks.
        let answer_count_offset = HEADER_BYTES as usize + 8 + 8 + 8 + 16 + 8 + 8;
        buffer[answer_count_offset..answer_count_offset + 8].copy_from_slice(&claimed.to_le_bytes());
        prop_assert!(
            matches!(
                read_products(&mut Cursor::new(&buffer)),
                Err(CodecError::CountMismatch {
                    name: "answer count",
                    expected: 1,
                    actual
                }) if actual == claimed
            ),
            "an inflated answer count must contradict the summary total"
        );
    }

    #[test]
    fn inflated_summary_query_counts_contradict_the_entries(
        matrix_id in any::<u64>(),
        claimed in 2_u64..=5_u64,
    ) {
        let mut buffer = Vec::new();
        write_evaluate(&mut buffer, &[one_query_entry(matrix_id)]).unwrap();
        // Inflating the summary total instead contradicts the unchanged
        // per-entry count; the larger query table still fits the payload.
        let total_offset = HEADER_BYTES as usize + 8;
        buffer[total_offset..total_offset + 8].copy_from_slice(&claimed.to_le_bytes());
        prop_assert!(
            matches!(
                read_evaluate(&mut Cursor::new(&buffer)),
                Err(CodecError::CountMismatch {
                    name: "query count",
                    expected,
                    actual: 1
                }) if expected == claimed
            ),
            "an inflated summary total must contradict the entries"
        );
    }

    #[test]
    fn inflated_upload_dimensions_break_the_value_budget(extra in 1_u64..=4_u64) {
        let mut buffer = Vec::new();
        write_upload_matrices(&mut buffer, &[small_upload()]).unwrap();
        // One record: k, ell, b (8 each), lambda (4), instance id (16),
        // then rows and columns (8 each). There is no declared value count
        // in v2: inflating the rows grows the derived value region beyond
        // the declared payload, which the plan must reject.
        let rows_offset = HEADER_BYTES as usize + 8 + 28 + 16;
        let declared = u64::from_le_bytes(
            buffer[rows_offset..rows_offset + 8].try_into().unwrap(),
        );
        buffer[rows_offset..rows_offset + 8].copy_from_slice(&(declared + extra).to_le_bytes());
        prop_assert!(
            matches!(
                read_upload_matrices(&mut Cursor::new(&buffer)),
                Err(CodecError::TruncatedFrame)
            ),
            "an inflated matrix dimension must break the value budget"
        );
    }

    #[test]
    fn inflated_answer_dimensions_break_the_value_budget(extra in 1_u64..=4_u64) {
        let mut buffer = Vec::new();
        write_products(&mut buffer, &[one_answer_entry(7)]).unwrap();
        // Per entry: matrix id (8), instance id (16), then rows (8). An
        // inflated row count grows every answer of the entry beyond the
        // declared payload.
        let rows_offset = HEADER_BYTES as usize + 8 + 8 + 8 + 16;
        let declared = u64::from_le_bytes(
            buffer[rows_offset..rows_offset + 8].try_into().unwrap(),
        );
        buffer[rows_offset..rows_offset + 8].copy_from_slice(&(declared + extra).to_le_bytes());
        prop_assert!(
            matches!(
                read_products(&mut Cursor::new(&buffer)),
                Err(CodecError::TruncatedFrame)
            ),
            "an inflated answer dimension must break the value budget"
        );
    }

    #[test]
    fn noncanonical_upload_words_are_rejected(word in PROTOCOL_MODULUS..=u32::MAX) {
        let mut buffer = Vec::new();
        write_upload_matrices(&mut buffer, &[small_upload()]).unwrap();
        // The first encoded element trails the header, the list count,
        // and the 60-byte descriptor.
        let first_element = HEADER_BYTES as usize + 8 + 60;
        buffer[first_element..first_element + 4].copy_from_slice(&word.to_le_bytes());
        prop_assert!(
            matches!(
                read_upload_matrices(&mut Cursor::new(&buffer)),
                Err(CodecError::NonCanonicalField { value }) if value == word
            ),
            "a word at or above the modulus must be rejected"
        );
    }

    #[test]
    fn noncanonical_evaluate_words_are_rejected(
        instance_id in any::<u128>(),
        query_id in any::<u64>(),
        word in PROTOCOL_MODULUS..=u32::MAX,
    ) {
        let mut buffer = Vec::new();
        let query = EncryptedQuery::from_parts(instance_id, query_id, values(1));
        write_evaluate(
            &mut buffer,
            &[EvaluateEntry {
                matrix_id: 1,
                queries: vec![query],
            }],
        )
        .unwrap();
        let first_element = HEADER_BYTES as usize + 16 + 24 + 24;
        buffer[first_element..first_element + 4].copy_from_slice(&word.to_le_bytes());
        prop_assert!(
            matches!(
                read_evaluate(&mut Cursor::new(&buffer)),
                Err(CodecError::NonCanonicalField { value }) if value == word
            ),
            "a word at or above the modulus must be rejected"
        );
    }

    #[test]
    fn unknown_frame_kinds_are_reported(kind in 6_u8..=u8::MAX) {
        let crafted = [kind, 0, 0, 0, 0, 0, 0, 0, 0];
        prop_assert!(
            matches!(
                read_frame_header(&mut Cursor::new(&crafted)),
                Err(CodecError::UnknownFrameKind { .. })
            ),
            "a kind byte outside the protocol must be rejected"
        );
        prop_assert_eq!(FrameKind::from_u8(kind), None);
    }
}

#[test]
fn deflated_upload_dimensions_leave_trailing_bytes() {
    let mut buffer = Vec::new();
    write_upload_matrices(&mut buffer, &[upload_2x2()]).unwrap();
    // Deflating the rows from 2 to 1 halves the derived value region from
    // 16 payload bytes to 8, leaving 8 trailing.
    let rows_offset = HEADER_BYTES as usize + 8 + 28 + 16;
    buffer[rows_offset..rows_offset + 8].copy_from_slice(&1_u64.to_le_bytes());
    assert!(matches!(
        read_upload_matrices(&mut Cursor::new(&buffer)),
        Err(CodecError::TrailingFrameBytes { extra: 8 })
    ));
}

/// Every proper prefix of `frame` must fail `read`, without panicking.
fn truncated_prefixes_always_fail(
    frame: &[u8],
    read: &mut dyn FnMut(&[u8]) -> Result<(), CodecError>,
) {
    for cut in 0..frame.len() {
        let outcome = read(&frame[..cut]);
        assert!(
            outcome.is_err(),
            "a {cut}-byte prefix of a {}-byte frame decoded successfully",
            frame.len()
        );
    }
}

const fn field() -> PrimeField<PROTOCOL_MODULUS> {
    PrimeField::<PROTOCOL_MODULUS>::new()
}

fn values(count: usize) -> Vec<FieldElement<PROTOCOL_MODULUS>> {
    (0..count)
        .map(|index| field().element_u32((index % 7 + 1) as u32))
        .collect()
}

fn field_strategy() -> impl Strategy<Value = FieldElement<PROTOCOL_MODULUS>> {
    (0_u32..PROTOCOL_MODULUS).prop_map(|value| field().element_u32(value))
}

/// Parameters the codec round-trips verbatim; the codec validates none of
/// their relationships (that is the server's job), so the property
/// generates arbitrary small shapes.
fn params_strategy() -> impl Strategy<Value = EmvpParams> {
    (1_usize..=32, 1_usize..=32, 2_usize..=16, any::<u32>()).prop_map(|(k, ell, b, lambda)| {
        EmvpParams {
            k,
            ell: ell.min(k),
            b,
            lambda,
        }
    })
}

fn uploads_strategy() -> impl Strategy<Value = Vec<MatrixUpload>> {
    prop::collection::vec(upload_strategy(), 0..=2)
}

fn upload_strategy() -> impl Strategy<Value = MatrixUpload> {
    (1_usize..=4, 1_usize..=8).prop_flat_map(|(rows, columns)| {
        (
            params_strategy(),
            any::<u128>(),
            prop::collection::vec(field_strategy(), rows * columns),
        )
            .prop_map(move |(params, instance_id, values)| MatrixUpload {
                params,
                matrix: EncryptedMatrix::from_parts(instance_id, rows, columns, values).unwrap(),
            })
    })
}

/// One 1x1 upload, keeping the patched-byte fixtures tiny.
fn small_upload() -> MatrixUpload {
    MatrixUpload {
        params: EmvpParams {
            k: 4,
            ell: 4,
            b: 2,
            lambda: 7,
        },
        matrix: EncryptedMatrix::from_parts(42, 1, 1, values(1)).unwrap(),
    }
}

/// One 2x2 upload, for value-budget arithmetic with headroom to deflate.
fn upload_2x2() -> MatrixUpload {
    MatrixUpload {
        params: EmvpParams {
            k: 4,
            ell: 4,
            b: 2,
            lambda: 7,
        },
        matrix: EncryptedMatrix::from_parts(42, 2, 2, values(4)).unwrap(),
    }
}

/// The server only ever acknowledges one-based identifiers.
fn accepted_identifiers_strategy() -> impl Strategy<Value = Vec<u64>> {
    prop::collection::vec(1_u64..=u64::MAX, 0..=4)
}

/// v2 gives one entry descriptor a single `query_width`, so every query
/// of an entry carries exactly that many coordinates (and a zero-width
/// entry carries none at all).
fn evaluate_entries_strategy() -> impl Strategy<Value = Vec<EvaluateEntry>> {
    prop::collection::vec(
        (any::<u64>(), 0_usize..=8).prop_flat_map(|(matrix_id, query_width)| {
            (
                Just(matrix_id),
                prop::collection::vec(
                    (
                        any::<u128>(),
                        any::<u64>(),
                        prop::collection::vec(field_strategy(), query_width),
                    )
                        .prop_map(move |(instance_id, query_id, values)| {
                            EncryptedQuery::from_parts(instance_id, query_id, values)
                        }),
                    if query_width == 0 {
                        0..=0_usize
                    } else {
                        0..=2_usize
                    },
                ),
            )
                .prop_map(move |(matrix_id, queries)| EvaluateEntry { matrix_id, queries })
        }),
        0..=2,
    )
}

/// v2 carries a single instance identifier per products entry, so every
/// answer of an entry shares it; only the query identifiers vary.
fn products_strategy() -> impl Strategy<Value = Vec<ProductEntry>> {
    prop::collection::vec(
        (any::<u64>(), 1_usize..=4, 1_usize..=4, any::<u128>()).prop_flat_map(
            move |(matrix_id, rows, blocks, instance_id)| {
                prop::collection::vec(
                    (any::<u64>(), Just(rows * blocks)).prop_flat_map(move |(query_id, words)| {
                        prop::collection::vec(field_strategy(), words).prop_map(move |values| {
                            AnswerMatrix::from_parts(instance_id, query_id, values, rows, blocks)
                        })
                    }),
                    1..=2,
                )
                .prop_map(move |answers| ProductEntry { matrix_id, answers })
            },
        ),
        0..=2,
    )
}

fn diagnostic_strategy() -> impl Strategy<Value = String> {
    prop::collection::vec(prop::char::range(' ', '~'), 0..=48)
        .prop_map(|characters| characters.into_iter().collect())
}

fn error_code_strategy() -> impl Strategy<Value = ErrorCode> {
    prop::sample::select(&ALL_ERROR_CODES[..])
}

fn one_query_entry(matrix_id: u64) -> EvaluateEntry {
    EvaluateEntry {
        matrix_id,
        queries: vec![EncryptedQuery::from_parts(42, 0, values(40))],
    }
}

fn one_answer_entry(matrix_id: u64) -> ProductEntry {
    ProductEntry {
        matrix_id,
        answers: vec![AnswerMatrix::from_parts(42, 3, values(8), 4, 2)],
    }
}
