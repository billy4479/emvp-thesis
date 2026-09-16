//! One client connection: the handshake, the load-once session state
//! machine, and request validation and evaluation.
//!
//! The session phase is explicitly `Empty` or `Loaded`: the one permitted
//! matrix-set upload moves `Empty` to `Loaded` atomically. Upload failures
//! are atomic too: the engine validates the whole set, reserves its entire
//! device residency with one budget transition, and only then uploads, so a
//! failure releases the whole reservation and returns no handles — no
//! identifiers are assigned, and the connection closes so the client can
//! retry with a fresh session.
//! Both state violations — a second upload on a loaded session, and an
//! evaluation before any upload — are answered with the structured error
//! and close the session immediately, without the offending payload being
//! read or drained. Request-validation failures (malformed frames,
//! rejected parameters, unknown identifiers) are answered in-band; a
//! validation failure of an evaluation leaves the session usable, while
//! an engine failure after registration closes the session and releases
//! its matrices. Every Evaluate frame is answered with a single
//! [`AnswerEngine::answer_many_with_report`] call, whose report feeds the
//! per-frame timing logs.

use std::collections::{HashMap, HashSet};
use std::fmt;

use emvp::{
    AnswerEngine, AnswerEngineError, AnswerJob, AnswerReport, EmvpParams, EncryptedQuery,
    PrepareBatchError, PrepareMatrixError, PreparedMatrix,
};
use emvp_network::{
    CodecError, ErrorCode, EvaluateEntry, FrameHeader, FrameKind, FrameReader, HandshakeError,
    MatrixUpload, PROTOCOL_MODULUS, ProductEntry, read_evaluate_payload, read_frame_header,
    read_upload_matrices_payload, server_handshake, write_error, write_products,
    write_upload_accepted,
};

use crate::failure::RequestFailure;

/// How a session continues after one frame.
enum Outcome {
    /// The connection stays open for the next frame.
    Continue,
    /// The connection closes; the optional error is the session's fate.
    Close(Option<SessionError>),
}

/// Why a session ended abnormally.
#[derive(Debug)]
#[non_exhaustive]
pub enum SessionError {
    /// The version and modulus handshake failed.
    Handshake(HandshakeError),
    /// The transport or a frame failed mid-session.
    Transport(CodecError),
    /// A matrix upload failed inside the answer engine after the request
    /// had passed prevalidation.
    Upload(PrepareMatrixError),
    /// An evaluation failed inside the answer engine.
    Engine(AnswerEngineError),
}

impl fmt::Display for SessionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Handshake(error) => error.fmt(formatter),
            Self::Transport(error) => error.fmt(formatter),
            Self::Upload(error) => error.fmt(formatter),
            Self::Engine(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for SessionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Handshake(error) => Some(error),
            Self::Transport(error) => Some(error),
            Self::Upload(error) => Some(error),
            Self::Engine(error) => Some(error),
        }
    }
}

/// Serves one connection until the client disconnects or the session fails.
///
/// The stream must be a fresh connection; the handshake runs first and a
/// rejection closes it. `Ok` is a clean client disconnect, `Err` the
/// reason an abnormal session ended; in both cases every resource the
/// session held has been released.
///
/// # Errors
///
/// Returns the handshake, transport, or engine failure that ended the
/// session. Protocol-level rejections are answered in-band; state
/// violations and unknown frame kinds are answered and close the
/// connection without surfacing here.
pub fn serve_connection<S>(
    stream: &mut S,
    engine: &AnswerEngine<PROTOCOL_MODULUS>,
) -> Result<(), SessionError>
where
    S: std::io::Read + std::io::Write,
{
    server_handshake(stream).map_err(SessionError::Handshake)?;
    let mut session = Session::new(engine);
    loop {
        let header = match read_frame_header(stream) {
            Ok(Some(header)) => header,
            Ok(None) => return Ok(()),
            // The header byte and length are already consumed, and the
            // session is closing, so the payload is never read.
            Err(CodecError::UnknownFrameKind { kind }) => {
                send_error(
                    stream,
                    ErrorCode::UnknownFrameKind,
                    &format!("unknown frame kind {kind}"),
                );
                return Err(SessionError::Transport(CodecError::UnknownFrameKind {
                    kind,
                }));
            }
            Err(error) => return Err(SessionError::Transport(error)),
        };
        match session.handle_frame(stream, header) {
            Outcome::Continue => {}
            Outcome::Close(error) => return error.map_or(Ok(()), Err),
        }
    }
}

/// The session-side identity of one stored encrypted matrix.
#[derive(Clone, Copy, Debug)]
struct MatrixInfo {
    /// The public instance identifier the matrix was encrypted with.
    instance_id: u128,
    /// The matrix width (`n = 2k`), which every query must match.
    width: usize,
}

/// One uploaded matrix: the engine-resident handle plus the metadata
/// request prevalidation consults before any engine work.
struct StoredMatrix {
    info: MatrixInfo,
    prepared: PreparedMatrix<PROTOCOL_MODULUS>,
}

/// The explicit phase of the load-once session state machine.
enum Phase {
    /// No matrix set has been loaded on this connection.
    Empty,
    /// The one permitted upload succeeded; matrices are keyed by the
    /// sequential identifiers `1..=len` assigned at upload time.
    Loaded {
        /// The stored matrices by assigned identifier.
        matrices: HashMap<u64, StoredMatrix>,
    },
}

impl Phase {
    /// The stored matrices, if the session has loaded its matrix set.
    const fn matrices(&self) -> Option<&HashMap<u64, StoredMatrix>> {
        match self {
            Self::Empty => None,
            Self::Loaded { matrices } => Some(matrices),
        }
    }
}

/// The mutable state of one connection.
struct Session<'a> {
    engine: &'a AnswerEngine<PROTOCOL_MODULUS>,
    phase: Phase,
}

impl Session<'_> {
    const fn new(engine: &AnswerEngine<PROTOCOL_MODULUS>) -> Session<'_> {
        Session {
            engine,
            phase: Phase::Empty,
        }
    }

    /// The stored matrix metadata for an assigned identifier.
    fn matrix_info(&self, matrix_id: u64) -> Option<MatrixInfo> {
        self.phase.matrices()?.get(&matrix_id).map(|m| m.info)
    }

    /// The stored engine handle for an assigned identifier.
    fn matrix(&self, matrix_id: u64) -> Option<&PreparedMatrix<PROTOCOL_MODULUS>> {
        self.phase.matrices()?.get(&matrix_id).map(|m| &m.prepared)
    }

    fn handle_frame<S: std::io::Read + std::io::Write>(
        &mut self,
        stream: &mut S,
        header: FrameHeader,
    ) -> Outcome {
        match header.kind {
            FrameKind::UploadMatrices => self.handle_upload(stream, header.payload_len),
            FrameKind::Evaluate => self.handle_evaluate(stream, header.payload_len),
            FrameKind::UploadAccepted | FrameKind::Products | FrameKind::Error => {
                // Those kinds flow server to client only; a client sending
                // one is speaking the wrong half of the protocol.
                let message = format!("unexpected {} frame from the client", header.kind);
                send_error(stream, ErrorCode::UnexpectedFrame, &message);
                Outcome::Close(Some(SessionError::Transport(CodecError::UnexpectedFrame {
                    expected: FrameKind::UploadMatrices,
                    actual: header.kind.to_u8(),
                })))
            }
        }
    }

    fn handle_upload<S: std::io::Read + std::io::Write>(
        &mut self,
        stream: &mut S,
        payload_len: u64,
    ) -> Outcome {
        // A state violation is fatal: the error is sent without reading
        // or draining the payload, and the session closes immediately.
        if matches!(self.phase, Phase::Loaded { .. }) {
            send_error(
                stream,
                ErrorCode::AlreadyLoaded,
                "this session already loaded a matrix set",
            );
            return Outcome::Close(None);
        }
        let mut frame = FrameReader::new(stream, payload_len);
        let uploads = match read_upload_matrices_payload(&mut frame)
            .and_then(|uploads| frame.finish().map(|()| uploads))
        {
            Ok(uploads) => uploads,
            Err(error) => {
                let message = error.to_string();
                send_error(stream, error.error_code(), &message);
                return Outcome::Close(Some(SessionError::Transport(error)));
            }
        };
        // Every failure past this point is an atomic upload failure: the
        // engine validates the whole set, reserves its entire residency
        // with one budget transition, and only then uploads — so a failure
        // releases the whole reservation, no identifier is assigned, no
        // handle is returned, and the connection closes so the client can
        // retry with a fresh session.
        let total_bytes = match validate_uploads(&uploads) {
            Ok(bytes) => bytes,
            Err(failure) => {
                send_error(stream, failure.code(), &failure.to_string());
                return Outcome::Close(None);
            }
        };
        let infos: Vec<MatrixInfo> = uploads
            .iter()
            .map(|upload| MatrixInfo {
                instance_id: upload.matrix.instance_id(),
                width: upload.matrix.columns(),
            })
            .collect();
        let prepared = match self.engine.prepare_batch(
            uploads
                .into_iter()
                .map(|MatrixUpload { params, matrix }| (params, matrix)),
        ) {
            Ok(prepared) => prepared,
            Err(error) => {
                let message = error.to_string();
                send_error(stream, ErrorCode::UploadFailed, &message);
                return match error {
                    PrepareBatchError::Budget(_) => Outcome::Close(None),
                    error => Outcome::Close(Some(SessionError::Upload(error.into()))),
                };
            }
        };
        // Identifiers are assigned once, consecutively from one, in input
        // order, so they map back onto the stored handles by key.
        let mut matrices = HashMap::with_capacity(prepared.len());
        for (index, (prepared, info)) in prepared.into_iter().zip(infos).enumerate() {
            matrices.insert(index as u64 + 1, StoredMatrix { info, prepared });
        }
        let identifiers: Vec<u64> = (1..=matrices.len() as u64).collect();
        eprintln!(
            "server: loaded {} matrices ({} bytes of engine matrix residency)",
            identifiers.len(),
            total_bytes
        );
        self.phase = Phase::Loaded { matrices };
        match write_upload_accepted(stream, &identifiers) {
            Ok(_) => Outcome::Continue,
            Err(error) => Outcome::Close(Some(SessionError::Transport(error))),
        }
    }

    fn handle_evaluate<S: std::io::Read + std::io::Write>(
        &self,
        stream: &mut S,
        payload_len: u64,
    ) -> Outcome {
        // A state violation is fatal: the error is sent without reading
        // or draining the payload, and the session closes immediately.
        if self.phase.matrices().is_none() {
            send_error(
                stream,
                ErrorCode::NotLoaded,
                "no matrix set has been loaded on this connection",
            );
            return Outcome::Close(None);
        }
        let mut frame = FrameReader::new(stream, payload_len);
        let entries = match read_evaluate_payload(&mut frame)
            .and_then(|entries| frame.finish().map(|()| entries))
        {
            Ok(entries) => entries,
            Err(error) => {
                let message = error.to_string();
                send_error(stream, error.error_code(), &message);
                return Outcome::Close(Some(SessionError::Transport(error)));
            }
        };
        let prepared = match prepare_evaluate(entries, &mut |matrix_id| self.matrix_info(matrix_id))
        {
            Ok(prepared) => prepared,
            Err(failure) => {
                let code = failure.code();
                send_error(stream, code, &failure.to_string());
                return Outcome::Continue;
            }
        };
        let mut jobs = Vec::with_capacity(prepared.len());
        for (matrix_id, queries) in &prepared {
            // Validation pinned every identifier to a stored matrix, and
            // nothing mutates the phase during evaluation.
            let Some(matrix) = self.matrix(*matrix_id) else {
                send_error(
                    stream,
                    ErrorCode::Internal,
                    "a validated matrix identifier disappeared",
                );
                return Outcome::Close(None);
            };
            jobs.push(AnswerJob { matrix, queries });
        }
        // One engine call answers the whole frame; the engine validates,
        // plans, and computes every entry and returns no partial output
        // on failure.
        let (answers, report) = match self.engine.answer_many_with_report(&jobs) {
            Ok(outcome) => outcome,
            Err(error) => {
                send_error(stream, ErrorCode::GpuFailure, &error.to_string());
                return Outcome::Close(Some(SessionError::Engine(error)));
            }
        };
        log_report(&report, &prepared);
        let mut products = Vec::with_capacity(prepared.len());
        for ((matrix_id, _), answers) in prepared.iter().zip(answers) {
            products.push(ProductEntry {
                matrix_id: *matrix_id,
                answers,
            });
        }
        match write_products(stream, &products) {
            Ok(_) => Outcome::Continue,
            Err(error) => Outcome::Close(Some(SessionError::Transport(error))),
        }
    }
}

/// Writes an `Error` frame; a failed write means the transport is already
/// gone, which the caller handles by closing.
fn send_error<S: std::io::Write>(stream: &mut S, code: ErrorCode, message: &str) {
    if let Err(error) = write_error(stream, code, message) {
        eprintln!("server: failed to deliver an error response: {error}");
    }
}

/// Logs the aggregate timing breakdown of one answered evaluation frame
/// and one line per entry with its backend and shape statistics.
fn log_report(report: &AnswerReport, prepared: &[(u64, Vec<EncryptedQuery<PROTOCOL_MODULUS>>)]) {
    let queries: usize = report.entries.iter().map(|entry| entry.queries).sum();
    eprintln!(
        "server: answered {} queries across {} matrices in {} ms (planning {} ms, cpu {} ms, gpu buffers {} ms, upload {} ms, submit {} ms, readback {} ms, reconstruct {} ms)",
        queries,
        report.entries.len(),
        report.total.as_millis(),
        report.planning.as_millis(),
        report.cpu_compute.as_millis(),
        report.gpu_prepare_buffers.as_millis(),
        report.gpu_upload.as_millis(),
        report.gpu_submit.as_millis(),
        report.gpu_wait.as_millis(),
        report.gpu_reconstruct.as_millis(),
    );
    for ((matrix_id, _), entry) in prepared.iter().zip(&report.entries) {
        eprintln!(
            "server: matrix {matrix_id}: {} queries via {:?} ({} multiplications)",
            entry.queries, entry.backend, entry.multiplications
        );
    }
}

/// Validates one matrix set for upload and totals its GPU byte footprint.
///
/// The set must be nonempty and is validated in full before any engine
/// work, so a rejection cannot leave a partial upload behind.
///
/// # Errors
///
/// Returns the first structural problem: an empty set, malformed
/// parameters, a matrix whose width disagrees with its parameters, or
/// arithmetic overflow while totaling the footprint.
pub fn validate_uploads(uploads: &[MatrixUpload]) -> Result<u64, RequestFailure> {
    if uploads.is_empty() {
        return Err(RequestFailure::EmptyRequest);
    }
    let mut total_bytes = 0_u64;
    for upload in uploads {
        validate_params(&upload.params)?;
        let width = upload.params.n().map_err(RequestFailure::Params)?;
        let columns = upload.matrix.columns();
        if columns != width {
            return Err(RequestFailure::ColumnsMismatch {
                expected: width,
                actual: columns,
            });
        }
        let words = upload
            .matrix
            .rows()
            .checked_mul(columns)
            .ok_or(RequestFailure::DimensionOverflow)?;
        let bytes = u64::try_from(words)
            .ok()
            .and_then(|word_count| word_count.checked_mul(4))
            .ok_or(RequestFailure::DimensionOverflow)?;
        total_bytes = total_bytes
            .checked_add(bytes)
            .ok_or(RequestFailure::DimensionOverflow)?;
    }
    Ok(total_bytes)
}

/// Validates uploaded protocol parameters.
fn validate_params(params: &EmvpParams) -> Result<(), RequestFailure> {
    params.validate_dimensions().map_err(RequestFailure::Params)
}

/// Validates an evaluation request in full and flattens it into one
/// `(matrix id, queries)` pair per entry, in request order.
///
/// Every entry must reference a distinct loaded matrix, carry at least one
/// query, and every query must match its matrix's width, instance
/// identifier, and carry a query identifier unique within its entry. The
/// whole request is checked before any engine work begins.
///
/// # Errors
///
/// Returns the first validation failure.
fn prepare_evaluate(
    entries: Vec<EvaluateEntry>,
    lookup: &mut dyn FnMut(u64) -> Option<MatrixInfo>,
) -> Result<Vec<(u64, Vec<EncryptedQuery<PROTOCOL_MODULUS>>)>, RequestFailure> {
    if entries.is_empty() {
        return Err(RequestFailure::EmptyRequest);
    }
    let mut matrix_ids = HashSet::with_capacity(entries.len());
    let mut prepared = Vec::with_capacity(entries.len());
    for entry in entries {
        let Some(matrix) = lookup(entry.matrix_id) else {
            return Err(RequestFailure::UnknownMatrix {
                id: entry.matrix_id,
            });
        };
        if !matrix_ids.insert(entry.matrix_id) {
            return Err(RequestFailure::DuplicateMatrix {
                id: entry.matrix_id,
            });
        }
        if entry.queries.is_empty() {
            return Err(RequestFailure::EmptyRequest);
        }
        let mut query_ids = HashSet::with_capacity(entry.queries.len());
        for query in &entry.queries {
            let width = query.values().len();
            if width != matrix.width {
                return Err(RequestFailure::QueryWidth {
                    expected: matrix.width,
                    actual: width,
                });
            }
            if query.instance_id() != matrix.instance_id {
                return Err(RequestFailure::InstanceMismatch {
                    expected: matrix.instance_id,
                    actual: query.instance_id(),
                });
            }
            if !query_ids.insert(query.query_id()) {
                return Err(RequestFailure::DuplicateQuery {
                    matrix_id: entry.matrix_id,
                    query_id: query.query_id(),
                });
            }
        }
        prepared.push((entry.matrix_id, entry.queries));
    }
    Ok(prepared)
}

#[cfg(test)]
mod tests {
    use std::io::Write;
    use std::os::unix::net::UnixStream;
    use std::sync::Arc;
    use std::thread;

    use emvp::{
        AnswerEngine, AnswerMatrix, DecodingKey, DerivedState, EmvpParams, EncryptedQuery,
        MaskContextId, ProtocolError, SecretKey, encrypt, query,
    };
    use emvp_network::{
        ErrorCode, EvaluateEntry, MatrixUpload, PROTOCOL_MODULUS, ProductEntry, client_handshake,
        read_error, read_frame_header, read_products, read_upload_accepted, write_evaluate,
        write_upload_matrices,
    };
    use prime_field_layer::{FieldElement, PrimeField};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;
    use trapdoor_matrices::ToeplitzFastProduct;

    use super::{
        RequestFailure, SessionError, prepare_evaluate, serve_connection, validate_uploads,
    };

    const PARAMS: EmvpParams = EmvpParams {
        k: 8,
        ell: 8,
        b: 2,
        lambda: 7,
    };

    const CONTEXT: MaskContextId = MaskContextId::from_u64(0x0100);

    const fn field() -> PrimeField<PROTOCOL_MODULUS> {
        PrimeField::<PROTOCOL_MODULUS>::new()
    }

    fn values(count: usize, seed: u64) -> Vec<FieldElement<PROTOCOL_MODULUS>> {
        let field = field();
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        (0..count).map(|_| field.sample_uniform(&mut rng)).collect()
    }

    fn toeplitz_block(
        stream: &mut ChaCha20Rng,
        _index: usize,
    ) -> Result<ToeplitzFastProduct<PROTOCOL_MODULUS>, ProtocolError> {
        Ok(ToeplitzFastProduct::sample(2 * PARAMS.k, stream)?)
    }

    fn upload(
        rows: usize,
        key: u8,
        seed: u64,
    ) -> (
        MatrixUpload,
        DerivedState<PROTOCOL_MODULUS, ToeplitzFastProduct<PROTOCOL_MODULUS>>,
        Vec<FieldElement<PROTOCOL_MODULUS>>,
    ) {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut state = SecretKey::<PROTOCOL_MODULUS>::new_insecure(PARAMS, [key; 32])
            .unwrap()
            .derive(CONTEXT, rows, &mut rng, toeplitz_block)
            .unwrap();
        let plaintext = values(rows * PARAMS.ell, seed ^ 0xFF);
        let matrix = encrypt(&mut state, &plaintext).unwrap();
        (
            MatrixUpload {
                params: PARAMS,
                matrix,
            },
            state,
            plaintext,
        )
    }

    #[test]
    fn uploads_validate_and_total_their_footprint() {
        let (small, _, _) = upload(3, 1, 1);
        let (tall, _, _) = upload(35, 2, 2);
        let bytes = validate_uploads(&[small, tall]).unwrap();
        assert_eq!(bytes, (3 * 16 + 35 * 16) * 4);
    }

    #[test]
    fn uploads_reject_malformed_dimensions() {
        let (mut malformed, _, _) = upload(3, 1, 3);
        malformed.params.b = 3;
        assert!(matches!(
            validate_uploads(&[malformed]),
            Err(RequestFailure::Params(_))
        ));

        let (good, _, _) = upload(3, 1, 4);
        let matrix = emvp::EncryptedMatrix::from_parts(1, 3, 15, values(45, 5)).unwrap();
        let uploads = [MatrixUpload {
            params: good.params,
            matrix,
        }];
        assert!(matches!(
            validate_uploads(&uploads),
            Err(RequestFailure::ColumnsMismatch {
                expected: 16,
                actual: 15
            })
        ));
    }

    #[test]
    fn validate_uploads_rejects_an_empty_set() {
        assert!(matches!(
            validate_uploads(&[]),
            Err(RequestFailure::EmptyRequest)
        ));
    }

    #[test]
    fn evaluate_preparation_enforces_the_request_contract() {
        let mut lookup = |matrix_id: u64| {
            if matrix_id == 7 {
                Some(super::MatrixInfo {
                    instance_id: 42,
                    width: 16,
                })
            } else {
                None
            }
        };
        let entry = |matrix_id: u64, queries: Vec<emvp::EncryptedQuery<PROTOCOL_MODULUS>>| {
            EvaluateEntry { matrix_id, queries }
        };
        let make_query = |query_id: u64, width: usize, instance_id: u128| {
            emvp::EncryptedQuery::from_parts(instance_id, query_id, values(width, query_id))
        };

        let ok = prepare_evaluate(
            vec![entry(7, vec![make_query(0, 16, 42), make_query(1, 16, 42)])],
            &mut lookup,
        )
        .unwrap();
        assert_eq!(ok.len(), 1);
        assert_eq!(ok[0].0, 7);
        assert_eq!(ok[0].1.len(), 2);

        assert!(matches!(
            prepare_evaluate(vec![], &mut lookup),
            Err(RequestFailure::EmptyRequest)
        ));
        assert!(matches!(
            prepare_evaluate(vec![entry(9, vec![make_query(0, 16, 42)])], &mut lookup),
            Err(RequestFailure::UnknownMatrix { id: 9 })
        ));
        assert!(matches!(
            prepare_evaluate(
                vec![
                    entry(7, vec![make_query(0, 16, 42)]),
                    entry(7, vec![make_query(1, 16, 42)]),
                ],
                &mut lookup,
            ),
            Err(RequestFailure::DuplicateMatrix { id: 7 })
        ));
        assert!(matches!(
            prepare_evaluate(vec![entry(7, vec![make_query(3, 15, 42)])], &mut lookup,),
            Err(RequestFailure::QueryWidth {
                expected: 16,
                actual: 15
            })
        ));
        assert!(matches!(
            prepare_evaluate(vec![entry(7, vec![make_query(3, 16, 43)])], &mut lookup,),
            Err(RequestFailure::InstanceMismatch {
                expected: 42,
                actual: 43
            })
        ));
        assert!(matches!(
            prepare_evaluate(
                vec![entry(
                    7,
                    vec![make_query(5, 16, 42), make_query(5, 16, 42),]
                )],
                &mut lookup,
            ),
            Err(RequestFailure::DuplicateQuery {
                matrix_id: 7,
                query_id: 5
            })
        ));
        assert!(matches!(
            prepare_evaluate(vec![entry(7, vec![])], &mut lookup),
            Err(RequestFailure::EmptyRequest)
        ));
        // A zero identifier maps to no slot: one-based identifiers only.
        assert!(lookup(0).is_none());
        assert!(lookup(1).is_none());
    }

    fn spawn_server(
        server_stream: UnixStream,
        engine: Arc<AnswerEngine<PROTOCOL_MODULUS>>,
    ) -> thread::JoinHandle<Result<(), SessionError>> {
        thread::spawn(move || {
            let mut server_stream = server_stream;
            serve_connection(&mut server_stream, &engine)
        })
    }

    fn random_query(
        state: &mut DerivedState<PROTOCOL_MODULUS, ToeplitzFastProduct<PROTOCOL_MODULUS>>,
        seed: u64,
    ) -> (
        EncryptedQuery<PROTOCOL_MODULUS>,
        DecodingKey<PROTOCOL_MODULUS>,
        Vec<FieldElement<PROTOCOL_MODULUS>>,
    ) {
        let q = values(PARAMS.ell, seed);
        let (encrypted, key) = query(state, &q).unwrap();
        (encrypted, key, q)
    }

    fn naive_product(
        plaintext: &[FieldElement<PROTOCOL_MODULUS>],
        q: &[FieldElement<PROTOCOL_MODULUS>],
        rows: usize,
    ) -> Vec<FieldElement<PROTOCOL_MODULUS>> {
        let field = field();
        (0..rows)
            .map(|row| {
                let mut accumulator = field.element_u32(0);
                for (column, &coefficient) in q.iter().enumerate() {
                    accumulator += plaintext[row * PARAMS.ell + column] * coefficient;
                }
                accumulator
            })
            .collect()
    }

    fn decoded(
        answer: &AnswerMatrix<PROTOCOL_MODULUS>,
        key: &DecodingKey<PROTOCOL_MODULUS>,
    ) -> Vec<FieldElement<PROTOCOL_MODULUS>> {
        let mut output = vec![field().element_u32(0); answer.rows()];
        emvp::decode_into(answer, key, &mut output).unwrap();
        output
    }

    #[test]
    fn full_session_uploads_evaluates_and_answers_correctly() {
        let engine = Arc::new(AnswerEngine::cpu());
        let (mut client, server_stream) = UnixStream::pair().unwrap();
        let server = spawn_server(server_stream, engine);
        client_handshake(&mut client).unwrap();

        let (upload_a, mut state_a, plaintext_a) = upload(5, 0x11, 100);
        let (upload_b, mut state_b, plaintext_b) = upload(21, 0x22, 101);
        write_upload_matrices(&mut client, &[upload_a, upload_b]).unwrap();
        let (ids, _) = read_upload_accepted(&mut client).unwrap();
        assert_eq!(ids, vec![1, 2]);

        let (query_a0, key_a0, plain_a0) = random_query(&mut state_a, 200);
        let (query_a1, key_a1, plain_a1) = random_query(&mut state_a, 201);
        let (query_of_b, key_of_b, plain_b) = random_query(&mut state_b, 202);
        let entries = vec![
            EvaluateEntry {
                matrix_id: ids[0],
                queries: vec![query_a0, query_a1],
            },
            EvaluateEntry {
                matrix_id: ids[1],
                queries: vec![query_of_b],
            },
        ];
        write_evaluate(&mut client, &entries).unwrap();
        let (products, _) = read_products(&mut client).unwrap();
        assert_eq!(products.len(), 2);

        let ProductEntry { matrix_id, answers } = &products[0];
        assert_eq!(*matrix_id, ids[0]);
        assert_eq!(answers.len(), 2);
        for ((answer, key), plaintext_query) in answers
            .iter()
            .zip([&key_a0, &key_a1])
            .zip([&plain_a0, &plain_a1])
        {
            assert_eq!(answer.instance_id(), state_a.instance_id());
            assert_eq!(answer.query_id(), key.query_id());
            assert_eq!(answer.rows(), 5);
            assert_eq!(answer.blocks(), 8);
            let decoded_product = decoded(answer, key);
            assert_eq!(
                decoded_product,
                naive_product(&plaintext_a, plaintext_query, 5)
            );
        }
        let ProductEntry { matrix_id, answers } = &products[1];
        assert_eq!(*matrix_id, ids[1]);
        assert_eq!(answers.len(), 1);
        assert_eq!(answers[0].rows(), 21);
        let decoded_product = decoded(&answers[0], &key_of_b);
        assert_eq!(decoded_product, naive_product(&plaintext_b, &plain_b, 21));

        drop(client);
        server.join().unwrap().unwrap();
    }

    #[test]
    fn a_second_upload_is_rejected_and_closes_the_session() {
        let engine = Arc::new(AnswerEngine::cpu());
        let (mut client, server_stream) = UnixStream::pair().unwrap();
        let server = spawn_server(server_stream, engine);
        client_handshake(&mut client).unwrap();

        let (upload_a, _, _) = upload(5, 0x11, 300);
        write_upload_matrices(&mut client, &[upload_a]).unwrap();
        let (ids, _) = read_upload_accepted(&mut client).unwrap();
        assert_eq!(ids, vec![1]);

        // The server answers the state violation and closes immediately;
        // the second payload is never read or drained. The client sends
        // only the frame header, so the close carries no unread data and
        // the error frame is delivered deterministically.
        let (upload_b, _, _) = upload(7, 0x33, 301);
        let payload_len = emvp_network::write_upload_matrices(&mut Vec::new(), &[upload_b])
            .unwrap()
            - emvp_network::HEADER_BYTES;
        emvp_network::write_frame_header(
            &mut client,
            emvp_network::FrameKind::UploadMatrices,
            payload_len,
        )
        .unwrap();
        let (error, _) = read_error(&mut client).unwrap();
        assert_eq!(error.code(), Some(ErrorCode::AlreadyLoaded));
        assert!(read_frame_header(&mut client).unwrap().is_none());

        server.join().unwrap().unwrap();
    }

    #[test]
    fn a_failed_upload_is_atomic_and_closes_the_session() {
        let engine = Arc::new(AnswerEngine::cpu());
        let (mut client, server_stream) = UnixStream::pair().unwrap();
        let server = spawn_server(server_stream, engine);
        client_handshake(&mut client).unwrap();

        // `b = 3` does not divide `n = 16`: structurally invalid parameters.
        let (upload_a, _, _) = upload(5, 0x11, 400);
        let malformed = MatrixUpload {
            params: EmvpParams {
                b: 3,
                ..upload_a.params
            },
            matrix: upload_a.matrix,
        };
        write_upload_matrices(&mut client, &[malformed]).unwrap();
        let (error, _) = read_error(&mut client).unwrap();
        assert_eq!(error.code(), Some(ErrorCode::InvalidParameters));

        // The server closed the connection without assigning identifiers.
        assert!(read_frame_header(&mut client).unwrap().is_none());
        server.join().unwrap().unwrap();
    }

    #[test]
    fn an_empty_upload_set_is_rejected() {
        let engine = Arc::new(AnswerEngine::cpu());
        let (mut client, server_stream) = UnixStream::pair().unwrap();
        let server = spawn_server(server_stream, engine);
        client_handshake(&mut client).unwrap();

        write_upload_matrices(&mut client, &[]).unwrap();
        let (error, _) = read_error(&mut client).unwrap();
        assert_eq!(error.code(), Some(ErrorCode::EmptyRequest));
        assert!(read_frame_header(&mut client).unwrap().is_none());

        server.join().unwrap().unwrap();
    }

    #[test]
    fn evaluation_before_an_upload_is_answered_and_closes() {
        let engine = Arc::new(AnswerEngine::cpu());
        let (mut client, server_stream) = UnixStream::pair().unwrap();
        let server = spawn_server(server_stream, engine);
        client_handshake(&mut client).unwrap();

        let entry = EvaluateEntry {
            matrix_id: 1,
            queries: vec![emvp::EncryptedQuery::from_parts(1, 0, values(16, 600))],
        };
        // The client sends only the frame header, so the session closes
        // with no unread data and the error frame is delivered cleanly.
        let payload_len =
            emvp_network::write_evaluate(&mut Vec::new(), std::slice::from_ref(&entry)).unwrap()
                - emvp_network::HEADER_BYTES;
        emvp_network::write_frame_header(
            &mut client,
            emvp_network::FrameKind::Evaluate,
            payload_len,
        )
        .unwrap();
        let (error, _) = read_error(&mut client).unwrap();
        assert_eq!(error.code(), Some(ErrorCode::NotLoaded));

        // The session closed: the answer never leaves the connection
        // usable for a second request.
        assert!(read_frame_header(&mut client).unwrap().is_none());
        server.join().unwrap().unwrap();
    }

    #[test]
    fn server_directed_frames_close_the_session() {
        let engine = Arc::new(AnswerEngine::cpu());
        let (mut client, server_stream) = UnixStream::pair().unwrap();
        let server = spawn_server(server_stream, engine);
        client_handshake(&mut client).unwrap();

        // A client must never send server-to-client frames. The client sends
        // only the frame header, so the session closes with no unread data
        // and the error frame is delivered deterministically.
        let mut buffer = Vec::new();
        let written = emvp_network::write_products(
            &mut buffer,
            &[ProductEntry {
                matrix_id: 1,
                answers: vec![AnswerMatrix::from_parts(1, 0, values(8, 700), 4, 2)],
            }],
        )
        .unwrap();
        emvp_network::write_frame_header(
            &mut client,
            emvp_network::FrameKind::Products,
            written - emvp_network::HEADER_BYTES,
        )
        .unwrap();
        let (error, _) = read_error(&mut client).unwrap();
        assert_eq!(error.code(), Some(ErrorCode::UnexpectedFrame));

        // The server closed the connection after the violation.
        assert!(read_frame_header(&mut client).unwrap().is_none());
        let outcome = server.join().unwrap();
        assert!(matches!(
            outcome,
            Err(SessionError::Transport(
                emvp_network::CodecError::UnexpectedFrame { .. }
            ))
        ));
    }

    #[test]
    fn an_unknown_frame_kind_gets_a_structured_error() {
        let engine = Arc::new(AnswerEngine::cpu());
        let (mut client, server_stream) = UnixStream::pair().unwrap();
        let server = spawn_server(server_stream, engine);
        client_handshake(&mut client).unwrap();

        // A kind byte outside the protocol: the server reads the header,
        // answers the structured error, and closes without the payload.
        client.write_all(&[0x2A, 0, 0, 0, 0, 0, 0, 0, 0]).unwrap();
        let (error, _) = read_error(&mut client).unwrap();
        assert_eq!(error.code(), Some(ErrorCode::UnknownFrameKind));
        assert!(read_frame_header(&mut client).unwrap().is_none());

        let outcome = server.join().unwrap();
        assert!(matches!(
            outcome,
            Err(SessionError::Transport(
                emvp_network::CodecError::UnknownFrameKind { kind: 0x2A }
            ))
        ));
    }
}
