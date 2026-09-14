//! One client connection: the handshake, the load-once session state
//! machine, and request validation and evaluation.
//!
//! The session lifecycle is `Empty` until the one permitted matrix-set
//! upload succeeds, then `Loaded`. Upload failures are atomic: no
//! identifiers are assigned, no budget stays reserved, and the connection
//! closes so the client can retry with a fresh session. Evaluation
//! requests are validated in full before any GPU work; a validation
//! failure is answered and leaves the session usable, while a GPU failure
//! after registration closes the session and releases its matrices.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;

use emvp::{EmvpParams, EncryptedQuery, GpuEncryptedMatrix, GpuError};
use emvp_network::{
    CodecError, ErrorCode, EvaluateEntry, FrameHeader, FrameKind, FrameReader, HandshakeError,
    MatrixUpload, PROTOCOL_MODULUS, ProductEntry, read_evaluate_payload, read_frame_header,
    read_upload_matrices_payload, server_handshake, write_error, write_products,
    write_upload_accepted,
};

use crate::coordinator::GpuCoordinator;
use crate::failure::RequestFailure;

/// The per-connection view of one stored encrypted matrix.
struct MatrixInfo {
    instance_id: u128,
    width: usize,
}

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
    /// A GPU operation failed after the session had loaded matrices.
    Gpu(GpuError),
}

impl fmt::Display for SessionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Handshake(error) => error.fmt(formatter),
            Self::Transport(error) => error.fmt(formatter),
            Self::Gpu(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for SessionError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Handshake(error) => Some(error),
            Self::Transport(error) => Some(error),
            Self::Gpu(error) => Some(error),
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
/// Returns the handshake, transport, or GPU failure that ended the
/// session. Protocol-level rejections are answered in-band and close the
/// connection without surfacing here.
pub fn serve_connection<S: std::io::Read + std::io::Write>(
    stream: &mut S,
    gpu: &GpuCoordinator,
) -> Result<(), SessionError> {
    server_handshake(stream).map_err(SessionError::Handshake)?;
    let mut session = Session::new(gpu);
    loop {
        let Some(header) = read_frame_header(stream).map_err(SessionError::Transport)? else {
            return Ok(());
        };
        match session.handle_frame(stream, header) {
            Outcome::Continue => {}
            Outcome::Close(error) => return error.map_or(Ok(()), Err),
        }
    }
}

/// The mutable state of one connection.
struct Session<'a> {
    gpu: &'a GpuCoordinator,
    matrices: BTreeMap<u64, GpuEncryptedMatrix<PROTOCOL_MODULUS>>,
    reserved_bytes: u64,
    next_id: u64,
}

impl Session<'_> {
    const fn new(gpu: &GpuCoordinator) -> Session<'_> {
        Session {
            gpu,
            matrices: BTreeMap::new(),
            reserved_bytes: 0,
            next_id: 1,
        }
    }

    /// The stored matrix's public identity, if the identifier is loaded.
    fn info(&self, matrix_id: u64) -> Option<MatrixInfo> {
        let matrix = self.matrices.get(&matrix_id)?;
        Some(MatrixInfo {
            instance_id: matrix.instance_id(),
            width: matrix.params().n().ok()?,
        })
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
        if !self.matrices.is_empty() {
            send_error(
                stream,
                ErrorCode::AlreadyLoaded,
                "this session already loaded a matrix set",
            );
            return Outcome::Continue;
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
        // session stays empty, no identifiers are assigned, and the
        // connection closes so the client can retry fresh.
        let total_bytes = match validate_uploads(&uploads) {
            Ok(bytes) => bytes,
            Err(failure) => {
                send_error(stream, ErrorCode::UploadFailed, &failure.to_string());
                return Outcome::Close(None);
            }
        };
        if !self.gpu.reserve(total_bytes) {
            let message = format!(
                "the matrix set needs {total_bytes} GPU bytes and exceeds the residency budget"
            );
            send_error(stream, ErrorCode::UploadFailed, &message);
            return Outcome::Close(None);
        }
        let gpu_guard = self.gpu.lock_gpu();
        let mut handles = Vec::new();
        for upload in &uploads {
            match self
                .gpu
                .answerer()
                .upload_matrix_sync(&upload.params, &upload.matrix)
            {
                Ok(handle) => handles.push(handle),
                Err(error) => {
                    drop(handles);
                    drop(gpu_guard);
                    self.gpu.release(total_bytes);
                    send_error(stream, ErrorCode::UploadFailed, &error.to_string());
                    return Outcome::Close(Some(SessionError::Gpu(error)));
                }
            }
        }
        drop(gpu_guard);
        let mut identifiers = Vec::with_capacity(handles.len());
        for handle in handles {
            identifiers.push(self.next_id);
            self.matrices.insert(self.next_id, handle);
            self.next_id += 1;
        }
        self.reserved_bytes += total_bytes;
        eprintln!(
            "server: loaded {} matrices ({} bytes, {} bytes reserved process-wide)",
            identifiers.len(),
            total_bytes,
            self.gpu.reserved_bytes()
        );
        match write_upload_accepted(stream, &identifiers) {
            Ok(_) => Outcome::Continue,
            Err(error) => Outcome::Close(Some(SessionError::Transport(error))),
        }
    }

    fn handle_evaluate<S: std::io::Read + std::io::Write>(
        &mut self,
        stream: &mut S,
        payload_len: u64,
    ) -> Outcome {
        if self.matrices.is_empty() {
            send_error(
                stream,
                ErrorCode::NotLoaded,
                "no matrix set has been loaded on this connection",
            );
            return Outcome::Continue;
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
        let prepared = match prepare_evaluate(entries, &mut |matrix_id| self.info(matrix_id)) {
            Ok(prepared) => prepared,
            Err(failure) => {
                let code = failure.code();
                send_error(stream, code, &failure.to_string());
                return Outcome::Continue;
            }
        };
        let gpu_guard = self.gpu.lock_gpu();
        let mut products = Vec::with_capacity(prepared.len());
        for (matrix_id, queries) in prepared {
            // Validation pinned every identifier to a stored matrix, and
            // nothing mutates the map during evaluation.
            let Some(stored) = self.matrices.get(&matrix_id) else {
                drop(gpu_guard);
                send_error(
                    stream,
                    ErrorCode::Internal,
                    "a validated matrix identifier disappeared",
                );
                return Outcome::Close(Some(SessionError::Transport(CodecError::TruncatedFrame)));
            };
            match self
                .gpu
                .answerer()
                .answer_batch_sync_with_timings(stored, &queries)
            {
                Ok((answers, timings)) => {
                    eprintln!(
                        "server: matrix {matrix_id}: answered {} queries in {} ms",
                        queries.len(),
                        timings.total().as_millis()
                    );
                    products.push(ProductEntry { matrix_id, answers });
                }
                Err(error) => {
                    drop(gpu_guard);
                    self.matrices.clear();
                    send_error(stream, ErrorCode::GpuFailure, &error.to_string());
                    return Outcome::Close(Some(SessionError::Gpu(error)));
                }
            }
        }
        drop(gpu_guard);
        match write_products(stream, &products) {
            Ok(_) => Outcome::Continue,
            Err(error) => Outcome::Close(Some(SessionError::Transport(error))),
        }
    }
}

impl Drop for Session<'_> {
    fn drop(&mut self) {
        self.matrices.clear();
        if self.reserved_bytes > 0 {
            self.gpu.release(self.reserved_bytes);
            self.reserved_bytes = 0;
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

/// Validates one matrix set for upload and totals its GPU byte footprint.
///
/// The set is validated in full before any reservation or device work, so
/// a rejection cannot leave a partial upload behind.
///
/// # Errors
///
/// Returns the first structural problem: malformed parameters, a matrix
/// whose width disagrees with its parameters, or arithmetic overflow while
/// totaling the footprint.
pub fn validate_uploads(uploads: &[MatrixUpload]) -> Result<u64, RequestFailure> {
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
/// whole request is checked before any GPU work begins.
///
/// # Errors
///
/// Returns the first validation failure.
fn prepare_evaluate(
    entries: Vec<EvaluateEntry>,
    info: &mut dyn FnMut(u64) -> Option<MatrixInfo>,
) -> Result<Vec<(u64, Vec<EncryptedQuery<PROTOCOL_MODULUS>>)>, RequestFailure> {
    if entries.is_empty() {
        return Err(RequestFailure::EmptyRequest);
    }
    let mut matrix_ids = BTreeSet::new();
    let mut prepared = Vec::with_capacity(entries.len());
    for entry in entries {
        let Some(matrix) = info(entry.matrix_id) else {
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
        let mut query_ids = BTreeSet::new();
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
        AnswerMatrix, DecodingKey, DerivedState, EmvpParams, GpuError, ProtocolError, SecretKey,
        encrypt, query,
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

    use super::{RequestFailure, prepare_evaluate, serve_connection, validate_uploads};
    use crate::coordinator::GpuCoordinator;

    const PARAMS: EmvpParams = EmvpParams {
        k: 8,
        ell: 8,
        b: 2,
        lambda: 7,
    };

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
            .derive(rows, &mut rng, toeplitz_block)
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
    fn evaluate_preparation_enforces_the_request_contract() {
        let mut info = |matrix_id: u64| {
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
            &mut info,
        )
        .unwrap();
        assert_eq!(ok.len(), 1);
        assert_eq!(ok[0].0, 7);
        assert_eq!(ok[0].1.len(), 2);

        assert!(matches!(
            prepare_evaluate(vec![], &mut info),
            Err(RequestFailure::EmptyRequest)
        ));
        assert!(matches!(
            prepare_evaluate(vec![entry(9, vec![make_query(0, 16, 42)])], &mut info),
            Err(RequestFailure::UnknownMatrix { id: 9 })
        ));
        assert!(matches!(
            prepare_evaluate(
                vec![
                    entry(7, vec![make_query(0, 16, 42)]),
                    entry(7, vec![make_query(1, 16, 42)]),
                ],
                &mut info,
            ),
            Err(RequestFailure::DuplicateMatrix { id: 7 })
        ));
        assert!(matches!(
            prepare_evaluate(vec![entry(7, vec![make_query(3, 15, 42)])], &mut info,),
            Err(RequestFailure::QueryWidth {
                expected: 16,
                actual: 15
            })
        ));
        assert!(matches!(
            prepare_evaluate(vec![entry(7, vec![make_query(3, 16, 43)])], &mut info,),
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
                &mut info,
            ),
            Err(RequestFailure::DuplicateQuery {
                matrix_id: 7,
                query_id: 5
            })
        ));
        assert!(matches!(
            prepare_evaluate(vec![entry(7, vec![])], &mut info),
            Err(RequestFailure::EmptyRequest)
        ));
    }

    /// Probes for a compute device; `None` means "skip this test".
    fn coordinator() -> Option<Arc<GpuCoordinator>> {
        match GpuCoordinator::new(1 << 30) {
            Ok(gpu) => Some(Arc::new(gpu)),
            Err(GpuError::NoAdapter { reason }) => {
                println!("skipping server test: no compute adapter ({reason})");
                None
            }
            Err(error) => panic!("unexpected coordinator failure: {error}"),
        }
    }

    fn spawn_server(
        server_stream: UnixStream,
        gpu: Arc<GpuCoordinator>,
    ) -> thread::JoinHandle<Result<(), super::SessionError>> {
        thread::spawn(move || {
            let mut server_stream = server_stream;
            serve_connection(&mut server_stream, &gpu)
        })
    }

    fn random_query(
        state: &mut DerivedState<PROTOCOL_MODULUS, ToeplitzFastProduct<PROTOCOL_MODULUS>>,
        seed: u64,
    ) -> (
        emvp::EncryptedQuery<PROTOCOL_MODULUS>,
        DecodingKey<PROTOCOL_MODULUS>,
    ) {
        let q = values(PARAMS.ell, seed);
        query(state, &q).unwrap()
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
        let Some(gpu) = coordinator() else {
            return;
        };
        let (mut client, server_stream) = UnixStream::pair().unwrap();
        let server = spawn_server(server_stream, gpu);
        client_handshake(&mut client).unwrap();

        let (upload_a, mut state_a, plaintext_a) = upload(5, 0x11, 100);
        let (upload_b, mut state_b, plaintext_b) = upload(21, 0x22, 101);
        write_upload_matrices(&mut client, &[upload_a, upload_b]).unwrap();
        let (ids, _) = read_upload_accepted(&mut client).unwrap();
        assert_eq!(ids, vec![1, 2]);

        let (query_a0, key_a0) = random_query(&mut state_a, 200);
        let (query_a1, key_a1) = random_query(&mut state_a, 201);
        let (query_of_b, key_of_b) = random_query(&mut state_b, 202);
        let entries = vec![
            EvaluateEntry {
                matrix_id: ids[0],
                queries: vec![query_a0, query_a1],
            },
            EvaluateEntry {
                matrix_id: ids[1],
                queries: vec![query_of_b.clone()],
            },
        ];
        write_evaluate(&mut client, &entries).unwrap();
        let (products, _) = read_products(&mut client).unwrap();
        assert_eq!(products.len(), 2);

        let ProductEntry { matrix_id, answers } = &products[0];
        assert_eq!(*matrix_id, ids[0]);
        assert_eq!(answers.len(), 2);
        for ((answer, key), sent_query) in answers
            .iter()
            .zip([&key_a0, &key_a1])
            .zip(&entries[0].queries)
        {
            assert_eq!(answer.instance_id(), state_a.instance_id());
            assert_eq!(answer.query_id(), key.query_id());
            assert_eq!(answer.rows(), 5);
            assert_eq!(answer.blocks(), 8);
            let decoded_product = decoded(answer, key);
            assert_eq!(
                decoded_product,
                naive_product(&plaintext_a, sent_query.values(), 5)
            );
        }
        let ProductEntry { matrix_id, answers } = &products[1];
        assert_eq!(*matrix_id, ids[1]);
        assert_eq!(answers.len(), 1);
        assert_eq!(answers[0].rows(), 21);
        let decoded_product = decoded(&answers[0], &key_of_b);
        assert_eq!(
            decoded_product,
            naive_product(&plaintext_b, query_of_b.values(), 21)
        );

        drop(client);
        server.join().unwrap().unwrap();
    }

    #[test]
    fn a_second_upload_is_rejected_and_the_session_survives() {
        let Some(gpu) = coordinator() else {
            return;
        };
        let (mut client, server_stream) = UnixStream::pair().unwrap();
        let server = spawn_server(server_stream, gpu);
        client_handshake(&mut client).unwrap();

        let (upload_a, mut state_a, _) = upload(5, 0x11, 300);
        write_upload_matrices(&mut client, &[upload_a]).unwrap();
        let (ids, _) = read_upload_accepted(&mut client).unwrap();

        let (upload_b, _, _) = upload(7, 0x33, 301);
        write_upload_matrices(&mut client, &[upload_b]).unwrap();
        let (error, _) = read_error(&mut client).unwrap();
        assert_eq!(error.code(), Some(ErrorCode::AlreadyLoaded));

        let (query_a0, key_a0) = random_query(&mut state_a, 302);
        write_evaluate(
            &mut client,
            &[EvaluateEntry {
                matrix_id: ids[0],
                queries: vec![query_a0],
            }],
        )
        .unwrap();
        let (products, _) = read_products(&mut client).unwrap();
        assert_eq!(products[0].answers[0].query_id(), key_a0.query_id());

        drop(client);
        server.join().unwrap().unwrap();
    }

    #[test]
    fn a_failed_upload_is_atomic_and_closes_the_session() {
        let Some(gpu) = coordinator() else {
            return;
        };
        let (mut client, server_stream) = UnixStream::pair().unwrap();
        let server = spawn_server(server_stream, gpu);
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
        assert_eq!(error.code(), Some(ErrorCode::UploadFailed));

        // The server closed the connection without assigning identifiers.
        assert!(read_frame_header(&mut client).unwrap().is_none());
        server.join().unwrap().unwrap();
    }

    #[test]
    fn evaluation_before_an_upload_is_answered_with_an_error() {
        let Some(gpu) = coordinator() else {
            return;
        };
        let (mut client, server_stream) = UnixStream::pair().unwrap();
        let server = spawn_server(server_stream, gpu);
        client_handshake(&mut client).unwrap();

        let entry = EvaluateEntry {
            matrix_id: 1,
            queries: vec![emvp::EncryptedQuery::from_parts(1, 0, values(16, 500))],
        };
        for _ in 0..2 {
            write_evaluate(&mut client, std::slice::from_ref(&entry)).unwrap();
            let (error, _) = read_error(&mut client).unwrap();
            assert_eq!(error.code(), Some(ErrorCode::NotLoaded));
        }

        drop(client);
        server.join().unwrap().unwrap();
    }

    #[test]
    fn server_directed_frames_close_the_session() {
        let Some(gpu) = coordinator() else {
            return;
        };
        let (mut client, server_stream) = UnixStream::pair().unwrap();
        let server = spawn_server(server_stream, gpu);
        client_handshake(&mut client).unwrap();

        // A client must never send server-to-client frames.
        let mut buffer = Vec::new();
        emvp_network::write_products(
            &mut buffer,
            &[ProductEntry {
                matrix_id: 1,
                answers: vec![AnswerMatrix::from_parts(1, 0, values(8, 600), 4, 2)],
            }],
        )
        .unwrap();
        client.write_all(&buffer).unwrap();
        let (error, _) = read_error(&mut client).unwrap();
        assert_eq!(error.code(), Some(ErrorCode::UnexpectedFrame));

        // The server closed the connection after the violation.
        assert!(read_frame_header(&mut client).unwrap().is_none());
        let outcome = server.join().unwrap();
        assert!(matches!(
            outcome,
            Err(super::SessionError::Transport(
                emvp_network::CodecError::UnexpectedFrame { .. }
            ))
        ));
    }
}
