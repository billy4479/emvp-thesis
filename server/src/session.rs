//! One client connection: the handshake, the load-once session state
//! machine, and request validation and evaluation.
//!
//! The session phase is explicitly `Empty` or `Loaded`: the one permitted
//! matrix-set upload moves `Empty` to `Loaded` atomically. Upload failures
//! are atomic too: the upload runs the plan → reserve → copy → commit
//! pipeline — the whole set is validated, staged into a reusable prepare
//! workspace, and only then committed — so a failure releases the whole
//! reservation and returns no handles, no identifiers are assigned, and
//! the connection closes so the client can retry with a fresh session.
//! Both state violations — a second upload on a loaded session, and an
//! evaluation before any upload — are answered with the structured error
//! and close the session immediately, without the offending payload being
//! read or drained. Request-validation failures (malformed frames,
//! rejected parameters, unknown identifiers) are answered in-band; a
//! validation failure of an evaluation leaves the session usable, while
//! an engine failure after registration closes the session and releases
//! its matrices. Every `Evaluate` frame is answered with one
//! plan → reserve → execute engine cycle, whose report feeds the
//! per-frame timing logs.
//!
//! # Workspace lifetime pattern
//!
//! The session owns every reusable arena for the connection's lifetime:
//! the upload and evaluate decode workspaces, the engine prepare
//! workspace, the engine answer workspace, and the upload-accepted
//! identifier buffer. Each grows only through its `reserve` step and
//! never shrinks, so after the first frame of a given shape a
//! steady-state `Evaluate` performs no allocation beyond small per-frame
//! metadata Vecs. Refs that borrow those arenas ([`EncryptedQueryRef`]
//! over the decoded queries, [`AnswerRef`] over the computed answers)
//! cannot be stored in the session next to the arenas they borrow — a
//! field's lifetime parameter is fixed for the whole connection, while
//! such a borrow lives for one frame only. They are therefore rebuilt
//! per frame into `Vec::with_capacity` scratch locals, and the per-frame
//! job and product input Vecs carve their query and answer slices from
//! those locals. This keeps every use of an arena inside the frame that
//! borrowed it, while the arenas themselves, the validation
//! [`HashSet`]s, and the identifier buffer keep their capacity across
//! frames.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::ops::Range;

use emvp::view::{AnswerRef, EncryptedQueryRef};
use emvp::{
    AnswerEngine, AnswerEngineError, AnswerJob, EmvpParams, EngineAnswers, EnginePlan,
    EngineWorkspace, PrepareError, PrepareMatrixError, PrepareWorkspace, PreparedMatrix, UploadRef,
};
use emvp_network::{
    CodecError, ErrorCode, FrameHeader, FrameKind, FrameReader, HandshakeError, PROTOCOL_MODULUS,
    read_frame_header, server_handshake, write_error, write_upload_accepted,
};
use emvp_network::{
    EvaluateViews, EvaluateWorkspace, ProductEntryInput, UploadAcceptedPlan,
    UploadAcceptedWorkspace, UploadViews, UploadWorkspace, decode_evaluate, decode_upload,
    plan_evaluate, plan_upload, write_products,
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

    /// The stored matrix metadata for an assigned identifier.
    fn info(&self, matrix_id: u64) -> Option<MatrixInfo> {
        self.matrices()?.get(&matrix_id).map(|m| m.info)
    }

    /// The stored engine handle for an assigned identifier.
    fn matrix(&self, matrix_id: u64) -> Option<&PreparedMatrix<PROTOCOL_MODULUS>> {
        self.matrices()?.get(&matrix_id).map(|m| &m.prepared)
    }
}

/// The mutable state of one connection.
///
/// The workspace fields are the connection's reusable arenas; see the
/// module documentation for the lifetime pattern that keeps borrowed refs
/// out of them.
struct Session<'a> {
    engine: &'a AnswerEngine<PROTOCOL_MODULUS>,
    phase: Phase,
    /// The decode arena of the one permitted `UploadMatrices` frame.
    upload_ws: UploadWorkspace,
    /// The staging slots of the plan → reserve → copy → commit upload.
    prepare_ws: PrepareWorkspace<PROTOCOL_MODULUS>,
    /// The decode arena of the current `Evaluate` frame.
    evaluate_ws: EvaluateWorkspace,
    /// The answer arena of the current `Evaluate` frame.
    engine_ws: EngineWorkspace<PROTOCOL_MODULUS>,
    /// The identifier buffer the upload acknowledgment streams from.
    accepted_ws: UploadAcceptedWorkspace,
    /// The matrix identifiers seen in the current evaluation request.
    seen_matrices: HashSet<u64>,
    /// The query identifiers seen in the current evaluation entry.
    seen_queries: HashSet<u64>,
}

impl Session<'_> {
    fn new(engine: &AnswerEngine<PROTOCOL_MODULUS>) -> Session<'_> {
        Session {
            engine,
            phase: Phase::Empty,
            upload_ws: UploadWorkspace::new(),
            prepare_ws: PrepareWorkspace::new(),
            evaluate_ws: EvaluateWorkspace::new(),
            engine_ws: EngineWorkspace::new(),
            accepted_ws: UploadAcceptedWorkspace::new(),
            seen_matrices: HashSet::new(),
            seen_queries: HashSet::new(),
        }
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

    /// Handles the one permitted matrix-set upload through the plan →
    /// reserve → copy → commit pipeline: the frame is planned and decoded
    /// into the session's upload workspace, the decoded views are
    /// validated in full, staged into the prepare workspace, and committed
    /// into engine-resident handles.
    /// Plans, reserves, and decodes one upload frame into `workspace`.
    /// The workspace is a separate parameter so the caller's borrow stays
    /// field-scoped while the decoded views are alive.
    ///
    /// # Errors
    ///
    /// Returns the planning, reservation, or decoding failure of the
    /// frame.
    fn read_upload<'ws, R: std::io::Read>(
        workspace: &'ws mut UploadWorkspace,
        frame: &mut FrameReader<'_, R>,
    ) -> Result<UploadViews<'ws>, CodecError> {
        let plan = plan_upload(frame)?;
        workspace.reserve(&plan)?;
        decode_upload(frame, &plan, workspace)
    }

    /// Handles the one permitted matrix-set upload through the plan →
    /// reserve → copy → commit pipeline: the frame is planned and decoded
    /// into the session's upload workspace, the decoded views are
    /// validated in full, staged into the prepare workspace, and committed
    /// into engine-resident handles.
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
        let views = match Self::read_upload(&mut self.upload_ws, &mut frame) {
            Ok(views) => views,
            Err(error) => {
                let message = error.to_string();
                send_error(stream, error.error_code(), &message);
                return Outcome::Close(Some(SessionError::Transport(error)));
            }
        };
        // Every failure past this point is an atomic upload failure: the
        // set is validated whole, staged with one workspace transition,
        // and committed with one budget transition — so a failure releases
        // the whole reservation, no identifier is assigned, no handle is
        // returned, and the connection closes so the client can retry
        // with a fresh session.
        let total_bytes = match validate_upload_views(views) {
            Ok(bytes) => bytes,
            Err(failure) => {
                send_error(stream, failure.code(), &failure.to_string());
                return Outcome::Close(None);
            }
        };
        let infos: Vec<MatrixInfo> = views
            .iter()
            .map(|view| MatrixInfo {
                instance_id: view.instance_id,
                width: view.columns,
            })
            .collect();
        // Borrowed upload candidates straight from the decoded workspace:
        // a validated decode always satisfies the view's shape checks, so
        // the `matrix` construction cannot fail.
        let uploads = views
            .iter()
            .map(|view| {
                view.matrix()
                    .map(|matrix| UploadRef::new(view.params, matrix))
            })
            .collect::<Result<Vec<_>, _>>()
            .ok();
        let Some(uploads) = uploads else {
            send_error(
                stream,
                ErrorCode::UploadFailed,
                "an uploaded matrix was malformed",
            );
            return Outcome::Close(None);
        };
        let prepare_plan = match self.engine.plan_prepare(&uploads) {
            Ok(plan) => plan,
            Err(error) => {
                send_error(stream, ErrorCode::UploadFailed, &error.to_string());
                return Outcome::Close(None);
            }
        };
        if let Err(error) = self.prepare_ws.reserve(&prepare_plan) {
            // A refused slot allocation releases the workspace untouched;
            // like a budget refusal it closes the session without an
            // engine error of its own.
            send_error(stream, ErrorCode::UploadFailed, &error.to_string());
            return Outcome::Close(None);
        }
        if let Err(error) = self.prepare_ws.copy(&prepare_plan, &uploads) {
            // Copy validates the plan pairing itself; a mismatch is an
            // internal invariant break, surfaced as an upload failure.
            send_error(stream, ErrorCode::UploadFailed, &error.to_string());
            return Outcome::Close(Some(SessionError::Upload(prepare_matrix_error(error))));
        }
        // The commit consumes the workspace either way: its slots move
        // into the committed matrices, so the session rebuilds an empty
        // one and keeps no stale staging state.
        let workspace = std::mem::take(&mut self.prepare_ws);
        let prepared = match self.engine.prepare_committed(&prepare_plan, workspace) {
            Ok(prepared) => prepared,
            Err(error) => {
                self.prepare_ws = PrepareWorkspace::new();
                send_error(stream, ErrorCode::UploadFailed, &error.to_string());
                return match error {
                    PrepareError::Capacity { .. } | PrepareError::Budget(_) => Outcome::Close(None),
                    error => {
                        Outcome::Close(Some(SessionError::Upload(prepare_matrix_error(error))))
                    }
                };
            }
        };
        self.prepare_ws = PrepareWorkspace::new();
        self.load_matrices_and_reply(stream, prepared, infos, total_bytes)
    }

    /// Installs the committed matrices as the session's loaded phase,
    /// assigning the sequential identifiers, and streams the upload
    /// acknowledgment from the session's identifier buffer.
    fn load_matrices_and_reply<S: std::io::Write>(
        &mut self,
        stream: &mut S,
        prepared: Vec<PreparedMatrix<PROTOCOL_MODULUS>>,
        infos: Vec<MatrixInfo>,
        total_bytes: u64,
    ) -> Outcome {
        // Identifiers are assigned once, consecutively from one, in input
        // order, so they map back onto the stored handles by key.
        let count = prepared.len();
        let mut matrices = HashMap::with_capacity(count);
        for (index, (prepared, info)) in prepared.into_iter().zip(infos).enumerate() {
            matrices.insert(index as u64 + 1, StoredMatrix { info, prepared });
        }
        eprintln!(
            "server: loaded {count} matrices ({total_bytes} bytes of engine matrix residency)"
        );
        self.phase = Phase::Loaded { matrices };
        let accepted = UploadAcceptedPlan {
            identifiers: (1..=count as u64).collect(),
        };
        if let Err(error) = self.accepted_ws.reserve(&accepted) {
            let message = error.to_string();
            send_error(stream, error.error_code(), &message);
            return Outcome::Close(Some(SessionError::Transport(error)));
        }
        match write_upload_accepted(stream, self.accepted_ws.identifiers()) {
            Ok(_) => Outcome::Continue,
            Err(error) => Outcome::Close(Some(SessionError::Transport(error))),
        }
    }

    /// Handles one evaluation request through the plan → reserve →
    /// execute pipeline: the frame is planned and decoded into the
    /// session's evaluate workspace, validated in full against the loaded
    /// phase, planned against the engine, executed into the session's
    /// answer workspace, and streamed back as one products frame.
    /// Plans, reserves, and decodes one evaluation frame into
    /// `workspace`, returning the decoded views plus the frame's total
    /// query count. The workspace is a separate parameter so the caller's
    /// borrow stays field-scoped while the decoded views are alive.
    ///
    /// # Errors
    ///
    /// Returns the planning, reservation, or decoding failure of the
    /// frame.
    fn read_evaluate<'ws, R: std::io::Read>(
        workspace: &'ws mut EvaluateWorkspace,
        frame: &mut FrameReader<'_, R>,
    ) -> Result<(EvaluateViews<'ws>, usize), CodecError> {
        let plan = plan_evaluate(frame)?;
        workspace.reserve(&plan)?;
        let total_queries = plan.queries.len();
        let views = decode_evaluate(frame, &plan, workspace)?;
        Ok((views, total_queries))
    }

    /// Handles one evaluation request through the plan → reserve →
    /// execute pipeline: the frame is planned and decoded into the
    /// session's evaluate workspace, validated in full against the loaded
    /// phase, planned against the engine, executed into the session's
    /// answer workspace, and streamed back as one products frame.
    fn handle_evaluate<S: std::io::Read + std::io::Write>(
        &mut self,
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
        let (views, total_queries) = match Self::read_evaluate(&mut self.evaluate_ws, &mut frame) {
            Ok(views) => views,
            Err(error) => {
                let message = error.to_string();
                send_error(stream, error.error_code(), &message);
                return Outcome::Close(Some(SessionError::Transport(error)));
            }
        };
        // Validation materializes the decoded queries as borrowed refs,
        // in entry order, and checks every entry against the phase
        // before any engine work begins.
        let mut query_refs: Vec<EncryptedQueryRef<'_, PROTOCOL_MODULUS>> =
            Vec::with_capacity(total_queries);
        let phase = &self.phase;
        let mut seen_matrices = std::mem::take(&mut self.seen_matrices);
        let mut seen_queries = std::mem::take(&mut self.seen_queries);
        let evaluated = prepare_evaluate(
            views,
            &mut |matrix_id| phase.info(matrix_id),
            &mut seen_matrices,
            &mut seen_queries,
            &mut query_refs,
        );
        self.seen_matrices = seen_matrices;
        self.seen_queries = seen_queries;
        let evaluated = match evaluated {
            Ok(evaluated) => evaluated,
            Err(failure) => {
                let code = failure.code();
                send_error(stream, code, &failure.to_string());
                return Outcome::Continue;
            }
        };
        // One engine cycle answers the whole frame: plan binds the jobs,
        // reserve grows the answer arena once, execute fills it.
        let mut jobs: Vec<AnswerJob<'_, PROTOCOL_MODULUS, _>> = Vec::with_capacity(evaluated.len());
        for entry in &evaluated {
            // Validation pinned every identifier to a stored matrix, and
            // nothing mutates the phase during evaluation.
            let Some(matrix) = self.phase.matrix(entry.matrix_id) else {
                send_error(
                    stream,
                    ErrorCode::Internal,
                    "a validated matrix identifier disappeared",
                );
                return Outcome::Close(None);
            };
            jobs.push(AnswerJob {
                matrix,
                queries: &query_refs[entry.queries.clone()],
            });
        }
        let engine_plan = match self.engine.plan(&jobs) {
            Ok(plan) => plan,
            Err(error) => {
                send_error(stream, ErrorCode::GpuFailure, &error.to_string());
                return Outcome::Close(Some(SessionError::Engine(error)));
            }
        };
        if let Err(error) = self.engine_ws.reserve(&engine_plan) {
            send_error(stream, ErrorCode::GpuFailure, &error.to_string());
            return Outcome::Close(Some(SessionError::Engine(error)));
        }
        let (answers, report) = match self.engine.execute(&engine_plan, &mut self.engine_ws) {
            Ok(outcome) => outcome,
            Err(error) => {
                send_error(stream, ErrorCode::GpuFailure, &error.to_string());
                return Outcome::Close(Some(SessionError::Engine(error)));
            }
        };
        log_report(&report, &engine_plan, &evaluated);
        send_products(stream, &answers, &engine_plan, &evaluated)
    }
}

/// Materializes the executed answers as borrowed refs, in entry order
/// paired through the planned query slices, and streams one products
/// frame. The session workspaces stay behind `&mut` in the caller.
fn send_products<S: std::io::Write>(
    stream: &mut S,
    answers: &EngineAnswers<'_, '_, PROTOCOL_MODULUS, EncryptedQueryRef<'_, PROTOCOL_MODULUS>>,
    engine_plan: &EnginePlan<'_, PROTOCOL_MODULUS, EncryptedQueryRef<'_, PROTOCOL_MODULUS>>,
    evaluated: &[EvaluatedEntry],
) -> Outcome {
    let total_queries: usize = engine_plan
        .iter()
        .map(|entry| entry.shape().queries())
        .sum();
    let mut answer_refs: Vec<AnswerRef<'_, PROTOCOL_MODULUS>> = Vec::with_capacity(total_queries);
    let mut answer_ranges: Vec<Range<usize>> = Vec::with_capacity(evaluated.len());
    for index in 0..answers.len() {
        let Some(entry_plan) = engine_plan.entry(index) else {
            send_error(
                stream,
                ErrorCode::Internal,
                "a planned evaluation entry disappeared",
            );
            return Outcome::Close(None);
        };
        let batch = match answers.entry_answers(index) {
            Ok(batch) => batch,
            Err(error) => {
                send_error(stream, ErrorCode::Internal, &error.to_string());
                return Outcome::Close(None);
            }
        };
        let start = answer_refs.len();
        match batch.iter(entry_plan.queries()) {
            Ok(iter) => answer_refs.extend(iter),
            Err(error) => {
                send_error(stream, ErrorCode::Internal, &error.to_string());
                return Outcome::Close(None);
            }
        }
        answer_ranges.push(start..answer_refs.len());
    }
    let mut products: Vec<ProductEntryInput<'_>> = Vec::with_capacity(evaluated.len());
    for (entry, range) in evaluated.iter().zip(&answer_ranges) {
        // Validation required one query per entry, so the engine answered
        // at least one per entry; the entry's shape metadata comes from
        // its first answer, as the wire descriptor shares it across the
        // entry.
        let Some(first) = answer_refs.get(range.start) else {
            send_error(
                stream,
                ErrorCode::Internal,
                "an evaluated entry produced no answers",
            );
            return Outcome::Close(None);
        };
        products.push(ProductEntryInput {
            matrix_id: entry.matrix_id,
            instance_id: first.instance_id(),
            rows: first.rows(),
            blocks: first.blocks(),
            answers: &answer_refs[range.clone()],
        });
    }
    match write_products(stream, &products) {
        Ok(_) => Outcome::Continue,
        Err(error) => Outcome::Close(Some(SessionError::Transport(error))),
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
fn log_report(
    report: &emvp::EngineReport,
    plan: &EnginePlan<'_, PROTOCOL_MODULUS, EncryptedQueryRef<'_, PROTOCOL_MODULUS>>,
    evaluated: &[EvaluatedEntry],
) {
    let queries: usize = plan.iter().map(|entry| entry.shape().queries()).sum();
    eprintln!(
        "server: answered {queries} queries across {} matrices in {} ms (planning {} ms, cpu {} ms, gpu buffers {} ms, upload {} ms, submit {} ms, readback {} ms, reconstruct {} ms)",
        plan.len(),
        report.total.as_millis(),
        report.planning.as_millis(),
        report.cpu_compute.as_millis(),
        report.gpu_prepare_buffers.as_millis(),
        report.gpu_upload.as_millis(),
        report.gpu_submit.as_millis(),
        report.gpu_wait.as_millis(),
        report.gpu_reconstruct.as_millis(),
    );
    for (entry, evaluated) in plan.iter().zip(evaluated) {
        eprintln!(
            "server: matrix {}: {} queries via {:?} ({} multiplications)",
            evaluated.matrix_id,
            entry.shape().queries(),
            entry.backend(),
            entry.multiplications(),
        );
    }
}

/// Validates a decoded upload set and totals its GPU byte footprint.
///
/// The set must be nonempty and is validated in full before any engine
/// work, so a rejection cannot leave a partial upload behind.
///
/// # Errors
///
/// Returns the first structural problem: an empty set, malformed
/// parameters, a matrix whose width disagrees with its parameters, or
/// arithmetic overflow while totaling the footprint.
fn validate_upload_views(views: UploadViews<'_>) -> Result<u64, RequestFailure> {
    if views.is_empty() {
        return Err(RequestFailure::EmptyRequest);
    }
    let mut total_bytes = 0_u64;
    for view in views {
        validate_upload_shape(&view.params, view.rows, view.columns, &mut total_bytes)?;
    }
    Ok(total_bytes)
}

/// Validates one uploaded matrix's parameters and shape against its
/// declared width and adds its byte footprint to `total_bytes`.
///
/// # Errors
///
/// Returns the first structural problem of the record.
fn validate_upload_shape(
    params: &EmvpParams,
    rows: usize,
    columns: usize,
    total_bytes: &mut u64,
) -> Result<(), RequestFailure> {
    validate_params(params)?;
    let width = params.n().map_err(RequestFailure::Params)?;
    if columns != width {
        return Err(RequestFailure::ColumnsMismatch {
            expected: width,
            actual: columns,
        });
    }
    let words = rows
        .checked_mul(columns)
        .ok_or(RequestFailure::DimensionOverflow)?;
    let bytes = u64::try_from(words)
        .ok()
        .and_then(|word_count| word_count.checked_mul(4))
        .ok_or(RequestFailure::DimensionOverflow)?;
    *total_bytes = total_bytes
        .checked_add(bytes)
        .ok_or(RequestFailure::DimensionOverflow)?;
    Ok(())
}

/// Validates uploaded protocol parameters.
fn validate_params(params: &EmvpParams) -> Result<(), RequestFailure> {
    params.validate_dimensions().map_err(RequestFailure::Params)
}

/// One validated evaluation entry: its matrix identifier and the range of
/// the entry's materialized query refs, in request order.
struct EvaluatedEntry {
    /// The server identifier of the targeted matrix.
    matrix_id: u64,
    /// The entry's query refs in the frame's query-ref scratch.
    queries: Range<usize>,
}

/// Validates an evaluation request in full and materializes its queries
/// as borrowed refs into `query_refs`, returning one
/// `(matrix id, query range)` record per entry, in request order.
///
/// Every entry must reference a distinct loaded matrix, carry at least one
/// query, and every query must match its matrix's width, instance
/// identifier, and carry a query identifier unique within its entry. The
/// whole request is checked before any engine work begins; the
/// `seen_matrices` and `seen_queries` sets are scratch the caller reuses
/// across frames and are cleared here.
///
/// # Errors
///
/// Returns the first validation failure.
fn prepare_evaluate<'v>(
    views: EvaluateViews<'v>,
    lookup: &mut dyn FnMut(u64) -> Option<MatrixInfo>,
    seen_matrices: &mut HashSet<u64>,
    seen_queries: &mut HashSet<u64>,
    query_refs: &mut Vec<EncryptedQueryRef<'v, PROTOCOL_MODULUS>>,
) -> Result<Vec<EvaluatedEntry>, RequestFailure> {
    if views.is_empty() {
        return Err(RequestFailure::EmptyRequest);
    }
    seen_matrices.clear();
    let mut evaluated = Vec::with_capacity(views.len());
    for entry in views {
        let Some(matrix) = lookup(entry.matrix_id()) else {
            return Err(RequestFailure::UnknownMatrix {
                id: entry.matrix_id(),
            });
        };
        if !seen_matrices.insert(entry.matrix_id()) {
            return Err(RequestFailure::DuplicateMatrix {
                id: entry.matrix_id(),
            });
        }
        if entry.is_empty() {
            return Err(RequestFailure::EmptyRequest);
        }
        seen_queries.clear();
        let start = query_refs.len();
        #[expect(
            clippy::explicit_iter_loop,
            reason = "the entry view is `Copy` and its `iter` yields refs for exactly the view's lifetime; `&entry` would require the local binding to outlive the frame"
        )]
        for query in entry.iter() {
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
            if !seen_queries.insert(query.query_id()) {
                return Err(RequestFailure::DuplicateQuery {
                    matrix_id: entry.matrix_id(),
                    query_id: query.query_id(),
                });
            }
            query_refs.push(query);
        }
        evaluated.push(EvaluatedEntry {
            matrix_id: entry.matrix_id(),
            queries: start..query_refs.len(),
        });
    }
    Ok(evaluated)
}

/// Maps a whole-set prepare failure onto the single-matrix upload error
/// the session reports, folding the plan-pairing and slot-capacity
/// failures that a validated server cannot produce into internal-state
/// errors. Budget and device-upload failures keep their source.
fn prepare_matrix_error(error: PrepareError) -> PrepareMatrixError {
    match error {
        PrepareError::Protocol { source, .. } => PrepareMatrixError::Protocol(source),
        PrepareError::Budget(source) | PrepareError::Upload { source, .. } => {
            PrepareMatrixError::Gpu(source)
        }
        PrepareError::InternalState(reason) => PrepareMatrixError::InternalState(reason),
        _ => PrepareMatrixError::InternalState("an unexpected prepare failure"),
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::io::Write;
    use std::os::unix::net::UnixStream;
    use std::sync::Arc;
    use std::thread;

    use emvp::{
        AnswerEngine, AnswerRef, DecodingKey, DerivedState, EmvpParams, EncryptedMatrix,
        EncryptedQuery, EncryptedQueryRef, MaskContextId, ProtocolError, SecretKey, encrypt, query,
    };
    use emvp_network::{
        CodecError, ErrorCode, FrameKind, FrameReader, HEADER_BYTES, PROTOCOL_MODULUS,
        client_handshake, read_error, read_frame_header, read_upload_accepted, write_evaluate,
        write_frame_header,
    };
    use emvp_network::{
        EvaluateEntryInput, EvaluateViews, EvaluateWorkspace, ProductEntryInput, ProductsViews,
        ProductsWorkspace, UploadMatrixView, UploadViews, UploadWorkspace, decode_evaluate,
        decode_products, decode_upload, plan_evaluate, plan_products, plan_upload, write_products,
        write_upload_matrices,
    };
    use prime_field_layer::{FieldElement, PrimeField};
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;
    use trapdoor_matrices::ToeplitzFastProduct;

    use super::{
        EvaluatedEntry, MatrixInfo, RequestFailure, SessionError, prepare_evaluate,
        serve_connection, validate_upload_views,
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

    /// One derived, encrypted matrix plus everything needed to derive
    /// queries against it and check its answers.
    struct Uploaded {
        params: EmvpParams,
        matrix: EncryptedMatrix<PROTOCOL_MODULUS>,
        state: DerivedState<PROTOCOL_MODULUS, ToeplitzFastProduct<PROTOCOL_MODULUS>>,
        plaintext: Vec<FieldElement<PROTOCOL_MODULUS>>,
    }

    impl Uploaded {
        /// The borrowed upload view the v2 writer consumes.
        fn view(&self) -> UploadMatrixView<'_> {
            UploadMatrixView {
                params: self.params,
                instance_id: self.matrix.instance_id(),
                rows: self.matrix.rows(),
                columns: self.matrix.columns(),
                values: self.matrix.values(),
            }
        }
    }

    fn upload(rows: usize, key: u8, seed: u64) -> Uploaded {
        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let mut state = SecretKey::<PROTOCOL_MODULUS>::new_insecure(PARAMS, [key; 32])
            .unwrap()
            .derive(CONTEXT, rows, &mut rng, toeplitz_block)
            .unwrap();
        let plaintext = values(rows * PARAMS.ell, seed ^ 0xFF);
        let matrix = encrypt(&mut state, &plaintext).unwrap();
        Uploaded {
            params: PARAMS,
            matrix,
            state,
            plaintext,
        }
    }

    /// Encodes `uploads` with the v2 writer and decodes them through the
    /// v2 pipeline, mirroring the server's per-frame path, then hands the
    /// decoded views to `consume`.
    fn with_decoded_uploads<T>(
        uploads: &[UploadMatrixView<'_>],
        consume: impl FnOnce(UploadViews<'_>) -> T,
    ) -> T {
        let mut bytes = Vec::new();
        write_upload_matrices(&mut bytes, uploads).unwrap();
        let mut reader = &bytes[..];
        let header = read_frame_header(&mut reader).unwrap().unwrap();
        assert_eq!(header.kind, FrameKind::UploadMatrices);
        let mut frame = FrameReader::new(&mut reader, header.payload_len);
        let plan = plan_upload(&mut frame).unwrap();
        let mut workspace = UploadWorkspace::new();
        workspace.reserve(&plan).unwrap();
        let views = decode_upload(&mut frame, &plan, &mut workspace).unwrap();
        consume(views)
    }

    #[test]
    fn uploads_validate_and_total_their_footprint() {
        let small = upload(3, 1, 1);
        let tall = upload(35, 2, 2);
        let bytes =
            with_decoded_uploads(&[small.view(), tall.view()], validate_upload_views).unwrap();
        assert_eq!(bytes, (3 * 16 + 35 * 16) * 4);
    }

    #[test]
    fn uploads_reject_malformed_dimensions() {
        let good = upload(3, 1, 4);
        // `b = 3` does not divide `n = 16`: structurally invalid parameters.
        let malformed = UploadMatrixView {
            params: EmvpParams {
                b: 3,
                ..good.params
            },
            instance_id: good.matrix.instance_id(),
            rows: good.matrix.rows(),
            columns: good.matrix.columns(),
            values: good.matrix.values(),
        };
        let failure = with_decoded_uploads(&[malformed], validate_upload_views).unwrap_err();
        assert!(matches!(failure, RequestFailure::Params(_)));

        let matrix = emvp::EncryptedMatrix::from_parts(1, 3, 15, values(45, 5)).unwrap();
        let mismatched = UploadMatrixView {
            params: good.params,
            instance_id: 1,
            rows: 3,
            columns: 15,
            values: matrix.values(),
        };
        let failure = with_decoded_uploads(&[mismatched], validate_upload_views).unwrap_err();
        assert!(matches!(
            failure,
            RequestFailure::ColumnsMismatch {
                expected: 16,
                actual: 15
            }
        ));
    }

    #[test]
    fn an_empty_upload_set_is_rejected_by_the_validator() {
        let failure = with_decoded_uploads(&[], validate_upload_views).unwrap_err();
        assert!(matches!(failure, RequestFailure::EmptyRequest));
    }

    /// One evaluation request fixture: a matrix identifier plus its owned
    /// queries.
    struct EntryFixture {
        matrix_id: u64,
        queries: Vec<EncryptedQuery<PROTOCOL_MODULUS>>,
    }

    fn entry(matrix_id: u64, queries: Vec<EncryptedQuery<PROTOCOL_MODULUS>>) -> EntryFixture {
        EntryFixture { matrix_id, queries }
    }

    /// Encodes `entries` with the v2 writer and decodes them through the
    /// v2 pipeline, mirroring the server's per-frame path, then hands the
    /// decoded views and an empty query-ref scratch to `consume`.
    fn with_decoded_entries<T>(
        entries: &[EntryFixture],
        consume: impl for<'v> FnOnce(
            EvaluateViews<'v>,
            &mut Vec<EncryptedQueryRef<'v, PROTOCOL_MODULUS>>,
        ) -> T,
    ) -> T {
        let query_refs: Vec<Vec<EncryptedQueryRef<'_, PROTOCOL_MODULUS>>> = entries
            .iter()
            .map(|fixture| {
                fixture
                    .queries
                    .iter()
                    .map(EncryptedQueryRef::from)
                    .collect()
            })
            .collect();
        let inputs: Vec<EvaluateEntryInput<'_>> = entries
            .iter()
            .zip(&query_refs)
            .map(|(fixture, queries)| EvaluateEntryInput {
                matrix_id: fixture.matrix_id,
                queries,
            })
            .collect();
        let mut bytes = Vec::new();
        write_evaluate(&mut bytes, &inputs).unwrap();
        let mut reader = &bytes[..];
        let header = read_frame_header(&mut reader).unwrap().unwrap();
        assert_eq!(header.kind, FrameKind::Evaluate);
        let mut frame = FrameReader::new(&mut reader, header.payload_len);
        let plan = plan_evaluate(&mut frame).unwrap();
        let mut workspace = EvaluateWorkspace::new();
        workspace.reserve(&plan).unwrap();
        let views = decode_evaluate(&mut frame, &plan, &mut workspace).unwrap();
        let mut query_refs = Vec::new();
        consume(views, &mut query_refs)
    }

    /// Runs the session's evaluation validator over a fixture request.
    /// Returns one `(matrix id, query count)` pair per entry, in request
    /// order.
    fn prepared_evaluate(
        entries: &[EntryFixture],
        lookup: &mut dyn FnMut(u64) -> Option<MatrixInfo>,
    ) -> Result<Vec<(u64, usize)>, RequestFailure> {
        with_decoded_entries(entries, |views, query_refs| {
            let mut seen_matrices = HashSet::new();
            let mut seen_queries = HashSet::new();
            prepare_evaluate(
                views,
                lookup,
                &mut seen_matrices,
                &mut seen_queries,
                query_refs,
            )
            .map(|prepared: Vec<EvaluatedEntry>| {
                prepared
                    .into_iter()
                    .map(|entry| (entry.matrix_id, entry.queries.len()))
                    .collect()
            })
        })
    }

    #[test]
    fn evaluate_preparation_enforces_the_request_contract() {
        let mut lookup = |matrix_id: u64| {
            if matrix_id == 7 {
                Some(MatrixInfo {
                    instance_id: 42,
                    width: 16,
                })
            } else {
                None
            }
        };
        let make_query = |query_id: u64, width: usize, instance_id: u128| {
            emvp::EncryptedQuery::from_parts(instance_id, query_id, values(width, query_id))
        };

        let ok = prepared_evaluate(
            &[entry(7, vec![make_query(0, 16, 42), make_query(1, 16, 42)])],
            &mut lookup,
        )
        .unwrap();
        assert_eq!(ok.len(), 1);
        assert_eq!(ok[0].0, 7);
        assert_eq!(ok[0].1, 2);

        assert!(matches!(
            prepared_evaluate(&[], &mut lookup),
            Err(RequestFailure::EmptyRequest)
        ));
        assert!(matches!(
            prepared_evaluate(&[entry(9, vec![make_query(0, 16, 42)])], &mut lookup),
            Err(RequestFailure::UnknownMatrix { id: 9 })
        ));
        assert!(matches!(
            prepared_evaluate(
                &[
                    entry(7, vec![make_query(0, 16, 42)]),
                    entry(7, vec![make_query(1, 16, 42)]),
                ],
                &mut lookup,
            ),
            Err(RequestFailure::DuplicateMatrix { id: 7 })
        ));
        assert!(matches!(
            prepared_evaluate(&[entry(7, vec![make_query(3, 15, 42)])], &mut lookup,),
            Err(RequestFailure::QueryWidth {
                expected: 16,
                actual: 15
            })
        ));
        assert!(matches!(
            prepared_evaluate(&[entry(7, vec![make_query(3, 16, 43)])], &mut lookup,),
            Err(RequestFailure::InstanceMismatch {
                expected: 42,
                actual: 43
            })
        ));
        assert!(matches!(
            prepared_evaluate(
                &[entry(
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
            prepared_evaluate(&[entry(7, vec![])], &mut lookup),
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
        answer: &AnswerRef<PROTOCOL_MODULUS>,
        key: &DecodingKey<PROTOCOL_MODULUS>,
    ) -> Vec<FieldElement<PROTOCOL_MODULUS>> {
        let mut output = vec![field().element_u32(0); answer.rows()];
        emvp::decode_into(answer, key, &mut output).unwrap();
        output
    }

    /// Reads the server's products answer into a fresh decode workspace
    /// and hands the views to `consume`; tests may allocate.
    fn read_products<T>(
        client: &mut UnixStream,
        consume: impl FnOnce(ProductsViews<'_>) -> T,
    ) -> T {
        let header = read_frame_header(client).unwrap().unwrap();
        assert_eq!(header.kind, FrameKind::Products);
        let mut frame = FrameReader::new(client, header.payload_len);
        let plan = plan_products(&mut frame).unwrap();
        let mut workspace = ProductsWorkspace::new();
        workspace.reserve(&plan).unwrap();
        let views = decode_products(&mut frame, &plan, &mut workspace).unwrap();
        consume(views)
    }

    #[test]
    fn full_session_uploads_evaluates_and_answers_correctly() {
        let engine = Arc::new(AnswerEngine::cpu());
        let (mut client, server_stream) = UnixStream::pair().unwrap();
        let server = spawn_server(server_stream, engine);
        client_handshake(&mut client).unwrap();

        let mut loaded_a = upload(5, 0x11, 100);
        let mut loaded_b = upload(21, 0x22, 101);
        write_upload_matrices(&mut client, &[loaded_a.view(), loaded_b.view()]).unwrap();
        let (ids, _) = read_upload_accepted(&mut client).unwrap();
        assert_eq!(ids, vec![1, 2]);

        let (query_a0, key_a0, plain_a0) = random_query(&mut loaded_a.state, 200);
        let (query_a1, key_a1, plain_a1) = random_query(&mut loaded_a.state, 201);
        let (query_of_b, key_of_b, plain_b) = random_query(&mut loaded_b.state, 202);
        let queries_a: Vec<EncryptedQueryRef<'_, PROTOCOL_MODULUS>> = [&query_a0, &query_a1]
            .into_iter()
            .map(EncryptedQueryRef::from)
            .collect();
        let queries_b: Vec<EncryptedQueryRef<'_, PROTOCOL_MODULUS>> = std::iter::once(&query_of_b)
            .map(EncryptedQueryRef::from)
            .collect();
        let entries = [
            EvaluateEntryInput {
                matrix_id: ids[0],
                queries: &queries_a,
            },
            EvaluateEntryInput {
                matrix_id: ids[1],
                queries: &queries_b,
            },
        ];
        write_evaluate(&mut client, &entries).unwrap();
        read_products(&mut client, |products| {
            assert_eq!(products.len(), 2);

            let first = products.get(0).unwrap();
            assert_eq!(first.matrix_id(), ids[0]);
            assert_eq!(first.len(), 2);
            for ((answer, key), plaintext_query) in first
                .iter()
                .zip([&key_a0, &key_a1])
                .zip([&plain_a0, &plain_a1])
            {
                assert_eq!(answer.instance_id(), loaded_a.matrix.instance_id());
                assert_eq!(answer.query_id(), key.query_id());
                assert_eq!(answer.rows(), 5);
                assert_eq!(answer.blocks(), 8);
                let decoded_product = decoded(&answer, key);
                assert_eq!(
                    decoded_product,
                    naive_product(&loaded_a.plaintext, plaintext_query, 5)
                );
            }
            let second = products.get(1).unwrap();
            assert_eq!(second.matrix_id(), ids[1]);
            assert_eq!(second.len(), 1);
            assert_eq!(second.rows(), 21);
            let answer = second.answer(0).unwrap();
            let decoded_product = decoded(&answer, &key_of_b);
            assert_eq!(
                decoded_product,
                naive_product(&loaded_b.plaintext, &plain_b, 21)
            );
        });

        drop(client);
        server.join().unwrap().unwrap();
    }

    /// Two consecutive `Evaluate` frames of the same shape run over one
    /// connection: the second answers through the workspaces the first
    /// already reserved.
    #[test]
    fn two_consecutive_same_shape_evaluates_reuse_the_session_workspaces() {
        let engine = Arc::new(AnswerEngine::cpu());
        let (mut client, server_stream) = UnixStream::pair().unwrap();
        let server = spawn_server(server_stream, engine);
        client_handshake(&mut client).unwrap();

        let mut loaded_a = upload(5, 0x11, 110);
        let mut loaded_b = upload(21, 0x22, 111);
        write_upload_matrices(&mut client, &[loaded_a.view(), loaded_b.view()]).unwrap();
        let (ids, _) = read_upload_accepted(&mut client).unwrap();
        assert_eq!(ids, vec![1, 2]);

        for round in 0_u64..2 {
            let (query_a0, key_a0, plain_a0) = random_query(&mut loaded_a.state, 200 + round);
            let (query_a1, key_a1, plain_a1) = random_query(&mut loaded_a.state, 210 + round);
            let (query_of_b, key_of_b, plain_b) = random_query(&mut loaded_b.state, 220 + round);
            let queries_a: Vec<EncryptedQueryRef<'_, PROTOCOL_MODULUS>> = [&query_a0, &query_a1]
                .into_iter()
                .map(EncryptedQueryRef::from)
                .collect();
            let queries_b: Vec<EncryptedQueryRef<'_, PROTOCOL_MODULUS>> =
                std::iter::once(&query_of_b)
                    .map(EncryptedQueryRef::from)
                    .collect();
            let entries = [
                EvaluateEntryInput {
                    matrix_id: ids[0],
                    queries: &queries_a,
                },
                EvaluateEntryInput {
                    matrix_id: ids[1],
                    queries: &queries_b,
                },
            ];
            write_evaluate(&mut client, &entries).unwrap();
            read_products(&mut client, |products| {
                assert_eq!(products.len(), 2);

                let first = products.get(0).unwrap();
                assert_eq!(first.matrix_id(), ids[0]);
                assert_eq!(first.len(), 2);
                for ((answer, key), plaintext_query) in first
                    .iter()
                    .zip([&key_a0, &key_a1])
                    .zip([&plain_a0, &plain_a1])
                {
                    assert_eq!(answer.instance_id(), loaded_a.matrix.instance_id());
                    assert_eq!(answer.query_id(), key.query_id());
                    assert_eq!(answer.rows(), 5);
                    assert_eq!(answer.blocks(), 8);
                    assert_eq!(
                        decoded(&answer, key),
                        naive_product(&loaded_a.plaintext, plaintext_query, 5)
                    );
                }
                let second = products.get(1).unwrap();
                assert_eq!(second.matrix_id(), ids[1]);
                assert_eq!(second.len(), 1);
                assert_eq!(
                    decoded(&second.answer(0).unwrap(), &key_of_b),
                    naive_product(&loaded_b.plaintext, &plain_b, 21)
                );
            });
        }

        drop(client);
        server.join().unwrap().unwrap();
    }

    /// A larger `Evaluate` frame grows the session's workspaces; the
    /// following smaller frame answers correctly through the same,
    /// now-oversized reservations.
    #[test]
    fn a_larger_then_smaller_evaluate_sequence_works() {
        let engine = Arc::new(AnswerEngine::cpu());
        let (mut client, server_stream) = UnixStream::pair().unwrap();
        let server = spawn_server(server_stream, engine);
        client_handshake(&mut client).unwrap();

        let mut loaded = upload(5, 0x11, 120);
        write_upload_matrices(&mut client, &[loaded.view()]).unwrap();
        let (ids, _) = read_upload_accepted(&mut client).unwrap();
        assert_eq!(ids, vec![1]);

        // The larger frame: three queries in one entry.
        let larger: Vec<_> = (0_u64..3)
            .map(|index| random_query(&mut loaded.state, 300 + index))
            .collect();
        let query_refs: Vec<EncryptedQueryRef<'_, PROTOCOL_MODULUS>> = larger
            .iter()
            .map(|(query, _, _)| EncryptedQueryRef::from(query))
            .collect();
        let entries = [EvaluateEntryInput {
            matrix_id: ids[0],
            queries: &query_refs,
        }];
        write_evaluate(&mut client, &entries).unwrap();
        read_products(&mut client, |products| {
            assert_eq!(products.len(), 1);
            let entry = products.get(0).unwrap();
            assert_eq!(entry.len(), 3);
            for ((answer, key), plain) in entry
                .iter()
                .zip(larger.iter().map(|(_, key, _)| key))
                .zip(larger.iter().map(|(_, _, plain)| plain))
            {
                assert_eq!(answer.query_id(), key.query_id());
                assert_eq!(
                    decoded(&answer, key),
                    naive_product(&loaded.plaintext, plain, 5)
                );
            }
        });

        // The smaller frame: one query, answered through the same
        // workspaces without shrinking them.
        let (query, key, plain) = random_query(&mut loaded.state, 310);
        let query_refs: Vec<EncryptedQueryRef<'_, PROTOCOL_MODULUS>> = std::iter::once(&query)
            .map(EncryptedQueryRef::from)
            .collect();
        let entries = [EvaluateEntryInput {
            matrix_id: ids[0],
            queries: &query_refs,
        }];
        write_evaluate(&mut client, &entries).unwrap();
        read_products(&mut client, |products| {
            assert_eq!(products.len(), 1);
            let entry = products.get(0).unwrap();
            assert_eq!(entry.len(), 1);
            let answer = entry.answer(0).unwrap();
            assert_eq!(answer.query_id(), key.query_id());
            assert_eq!(
                decoded(&answer, &key),
                naive_product(&loaded.plaintext, &plain, 5)
            );
        });

        drop(client);
        server.join().unwrap().unwrap();
    }

    /// An `Evaluate` frame whose value region carries a noncanonical field
    /// word is answered with the codec's structured error and closes the
    /// session, exactly like any other decode failure.
    #[test]
    fn a_corrupted_evaluate_frame_is_answered_and_closes() {
        let engine = Arc::new(AnswerEngine::cpu());
        let (mut client, server_stream) = UnixStream::pair().unwrap();
        let server = spawn_server(server_stream, engine);
        client_handshake(&mut client).unwrap();

        let mut loaded = upload(5, 0x11, 130);
        write_upload_matrices(&mut client, &[loaded.view()]).unwrap();
        let (ids, _) = read_upload_accepted(&mut client).unwrap();

        // A well-formed frame, then corrupted on the wire: the final value
        // word becomes `u32::MAX`, which is at or above the modulus.
        let (query, _, _) = random_query(&mut loaded.state, 320);
        let query_refs: Vec<EncryptedQueryRef<'_, PROTOCOL_MODULUS>> = std::iter::once(&query)
            .map(EncryptedQueryRef::from)
            .collect();
        let entries = [EvaluateEntryInput {
            matrix_id: ids[0],
            queries: &query_refs,
        }];
        let mut frame_bytes = Vec::new();
        write_evaluate(&mut frame_bytes, &entries).unwrap();
        let last_word = frame_bytes.len() - 4..;
        frame_bytes[last_word].copy_from_slice(&[0xFF, 0xFF, 0xFF, 0xFF]);
        client.write_all(&frame_bytes).unwrap();

        let (error, _) = read_error(&mut client).unwrap();
        assert_eq!(error.code(), Some(ErrorCode::NonCanonicalFieldElement));
        assert!(read_frame_header(&mut client).unwrap().is_none());

        let outcome = server.join().unwrap();
        assert!(matches!(
            outcome,
            Err(SessionError::Transport(
                CodecError::NonCanonicalField { .. }
            ))
        ));
    }

    #[test]
    fn a_failed_upload_is_atomic_and_closes_the_session() {
        let engine = Arc::new(AnswerEngine::cpu());
        let (mut client, server_stream) = UnixStream::pair().unwrap();
        let server = spawn_server(server_stream, engine);
        client_handshake(&mut client).unwrap();

        // `b = 3` does not divide `n = 16`: structurally invalid parameters.
        let uploaded = upload(5, 0x11, 400);
        let malformed = UploadMatrixView {
            params: EmvpParams {
                b: 3,
                ..uploaded.params
            },
            instance_id: uploaded.matrix.instance_id(),
            rows: uploaded.matrix.rows(),
            columns: uploaded.matrix.columns(),
            values: uploaded.matrix.values(),
        };
        write_upload_matrices(&mut client, &[malformed]).unwrap();
        let (error, _) = read_error(&mut client).unwrap();
        assert_eq!(error.code(), Some(ErrorCode::InvalidParameters));

        // The server closed the connection without assigning identifiers.
        assert!(read_frame_header(&mut client).unwrap().is_none());
        server.join().unwrap().unwrap();
    }

    /// A failed upload assigns no identifiers and leaves no prepared
    /// matrices behind: a fresh connection over the same engine uploads
    /// and evaluates normally.
    #[test]
    fn a_failed_upload_leaves_no_prepared_matrices_behind() {
        let engine = Arc::new(AnswerEngine::cpu());

        // Connection 1: a structurally invalid upload; the session closes
        // without assigning identifiers.
        let (mut client, server_stream) = UnixStream::pair().unwrap();
        let server = spawn_server(server_stream, Arc::clone(&engine));
        client_handshake(&mut client).unwrap();
        let uploaded = upload(5, 0x11, 400);
        let malformed = UploadMatrixView {
            params: EmvpParams {
                b: 3,
                ..uploaded.params
            },
            instance_id: uploaded.matrix.instance_id(),
            rows: uploaded.matrix.rows(),
            columns: uploaded.matrix.columns(),
            values: uploaded.matrix.values(),
        };
        write_upload_matrices(&mut client, &[malformed]).unwrap();
        let (error, _) = read_error(&mut client).unwrap();
        assert_eq!(error.code(), Some(ErrorCode::InvalidParameters));
        assert!(read_frame_header(&mut client).unwrap().is_none());
        server.join().unwrap().unwrap();
        drop(client);

        // Connection 2 over the same engine: the whole flow succeeds.
        let (mut client, server_stream) = UnixStream::pair().unwrap();
        let server = spawn_server(server_stream, engine);
        client_handshake(&mut client).unwrap();
        let mut loaded = upload(5, 0x22, 401);
        write_upload_matrices(&mut client, &[loaded.view()]).unwrap();
        let (ids, _) = read_upload_accepted(&mut client).unwrap();
        assert_eq!(ids, vec![1]);
        let (query, key, plain) = random_query(&mut loaded.state, 402);
        let query_refs: Vec<EncryptedQueryRef<'_, PROTOCOL_MODULUS>> = std::iter::once(&query)
            .map(EncryptedQueryRef::from)
            .collect();
        let entries = [EvaluateEntryInput {
            matrix_id: ids[0],
            queries: &query_refs,
        }];
        write_evaluate(&mut client, &entries).unwrap();
        read_products(&mut client, |products| {
            let entry = products.get(0).unwrap();
            assert_eq!(entry.len(), 1);
            assert_eq!(
                decoded(&entry.answer(0).unwrap(), &key),
                naive_product(&loaded.plaintext, &plain, 5)
            );
        });

        drop(client);
        server.join().unwrap().unwrap();
    }

    #[test]
    fn a_second_upload_is_rejected_and_closes_the_session() {
        let engine = Arc::new(AnswerEngine::cpu());
        let (mut client, server_stream) = UnixStream::pair().unwrap();
        let server = spawn_server(server_stream, engine);
        client_handshake(&mut client).unwrap();

        let first = upload(5, 0x11, 300);
        write_upload_matrices(&mut client, &[first.view()]).unwrap();
        let (ids, _) = read_upload_accepted(&mut client).unwrap();
        assert_eq!(ids, vec![1]);

        // The server answers the state violation and closes immediately;
        // the second payload is never read or drained. The client sends
        // only the frame header, so the close carries no unread data and
        // the error frame is delivered deterministically.
        let second = upload(7, 0x33, 301);
        let payload_len =
            write_upload_matrices(&mut Vec::new(), &[second.view()]).unwrap() - HEADER_BYTES;
        write_frame_header(&mut client, FrameKind::UploadMatrices, payload_len).unwrap();
        let (error, _) = read_error(&mut client).unwrap();
        assert_eq!(error.code(), Some(ErrorCode::AlreadyLoaded));
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

        let query_values = values(16, 600);
        let queries = [EncryptedQueryRef::new(1, 0, &query_values).unwrap()];
        let entries = [EvaluateEntryInput {
            matrix_id: 1,
            queries: &queries,
        }];
        // The client sends only the frame header, so the session closes
        // with no unread data and the error frame is delivered cleanly.
        let payload_len = write_evaluate(&mut Vec::new(), &entries).unwrap() - HEADER_BYTES;
        write_frame_header(&mut client, FrameKind::Evaluate, payload_len).unwrap();
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
        let answer_values = values(8, 700);
        let answer = AnswerRef::new(1, 0, &answer_values, 4, 2).unwrap();
        let products = [ProductEntryInput {
            matrix_id: 1,
            instance_id: 1,
            rows: 4,
            blocks: 2,
            answers: std::slice::from_ref(&answer),
        }];
        let mut buffer = Vec::new();
        let written = write_products(&mut buffer, &products).unwrap();
        write_frame_header(&mut client, FrameKind::Products, written - HEADER_BYTES).unwrap();
        let (error, _) = read_error(&mut client).unwrap();
        assert_eq!(error.code(), Some(ErrorCode::UnexpectedFrame));

        // The server closed the connection after the violation.
        assert!(read_frame_header(&mut client).unwrap().is_none());
        let outcome = server.join().unwrap();
        assert!(matches!(
            outcome,
            Err(SessionError::Transport(CodecError::UnexpectedFrame { .. }))
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
            Err(SessionError::Transport(CodecError::UnknownFrameKind {
                kind: 0x2A
            }))
        ));
    }
}
