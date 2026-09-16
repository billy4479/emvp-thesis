//! Backend-agnostic orchestration for answering several encrypted matrices.
//!
//! The engine offers two execution paths over the same validation. The
//! allocating path ([`AnswerEngine::answer_many`]) validates, allocates,
//! fills, and collects per-call. The plan → reserve → execute path
//! ([`AnswerEngine::plan`], [`EngineWorkspace::reserve`],
//! [`AnswerEngine::execute`]) splits the same work so a steady-state
//! server cycle allocates nothing project-owned: the plan binds the exact
//! inputs by borrowing (executing against different inputs is
//! unrepresentable), the workspace grows only through `reserve` and never
//! shrinks, and `execute` fills the caller's arena in place — CPU entries
//! through the shared CPU row kernels, GPU entries through one packed
//! device flight streaming into absolute arena offsets.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use prime_field_layer::{FieldElement, PrimeField};
use rayon::prelude::*;

use crate::answer::{AnswerShape, Answers};
#[cfg(feature = "gpu")]
use crate::dispatch::select_answer_backend;
use crate::dispatch::{AnswerBackend, select_cpu_backend};
#[cfg(feature = "gpu")]
use crate::gpu::packed::{PackedJob, PackedPlan, PackedStats, PackedTimings};
#[cfg(feature = "gpu")]
use crate::gpu::{GpuAnswerer, GpuEncryptedMatrix, GpuError};
use crate::protocol::{fill_answer_batch_serial, fill_answer_row, validate_query_against_matrix};
use crate::view::QueryValues;
use crate::{
    AnswerMatrix, EmvpParams, EncryptedMatrix, EncryptedQuery, ProtocolError, answer_batch,
};

static NEXT_ENGINE_ID: AtomicU64 = AtomicU64::new(1);
static NEXT_MATRIX_ID: AtomicU64 = AtomicU64::new(1);

/// An answer runner and the resources shared by its prepared matrices.
pub struct AnswerEngine<const MODULUS: u32> {
    id: u64,
    #[cfg(feature = "gpu")]
    gpu: Option<Arc<GpuEngine>>,
}

#[cfg(feature = "gpu")]
struct GpuEngine {
    answerer: GpuAnswerer,
    max_bytes: u64,
    reserved_bytes: AtomicU64,
}

/// A matrix validated and retained for repeated answer operations.
#[derive(Clone)]
pub struct PreparedMatrix<const MODULUS: u32> {
    inner: Arc<PreparedMatrixInner<MODULUS>>,
}

struct PreparedMatrixInner<const MODULUS: u32> {
    id: u64,
    engine_id: u64,
    params: EmvpParams,
    matrix: EncryptedMatrix<MODULUS>,
    #[cfg(feature = "gpu")]
    gpu_matrix: Option<GpuEncryptedMatrix<MODULUS>>,
    #[cfg(feature = "gpu")]
    _reservation: Option<Arc<GpuReservation>>,
}

#[cfg(feature = "gpu")]
struct GpuReservation {
    engine: Arc<GpuEngine>,
    bytes: u64,
}

#[cfg(feature = "gpu")]
impl Drop for GpuReservation {
    fn drop(&mut self) {
        self.engine
            .reserved_bytes
            .fetch_sub(self.bytes, Ordering::AcqRel);
    }
}

/// One entry in a multi-matrix answer operation.
///
/// Generic over the query representation `Q` (the owned
/// [`EncryptedQuery`] by default, or the borrowed
/// [`EncryptedQueryRef`](crate::view::EncryptedQueryRef) wire view), so
/// both the allocating and the plan → reserve → execute paths answer
/// batches without copying queries into owned storage.
pub struct AnswerJob<'a, const MODULUS: u32, Q: QueryValues<MODULUS> = EncryptedQuery<MODULUS>> {
    /// The prepared matrix to evaluate.
    pub matrix: &'a PreparedMatrix<MODULUS>,
    /// The encrypted queries to answer, in output order.
    pub queries: &'a [Q],
}

/// Why a multi-matrix operation was rejected.
#[derive(Debug)]
#[non_exhaustive]
pub enum AnswerEngineError {
    /// The operation contained no jobs.
    EmptyJobs,
    /// One job contained no queries.
    EmptyQueries { entry: usize },
    /// A prepared matrix occurred more than once.
    RepeatedMatrix { entry: usize },
    /// A matrix was prepared by another engine.
    ForeignMatrix { entry: usize },
    /// Protocol validation or computation failed for one entry.
    Protocol { entry: usize, source: ProtocolError },
    /// A workspace refused the arena growth an operation needs, or an
    /// execute call found the workspace's arena too small. Reported before
    /// any arena mutation, so the caller can grow the workspace and retry
    /// whole.
    Capacity {
        /// The required arena word count.
        required: usize,
        /// The available arena word count.
        available: usize,
    },
    /// Device execution failed after all jobs had been validated.
    #[cfg(feature = "gpu")]
    Gpu(GpuError),
    /// An internal execution invariant was violated without producing partial output.
    InternalState(&'static str),
}

/// A failure while validating and retaining a matrix.
#[derive(Debug)]
#[non_exhaustive]
pub enum PrepareMatrixError {
    /// The matrix or its protocol parameters are malformed.
    Protocol(ProtocolError),
    /// The matrix could not be reserved or uploaded to the device.
    #[cfg(feature = "gpu")]
    Gpu(GpuError),
    /// An internal execution invariant was violated without producing a
    /// handle.
    InternalState(&'static str),
}

impl fmt::Display for PrepareMatrixError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Protocol(error) => error.fmt(formatter),
            #[cfg(feature = "gpu")]
            Self::Gpu(error) => error.fmt(formatter),
            Self::InternalState(reason) => write!(formatter, "answer engine state error: {reason}"),
        }
    }
}

impl std::error::Error for PrepareMatrixError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Protocol(error) => Some(error),
            #[cfg(feature = "gpu")]
            Self::Gpu(error) => Some(error),
            Self::InternalState(_) => None,
        }
    }
}

impl From<ProtocolError> for PrepareMatrixError {
    fn from(error: ProtocolError) -> Self {
        Self::Protocol(error)
    }
}

#[cfg(feature = "gpu")]
impl From<GpuError> for PrepareMatrixError {
    fn from(error: GpuError) -> Self {
        Self::Gpu(error)
    }
}

/// A failure while validating, reserving, or uploading a whole matrix set.
///
/// Every failure is all-or-nothing: no handle of the rejected set is
/// returned and none of its residency stays reserved.
#[derive(Debug)]
#[non_exhaustive]
pub enum PrepareBatchError {
    /// The batch contained no matrices.
    Empty,
    /// The matrix at `index`, or its protocol parameters, are malformed.
    /// Nothing was reserved or uploaded.
    Protocol {
        /// The input position of the rejected matrix.
        index: usize,
        /// The validation failure.
        source: ProtocolError,
    },
    /// The whole set does not fit the engine's device residency budget.
    /// Nothing was uploaded.
    #[cfg(feature = "gpu")]
    Budget(GpuError),
    /// The matrix at `index` failed its device upload. The handles already
    /// prepared for earlier entries were discarded together with the whole
    /// set's reservation.
    #[cfg(feature = "gpu")]
    Upload {
        /// The input position of the failed matrix.
        index: usize,
        /// The device failure.
        source: GpuError,
    },
    /// An internal execution invariant was violated without producing
    /// partial output.
    InternalState(&'static str),
}

impl fmt::Display for PrepareBatchError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Empty => formatter.write_str("matrix batch is empty"),
            Self::Protocol { index, source } => write!(formatter, "batch matrix {index}: {source}"),
            #[cfg(feature = "gpu")]
            Self::Budget(error) => error.fmt(formatter),
            #[cfg(feature = "gpu")]
            Self::Upload { index, source } => {
                write!(formatter, "batch matrix {index}: {source}")
            }
            Self::InternalState(reason) => write!(formatter, "answer engine state error: {reason}"),
        }
    }
}

impl std::error::Error for PrepareBatchError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Protocol { source, .. } => Some(source),
            #[cfg(feature = "gpu")]
            Self::Budget(error) | Self::Upload { source: error, .. } => Some(error),
            Self::Empty | Self::InternalState(_) => None,
        }
    }
}

impl From<PrepareBatchError> for PrepareMatrixError {
    fn from(error: PrepareBatchError) -> Self {
        match error {
            PrepareBatchError::Protocol { source, .. } => Self::Protocol(source),
            #[cfg(feature = "gpu")]
            PrepareBatchError::Budget(source) | PrepareBatchError::Upload { source, .. } => {
                Self::Gpu(source)
            }
            PrepareBatchError::Empty => {
                Self::InternalState("an empty batch reached the single-matrix preparation path")
            }
            PrepareBatchError::InternalState(reason) => Self::InternalState(reason),
        }
    }
}

impl fmt::Display for AnswerEngineError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::EmptyJobs => formatter.write_str("multi-matrix answer request is empty"),
            Self::EmptyQueries { entry } => {
                write!(formatter, "answer entry {entry} has no queries")
            }
            Self::RepeatedMatrix { entry } => {
                write!(formatter, "answer entry {entry} repeats a prepared matrix")
            }
            Self::ForeignMatrix { entry } => {
                write!(formatter, "answer entry {entry} belongs to another engine")
            }
            Self::Protocol { entry, source } => write!(formatter, "answer entry {entry}: {source}"),
            Self::Capacity {
                required,
                available,
            } => write!(
                formatter,
                "engine answer workspace holds {available} words but the plan needs {required}"
            ),
            #[cfg(feature = "gpu")]
            Self::Gpu(error) => error.fmt(formatter),
            Self::InternalState(reason) => write!(formatter, "answer engine state error: {reason}"),
        }
    }
}

impl std::error::Error for AnswerEngineError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Protocol { source, .. } => Some(source),
            #[cfg(feature = "gpu")]
            Self::Gpu(error) => Some(error),
            _ => None,
        }
    }
}

/// Execution statistics for one input entry.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnswerEntryReport {
    /// The CPU tier selected for this entry.
    pub backend: AnswerBackend,
    /// Number of queries in the entry.
    pub queries: usize,
    /// Estimated field multiplications.
    pub multiplications: usize,
    /// Number of query field words.
    pub query_words: usize,
    /// Number of answer field words.
    pub answer_words: usize,
    /// Number of GPU segments; zero on a CPU engine.
    pub gpu_segments: usize,
}

/// Wall-clock and shape statistics for one multi-matrix operation.
#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct AnswerReport {
    /// Reports in input order.
    pub entries: Vec<AnswerEntryReport>,
    /// Entries evaluated on the CPU.
    pub cpu_entries: usize,
    /// Entries evaluated on the GPU.
    pub gpu_entries: usize,
    /// Sum of all estimated field multiplications.
    pub multiplications: usize,
    /// Queries evaluated on the GPU.
    pub gpu_queries: usize,
    /// Bytes occupied by live query words in GPU packs, including alignment gaps.
    pub gpu_query_bytes: u64,
    /// Bytes occupied by live answer words in GPU packs, including alignment gaps.
    pub gpu_answer_bytes: u64,
    /// Number of packed GPU chunks.
    pub gpu_chunks: usize,
    /// Number of GPU dispatch segments across all chunks.
    pub gpu_segments: usize,
    /// Time spent validating and planning.
    pub planning: Duration,
    /// Time spent evaluating CPU entries.
    pub cpu_compute: Duration,
    /// Time leasing and growing packed device buffers.
    pub gpu_prepare_buffers: Duration,
    /// Time encoding packed query uploads.
    pub gpu_upload: Duration,
    /// Time recording and submitting the single GPU command buffer.
    pub gpu_submit: Duration,
    /// Time waiting for mapped GPU readback.
    pub gpu_wait: Duration,
    /// Time reconstructing GPU answers.
    pub gpu_reconstruct: Duration,
    /// End-to-end wall time.
    pub total: Duration,
}

/// The validated execution facts of one entry in an [`EnginePlan`].
///
/// The plan phase produces one per input entry and the entry stays bound
/// to its inputs by borrowing: the matrix and the queries are shared
/// references, so executing the plan against different inputs is
/// unrepresentable.
#[derive(Clone, Copy)]
pub struct EntryPlan<'a, const MODULUS: u32, Q: QueryValues<MODULUS>> {
    matrix: &'a PreparedMatrix<MODULUS>,
    queries: &'a [Q],
    shape: AnswerShape,
    backend: AnswerBackend,
    offset_words: usize,
    multiplications: usize,
    gpu_segments: usize,
}

impl<'a, const MODULUS: u32, Q: QueryValues<MODULUS>> EntryPlan<'a, MODULUS, Q> {
    /// The entry's borrowed queries, in output order.
    #[must_use]
    pub const fn queries(&self) -> &'a [Q] {
        self.queries
    }

    /// The entry's prepared matrix.
    #[must_use]
    pub const fn matrix(&self) -> &'a PreparedMatrix<MODULUS> {
        self.matrix
    }

    /// The validated answer shape: rows, blocks, query count, and the
    /// public instance identifier.
    #[must_use]
    pub const fn shape(&self) -> AnswerShape {
        self.shape
    }

    /// The backend this entry will run on, fixed from a rayon pool
    /// snapshot at plan time.
    #[must_use]
    pub const fn backend(&self) -> AnswerBackend {
        self.backend
    }

    /// The entry's arena start: the number of answer words planned before
    /// this entry, so consecutive entries occupy disjoint consecutive
    /// arena ranges.
    #[must_use]
    pub const fn offset_words(&self) -> usize {
        self.offset_words
    }

    /// The estimated field-multiplication count of the entry.
    #[must_use]
    pub const fn multiplications(&self) -> usize {
        self.multiplications
    }

    /// The number of GPU dispatch segments planned for the entry; zero for
    /// CPU entries.
    #[must_use]
    pub const fn gpu_segments(&self) -> usize {
        self.gpu_segments
    }
}

/// The packed device flight of a plan's GPU-tier entries.
///
/// Built once in the plan phase: the packed chunk plan, the jobs (which
/// borrow the GPU-resident matrices and the entries' query slices), and
/// each job's absolute arena base in engine-entry order, so execute can
/// stream one flight straight into the interleaved arena without building
/// anything.
#[cfg(feature = "gpu")]
struct PackedGpuPlan<'a, const MODULUS: u32, Q: QueryValues<MODULUS>> {
    packed: PackedPlan,
    jobs: Vec<PackedJob<'a, MODULUS, Q>>,
    /// Absolute arena base per packed job, in subset order: the engine
    /// entry's `offset_words`.
    bases: Vec<usize>,
}

/// A fully validated multi-matrix answer plan bound to its inputs by
/// borrowing.
///
/// [`AnswerEngine::plan`] runs the same all-or-nothing validation as
/// [`AnswerEngine::answer_many`] and additionally fixes, per entry, the
/// answer shape, the backend under a snapshot of the current rayon pool,
/// and the entry's arena offset. The plan holds every input only by
/// reference, so executing it against other inputs is unrepresentable; a
/// plan can be executed repeatedly and yields identical answers every
/// time.
pub struct EnginePlan<'a, const MODULUS: u32, Q: QueryValues<MODULUS>> {
    entries: Vec<EntryPlan<'a, MODULUS, Q>>,
    total_words: usize,
    /// The packed device flight over the plan's GPU-tier entries, in entry
    /// order; `None` when no entry selected the GPU tier.
    #[cfg(feature = "gpu")]
    gpu: Option<PackedGpuPlan<'a, MODULUS, Q>>,
}

impl<'a, const MODULUS: u32, Q: QueryValues<MODULUS>> EnginePlan<'a, MODULUS, Q> {
    /// The number of planned entries.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.entries.len()
    }

    /// Whether the plan has no entries; [`AnswerEngine::plan`] rejects
    /// empty job lists, so an executed plan is never empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// The exact field-element word count
    /// [`AnswerEngine::execute`] writes: the sum of every entry's
    /// query-major answer words.
    #[must_use]
    pub const fn total_words(&self) -> usize {
        self.total_words
    }

    /// The `index`th planned entry, or `None` when out of range.
    #[must_use]
    pub fn entry(&self, index: usize) -> Option<&EntryPlan<'a, MODULUS, Q>> {
        self.entries.get(index)
    }

    /// Iterates the planned entries in input order.
    pub fn iter(&self) -> std::slice::Iter<'_, EntryPlan<'a, MODULUS, Q>> {
        self.entries.iter()
    }
}

impl<'a, const MODULUS: u32, Q: QueryValues<MODULUS>> IntoIterator
    for &'a EnginePlan<'a, MODULUS, Q>
{
    type Item = &'a EntryPlan<'a, MODULUS, Q>;
    type IntoIter = std::slice::Iter<'a, EntryPlan<'a, MODULUS, Q>>;

    fn into_iter(self) -> Self::IntoIter {
        self.entries.iter()
    }
}

/// Wall-clock and shape statistics for one plan-execute operation.
///
/// A plain copyable aggregate: the per-entry facts (backend, shapes,
/// offsets, segment counts) live in the [`EnginePlan`], which is the
/// artifact a caller keeps, and this report only summarizes the execute
/// call itself.
#[derive(Clone, Copy, Debug)]
pub struct EngineReport {
    /// Time spent on execute-phase plan bookkeeping: the arena capacity
    /// check and entry metadata. The plan itself is built in
    /// [`AnswerEngine::plan`], before this timer starts.
    pub planning: Duration,
    /// Time spent evaluating CPU entries.
    pub cpu_compute: Duration,
    /// Time leasing and growing packed device buffers.
    pub gpu_prepare_buffers: Duration,
    /// Time encoding packed query uploads.
    pub gpu_upload: Duration,
    /// Time recording and submitting the single GPU command buffer.
    pub gpu_submit: Duration,
    /// Time waiting for mapped GPU readback.
    pub gpu_wait: Duration,
    /// Time reconstructing GPU answers into the engine arena.
    pub gpu_reconstruct: Duration,
    /// End-to-end wall time of the execute call.
    pub total: Duration,
    /// Entries evaluated on the GPU.
    pub gpu_entries: usize,
    /// Entries evaluated on the CPU.
    pub cpu_entries: usize,
    /// Queries evaluated on the GPU.
    pub gpu_queries: usize,
    /// Sum of all estimated field multiplications.
    pub multiplications: usize,
    /// Number of packed GPU chunks; zero without GPU entries.
    pub gpu_chunks: usize,
    /// Number of GPU dispatch segments across all chunks.
    pub gpu_segments: usize,
    /// Bytes occupied by live query words in GPU packs, including alignment gaps.
    pub gpu_query_bytes: u64,
    /// Bytes occupied by live answer words in GPU packs, including alignment gaps.
    pub gpu_answer_bytes: u64,
}

/// Reusable staging for one answer row of the engine's parallel CPU
/// kernel.
///
/// The staging row is filled by the shared row kernel and then copied into
/// its arena chunk, so leased scratch never aliases the arena. The buffer
/// serves the widest row its leases have seen; it is returned to the pool
/// afterwards.
#[derive(Debug)]
struct EngineRowScratch<const MODULUS: u32> {
    row: Vec<FieldElement<MODULUS>>,
}

/// A pool of row staging buffers leased by the engine's parallel CPU
/// kernel, mirroring the scratch pool of
/// [`crate::answer::AnswerWorkspace`]: a call pops a staging row and
/// returns it afterwards, so after a warm-up call the pool serves every
/// lease without allocating.
#[derive(Debug)]
struct RowScratchPool<const MODULUS: u32> {
    rows: Mutex<Vec<EngineRowScratch<MODULUS>>>,
}

impl<const MODULUS: u32> RowScratchPool<MODULUS> {
    /// Pops a staging row of exactly `answer_words` elements.
    ///
    /// An empty pool, or one whose returned rows have a different width,
    /// pays one allocation; a warm pool serves the shape untouched.
    fn pop(&self, answer_words: usize) -> EngineRowScratch<MODULUS> {
        let mut scratch = engine_lock_recovered(&self.rows)
            .pop()
            .unwrap_or_else(|| EngineRowScratch { row: Vec::new() });
        if scratch.row.len() != answer_words {
            let zero = PrimeField::<MODULUS>::new().element_u32(0);
            scratch.row.resize(answer_words, zero);
        }
        scratch
    }

    /// Returns a spent staging row to the pool.
    fn push(&self, scratch: EngineRowScratch<MODULUS>) {
        engine_lock_recovered(&self.rows).push(scratch);
    }
}

/// Locks `mutex`, recovering from poisoning like the workspace scratch
/// pools elsewhere in the crate: the pooled staging rows are
/// self-consistent between calls, so a panic in another thread mid-lease
/// cannot have corrupted anything.
fn engine_lock_recovered<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// A caller-owned reusable multi-matrix answer arena plus the parallel
/// CPU kernel's scratch pool.
///
/// The workspace owns the flat answer arena shared by every entry of an
/// [`EnginePlan`], laid out entry by entry in plan order. Its capacity
/// only ever grows, and only through [`Self::reserve`]: a steady-state
/// plan-reserve-execute cycle never allocates. The arena never shrinks, so
/// the workspace keeps the peak words a server actually uses. The
/// workspace additionally pools the per-row staging the parallel CPU tier
/// leases, mirroring [`crate::answer::AnswerWorkspace`].
#[derive(Debug)]
pub struct EngineWorkspace<const MODULUS: u32> {
    arena: Vec<FieldElement<MODULUS>>,
    row_scratch: RowScratchPool<MODULUS>,
}

impl<const MODULUS: u32> Default for EngineWorkspace<MODULUS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const MODULUS: u32> EngineWorkspace<MODULUS> {
    /// Creates an empty workspace.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            arena: Vec::new(),
            row_scratch: RowScratchPool {
                rows: Mutex::new(Vec::new()),
            },
        }
    }

    /// Grows the arena so it holds `plan`'s answers, and never shrinks it.
    ///
    /// This is the only growth point of the workspace: the growth is
    /// requested with [`Vec::try_reserve_exact`] so allocation failure is
    /// an error instead of an abort, and the arena is then resized to the
    /// plan's word count with zero elements so safe slices exist before
    /// [`AnswerEngine::execute`] writes into them. Reserving a plan that
    /// fits the current capacity is free and leaves the arena's storage
    /// pointer and capacity untouched.
    ///
    /// # Errors
    ///
    /// Returns [`AnswerEngineError::Capacity`] when the allocator refuses
    /// the growth, naming the plan's requirement and the workspace's
    /// usable words.
    pub fn reserve<Q: QueryValues<MODULUS>>(
        &mut self,
        plan: &EnginePlan<'_, MODULUS, Q>,
    ) -> Result<(), AnswerEngineError> {
        let required = plan.total_words;
        let available = self.arena.len();
        if required <= available {
            return Ok(());
        }
        self.arena
            .try_reserve_exact(required - available)
            .map_err(|_allocation_failure| AnswerEngineError::Capacity {
                required,
                available,
            })?;
        let zero = PrimeField::<MODULUS>::new().element_u32(0);
        self.arena.resize(required, zero);
        Ok(())
    }

    /// Returns the usable arena words: the high-water mark of every
    /// successful [`Self::reserve`].
    #[must_use]
    pub const fn capacity_words(&self) -> usize {
        self.arena.len()
    }

    /// Releases the workspace and returns the freed arena capacity in
    /// words.
    ///
    /// The workspace is also released through [`Drop`]; this form reports
    /// the freed capacity explicitly.
    #[must_use]
    pub fn release(self) -> usize {
        self.arena.capacity()
    }
}

/// Filled engine answers borrowed from an [`EngineWorkspace`].
///
/// No per-entry views are stored: the answers borrow the workspace's
/// filled arena immutably together with the plan that describes its
/// layout, and every entry's [`Answers`] view is built on access over that
/// entry's contiguous arena range.
pub struct EngineAnswers<'ws, 'p, const MODULUS: u32, Q: QueryValues<MODULUS>> {
    arena: &'ws [FieldElement<MODULUS>],
    plan: &'p EnginePlan<'p, MODULUS, Q>,
}

impl<'ws, const MODULUS: u32, Q: QueryValues<MODULUS>> EngineAnswers<'ws, '_, MODULUS, Q> {
    /// The number of answered entries.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.plan.entries.len()
    }

    /// Whether there are no answered entries; a plan rejects empty job
    /// lists, so executed answers are never empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.plan.entries.is_empty()
    }

    /// The filled arena's field-element word count.
    #[must_use]
    pub const fn total_words(&self) -> usize {
        self.arena.len()
    }

    /// The validated answer shape of the `index`th entry, or `None` when
    /// out of range.
    #[must_use]
    pub fn shape_of(&self, index: usize) -> Option<AnswerShape> {
        self.plan.entries.get(index).map(|entry| entry.shape)
    }

    /// The answered batch of the `index`th entry, as a view over the
    /// entry's contiguous arena range.
    ///
    /// Pair the view with [`AnswerEngine::plan`]'s borrowed queries (the
    /// same slice the entry was planned with) through
    /// [`Answers::answer`] or [`Answers::iter`] to pair every answer with
    /// its public identifiers.
    ///
    /// # Errors
    ///
    /// Returns an indexed length mismatch when `index` is out of range,
    /// and an internal-state error when a plan invariant was somehow
    /// violated; neither writes anything.
    pub fn entry_answers(&self, index: usize) -> Result<Answers<'ws, MODULUS>, AnswerEngineError> {
        let entry = self
            .plan
            .entries
            .get(index)
            .ok_or(AnswerEngineError::Protocol {
                entry: index,
                source: ProtocolError::LengthMismatch {
                    name: "entry index",
                    expected: self.plan.entries.len(),
                    actual: index,
                },
            })?;
        let words = entry
            .shape
            .arena_words()
            .ok_or(AnswerEngineError::InternalState(
                "a planned entry's arena words overflowed",
            ))?;
        let end = entry
            .offset_words
            .checked_add(words)
            .ok_or(AnswerEngineError::InternalState(
                "a planned entry's arena range overflowed",
            ))?;
        let range =
            self.arena
                .get(entry.offset_words..end)
                .ok_or(AnswerEngineError::InternalState(
                    "a planned entry escaped the engine arena",
                ))?;
        Ok(Answers::from_arena(range, entry.shape))
    }
}

impl<const MODULUS: u32> AnswerEngine<MODULUS> {
    /// Creates an engine that evaluates every matrix on the CPU.
    #[must_use]
    pub fn cpu() -> Self {
        Self {
            id: NEXT_ENGINE_ID.fetch_add(1, Ordering::Relaxed),
            #[cfg(feature = "gpu")]
            gpu: None,
        }
    }

    /// Creates a GPU-capable engine with a long-lived matrix residency budget.
    /// A machine without an adapter becomes a CPU engine; every other device
    /// initialization failure is returned.
    ///
    /// # Errors
    ///
    /// Returns device-request failures and any adapter error other than
    /// [`GpuError::NoAdapter`].
    #[cfg(feature = "gpu")]
    pub fn new(max_gpu_bytes: u64) -> Result<Self, GpuError> {
        match GpuAnswerer::new() {
            Ok(answerer) => Ok(Self {
                id: NEXT_ENGINE_ID.fetch_add(1, Ordering::Relaxed),
                gpu: Some(Arc::new(GpuEngine {
                    answerer,
                    max_bytes: max_gpu_bytes,
                    reserved_bytes: AtomicU64::new(0),
                })),
            }),
            Err(GpuError::NoAdapter { .. }) => Ok(Self::cpu()),
            Err(error) => Err(error),
        }
    }

    /// Validates and retains an encrypted matrix for repeated evaluation.
    ///
    /// A one-matrix [`Self::prepare_batch`]: the whole set is validated and
    /// its residency reserved before any upload begins.
    ///
    /// # Errors
    ///
    /// Returns a protocol error when the parameters and matrix disagree. A
    /// GPU engine also returns strict budget, capacity, and upload failures.
    pub fn prepare(
        &self,
        params: EmvpParams,
        matrix: EncryptedMatrix<MODULUS>,
    ) -> Result<PreparedMatrix<MODULUS>, PrepareMatrixError> {
        let mut prepared = self
            .prepare_batch_core(vec![(params, matrix)])
            .map_err(PrepareMatrixError::from)?;
        prepared.pop().ok_or(PrepareMatrixError::InternalState(
            "a one-matrix batch produced no handle",
        ))
    }

    /// Validates and retains a whole matrix set atomically.
    ///
    /// Every matrix is validated first, the device residency of the whole
    /// set is reserved with one atomic budget transition, and only then are
    /// the matrices uploaded. Concurrent callers therefore observe each set
    /// as all-or-nothing: a set either fits entirely alongside every set
    /// reserved before it, or it is rejected whole. On any failure no
    /// handle is returned and the entire reservation is released. On
    /// success each returned handle retains the reservation of its own
    /// matrix's bytes, released when that handle's final clone drops, so
    /// dropping one handle never holds another matrix's bytes.
    ///
    /// The returned handles keep the input order.
    ///
    /// # Errors
    ///
    /// Returns [`PrepareBatchError::Empty`] for an empty batch, the input
    /// index of the first malformed matrix as
    /// [`PrepareBatchError::Protocol`], the budget rejection of the whole
    /// set as [`PrepareBatchError::Budget`], or the input index of the
    /// first device upload failure as [`PrepareBatchError::Upload`]. A CPU
    /// engine can only produce the validation failures.
    pub fn prepare_batch<I>(
        &self,
        batch: I,
    ) -> Result<Vec<PreparedMatrix<MODULUS>>, PrepareBatchError>
    where
        I: IntoIterator<Item = (EmvpParams, EncryptedMatrix<MODULUS>)>,
    {
        self.prepare_batch_core(batch.into_iter().collect())
    }

    /// The shared validate, reserve, and upload pipeline behind
    /// [`Self::prepare_batch`] and [`Self::prepare`].
    fn prepare_batch_core(
        &self,
        matrices: Vec<(EmvpParams, EncryptedMatrix<MODULUS>)>,
    ) -> Result<Vec<PreparedMatrix<MODULUS>>, PrepareBatchError> {
        if matrices.is_empty() {
            return Err(PrepareBatchError::Empty);
        }
        // Phase 1: validate the whole set and total its residency before
        // touching the budget or the device.
        let mut total_bytes = 0_u64;
        #[cfg(feature = "gpu")]
        let mut residencies = Vec::with_capacity(matrices.len());
        for (index, (params, matrix)) in matrices.iter().enumerate() {
            let words = validate_matrix_shape(params, matrix)
                .map_err(|source| PrepareBatchError::Protocol { index, source })?;
            let bytes = residency_bytes(words).ok_or(PrepareBatchError::Protocol {
                index,
                source: ProtocolError::DimensionOverflow,
            })?;
            total_bytes = total_bytes
                .checked_add(bytes)
                .ok_or(PrepareBatchError::Protocol {
                    index,
                    source: ProtocolError::DimensionOverflow,
                })?;
            #[cfg(feature = "gpu")]
            residencies.push(bytes);
        }
        // Phase 2: reserve the whole set with one atomic transition before
        // any upload; failure rejects the set whole.
        #[cfg(feature = "gpu")]
        let gpu = self.gpu.as_ref();
        #[cfg(feature = "gpu")]
        if let Some(gpu) = gpu {
            reserve_budget(gpu, total_bytes).map_err(PrepareBatchError::Budget)?;
        }
        #[cfg(feature = "gpu")]
        let mut reservations = gpu.map_or_else(
            || Box::new(std::iter::empty()) as Box<dyn Iterator<Item = Arc<GpuReservation>>>,
            |gpu| {
                // Split the reserved total into one reservation per matrix
                // so every handle releases exactly its own bytes on drop.
                // Reservations not yet attached release the un-uploaded
                // remainder when this pipeline aborts.
                Box::new(residencies.iter().copied().map(move |bytes| {
                    Arc::new(GpuReservation {
                        engine: Arc::clone(gpu),
                        bytes,
                    })
                })) as Box<dyn Iterator<Item = Arc<GpuReservation>>>
            },
        );
        // Phase 3: upload. Any failure drops the handles prepared so far
        // and the reservations not yet attached, which together release
        // exactly the reserved total.
        let mut prepared = Vec::with_capacity(matrices.len());
        #[cfg_attr(
            not(feature = "gpu"),
            expect(
                unused_variables,
                reason = "the input index only names GPU upload failures"
            )
        )]
        for (index, (params, matrix)) in matrices.into_iter().enumerate() {
            #[cfg(feature = "gpu")]
            let (gpu_matrix, reservation) = match (gpu, reservations.next()) {
                (Some(gpu), Some(reservation)) => {
                    let uploaded = gpu
                        .answerer
                        .upload_matrix(&params, &matrix)
                        .map_err(|source| PrepareBatchError::Upload { index, source })?;
                    (Some(uploaded), Some(reservation))
                }
                (None, None) => (None, None),
                _ => {
                    return Err(PrepareBatchError::InternalState(
                        "the reservation count diverged from the matrix count",
                    ));
                }
            };
            prepared.push(PreparedMatrix {
                inner: Arc::new(PreparedMatrixInner {
                    id: NEXT_MATRIX_ID.fetch_add(1, Ordering::Relaxed),
                    engine_id: self.id,
                    params,
                    matrix,
                    #[cfg(feature = "gpu")]
                    gpu_matrix,
                    #[cfg(feature = "gpu")]
                    _reservation: reservation,
                }),
            });
        }
        Ok(prepared)
    }

    /// Whether this engine holds a live compute device, so prepared
    /// matrices stay resident under the configured device budget. `false`
    /// means every matrix is answered on the CPU and no device budget is
    /// active.
    #[must_use]
    #[cfg(feature = "gpu")]
    pub const fn has_device(&self) -> bool {
        self.gpu.is_some()
    }

    /// Whether this engine holds a live compute device, so prepared
    /// matrices stay resident under the configured device budget. `false`
    /// means every matrix is answered on the CPU and no device budget is
    /// active.
    #[must_use]
    #[cfg(not(feature = "gpu"))]
    pub const fn has_device(&self) -> bool {
        false
    }

    /// Answers all jobs and returns answer groups in input order.
    ///
    /// # Errors
    ///
    /// Returns an indexed validation or protocol failure without returning
    /// partial results.
    pub fn answer_many(
        &self,
        jobs: &[AnswerJob<'_, MODULUS>],
    ) -> Result<Vec<Vec<AnswerMatrix<MODULUS>>>, AnswerEngineError> {
        self.answer_many_with_report(jobs)
            .map(|(answers, _)| answers)
    }

    /// Answers all jobs and returns answers plus execution statistics.
    ///
    /// # Errors
    ///
    /// Returns the same failures as [`Self::answer_many`].
    pub fn answer_many_with_report(
        &self,
        jobs: &[AnswerJob<'_, MODULUS>],
    ) -> Result<(Vec<Vec<AnswerMatrix<MODULUS>>>, AnswerReport), AnswerEngineError> {
        let total_start = Instant::now();
        let planning_start = Instant::now();
        let threads = rayon::current_num_threads();
        let entries = validate_jobs(self.id, jobs, threads)?;
        #[cfg(feature = "gpu")]
        let mut entries = entries;

        #[cfg(feature = "gpu")]
        let gpu_jobs = collect_gpu_jobs(jobs, &entries);
        #[cfg(feature = "gpu")]
        let packed_plan = if gpu_jobs.is_empty() {
            None
        } else {
            let gpu = self.gpu_answerer()?;
            Some(
                gpu.answerer
                    .plan_packed(&gpu_jobs)
                    .map_err(AnswerEngineError::Gpu)?,
            )
        };
        let planning = planning_start.elapsed();

        #[cfg(feature = "gpu")]
        let flight = if let Some(plan) = packed_plan {
            let flight = self
                .gpu_answerer()?
                .answerer
                .begin_packed(&gpu_jobs, plan)
                .map_err(AnswerEngineError::Gpu)?;
            for (entry, count) in flight.segment_counts() {
                entries[entry].gpu_segments = count;
            }
            Some(flight)
        } else {
            None
        };

        let cpu_start = Instant::now();
        let computed = compute_cpu_jobs(jobs, &entries);
        let cpu_compute = cpu_start.elapsed();
        let mut answers: Vec<Option<Vec<AnswerMatrix<MODULUS>>>> =
            (0..jobs.len()).map(|_| None).collect();
        for result in computed {
            let (entry, entry_answers) = result?;
            answers[entry] = Some(entry_answers);
        }
        #[cfg(feature = "gpu")]
        let (gpu_stats, gpu_timings) = if let Some(flight) = flight {
            let (gpu_answers, stats, timings) = self
                .gpu_answerer()?
                .answerer
                .finish_packed(flight)
                .map_err(AnswerEngineError::Gpu)?;
            for (entry, entry_answers) in gpu_answers {
                answers[entry] = Some(entry_answers);
            }
            (stats, timings)
        } else {
            (PackedStats::default(), PackedTimings::default())
        };
        let answers = collect_answers(answers)?;
        #[cfg(feature = "gpu")]
        let report = build_report(
            entries,
            planning,
            cpu_compute,
            gpu_stats,
            gpu_timings,
            total_start.elapsed(),
        );
        #[cfg(not(feature = "gpu"))]
        let report = build_report(entries, planning, cpu_compute, total_start.elapsed());
        Ok((answers, report))
    }

    #[cfg(feature = "gpu")]
    fn gpu_answerer(&self) -> Result<&GpuEngine, AnswerEngineError> {
        self.gpu.as_deref().ok_or(AnswerEngineError::InternalState(
            "GPU work was selected without a GPU engine",
        ))
    }

    /// Validates all jobs and plans their execution without evaluating
    /// anything.
    ///
    /// This is the plan step of the plan → reserve → execute path. It runs
    /// the same all-or-nothing validation as [`Self::answer_many`] — the
    /// engine identifier of every matrix, no repeated matrix, nonempty job
    /// list, and per-query width and instance identifier — and fixes, per
    /// entry, the validated answer shape, the backend under a snapshot of
    /// the current rayon pool, the entry's arena offset (the prefix sum of
    /// the preceding entries' answer words), and for GPU-tier entries the
    /// single packed device flight over the GPU subset in entry order. The
    /// returned plan borrows the exact inputs, so executing it against
    /// different inputs is unrepresentable, and it can be executed
    /// repeatedly against one workspace.
    ///
    /// On a CPU engine every entry plans on a CPU tier even above the GPU
    /// threshold, because no matrix of a CPU engine is device-resident.
    ///
    /// # Errors
    ///
    /// Returns the same indexed validation failures as
    /// [`Self::answer_many`], plus device planning failures for the packed
    /// GPU flight of a GPU engine.
    pub fn plan<'a, Q: QueryValues<MODULUS> + Sync>(
        &self,
        jobs: &'a [AnswerJob<'a, MODULUS, Q>],
    ) -> Result<EnginePlan<'a, MODULUS, Q>, AnswerEngineError> {
        let threads = rayon::current_num_threads();
        let (entries, total_words) = plan_jobs(self.id, jobs, threads)?;
        #[cfg(feature = "gpu")]
        let mut entries = entries;
        #[cfg(feature = "gpu")]
        let gpu = {
            // The GPU subset in entry order. A GPU-tier entry is always
            // device-resident by construction of the backend selection;
            // anything else is a plan invariant violation rejected
            // fail-closed.
            let mut gpu_jobs = Vec::new();
            for (entry, job) in jobs.iter().enumerate() {
                if entries[entry].backend != AnswerBackend::Gpu {
                    continue;
                }
                let Some(matrix) = job.matrix.inner.gpu_matrix.as_ref() else {
                    return Err(AnswerEngineError::InternalState(
                        "the GPU tier was selected without a device-resident matrix",
                    ));
                };
                gpu_jobs.push(PackedJob {
                    entry,
                    matrix,
                    queries: job.queries,
                });
            }
            if gpu_jobs.is_empty() {
                None
            } else {
                let packed = self
                    .gpu_answerer()?
                    .answerer
                    .plan_packed(&gpu_jobs)
                    .map_err(AnswerEngineError::Gpu)?;
                // Per-entry segment counts are a plan-phase fact: the
                // planner already knows how each job's queries split into
                // dispatch segments.
                for (job, count) in packed.segment_counts() {
                    entries[gpu_jobs[job].entry].gpu_segments = count;
                }
                let bases = gpu_jobs
                    .iter()
                    .map(|job| entries[job.entry].offset_words)
                    .collect();
                Some(PackedGpuPlan {
                    packed,
                    jobs: gpu_jobs,
                    bases,
                })
            }
        };
        Ok(EnginePlan {
            entries,
            total_words,
            #[cfg(feature = "gpu")]
            gpu,
        })
    }

    /// Executes a validated plan into a caller-owned workspace, without
    /// allocating.
    ///
    /// The execute step of the plan → reserve → execute path. It checks
    /// the workspace's capacity first (rejecting before any write), fills
    /// every CPU entry's disjoint arena range one entry at a time through
    /// the same CPU row kernels the one-shot path runs — the plan's fixed
    /// tier decides serial versus the rayon grid, whose per-row staging
    /// leases from the workspace's scratch pool — and evaluates all
    /// GPU-tier entries as one packed device flight streaming into the
    /// entries' absolute arena offsets. Rayon parallelism lives inside each
    /// CPU entry exactly as in [`Self::answer_many`]; the entries
    /// themselves are filled sequentially in plan order.
    ///
    /// The returned [`EngineAnswers`] borrows the filled arena immutably;
    /// on any error no views are returned and the arena's contents are
    /// unspecified — callers must treat the whole workspace as unwritten.
    ///
    /// # Errors
    ///
    /// Returns [`AnswerEngineError::Capacity`] before touching the arena
    /// when the workspace's usable words are below the plan's requirement;
    /// reserve the workspace for the plan first. A GPU engine additionally
    /// returns the packed flight's device failures.
    pub fn execute<'ws, 'p, Q: QueryValues<MODULUS> + Sync>(
        &self,
        plan: &'p EnginePlan<'p, MODULUS, Q>,
        workspace: &'ws mut EngineWorkspace<MODULUS>,
    ) -> Result<(EngineAnswers<'ws, 'p, MODULUS, Q>, EngineReport), AnswerEngineError> {
        let total_start = Instant::now();
        let planning_start = Instant::now();
        // Capacity first: reject before any write.
        let required = plan.total_words;
        let available = workspace.arena.len();
        if available < required {
            return Err(AnswerEngineError::Capacity {
                required,
                available,
            });
        }
        let planning = planning_start.elapsed();

        let cpu_start = Instant::now();
        fill_cpu_entries(plan, workspace)?;
        let cpu_compute = cpu_start.elapsed();

        #[cfg(feature = "gpu")]
        let (gpu_stats, gpu_timings) = match &plan.gpu {
            Some(gpu) => {
                let answerer = self.gpu_answerer()?;
                let (stats, timings) = answerer
                    .answerer
                    .execute_packed_into_at(
                        &gpu.packed,
                        &gpu.jobs,
                        &gpu.bases,
                        &mut workspace.arena,
                    )
                    .map_err(AnswerEngineError::Gpu)?;
                (stats, timings)
            }
            None => (PackedStats::default(), PackedTimings::default()),
        };
        #[cfg(not(feature = "gpu"))]
        let cpu_entries = plan.entries.len();
        #[cfg(feature = "gpu")]
        let (cpu_entries, gpu_entries, gpu_queries) = {
            let gpu_entries = plan
                .entries
                .iter()
                .filter(|entry| entry.backend == AnswerBackend::Gpu)
                .count();
            let gpu_queries = plan
                .entries
                .iter()
                .filter(|entry| entry.backend == AnswerBackend::Gpu)
                .map(|entry| entry.shape.queries())
                .sum();
            (plan.entries.len() - gpu_entries, gpu_entries, gpu_queries)
        };
        let multiplications = plan.entries.iter().fold(0_usize, |total, entry| {
            total.saturating_add(entry.multiplications)
        });
        #[cfg(feature = "gpu")]
        let report = EngineReport {
            planning,
            cpu_compute,
            gpu_prepare_buffers: gpu_timings.prepare_buffers,
            gpu_upload: gpu_timings.encode_upload_queries,
            gpu_submit: gpu_timings.dispatch_submit,
            gpu_wait: gpu_timings.wait_readback,
            gpu_reconstruct: gpu_timings.reconstruct,
            total: total_start.elapsed(),
            gpu_entries,
            cpu_entries,
            gpu_queries,
            multiplications,
            gpu_chunks: gpu_stats.chunks,
            gpu_segments: gpu_stats.segments,
            gpu_query_bytes: gpu_stats.query_bytes,
            gpu_answer_bytes: gpu_stats.answer_bytes,
        };
        #[cfg(not(feature = "gpu"))]
        let report = EngineReport {
            planning,
            cpu_compute,
            gpu_prepare_buffers: Duration::ZERO,
            gpu_upload: Duration::ZERO,
            gpu_submit: Duration::ZERO,
            gpu_wait: Duration::ZERO,
            gpu_reconstruct: Duration::ZERO,
            total: total_start.elapsed(),
            gpu_entries: 0,
            cpu_entries,
            gpu_queries: 0,
            multiplications,
            gpu_chunks: 0,
            gpu_segments: 0,
            gpu_query_bytes: 0,
            gpu_answer_bytes: 0,
        };
        Ok((
            EngineAnswers {
                arena: &workspace.arena[..required],
                plan,
            },
            report,
        ))
    }
}

/// Validates one matrix against its parameters and returns its field-word
/// count `rows * n`.
///
/// This is the shape contract of both the CPU answer path and the device
/// upload path, applied before any budget or device work.
fn validate_matrix_shape<const MODULUS: u32>(
    params: &EmvpParams,
    matrix: &EncryptedMatrix<MODULUS>,
) -> Result<usize, ProtocolError> {
    params.validate_dimensions()?;
    let n = params.n()?;
    if matrix.columns() != n {
        return Err(ProtocolError::LengthMismatch {
            name: "encrypted matrix columns",
            expected: n,
            actual: matrix.columns(),
        });
    }
    if matrix.rows() == 0 {
        return Err(ProtocolError::LengthMismatch {
            name: "matrix rows",
            expected: 1,
            actual: 0,
        });
    }
    let expected = matrix
        .rows()
        .checked_mul(n)
        .ok_or(ProtocolError::DimensionOverflow)?;
    if matrix.values().len() != expected {
        return Err(ProtocolError::LengthMismatch {
            name: "encrypted matrix values",
            expected,
            actual: matrix.values().len(),
        });
    }
    Ok(expected)
}

/// The device residency of `words` field words, one `u32` each.
fn residency_bytes(words: usize) -> Option<u64> {
    u64::try_from(words)
        .ok()
        .and_then(|words| words.checked_mul(4))
}

/// Reserves `bytes` of long-lived matrix residency with one compare-and-
/// swap transition of the engine's budget counter, so concurrent callers
/// each reserve their whole request or nothing of it.
#[cfg(feature = "gpu")]
fn reserve_budget(gpu: &GpuEngine, bytes: u64) -> Result<(), GpuError> {
    let mut current = gpu.reserved_bytes.load(Ordering::Acquire);
    loop {
        let available = gpu.max_bytes.saturating_sub(current);
        let exhausted = || GpuError::BudgetExceeded {
            requested_bytes: bytes,
            available_bytes: available,
        };
        let Some(next) = current.checked_add(bytes) else {
            return Err(exhausted());
        };
        if next > gpu.max_bytes {
            return Err(exhausted());
        }
        match gpu.reserved_bytes.compare_exchange_weak(
            current,
            next,
            Ordering::AcqRel,
            Ordering::Acquire,
        ) {
            Ok(_) => return Ok(()),
            Err(observed) => current = observed,
        }
    }
}

/// The validated, tier-selected execution facts of one answer entry.
///
/// Everything [`Self::answer_many`]'s per-entry report and the plan phase
/// need, computed once and shared by both paths so their validation and
/// backend selection cannot drift apart.
struct ValidatedEntry {
    rows: usize,
    blocks: usize,
    instance_id: u128,
    queries: usize,
    backend: AnswerBackend,
    multiplications: usize,
    query_words: usize,
    answer_words: usize,
}

fn validate_jobs<'a, const MODULUS: u32, Q: QueryValues<MODULUS>>(
    engine_id: u64,
    jobs: &'a [AnswerJob<'a, MODULUS, Q>],
    threads: usize,
) -> Result<Vec<AnswerEntryReport>, AnswerEngineError> {
    if jobs.is_empty() {
        return Err(AnswerEngineError::EmptyJobs);
    }
    let mut seen = HashSet::with_capacity(jobs.len());
    let mut entries = Vec::with_capacity(jobs.len());
    for (entry, job) in jobs.iter().enumerate() {
        if job.matrix.inner.engine_id != engine_id {
            return Err(AnswerEngineError::ForeignMatrix { entry });
        }
        if !seen.insert(job.matrix.inner.id) {
            return Err(AnswerEngineError::RepeatedMatrix { entry });
        }
        if job.queries.is_empty() {
            return Err(AnswerEngineError::EmptyQueries { entry });
        }
        let validated = validate_entry(entry, job, threads)?;
        entries.push(AnswerEntryReport {
            backend: validated.backend,
            queries: validated.queries,
            multiplications: validated.multiplications,
            query_words: validated.query_words,
            answer_words: validated.answer_words,
            gpu_segments: 0,
        });
    }
    Ok(entries)
}

/// The plan phase's per-entry validation: the same checks and selections
/// as [`validate_jobs`], kept as [`EntryPlan`]s with running arena
/// offsets, plus the total arena word count.
fn plan_jobs<'a, const MODULUS: u32, Q: QueryValues<MODULUS>>(
    engine_id: u64,
    jobs: &'a [AnswerJob<'a, MODULUS, Q>],
    threads: usize,
) -> Result<(Vec<EntryPlan<'a, MODULUS, Q>>, usize), AnswerEngineError> {
    if jobs.is_empty() {
        return Err(AnswerEngineError::EmptyJobs);
    }
    let mut seen = HashSet::with_capacity(jobs.len());
    let mut entries = Vec::with_capacity(jobs.len());
    let mut offset_words = 0_usize;
    for (entry, job) in jobs.iter().enumerate() {
        if job.matrix.inner.engine_id != engine_id {
            return Err(AnswerEngineError::ForeignMatrix { entry });
        }
        if !seen.insert(job.matrix.inner.id) {
            return Err(AnswerEngineError::RepeatedMatrix { entry });
        }
        if job.queries.is_empty() {
            return Err(AnswerEngineError::EmptyQueries { entry });
        }
        let validated = validate_entry(entry, job, threads)?;
        entries.push(EntryPlan {
            matrix: job.matrix,
            queries: job.queries,
            shape: AnswerShape::new(
                validated.rows,
                validated.blocks,
                validated.queries,
                validated.instance_id,
            ),
            backend: validated.backend,
            offset_words,
            multiplications: validated.multiplications,
            gpu_segments: 0,
        });
        // `answer_words` is the entry's whole query-major arena:
        // `queries * rows * blocks`.
        offset_words = offset_words.checked_add(validated.answer_words).ok_or(
            AnswerEngineError::Protocol {
                entry,
                source: ProtocolError::DimensionOverflow,
            },
        )?;
    }
    Ok((entries, offset_words))
}

fn validate_entry<const MODULUS: u32, Q: QueryValues<MODULUS>>(
    entry: usize,
    job: &AnswerJob<'_, MODULUS, Q>,
    threads: usize,
) -> Result<ValidatedEntry, AnswerEngineError> {
    let inner = &job.matrix.inner;
    let n = inner.matrix.columns();
    for query in job.queries {
        validate_query(entry, inner, query, n)?;
    }
    let blocks = inner
        .params
        .blocks()
        .map_err(|source| AnswerEngineError::Protocol {
            entry,
            source: source.into(),
        })?;
    let rows = inner.matrix.rows();
    let answer_words = rows
        .checked_mul(blocks)
        .and_then(|words| words.checked_mul(job.queries.len()))
        .ok_or(AnswerEngineError::Protocol {
            entry,
            source: ProtocolError::DimensionOverflow,
        })?;
    let multiplications = job.queries.len().saturating_mul(rows).saturating_mul(n);
    let cpu_backend = select_cpu_backend(job.queries.len(), rows, n, threads);
    #[cfg(feature = "gpu")]
    let backend = inner.gpu_matrix.as_ref().map_or(cpu_backend, |_| {
        select_answer_backend(job.queries.len(), rows, n, threads)
    });
    #[cfg(not(feature = "gpu"))]
    let backend = cpu_backend;
    Ok(ValidatedEntry {
        rows,
        blocks,
        instance_id: inner.matrix.instance_id(),
        queries: job.queries.len(),
        backend,
        multiplications,
        query_words: job.queries.len().saturating_mul(n),
        answer_words,
    })
}

fn validate_query<const MODULUS: u32, Q: QueryValues<MODULUS>>(
    entry: usize,
    inner: &PreparedMatrixInner<MODULUS>,
    query: &Q,
    n: usize,
) -> Result<(), AnswerEngineError> {
    validate_query_against_matrix(n, inner.matrix.instance_id(), query)
        .map_err(|source| AnswerEngineError::Protocol { entry, source })
}

#[cfg(feature = "gpu")]
fn collect_gpu_jobs<'a, const MODULUS: u32, Q: QueryValues<MODULUS>>(
    jobs: &'a [AnswerJob<'a, MODULUS, Q>],
    entries: &[AnswerEntryReport],
) -> Vec<PackedJob<'a, MODULUS, Q>> {
    jobs.iter()
        .enumerate()
        .filter_map(|(entry, job)| {
            if entries[entry].backend != AnswerBackend::Gpu {
                return None;
            }
            job.matrix
                .inner
                .gpu_matrix
                .as_ref()
                .map(|matrix| PackedJob {
                    entry,
                    matrix,
                    queries: job.queries,
                })
        })
        .collect()
}

#[cfg(feature = "gpu")]
fn compute_cpu_jobs<const MODULUS: u32>(
    jobs: &[AnswerJob<'_, MODULUS>],
    entries: &[AnswerEntryReport],
) -> Vec<Result<(usize, Vec<AnswerMatrix<MODULUS>>), AnswerEngineError>> {
    jobs.par_iter()
        .enumerate()
        .filter(|(entry, _)| entries[*entry].backend != AnswerBackend::Gpu)
        .map(|(entry, job)| {
            let inner = &job.matrix.inner;
            answer_batch(&inner.params, &inner.matrix, job.queries)
                .map(|answers| (entry, answers))
                .map_err(|source| AnswerEngineError::Protocol { entry, source })
        })
        .collect()
}

#[cfg(not(feature = "gpu"))]
fn compute_cpu_jobs<const MODULUS: u32>(
    jobs: &[AnswerJob<'_, MODULUS>],
    _entries: &[AnswerEntryReport],
) -> Vec<Result<(usize, Vec<AnswerMatrix<MODULUS>>), AnswerEngineError>> {
    jobs.par_iter()
        .enumerate()
        .map(|(entry, job)| {
            let inner = &job.matrix.inner;
            answer_batch(&inner.params, &inner.matrix, job.queries)
                .map(|answers| (entry, answers))
                .map_err(|source| AnswerEngineError::Protocol { entry, source })
        })
        .collect()
}

fn collect_answers<const MODULUS: u32>(
    answers: Vec<Option<Vec<AnswerMatrix<MODULUS>>>>,
) -> Result<Vec<Vec<AnswerMatrix<MODULUS>>>, AnswerEngineError> {
    answers
        .into_iter()
        .map(|answer| {
            answer.ok_or(AnswerEngineError::InternalState(
                "a planned job produced no answer",
            ))
        })
        .collect()
}

#[cfg(feature = "gpu")]
fn build_report(
    entries: Vec<AnswerEntryReport>,
    planning: Duration,
    cpu_compute: Duration,
    gpu_stats: PackedStats,
    gpu_timings: PackedTimings,
    total: Duration,
) -> AnswerReport {
    let cpu_entries = entries
        .iter()
        .filter(|entry| entry.backend != AnswerBackend::Gpu)
        .count();
    let gpu_entries = entries.len() - cpu_entries;
    let gpu_queries = entries
        .iter()
        .filter(|entry| entry.backend == AnswerBackend::Gpu)
        .map(|entry| entry.queries)
        .sum();
    AnswerReport {
        multiplications: total_multiplications(&entries),
        entries,
        cpu_entries,
        gpu_entries,
        gpu_queries,
        gpu_query_bytes: gpu_stats.query_bytes,
        gpu_answer_bytes: gpu_stats.answer_bytes,
        gpu_chunks: gpu_stats.chunks,
        gpu_segments: gpu_stats.segments,
        planning,
        cpu_compute,
        gpu_prepare_buffers: gpu_timings.prepare_buffers,
        gpu_upload: gpu_timings.encode_upload_queries,
        gpu_submit: gpu_timings.dispatch_submit,
        gpu_wait: gpu_timings.wait_readback,
        gpu_reconstruct: gpu_timings.reconstruct,
        total,
    }
}

#[cfg(not(feature = "gpu"))]
fn build_report(
    entries: Vec<AnswerEntryReport>,
    planning: Duration,
    cpu_compute: Duration,
    total: Duration,
) -> AnswerReport {
    AnswerReport {
        multiplications: total_multiplications(&entries),
        cpu_entries: entries.len(),
        entries,
        planning,
        cpu_compute,
        total,
        ..AnswerReport::default()
    }
}

fn total_multiplications(entries: &[AnswerEntryReport]) -> usize {
    entries.iter().fold(0, |total, entry| {
        total.saturating_add(entry.multiplications)
    })
}

/// Fills every CPU-tier entry's disjoint arena range of a plan, in plan
/// order.
///
/// The plan's fixed per-entry tier decides serial versus the rayon grid;
/// rayon parallelism lives inside each entry exactly as in
/// [`AnswerEngine::answer_many`]. GPU-tier entries are skipped — execute
/// handles them as one packed flight.
///
/// # Errors
///
/// Returns an internal-state error if a plan invariant was somehow
/// violated; no entry is filled after that point.
fn fill_cpu_entries<const MODULUS: u32, Q: QueryValues<MODULUS> + Sync>(
    plan: &EnginePlan<'_, MODULUS, Q>,
    workspace: &mut EngineWorkspace<MODULUS>,
) -> Result<(), AnswerEngineError> {
    for entry in &plan.entries {
        let words = entry
            .shape
            .arena_words()
            .ok_or(AnswerEngineError::InternalState(
                "a planned entry's arena words overflowed",
            ))?;
        let end = entry
            .offset_words
            .checked_add(words)
            .ok_or(AnswerEngineError::InternalState(
                "a planned entry's arena range overflowed",
            ))?;
        let arena = &mut workspace.arena[entry.offset_words..end];
        let inner = &entry.matrix.inner;
        let n = inner.matrix.columns();
        let b = inner.params.block_size();
        let s = entry.shape.blocks();
        let rows = entry.shape.rows();
        match entry.backend {
            AnswerBackend::SingleCore => {
                fill_answer_batch_serial(&inner.matrix, entry.queries, arena, n, b, s, rows);
            }
            AnswerBackend::Rayon => {
                fill_entry_parallel(
                    &inner.matrix,
                    entry.queries,
                    arena,
                    n,
                    b,
                    s,
                    rows,
                    &workspace.row_scratch,
                );
            }
            // GPU-tier entries are handled by the single packed flight in
            // execute.
            #[cfg(feature = "gpu")]
            AnswerBackend::Gpu => {}
        }
    }
    Ok(())
}

/// The parallel tier of the engine's execute path.
///
/// The same flattened (query, row) grid the one-shot batch and the
/// workspace answer path run, with each answer row staged through a
/// scratch leased from `pool` and returned afterwards; the arena chunks,
/// the row order, and the row kernel ([`fill_answer_row`]) are identical,
/// so the tier's output matches [`Self::answer_many`] exactly.
#[expect(
    clippy::too_many_arguments,
    reason = "the kernel mirrors fill_answer_batch_serial's argument layout and adds the scratch pool"
)]
fn fill_entry_parallel<const MODULUS: u32, Q: QueryValues<MODULUS> + Sync>(
    matrix: &EncryptedMatrix<MODULUS>,
    queries: &[Q],
    arena: &mut [FieldElement<MODULUS>],
    n: usize,
    b: usize,
    s: usize,
    rows: usize,
    pool: &RowScratchPool<MODULUS>,
) {
    arena
        .par_chunks_mut(s)
        .enumerate()
        .for_each(|(flat_row, answer_row)| {
            let query = &queries[flat_row / rows];
            let matrix_row_index = flat_row % rows;
            let matrix_row = &matrix.values()[matrix_row_index * n..(matrix_row_index + 1) * n];
            let mut scratch = pool.pop(answer_row.len());
            fill_answer_row(matrix_row, query.values(), b, &mut scratch.row);
            answer_row.copy_from_slice(&scratch.row);
            pool.push(scratch);
        });
}

#[cfg(test)]
mod tests {
    use prime_field_layer::PrimeField;

    use super::*;
    use crate::view::EncryptedQueryRef;

    const MODULUS: u32 = 1_073_479_681;
    const PARAMS: EmvpParams = EmvpParams {
        k: 8,
        ell: 8,
        b: 2,
        lambda: 7,
    };
    #[cfg(feature = "gpu")]
    const GPU_PARAMS: EmvpParams = EmvpParams {
        k: 1,
        ell: 1,
        b: 2,
        lambda: 7,
    };

    fn matrix(instance_id: u128, rows: usize) -> EncryptedMatrix<MODULUS> {
        let zero = PrimeField::<MODULUS>::new().element_u32(0);
        EncryptedMatrix::from_parts(instance_id, rows, 16, vec![zero; rows * 16]).unwrap()
    }

    /// A well-formed matrix whose declared width disagrees with [`PARAMS`],
    /// whose `n` is 16.
    fn narrow_matrix(instance_id: u128) -> EncryptedMatrix<MODULUS> {
        let zero = PrimeField::<MODULUS>::new().element_u32(0);
        EncryptedMatrix::from_parts(instance_id, 2, 15, vec![zero; 2 * 15]).unwrap()
    }

    fn query(instance_id: u128, query_id: u64) -> EncryptedQuery<MODULUS> {
        let one = PrimeField::<MODULUS>::new().element_u32(1);
        EncryptedQuery::from_parts(instance_id, query_id, vec![one; 16])
    }

    /// Asserts an executed [`Answers`] view pairs exactly the reference
    /// answers: values, identifiers, and shape, answer by answer.
    fn assert_view_matches<Q: QueryValues<MODULUS>>(
        view: &Answers<'_, MODULUS>,
        queries: &[Q],
        expected: &[AnswerMatrix<MODULUS>],
    ) {
        assert_eq!(view.shape().queries(), expected.len());
        for (index, expected_answer) in expected.iter().enumerate() {
            let answer = view.answer(queries, index).unwrap();
            assert_eq!(answer.values(), expected_answer.values());
            assert_eq!(answer.query_id(), expected_answer.query_id());
            assert_eq!(answer.instance_id(), expected_answer.instance_id());
            assert_eq!(answer.rows(), expected_answer.rows());
            assert_eq!(answer.blocks(), expected_answer.blocks());
        }
    }

    #[test]
    fn cpu_engine_preserves_entry_and_query_order() {
        let engine = AnswerEngine::cpu();
        let first = engine.prepare(PARAMS, matrix(11, 2)).unwrap();
        let second = engine.prepare(PARAMS, matrix(22, 3)).unwrap();
        let first_queries = [query(11, 7), query(11, 8)];
        let second_queries = [query(22, 9)];
        let jobs = [
            AnswerJob {
                matrix: &first,
                queries: &first_queries,
            },
            AnswerJob {
                matrix: &second,
                queries: &second_queries,
            },
        ];
        let (answers, report) = engine.answer_many_with_report(&jobs).unwrap();
        assert_eq!(answers.len(), 2);
        assert_eq!(
            answers[0]
                .iter()
                .map(AnswerMatrix::query_id)
                .collect::<Vec<_>>(),
            [7, 8]
        );
        assert_eq!(answers[1][0].query_id(), 9);
        assert_eq!(answers[0][0].rows(), 2);
        assert_eq!(answers[1][0].rows(), 3);
        assert_eq!(report.cpu_entries, 2);
        assert_eq!(report.entries.len(), 2);
    }

    #[test]
    fn validation_rejects_empty_repeated_and_foreign_jobs() {
        let engine = AnswerEngine::cpu();
        let foreign_engine = AnswerEngine::cpu();
        let prepared = engine.prepare(PARAMS, matrix(11, 2)).unwrap();
        let foreign = foreign_engine.prepare(PARAMS, matrix(22, 2)).unwrap();
        let queries = [query(11, 1)];
        let foreign_queries = [query(22, 1)];

        assert!(matches!(
            engine.answer_many(&[]),
            Err(AnswerEngineError::EmptyJobs)
        ));
        assert!(matches!(
            engine.answer_many(&[AnswerJob {
                matrix: &prepared,
                queries: &[]
            }]),
            Err(AnswerEngineError::EmptyQueries { entry: 0 })
        ));
        assert!(matches!(
            engine.answer_many(&[
                AnswerJob {
                    matrix: &prepared,
                    queries: &queries
                },
                AnswerJob {
                    matrix: &prepared,
                    queries: &queries
                },
            ]),
            Err(AnswerEngineError::RepeatedMatrix { entry: 1 })
        ));
        assert!(matches!(
            engine.answer_many(&[AnswerJob {
                matrix: &foreign,
                queries: &foreign_queries
            }]),
            Err(AnswerEngineError::ForeignMatrix { entry: 0 })
        ));
    }

    #[test]
    fn batch_prepare_returns_handles_in_input_order() {
        let engine = AnswerEngine::cpu();
        let batch = engine
            .prepare_batch([
                (PARAMS, matrix(11, 2)),
                (PARAMS, matrix(22, 3)),
                (PARAMS, matrix(33, 5)),
            ])
            .unwrap();
        assert_eq!(batch.len(), 3);
        let query_sets: Vec<Vec<EncryptedQuery<MODULUS>>> = vec![
            vec![query(11, 1)],
            vec![query(22, 2), query(22, 3)],
            vec![query(33, 4)],
        ];
        let jobs: Vec<AnswerJob<'_, MODULUS>> = batch
            .iter()
            .zip(query_sets.iter())
            .map(|(matrix, queries)| AnswerJob {
                matrix,
                queries: queries.as_slice(),
            })
            .collect();
        let answers = engine.answer_many(&jobs).unwrap();
        let answer_rows: Vec<usize> = answers
            .iter()
            .map(|group| group.iter().map(AnswerMatrix::rows).sum())
            .collect();
        assert_eq!(answer_rows, [2, 6, 5]);
    }

    #[test]
    fn batch_prepare_rejects_the_whole_set_and_indexes_the_failure() {
        let engine = AnswerEngine::cpu();
        assert!(matches!(
            engine.prepare_batch(Vec::<(EmvpParams, EncryptedMatrix<MODULUS>)>::new()),
            Err(PrepareBatchError::Empty)
        ));
        // A leading malformed matrix is indexed at its input position.
        assert!(matches!(
            engine.prepare_batch([(PARAMS, narrow_matrix(12)), (PARAMS, matrix(11, 2))]),
            Err(PrepareBatchError::Protocol { index: 0, .. })
        ));
        // A later malformed matrix rejects the whole set, including the
        // matrices before it: no handle comes back.
        assert!(matches!(
            engine.prepare_batch([(PARAMS, matrix(11, 2)), (PARAMS, narrow_matrix(12))]),
            Err(PrepareBatchError::Protocol { index: 1, .. })
        ));
        // The rejected set left nothing behind: the engine still works.
        engine.prepare(PARAMS, matrix(13, 2)).unwrap();
    }

    #[test]
    fn single_prepare_shares_the_batch_validation() {
        let engine = AnswerEngine::cpu();
        assert!(matches!(
            engine.prepare(PARAMS, narrow_matrix(12)),
            Err(PrepareMatrixError::Protocol(
                ProtocolError::LengthMismatch { .. }
            ))
        ));
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_engine_matches_cpu_and_reports_packed_execution_when_available() {
        let engine = AnswerEngine::new(32 * 1024 * 1024).unwrap();
        if engine.gpu.is_none() {
            return;
        }
        let rows = crate::MIN_GPU_MULTIPLICATIONS / 2;
        let field = PrimeField::<MODULUS>::new();
        let values: Vec<_> = (0..rows * 2)
            .map(|index| field.element_u32((index % 17) as u32))
            .collect();
        let host = EncryptedMatrix::from_parts(91, rows, 2, values).unwrap();
        let expected_query =
            EncryptedQuery::from_parts(91, 7, vec![field.element_u32(3), field.element_u32(5)]);
        let expected =
            answer_batch(&GPU_PARAMS, &host, std::slice::from_ref(&expected_query)).unwrap();
        let prepared = engine.prepare(GPU_PARAMS, host).unwrap();
        let queries = [expected_query];
        let (actual, report) = engine
            .answer_many_with_report(&[AnswerJob {
                matrix: &prepared,
                queries: &queries,
            }])
            .unwrap();
        assert_eq!(actual[0], expected);
        assert_eq!(report.gpu_entries, 1);
        assert_eq!(report.gpu_queries, 1);
        assert_eq!(report.gpu_chunks, 1);
        assert_eq!(report.gpu_segments, 1);
        assert_eq!(report.entries[0].gpu_segments, 1);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn gpu_budget_is_held_until_the_final_prepared_clone_drops() {
        let bytes = (2 * PARAMS.n().unwrap() * 4) as u64;
        let engine = AnswerEngine::new(bytes).unwrap();
        if engine.gpu.is_none() {
            return;
        }
        let prepared = engine.prepare(PARAMS, matrix(31, 2)).unwrap();
        let final_clone = prepared.clone();
        drop(prepared);
        assert!(matches!(
            engine.prepare(PARAMS, matrix(32, 2)),
            Err(PrepareMatrixError::Gpu(GpuError::BudgetExceeded { .. }))
        ));
        drop(final_clone);
        engine.prepare(PARAMS, matrix(32, 2)).unwrap();
    }

    /// A GPU engine with the requested budget, or a skipped notice when the
    /// machine has no compute adapter.
    #[cfg(feature = "gpu")]
    fn gpu_engine_with_budget(bytes: u64) -> AnswerEngine<MODULUS> {
        let engine = AnswerEngine::new(bytes).unwrap();
        if engine.gpu.is_none() {
            eprintln!("gpu test skipped: no compute adapter");
        }
        engine
    }

    /// The engine's live matrix residency, readable because the tests are
    /// in this module.
    #[cfg(feature = "gpu")]
    fn reserved_bytes(engine: &AnswerEngine<MODULUS>) -> u64 {
        engine
            .gpu
            .as_ref()
            .map_or(0, |gpu| gpu.reserved_bytes.load(Ordering::Acquire))
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn batch_prepare_reserves_once_and_releases_per_handle() {
        let one = (2 * PARAMS.n().unwrap() * 4) as u64;
        let engine = gpu_engine_with_budget(3 * one);
        if engine.gpu.is_none() {
            return;
        }
        let batch = engine
            .prepare_batch([
                (PARAMS, matrix(41, 2)),
                (PARAMS, matrix(42, 2)),
                (PARAMS, matrix(43, 2)),
            ])
            .unwrap();
        // The whole set was reserved with one transition; nothing partial.
        assert_eq!(reserved_bytes(&engine), 3 * one);
        let another_set = [(PARAMS, matrix(44, 2)), (PARAMS, matrix(45, 2))];
        assert!(matches!(
            engine.prepare_batch(another_set),
            Err(PrepareBatchError::Budget(GpuError::BudgetExceeded {
                requested_bytes,
                available_bytes
            })) if requested_bytes == 2 * one && available_bytes == 0
        ));
        assert_eq!(reserved_bytes(&engine), 3 * one);
        // Dropping one handle releases exactly its own matrix's bytes; the
        // unrelated matrices stay resident.
        let keep = batch[0].clone();
        drop(batch);
        assert_eq!(reserved_bytes(&engine), one);
        // The released room fits exactly one more matrix.
        engine.prepare(PARAMS, matrix(44, 2)).unwrap();
        assert_eq!(reserved_bytes(&engine), 2 * one);
        // The final clone of the first handle releases the rest.
        drop(keep);
        assert_eq!(reserved_bytes(&engine), one);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn a_batch_validation_failure_leaves_no_reservation() {
        let one = (2 * PARAMS.n().unwrap() * 4) as u64;
        let engine = gpu_engine_with_budget(4 * one);
        if engine.gpu.is_none() {
            return;
        }
        assert!(matches!(
            engine.prepare_batch([
                (PARAMS, matrix(41, 2)),
                (PARAMS, narrow_matrix(42)),
                (PARAMS, matrix(43, 2)),
            ]),
            Err(PrepareBatchError::Protocol { index: 1, .. })
        ));
        // Validation precedes the reservation: the whole set was rejected
        // before any byte was booked, so a fresh set still fits.
        assert_eq!(reserved_bytes(&engine), 0);
        let batch = engine
            .prepare_batch([(PARAMS, matrix(41, 2)), (PARAMS, matrix(43, 2))])
            .unwrap();
        assert_eq!(reserved_bytes(&engine), 2 * one);
        drop(batch);
        assert_eq!(reserved_bytes(&engine), 0);
    }

    #[cfg(feature = "gpu")]
    #[test]
    fn concurrent_batches_reserve_the_whole_budget_atomically() {
        let one = (2 * PARAMS.n().unwrap() * 4) as u64;
        let engine = Arc::new(gpu_engine_with_budget(2 * one));
        if engine.gpu.is_none() {
            return;
        }
        // Two concurrent sets, each exactly the whole budget: exactly one
        // set wins its single whole-set reservation, the loser is rejected
        // whole, and no half-reserved set survives either outcome.
        let barrier = Arc::new(std::sync::Barrier::new(2));
        let workers: Vec<_> = (0_usize..2)
            .map(|worker| {
                let engine = Arc::clone(&engine);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    engine.prepare_batch([
                        (PARAMS, matrix(u128::try_from(100 + worker).unwrap(), 2)),
                        (PARAMS, matrix(u128::try_from(200 + worker).unwrap(), 2)),
                    ])
                })
            })
            .collect();
        let mut won = None;
        for result in workers.into_iter().map(|worker| worker.join().unwrap()) {
            match result {
                Ok(batch) => {
                    assert!(won.is_none(), "two batches won one budget");
                    won = Some(batch);
                }
                Err(PrepareBatchError::Budget(_)) => {}
                Err(error) => panic!("unexpected batch failure: {error}"),
            }
        }
        // The winner's whole-set residency is exactly the budget, and
        // releasing it releases everything.
        assert_eq!(reserved_bytes(&engine), 2 * one);
        drop(won);
        assert_eq!(reserved_bytes(&engine), 0);
    }

    /// The engine plan binds exact inputs by borrowing, so plan → reserve →
    /// execute must reproduce the allocating path's answers entry by entry,
    /// across entries whose shapes and CPU tiers differ.
    #[test]
    fn plan_execute_matches_answer_many_across_mixed_tiers() {
        let engine = AnswerEngine::cpu();
        let first = engine.prepare(PARAMS, matrix(11, 512)).unwrap();
        let second = engine.prepare(PARAMS, matrix(22, 2)).unwrap();
        let third = engine.prepare(PARAMS, matrix(33, 3)).unwrap();
        let first_queries: Vec<_> = (0..8_u64).map(|index| query(11, 40 + index)).collect();
        let second_queries = [query(22, 50)];
        let third_queries = [query(33, 60), query(33, 61)];
        let jobs = [
            AnswerJob {
                matrix: &first,
                queries: &first_queries,
            },
            AnswerJob {
                matrix: &second,
                queries: &second_queries,
            },
            AnswerJob {
                matrix: &third,
                queries: &third_queries,
            },
        ];
        let expected = engine.answer_many(&jobs).unwrap();
        let plan = engine.plan(&jobs).unwrap();
        assert_eq!(plan.len(), 3);
        assert!(!plan.is_empty());
        assert_eq!(plan.iter().count(), 3);
        // The tiers follow the shared dispatch policy under the current
        // pool; the big entry clears the parallel-work threshold.
        let threads = rayon::current_num_threads();
        assert!(
            8 * 512 * PARAMS.n().unwrap() >= crate::MIN_PARALLEL_MULTIPLICATIONS,
            "the fixture must clear the parallel threshold"
        );
        assert_eq!(
            plan.entry(0).unwrap().backend(),
            select_cpu_backend(8, 512, PARAMS.n().unwrap(), threads)
        );
        assert_eq!(
            plan.entry(1).unwrap().backend(),
            select_cpu_backend(1, 2, PARAMS.n().unwrap(), threads)
        );
        assert_eq!(plan.entry(0).unwrap().multiplications(), 8 * 512 * 16);
        assert_eq!(plan.entry(0).unwrap().shape().rows(), 512);
        assert_eq!(plan.entry(0).unwrap().shape().queries(), 8);
        assert_eq!(
            plan.total_words(),
            // blocks = n / b = 8; entry words = queries * rows * blocks:
            // 8 queries, 1 query, and 2 queries respectively.
            8 * 512 * 8 + 2 * 8 + 2 * 3 * 8,
        );
        let mut workspace = EngineWorkspace::new();
        workspace.reserve(&plan).unwrap();
        let (answers, report) = engine.execute(&plan, &mut workspace).unwrap();
        assert_eq!(report.cpu_entries, 3);
        assert_eq!(report.gpu_entries, 0);
        assert_eq!(report.gpu_queries, 0);
        assert_eq!(report.multiplications, 8 * 512 * 16 + 2 * 16 + 2 * 3 * 16);
        assert!(report.total >= report.cpu_compute);
        assert_eq!(answers.len(), 3);
        assert_eq!(answers.total_words(), plan.total_words());
        for index in 0..answers.len() {
            let view = answers.entry_answers(index).unwrap();
            assert_view_matches(&view, jobs[index].queries, &expected[index]);
        }
    }

    /// Entries must occupy disjoint consecutive arena ranges in plan order:
    /// the second entry's answers are contiguous right after the first's.
    #[test]
    fn entries_are_laid_out_back_to_back_in_plan_order() {
        let engine = AnswerEngine::cpu();
        let first_host = matrix(11, 2);
        let second_host = matrix(22, 3);
        let first = engine.prepare(PARAMS, first_host.clone()).unwrap();
        let second = engine.prepare(PARAMS, second_host.clone()).unwrap();
        let first_queries = [query(11, 7), query(11, 8)];
        let second_queries = [query(22, 9)];
        let jobs = [
            AnswerJob {
                matrix: &first,
                queries: &first_queries,
            },
            AnswerJob {
                matrix: &second,
                queries: &second_queries,
            },
        ];
        let first_words = 2 * 2 * 8;
        let second_words = 3 * 8; // 1 query * rows 3 * blocks 8
        let plan = engine.plan(&jobs).unwrap();
        assert_eq!(plan.entry(0).unwrap().offset_words(), 0);
        assert_eq!(plan.entry(1).unwrap().offset_words(), first_words);
        let mut workspace = EngineWorkspace::new();
        workspace.reserve(&plan).unwrap();
        let (arena0, arena1) = {
            let (answers, _) = engine.execute(&plan, &mut workspace).unwrap();
            let view0 = answers.entry_answers(0).unwrap();
            let view1 = answers.entry_answers(1).unwrap();
            assert_eq!(view0.len(), first_words);
            assert_eq!(view1.len(), second_words);
            assert_eq!(answers.total_words(), first_words + second_words);
            // Each entry matches the one-shot path's answers for the same
            // inputs.
            assert_view_matches(
                &view0,
                &first_queries,
                &answer_batch(&PARAMS, &first_host, &first_queries).unwrap(),
            );
            assert_view_matches(
                &view1,
                &second_queries,
                &answer_batch(&PARAMS, &second_host, &second_queries).unwrap(),
            );
            (view0.arena().to_vec(), view1.arena().to_vec())
        };
        // The views were exactly the consecutive workspace arena slices:
        // entry 1's answers are contiguous right after entry 0's.
        assert_eq!(&workspace.arena[..first_words], arena0.as_slice());
        assert_eq!(
            &workspace.arena[first_words..first_words + second_words],
            arena1.as_slice()
        );
    }

    /// An insufficient workspace capacity must be rejected before any
    /// mutation: the poisoned arena keeps every sentinel word.
    #[test]
    fn insufficient_capacity_is_rejected_before_any_mutation() {
        let engine = AnswerEngine::cpu();
        let small_host = matrix(11, 2);
        let big_host = matrix(22, 2);
        let prepared_small = engine.prepare(PARAMS, small_host).unwrap();
        let prepared_big = engine.prepare(PARAMS, big_host).unwrap();
        let small_queries = [query(11, 1)];
        let big_queries = [query(22, 1), query(22, 2), query(22, 3), query(22, 4)];
        let small_jobs = [AnswerJob {
            matrix: &prepared_small,
            queries: &small_queries,
        }];
        let big_jobs = [AnswerJob {
            matrix: &prepared_big,
            queries: &big_queries,
        }];
        let small_plan = engine.plan(&small_jobs).unwrap();
        let big_plan = engine.plan(&big_jobs).unwrap();
        assert!(big_plan.total_words() > small_plan.total_words());
        let mut workspace = EngineWorkspace::new();
        workspace.reserve(&small_plan).unwrap();
        // Poison the arena: the execute path must not touch a single word
        // when the capacity check fails.
        let sentinel = PrimeField::<MODULUS>::new().element_u32(MODULUS - 1);
        for slot in &mut workspace.arena {
            *slot = sentinel;
        }
        assert!(matches!(
            engine.execute(&big_plan, &mut workspace),
            Err(AnswerEngineError::Capacity { required, available })
                if required == big_plan.total_words()
                    && available == small_plan.total_words()
        ));
        assert!(workspace.arena.iter().all(|slot| *slot == sentinel));
        // After reserving, the same execute succeeds and the poisoned
        // prefix is fully overwritten.
        workspace.reserve(&big_plan).unwrap();
        {
            let (answers, _) = engine.execute(&big_plan, &mut workspace).unwrap();
            assert_eq!(answers.total_words(), big_plan.total_words());
        }
        assert!(!workspace.arena.iter().all(|slot| *slot == sentinel));
    }

    /// Reserve is the only growth point: the capacity survives a
    /// smaller-plan reserve untouched, execute never grows it, and release
    /// reports at least the peak.
    #[test]
    fn reserve_grows_never_shrinks_and_execute_never_grows() {
        let engine = AnswerEngine::cpu();
        let small_host = matrix(11, 2);
        let big_host = matrix(22, 512);
        let prepared_small = engine.prepare(PARAMS, small_host).unwrap();
        let prepared_big = engine.prepare(PARAMS, big_host).unwrap();
        let small_queries = [query(11, 1)];
        let big_queries: Vec<_> = (0..8_u64).map(|index| query(22, 20 + index)).collect();
        let small_jobs = [AnswerJob {
            matrix: &prepared_small,
            queries: &small_queries,
        }];
        let big_jobs = [AnswerJob {
            matrix: &prepared_big,
            queries: &big_queries,
        }];
        let small_plan = engine.plan(&small_jobs).unwrap();
        let big_plan = engine.plan(&big_jobs).unwrap();
        let peak = big_plan.total_words();
        assert!(peak > small_plan.total_words());
        let mut workspace = EngineWorkspace::new();
        assert_eq!(workspace.capacity_words(), 0);
        workspace.reserve(&big_plan).unwrap();
        assert_eq!(workspace.capacity_words(), peak);
        // A smaller plan's reserve is free and never shrinks the arena.
        workspace.reserve(&small_plan).unwrap();
        assert_eq!(workspace.capacity_words(), peak);
        // Execute never grows the workspace either.
        {
            let (answers, _) = engine.execute(&small_plan, &mut workspace).unwrap();
            assert_eq!(answers.total_words(), small_plan.total_words());
        }
        assert_eq!(workspace.capacity_words(), peak);
        // Re-executing the big plan still works on the same workspace.
        {
            let (answers, _) = engine.execute(&big_plan, &mut workspace).unwrap();
            assert_eq!(answers.total_words(), peak);
        }
        assert_eq!(workspace.capacity_words(), peak);
        assert!(workspace.release() >= peak);
    }

    /// Re-executing the same plan into the same workspace must be
    /// idempotent: identical answers, word for word.
    #[test]
    fn re_executing_the_same_plan_yields_identical_answers() {
        let engine = AnswerEngine::cpu();
        let first = engine.prepare(PARAMS, matrix(11, 2)).unwrap();
        let second = engine.prepare(PARAMS, matrix(22, 3)).unwrap();
        let first_queries = [query(11, 7), query(11, 8)];
        let second_queries = [query(22, 9)];
        let jobs = [
            AnswerJob {
                matrix: &first,
                queries: &first_queries,
            },
            AnswerJob {
                matrix: &second,
                queries: &second_queries,
            },
        ];
        let plan = engine.plan(&jobs).unwrap();
        let mut workspace = EngineWorkspace::new();
        workspace.reserve(&plan).unwrap();
        let _ = engine.execute(&plan, &mut workspace).unwrap();
        let first_run = workspace.arena.clone();
        let (total, view_values, view_ids) = {
            let (answers, _) = engine.execute(&plan, &mut workspace).unwrap();
            let view = answers.entry_answers(1).unwrap();
            // The re-executed views still pair with the planned queries.
            let reference = answer_batch(&PARAMS, &second.inner.matrix, &second_queries).unwrap();
            let values: Vec<_> = view.answer(&second_queries, 0).unwrap().values().to_vec();
            (answers.total_words(), values, reference[0].query_id())
        };
        assert_eq!(total, plan.total_words());
        assert_eq!(&workspace.arena[..total], &first_run[..]);
        assert_eq!(
            workspace.arena[plan.entry(1).unwrap().offset_words()
                ..plan.entry(1).unwrap().offset_words() + view_values.len()],
            view_values[..]
        );
        assert_eq!(view_ids, 9);
    }

    /// The plan and execute paths must accept the borrowed wire view as
    /// the query type and produce exactly the owned path's answers.
    #[test]
    fn plan_execute_accepts_borrowed_query_views() {
        let engine = AnswerEngine::cpu();
        let first_host = matrix(11, 2);
        let second_host = matrix(22, 3);
        let first = engine.prepare(PARAMS, first_host).unwrap();
        let second = engine.prepare(PARAMS, second_host).unwrap();
        let owned_first = [query(11, 7), query(11, 8)];
        let owned_second = [query(22, 9)];
        let borrowed_first: Vec<EncryptedQueryRef<'_, MODULUS>> =
            owned_first.iter().map(|query| query.as_ref()).collect();
        let borrowed_second: Vec<_> = owned_second.iter().map(|query| query.as_ref()).collect();
        // The old owned path pins the reference answers.
        let owned_jobs = [
            AnswerJob {
                matrix: &first,
                queries: &owned_first,
            },
            AnswerJob {
                matrix: &second,
                queries: &owned_second,
            },
        ];
        let expected = engine.answer_many(&owned_jobs).unwrap();
        // The borrowed-view path plans and executes the same answers.
        let jobs = [
            AnswerJob {
                matrix: &first,
                queries: &borrowed_first,
            },
            AnswerJob {
                matrix: &second,
                queries: &borrowed_second,
            },
        ];
        let plan = engine.plan(&jobs).unwrap();
        let mut workspace = EngineWorkspace::new();
        workspace.reserve(&plan).unwrap();
        let (answers, report) = engine.execute(&plan, &mut workspace).unwrap();
        assert_eq!(report.cpu_entries, 2);
        for (index, queries) in [&borrowed_first, &borrowed_second].into_iter().enumerate() {
            let view = answers.entry_answers(index).unwrap();
            assert_view_matches(&view, queries, &expected[index]);
        }
    }

    /// A CPU engine has no device-resident matrices, so work above the GPU
    /// threshold must still plan — on the actual CPU tier — and execute.
    #[cfg(feature = "gpu")]
    #[test]
    fn a_cpu_engine_plans_gpu_tier_work_on_the_cpu() {
        let engine = AnswerEngine::cpu();
        assert!(!engine.has_device());
        // 64 queries * rows * n(16) = MIN_GPU_MULTIPLICATIONS exactly.
        let rows = crate::MIN_GPU_MULTIPLICATIONS / (64 * 16);
        let host = matrix(71, rows);
        let prepared = engine.prepare(PARAMS, host.clone()).unwrap();
        let queries: Vec<_> = (0..64_u64).map(|index| query(71, 100 + index)).collect();
        let jobs = [AnswerJob {
            matrix: &prepared,
            queries: &queries,
        }];
        let threads = rayon::current_num_threads();
        // The raw policy would select the GPU tier, but the CPU engine
        // plans the CPU tier for the same shape.
        assert_eq!(
            select_answer_backend(64, rows, 16, threads),
            AnswerBackend::Gpu
        );
        let plan = engine.plan(&jobs).unwrap();
        assert_eq!(
            plan.entry(0).unwrap().backend(),
            select_cpu_backend(64, rows, 16, threads)
        );
        let mut workspace = EngineWorkspace::new();
        workspace.reserve(&plan).unwrap();
        let (answers, report) = engine.execute(&plan, &mut workspace).unwrap();
        assert_eq!(report.cpu_entries, 1);
        assert_eq!(report.gpu_entries, 0);
        assert_view_matches(
            &answers.entry_answers(0).unwrap(),
            &queries,
            &answer_batch(&PARAMS, &host, &queries).unwrap(),
        );
    }

    /// The true mixed-tier plan-execute run: one GPU-tier and one
    /// CPU-tier entry through the new API, the GPU answers bit-identical
    /// to the CPU reference. Requires a compute adapter.
    #[cfg(feature = "gpu")]
    #[ignore = "requires a compute adapter"]
    #[test]
    fn mixed_tier_plan_execute_matches_the_cpu_reference() {
        let engine = AnswerEngine::new(32 * 1024 * 1024).unwrap();
        assert!(engine.has_device(), "this test requires a compute adapter");
        let field = PrimeField::<MODULUS>::new();
        // GPU-tier entry: rows * 1 query * n(2) = MIN_GPU_MULTIPLICATIONS.
        let gpu_rows = crate::MIN_GPU_MULTIPLICATIONS / 2;
        let values: Vec<_> = (0..gpu_rows * 2)
            .map(|index| field.element_u32((index % 17) as u32))
            .collect();
        let gpu_host = EncryptedMatrix::from_parts(91, gpu_rows, 2, values).unwrap();
        let cpu_host = matrix(92, 2);
        let gpu_prepared = engine.prepare(GPU_PARAMS, gpu_host.clone()).unwrap();
        let cpu_prepared = engine.prepare(PARAMS, cpu_host.clone()).unwrap();
        let gpu_queries = [EncryptedQuery::from_parts(
            91,
            7,
            vec![field.element_u32(3), field.element_u32(5)],
        )];
        let cpu_queries = [query(92, 1)];
        let jobs = [
            AnswerJob {
                matrix: &gpu_prepared,
                queries: &gpu_queries,
            },
            AnswerJob {
                matrix: &cpu_prepared,
                queries: &cpu_queries,
            },
        ];
        let device_reference = answer_batch(&GPU_PARAMS, &gpu_host, &gpu_queries).unwrap();
        let core_reference = answer_batch(&PARAMS, &cpu_host, &cpu_queries).unwrap();

        let plan = engine.plan(&jobs).unwrap();
        assert_eq!(plan.entry(0).unwrap().backend(), AnswerBackend::Gpu);
        assert!(plan.entry(0).unwrap().gpu_segments() >= 1);
        assert_eq!(
            plan.entry(1).unwrap().backend(),
            select_cpu_backend(1, 2, 16, rayon::current_num_threads())
        );
        let mut workspace = EngineWorkspace::new();
        workspace.reserve(&plan).unwrap();
        let (answers, report) = engine.execute(&plan, &mut workspace).unwrap();
        assert_eq!(report.gpu_entries, 1);
        assert_eq!(report.cpu_entries, 1);
        assert_eq!(report.gpu_queries, 1);
        assert_eq!(report.gpu_chunks, 1);
        assert!(report.gpu_wait > Duration::ZERO);
        assert_view_matches(
            &answers.entry_answers(0).unwrap(),
            &gpu_queries,
            &device_reference,
        );
        assert_view_matches(
            &answers.entry_answers(1).unwrap(),
            &cpu_queries,
            &core_reference,
        );
    }
}
