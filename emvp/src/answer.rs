//! The plan → reserve → execute answer API over caller-owned arenas.
//!
//! A server answering many batches against one encrypted matrix would
//! otherwise re-allocate the query-major answer arena every time. This
//! module splits the answer work into three steps with a fixed allocation
//! profile, reusing the exact fill machinery of the single-query kernel so
//! every path produces bit-identical answers for the same inputs and CPU
//! tier:
//!
//! - [`AnswerPlan::plan`] validates the whole batch up front (parameters,
//!   matrix shape and values, per-query width and instance identifier) and
//!   fixes the dimensions, the CPU tier under the current rayon pool, and
//!   the inputs themselves, which the plan only borrows.
//! - [`AnswerWorkspace::reserve`] is the single growth point of a
//!   workspace: it grows the arena to the plan's word count and never
//!   shrinks it, so reserved memory stays at the peak a server actually
//!   uses.
//! - [`execute_answer_batch`] writes exactly the planned arena words,
//!   query-major, through the shared row kernel and the same dispatch grid
//!   as [`crate::answer_into`], without allocating; in steady state the
//!   parallel tier also leases its per-row staging from the workspace's
//!   scratch pool without allocating.
//!
//! The returned [`Answers`] borrows the filled arena immutably and pairs
//! contiguous arena slices with the batch's query identifiers, so clients
//! decode straight from the workspace without copying.

use std::sync::{Mutex, MutexGuard};

use prime_field_layer::{FieldElement, PrimeField};
use rayon::prelude::*;

use crate::dispatch::{AnswerBackend, select_cpu_backend};
use crate::params::EmvpParams;
use crate::protocol::{
    EncryptedMatrix, ProtocolError, check_len, fill_answer_batch_serial, fill_answer_row,
    validate_matrix_shape, validate_query_against_matrix,
};
use crate::view::{AnswerRef, QueryValues};

/// The validated dimensions and instance identifier of one answer batch.
///
/// A plan fixes these once and every arena size derives from them with
/// checked arithmetic.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AnswerShape {
    /// The matrix row count `m`, which is also the answer row count.
    rows: usize,
    /// The answer column count `s = n / b`.
    blocks: usize,
    /// The batch's query count.
    queries: usize,
    /// The public matrix-instance identifier every answer belongs to.
    instance_id: u128,
}

impl AnswerShape {
    /// Builds a shape from already-validated parts.
    ///
    /// Crate-internal constructor for executors whose batch validation runs
    /// outside this module and produce the same rows, blocks, query count,
    /// and instance identifier this type pins for the CPU path: the GPU
    /// answer path (device-resident shapes in `crate::gpu`) and the multi-
    /// matrix engine plan in `crate::engine`.
    pub(crate) const fn new(rows: usize, blocks: usize, queries: usize, instance_id: u128) -> Self {
        Self {
            rows,
            blocks,
            queries,
            instance_id,
        }
    }

    /// Returns the answer row count (`m`).
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the answer column count (`s = n / b`).
    #[must_use]
    pub const fn blocks(&self) -> usize {
        self.blocks
    }

    /// Returns the batch's query count.
    #[must_use]
    pub const fn queries(&self) -> usize {
        self.queries
    }

    /// Returns the public matrix-instance identifier.
    #[must_use]
    pub const fn instance_id(&self) -> u128 {
        self.instance_id
    }

    /// The field-element word count of one answer, `rows * blocks`.
    ///
    /// Returns `None` when the product overflows `usize`.
    #[must_use]
    pub const fn answer_words(&self) -> Option<usize> {
        self.rows.checked_mul(self.blocks)
    }

    /// The field-element word count of the whole query-major arena,
    /// `queries * answer_words`.
    ///
    /// Returns `None` when the product overflows `usize`.
    #[must_use]
    pub const fn arena_words(&self) -> Option<usize> {
        match self.answer_words() {
            Some(words) => self.queries.checked_mul(words),
            None => None,
        }
    }
}

/// A fully validated answer batch bound to its inputs by borrowing.
///
/// [`Self::plan`] runs the batch validation — parameters, matrix shape
/// and values, per-query width and instance identifier — and additionally
/// fixes the CPU tier from a snapshot of the current rayon pool, so
/// executing the plan is infallible except for arena capacity. The plan
/// holds the matrix and the queries only by reference, so executing it
/// against other inputs is unrepresentable; a plan can be executed
/// repeatedly and yields identical answers every time.
pub struct AnswerPlan<'a, const MODULUS: u32, Q: QueryValues<MODULUS>> {
    matrix: &'a EncryptedMatrix<MODULUS>,
    queries: &'a [Q],
    shape: AnswerShape,
    codeword_len: usize,
    block_size: usize,
    backend: AnswerBackend,
    answer_words: usize,
    arena_words: usize,
}

impl<'a, const MODULUS: u32, Q: QueryValues<MODULUS>> AnswerPlan<'a, MODULUS, Q> {
    /// Validates a full batch against `params` and `matrix` and plans its
    /// execution.
    ///
    /// Validation is all-or-nothing: the parameters and the matrix
    /// shape and values are checked once, and every query must have length
    /// `n` and carry the matrix's instance identifier. An empty batch is
    /// rejected with a `queries` length mismatch. The reported
    /// [`AnswerBackend`] is the CPU tier
    /// [`select_cpu_backend`] picks for the batch under the rayon pool as
    /// it exists at plan time; [`crate::execute_answer_batch`] runs exactly
    /// that tier.
    ///
    /// # Errors
    ///
    /// Returns an error for malformed parameters, a matrix whose shape or
    /// values disagree with `params`, an empty batch, a query with the
    /// wrong length or a foreign instance identifier, or dimension
    /// overflow.
    pub fn plan(
        params: &EmvpParams,
        matrix: &'a EncryptedMatrix<MODULUS>,
        queries: &'a [Q],
    ) -> Result<Self, ProtocolError> {
        let Some(first) = queries.first() else {
            return Err(ProtocolError::LengthMismatch {
                name: "queries",
                expected: 1,
                actual: 0,
            });
        };
        let n = validate_matrix_shape(params, matrix)?;
        validate_query_against_matrix(n, matrix.instance_id(), first)?;
        for query in &queries[1..] {
            validate_query_against_matrix(n, matrix.instance_id(), query)?;
        }
        let blocks = params.blocks()?;
        let rows = matrix.rows();
        let answer_words = rows
            .checked_mul(blocks)
            .ok_or(ProtocolError::DimensionOverflow)?;
        let arena_words = queries
            .len()
            .checked_mul(answer_words)
            .ok_or(ProtocolError::DimensionOverflow)?;
        let backend = select_cpu_backend(queries.len(), rows, n, rayon::current_num_threads());
        Ok(Self {
            matrix,
            queries,
            shape: AnswerShape {
                rows,
                blocks,
                queries: queries.len(),
                instance_id: matrix.instance_id(),
            },
            codeword_len: n,
            block_size: params.block_size(),
            backend,
            answer_words,
            arena_words,
        })
    }

    /// Returns the validated batch shape.
    #[must_use]
    pub const fn shape(&self) -> AnswerShape {
        self.shape
    }

    /// Returns the field-element word count of one answer, `rows * blocks`.
    #[must_use]
    pub const fn answer_words(&self) -> usize {
        self.answer_words
    }

    /// Returns the field-element word count of the whole query-major arena.
    #[must_use]
    pub const fn arena_words(&self) -> usize {
        self.arena_words
    }

    /// Returns the CPU tier [`crate::execute_answer_batch`] runs for this
    /// plan, fixed from a rayon pool snapshot at plan time.
    #[must_use]
    pub const fn backend(&self) -> AnswerBackend {
        self.backend
    }
}

/// Reusable staging for one answer row of the parallel kernel.
///
/// The staging row is filled by the shared row kernel and then copied into
/// its arena chunk, so leased scratch never aliases the arena. The buffer
/// serves the widest row its leases have seen; it is returned to the pool
/// afterwards.
#[derive(Debug)]
struct RowScratch<const MODULUS: u32> {
    row: Vec<FieldElement<MODULUS>>,
}

/// A pool of row staging buffers leased by the parallel answer kernel.
///
/// This mirrors the device buffer pool of the GPU answerer: a call pops a
/// staging row and returns it afterwards, so after a warm-up call the pool
/// serves every lease without allocating.
#[derive(Debug)]
struct ScratchPool<const MODULUS: u32> {
    rows: Mutex<Vec<RowScratch<MODULUS>>>,
}

impl<const MODULUS: u32> ScratchPool<MODULUS> {
    /// Pops a staging row of exactly `answer_words` elements.
    ///
    /// An empty pool, or one whose returned rows have a different width,
    /// pays one allocation; a warm pool serves the shape untouched.
    fn pop(&self, answer_words: usize) -> RowScratch<MODULUS> {
        let mut scratch = lock_recovered(&self.rows)
            .pop()
            .unwrap_or_else(|| RowScratch { row: Vec::new() });
        if scratch.row.len() != answer_words {
            let zero = PrimeField::<MODULUS>::new().element_u32(0);
            scratch.row.resize(answer_words, zero);
        }
        scratch
    }

    /// Returns a spent staging row to the pool.
    fn push(&self, scratch: RowScratch<MODULUS>) {
        lock_recovered(&self.rows).push(scratch);
    }
}

/// Locks `mutex`, recovering from poisoning like the GPU module's helper of
/// the same role: the pooled staging rows are self-consistent between
/// calls, so a panic in another thread mid-lease cannot have corrupted
/// anything.
fn lock_recovered<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// A caller-owned reusable answer arena plus the parallel kernel's scratch
/// pool.
///
/// The workspace owns the query-major answer arena of a batch. Its capacity
/// only ever grows, and only through [`Self::reserve`]: a steady-state
/// plan-reserve-execute cycle never allocates. The arena never shrinks, so
/// the workspace keeps the peak words a server actually uses. The
/// workspace additionally pools the per-row staging the parallel tier
/// leases, mirroring the GPU answerer's device buffer pool.
#[derive(Debug)]
pub struct AnswerWorkspace<const MODULUS: u32> {
    arena: Vec<FieldElement<MODULUS>>,
    row_scratch: ScratchPool<MODULUS>,
}

impl<const MODULUS: u32> Default for AnswerWorkspace<MODULUS> {
    fn default() -> Self {
        Self::new()
    }
}

impl<const MODULUS: u32> AnswerWorkspace<MODULUS> {
    /// Creates an empty workspace.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            arena: Vec::new(),
            row_scratch: ScratchPool {
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
    /// [`crate::execute_answer_batch`] writes into them. Reserving a plan
    /// that fits the current capacity is free and leaves the arena's
    /// storage pointer and capacity untouched.
    ///
    /// # Errors
    ///
    /// Returns [`ProtocolError::Capacity`] when the allocator refuses the
    /// growth, naming the plan's requirement and the workspace's usable
    /// words.
    pub fn reserve<Q: QueryValues<MODULUS>>(
        &mut self,
        plan: &AnswerPlan<'_, MODULUS, Q>,
    ) -> Result<(), ProtocolError> {
        let required = plan.arena_words();
        let available = self.arena.len();
        if required <= available {
            return Ok(());
        }
        self.arena
            .try_reserve_exact(required - available)
            .map_err(|_allocation_failure| ProtocolError::Capacity {
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

    /// Returns the arena for reading. Crate-internal view for consumers
    /// that assemble [`Answers`] outside this module.
    #[cfg(feature = "gpu")]
    pub(crate) fn arena(&self) -> &[FieldElement<MODULUS>] {
        &self.arena
    }

    /// Returns the arena for writing. Crate-internal view for consumers
    /// that fill the arena outside this module, such as the GPU answer
    /// path's staged-readback writer.
    #[cfg(feature = "gpu")]
    pub(crate) fn arena_mut(&mut self) -> &mut [FieldElement<MODULUS>] {
        &mut self.arena
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

/// A filled answer arena borrowed from an [`AnswerWorkspace`].
///
/// The arena is laid out query-major: query `i`'s answer occupies the
/// `rows * blocks` words starting at `i * answer_words`. [`Self::answer`]
/// and [`Self::iter`] pair those slices with the batch's query identifiers
/// without copying, so a client can decode straight from the workspace.
#[derive(Debug)]
pub struct Answers<'ws, const MODULUS: u32> {
    arena: &'ws [FieldElement<MODULUS>],
    shape: AnswerShape,
}

impl<'ws, const MODULUS: u32> Answers<'ws, MODULUS> {
    /// Pairs an already-filled query-major arena with its validated shape.
    ///
    /// Crate-internal constructor for executors that fill the arena outside
    /// this module and can only produce views over arenas they validated
    /// before filling: the GPU staged-readback writer and the multi-matrix
    /// engine's execute step.
    pub(crate) const fn from_arena(
        arena: &'ws [FieldElement<MODULUS>],
        shape: AnswerShape,
    ) -> Self {
        Self { arena, shape }
    }

    /// Returns the validated batch shape.
    #[must_use]
    pub const fn shape(&self) -> AnswerShape {
        self.shape
    }

    /// Returns the whole filled arena, query-major.
    #[must_use]
    pub const fn arena(&self) -> &'ws [FieldElement<MODULUS>] {
        self.arena
    }

    /// Returns the arena's field-element word count.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.arena.len()
    }

    /// Whether the arena holds no words; a plan rejects empty batches, so
    /// executed answers are never empty.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.arena.is_empty()
    }

    /// Pairs the answer of `queries[index]` with its public identifiers.
    ///
    /// The query slice must be the one the answers were planned with, so
    /// pairing stays honest: its length and, at `index`, the query's
    /// instance identifier are checked against the shape.
    ///
    /// # Errors
    ///
    /// Returns an error when `queries` does not match the batch length,
    /// `index` is out of range, the paired query carries a foreign
    /// instance identifier, or dimension arithmetic overflows.
    pub fn answer<Q: QueryValues<MODULUS>>(
        &self,
        queries: &[Q],
        index: usize,
    ) -> Result<AnswerRef<'ws, MODULUS>, ProtocolError> {
        check_len("queries", self.shape.queries, queries.len())?;
        let query = queries.get(index).ok_or(ProtocolError::LengthMismatch {
            name: "query index",
            expected: self.shape.queries,
            actual: index,
        })?;
        Self::check_instance(self.shape.instance_id, query)?;
        let answer_words = self.answer_words()?;
        let start = index
            .checked_mul(answer_words)
            .ok_or(ProtocolError::DimensionOverflow)?;
        let end = start
            .checked_add(answer_words)
            .ok_or(ProtocolError::DimensionOverflow)?;
        AnswerRef::new(
            self.shape.instance_id,
            query.query_id(),
            &self.arena[start..end],
            self.shape.rows,
            self.shape.blocks,
        )
    }

    /// Iterates every answer paired with its query's public identifiers.
    ///
    /// The query slice must be the one the answers were planned with, so
    /// pairing stays honest: its length and every query's instance
    /// identifier are checked against the shape before iteration starts.
    ///
    /// # Errors
    ///
    /// Returns an error when `queries` does not match the batch length, any
    /// query carries a foreign instance identifier, or dimension arithmetic
    /// overflows.
    #[expect(
        clippy::iter_not_returning_iterator,
        reason = "the all-or-nothing validation of `queries` needs the Result wrapper, and `IntoIterator for &Answers` is unrepresentable because pairing binds the iterator to the query type `Q`"
    )]
    pub fn iter<Q: QueryValues<MODULUS>>(
        &self,
        queries: &[Q],
    ) -> Result<impl ExactSizeIterator<Item = AnswerRef<'ws, MODULUS>>, ProtocolError> {
        check_len("queries", self.shape.queries, queries.len())?;
        for query in queries {
            Self::check_instance(self.shape.instance_id, query)?;
        }
        let answer_words = self.answer_words()?;
        Ok(self
            .arena
            .chunks_exact(answer_words)
            .zip(queries.iter())
            .map(|(values, query)| {
                AnswerRef::new_unchecked(
                    self.shape.instance_id,
                    query.query_id(),
                    self.shape.rows,
                    self.shape.blocks,
                    values,
                )
            }))
    }

    fn answer_words(&self) -> Result<usize, ProtocolError> {
        self.shape
            .answer_words()
            .ok_or(ProtocolError::DimensionOverflow)
    }

    fn check_instance<Q: QueryValues<MODULUS>>(
        expected: u128,
        query: &Q,
    ) -> Result<(), ProtocolError> {
        if query.instance_id() == expected {
            Ok(())
        } else {
            Err(ProtocolError::InstanceMismatch {
                name: "encrypted query",
                expected,
                actual: query.instance_id(),
            })
        }
    }
}

/// Executes a validated plan into a caller-owned workspace, without
/// allocating.
///
/// Writes exactly `plan.arena_words()` field elements into the workspace's
/// arena, query-major, through the same row kernel and dispatch grid as
/// [`crate::answer_into`]: the serial tier is the single-query path's
/// serial row loop, and the parallel tier distributes the same
/// flattened (query, row) grid across rayon workers, leasing one staging
/// row per row chunk from the workspace's scratch pool and returning it
/// afterwards, so a warm pool serves the call without allocating. The tier
/// is the plan's fixed [`AnswerBackend`], not a fresh decision, so
/// executing a plan is reproducible.
///
/// The returned [`Answers`] borrows the workspace's arena; the workspace is
/// only mutably borrowed for the duration of the call.
///
/// The [`Sync`] bound on `Q` lets the parallel tier read the queries across
/// rayon workers; every view and owned query is a plain shared-slice or
/// `Vec` aggregate, so the bound costs nothing.
///
/// # Errors
///
/// Returns [`ProtocolError::Capacity`] before touching the arena when the
/// workspace's usable words are below the plan's requirement; reserve the
/// workspace for the plan first.
pub fn execute_answer_batch<'ws, const MODULUS: u32, Q: QueryValues<MODULUS> + Sync>(
    plan: &AnswerPlan<'_, MODULUS, Q>,
    workspace: &'ws mut AnswerWorkspace<MODULUS>,
) -> Result<Answers<'ws, MODULUS>, ProtocolError> {
    let required = plan.arena_words();
    if workspace.capacity_words() < required {
        return Err(ProtocolError::Capacity {
            required,
            available: workspace.capacity_words(),
        });
    }
    {
        let arena = &mut workspace.arena[..required];
        match plan.backend {
            AnswerBackend::Rayon => fill_answer_batch_pooled(
                plan.matrix,
                plan.queries,
                arena,
                plan.codeword_len,
                plan.block_size,
                plan.shape.blocks,
                plan.shape.rows,
                &workspace.row_scratch,
            ),
            // A plan never records the GPU tier, so treating a hypothetical
            // one as serial keeps the match exhaustive without a wildcard;
            // the serial row loop is the only other CPU tier.
            #[cfg(feature = "gpu")]
            AnswerBackend::Gpu | AnswerBackend::SingleCore => fill_answer_batch_serial(
                plan.matrix,
                plan.queries,
                arena,
                plan.codeword_len,
                plan.block_size,
                plan.shape.blocks,
                plan.shape.rows,
            ),
            #[cfg(not(feature = "gpu"))]
            AnswerBackend::SingleCore => fill_answer_batch_serial(
                plan.matrix,
                plan.queries,
                arena,
                plan.codeword_len,
                plan.block_size,
                plan.shape.blocks,
                plan.shape.rows,
            ),
        }
    }
    Ok(Answers {
        arena: &workspace.arena[..required],
        shape: plan.shape,
    })
}

/// The parallel tier of the plan-reserve-execute path.
///
/// The same flattened (query, row) grid the serial tier walks, with each
/// answer row staged through a scratch leased from `pool` and returned
/// afterwards; the arena chunks, the row order, and the row kernel
/// ([`fill_answer_row`](crate::protocol)) are identical, so the tier's
/// output matches the serial tier exactly.
#[expect(
    clippy::too_many_arguments,
    reason = "the kernel mirrors fill_answer_batch_serial's argument layout and adds the scratch pool"
)]
fn fill_answer_batch_pooled<const MODULUS: u32, Q: QueryValues<MODULUS> + Sync>(
    matrix: &EncryptedMatrix<MODULUS>,
    queries: &[Q],
    arena: &mut [FieldElement<MODULUS>],
    n: usize,
    b: usize,
    s: usize,
    rows: usize,
    pool: &ScratchPool<MODULUS>,
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
