//! Server-side GPU answer path for the EMVP protocol (wgpu + WGSL).
//!
//! The server's answer phase dominates the protocol's online cost: it
//! performs `queries * rows * n` field multiplications per batch against
//! data that is entirely public (the encrypted matrix `M_hat`, the encrypted
//! queries `q_hat`, and the answers `M'`). This module moves that phase onto
//! a compute device through [`wgpu`] while the client stays on the CPU. The
//! trust boundary is unchanged: every word the GPU reads or writes is public
//! protocol data, no secret key material ever reaches the device, and
//! constant-time discipline is unnecessary on this path. The workload
//! threshold that decides between this path and the CPU path lives in
//! [`crate::dispatch`], whose [`AnswerDispatcher`](crate::dispatch::AnswerDispatcher)
//! runner owns the hand-off.
//!
//! # Montgomery-native interchange
//!
//! [`prime_field_layer::FieldElement`] stores each value as its canonical
//! Montgomery residue `a * 2^32 mod p`. The WGSL kernel in
//! [`ANSWER_WGSL`] reproduces the crate's Montgomery multiplication (REDC)
//! on those raw words, so uploads and readbacks move the words as-is:
//! [`GpuAnswerer::upload_matrix`] streams the raw words into the device
//! buffer, and [`GpuAnswerer::answer_batch`] wraps the returned words with
//! the checked [`prime_field_layer::FieldElement::try_from_raw`]. The
//! results are bit-identical to the CPU [`answer_batch`](crate::answer_batch),
//! which remains the reference implementation; the parity tests in
//! `emvp/tests/gpu.rs` pin this property empirically. Because the kernel's
//! final fold returns a canonical word, equality on raw words matches
//! equality on elements and no normalization pass is needed; a device word
//! outside `0..p` would be a broken readback and is rejected as
//! [`GpuError::NonCanonicalWord`] instead of silently canonicalized.
//!
//! # Pipelines and constants
//!
//! One compute pipeline exists per field modulus, cached in
//! [`GpuAnswerer`]. The shader receives the modulus `p`, the REDC constant
//! `-p^{-1} mod 2^32` (from [`PrimeField::montgomery_neg_inv`]), and the
//! Montgomery conversion constant `2^64 mod p` (from
//! [`PrimeField::montgomery_r2`]) as pipeline-overridable constants compiled
//! into the kernel. `R2` does not convert into Montgomery form — the buffers
//! already hold Montgomery residues — it cancels the lazy accumulator's
//! double `2^-32` fold in the kernel's final reduction. Per-call dimensions
//! (`n`, `b`, `s`, rows, batch, dispatch width) travel in a small uniform
//! buffer so changing protocol parameters never recompiles the pipeline.
//!
//! # Data flow and resource ownership
//!
//! [`upload_matrix`] is the explicit upload-once step: it copies the
//! row-major encrypted matrix into a device-side storage buffer held by the
//! returned [`GpuEncryptedMatrix`]. [`answer_batch`] then uploads the
//! query batch, dispatches one thread per output element, and reads the
//! answers back through a staging buffer.
//!
//! The per-batch device buffers (queries, uniform, output, staging
//! readback) are not created per call: each batch pops a scratch set from
//! a pool inside the answerer, grows it only when the batch exceeds every
//! shape the set has ever served, and pushes it back when the call
//! completes. A batch that errors discards its set instead, which only
//! costs the next call a fresh allocation. A set's capacities never shrink,
//! so a server answering steady shapes allocates device memory once.
//! Batches that run concurrently pop distinct sets, so the pool grows to
//! the concurrency level and stays there. Uploads go through `Queue::write_buffer_with` staging views, so
//! the `FieldElement::to_raw` word pass, the little-endian byte conversion,
//! and the staging copy are fused into a single pass with no intermediate
//! allocation (all wgpu-supported hosts are little-endian, and the
//! workspace denies `unsafe`). Readbacks map only the live slice of the
//! staging buffer and stream the answer words straight into a caller-owned
//! arena in one pass over the mapped bytes
//! ([`GpuAnswerer::execute_answer_batch_into`]); the legacy
//! [`GpuAnswerer::answer_batch`] adapters run that same into-write and
//! convert the arena into per-query [`AnswerMatrix`]s on top.
//!
//! # Blocking behavior
//!
//! The API is synchronous. wgpu's adapter search, device request, and
//! `map_async` completion genuinely require waiting, so
//! [`GpuAnswerer::new`] blocks on adapter and device acquisition through
//! [`pollster`] and the readback phase blocks the calling thread with a
//! [`wgpu::PollType::Wait`] device poll. No method fakes nonblocking
//! behavior; latency-sensitive callers should run batches on a thread that
//! is allowed to block.
//!
//! # Phase timings
//!
//! [`GpuAnswerer::answer_batch_with_timings`] reports the host-side
//! wall-clock breakdown of a batch in [`PhaseTimings`]. The GPU benchmark
//! prints a fixed set of diagnostic calls collected outside Criterion timing.
//!
//! # Errors
//!
//! Every failure mode surfaces as a [`GpuError`] rather than a
//! [`crate::ProtocolError`]: the GPU path is server plumbing, servers use
//! [`GpuAnswerer`] directly, and folding it into the protocol error type
//! would couple client-side decoding to the server's hardware. Shape
//! validation reuses the protocol crate's server-side checks
//! ([`crate::protocol::validate_matrix_shape`] and the per-query validator
//! next to it) so CPU, dispatcher, and GPU reject the same shapes before
//! any work; those failures are mapped onto their [`GpuError`]
//! equivalents.
//!
//! # Experimental
//!
//! This is experimental cryptography running on an unaudited kernel; the
//! WGSL is compiled by naga at runtime, so shader failures surface as
//! device errors rather than compile-time failures. The parse-and-validate
//! test in this module pins the embedded shader's WGSL well-formedness
//! device-independently; only the device-side execution itself needs
//! hardware.

use std::collections::HashMap;
use std::fmt;
use std::sync::{Mutex, MutexGuard};
use std::time::{Duration, Instant};

use prime_field_layer::{FieldElement, PrimeField};

use crate::answer::{AnswerWorkspace, Answers};
use crate::params::EmvpParams;
use crate::protocol::{
    AnswerMatrix, EncryptedMatrix, EncryptedQuery, ProtocolError, validate_matrix_shape,
    validate_query_against_matrix,
};

pub(crate) mod packed;

/// The answer compute kernel, compiled once per modulus.
const ANSWER_WGSL: &str = include_str!("answer.wgsl");

/// Threads per workgroup; must match the `WORKGROUP_SIZE` const in
/// [`ANSWER_WGSL`].
const WORKGROUP_SIZE: u32 = 256;

/// [`WORKGROUP_SIZE`] as a host word count.
const WORKGROUP_SIZE_USIZE: usize = WORKGROUP_SIZE as usize;

/// Bytes in one field word; the on-the-wire width of a
/// [`FieldElement::to_raw`] residue.
const WORD_BYTES: usize = 4;

/// Words in the `Dims` uniform: six used words plus two padding words.
const DIMS_UNIFORM_WORDS: usize = 8;

/// [`DIMS_UNIFORM_WORDS`] in bytes; the fixed size of every uniform buffer.
const DIMS_UNIFORM_BYTES: u64 = (DIMS_UNIFORM_WORDS * WORD_BYTES) as u64;

/// Largest dispatchable workgroup count per dimension guaranteed by wgpu.
const MAX_WORKGROUPS_PER_DIMENSION: u32 = 65_535;

/// Largest answer count the kernel can address without wrapping its `u32`
/// thread index.
///
/// The shader reconstructs each thread's linear index from the `u32`
/// workgroup coordinates as `((y * workgroups_x) + x) * WORKGROUP_SIZE +
/// local_id.x`. Padding `answer_words` up to whole workgroups plus the
/// partial last workgroup row costs at most `WORKGROUP_SIZE` words
/// horizontally and `MAX_WORKGROUPS_PER_DIMENSION` workgroups vertically, so
/// the largest issued index is below
/// `answer_words + MAX_WORKGROUPS_PER_DIMENSION * WORKGROUP_SIZE`. Capping
/// `answer_words` at `u32::MAX + 1 - MAX_WORKGROUPS_PER_DIMENSION *
/// WORKGROUP_SIZE` keeps every issued index inside `u32`; larger indices
/// would wrap around, pass the kernel's `index >= total` guard, and race
/// other threads' output slots. It also keeps `workgroups_y` far below the
/// per-dimension dispatch limit.
const MAX_ANSWER_WORDS: usize =
    (u32::MAX as usize + 1) - (MAX_WORKGROUPS_PER_DIMENSION as usize) * WORKGROUP_SIZE_USIZE;

/// A rejected GPU operation.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum GpuError {
    /// No compute adapter matched the request. The machine may have no GPU,
    /// no Vulkan-compatible driver, or the loader may be missing; tests
    /// mark their adapter-dependent cases `#[ignore = "requires a compute
    /// adapter"]` and benchmarks print a notice.
    NoAdapter {
        /// The adapter-search failure description.
        reason: String,
    },
    /// The adapter was found but the logical device request failed, usually
    /// because the requested limits exceed the adapter's capabilities.
    RequestDevice(String),
    /// Protocol parameters were malformed.
    Params(crate::params::ParamsError),
    /// Dimension arithmetic overflowed.
    DimensionOverflow,
    /// A buffer exceeded the device's capacity, either the adapter's storage
    /// limits or the u32 indexing bound the kernel places on `rows * n` and
    /// on the batched output count.
    UploadTooLarge {
        /// The requested element (field element or u32 word) count.
        elements: usize,
        /// The largest element count the device accepts for one buffer.
        max_elements: usize,
    },
    /// The driver refused a buffer allocation because the device ran out of
    /// memory. Unlike [`Self::UploadTooLarge`], which is decided by the
    /// adapter's reported limits before any allocation, this surfaces the
    /// scoped out-of-memory error of an allocation the limits allowed.
    OutOfMemory {
        /// Which buffer allocation failed.
        context: &'static str,
    },
    /// Long-lived matrix residency would exceed the engine's configured
    /// device-memory budget.
    BudgetExceeded {
        /// Bytes required by this matrix.
        requested_bytes: u64,
        /// Bytes still available in the configured budget.
        available_bytes: u64,
    },
    /// A slice did not have the required length.
    LengthMismatch {
        /// The rejected slice.
        name: &'static str,
        /// The required length.
        expected: usize,
        /// The observed length.
        actual: usize,
    },
    /// A caller-owned answer arena was too small for the batch's answers.
    /// Reported before any device work or arena mutation, so the caller can
    /// grow the workspace and retry whole.
    Capacity {
        /// The required arena word count.
        required: usize,
        /// The available arena word count.
        available: usize,
    },
    /// Two protocol artifacts belong to different matrix instances.
    InstanceMismatch {
        /// The required public instance identifier.
        expected: u128,
        /// The observed public instance identifier.
        actual: u128,
    },
    /// The field modulus is outside the kernel's supported range; the WGSL
    /// arithmetic requires `2 < MODULUS < 2^31`, which every protocol field
    /// satisfies.
    UnsupportedModulus {
        /// The rejected field modulus.
        modulus: u32,
    },
    /// A queue submission or device poll failed.
    Submission(String),
    /// Mapping or reading the staging buffer failed.
    Map(String),
    /// A device answer word was not a canonical Montgomery residue, so the
    /// readback is corrupt and cannot be wrapped into a
    /// [`prime_field_layer::FieldElement`] without silently canonicalizing
    /// it. Carries the rejected word and the modulus it was checked against.
    NonCanonicalWord {
        /// The rejected raw word.
        word: u32,
        /// The field modulus the word was checked against.
        modulus: u32,
    },
    /// A protocol validation failure outside the four shapes the shared
    /// server-side validation produces ([`Self::LengthMismatch`],
    /// [`Self::DimensionOverflow`], [`Self::Params`], and
    /// [`Self::InstanceMismatch`]). Unreachable from that validation;
    /// retained so the mapping from [`ProtocolError`] stays total and
    /// fail-closed rather than dropping an unexpected variant.
    Protocol(ProtocolError),
}

impl fmt::Display for GpuError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NoAdapter { reason } => {
                write!(formatter, "no suitable compute adapter: {reason}")
            }
            Self::RequestDevice(reason) => {
                write!(formatter, "logical device request failed: {reason}")
            }
            Self::Params(error) => error.fmt(formatter),
            Self::DimensionOverflow => formatter.write_str("dimension arithmetic overflowed"),
            Self::UploadTooLarge {
                elements,
                max_elements,
            } => write!(
                formatter,
                "buffer of {elements} elements exceeds the device maximum of {max_elements}"
            ),
            Self::OutOfMemory { context } => {
                write!(
                    formatter,
                    "device ran out of memory allocating the {context}"
                )
            }
            Self::BudgetExceeded {
                requested_bytes,
                available_bytes,
            } => write!(
                formatter,
                "GPU matrix needs {requested_bytes} bytes but only {available_bytes} budget bytes remain"
            ),
            Self::LengthMismatch {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "{name} length mismatch: expected {expected}, got {actual}"
            ),
            Self::Capacity {
                required,
                available,
            } => write!(
                formatter,
                "answer workspace holds {available} words but the batch needs {required}"
            ),
            Self::InstanceMismatch { expected, actual } => write!(
                formatter,
                "encrypted query instance mismatch: expected {expected}, got {actual}"
            ),
            Self::UnsupportedModulus { modulus } => write!(
                formatter,
                "the GPU answer kernel requires 2 < modulus < 2^31, got modulus {modulus}"
            ),
            Self::Submission(reason) => {
                write!(formatter, "GPU submission or poll failed: {reason}")
            }
            Self::Map(reason) => write!(formatter, "GPU readback mapping failed: {reason}"),
            Self::NonCanonicalWord { word, modulus } => write!(
                formatter,
                "device answer word {word} is not a canonical Montgomery residue below modulus {modulus}"
            ),
            Self::Protocol(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for GpuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Params(error) => Some(error),
            Self::Protocol(error) => Some(error),
            _ => None,
        }
    }
}

impl From<crate::params::ParamsError> for GpuError {
    fn from(error: crate::params::ParamsError) -> Self {
        Self::Params(error)
    }
}

impl From<ProtocolError> for GpuError {
    fn from(error: ProtocolError) -> Self {
        match error {
            // The exact variants the shared server-side shape validation can
            // produce, mapped onto their precise GPU equivalents.
            ProtocolError::LengthMismatch {
                name,
                expected,
                actual,
            } => Self::LengthMismatch {
                name,
                expected,
                actual,
            },
            ProtocolError::DimensionOverflow => Self::DimensionOverflow,
            ProtocolError::Params(error) => Self::Params(error),
            ProtocolError::InstanceMismatch {
                name: _,
                expected,
                actual,
            } => Self::InstanceMismatch { expected, actual },
            // Every other protocol failure is unreachable from the shared
            // shape validation; keeping the arm total means a future variant
            // cannot be silently dropped here.
            other => Self::Protocol(other),
        }
    }
}

/// Wall-clock host-side breakdown of one
/// [`GpuAnswerer::answer_batch_with_timings`] call.
///
/// Every field is the elapsed [`Instant`] delta measured on the calling
/// thread while executing that phase, and the phases run sequentially, so
/// [`Self::total`] approximates the call's wall time across the hot path.
/// One-time costs are not attributed to any phase: shader compilation on a
/// modulus's first call happens before the first phase, and every later
/// call reuses the cached pipeline.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PhaseTimings {
    /// Locking the scratch pool and growing the reused device buffers when
    /// this batch is larger than every batch the leased set has served.
    /// Steady-state calls pay only the pool lock here; a growth spike means
    /// a new peak shape arrived.
    pub prepare_buffers: Duration,
    /// Streaming the queries into staging memory as little-endian raw
    /// Montgomery words and scheduling the query and uniform uploads. The
    /// `FieldElement::to_raw` word pass and the byte conversion are fused
    /// into the staging write; wgpu does not transfer staged writes until
    /// the next `Queue::submit`, so the device-side transfer itself is
    /// waited on in [`Self::wait_readback`].
    pub encode_upload_queries: Duration,
    /// Creating the bind group, recording the dispatch and the readback
    /// copy into a command encoder, and `Queue::submit`. The submit is
    /// asynchronous by wgpu's contract, so this phase measures host-side
    /// recording cost only.
    pub dispatch_submit: Duration,
    /// `map_async` on the staging slice plus `Device::poll` with
    /// [`wgpu::PollType::Wait`] on this batch's submission, which blocks
    /// the calling thread until the device has drained the uploads, the
    /// dispatch, and the device-to-host copy, and the mapping callback has
    /// run. This is the phase that waits on GPU execution.
    pub wait_readback: Duration,
    /// Reconstructing the per-query answer vectors from the mapped staging
    /// bytes: the checked `FieldElement::try_from_raw` word pass and the
    /// split into [`AnswerMatrix`]s. Pure host work with no device
    /// dependency.
    pub reconstruct: Duration,
}

impl PhaseTimings {
    /// Sum of all phases; approximately the call's wall time across the
    /// measured hot path.
    #[must_use]
    pub fn total(&self) -> Duration {
        self.prepare_buffers
            + self.encode_upload_queries
            + self.dispatch_submit
            + self.wait_readback
            + self.reconstruct
    }
}

/// An encrypted matrix uploaded to the device as raw Montgomery words.
///
/// The handle owns the device buffer, so the host copy of the ciphertext can
/// be dropped after upload; [`GpuAnswerer::answer_batch`] reuses the buffer
/// for every query batch. It also carries the protocol parameters the
/// matrix was encrypted under, which fix the block structure `b` and `s`
/// the kernel answers with. Drop the handle to release the device memory.
pub struct GpuEncryptedMatrix<const MODULUS: u32> {
    buffer: wgpu::Buffer,
    params: EmvpParams,
    rows: usize,
    columns: usize,
    instance_id: u128,
}

impl<const MODULUS: u32> GpuEncryptedMatrix<MODULUS> {
    /// Returns the encrypted matrix rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the encrypted matrix columns (`n = 2k`).
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Returns the protocol parameters the matrix was encrypted under.
    #[must_use]
    pub const fn params(&self) -> EmvpParams {
        self.params
    }

    /// Returns the public matrix-instance identifier.
    #[must_use]
    pub const fn instance_id(&self) -> u128 {
        self.instance_id
    }
}

impl<const MODULUS: u32> fmt::Debug for GpuEncryptedMatrix<MODULUS> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("GpuEncryptedMatrix")
            .field("rows", &self.rows)
            .field("columns", &self.columns)
            .field("instance_id", &self.instance_id)
            .finish_non_exhaustive()
    }
}

/// The GPU answer server: a logical device, a command queue, and the
/// per-modulus pipeline and buffer caches.
///
/// Batches lease reusable buffer sets from [`Self::scratch_pool`], so a
/// steady workload allocates device memory once per concurrent caller
/// rather than once per call. See [`AnswerScratch`] for the growth
/// contract.
pub struct GpuAnswerer {
    pub(crate) device: wgpu::Device,
    pub(crate) queue: wgpu::Queue,
    pipelines: Mutex<HashMap<u32, wgpu::ComputePipeline>>,
    scratch_pool: Mutex<Vec<AnswerScratch>>,
    pub(crate) packed_scratch_pool: Mutex<Vec<packed::PackedScratch>>,
}

impl GpuAnswerer {
    /// Initializes a compute device for the answer phase.
    ///
    /// The high-performance adapter is requested; on machines with a single
    /// discrete GPU that selects it. Buffer limits are raised to whatever
    /// the adapter supports, because production encrypted matrices reach
    /// gibibytes while wgpu's default limits cap buffers at 256 MiB.
    ///
    /// This blocks the calling thread: wgpu's adapter search and device
    /// request are genuinely asynchronous interfaces, and the acquisition
    /// waits on them through [`pollster`]. See the module's [blocking
    /// behavior](self#blocking-behavior) documentation.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::NoAdapter`] when no Vulkan/GL adapter is
    /// available (the container case), and [`GpuError::RequestDevice`] when
    /// the adapter rejects the device request.
    pub fn new() -> Result<Self, GpuError> {
        pollster::block_on(Self::acquire())
    }

    /// The async half of [`Self::new`]: adapter and device acquisition.
    ///
    /// wgpu's request interfaces are async; [`pollster`] bridges them to
    /// the synchronous API. Kept private so no public method pretends the
    /// wait is optional.
    async fn acquire() -> Result<Self, GpuError> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                apply_limit_buckets: false,
            })
            .await
            .map_err(|error| GpuError::NoAdapter {
                reason: error.to_string(),
            })?;

        let adapter_limits = adapter.limits();
        let mut required_limits = wgpu::Limits::default();
        required_limits.max_buffer_size = required_limits
            .max_buffer_size
            .max(adapter_limits.max_buffer_size);
        required_limits.max_storage_buffer_binding_size = required_limits
            .max_storage_buffer_binding_size
            .max(adapter_limits.max_storage_buffer_binding_size);
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("emvp-answer-device"),
                required_features: wgpu::Features::empty(),
                required_limits,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            })
            .await
            .map_err(|error| GpuError::RequestDevice(error.to_string()))?;
        Ok(Self {
            device,
            queue,
            pipelines: Mutex::new(HashMap::new()),
            scratch_pool: Mutex::new(Vec::new()),
            packed_scratch_pool: Mutex::new(Vec::new()),
        })
    }

    /// Uploads an encrypted matrix as raw Montgomery words.
    ///
    /// This is the one-time transfer per matrix: the returned
    /// [`GpuEncryptedMatrix`] owns the device buffer and every later
    /// [`Self::answer_batch`] against it reuses the words in place. The
    /// upload streams the raw words straight into a
    /// `Queue::write_buffer_with` staging view and flushes the transfer
    /// with an immediate empty submission, so the copy overlaps whatever
    /// the host does next instead of hiding behind the first answer
    /// batch's submission. The parameters must match the encryption
    /// parameters of `matrix`; they fix the block structure the kernel
    /// answers with.
    ///
    /// Shape validation reuses
    /// [`crate::protocol::validate_matrix_shape`], the same check the CPU
    /// answer path and the [`AnswerDispatcher`](crate::dispatch::AnswerDispatcher)
    /// run, so all server paths reject the same malformed shapes.
    ///
    /// # Errors
    ///
    /// Returns an error before any transfer if the parameters are malformed,
    /// the parameter dimensions disagree with the matrix, the matrix is
    /// empty, its value length differs from `rows * columns`, the element
    /// count exceeds the device's per-buffer capacity or the kernel's u32
    /// index bound, or the driver refuses the buffer allocation.
    pub fn upload_matrix<const MODULUS: u32>(
        &self,
        params: &EmvpParams,
        matrix: &EncryptedMatrix<MODULUS>,
    ) -> Result<GpuEncryptedMatrix<MODULUS>, GpuError> {
        check_modulus::<MODULUS>()?;
        let n = validate_matrix_shape(params, matrix)?;
        let rows = matrix.rows();
        // The shared validator already multiplied `rows * n` successfully,
        // so this product cannot overflow.
        let words = rows.checked_mul(n).ok_or(GpuError::DimensionOverflow)?;
        // The u32 narrowing doubles as the kernel's u32 index bound: every
        // matrix index is below `rows * n = words`.
        let words_u32 = u32::try_from(words).map_err(|_conversion| GpuError::UploadTooLarge {
            elements: words,
            max_elements: u32::MAX as usize,
        })?;
        self.check_buffer_words(u64::from(words_u32))?;

        let buffer = create_buffer_checked(
            &self.device,
            "encrypted matrix buffer",
            &wgpu::BufferDescriptor {
                label: Some("emvp-encrypted-matrix"),
                size: byte_len_of_words(words_u32),
                usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;
        let mut view = staged_write_view(&self.queue, &buffer, words_u32)?;
        fill_matrix_view(&mut view, matrix.values());
        drop(view);
        // Flush the one-time upload now rather than leaving it queued behind
        // the next answer batch's submission.
        self.queue.submit([]);

        Ok(GpuEncryptedMatrix {
            buffer,
            params: *params,
            rows,
            columns: n,
            instance_id: matrix.instance_id(),
        })
    }

    /// Answers a batch of encrypted queries against one uploaded matrix on
    /// the device.
    ///
    /// Every query is answered exactly as the CPU
    /// [`answer_batch`](crate::answer_batch) answers it alone: for each
    /// output `(query, row, block)` the kernel accumulates the `b` raw word
    /// products into a 96-bit integer accumulator and reduces it once with
    /// a two-step Montgomery fold plus an `R2` correction, writing one
    /// canonical word per element of the query-major answer arena.
    /// Validation is all-or-nothing and
    /// mirrors the CPU path; results are bit-identical to it. This is the
    /// temporary allocating adapter over the plan-reserve-execute path
    /// ([`Self::answer_batch_plan`] and [`Self::execute_answer_batch_into`]);
    /// it pays one arena allocation plus one per-query [`AnswerMatrix`]
    /// conversion the into-API avoids.
    ///
    /// # Errors
    ///
    /// Returns an error before any device work if the parameters are
    /// malformed, the batch is empty, any query has the wrong length or a
    /// foreign instance identifier, an index or size would overflow, the
    /// query/output buffers exceed the device's capacity, or the driver
    /// refuses a buffer allocation.
    pub fn answer_batch<const MODULUS: u32>(
        &self,
        matrix: &GpuEncryptedMatrix<MODULUS>,
        queries: &[EncryptedQuery<MODULUS>],
    ) -> Result<Vec<AnswerMatrix<MODULUS>>, GpuError> {
        self.answer_batch_with_timings(matrix, queries)
            .map(|(answers, _timings)| answers)
    }

    /// [`Self::answer_batch`] with the host-side phase breakdown of
    /// [`PhaseTimings`] attached.
    ///
    /// The timings make the host-versus-device split of one batch
    /// observable: how long the host spends preparing and submitting device
    /// work ([`PhaseTimings::prepare_buffers`],
    /// [`PhaseTimings::encode_upload_queries`],
    /// [`PhaseTimings::dispatch_submit`]), how long the calling thread
    /// blocks on the device ([`PhaseTimings::wait_readback`]), and how long
    /// the host spends reconstructing the answers afterwards
    /// ([`PhaseTimings::reconstruct`], which on this adapter covers the
    /// into-arena write and the per-query [`AnswerMatrix`] conversion).
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::answer_batch`].
    pub fn answer_batch_with_timings<const MODULUS: u32>(
        &self,
        matrix: &GpuEncryptedMatrix<MODULUS>,
        queries: &[EncryptedQuery<MODULUS>],
    ) -> Result<(Vec<AnswerMatrix<MODULUS>>, PhaseTimings), GpuError> {
        check_modulus::<MODULUS>()?;
        let shape = self.answer_shape(matrix, queries)?;
        let mut workspace = AnswerWorkspace::<MODULUS>::new();
        workspace
            .reserve_words(shape.answer_words)
            .map_err(map_capacity_error)?;
        let host_shape =
            crate::answer::AnswerShape::new(shape.rows, shape.s, shape.batch, matrix.instance_id());
        let mut timings = {
            let arena = workspace.arena_mut();
            self.execute_validated_batch_into(
                matrix,
                queries,
                &shape,
                &mut arena[..shape.answer_words],
            )?
        };
        let conversion_start = Instant::now();
        let answers = owned_answers(
            &workspace.arena()[..shape.answer_words],
            queries,
            &host_shape,
        )?;
        timings.reconstruct += conversion_start.elapsed();
        Ok((answers, timings))
    }

    /// Plans a single-batch GPU answer without device-side side effects on
    /// the batch itself.
    ///
    /// This is the plan step of the GPU plan-reserve-execute path: it runs
    /// the same all-or-nothing validation as
    /// [`Self::answer_batch_with_timings`] and derives and validates the
    /// dispatch grid and uniform bytes, so a plan that returns here is
    /// executable except for caller-side arena capacity. It also warms the
    /// per-modulus pipeline cache, moving one-time shader compilation out
    /// of the execute call.
    ///
    /// The returned [`crate::answer::AnswerShape`] is the CPU path's shape
    /// type: `shape.arena_words()` is the exact word count
    /// [`Self::execute_answer_batch_into`] writes into the caller's
    /// workspace, so a caller can size an [`AnswerWorkspace`] from it.
    ///
    /// # Errors
    ///
    /// Returns an error before any device work if the parameters are
    /// malformed, the batch is empty, any query has the wrong length or a
    /// foreign instance identifier, an index or size would overflow, the
    /// query/output buffers exceed the device's capacity, or the kernel's
    /// dispatch grid cannot be derived.
    pub fn answer_batch_plan<const MODULUS: u32>(
        &self,
        matrix: &GpuEncryptedMatrix<MODULUS>,
        queries: &[EncryptedQuery<MODULUS>],
    ) -> Result<crate::answer::AnswerShape, GpuError> {
        check_modulus::<MODULUS>()?;
        let shape = self.answer_shape(matrix, queries)?;
        let _dispatch = self.answer_dispatch::<MODULUS>(&shape)?;
        Ok(crate::answer::AnswerShape::new(
            shape.rows,
            shape.s,
            shape.batch,
            matrix.instance_id(),
        ))
    }

    /// Executes a planned single-batch GPU answer into a caller-owned
    /// workspace, without allocating.
    ///
    /// The execute step of the GPU plan-reserve-execute path: it re-runs
    /// the plan's validation, checks the workspace's capacity first, and
    /// then runs the same pipeline as [`Self::answer_batch_with_timings`]
    /// — scratch lease from the pool, query upload, dispatch, staged
    /// readback — but streams the readback words straight into the
    /// workspace's arena subslice `[0, shape.arena_words())` in the CPU
    /// path's query-major layout: query `q` of the batch occupies
    /// `[q * rows * s, (q + 1) * rows * s)`. A warm scratch pool serves the
    /// whole call without allocating; the scratch set returns to the pool
    /// on success and is dropped on any error path, as before.
    ///
    /// The returned [`Answers`] borrows the workspace's arena, so a client
    /// decodes straight from it without copying.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::Capacity`] before any device work or arena
    /// mutation when `workspace.capacity_words()` is below
    /// `shape.arena_words()`; returns the plan's validation errors and the
    /// device errors of [`Self::answer_batch`] otherwise. On an error the
    /// arena's contents are unspecified: callers must treat it as
    /// unwritten.
    pub fn execute_answer_batch_into<'ws, const MODULUS: u32>(
        &self,
        matrix: &GpuEncryptedMatrix<MODULUS>,
        queries: &[EncryptedQuery<MODULUS>],
        workspace: &'ws mut AnswerWorkspace<MODULUS>,
    ) -> Result<(Answers<'ws, MODULUS>, PhaseTimings), GpuError> {
        check_modulus::<MODULUS>()?;
        let shape = self.answer_shape(matrix, queries)?;
        // Capacity first: reject before any device work or mutation.
        check_arena_capacity(workspace.capacity_words(), shape.answer_words)?;
        let host_shape =
            crate::answer::AnswerShape::new(shape.rows, shape.s, shape.batch, matrix.instance_id());
        let timings = {
            let arena = workspace.arena_mut();
            self.execute_validated_batch_into(
                matrix,
                queries,
                &shape,
                &mut arena[..shape.answer_words],
            )?
        };
        Ok((
            Answers::from_arena(&workspace.arena()[..shape.answer_words], host_shape),
            timings,
        ))
    }

    /// The shared validated pipeline behind
    /// [`Self::execute_answer_batch_into`] and the legacy allocating
    /// adapters: lease scratch, prepare, encode and upload, submit, wait,
    /// and stream the staged answers into `dest`.
    ///
    /// `dest` must hold at least `shape.answer_words` words; exactly that
    /// prefix is written, query-major. On every success path the leased
    /// scratch set returns to the pool; on any error path it is dropped
    /// instead, which only costs the next call a fresh allocation.
    fn execute_validated_batch_into<const MODULUS: u32>(
        &self,
        matrix: &GpuEncryptedMatrix<MODULUS>,
        queries: &[EncryptedQuery<MODULUS>],
        shape: &AnswerShape,
        dest: &mut [FieldElement<MODULUS>],
    ) -> Result<PhaseTimings, GpuError> {
        // Fail-closed: the callers slice `dest` to the validated length,
        // and the write below trusts that slice.
        check_arena_capacity(dest.len(), shape.answer_words)?;
        let dispatch = self.answer_dispatch::<MODULUS>(shape)?;
        let mut timings = PhaseTimings::default();
        let pooled = {
            let mut pool = lock_recovered(&self.scratch_pool);
            pool.pop()
        };
        let mut scratch = match pooled {
            Some(scratch) => scratch,
            None => {
                AnswerScratch::new(&self.device, shape.query_words_u32, shape.answer_words_u32)?
            }
        };
        let (query_buffer, uniform_buffer, output_buffer, staging_buffer) = {
            let start = Instant::now();
            let buffers =
                scratch.prepare(&self.device, shape.query_words_u32, shape.answer_words_u32)?;
            timings.prepare_buffers = start.elapsed();
            buffers
        };

        timings.encode_upload_queries = {
            let start = Instant::now();
            let mut view = staged_write_view(&self.queue, query_buffer, shape.query_words_u32)?;
            fill_query_view(&mut view, queries);
            drop(view);
            self.queue
                .write_buffer(uniform_buffer, 0, &dispatch.uniform);
            start.elapsed()
        };

        let answer_bytes = byte_len_of_words(shape.answer_words_u32);
        let submission = {
            let start = Instant::now();
            let submission = self.submit_answer_dispatch::<MODULUS>(
                matrix,
                &dispatch.pipeline,
                (query_buffer, uniform_buffer, output_buffer, staging_buffer),
                dispatch.workgroups,
                answer_bytes,
            );
            timings.dispatch_submit = start.elapsed();
            submission
        };

        {
            let start = Instant::now();
            self.wait_for_staged_slice(staging_buffer, answer_bytes, submission)?;
            timings.wait_readback = start.elapsed();
        }

        {
            let start = Instant::now();
            reconstruct_staged_into::<MODULUS>(
                staging_buffer,
                &mut dest[..shape.answer_words],
                shape.answer_words_u32,
            )?;
            timings.reconstruct = start.elapsed();
        }
        // Return the set to the pool. Error paths above drop it instead,
        // which only costs the next call a fresh allocation.
        lock_recovered(&self.scratch_pool).push(scratch);
        Ok(timings)
    }

    /// Derives the pipeline, dispatch grid, and uniform bytes for one
    /// validated answer shape — the shared preflight of the plan and
    /// execute paths, so both reject the same undeliverable shapes.
    fn answer_dispatch<const MODULUS: u32>(
        &self,
        shape: &AnswerShape,
    ) -> Result<AnswerDispatch, GpuError> {
        let pipeline = self.answer_pipeline::<MODULUS>();
        let workgroups = dispatch_grid(shape.answer_words)?;
        let uniform = dims_uniform_bytes(
            shape.n,
            shape.b,
            shape.s,
            shape.rows,
            shape.batch,
            workgroups.0,
        )?;
        Ok(AnswerDispatch {
            pipeline,
            workgroups,
            uniform,
        })
    }

    /// Returns the cached answer pipeline for this modulus, compiling it on
    /// first use.
    pub(crate) fn answer_pipeline<const MODULUS: u32>(&self) -> wgpu::ComputePipeline {
        // Pipelines are immutable once built, so a panic in another thread
        // mid-insert cannot have corrupted anything: recover the guard. The
        // guard is dropped before compilation so concurrent batches are not
        // blocked behind shader work.
        let cached = lock_recovered(&self.pipelines).get(&MODULUS).cloned();
        if let Some(pipeline) = cached {
            return pipeline;
        }
        let pipeline = Self::build_pipeline::<MODULUS>(&self.device);
        lock_recovered(&self.pipelines).insert(MODULUS, pipeline.clone());
        pipeline
    }

    /// Compiles [`ANSWER_WGSL`] with this modulus's Montgomery constants.
    fn build_pipeline<const MODULUS: u32>(device: &wgpu::Device) -> wgpu::ComputePipeline {
        let field = PrimeField::<MODULUS>::new();
        let module = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("emvp-answer.wgsl"),
            source: wgpu::ShaderSource::Wgsl(ANSWER_WGSL.into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("emvp-answer-bind-group-layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("emvp-answer-pipeline-layout"),
            bind_group_layouts: &[Some(&layout)],
            immediate_size: 0,
        });
        // u32 constants pass through f64 exactly: both words are below 2^53.
        let constants = [
            ("MODULUS", f64::from(MODULUS)),
            ("NEG_INV", f64::from(field.montgomery_neg_inv())),
            ("R2", f64::from(field.montgomery_r2())),
        ];
        device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("emvp-answer-pipeline"),
            layout: Some(&pipeline_layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions {
                constants: &constants,
                zero_initialize_workgroup_memory: true,
            },
            cache: None,
        })
    }

    /// Rejects a buffer of `words` u32 words that exceeds the device's
    /// per-buffer capacity. The storage-binding limit is used for every
    /// buffer, which is conservative for the staging buffer.
    fn check_buffer_words(&self, words: u64) -> Result<(), GpuError> {
        let limits = self.device.limits();
        let max_words = limits
            .max_buffer_size
            .min(limits.max_storage_buffer_binding_size)
            / 4;
        if words > max_words {
            return Err(GpuError::UploadTooLarge {
                elements: usize::try_from(words).unwrap_or(usize::MAX),
                max_elements: usize::try_from(max_words).unwrap_or(usize::MAX),
            });
        }
        Ok(())
    }

    /// Runs the all-or-nothing validation for one answer batch and returns
    /// the kernel's dimensions.
    ///
    /// The per-query checks reuse
    /// [`crate::protocol::validate_query_against_matrix`], the same
    /// validation the CPU `answer_batch` runs, so CPU, dispatcher, and GPU
    /// reject the same malformed batches. The matrix shape itself was
    /// validated at upload time (the stored parameters and dimensions are
    /// pinned by [`GpuEncryptedMatrix`]); the batch-specific work here is
    /// the empty-batch rejection and the kernel's device and index bounds.
    fn answer_shape<const MODULUS: u32>(
        &self,
        matrix: &GpuEncryptedMatrix<MODULUS>,
        queries: &[EncryptedQuery<MODULUS>],
    ) -> Result<AnswerShape, GpuError> {
        let Some(_first) = queries.first() else {
            return Err(GpuError::LengthMismatch {
                name: "queries",
                expected: 1,
                actual: 0,
            });
        };
        let params = matrix.params();
        params.validate_dimensions()?;
        let n = matrix.columns();
        for query in queries {
            validate_query_against_matrix(n, matrix.instance_id(), query)?;
        }
        let b = params.block_size();
        let s = params.blocks()?;
        let rows = matrix.rows();
        let query_words = queries
            .len()
            .checked_mul(n)
            .ok_or(GpuError::DimensionOverflow)?;
        let answer_words_per_query = rows.checked_mul(s).ok_or(GpuError::DimensionOverflow)?;
        let answer_words = queries
            .len()
            .checked_mul(answer_words_per_query)
            .ok_or(GpuError::DimensionOverflow)?;
        // The u32 narrowings double as the kernel's u32 index bounds: query
        // indices stay below `batch * n` and answer indices below
        // `batch * rows * s`. The answer count additionally respects
        // [`MAX_ANSWER_WORDS`] so the kernel's reconstructed thread index
        // cannot wrap.
        let query_words_u32 =
            u32::try_from(query_words).map_err(|_conversion| GpuError::UploadTooLarge {
                elements: query_words,
                max_elements: u32::MAX as usize,
            })?;
        if answer_words > MAX_ANSWER_WORDS {
            return Err(GpuError::UploadTooLarge {
                elements: answer_words,
                max_elements: MAX_ANSWER_WORDS,
            });
        }
        let answer_words_u32 =
            u32::try_from(answer_words).map_err(|_conversion| GpuError::UploadTooLarge {
                elements: answer_words,
                max_elements: u32::MAX as usize,
            })?;
        self.check_buffer_words(u64::from(query_words_u32))?;
        self.check_buffer_words(u64::from(answer_words_u32))?;
        Ok(AnswerShape {
            n,
            b,
            s,
            rows,
            batch: queries.len(),
            query_words_u32,
            answer_words_u32,
            answer_words,
        })
    }

    /// Records the compute dispatch and the readback copy, submits them,
    /// and returns the submission index the readback wait blocks on.
    fn submit_answer_dispatch<const MODULUS: u32>(
        &self,
        matrix: &GpuEncryptedMatrix<MODULUS>,
        pipeline: &wgpu::ComputePipeline,
        buffers: (&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer),
        workgroups: (u32, u32),
        copy_bytes: u64,
    ) -> wgpu::SubmissionIndex {
        let (query_buffer, uniform_buffer, output_buffer, staging_buffer) = buffers;
        let (workgroups_x, workgroups_y) = workgroups;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("emvp-answer-bind-group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &matrix.buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: query_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: output_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: uniform_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("emvp-answer-encoder"),
            });
        {
            let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("emvp-answer-pass"),
                timestamp_writes: None,
            });
            compute_pass.set_pipeline(pipeline);
            compute_pass.set_bind_group(0, &bind_group, &[]);
            compute_pass.dispatch_workgroups(workgroups_x, workgroups_y, 1);
        }
        encoder.copy_buffer_to_buffer(output_buffer, 0, staging_buffer, 0, copy_bytes);
        self.queue.submit([encoder.finish()])
    }

    /// Maps `byte_len` of the staging buffer and blocks the calling thread
    /// until that mapping is valid, which for
    /// [`wgpu::PollType::wait_indefinitely`] means the device has drained
    /// the given submission: the staged uploads, the dispatch, and the
    /// device-to-host copy.
    pub(crate) fn wait_for_staged_slice(
        &self,
        staging_buffer: &wgpu::Buffer,
        byte_len: u64,
        submission: wgpu::SubmissionIndex,
    ) -> Result<(), GpuError> {
        let slice = staging_buffer.slice(0..byte_len);
        let (sender, receiver) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _delivery = sender.send(result.map_err(|error| error.to_string()));
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(submission),
                timeout: None,
            })
            .map_err(|error| GpuError::Submission(error.to_string()))?;
        receiver
            .recv()
            .map_err(|_dropped| GpuError::Map("mapping callback result unavailable".into()))?
            .map_err(GpuError::Map)
    }
}

/// The device-ready dispatch plan for one validated answer shape: the
/// modulus's compiled pipeline, the 2D workgroup grid the kernel
/// reconstructs its linear index from, and the encoded `Dims` uniform.
struct AnswerDispatch {
    pipeline: wgpu::ComputePipeline,
    workgroups: (u32, u32),
    uniform: [u8; DIMS_UNIFORM_WORDS * WORD_BYTES],
}

/// Kernel dimensions and buffer sizes for one validated answer batch.
struct AnswerShape {
    /// The codeword length `n = 2k`.
    n: usize,
    /// The block size `b`.
    b: usize,
    /// The block count `s = n / b`.
    s: usize,
    /// The matrix row count `m`.
    rows: usize,
    /// The query batch size `B`.
    batch: usize,
    /// `B * n` as a u32 word count for the query buffer size.
    query_words_u32: u32,
    /// `B * rows * s` as a u32 word count for the output buffer size.
    answer_words_u32: u32,
    /// `B * rows * s` in host arithmetic, for the readback allocation.
    answer_words: usize,
}

/// A reusable set of device buffers for one answer batch.
///
/// The set is leased from a [`GpuAnswerer`]'s pool, so consecutive batches
/// skip device allocation entirely. Growth contract: each buffer remembers
/// the largest word count its set has served; a call needing more replaces
/// the affected buffers and raises the record. Capacities never shrink, so
/// pooled memory stays at the peak a server actually uses. Every set owns
/// its uniform buffer because the `Dims` contents of an in-flight batch
/// must not be overwritten by a concurrent one.
#[derive(Debug)]
struct AnswerScratch {
    /// Query words (`batch * n`), `STORAGE | COPY_DST`.
    query_buffer: wgpu::Buffer,
    /// The `Dims` uniform; fixed [`DIMS_UNIFORM_BYTES`] size.
    uniform_buffer: wgpu::Buffer,
    /// Answer words (`batch * rows * s`), `STORAGE | COPY_SRC`.
    output_buffer: wgpu::Buffer,
    /// Readback copy of the answers, `MAP_READ | COPY_DST`.
    staging_buffer: wgpu::Buffer,
    query_capacity_words: u32,
    answer_capacity_words: u32,
}

impl AnswerScratch {
    /// Creates a set whose buffers start sized for `query_words` and
    /// `answer_words`; both capacities are valid device-side sizes because
    /// the shape validation ran first.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::OutOfMemory`] if the driver refuses any of the
    /// four buffer allocations.
    fn new(device: &wgpu::Device, query_words: u32, answer_words: u32) -> Result<Self, GpuError> {
        let query_buffer = create_buffer_checked(
            device,
            "answer query buffer",
            &answer_buffer_descriptor(
                "emvp-answer-queries",
                query_words,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
        )?;
        let uniform_buffer = create_buffer_checked(
            device,
            "answer uniform buffer",
            &wgpu::BufferDescriptor {
                label: Some("emvp-answer-dims"),
                size: DIMS_UNIFORM_BYTES,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            },
        )?;
        let output_buffer = create_buffer_checked(
            device,
            "answer output buffer",
            &answer_buffer_descriptor(
                "emvp-answer-output",
                answer_words,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
        )?;
        let staging_buffer = create_buffer_checked(
            device,
            "answer staging buffer",
            &answer_buffer_descriptor(
                "emvp-answer-staging",
                answer_words,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
        )?;
        Ok(Self {
            query_buffer,
            uniform_buffer,
            output_buffer,
            staging_buffer,
            query_capacity_words: query_words,
            answer_capacity_words: answer_words,
        })
    }

    /// Grows the reused buffers to cover `query_words` and `answer_words`
    /// when the request exceeds the set's recorded peaks, and returns the
    /// ready buffers. Steady-state calls take the no-growth fast path.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::OutOfMemory`] if the driver refuses a growth
    /// allocation. The set is left inconsistent in that case and must be
    /// dropped rather than returned to its pool.
    fn prepare(
        &mut self,
        device: &wgpu::Device,
        query_words: u32,
        answer_words: u32,
    ) -> Result<(&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer), GpuError> {
        if query_words > self.query_capacity_words {
            self.query_buffer = create_buffer_checked(
                device,
                "answer query buffer",
                &answer_buffer_descriptor(
                    "emvp-answer-queries",
                    query_words,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                ),
            )?;
            self.query_capacity_words = query_words;
        }
        if answer_words > self.answer_capacity_words {
            self.output_buffer = create_buffer_checked(
                device,
                "answer output buffer",
                &answer_buffer_descriptor(
                    "emvp-answer-output",
                    answer_words,
                    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                ),
            )?;
            self.staging_buffer = create_buffer_checked(
                device,
                "answer staging buffer",
                &answer_buffer_descriptor(
                    "emvp-answer-staging",
                    answer_words,
                    wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                ),
            )?;
            self.answer_capacity_words = answer_words;
        }
        Ok((
            &self.query_buffer,
            &self.uniform_buffer,
            &self.output_buffer,
            &self.staging_buffer,
        ))
    }
}

/// Locks `mutex`, recovering from poisoning: the guarded values (pipeline
/// cache, buffer pool) are immutable or self-consistent between calls, so a
/// panic in another thread mid-insert cannot have corrupted anything.
pub(crate) fn lock_recovered<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// Buffer descriptor for a reusable answer buffer of `words` words.
fn answer_buffer_descriptor(
    label: &'static str,
    words: u32,
    usage: wgpu::BufferUsages,
) -> wgpu::BufferDescriptor<'static> {
    wgpu::BufferDescriptor {
        label: Some(label),
        size: byte_len_of_words(words),
        usage,
        mapped_at_creation: false,
    }
}

/// Creates a buffer, surfacing a scoped device out-of-memory error as a
/// recoverable [`GpuError::OutOfMemory`].
///
/// `create_buffer` reports allocation failures through wgpu's error-scope
/// machinery rather than its return type. This helper brackets the call in
/// an [`wgpu::ErrorFilter::OutOfMemory`] scope and blocks on popping it. On
/// the native backends the underlying scope future is already resolved when
/// popped (buffer validation and its OOM reporting are synchronous), so the
/// blocking wait never parks meaningfully; error scopes are thread-local,
/// which every caller satisfies by creating and popping on one thread.
pub(crate) fn create_buffer_checked(
    device: &wgpu::Device,
    context: &'static str,
    descriptor: &wgpu::BufferDescriptor<'_>,
) -> Result<wgpu::Buffer, GpuError> {
    let scope = device.push_error_scope(wgpu::ErrorFilter::OutOfMemory);
    let buffer = device.create_buffer(descriptor);
    pollster::block_on(scope.pop()).map_or_else(
        || Ok(buffer),
        |_out_of_memory| Err(GpuError::OutOfMemory { context }),
    )
}

/// Opens a write-only staging view over the first `words` words of `buffer`
/// for a `Queue::write_buffer_with` upload.
///
/// The view writes straight into wgpu's staging memory: uploading costs one
/// pass over the data with no intermediate byte `Vec` and no separate
/// `write_buffer` staging copy. The staged transfer is flushed to the
/// device by the next `Queue::submit`.
///
/// # Errors
///
/// Returns [`GpuError::UploadTooLarge`] when the byte length does not fit
/// the staging API's u32 size, and [`GpuError::Submission`] when wgpu
/// refuses to allocate the staging memory.
fn staged_write_view(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    words: u32,
) -> Result<wgpu::QueueWriteBufferView, GpuError> {
    let size = wgpu::BufferSize::new(byte_len_of_words(words)).ok_or(GpuError::UploadTooLarge {
        elements: words as usize,
        max_elements: (u32::MAX as usize) / WORD_BYTES,
    })?;
    queue.write_buffer_with(buffer, 0, size).ok_or_else(|| {
        GpuError::Submission("the queue refused to allocate staging memory for an upload".into())
    })
}

/// Streams the matrix's raw Montgomery words into the staged upload view as
/// little-endian bytes.
///
/// `view` is exactly `values.len()` words of `WORD_BYTES` bytes by
/// construction, so the write fills it completely. This fuses the `to_raw`
/// word pass, the byte conversion, and the staging copy into a single pass
/// with no intermediate allocation; the element-wise walk is the safe
/// (workspace `unsafe`-denying) alternative to reinterpreting the slice and
/// is dominated by the `PCIe` transfer it feeds.
fn fill_matrix_view<const MODULUS: u32>(
    view: &mut wgpu::QueueWriteBufferView,
    values: &[FieldElement<MODULUS>],
) {
    let (word_bytes, _tail) = view.slice(..).into_chunks::<WORD_BYTES>();
    word_bytes.write_iter(values.iter().map(|element| element.to_raw().to_le_bytes()));
}

/// Streams the query batch's raw Montgomery words into the staged upload
/// view as little-endian bytes.
///
/// `view` is exactly `batch * n` words of `WORD_BYTES` bytes by
/// construction, so the flattened per-query words fill it completely and
/// `WriteOnly::write_iter`'s length check cannot trip. Same fused single
/// pass as [`fill_matrix_view`].
fn fill_query_view<const MODULUS: u32>(
    view: &mut wgpu::QueueWriteBufferView,
    queries: &[EncryptedQuery<MODULUS>],
) {
    let (word_bytes, _tail) = view.slice(..).into_chunks::<WORD_BYTES>();
    word_bytes.write_iter(
        queries
            .iter()
            .flat_map(|query| query.values().iter())
            .map(|element| element.to_raw().to_le_bytes()),
    );
}

/// Maps the staged answer bytes and streams them into `dest`, releasing
/// the mapping on every path.
///
/// The staging buffer holds the kernel's query-major arena for the batch;
/// `dest` must hold exactly the same word count, which the validated shape
/// fixes. On an error the mapping is still released, so a pooled scratch
/// set is always returned to the pool unmapped, and `dest`'s contents are
/// unspecified: the caller treats the whole batch as failed.
fn reconstruct_staged_into<const MODULUS: u32>(
    staging_buffer: &wgpu::Buffer,
    dest: &mut [FieldElement<MODULUS>],
    answer_words_u32: u32,
) -> Result<(), GpuError> {
    let byte_len = byte_len_of_words(answer_words_u32);
    let data = match staging_buffer.slice(0..byte_len).get_mapped_range() {
        Ok(data) => data,
        Err(error) => {
            staging_buffer.unmap();
            return Err(GpuError::Map(error.to_string()));
        }
    };
    let result = reconstruct_bytes_into::<MODULUS>(dest, &data);
    drop(data);
    staging_buffer.unmap();
    result
}

/// Streams little-endian staged answer words into `dest` as canonical
/// field elements, rejecting a corrupt word precisely.
///
/// Each word is wrapped with the checked
/// [`FieldElement::try_from_raw`](prime_field_layer::FieldElement::try_from_raw):
/// the kernel's final fold produces canonical words, so a word outside
/// `0..MODULUS` means a corrupt readback and is rejected as
/// [`GpuError::NonCanonicalWord`] with the offending word, never
/// canonicalized. The byte length is checked against `dest` before any
/// write, so a length mismatch leaves `dest` untouched.
///
/// This is the device-free core of the into-arena readback: one pass over
/// the mapped bytes, no intermediate allocation.
fn reconstruct_bytes_into<const MODULUS: u32>(
    dest: &mut [FieldElement<MODULUS>],
    bytes: &[u8],
) -> Result<(), GpuError> {
    let expected = dest
        .len()
        .checked_mul(WORD_BYTES)
        .ok_or(GpuError::DimensionOverflow)?;
    if bytes.len() != expected {
        return Err(GpuError::LengthMismatch {
            name: "staged answer bytes",
            expected,
            actual: bytes.len(),
        });
    }
    for (slot, word) in dest.iter_mut().zip(bytes.chunks_exact(WORD_BYTES)) {
        *slot = staged_answer_word::<MODULUS>(word)?;
    }
    Ok(())
}

/// Checks a destination arena can hold a batch's answers before any work
/// or mutation happens.
const fn check_arena_capacity(dest_words: usize, required: usize) -> Result<(), GpuError> {
    if dest_words >= required {
        Ok(())
    } else {
        Err(GpuError::Capacity {
            required,
            available: dest_words,
        })
    }
}

/// Maps the workspace-growth failure onto its GPU equivalent. Only
/// [`ProtocolError::Capacity`] is reachable from [`AnswerWorkspace::reserve_words`],
/// and it maps losslessly; the catch-all arm keeps the conversion total
/// and fail-closed.
fn map_capacity_error(error: ProtocolError) -> GpuError {
    match error {
        ProtocolError::Capacity {
            required,
            available,
        } => GpuError::Capacity {
            required,
            available,
        },
        other => GpuError::from(other),
    }
}

/// Splits a filled query-major arena prefix into one owned
/// [`AnswerMatrix`] per query — the temporary conversion the legacy
/// allocating adapters run over the into-API's output.
///
/// `shape` must be the shape the arena was validated and filled with, so
/// the arena length is checked against it only fail-closed. The split
/// allocates one `Vec` per query, which is exactly the allocation profile
/// the old pre-arena path had.
fn owned_answers<const MODULUS: u32>(
    arena: &[FieldElement<MODULUS>],
    queries: &[EncryptedQuery<MODULUS>],
    shape: &crate::answer::AnswerShape,
) -> Result<Vec<AnswerMatrix<MODULUS>>, GpuError> {
    let answer_words = shape.answer_words().ok_or(GpuError::DimensionOverflow)?;
    let expected = queries
        .len()
        .checked_mul(answer_words)
        .ok_or(GpuError::DimensionOverflow)?;
    if arena.len() != expected {
        return Err(GpuError::LengthMismatch {
            name: "answer arena",
            expected,
            actual: arena.len(),
        });
    }
    queries
        .iter()
        .zip(arena.chunks_exact(answer_words))
        .map(|(query, values)| {
            Ok(AnswerMatrix::from_parts(
                shape.instance_id(),
                query.query_id(),
                values.to_vec(),
                shape.rows(),
                shape.blocks(),
            ))
        })
        .collect()
}

/// Wraps one little-endian staged answer word as a canonical field
/// element, rejecting a corrupt word precisely.
pub(crate) fn staged_answer_word<const MODULUS: u32>(
    word: &[u8],
) -> Result<FieldElement<MODULUS>, GpuError> {
    let raw = u32::from_le_bytes([word[0], word[1], word[2], word[3]]);
    FieldElement::<MODULUS>::try_from_raw(raw).map_err(|_field_error| GpuError::NonCanonicalWord {
        word: raw,
        modulus: MODULUS,
    })
}

/// Folds the linear output count into a 2D workgroup dispatch that stays
/// within the wgpu-guaranteed 65535 workgroups per dimension. The kernel
/// reconstructs the linear index from the returned `workgroups_x`, which the
/// uniform carries.
///
/// Callers must have capped `answer_words` at [`MAX_ANSWER_WORDS`], which
/// keeps every issued thread index (including workgroup padding) inside
/// `u32` and `workgroups_y` near its lower bound.
pub(crate) fn dispatch_grid(answer_words: usize) -> Result<(u32, u32), GpuError> {
    let workgroup_total = answer_words.div_ceil(WORKGROUP_SIZE_USIZE);
    let workgroups_x = u32::try_from(workgroup_total)
        .map_err(|_conversion| GpuError::UploadTooLarge {
            elements: answer_words,
            max_elements: u32::MAX as usize,
        })?
        .min(MAX_WORKGROUPS_PER_DIMENSION);
    let workgroups_y = u32::try_from(workgroup_total.div_ceil(workgroups_x as usize))
        .map_err(|_conversion| GpuError::DimensionOverflow)?;
    Ok((workgroups_x, workgroups_y))
}

/// Encodes the `Dims` uniform: `n`, `b`, `s`, `rows`, `batch`,
/// `workgroups_x`, then two padding words.
///
/// Callers must have validated the usize dimensions against the kernel's
/// u32 index bounds before calling; the narrowings here fail closed with
/// [`GpuError::DimensionOverflow`].
pub(crate) fn dims_uniform_bytes(
    n: usize,
    b: usize,
    s: usize,
    rows: usize,
    batch: usize,
    workgroups_x: u32,
) -> Result<[u8; DIMS_UNIFORM_WORDS * WORD_BYTES], GpuError> {
    let mut bytes = [0_u8; DIMS_UNIFORM_WORDS * WORD_BYTES];
    for (slot, dimension) in
        bytes
            .chunks_exact_mut(WORD_BYTES)
            .zip([n, b, s, rows, batch, workgroups_x as usize])
    {
        let word = u32::try_from(dimension).map_err(|_conversion| GpuError::DimensionOverflow)?;
        slot.copy_from_slice(&word.to_le_bytes());
    }
    Ok(bytes)
}

/// Rejects moduli the WGSL arithmetic cannot support: the REDC bound
/// arguments require `p < 2^31`, and Montgomery form requires `p > 2`.
const fn check_modulus<const MODULUS: u32>() -> Result<(), GpuError> {
    if MODULUS > 2 && MODULUS < 1_u32 << 31 {
        Ok(())
    } else {
        Err(GpuError::UnsupportedModulus { modulus: MODULUS })
    }
}

/// Byte length of a buffer holding `words` u32 words.
pub(crate) fn byte_len_of_words(words: u32) -> wgpu::BufferAddress {
    u64::from(words) * 4
}

#[cfg(test)]
mod tests {
    use super::{
        ANSWER_WGSL, MAX_ANSWER_WORDS, MAX_WORKGROUPS_PER_DIMENSION, WORKGROUP_SIZE_USIZE,
        check_arena_capacity, dispatch_grid, owned_answers, reconstruct_bytes_into,
    };
    use crate::answer::AnswerShape;
    use crate::params::EmvpParams;
    use crate::protocol::{EncryptedMatrix, EncryptedQuery};
    use prime_field_layer::{FieldElement, PrimeField};

    // The deployment field: p = 998244353 is prime and below 2^30. The
    // host-reconstruction tests need no adapter, only the field type.
    const MODULUS: u32 = 998_244_353;

    /// The answer shader must parse as WGSL and pass naga's full validation
    /// without any adapter: this pins the embedded kernel's well-formedness
    /// on every machine, so only its execution needs hardware.
    #[test]
    fn answer_wgsl_parses_and_validates_device_independently() {
        let module =
            naga::front::wgsl::parse_str(ANSWER_WGSL).expect("the answer shader must parse");
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::all(),
        )
        .validate(&module)
        .expect("the answer shader must validate");
    }

    /// The validated shape of the two-query host-reconstruction fixture:
    /// n = 2 (k = 1), b = 2, s = 1, two rows, two queries. The host shape
    /// is the plan/execute API's shape type, whose arena holds exactly
    /// `queries * rows * s = 4` words.
    fn host_shape() -> AnswerShape {
        AnswerShape::new(2, 1, 2, 7)
    }

    /// The canonical Montgomery words of the sums 17, 39, 23, 53: each is
    /// `value * 2^32 mod p`, derived by hand arithmetic, independently of
    /// both the kernel and the CPU answer path.
    const HOST_GOLDEN_WORDS: [u32; 4] = [
        142_606_263, // 17 * 301989884 mod 998244353
        796_917_593, // 39 * 301989884 mod 998244353
        956_301_214, // 23 * 301989884 mod 998244353
        33_554_204,  // 53 * 301989884 mod 998244353
    ];

    fn host_queries() -> Vec<EncryptedQuery<MODULUS>> {
        let field = PrimeField::<MODULUS>::new();
        vec![
            EncryptedQuery::from_parts(7, 10, vec![field.element_u32(0); 2]),
            EncryptedQuery::from_parts(7, 11, vec![field.element_u32(0); 2]),
        ]
    }

    fn host_bytes(words: &[u32]) -> Vec<u8> {
        words.iter().flat_map(|word| word.to_le_bytes()).collect()
    }

    /// A destination arena poisoned with a sentinel element (the field
    /// value `p - 1`, distinct from every fixture answer), so a test can
    /// observe exactly which slots a rejected write touched.
    fn poison_sentinel() -> FieldElement<MODULUS> {
        PrimeField::<MODULUS>::new().element_u32(MODULUS - 1)
    }

    fn poisoned_dest() -> Vec<FieldElement<MODULUS>> {
        vec![poison_sentinel(); HOST_GOLDEN_WORDS.len()]
    }

    /// Canonical device words must stream into the caller's arena as
    /// exactly the elements they are the Montgomery images of, in the
    /// kernel's query-major order, leaving no allocation behind.
    #[test]
    fn reconstruct_into_arena_accepts_canonical_device_words() {
        let field = PrimeField::<MODULUS>::new();
        let mut dest = poisoned_dest();
        reconstruct_bytes_into::<MODULUS>(&mut dest, &host_bytes(&HOST_GOLDEN_WORDS))
            .expect("canonical words must reconstruct");
        let expected = [17_u32, 39, 23, 53]
            .iter()
            .map(|&sum| field.element_u32(sum));
        for (slot, expected_element) in dest.iter().zip(expected) {
            assert_eq!(slot, &expected_element);
        }
        // The owned split the legacy adapter performs over the filled arena
        // pairs the same elements with the batch's public identifiers.
        let queries = host_queries();
        let answers =
            owned_answers(&dest, &queries, &host_shape()).expect("the fixture shape fits");
        assert_eq!(answers.len(), 2);
        let expected_per_query: Vec<Vec<FieldElement<MODULUS>>> = [[17_u32, 39], [23, 53]]
            .iter()
            .map(|sums| sums.iter().map(|&sum| field.element_u32(sum)).collect())
            .collect();
        for (answer, expected_values) in answers.iter().zip(&expected_per_query) {
            assert_eq!(answer.values(), expected_values.as_slice());
            assert_eq!(answer.rows(), 2);
            assert_eq!(answer.blocks(), 1);
            assert_eq!(answer.instance_id(), 7);
        }
        assert_eq!(answers[0].query_id(), 10);
        assert_eq!(answers[1].query_id(), 11);
    }

    /// A device word at or above the modulus is not a canonical Montgomery
    /// residue: the into-write must reject it precisely, carrying the
    /// offending word, instead of silently canonicalizing it.
    #[test]
    fn reconstruct_into_arena_rejects_noncanonical_device_words() {
        // The third word is the smallest noncanonical residue, the modulus
        // itself; the first two were already streamed when the rejection
        // fires, so the caller treats the whole arena as failed.
        let mut words = HOST_GOLDEN_WORDS.to_vec();
        words[2] = MODULUS;
        let mut dest = poisoned_dest();
        let error = reconstruct_bytes_into::<MODULUS>(&mut dest, &host_bytes(&words)).unwrap_err();
        assert_eq!(
            error,
            super::GpuError::NonCanonicalWord {
                word: MODULUS,
                modulus: MODULUS,
            }
        );

        // The maximum u32 word is rejected the same way.
        let mut words = HOST_GOLDEN_WORDS.to_vec();
        words[3] = u32::MAX;
        assert!(matches!(
            reconstruct_bytes_into::<MODULUS>(&mut poisoned_dest(), &host_bytes(&words)),
            Err(super::GpuError::NonCanonicalWord {
                word: u32::MAX,
                modulus: MODULUS,
            })
        ));

        // A byte buffer shorter than the destination is rejected
        // fail-closed before any word is wrapped, leaving the destination
        // poisoned untouched.
        let bytes = host_bytes(&HOST_GOLDEN_WORDS);
        let mut dest = poisoned_dest();
        assert!(matches!(
            reconstruct_bytes_into::<MODULUS>(&mut dest, &bytes[..bytes.len() - 1]),
            Err(super::GpuError::LengthMismatch {
                name: "staged answer bytes",
                ..
            })
        ));
        assert!(dest.iter().all(|slot| *slot == poison_sentinel()));
    }

    /// An insufficient arena capacity must be rejected before any device
    /// work and before any arena mutation: the rejection carries both
    /// counts and leaves the caller's words untouched.
    #[test]
    fn insufficient_capacity_is_rejected_before_any_mutation() {
        // No `mut`: the rejection happens before any mutation could, which
        // is exactly the property under test.
        let dest = poisoned_dest();
        assert_eq!(
            check_arena_capacity(dest.len(), HOST_GOLDEN_WORDS.len() + 1),
            Err(super::GpuError::Capacity {
                required: HOST_GOLDEN_WORDS.len() + 1,
                available: HOST_GOLDEN_WORDS.len(),
            })
        );
        // Exactly enough capacity is accepted; the caller proceeds.
        assert_eq!(
            check_arena_capacity(dest.len(), HOST_GOLDEN_WORDS.len()),
            Ok(())
        );
        assert!(dest.iter().all(|slot| *slot == poison_sentinel()));
    }

    /// The shared matrix-shape validation the GPU upload runs (and the CPU
    /// answer path and dispatcher reuse) rejects mismatched shapes before
    /// any device work could be reached.
    #[test]
    fn upload_shape_validation_matches_the_cpu_path() {
        use crate::protocol::validate_matrix_shape;
        let field = PrimeField::<MODULUS>::new();
        let params = EmvpParams {
            k: 1,
            ell: 1,
            b: 2,
            lambda: 7,
        };
        let matrix = EncryptedMatrix::from_parts(1, 2, 2, vec![field.element_u32(0); 4]).unwrap();
        // Matching dimensions: the shared validator accepts and returns the
        // codeword length n = 2.
        assert_eq!(validate_matrix_shape(&params, &matrix), Ok(2));
        let mismatched = EmvpParams {
            k: 2,
            ell: 1,
            b: 2,
            lambda: 7,
        };
        assert!(matches!(
            validate_matrix_shape(&mismatched, &matrix),
            Err(crate::protocol::ProtocolError::LengthMismatch {
                name: "encrypted matrix columns",
                ..
            })
        ));
    }

    /// The largest issued thread index over the whole dispatch must stay
    /// inside `u32`, or the shader's wrapping index arithmetic would let
    /// padded threads overwrite real output slots.
    #[test]
    fn dispatch_indices_never_wrap_u32() {
        for answer_words in [1_usize, 256, 257, MAX_ANSWER_WORDS - 1, MAX_ANSWER_WORDS] {
            let (workgroups_x, workgroups_y) = dispatch_grid(answer_words).unwrap();
            let issued_threads =
                u64::from(workgroups_x) * u64::from(workgroups_y) * WORKGROUP_SIZE_USIZE as u64;
            assert!(issued_threads <= u64::from(u32::MAX) + 1);
            assert!(workgroups_x <= MAX_WORKGROUPS_PER_DIMENSION);
            assert!(workgroups_y <= MAX_WORKGROUPS_PER_DIMENSION);
            assert!(issued_threads >= answer_words as u64);
        }
    }

    /// The cap's defining inequality must hold at the extreme: the padded
    /// thread count for a maximal batch stays within `answer_words +
    /// MAX_WORKGROUPS_PER_DIMENSION * WORKGROUP_SIZE <= 2^32`, which is
    /// exactly what keeps the kernel's reconstructed u32 indices from
    /// wrapping.
    #[test]
    fn cap_bound_holds_at_the_extreme() {
        for answer_words in [1_usize, 256, 257, MAX_ANSWER_WORDS / 2, MAX_ANSWER_WORDS] {
            let (workgroups_x, workgroups_y) = dispatch_grid(answer_words).unwrap();
            let issued_threads =
                u64::from(workgroups_x) * u64::from(workgroups_y) * WORKGROUP_SIZE_USIZE as u64;
            let padding_allowance =
                u64::from(MAX_WORKGROUPS_PER_DIMENSION) * WORKGROUP_SIZE_USIZE as u64;
            assert!(issued_threads <= answer_words as u64 + padding_allowance);
            assert!(issued_threads <= u64::from(u32::MAX) + 1);
        }
    }
}
