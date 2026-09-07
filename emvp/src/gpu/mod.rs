//! Server-side GPU answer path for the EMVP protocol (wgpu + WGSL).
//!
//! The server's answer phase dominates the protocol's online cost: it
//! performs `queries * rows * n` field multiplications per batch against
//! data that is entirely public (the encrypted matrix `M_hat`, the encrypted
//! queries `q_hat`, and the answers `M'`). This module moves that phase onto
//! a compute device through [`wgpu`] while the client stays on the CPU. The
//! trust boundary is unchanged: every word the GPU reads or writes is public
//! protocol data, no secret key material ever reaches the device, and
//! constant-time discipline is unnecessary on this path.
//!
//! # Montgomery-native interchange
//!
//! [`prime_field_layer::FieldElement`] stores each value as its canonical
//! Montgomery residue `a * 2^32 mod p`. The WGSL kernel in
//! [`ANSWER_WGSL`] reproduces the crate's Montgomery multiplication (REDC)
//! on those raw words, so uploads and readbacks move the words as-is:
//! [`GpuAnswerer::upload_matrix`] copies the raw words out of the encrypted
//! matrix, and [`GpuAnswerer::answer_batch`] wraps the returned words with
//! [`prime_field_layer::FieldElement::from_raw`]. The results are
//! bit-identical to the CPU [`answer_batch`](crate::answer_batch), which
//! remains the reference implementation; the parity tests in
//! `emvp/tests/gpu.rs` pin this property empirically. Because REDC of
//! canonical residues yields canonical residues, and field addition of
//! canonical residues is canonical, equality on raw words matches equality
//! on elements and no normalization pass is needed.
//!
//! # Pipelines and constants
//!
//! One compute pipeline exists per field modulus, cached in
//! [`GpuAnswerer`]. The shader receives the modulus `p` and the REDC
//! constant `-p^{-1} mod 2^32` (from
//! [`PrimeField::montgomery_neg_inv`]) as pipeline-overridable constants
//! compiled into the kernel; `R2 = 2^64 mod p` is deliberately not passed
//! because these buffers already hold Montgomery residues and R2 only
//! matters when entering Montgomery form, which the CPU did when the
//! elements were built. Per-call dimensions (`n`, `b`, `s`, rows, batch,
//! dispatch width) travel in a small uniform buffer so changing protocol
//! parameters never recompiles the pipeline.
//!
//! # Data flow and resource ownership
//!
//! [`upload_matrix`] is the explicit upload-once step: it copies the
//! row-major encrypted matrix into a device-side storage buffer held by the
//! returned [`GpuEncryptedMatrix`]. [`answer_batch`] then uploads the
//! query batch, dispatches one thread per output element, and reads the
//! answers back through a staging buffer. The upload path converts
//! [`FieldElement`] words with a safe element-wise copy (the workspace
//! denies `unsafe` and the copy is dwarfed by the `PCIe` transfer); the same
//! applies to the little-endian byte staging, because all wgpu-supported
//! hosts are little-endian.
//!
//! The `answer_*_sync` wrappers block the calling thread with
//! [`pollster`]; the async methods exist so servers can integrate with
//! async executors, but note that the device wait inside
//! [`GpuAnswerer::answer_batch`] is itself a blocking poll, so latency-
//! sensitive executors should run the future on a blocking thread.
//!
//! # Errors
//!
//! Every failure mode surfaces as a [`GpuError`] rather than a
//! [`crate::ProtocolError`]: the GPU path is server plumbing, servers use
//! [`GpuAnswerer`] directly, and folding it into the protocol error type
//! would couple client-side decoding to the server's hardware.
//!
//! # Experimental
//!
//! This is experimental cryptography running on an unaudited kernel; the
//! WGSL is compiled by naga at runtime, so shader failures surface as
//! device errors rather than compile-time failures.

use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;

use prime_field_layer::{FieldElement, PrimeField};

use crate::params::EmvpParams;
use crate::protocol::{AnswerMatrix, EncryptedMatrix, EncryptedQuery};

/// The answer compute kernel, compiled once per modulus.
const ANSWER_WGSL: &str = include_str!("answer.wgsl");

/// Threads per workgroup; must match the `WORKGROUP_SIZE` const in
/// [`ANSWER_WGSL`].
const WORKGROUP_SIZE: u32 = 256;

/// [`WORKGROUP_SIZE`] as a host word count.
const WORKGROUP_SIZE_USIZE: usize = WORKGROUP_SIZE as usize;

/// Words in the `Dims` uniform: six used words plus two padding words.
const DIMS_UNIFORM_WORDS: usize = 8;

/// Largest dispatchable workgroup count per dimension guaranteed by wgpu.
const MAX_WORKGROUPS_PER_DIMENSION: u32 = 65_535;

/// A rejected GPU operation.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum GpuError {
    /// No compute adapter matched the request. The machine may have no GPU,
    /// no Vulkan-compatible driver, or the loader may be missing; callers
    /// (tests, benchmarks) treat this variant as a graceful skip.
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
    /// A slice did not have the required length.
    LengthMismatch {
        /// The rejected slice.
        name: &'static str,
        /// The required length.
        expected: usize,
        /// The observed length.
        actual: usize,
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
            Self::LengthMismatch {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "{name} length mismatch: expected {expected}, got {actual}"
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
        }
    }
}

impl std::error::Error for GpuError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Params(error) => Some(error),
            _ => None,
        }
    }
}

impl From<crate::params::ParamsError> for GpuError {
    fn from(error: crate::params::ParamsError) -> Self {
        Self::Params(error)
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
/// per-modulus pipeline cache.
pub struct GpuAnswerer {
    device: wgpu::Device,
    queue: wgpu::Queue,
    pipelines: Mutex<HashMap<u32, wgpu::ComputePipeline>>,
}

impl GpuAnswerer {
    /// Initializes a compute device for the answer phase.
    ///
    /// The high-performance adapter is requested; on machines with a single
    /// discrete GPU that selects it. Buffer limits are raised to whatever
    /// the adapter supports, because production encrypted matrices reach
    /// gibibytes while wgpu's default limits cap buffers at 256 MiB.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::NoAdapter`] when no Vulkan/GL adapter is
    /// available (the container case), and [`GpuError::RequestDevice`] when
    /// the adapter rejects the device request.
    pub async fn new() -> Result<Self, GpuError> {
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
        })
    }

    /// Uploads an encrypted matrix as raw Montgomery words.
    ///
    /// This is the one-time transfer per matrix: the returned
    /// [`GpuEncryptedMatrix`] owns the device buffer and every later
    /// [`Self::answer_batch`] against it reuses the words in place. The
    /// upload is a single `write_buffer` of the row-major words. The
    /// parameters must match the encryption parameters of `matrix`; they
    /// fix the block structure the kernel answers with.
    ///
    /// # Errors
    ///
    /// Returns an error before any transfer if the parameters are malformed,
    /// the parameter dimensions disagree with the matrix, the matrix is
    /// empty, its value length differs from `rows * columns`, the element
    /// count exceeds the device's per-buffer capacity or the kernel's u32
    /// index bound.
    #[expect(
        clippy::unused_async,
        reason = "the async surface stays uniform across the answerer API so callers integrate it with executors uniformly"
    )]
    pub async fn upload_matrix<const MODULUS: u32>(
        &self,
        params: &EmvpParams,
        matrix: &EncryptedMatrix<MODULUS>,
    ) -> Result<GpuEncryptedMatrix<MODULUS>, GpuError> {
        check_modulus::<MODULUS>()?;
        params.validate_dimensions()?;
        let n = params.n()?;
        check_len("encrypted matrix columns", n, matrix.columns())?;
        let rows = matrix.rows();
        check_len("matrix rows", 1, rows)?;
        let words = rows.checked_mul(n).ok_or(GpuError::DimensionOverflow)?;
        // The u32 narrowing doubles as the kernel's u32 index bound: every
        // matrix index is below `rows * n = words`.
        let words_u32 = u32::try_from(words).map_err(|_conversion| GpuError::UploadTooLarge {
            elements: words,
            max_elements: u32::MAX as usize,
        })?;
        self.check_buffer_words(u64::from(words_u32))?;

        let field = PrimeField::<MODULUS>::new();
        let mut words_buffer = vec![0_u32; words];
        field
            .write_raw_words(matrix.values(), &mut words_buffer)
            .map_err(|_field_error| GpuError::LengthMismatch {
                name: "encrypted matrix values",
                expected: words,
                actual: matrix.values().len(),
            })?;
        let bytes = words_to_le_bytes(&words_buffer);
        drop(words_buffer);

        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("emvp-encrypted-matrix"),
            size: byte_len_of_words(words_u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&buffer, 0, &bytes);

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
    /// output `(query, row, block)` the kernel accumulates `b` Montgomery
    /// products with field additions, writing one canonical word per element
    /// of the query-major answer arena. Validation is all-or-nothing and
    /// mirrors the CPU path; results are bit-identical to it.
    ///
    /// # Errors
    ///
    /// Returns an error before any device work if the parameters are
    /// malformed, the batch is empty, any query has the wrong length or a
    /// foreign instance identifier, an index or size would overflow, or the
    /// query/output buffers exceed the device's capacity.
    #[expect(
        clippy::unused_async,
        reason = "the async surface stays uniform across the answerer API so callers integrate it with executors uniformly"
    )]
    pub async fn answer_batch<const MODULUS: u32>(
        &self,
        matrix: &GpuEncryptedMatrix<MODULUS>,
        queries: &[EncryptedQuery<MODULUS>],
    ) -> Result<Vec<AnswerMatrix<MODULUS>>, GpuError> {
        check_modulus::<MODULUS>()?;
        let shape = self.answer_shape(matrix, queries)?;
        let query_bytes = encode_query_words::<MODULUS>(shape.n, queries)?;
        let (workgroups_x, workgroups_y) = dispatch_grid(shape.answer_words)?;
        let uniform_bytes = dims_uniform_bytes(
            shape.n,
            shape.b,
            shape.s,
            shape.rows,
            shape.batch,
            workgroups_x,
        )?;
        let pipeline = self.answer_pipeline::<MODULUS>();
        let (query_buffer, uniform_buffer, output_buffer, staging_buffer) = self
            .create_answer_buffers(
                &query_bytes,
                shape.query_words_u32,
                shape.answer_words_u32,
                &uniform_bytes,
            );
        self.submit_answer_dispatch::<MODULUS>(
            matrix,
            &pipeline,
            (
                &query_buffer,
                &uniform_buffer,
                &output_buffer,
                &staging_buffer,
            ),
            (workgroups_x, workgroups_y),
        );
        let answer_values =
            self.read_answer_values::<MODULUS>(&staging_buffer, shape.answer_words)?;
        Ok(answer_values
            .chunks_exact(shape.answer_words_per_query)
            .zip(queries)
            .map(|(values, query)| {
                AnswerMatrix::from_parts(
                    matrix.instance_id(),
                    query.query_id(),
                    values.to_vec(),
                    shape.rows,
                    shape.s,
                )
            })
            .collect())
    }

    /// Blocking single-threaded wrapper around [`Self::new`].
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::new`].
    pub fn new_sync() -> Result<Self, GpuError> {
        pollster::block_on(Self::new())
    }

    /// Blocking wrapper around [`Self::upload_matrix`].
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::upload_matrix`].
    pub fn upload_matrix_sync<const MODULUS: u32>(
        &self,
        params: &EmvpParams,
        matrix: &EncryptedMatrix<MODULUS>,
    ) -> Result<GpuEncryptedMatrix<MODULUS>, GpuError> {
        pollster::block_on(self.upload_matrix(params, matrix))
    }

    /// Blocking wrapper around [`Self::answer_batch`].
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::answer_batch`].
    pub fn answer_batch_sync<const MODULUS: u32>(
        &self,
        matrix: &GpuEncryptedMatrix<MODULUS>,
        queries: &[EncryptedQuery<MODULUS>],
    ) -> Result<Vec<AnswerMatrix<MODULUS>>, GpuError> {
        pollster::block_on(self.answer_batch(matrix, queries))
    }

    /// Returns the cached answer pipeline for this modulus, compiling it on
    /// first use.
    fn answer_pipeline<const MODULUS: u32>(&self) -> wgpu::ComputePipeline {
        // Pipelines are immutable once built, so a panic in another thread
        // mid-insert cannot have corrupted anything: recover the guard. The
        // guard is dropped before compilation so concurrent batches are not
        // blocked behind shader work.
        let cached = {
            let pipelines = match self.pipelines.lock() {
                Ok(pipelines) => pipelines,
                Err(poisoned) => poisoned.into_inner(),
            };
            pipelines.get(&MODULUS).cloned()
        };
        if let Some(pipeline) = cached {
            return pipeline;
        }
        let pipeline = Self::build_pipeline::<MODULUS>(&self.device);
        {
            let mut pipelines = match self.pipelines.lock() {
                Ok(pipelines) => pipelines,
                Err(poisoned) => poisoned.into_inner(),
            };
            pipelines.insert(MODULUS, pipeline.clone());
        }
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

    /// Runs the all-or-nothing validation for one answer batch, mirroring
    /// the CPU `answer_batch` checks, and returns the kernel's dimensions.
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
        let s = params.blocks()?;
        let rows = matrix.rows();
        for query in queries {
            check_len("encrypted query", n, query.values().len())?;
            if query.instance_id() != matrix.instance_id() {
                return Err(GpuError::InstanceMismatch {
                    expected: matrix.instance_id(),
                    actual: query.instance_id(),
                });
            }
        }
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
        // `batch * rows * s`.
        let query_words_u32 =
            u32::try_from(query_words).map_err(|_conversion| GpuError::UploadTooLarge {
                elements: query_words,
                max_elements: u32::MAX as usize,
            })?;
        let answer_words_u32 =
            u32::try_from(answer_words).map_err(|_conversion| GpuError::UploadTooLarge {
                elements: answer_words,
                max_elements: u32::MAX as usize,
            })?;
        self.check_buffer_words(u64::from(query_words_u32))?;
        self.check_buffer_words(u64::from(answer_words_u32))?;
        Ok(AnswerShape {
            n,
            b: params.block_size(),
            s,
            rows,
            batch: queries.len(),
            query_words_u32,
            answer_words_u32,
            answer_words,
            answer_words_per_query,
        })
    }

    /// Creates the per-call buffers and fills the query and uniform ones.
    fn create_answer_buffers(
        &self,
        query_bytes: &[u8],
        query_words_u32: u32,
        answer_words_u32: u32,
        uniform_bytes: &[u8],
    ) -> (wgpu::Buffer, wgpu::Buffer, wgpu::Buffer, wgpu::Buffer) {
        let query_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("emvp-answer-queries"),
            size: byte_len_of_words(query_words_u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&query_buffer, 0, query_bytes);
        let uniform_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("emvp-answer-dims"),
            size: uniform_bytes.len() as wgpu::BufferAddress,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        self.queue.write_buffer(&uniform_buffer, 0, uniform_bytes);
        let output_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("emvp-answer-output"),
            size: byte_len_of_words(answer_words_u32),
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let staging_buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("emvp-answer-staging"),
            size: output_buffer.size(),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        (query_buffer, uniform_buffer, output_buffer, staging_buffer)
    }

    /// Records the compute dispatch and the readback copy and submits them.
    fn submit_answer_dispatch<const MODULUS: u32>(
        &self,
        matrix: &GpuEncryptedMatrix<MODULUS>,
        pipeline: &wgpu::ComputePipeline,
        buffers: (&wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer, &wgpu::Buffer),
        workgroups: (u32, u32),
    ) {
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
        encoder.copy_buffer_to_buffer(output_buffer, 0, staging_buffer, 0, staging_buffer.size());
        self.queue.submit([encoder.finish()]);
    }

    /// Blocks until the staging buffer holds the answers and converts the
    /// little-endian words into canonical Montgomery field elements.
    fn read_answer_values<const MODULUS: u32>(
        &self,
        staging_buffer: &wgpu::Buffer,
        words: usize,
    ) -> Result<Vec<FieldElement<MODULUS>>, GpuError> {
        // Block until the submission completes and the mapping callback has
        // run; the wgpu polling model makes the wait synchronous by design.
        let (sender, receiver) = std::sync::mpsc::channel();
        staging_buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _delivery = sender.send(result.map_err(|error| error.to_string()));
            });
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .map_err(|error| GpuError::Submission(error.to_string()))?;
        receiver
            .recv()
            .map_err(|_dropped| GpuError::Map("mapping callback result unavailable".into()))?
            .map_err(GpuError::Map)?;
        let data = staging_buffer
            .slice(..)
            .get_mapped_range()
            .map_err(|error| GpuError::Map(error.to_string()))?;
        let mut answer_values = Vec::with_capacity(words);
        for chunk in data.chunks_exact(4) {
            let mut word = [0_u8; 4];
            word.copy_from_slice(chunk);
            answer_values.push(FieldElement::from_raw(u32::from_le_bytes(word)));
        }
        drop(data);
        staging_buffer.unmap();
        Ok(answer_values)
    }
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
    /// `rows * s`, the per-query answer slice length.
    answer_words_per_query: usize,
}

/// Copies `values` into a little-endian byte buffer for `write_buffer`.
///
/// All wgpu-supported hosts are little-endian, so this is the on-the-wire
/// encoding of the raw words. The element-wise copy is the safe (workspace
/// `unsafe`-denying) alternative to reinterpreting the slice and is
/// negligible next to the transfers it feeds.
fn words_to_le_bytes(words: &[u32]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(words.len() * 4);
    for word in words {
        bytes.extend_from_slice(&word.to_le_bytes());
    }
    bytes
}

/// Encodes a query batch into little-endian raw Montgomery words.
///
/// Validation has already fixed every query at length `n`, so a
/// [`FieldError::LengthMismatch`] here is unreachable; it is mapped into the
/// matching [`GpuError`] anyway to stay fail-closed.
fn encode_query_words<const MODULUS: u32>(
    n: usize,
    queries: &[EncryptedQuery<MODULUS>],
) -> Result<Vec<u8>, GpuError> {
    let word_count = queries
        .len()
        .checked_mul(n)
        .ok_or(GpuError::DimensionOverflow)?;
    let field = PrimeField::<MODULUS>::new();
    let mut query_word_buffer = vec![0_u32; word_count];
    for (slot, query) in query_word_buffer.chunks_mut(n).zip(queries) {
        field
            .write_raw_words(query.values(), slot)
            .map_err(|_field_error| GpuError::LengthMismatch {
                name: "encrypted query",
                expected: n,
                actual: query.values().len(),
            })?;
    }
    Ok(words_to_le_bytes(&query_word_buffer))
}

/// Folds the linear output count into a 2D workgroup dispatch that stays
/// within the wgpu-guaranteed 65535 workgroups per dimension. The kernel
/// reconstructs the linear index from the returned `workgroups_x`, which the
/// uniform carries.
fn dispatch_grid(answer_words: usize) -> Result<(u32, u32), GpuError> {
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
fn dims_uniform_bytes(
    n: usize,
    b: usize,
    s: usize,
    rows: usize,
    batch: usize,
    workgroups_x: u32,
) -> Result<[u8; DIMS_UNIFORM_WORDS * 4], GpuError> {
    let mut bytes = [0_u8; DIMS_UNIFORM_WORDS * 4];
    for (slot, dimension) in
        bytes
            .chunks_exact_mut(4)
            .zip([n, b, s, rows, batch, workgroups_x as usize])
    {
        let word = u32::try_from(dimension).map_err(|_conversion| GpuError::DimensionOverflow)?;
        slot.copy_from_slice(&word.to_le_bytes());
    }
    Ok(bytes)
}

/// Copies a length check into the GPU module, mirroring
/// `crate::protocol::check_len`.
const fn check_len(name: &'static str, expected: usize, actual: usize) -> Result<(), GpuError> {
    if expected == actual {
        Ok(())
    } else {
        Err(GpuError::LengthMismatch {
            name,
            expected,
            actual,
        })
    }
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
fn byte_len_of_words(words: u32) -> wgpu::BufferAddress {
    u64::from(words) * 4
}
