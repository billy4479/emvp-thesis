#![expect(
    clippy::unwrap_used,
    reason = "benchmarks use fixed valid parameters and keep setup beside measurement"
)]
#![expect(
    clippy::unnecessary_literal_unwrap,
    reason = "the adapter probe wraps an unexpected error in `Err` so the single `unwrap` path fails the bench loudly; static analysis flags the literal even though the error is dynamic"
)]

//! GPU answer-phase throughput: the EMVP server answer against cleartext
//! baselines on the same device.
//!
//! # Cases
//!
//! For `rows in {4096, 16384}` and `batch in {1, 8, 64, 256}` at
//! `ell = 4096` (the 7B-class record length), under the
//! `gpu_answer_cost_v1` group:
//!
//! - `emvp/batchB-rowsR`: one [`GpuAnswerer::execute_answer_batch_into`]
//!   call per iteration — the server answer phase alone against the
//!   device-resident encrypted matrix, covering query upload, dispatch,
//!   readback, and host reconstruction. The batch's queries are generated
//!   once per case outside timing, and the workspace is planned and
//!   reserved once per case, so steady-state iterations follow the
//!   library's plan-reserve-execute contract. This is the number a
//!   deployment pays per encrypted batch on the server; the client-side
//!   query and decode costs are the CPU suites' subject (`online`,
//!   `client_throughput`).
//! - `field/batchB-rowsR`: a cleartext field-element matvec of the same
//!   logical `rows x ell` problem, served by *the same WGSL answer kernel*
//!   the EMVP server runs, driven by bench-local wgpu plumbing with the
//!   trivial block decomposition `s = 1, b = ell`. The emvp-to-field ratio
//!   isolates what masking and encryption cost on identical hardware; the
//!   remaining emvp work is the `rows x n` expanded width the protocol
//!   answers over (`n = 2k > ell`), which is inherent to it.
//! - `f32/batchB-rowsR`: a tuned float32 GEMV (`gemv_f32.wgsl`:
//!   workgroup-per-row cooperative reduction over vector-shaped loads), the
//!   throughput an unencrypted f32 deployment gets from the same device.
//!
//! # Fairness contract
//!
//! - Same logical problem: every case reports
//!   `Throughput::Elements(rows * ell)`, so criterion's elem/s column and
//!   every plaintext-to-protocol ratio compare directly across cases, and
//!   with the `gpu` suite's batch grid.
//! - Same device discipline: EMVP and the field baseline run the identical
//!   kernel and pipeline constants; all three paths reuse their device
//!   buffers and output storage across iterations exactly like the
//!   `GpuAnswerer` scratch pool and `AnswerWorkspace` arena, create bind
//!   groups per dispatch like the library path, and upload fresh query data
//!   per batch like queries arriving over the network.
//! - No pinned pools: the bench-side query generation runs on rayon's
//!   global pool at its default width, strictly outside timing, so it never
//!   lands in a measured iteration. Deliberately different from the
//!   `online` suite's single-core contract; every CPU-side suite now shares
//!   the global pool.
//! - Answer phase only: the measured emvp iteration is exactly the server
//!   answer call; the same queries are answered every iteration (field
//!   arithmetic performance is data-independent, so re-answering them
//!   measures the same work fresh queries would).
//! - One mask construction: the suite runs the toeplitz suite only;
//!   client query cost varies slightly by construction and the `online`
//!   suite already covers that spread on the CPU.
//!
//! Without a compute adapter the binary prints a notice and benchmarks
//! nothing.
//!
//! # Correctness pins
//!
//! Once per row count, outside Criterion timing: the bench-local field path
//! must reproduce the CPU `dot_product` word-for-word, and the f32 kernel
//! must match a naive CPU GEMV within f32 round-off tolerance (summation
//! order differs, so bit equality is not expected there). Both checks
//! double as pipeline and buffer warm-up for the cleartext paths.
//!
//! ```text
//! cargo bench -p emvp --features gpu --bench gpu_online
//! ```

use std::hint::black_box;
use std::iter;
use std::sync::mpsc;
use std::time::Duration;

use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, criterion_group, criterion_main, measurement::WallTime,
};
use emvp::gpu::{WORKGROUP_SIZE, fold_fast_path, permuted_matrix_source, queries_per_tile};
use emvp::{
    AnswerPlan, AnswerWorkspace, DerivedState, EmvpParams, EncryptedMatrix, GpuAnswerer,
    GpuEncryptedMatrix, GpuError, encrypt, query_batch, search,
};
use prime_field_layer::arithmetic_kernels::dot_product;
use prime_field_layer::{FieldElement, PrimeField};
use rand_core::Rng;
use trapdoor_matrices::ToeplitzFastProduct;

use emvp_bench_common as common;

use common::{
    CONTEXT_TOEPLITZ, LLM_LAMBDA, MODULUS, derive_with, elements, field_values, seeded_rng,
    toeplitz_block,
};

// The answer kernel driving both the EMVP server and the cleartext field
// baseline: one source, so the emvp-to-field delta is pure protocol.
const ANSWER_WGSL: &str = include_str!("../src/gpu/answer.wgsl");
// The tuned float32 baseline kernel.
const GEMV_F32_WGSL: &str = include_str!("gemv_f32.wgsl");

// Threads per workgroup of the f32 baseline kernel, passed into its
// `@workgroup_size` override; the sweep value lives here so trying 64,
// 128, and 256 is a one-line change.
const F32_WORKGROUP_SIZE: u32 = 128;

// The 7B-class record length; the suite runs one record length to keep the
// case count bounded.
const ELL: usize = 4096;

// Row counts: the small occupancy-sensitive end and the large LLM-scale end.
const ONLINE_ROW_COUNTS: [usize; 2] = [4096, 16384];

// Query batches per measured iteration.
const BATCHES: [usize; 4] = [1, 8, 64, 256];

// Absolute f32 tolerance of the GEMV parity pin: random [-1, 1) operands
// over 4096-term dot products accumulate f32 round-off far below this, and
// a real kernel bug produces garbage far above it.
const F32_TOLERANCE: f32 = 1e-2;

// Uniform buffer sizes: the answer kernel's `Dims` struct (six used words,
// two padding) and the f32 kernel's four-word `Dims`.
const FIELD_UNIFORM_BYTES: u64 = 32;
const F32_UNIFORM_BYTES: u64 = 16;

// Bytes in one u32 word or one f32 value; the width every buffer moves.
const WORD_BYTES: u64 = 4;

/// Uniform f32 values in `[-1, 1)`, seeded like every other fixture.
///
/// The u32-to-f32 casts lose mantissa bits deliberately: the baseline needs
/// well-scaled random operands, not a lossless integer transport.
#[expect(
    clippy::cast_precision_loss,
    reason = "fixture values only need [-1, 1) coverage; the lost low bits are irrelevant to a performance baseline"
)]
fn f32_values(count: usize, domain: u8) -> Vec<f32> {
    let mut rng = seeded_rng(domain, count);
    (0..count)
        .map(|_| (rng.next_u32() as f32 / u32::MAX as f32).mul_add(2.0, -1.0))
        .collect()
}

/// The answer kernel's overrides for the bench modulus, exactly the
/// constants the library pipeline compiles (see
/// `GpuAnswerer::build_pipeline`).
fn field_constants() -> [(&'static str, f64); 5] {
    let field = PrimeField::<MODULUS>::new();
    [
        ("MODULUS", f64::from(MODULUS)),
        ("NEG_INV", f64::from(field.montgomery_neg_inv())),
        ("R1", f64::from(field.montgomery_r())),
        ("R2", f64::from(field.montgomery_r2())),
        ("WORKGROUP_SIZE", f64::from(WORKGROUP_SIZE)),
    ]
}

/// The bind layout both kernels share: matrix and query storage (read-only),
/// output storage (read-write), and the dimensions uniform.
fn bind_entries() -> [wgpu::BindGroupLayoutEntry; 4] {
    let storage = |read_only: bool| wgpu::BindingType::Buffer {
        ty: wgpu::BufferBindingType::Storage { read_only },
        has_dynamic_offset: false,
        min_binding_size: None,
    };
    [
        wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: storage(true),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 1,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: storage(true),
            count: None,
        },
        wgpu::BindGroupLayoutEntry {
            binding: 2,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty: storage(false),
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
    ]
}

/// One logical device shared by the bench-local cleartext pipelines.
struct BenchDevice {
    device: wgpu::Device,
    queue: wgpu::Queue,
}

impl BenchDevice {
    /// Mirrors `GpuAnswerer::acquire`'s adapter and device request: high
    /// performance preference, buffer limits raised to the adapter's
    /// maximum, and a blocking wait through `pollster` because the wgpu
    /// request interfaces are genuinely async. Returning an error string is
    /// enough for the bench, which either skips with a notice or fails the
    /// run loudly.
    fn new() -> Result<Self, String> {
        let attempt = async {
            let instance =
                wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
            let adapter = instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::HighPerformance,
                    compatible_surface: None,
                    force_fallback_adapter: false,
                    apply_limit_buckets: false,
                })
                .await
                .map_err(|error| error.to_string())?;
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
                    label: Some("gpu-online-bench-device"),
                    required_features: wgpu::Features::empty(),
                    required_limits,
                    experimental_features: wgpu::ExperimentalFeatures::disabled(),
                    memory_hints: wgpu::MemoryHints::Performance,
                    trace: wgpu::Trace::Off,
                })
                .await
                .map_err(|error| error.to_string())?;
            Ok(Self { device, queue })
        };
        pollster::block_on(attempt)
    }

    /// Compiles one compute pipeline from `source` with the shared bind
    /// layout, `constants`, and `entry_point`, mirroring the library's
    /// pipeline build.
    fn pipeline(
        &self,
        source: &'static str,
        label: &'static str,
        entry_point: &'static str,
        constants: &[(&'static str, f64)],
    ) -> wgpu::ComputePipeline {
        let module = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some(label),
                source: wgpu::ShaderSource::Wgsl(source.into()),
            });
        let layout = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some(label),
                entries: &bind_entries(),
            });
        let pipeline_layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: &[Some(&layout)],
                immediate_size: 0,
            });
        self.device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants,
                    zero_initialize_workgroup_memory: true,
                },
                cache: None,
            })
    }

    /// Uploads `byte_len` bytes produced by `fill` into a storage buffer and
    /// flushes the transfer with an immediate submission, mirroring
    /// `GpuAnswerer::upload_matrix`'s staged single-pass upload. Setup-time
    /// only: the measured iterations reuse the returned buffer.
    fn upload(
        &self,
        label: &'static str,
        byte_len: u64,
        fill: impl FnOnce(&mut wgpu::QueueWriteBufferView),
    ) -> wgpu::Buffer {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_len,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut view = self
            .queue
            .write_buffer_with(&buffer, 0, wgpu::BufferSize::new(byte_len).unwrap())
            .unwrap();
        fill(&mut view);
        drop(view);
        self.queue.submit([]);
        buffer
    }
}

/// The f32 baseline's two pipelines: the single-query entry point and the
/// four-query tile, compiled from one source with one override value.
struct F32Pipelines {
    single: wgpu::ComputePipeline,
    tiled: wgpu::ComputePipeline,
}

/// One cleartext path's run-wide device state: the shared bench device, the
/// compiled pipeline, and the device-resident matrix the measured batches
/// reuse. Created once per (family, row count) like the library's uploaded
/// matrix.
struct CleartextServer<'a> {
    bench: &'a BenchDevice,
    pipeline: &'a wgpu::ComputePipeline,
    matrix: &'a wgpu::Buffer,
}

/// Reused per-case device buffers for one cleartext batch shape, mirroring
/// the `GpuAnswerer` scratch contract: steady-state iterations skip device
/// allocation exactly like the protocol path they are compared against.
struct CaseBuffers {
    query: wgpu::Buffer,
    uniform: wgpu::Buffer,
    output: wgpu::Buffer,
    staging: wgpu::Buffer,
    query_bytes: u64,
    answer_bytes: u64,
}

impl CaseBuffers {
    /// Creates the four buffers sized for one case's batch. `query_words`
    /// is `batch * ell`, `answer_words` is `batch * rows` for both kernels
    /// (the answer kernel writes `s = 1` block per row, the f32 kernel one
    /// value per pair).
    fn new(
        bench: &BenchDevice,
        uniform_bytes: u64,
        query_words: usize,
        answer_words: usize,
    ) -> Self {
        let buffer = |label, byte_len, usage| {
            bench.device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: byte_len,
                usage,
                mapped_at_creation: false,
            })
        };
        Self {
            query: buffer(
                "gpu-online-cleartext-queries",
                WORD_BYTES * query_words as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            ),
            uniform: buffer(
                "gpu-online-cleartext-dims",
                uniform_bytes,
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            ),
            output: buffer(
                "gpu-online-cleartext-output",
                WORD_BYTES * answer_words as u64,
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            ),
            staging: buffer(
                "gpu-online-cleartext-staging",
                WORD_BYTES * answer_words as u64,
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            ),
            query_bytes: WORD_BYTES * query_words as u64,
            answer_bytes: WORD_BYTES * answer_words as u64,
        }
    }
}

/// Folds a linear workgroup count into the 2D dispatch the kernels
/// reconstruct their linear index from, matching the library's
/// `dispatch_grid` shape and the wgpu-guaranteed 65535 workgroups per
/// dimension.
fn grid_for(workgroup_total: usize) -> (u32, u32) {
    let x = u32::try_from(workgroup_total).unwrap().min(65_535);
    let y = u32::try_from(workgroup_total.div_ceil(usize::try_from(x).unwrap())).unwrap();
    (x, y)
}

/// The field baseline's dispatch grid for one `(rows, batch)`: the answer
/// kernel selects its query tile from the bench modulus and batch exactly
/// as the library path does, so the grid must cover
/// `ceil(batch / tile) * rows * s` threads at [`WORKGROUP_SIZE`] per
/// workgroup.
fn field_grid(rows: usize, batch: usize) -> (u32, u32) {
    let tile = usize::try_from(queries_per_tile(MODULUS, batch)).unwrap();
    let threads = batch.div_ceil(tile) * rows;
    grid_for(threads.div_ceil(WORKGROUP_SIZE as usize))
}

/// Encodes the answer kernel's `Dims` uniform for the cleartext baseline:
/// the same shader runs a plain matvec with the trivial block decomposition
/// `n = b = ell`, `s = 1`, plus `rows`, `batch`, the dispatch width, and
/// the direct final-fold flag (`b = ell` clears the exactness bound for
/// the bench modulus, so the fast fold is on).
fn field_uniform(ell: usize, rows: usize, batch: usize) -> [u8; 32] {
    let workgroups_x = field_grid(rows, batch).0 as usize;
    let mut bytes = [0_u8; 32];
    for (slot, dimension) in bytes
        .chunks_exact_mut(usize::try_from(WORD_BYTES).unwrap())
        .zip([ell, ell, 1, rows, batch, workgroups_x])
    {
        slot.copy_from_slice(&u32::try_from(dimension).unwrap().to_le_bytes());
    }
    let flag = fold_fast_path(MODULUS, ell);
    bytes[6 * WORD_BYTES as usize..6 * WORD_BYTES as usize + WORD_BYTES as usize]
        .copy_from_slice(&flag.to_le_bytes());
    bytes
}

/// The f32 baseline's query tile size: four queries per workgroup from the
/// same batch threshold the EMVP kernel uses (`f32_q4.wgsl`'s tiled entry
/// point), the single-query kernel below it.
const fn f32_tile(batch: usize) -> usize {
    if batch >= 4 { 4 } else { 1 }
}

/// The f32 baseline's dispatch grid for one `(rows, batch)`: one workgroup
/// per `ceil(batch / tile) * rows` output group.
fn f32_grid(rows: usize, batch: usize) -> (u32, u32) {
    let tile = f32_tile(batch);
    let groups = batch.div_ceil(tile) * rows;
    grid_for(groups)
}

/// Encodes the f32 kernel's `Dims` uniform: `rows`, `batch`, `ell`, and the
/// dispatch width over the tiled output groups.
fn f32_uniform(ell: usize, rows: usize, batch: usize) -> [u8; 16] {
    let workgroups_x = f32_grid(rows, batch).0 as usize;
    let mut bytes = [0_u8; 16];
    for (slot, dimension) in bytes
        .chunks_exact_mut(usize::try_from(WORD_BYTES).unwrap())
        .zip([rows, batch, ell, workgroups_x])
    {
        slot.copy_from_slice(&u32::try_from(dimension).unwrap().to_le_bytes());
    }
    bytes
}

/// Records the dispatch and the readback copy, submits, and blocks the
/// calling thread until the copy has landed in the staging buffer. This is
/// the phase that waits on GPU execution, exactly like the library's
/// `wait_readback`. The bind group is created per dispatch, matching the
/// library path's per-batch work.
fn submit_and_wait(
    server: &CleartextServer<'_>,
    buffers: &CaseBuffers,
    workgroups: (u32, u32),
) -> Result<(), String> {
    let device = &server.bench.device;
    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some("gpu-online-bench-bind-group"),
        layout: &server.pipeline.get_bind_group_layout(0),
        entries: &[
            wgpu::BindGroupEntry {
                binding: 0,
                resource: server.matrix.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 1,
                resource: buffers.query.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 2,
                resource: buffers.output.as_entire_binding(),
            },
            wgpu::BindGroupEntry {
                binding: 3,
                resource: buffers.uniform.as_entire_binding(),
            },
        ],
    });
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("gpu-online-bench-encoder"),
    });
    {
        let mut compute_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("gpu-online-bench-pass"),
            timestamp_writes: None,
        });
        compute_pass.set_pipeline(server.pipeline);
        compute_pass.set_bind_group(0, &bind_group, &[]);
        compute_pass.dispatch_workgroups(workgroups.0, workgroups.1, 1);
    }
    encoder.copy_buffer_to_buffer(
        &buffers.output,
        0,
        &buffers.staging,
        0,
        buffers.answer_bytes,
    );
    let submission = server.bench.queue.submit([encoder.finish()]);
    let slice = buffers.staging.slice(0..buffers.answer_bytes);
    let (sender, receiver) = mpsc::channel();
    slice.map_async(wgpu::MapMode::Read, move |result| {
        let _delivery = sender.send(result.map_err(|error| error.to_string()));
    });
    device
        .poll(wgpu::PollType::Wait {
            submission_index: Some(submission),
            timeout: None,
        })
        .map_err(|error| error.to_string())?;
    receiver
        .recv()
        .map_err(|_dropped| "mapping callback result unavailable".to_string())?
}

/// Uploads the batch's query words through a staged write view, mirroring
/// the fused single-pass upload the library's query phase runs.
fn upload_query_words(
    server: &CleartextServer<'_>,
    buffers: &CaseBuffers,
    fill: impl FnOnce(&mut wgpu::QueueWriteBufferView),
) {
    let mut view = server
        .bench
        .queue
        .write_buffer_with(
            &buffers.query,
            0,
            wgpu::BufferSize::new(buffers.query_bytes).unwrap(),
        )
        .unwrap();
    fill(&mut view);
}

/// Streams little-endian words out of the mapped staging buffer into the
/// case's reused output vector, four bytes at a time through `word`. The
/// vector is reserved once per case, so steady-state calls refill it
/// without growing, matching the `AnswerWorkspace` contract.
fn reconstruct_into<T>(
    staging: &wgpu::Buffer,
    byte_len: u64,
    out: &mut Vec<T>,
    word: fn(&[u8]) -> T,
) {
    out.clear();
    let mapped = staging.slice(0..byte_len).get_mapped_range().unwrap();
    out.extend(
        mapped
            .chunks_exact(usize::try_from(WORD_BYTES).unwrap())
            .map(word),
    );
    drop(mapped);
    staging.unmap();
}

/// Builds one output word from its little-endian bytes: the cleartext
/// analogue of the library's checked answer reconstruction, minus the
/// canonicalization a cleartext result does not need.
const fn answer_word(word: &[u8]) -> u32 {
    let mut raw = [0_u8; 4];
    raw.copy_from_slice(word);
    u32::from_le_bytes(raw)
}

/// Builds one output f32 value from its little-endian bytes.
const fn answer_value(word: &[u8]) -> f32 {
    let mut raw = [0_u8; 4];
    raw.copy_from_slice(word);
    f32::from_le_bytes(raw)
}

/// One cleartext field matvec batch on the shared answer kernel: writes the
/// dimensions uniform, uploads the query words, dispatches, and refills the
/// case's reused answer words. Every call uploads `batch` copies of the
/// record, mirroring queries arriving over the network.
fn field_answer_batch(
    server: &CleartextServer<'_>,
    buffers: &CaseBuffers,
    uniform: &[u8; 32],
    record: &[FieldElement<MODULUS>],
    rows: usize,
    batch: usize,
    answers: &mut Vec<u32>,
) {
    server
        .bench
        .queue
        .write_buffer(&buffers.uniform, 0, uniform);
    upload_query_words(server, buffers, |view| {
        let (word_bytes, _tail) = view.slice(..).into_chunks::<4>();
        word_bytes.write_iter(
            iter::repeat_n(record.iter(), batch)
                .flatten()
                .map(|element| element.to_raw().to_le_bytes()),
        );
    });
    let workgroups = field_grid(rows, batch);
    submit_and_wait(server, buffers, workgroups).unwrap();
    reconstruct_into(&buffers.staging, buffers.answer_bytes, answers, answer_word);
}

/// One tuned f32 GEMV batch: same phases as [`field_answer_batch`] against
/// the f32 kernel and buffers, refilling the case's reused output values.
fn f32_gemv_batch(
    server: &CleartextServer<'_>,
    buffers: &CaseBuffers,
    uniform: &[u8; 16],
    queries: &[f32],
    rows: usize,
    batch: usize,
    values: &mut Vec<f32>,
) {
    server
        .bench
        .queue
        .write_buffer(&buffers.uniform, 0, uniform);
    upload_query_words(server, buffers, |view| {
        let (word_bytes, _tail) = view.slice(..).into_chunks::<4>();
        word_bytes.write_iter(queries.iter().map(|value| value.to_le_bytes()));
    });
    let workgroups = f32_grid(rows, batch);
    submit_and_wait(server, buffers, workgroups).unwrap();
    reconstruct_into(&buffers.staging, buffers.answer_bytes, values, answer_value);
}

/// Pins the bench-local field path to the CPU reference: one batch-1
/// dispatch must reproduce `dot_product` word-for-word, because it runs the
/// identical kernel over the identical Montgomery-word representation.
/// Runs outside Criterion timing and doubles as pipeline warm-up.
fn check_field_parity(
    server: &CleartextServer<'_>,
    matrix: &[FieldElement<MODULUS>],
    record: &[FieldElement<MODULUS>],
    ell: usize,
    rows: usize,
) {
    let buffers = CaseBuffers::new(server.bench, FIELD_UNIFORM_BYTES, ell, rows);
    let uniform = field_uniform(ell, rows, 1);
    let mut answers = Vec::with_capacity(rows);
    field_answer_batch(server, &buffers, &uniform, record, rows, 1, &mut answers);
    assert_eq!(answers.len(), rows, "field readback length");
    for (row, word) in answers.iter().enumerate() {
        let expected = dot_product(&matrix[row * ell..(row + 1) * ell], record).unwrap();
        assert_eq!(
            *word,
            expected.to_raw(),
            "field GPU row {row} diverges from CPU dot_product"
        );
    }
}

/// Pins the f32 kernel to a naive CPU GEMV within f32 round-off tolerance.
/// Runs outside Criterion timing and doubles as pipeline warm-up.
fn check_f32_parity(
    server: &CleartextServer<'_>,
    matrix: &[f32],
    queries: &[f32],
    ell: usize,
    rows: usize,
) {
    let buffers = CaseBuffers::new(server.bench, F32_UNIFORM_BYTES, ell, rows);
    let uniform = f32_uniform(ell, rows, 1);
    let mut values = Vec::with_capacity(rows);
    f32_gemv_batch(
        server,
        &buffers,
        &uniform,
        &queries[..ell],
        rows,
        1,
        &mut values,
    );
    assert_eq!(values.len(), rows, "f32 readback length");
    for (row, value) in values.iter().enumerate() {
        let mut expected = 0.0_f32;
        for (a, b) in matrix[row * ell..(row + 1) * ell]
            .iter()
            .zip(queries[..ell].iter())
        {
            expected += a * b;
        }
        assert!(
            (value - expected).abs() <= F32_TOLERANCE,
            "f32 GPU row {row}: {value} vs CPU {expected}"
        );
    }
}

/// Pins the f32 kernel's tiled entry point to a naive CPU GEMV within f32
/// round-off tolerance, at a batch that fills one tile and clamps a
/// partial one (batch 5: one full tile plus a half tile with guarded
/// stores). Runs outside Criterion timing and doubles as the tiled
/// pipeline's warm-up.
fn check_f32_tiled_parity(
    server: &CleartextServer<'_>,
    matrix: &[f32],
    queries: &[f32],
    ell: usize,
    rows: usize,
) {
    let batch = 5;
    let buffers = CaseBuffers::new(server.bench, F32_UNIFORM_BYTES, batch * ell, batch * rows);
    let uniform = f32_uniform(ell, rows, batch);
    let mut values = Vec::with_capacity(batch * rows);
    f32_gemv_batch(
        server,
        &buffers,
        &uniform,
        &queries[..batch * ell],
        rows,
        batch,
        &mut values,
    );
    assert_eq!(values.len(), batch * rows, "tiled f32 readback length");
    for (pair, value) in values.iter().enumerate() {
        let (query, row) = (pair / rows, pair % rows);
        let mut expected = 0.0_f32;
        for (a, b) in matrix[row * ell..(row + 1) * ell]
            .iter()
            .zip(queries[query * ell..(query + 1) * ell].iter())
        {
            expected += a * b;
        }
        assert!(
            (*value - expected).abs() <= F32_TOLERANCE,
            "tiled f32 q{query} row{row}: {value} vs CPU {expected}"
        );
    }
}

/// One protocol instance shared by every batch case of one row count.
struct AnswerFixtures<'a> {
    /// The advancing client state; the case's query batch is generated
    /// from it once, strictly outside timing.
    state: &'a mut DerivedState<MODULUS, ToeplitzFastProduct<MODULUS>>,
    encrypted: &'a EncryptedMatrix<MODULUS>,
    gpu_matrix: &'a GpuEncryptedMatrix<MODULUS>,
    /// `batch` views of one record: the same vector queried `batch` times
    /// with fresh randomness, matching the `online` suite's round trip.
    record_slices: &'a [&'a [FieldElement<MODULUS>]],
    ell: usize,
    rows: usize,
}

/// Registers the EMVP answer-phase case for one (rows, batch): the batch's
/// queries are generated once and the workspace planned and reserved once,
/// both strictly outside timing, then every measured iteration is exactly
/// one [`GpuAnswerer::execute_answer_batch_into`] call into the once-
/// reserved workspace. No client work lands in the timed iteration.
fn bench_emvp_case(
    group: &mut BenchmarkGroup<'_, WallTime>,
    tag: &str,
    params: EmvpParams,
    fixtures: AnswerFixtures<'_>,
    answerer: &GpuAnswerer,
) {
    let AnswerFixtures {
        state,
        encrypted,
        gpu_matrix,
        record_slices,
        ell,
        rows,
    } = fixtures;
    // The case's query batch, generated once outside timing.
    let pairs = query_batch(state, record_slices).unwrap();
    let (queries, _decoding_keys): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
    // Plan and reserve once: the shape arithmetic only depends on (rows,
    // blocks, batch, instance), which every iteration shares.
    let host_plan = AnswerPlan::plan(&params, encrypted, &queries).unwrap();
    let gpu_shape = answerer.answer_batch_plan(gpu_matrix, &queries).unwrap();
    assert_eq!(gpu_shape, host_plan.shape(), "tier shapes must agree");
    let mut workspace = AnswerWorkspace::<MODULUS>::new();
    workspace.reserve(&host_plan).unwrap();

    group.throughput(elements(rows * ell));
    group.bench_function(BenchmarkId::new("emvp", tag.to_owned()), |b| {
        b.iter(|| {
            let (answers, _timings) = answerer
                .execute_answer_batch_into(gpu_matrix, &queries, &mut workspace)
                .unwrap();
            black_box(&answers);
        });
    });
}

/// Registers the cleartext field case for one (rows, batch): one dispatch of
/// the same WGSL answer kernel the EMVP server runs, on unencrypted field
/// elements with the trivial decomposition `s = 1, b = ell`.
fn bench_field_case(
    group: &mut BenchmarkGroup<'_, WallTime>,
    tag: &str,
    server: &CleartextServer<'_>,
    record: &[FieldElement<MODULUS>],
    ell: usize,
    rows: usize,
    batch: usize,
) {
    let buffers = CaseBuffers::new(server.bench, FIELD_UNIFORM_BYTES, batch * ell, batch * rows);
    let uniform = field_uniform(ell, rows, batch);
    // Reserved once, like the protocol path's workspace.
    let mut answers = Vec::with_capacity(batch * rows);
    group.throughput(elements(rows * ell));
    group.bench_function(BenchmarkId::new("field", tag.to_owned()), |b| {
        b.iter(|| {
            field_answer_batch(
                server,
                &buffers,
                &uniform,
                record,
                rows,
                batch,
                &mut answers,
            );
            black_box(&answers);
        });
    });
}

/// Registers the tuned float32 case for one (rows, batch): the f32 GEMV a
/// cleartext deployment would run on the same device, over the same logical
/// `rows x ell` problem.
fn bench_f32_case(
    group: &mut BenchmarkGroup<'_, WallTime>,
    tag: &str,
    server: &CleartextServer<'_>,
    queries: &[f32],
    ell: usize,
    rows: usize,
    batch: usize,
) {
    let buffers = CaseBuffers::new(server.bench, F32_UNIFORM_BYTES, batch * ell, batch * rows);
    let uniform = f32_uniform(ell, rows, batch);
    // Reserved once, like the protocol path's workspace.
    let mut values = Vec::with_capacity(batch * rows);
    group.throughput(elements(rows * ell));
    group.bench_function(BenchmarkId::new("f32", tag.to_owned()), |b| {
        b.iter(|| {
            f32_gemv_batch(
                server,
                &buffers,
                &uniform,
                &queries[..batch * ell],
                rows,
                batch,
                &mut values,
            );
            black_box(&values);
        });
    });
}

/// Uploads one field matrix from its Montgomery words, permuting the
/// wire's row-major order into the device layout the answer kernel reads
/// (the same transformation the library's upload runs).
fn upload_field_matrix(
    bench: &BenchDevice,
    values: &[FieldElement<MODULUS>],
    rows: usize,
    n: usize,
) -> wgpu::Buffer {
    bench.upload(
        "gpu-online-field-matrix",
        WORD_BYTES * values.len() as u64,
        |view| {
            let (word_bytes, _tail) = view.slice(..).into_chunks::<4>();
            for (destination, slot) in word_bytes.into_iter().enumerate() {
                let source = permuted_matrix_source(destination, rows, n);
                slot.write(values[source].to_raw().to_le_bytes());
            }
        },
    )
}

/// Uploads one f32 matrix from its values.
fn upload_f32_matrix(bench: &BenchDevice, values: &[f32]) -> wgpu::Buffer {
    bench.upload(
        "gpu-online-f32-matrix",
        WORD_BYTES * values.len() as u64,
        |view| {
            let (word_bytes, _tail) = view.slice(..).into_chunks::<4>();
            word_bytes.write_iter(values.iter().map(|value| value.to_le_bytes()));
        },
    )
}

/// All cases for one row count: the protocol instance is encrypted and
/// uploaded once, both cleartext matrices upload once, the parity pins run
/// once, and then every batch case registers its three families.
fn bench_row_count(
    group: &mut BenchmarkGroup<'_, WallTime>,
    answerer: &GpuAnswerer,
    bench: &BenchDevice,
    field_pipeline: &wgpu::ComputePipeline,
    f32_pipelines: &F32Pipelines,
    params: EmvpParams,
    rows: usize,
) {
    // One protocol instance per row count: encrypt and upload once, then
    // serve every batch case from the same device-resident matrix.
    let mut state = derive_with(params, rows, 0x06, CONTEXT_TOEPLITZ, toeplitz_block);
    let matrix = field_values(rows * params.ell, 0x07);
    let encrypted = encrypt(&mut state, &matrix).unwrap();
    let gpu_matrix = answerer.upload_matrix(&params, &encrypted).unwrap();
    let record = field_values(params.ell, 0x08);

    // Cleartext fixtures for the same logical rows x ell problem, plus the
    // maximum-batch f32 query pool the batch cases slice into.
    let f32_matrix_values = f32_values(rows * params.ell, 0x0a);
    let f32_queries = f32_values(BATCHES[BATCHES.len() - 1] * params.ell, 0x0b);
    let field_matrix = upload_field_matrix(bench, &matrix, rows, params.ell);
    let f32_matrix = upload_f32_matrix(bench, &f32_matrix_values);
    let field_server = CleartextServer {
        bench,
        pipeline: field_pipeline,
        matrix: &field_matrix,
    };
    let f32_server = CleartextServer {
        bench,
        pipeline: &f32_pipelines.single,
        matrix: &f32_matrix,
    };
    let f32_q4_server = CleartextServer {
        bench,
        pipeline: &f32_pipelines.tiled,
        matrix: &f32_matrix,
    };

    // One-time parity pins, outside Criterion timing.
    check_field_parity(&field_server, &matrix, &record, params.ell, rows);
    check_f32_parity(
        &f32_server,
        &f32_matrix_values,
        &f32_queries,
        params.ell,
        rows,
    );
    check_f32_tiled_parity(
        &f32_q4_server,
        &f32_matrix_values,
        &f32_queries,
        params.ell,
        rows,
    );

    for &batch in &BATCHES {
        let tag = format!("batch{batch}-rows{rows}");
        let record_slices: Vec<&[FieldElement<MODULUS>]> = vec![record.as_slice(); batch];
        bench_emvp_case(
            group,
            &tag,
            params,
            AnswerFixtures {
                state: &mut state,
                encrypted: &encrypted,
                gpu_matrix: &gpu_matrix,
                record_slices: &record_slices,
                ell: params.ell,
                rows,
            },
            answerer,
        );
        bench_field_case(group, &tag, &field_server, &record, params.ell, rows, batch);
        // The tiled f32 entry point serves batches that fill a tile,
        // mirroring the EMVP kernel's tile policy.
        let f32_case_server = if f32_tile(batch) == 4 {
            &f32_q4_server
        } else {
            &f32_server
        };
        bench_f32_case(
            group,
            &tag,
            f32_case_server,
            &f32_queries,
            params.ell,
            rows,
            batch,
        );
    }
}

fn gpu_online_benches(criterion: &mut Criterion) {
    let answerer = match GpuAnswerer::new() {
        Ok(answerer) => answerer,
        Err(GpuError::NoAdapter { reason }) => {
            println!("skipping gpu online benches: no compute adapter available ({reason})");
            return;
        }
        Err(error) => {
            let failure = Err::<GpuAnswerer, GpuError>(error);
            failure.unwrap()
        }
    };
    let bench = match BenchDevice::new() {
        Ok(bench) => bench,
        Err(reason) => {
            println!("skipping gpu online benches: no compute adapter available ({reason})");
            return;
        }
    };
    let field_pipeline = bench.pipeline(
        ANSWER_WGSL,
        "gpu-online-field-answer.wgsl",
        "main",
        &field_constants(),
    );
    let f32_constants = [("WORKGROUP_SIZE", f64::from(F32_WORKGROUP_SIZE))];
    let f32_pipelines = F32Pipelines {
        single: bench.pipeline(
            GEMV_F32_WGSL,
            "gpu-online-gemv-f32.wgsl",
            "main",
            &f32_constants,
        ),
        tiled: bench.pipeline(
            GEMV_F32_WGSL,
            "gpu-online-gemv-f32-q4.wgsl",
            "main_q4",
            &f32_constants,
        ),
    };

    let params = search(ELL, LLM_LAMBDA).unwrap();
    println!(
        "gpu online suite at ell = {ELL}: k = {}, b = {}, n = {}, s = {}, lambda = {}",
        params.k,
        params.b,
        params.n().unwrap(),
        params.blocks().unwrap(),
        params.lambda
    );

    let mut group = criterion.benchmark_group("gpu_answer_cost_v1");
    // LLM-scale iterations cost tens to hundreds of milliseconds; fewer
    // samples keep the run bounded, matching the gpu suite.
    group.sample_size(10);

    for &rows in &ONLINE_ROW_COUNTS {
        bench_row_count(
            &mut group,
            &answerer,
            &bench,
            &field_pipeline,
            &f32_pipelines,
            params,
            rows,
        );
    }
    group.finish();
}

fn criterion_config() -> Criterion {
    Criterion::default()
        .sample_size(10)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(4))
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = gpu_online_benches
}
criterion_main!(benches);
