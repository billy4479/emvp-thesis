//! Packed multi-matrix execution used by [`crate::engine::AnswerEngine`].

use std::ops::Range;
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use prime_field_layer::FieldElement;

use super::{
    DIMS_UNIFORM_BYTES, GpuAnswerer, GpuEncryptedMatrix, GpuError, MAX_ANSWER_WORDS,
    create_buffer_checked, dims_uniform_bytes, dispatch_grid, lock_recovered, queries_per_tile,
    staged_answer_word,
};
use crate::EncryptedQuery;
use crate::view::QueryValues;

const WORD_BYTES: u64 = 4;

/// One complete engine job selected for device execution.
///
/// Generic over the query representation so the engine's packed flight
/// answers owned [`EncryptedQuery`] batches and borrowed wire views alike;
/// the default keeps the historical owned-queries form.
pub struct PackedJob<'a, const MODULUS: u32, Q: QueryValues<MODULUS> = EncryptedQuery<MODULUS>> {
    pub entry: usize,
    pub matrix: &'a GpuEncryptedMatrix<MODULUS>,
    pub queries: &'a [Q],
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PackedTimings {
    pub prepare_buffers: Duration,
    pub encode_upload_queries: Duration,
    pub dispatch_submit: Duration,
    pub wait_readback: Duration,
    pub reconstruct: Duration,
}

#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct PackedStats {
    pub chunks: usize,
    pub segments: usize,
    pub query_bytes: u64,
    pub answer_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Segment {
    job: usize,
    query_start: usize,
    query_count: usize,
    query_offset_words: usize,
    answer_offset_words: usize,
    query_words: usize,
    answer_words: usize,
    /// Queries per kernel invocation for this segment (1 or 4, from
    /// `gpu::queries_per_tile`); the dispatch covers
    /// `ceil(query_count / tile) * rows * s` threads.
    tile: u32,
    workgroups_x: u32,
    workgroups_y: u32,
    uniform: [u8; DIMS_UNIFORM_BYTES as usize],
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ChunkPlan {
    segments: Vec<Segment>,
    query_words: usize,
    answer_words: usize,
}

#[derive(Clone, Copy)]
pub struct Shape {
    n: usize,
    b: usize,
    rows: usize,
    s: usize,
}

#[derive(Debug)]
pub struct PackedScratch {
    chunks: Vec<ChunkScratch>,
}

#[derive(Debug)]
struct ChunkScratch {
    query: wgpu::Buffer,
    output: wgpu::Buffer,
    staging: wgpu::Buffer,
    uniforms: wgpu::Buffer,
    query_capacity: u64,
    answer_capacity: u64,
    uniform_capacity: u64,
}

struct FlightChunk {
    plan: ChunkPlan,
    receiver: Receiver<Result<(), String>>,
}

/// A submitted packed operation whose answers reconstruct into a
/// caller-owned flat engine arena instead of per-query owned vectors.
///
/// Produced by [`GpuAnswerer::begin_packed_into`] and consumed by
/// [`GpuAnswerer::finish_packed_into_at`]. It carries the device-side
/// bookkeeping plus the per-job kernel shapes and engine-entry indices the
/// into-write streams against.
pub struct PackedFlightInto<const MODULUS: u32> {
    scratch: Option<PackedScratch>,
    chunks: Vec<FlightChunk>,
    /// Per-job kernel shapes, in job order; the rows and blocks fields fix
    /// each segment's words per query.
    shapes: Vec<Shape>,
    /// The engine entry index of each packed job, for reporting.
    entries: Vec<usize>,
    submission: wgpu::SubmissionIndex,
    stats: PackedStats,
    timings: PackedTimings,
}

impl<const MODULUS: u32> PackedFlightInto<MODULUS> {
    /// Returns `(engine entry, dispatched segment count)` per packed job,
    /// for reporting.
    pub fn segment_counts(&self) -> Vec<(usize, usize)> {
        let mut counts = vec![0; self.entries.len()];
        for segment in self.chunks.iter().flat_map(|chunk| &chunk.plan.segments) {
            counts[segment.job] += 1;
        }
        self.entries
            .iter()
            .zip(counts)
            .map(|(entry, count)| (*entry, count))
            .collect()
    }
}

impl<const MODULUS: u32> Drop for PackedFlightInto<MODULUS> {
    fn drop(&mut self) {
        if let Some(scratch) = &self.scratch {
            for buffers in scratch.chunks.iter().take(self.chunks.len()) {
                buffers.staging.unmap();
            }
        }
    }
}

pub struct PackedPlan {
    chunks: Vec<ChunkPlan>,
    uniform_alignment: u64,
    stats: PackedStats,
}

impl PackedPlan {
    /// Returns `(job, dispatched segment count)` per packed job, the
    /// device-free counterpart of [`PackedFlightInto::segment_counts`]: the
    /// planner already knows how a job's queries split into dispatch
    /// segments, so callers can report per-entry segment counts before any
    /// device work. Jobs without segments (none can be planned) are
    /// absent from the returned pairs.
    #[must_use]
    pub fn segment_counts(&self) -> Vec<(usize, usize)> {
        let mut counts: Vec<(usize, usize)> = Vec::new();
        for segment in self.chunks.iter().flat_map(|chunk| &chunk.segments) {
            match counts.iter_mut().find(|(job, _)| *job == segment.job) {
                Some((_, count)) => *count += 1,
                None => counts.push((segment.job, 1)),
            }
        }
        counts
    }
}

fn align_up(value: usize, alignment: usize) -> Result<usize, GpuError> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum / alignment * alignment)
        .ok_or(GpuError::DimensionOverflow)
}

fn plan_chunks(
    modulus: u32,
    shapes: &[Shape],
    query_counts: &[usize],
    max_buffer_words: usize,
    max_binding_words: usize,
    alignment_words: usize,
) -> Result<Vec<ChunkPlan>, GpuError> {
    let mut chunks = vec![ChunkPlan::default()];
    for (job, (&shape, &query_count)) in shapes.iter().zip(query_counts).enumerate() {
        let answer_per_query = shape
            .rows
            .checked_mul(shape.s)
            .ok_or(GpuError::DimensionOverflow)?;
        let max_queries = (max_binding_words / shape.n)
            .min(max_binding_words / answer_per_query)
            .min((u32::MAX as usize) / shape.n)
            .min(MAX_ANSWER_WORDS / answer_per_query);
        if max_queries == 0 {
            return Err(GpuError::UploadTooLarge {
                elements: shape.n.max(answer_per_query),
                max_elements: max_binding_words,
            });
        }
        let mut query_start = 0;
        while query_start < query_count {
            let mut count = (query_count - query_start).min(max_queries);
            loop {
                let Some(chunk) = chunks.last_mut() else {
                    return Err(GpuError::DimensionOverflow);
                };
                let query_offset = align_up(chunk.query_words, alignment_words)?;
                let answer_offset = align_up(chunk.answer_words, alignment_words)?;
                let query_words = count
                    .checked_mul(shape.n)
                    .ok_or(GpuError::DimensionOverflow)?;
                let answer_words = count
                    .checked_mul(answer_per_query)
                    .ok_or(GpuError::DimensionOverflow)?;
                let query_end = query_offset
                    .checked_add(query_words)
                    .ok_or(GpuError::DimensionOverflow)?;
                let answer_end = answer_offset
                    .checked_add(answer_words)
                    .ok_or(GpuError::DimensionOverflow)?;
                if query_end <= max_buffer_words && answer_end <= max_buffer_words {
                    let tile = queries_per_tile(modulus, count);
                    let threads = count
                        .div_ceil(tile as usize)
                        .checked_mul(answer_per_query)
                        .ok_or(GpuError::DimensionOverflow)?;
                    let (workgroups_x, workgroups_y) = dispatch_grid(threads)?;
                    let uniform = dims_uniform_bytes(
                        modulus,
                        shape.n,
                        shape.b,
                        shape.s,
                        shape.rows,
                        count,
                        workgroups_x,
                    )?;
                    chunk.segments.push(Segment {
                        job,
                        query_start,
                        query_count: count,
                        query_offset_words: query_offset,
                        answer_offset_words: answer_offset,
                        query_words,
                        answer_words,
                        tile,
                        workgroups_x,
                        workgroups_y,
                        uniform,
                    });
                    chunk.query_words = query_end;
                    chunk.answer_words = answer_end;
                    query_start += count;
                    break;
                }
                if chunk.segments.is_empty() {
                    count /= 2;
                    if count == 0 {
                        return Err(GpuError::UploadTooLarge {
                            elements: shape.n.max(answer_per_query),
                            max_elements: max_buffer_words,
                        });
                    }
                } else {
                    chunks.push(ChunkPlan::default());
                    count = (query_count - query_start).min(max_queries);
                }
            }
        }
    }
    Ok(chunks)
}

/// The query-major destination word range of one packed segment inside a
/// flat engine arena at an absolute base.
///
/// Given the segment's *absolute* arena base `job_base_words[job]`, the
/// within-entry query-major arithmetic is
/// `job_base_words[job] + query_start * rows * s ..
/// .. + query_count * rows * s`, so a packed flight can land inside a
/// larger caller-owned engine arena whose entries interleave CPU- and
/// GPU-tier answers.
///
/// Pure host arithmetic with checked overflow; `job_base_words` must have
/// one slot per packed job, and the caller owns the invariant that the
/// per-job runs are disjoint inside `dest`.
pub fn segment_dest_range_at(
    job_base_words: &[usize],
    job: usize,
    query_start: usize,
    query_count: usize,
    rows: usize,
    s: usize,
) -> Result<Range<usize>, GpuError> {
    let base = job_base_words
        .get(job)
        .copied()
        .ok_or(GpuError::LengthMismatch {
            name: "packed segment job",
            expected: job_base_words.len(),
            actual: job,
        })?;
    dest_range_at_base(base, query_start, query_count, rows, s)
}

/// The shared query-major arithmetic of the segment-destination ranges:
/// the run `[base + query_start * words_per_query,
/// base + (query_start + query_count) * words_per_query)` with checked
/// overflow.
fn dest_range_at_base(
    base: usize,
    query_start: usize,
    query_count: usize,
    rows: usize,
    s: usize,
) -> Result<Range<usize>, GpuError> {
    let words_per_query = rows.checked_mul(s).ok_or(GpuError::DimensionOverflow)?;
    let start = base
        .checked_add(
            query_start
                .checked_mul(words_per_query)
                .ok_or(GpuError::DimensionOverflow)?,
        )
        .ok_or(GpuError::DimensionOverflow)?;
    let end = start
        .checked_add(
            query_count
                .checked_mul(words_per_query)
                .ok_or(GpuError::DimensionOverflow)?,
        )
        .ok_or(GpuError::DimensionOverflow)?;
    Ok(start..end)
}

/// The per-job kernel shapes of a packed job list.
fn job_shapes<const MODULUS: u32, Q: QueryValues<MODULUS>>(
    jobs: &[PackedJob<'_, MODULUS, Q>],
) -> Result<Vec<Shape>, GpuError> {
    jobs.iter()
        .map(|job| {
            Ok(Shape {
                n: job.matrix.columns,
                b: job.matrix.params.block_size(),
                rows: job.matrix.rows,
                s: job.matrix.params.blocks()?,
            })
        })
        .collect()
}

/// Streams one mapped chunk's staged answer words into `dest` at each
/// segment's `segment_dest_range_at` absolute range — the device-free
/// core of the packed into-path.
///
/// `chunks` and `staged` must be parallel (`staged[i]` holds chunk `i`'s
/// mapped staging bytes, exactly `chunk.answer_words` little-endian
/// words). Each readback word is wrapped with the checked
/// [`staged_answer_word`]: a noncanonical word rejects the whole call and
/// leaves the not-yet-written part of `dest` untouched, so callers treat
/// the arena as failed. No per-query `Vec`, no intermediate collection:
/// one pass per chunk from mapped bytes into the caller's arena.
pub fn reconstruct_packed_into_at<const MODULUS: u32>(
    chunks: &[ChunkPlan],
    staged: &[&[u8]],
    job_base_words: &[usize],
    shapes: &[Shape],
    dest: &mut [FieldElement<MODULUS>],
) -> Result<(), GpuError> {
    if chunks.len() != staged.len() {
        return Err(GpuError::LengthMismatch {
            name: "packed staging views",
            expected: chunks.len(),
            actual: staged.len(),
        });
    }
    for (chunk, data) in chunks.iter().zip(staged.iter().copied()) {
        for segment in &chunk.segments {
            let shape = shapes.get(segment.job).ok_or(GpuError::LengthMismatch {
                name: "packed segment job",
                expected: shapes.len(),
                actual: segment.job,
            })?;
            let range = segment_dest_range_at(
                job_base_words,
                segment.job,
                segment.query_start,
                segment.query_count,
                shape.rows,
                shape.s,
            )?;
            stream_segment_into::<MODULUS>(segment, data, range, dest)?;
        }
    }
    Ok(())
}

/// Streams one segment's staged answer words into their destination range
/// — the shared per-segment tail of both reconstruct variants.
///
/// Fail-closed cross-checks: the planner pins every segment's answer run
/// to `query_count * rows * s` words inside its chunk's staging buffer,
/// and the caller pins `dest` to hold each segment's range.
fn stream_segment_into<const MODULUS: u32>(
    segment: &Segment,
    data: &[u8],
    range: Range<usize>,
    dest: &mut [FieldElement<MODULUS>],
) -> Result<(), GpuError> {
    let bytes_start = segment
        .answer_offset_words
        .checked_mul(WORD_BYTES as usize)
        .ok_or(GpuError::DimensionOverflow)?;
    let bytes_end = bytes_start
        .checked_add(
            segment
                .answer_words
                .checked_mul(WORD_BYTES as usize)
                .ok_or(GpuError::DimensionOverflow)?,
        )
        .ok_or(GpuError::DimensionOverflow)?;
    let words = data
        .get(bytes_start..bytes_end)
        .ok_or(GpuError::LengthMismatch {
            name: "packed staging bytes",
            expected: bytes_end,
            actual: data.len(),
        })?;
    let dest_len = dest.len();
    let range_end = range.end;
    let slots = dest.get_mut(range).ok_or(GpuError::LengthMismatch {
        name: "packed answer arena",
        expected: range_end,
        actual: dest_len,
    })?;
    if slots.len() != segment.answer_words {
        return Err(GpuError::LengthMismatch {
            name: "packed answer arena",
            expected: segment.answer_words,
            actual: slots.len(),
        });
    }
    for (slot, word) in slots
        .iter_mut()
        .zip(words.chunks_exact(WORD_BYTES as usize))
    {
        *slot = staged_answer_word::<MODULUS>(word)?;
    }
    Ok(())
}

impl PackedScratch {
    fn prepare(
        &mut self,
        device: &wgpu::Device,
        plans: &[ChunkPlan],
        uniform_alignment: u64,
    ) -> Result<(), GpuError> {
        while self.chunks.len() < plans.len() {
            self.chunks.push(ChunkScratch::new(device)?);
        }
        for (scratch, plan) in self.chunks.iter_mut().zip(plans) {
            scratch.prepare(device, plan, uniform_alignment)?;
        }
        Ok(())
    }
}

impl ChunkScratch {
    fn new(device: &wgpu::Device) -> Result<Self, GpuError> {
        let buffer = |label, usage| {
            create_buffer_checked(
                device,
                label,
                &wgpu::BufferDescriptor {
                    label: Some(label),
                    size: 4,
                    usage,
                    mapped_at_creation: false,
                },
            )
        };
        Ok(Self {
            query: buffer(
                "packed query buffer",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            )?,
            output: buffer(
                "packed output buffer",
                wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            )?,
            staging: buffer(
                "packed staging buffer",
                wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            )?,
            uniforms: buffer(
                "packed uniform buffer",
                wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            )?,
            query_capacity: 1,
            answer_capacity: 1,
            uniform_capacity: 4,
        })
    }

    fn prepare(
        &mut self,
        device: &wgpu::Device,
        plan: &ChunkPlan,
        uniform_alignment: u64,
    ) -> Result<(), GpuError> {
        let query_bytes = (plan.query_words as u64).max(1) * WORD_BYTES;
        let answer_bytes = (plan.answer_words as u64).max(1) * WORD_BYTES;
        let uniform_bytes = (plan.segments.len() as u64)
            .checked_mul(uniform_alignment)
            .ok_or(GpuError::DimensionOverflow)?
            .max(DIMS_UNIFORM_BYTES);
        if query_bytes > self.query_capacity {
            self.query = create_buffer_checked(
                device,
                "packed query buffer",
                &wgpu::BufferDescriptor {
                    label: Some("emvp-packed-queries"),
                    size: query_bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            )?;
            self.query_capacity = query_bytes;
        }
        if answer_bytes > self.answer_capacity {
            self.output = create_buffer_checked(
                device,
                "packed output buffer",
                &wgpu::BufferDescriptor {
                    label: Some("emvp-packed-output"),
                    size: answer_bytes,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
                    mapped_at_creation: false,
                },
            )?;
            self.staging = create_buffer_checked(
                device,
                "packed staging buffer",
                &wgpu::BufferDescriptor {
                    label: Some("emvp-packed-staging"),
                    size: answer_bytes,
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            )?;
            self.answer_capacity = answer_bytes;
        }
        if uniform_bytes > self.uniform_capacity {
            self.uniforms = create_buffer_checked(
                device,
                "packed uniform buffer",
                &wgpu::BufferDescriptor {
                    label: Some("emvp-packed-uniforms"),
                    size: uniform_bytes,
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                },
            )?;
            self.uniform_capacity = uniform_bytes;
        }
        Ok(())
    }
}

impl GpuAnswerer {
    /// Completes all shape and device-limit checks without submitting or
    /// allocating device work.
    ///
    /// # Errors
    ///
    /// Returns an error when dimensions overflow, parameters are invalid, or
    /// one query cannot fit the device's buffer and binding limits.
    pub fn plan_packed<const MODULUS: u32, Q: QueryValues<MODULUS>>(
        &self,
        jobs: &[PackedJob<'_, MODULUS, Q>],
    ) -> Result<PackedPlan, GpuError> {
        let limits = self.device.limits();
        let max_buffer_words =
            usize::try_from(limits.max_buffer_size / WORD_BYTES).unwrap_or(usize::MAX);
        let max_binding_words =
            usize::try_from(limits.max_storage_buffer_binding_size / WORD_BYTES)
                .unwrap_or(usize::MAX);
        let alignment_words = usize::try_from(
            u64::from(limits.min_storage_buffer_offset_alignment).div_ceil(WORD_BYTES),
        )
        .unwrap_or(usize::MAX)
        .max(1);
        let uniform_alignment =
            u64::from(limits.min_uniform_buffer_offset_alignment).max(DIMS_UNIFORM_BYTES);
        let shapes: Vec<_> = jobs
            .iter()
            .map(|job| {
                Ok(Shape {
                    n: job.matrix.columns,
                    b: job.matrix.params.block_size(),
                    rows: job.matrix.rows,
                    s: job.matrix.params.blocks()?,
                })
            })
            .collect::<Result<_, GpuError>>()?;
        let query_counts: Vec<_> = jobs.iter().map(|job| job.queries.len()).collect();
        let chunks = plan_chunks(
            MODULUS,
            &shapes,
            &query_counts,
            max_buffer_words,
            max_binding_words,
            alignment_words,
        )?;
        let stats = PackedStats {
            chunks: chunks.len(),
            segments: chunks.iter().map(|chunk| chunk.segments.len()).sum(),
            query_bytes: chunks
                .iter()
                .map(|chunk| chunk.query_words as u64 * WORD_BYTES)
                .sum(),
            answer_bytes: chunks
                .iter()
                .map(|chunk| chunk.answer_words as u64 * WORD_BYTES)
                .sum(),
        };
        Ok(PackedPlan {
            chunks,
            uniform_alignment,
            stats,
        })
    }

    fn upload_packed_inputs<const MODULUS: u32, Q: QueryValues<MODULUS>>(
        &self,
        jobs: &[PackedJob<'_, MODULUS, Q>],
        plan: &PackedPlan,
        scratch: &PackedScratch,
    ) -> Result<(), GpuError> {
        for (chunk, buffers) in plan.chunks.iter().zip(&scratch.chunks) {
            for (segment_index, segment) in chunk.segments.iter().enumerate() {
                let job = &jobs[segment.job];
                let queries =
                    &job.queries[segment.query_start..segment.query_start + segment.query_count];
                write_query_segment(
                    &self.queue,
                    &buffers.query,
                    segment.query_offset_words as u64 * WORD_BYTES,
                    segment.query_words,
                    queries,
                )?;
                self.queue.write_buffer(
                    &buffers.uniforms,
                    segment_index as u64 * plan.uniform_alignment,
                    &segment.uniform,
                );
            }
        }
        Ok(())
    }

    fn encode_packed_commands<const MODULUS: u32, Q: QueryValues<MODULUS>>(
        &self,
        jobs: &[PackedJob<'_, MODULUS, Q>],
        plan: &PackedPlan,
        scratch: &PackedScratch,
    ) -> Result<wgpu::CommandEncoder, GpuError> {
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("emvp-packed-answer-encoder"),
            });
        for (chunk, buffers) in plan.chunks.iter().zip(&scratch.chunks) {
            for (segment_index, segment) in chunk.segments.iter().enumerate() {
                self.encode_packed_segment(
                    &mut encoder,
                    &jobs[segment.job],
                    buffers,
                    segment,
                    segment_index as u64 * plan.uniform_alignment,
                )?;
            }
            encoder.copy_buffer_to_buffer(
                &buffers.output,
                0,
                &buffers.staging,
                0,
                chunk.answer_words as u64 * WORD_BYTES,
            );
        }
        Ok(encoder)
    }

    fn encode_packed_segment<const MODULUS: u32, Q: QueryValues<MODULUS>>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        job: &PackedJob<'_, MODULUS, Q>,
        buffers: &ChunkScratch,
        segment: &Segment,
        uniform_offset: u64,
    ) -> Result<(), GpuError> {
        // The segment's tile selects the entry point; both pipelines' bind
        // group layouts are identical by construction.
        let pipeline = self.answer_pipeline::<MODULUS>(segment.tile)?;
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("emvp-packed-answer-bind-group"),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: job.matrix.buffer.as_entire_binding(),
                },
                buffer_binding(
                    1,
                    &buffers.query,
                    segment.query_offset_words,
                    segment.query_words,
                ),
                buffer_binding(
                    2,
                    &buffers.output,
                    segment.answer_offset_words,
                    segment.answer_words,
                ),
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffers.uniforms,
                        offset: uniform_offset,
                        size: wgpu::BufferSize::new(DIMS_UNIFORM_BYTES),
                    }),
                },
            ],
        });
        // The plan stored this segment's dispatch (tile-derived thread
        // count included); issuing it verbatim keeps plan and execution in
        // lockstep.
        let (x, y) = (segment.workgroups_x, segment.workgroups_y);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("emvp-packed-answer-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(x, y, 1);
        Ok(())
    }

    /// Allocates or leases scratch, encodes uploads, and submits a
    /// preflighted plan whose answers will reconstruct into a caller-owned
    /// arena.
    ///
    /// The submit half of the split packed API, paired with
    /// [`Self::finish_packed_into_at`]: the device pipeline and scratch
    /// pool, plus the flat-arena layout the into-write streams against
    /// per-job kernel shapes and engine-entry indices. The per-job segment
    /// coverage of the plan must match the jobs — the engine's plan phase
    /// guarantees this by construction — so the checks along the write
    /// path are only fail-closed.
    ///
    /// # Errors
    ///
    /// Returns allocation, upload, encoding, or submission failures, and
    /// [`GpuError::LengthMismatch`] when the plan's segment coverage does
    /// not match the jobs' shapes.
    pub fn begin_packed_into<const MODULUS: u32, Q: QueryValues<MODULUS>>(
        &self,
        jobs: &[PackedJob<'_, MODULUS, Q>],
        plan: &PackedPlan,
    ) -> Result<PackedFlightInto<MODULUS>, GpuError> {
        let shapes = job_shapes(jobs)?;
        let mut timings = PackedTimings::default();
        let prepare_start = Instant::now();
        let mut scratch = lock_recovered(&self.packed_scratch_pool)
            .pop()
            .unwrap_or(PackedScratch { chunks: Vec::new() });
        scratch.prepare(&self.device, &plan.chunks, plan.uniform_alignment)?;
        timings.prepare_buffers = prepare_start.elapsed();

        let upload_start = Instant::now();
        self.upload_packed_inputs(jobs, plan, &scratch)?;
        timings.encode_upload_queries = upload_start.elapsed();

        let submit_start = Instant::now();
        let encoder = self.encode_packed_commands(jobs, plan, &scratch)?;
        let submission = self.queue.submit([encoder.finish()]);
        timings.dispatch_submit = submit_start.elapsed();

        let mut flight_chunks = Vec::with_capacity(plan.chunks.len());
        for (chunk, buffers) in plan.chunks.iter().zip(&scratch.chunks) {
            let (sender, receiver) = std::sync::mpsc::channel();
            buffers
                .staging
                .slice(0..chunk.answer_words as u64 * WORD_BYTES)
                .map_async(wgpu::MapMode::Read, move |result| {
                    drop(sender.send(result.map_err(|error| error.to_string())));
                });
            flight_chunks.push(FlightChunk {
                plan: chunk.clone(),
                receiver,
            });
        }
        Ok(PackedFlightInto {
            scratch: Some(scratch),
            chunks: flight_chunks,
            shapes,
            entries: jobs.iter().map(|job| job.entry).collect(),
            submission,
            stats: plan.stats,
            timings,
        })
    }

    /// Executes a planned packed operation end to end, streaming the
    /// answers into a caller-owned arena at *absolute* per-job offsets.
    ///
    /// The engine's execute step: job `k`'s answers land at
    /// `job_base_words[k]`, query-major inside the job — exactly what
    /// `reconstruct_packed_into_at` and `segment_dest_range_at`
    /// stream — so a packed flight can land inside a larger engine arena
    /// whose entries interleave CPU- and GPU-tier answers. Everything else
    /// is plan → submit → wait → reconstruct with the pooled
    /// `PackedScratch` semantics of the split API.
    ///
    /// Before any device work or mutation the call checks that every job's
    /// absolute run (`job_base_words[k] + queries_k * rows_k * s_k`) fits
    /// inside `dest`, rejecting with [`GpuError::Capacity`] otherwise. The
    /// caller owns the invariant that the per-job runs are disjoint inside
    /// `dest`; the engine's plan phase guarantees this by construction with
    /// prefix-sum offsets, and overlapping bases would silently overwrite
    /// answers rather than fault.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::LengthMismatch`] when `job_base_words` does not
    /// have one slot per packed job, [`GpuError::Capacity`] when a job's
    /// absolute run escapes `dest`, and the plan, upload, submission, wait,
    /// or readback failures of the split API otherwise.
    pub fn execute_packed_into_at<const MODULUS: u32, Q: QueryValues<MODULUS>>(
        &self,
        plan: &PackedPlan,
        jobs: &[PackedJob<'_, MODULUS, Q>],
        job_base_words: &[usize],
        dest: &mut [FieldElement<MODULUS>],
    ) -> Result<(PackedStats, PackedTimings), GpuError> {
        // Fail-closed layout checks before any device work or mutation.
        if job_base_words.len() != jobs.len() {
            return Err(GpuError::LengthMismatch {
                name: "packed answer bases",
                expected: jobs.len(),
                actual: job_base_words.len(),
            });
        }
        let shapes = job_shapes(jobs)?;
        let dest_len = dest.len();
        for (base, (job, shape)) in job_base_words.iter().zip(jobs.iter().zip(&shapes)) {
            let words_per_query = shape
                .rows
                .checked_mul(shape.s)
                .ok_or(GpuError::DimensionOverflow)?;
            let end = base
                .checked_add(
                    job.queries
                        .len()
                        .checked_mul(words_per_query)
                        .ok_or(GpuError::DimensionOverflow)?,
                )
                .ok_or(GpuError::DimensionOverflow)?;
            if end > dest_len {
                return Err(GpuError::Capacity {
                    required: end,
                    available: dest_len,
                });
            }
        }
        let flight = self.begin_packed_into(jobs, plan)?;
        self.finish_packed_into_at(flight, job_base_words, dest)
    }

    /// Waits for a submitted into-flight's staging mappings and returns the
    /// elapsed wait duration, so the finish step owns the poll and
    /// mapping-callback semantics.
    fn wait_packed_into_flight<const MODULUS: u32>(
        &self,
        flight: &PackedFlightInto<MODULUS>,
    ) -> Result<Duration, GpuError> {
        let wait_start = Instant::now();
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: Some(flight.submission.clone()),
                timeout: None,
            })
            .map_err(|error| GpuError::Submission(error.to_string()))?;
        for chunk in &flight.chunks {
            chunk
                .receiver
                .recv()
                .map_err(|_disconnected| {
                    GpuError::Map("mapping callback result unavailable".into())
                })?
                .map_err(GpuError::Map)?;
        }
        Ok(wait_start.elapsed())
    }

    /// Waits for a submitted packed operation and streams its answers into
    /// `dest` at *absolute* per-job offsets.
    ///
    /// The finish half of the split packed API, paired with
    /// [`Self::begin_packed_into`]: same wait and scratch-pool
    /// semantics, with the staged words streaming via
    /// `reconstruct_packed_into_at` to
    /// `job_base_words[k] + query_start * rows * s` per segment.
    /// `job_base_words` must have one slot per packed job; a segment's
    /// range escaping `dest` rejects the call.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::LengthMismatch`] when `job_base_words` does not
    /// match the flight's job count, and polling, mapping, or noncanonical
    /// readback failures otherwise.
    pub fn finish_packed_into_at<const MODULUS: u32>(
        &self,
        mut flight: PackedFlightInto<MODULUS>,
        job_base_words: &[usize],
        dest: &mut [FieldElement<MODULUS>],
    ) -> Result<(PackedStats, PackedTimings), GpuError> {
        if job_base_words.len() != flight.entries.len() {
            return Err(GpuError::LengthMismatch {
                name: "packed answer bases",
                expected: flight.entries.len(),
                actual: job_base_words.len(),
            });
        }
        flight.timings.wait_readback = self.wait_packed_into_flight(&flight)?;

        let reconstruct_start = Instant::now();
        let reconstruct = reconstruct_flight_into_at(&flight, job_base_words, dest);
        flight.timings.reconstruct = reconstruct_start.elapsed();
        reconstruct?;
        let Some(scratch) = flight.scratch.take() else {
            return Err(GpuError::Map("packed scratch unavailable".into()));
        };
        lock_recovered(&self.packed_scratch_pool).push(scratch);
        Ok((flight.stats, flight.timings))
    }
}

/// Maps each chunk's staging buffer and streams its words into `dest` via
/// `reconstruct_packed_into_at`, unmapping every chunk before returning.
fn reconstruct_flight_into_at<const MODULUS: u32>(
    flight: &PackedFlightInto<MODULUS>,
    job_base_words: &[usize],
    dest: &mut [FieldElement<MODULUS>],
) -> Result<(), GpuError> {
    let Some(scratch) = flight.scratch.as_ref() else {
        return Err(GpuError::Map("packed scratch unavailable".into()));
    };
    for (flight_chunk, buffers) in flight.chunks.iter().zip(&scratch.chunks) {
        let data = buffers
            .staging
            .slice(0..flight_chunk.plan.answer_words as u64 * WORD_BYTES)
            .get_mapped_range()
            .map_err(|error| GpuError::Map(error.to_string()))?;
        let result = reconstruct_packed_into_at::<MODULUS>(
            std::slice::from_ref(&flight_chunk.plan),
            &[&data[..]],
            job_base_words,
            &flight.shapes,
            dest,
        );
        drop(data);
        buffers.staging.unmap();
        result?;
    }
    Ok(())
}

fn write_query_segment<const MODULUS: u32, Q: QueryValues<MODULUS>>(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    offset: u64,
    words: usize,
    queries: &[Q],
) -> Result<(), GpuError> {
    let bytes = u64::try_from(words)
        .ok()
        .and_then(|count| count.checked_mul(WORD_BYTES))
        .ok_or(GpuError::DimensionOverflow)?;
    let size = wgpu::BufferSize::new(bytes).ok_or(GpuError::DimensionOverflow)?;
    let Some(mut view) = queue.write_buffer_with(buffer, offset, size) else {
        return Err(GpuError::Submission(
            "the queue refused packed query staging memory".into(),
        ));
    };
    let (word_bytes, _tail) = view.slice(..).into_chunks::<{ WORD_BYTES as usize }>();
    word_bytes.write_iter(
        queries
            .iter()
            .flat_map(QueryValues::values)
            .map(|value| value.to_raw().to_le_bytes()),
    );
    Ok(())
}

const fn buffer_binding(
    binding: u32,
    buffer: &wgpu::Buffer,
    offset_words: usize,
    words: usize,
) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer,
            offset: offset_words as u64 * WORD_BYTES,
            size: wgpu::BufferSize::new(words as u64 * WORD_BYTES),
        }),
    }
}

#[cfg(test)]
mod tests {
    use prime_field_layer::PrimeField;

    use super::{
        ChunkPlan, DIMS_UNIFORM_BYTES, GpuError, PackedPlan, Segment, Shape, plan_chunks,
        reconstruct_packed_into_at, segment_dest_range_at,
    };

    #[test]
    fn planner_splits_only_between_queries_and_preserves_jobs() {
        let shapes = [
            Shape {
                n: 16,
                b: 2,
                rows: 3,
                s: 8,
            },
            Shape {
                n: 8,
                b: 2,
                rows: 2,
                s: 4,
            },
        ];
        let chunks = plan_chunks(MODULUS, &shapes, &[9, 3], 256, 96, 16).unwrap();
        let segments: Vec<_> = chunks.iter().flat_map(|chunk| &chunk.segments).collect();
        assert_eq!(
            segments
                .iter()
                .map(|segment| segment.query_count)
                .sum::<usize>(),
            12
        );
        assert!(
            segments
                .iter()
                .all(|segment| segment.query_words % shapes[segment.job].n == 0)
        );
        assert!(segments.windows(2).all(|pair| {
            pair[0].job != pair[1].job
                || pair[1].query_start == pair[0].query_start + pair[0].query_count
        }));
    }

    #[test]
    fn planner_rejects_one_query_larger_than_a_binding() {
        let error = plan_chunks(
            MODULUS,
            &[Shape {
                n: 65,
                b: 1,
                rows: 1,
                s: 1,
            }],
            &[1],
            256,
            64,
            16,
        )
        .unwrap_err();
        assert!(matches!(error, super::GpuError::UploadTooLarge { .. }));
    }

    #[test]
    fn planner_aligns_every_binding_range() {
        let chunks = plan_chunks(
            MODULUS,
            &[Shape {
                n: 6,
                b: 2,
                rows: 1,
                s: 3,
            }],
            &[20],
            64,
            24,
            8,
        )
        .unwrap();
        for segment in chunks.iter().flat_map(|chunk| &chunk.segments) {
            assert_eq!(segment.query_offset_words % 8, 0);
            assert_eq!(segment.answer_offset_words % 8, 0);
        }
    }

    // The deployment field, shared with the single-batch host-reconstruction
    // fixtures in `super::super`'s tests: p = 998244353, and the Montgomery
    // image of `v` is `v * 301989884 mod p`.
    const MODULUS: u32 = 998_244_353;

    /// Hand-built segment with the given destination-relevant fields; the
    /// device-side fields are zero because the layout helpers never read
    /// them.
    fn seg(
        job: usize,
        query_start: usize,
        query_count: usize,
        answer_offset_words: usize,
        answer_words: usize,
    ) -> Segment {
        Segment {
            job,
            query_start,
            query_count,
            query_offset_words: 0,
            answer_offset_words,
            query_words: 0,
            answer_words,
            tile: 1,
            workgroups_x: 0,
            workgroups_y: 0,
            uniform: [0; DIMS_UNIFORM_BYTES as usize],
        }
    }

    fn chunk(segments: Vec<Segment>, answer_words: usize) -> ChunkPlan {
        ChunkPlan {
            segments,
            query_words: 0,
            answer_words,
        }
    }

    /// The destination range of a segment must be the running prefix of the
    /// preceding entries' word counts plus the segment's query-major offset
    /// inside its own entry, hand-computed here. The contiguous layout of a
    /// packed subset is exactly the absolute-base form at the prefix sums.
    #[test]
    fn segment_dest_ranges_match_hand_computed_layouts() {
        // Single entry, whole-entry segment.
        assert_eq!(segment_dest_range_at(&[0], 0, 0, 1, 2, 3).unwrap(), 0..6);
        // Multiple entries: job 1 starts after entry 0's 12 words; job 2
        // starts after 12 + 6 = 18 and its second query sits one
        // words-per-query (rows 1 * s 2) into the entry, at word 22.
        let entry_words = [12, 6, 8];
        let prefix = |job: usize| entry_words[..job].iter().sum::<usize>();
        assert_eq!(
            segment_dest_range_at(&[prefix(1)], 0, 0, 3, 1, 2).unwrap(),
            12..18
        );
        assert_eq!(
            segment_dest_range_at(&[prefix(2)], 0, 1, 1, 2, 2).unwrap(),
            22..26
        );
        // A query_start offset inside one entry.
        assert_eq!(segment_dest_range_at(&[0], 0, 2, 1, 2, 2).unwrap(), 8..12);
        // A segment naming a job outside the base table is rejected
        // fail-closed.
        assert!(matches!(
            segment_dest_range_at(&[0], 1, 0, 1, 2, 2),
            Err(GpuError::LengthMismatch {
                name: "packed segment job",
                ..
            })
        ));
    }

    /// Canonical staged words must stream into the flat arena exactly at
    /// the hand-computed segment ranges, query-major inside each entry,
    /// across a multi-chunk plan, with no per-query collection.
    #[test]
    fn reconstruct_packed_into_at_accepts_canonical_words_in_arena_layout() {
        let field = PrimeField::<MODULUS>::new();
        // Job 0: two queries, rows 2 * s 1 = 2 words each. Job 1: one
        // query, rows 1 * s 2 = 2 words. Segment runs sit back to back in
        // one chunk, and split across two chunks in the multi-chunk case.
        let shapes = [
            Shape {
                n: 2,
                b: 2,
                rows: 2,
                s: 1,
            },
            Shape {
                n: 2,
                b: 2,
                rows: 1,
                s: 2,
            },
        ];
        // The contiguous packed-subset layout is the absolute-base form at
        // the prefix sums [0, 4] of the entry words [4, 2].
        let entry_bases = [0, 4];
        // The Montgomery images of 17, 39, 23, 53 (job 0's query-major
        // answers) and 4, 9 (job 1's answer), hand-derived as in the
        // single-batch fixtures.
        let golden = [
            142_606_263,
            796_917_593,
            956_301_214,
            33_554_204,
            209_715_183,
            721_420_250,
        ];
        // Cross-check the hand-derived words against the field itself, so a
        // broken fixture cannot pass silently.
        for (&word, sum) in golden.iter().zip([17_u32, 39, 23, 53, 4, 9]) {
            assert_eq!(word, field.element_u32(sum).to_raw());
        }
        let bytes = |words: &[u32]| -> Vec<u8> {
            words.iter().flat_map(|word| word.to_le_bytes()).collect()
        };

        // One chunk, two segments: job 0's run at word offset 0, job 1's at
        // word offset 4.
        let single = chunk(vec![seg(0, 0, 2, 0, 4), seg(1, 0, 1, 4, 2)], 6);
        let mut dest = vec![field.element_u32(MODULUS - 1); 6];
        reconstruct_packed_into_at::<MODULUS>(
            std::slice::from_ref(&single),
            &[&bytes(&golden)],
            &entry_bases,
            &shapes,
            &mut dest,
        )
        .unwrap();
        let expected: Vec<_> = [17_u32, 39, 23, 53, 4, 9]
            .iter()
            .map(|&sum| field.element_u32(sum))
            .collect();
        assert_eq!(dest, expected);

        // The same layout split across two chunks: job 0's segment in the
        // first chunk, job 1's continuation in the second.
        let first = chunk(vec![seg(0, 0, 2, 0, 4)], 4);
        let second = chunk(vec![seg(1, 0, 1, 0, 2)], 2);
        let mut dest = vec![field.element_u32(MODULUS - 1); 6];
        reconstruct_packed_into_at::<MODULUS>(
            &[first, second],
            &[&bytes(&golden[..4]), &bytes(&golden[4..])],
            &entry_bases,
            &shapes,
            &mut dest,
        )
        .unwrap();
        assert_eq!(dest, expected);
    }

    /// A noncanonical staged word is rejected precisely instead of being
    /// canonicalized, and a destination that cannot hold the first
    /// segment's range is rejected before any slot is written.
    #[test]
    fn reconstruct_packed_into_at_rejects_noncanonical_and_mismatched() {
        let field = PrimeField::<MODULUS>::new();
        let shapes = [Shape {
            n: 2,
            b: 2,
            rows: 2,
            s: 1,
        }];
        let entry_bases = [0];
        let golden = [142_606_263_u32, 796_917_593, 956_301_214, 33_554_204];
        let bytes = |words: &[u32]| -> Vec<u8> {
            words.iter().flat_map(|word| word.to_le_bytes()).collect()
        };
        let plan = chunk(vec![seg(0, 0, 2, 0, 4)], 4);

        // The third word is the modulus itself: not a canonical residue.
        let mut corrupt = golden;
        corrupt[2] = MODULUS;
        let mut dest = vec![field.element_u32(MODULUS - 1); 4];
        assert!(matches!(
            reconstruct_packed_into_at::<MODULUS>(
                std::slice::from_ref(&plan),
                &[&bytes(&corrupt)],
                &entry_bases,
                &shapes,
                &mut dest,
            ),
            Err(GpuError::NonCanonicalWord {
                word: MODULUS,
                modulus: MODULUS,
            })
        ));

        // A destination shorter than the first segment's range is rejected
        // before any write: every slot keeps its poison sentinel.
        let mut dest = vec![field.element_u32(MODULUS - 1); 2];
        assert!(matches!(
            reconstruct_packed_into_at::<MODULUS>(
                std::slice::from_ref(&plan),
                &[&bytes(&golden)],
                &entry_bases,
                &shapes,
                &mut dest,
            ),
            Err(GpuError::LengthMismatch {
                name: "packed answer arena",
                ..
            })
        ));
        assert!(
            dest.iter()
                .all(|slot| *slot == field.element_u32(MODULUS - 1))
        );

        // Chunk plans and staging views must be parallel.
        let plans = [chunk(vec![seg(0, 0, 2, 0, 4)], 4), chunk(Vec::new(), 0)];
        assert!(matches!(
            reconstruct_packed_into_at::<MODULUS>(
                &plans,
                &[&bytes(&golden)],
                &entry_bases,
                &shapes,
                &mut dest,
            ),
            Err(GpuError::LengthMismatch {
                name: "packed staging views",
                ..
            })
        ));
    }

    /// The absolute-base destination ranges must be the hand-computed
    /// `[base + query_start * words_per_query, .. + query_count *
    /// words_per_query)` runs, independent of any contiguous entry table,
    /// and a job outside the base table is rejected fail-closed.
    #[test]
    fn segment_dest_range_at_matches_hand_computed_absolute_layouts() {
        // Bases interleave CPU-tier entries in a larger engine arena: job 0
        // answers start at word 100, job 1's at word 200.
        let bases = [100, 200];
        // Job 0 (rows 2, s 3): the whole one-query run sits at 100..106.
        assert_eq!(
            segment_dest_range_at(&bases, 0, 0, 1, 2, 3).unwrap(),
            100..106
        );
        // Job 1 (rows 1, s 2): its second query starts one words-per-query
        // (2) into the entry, at 202.
        assert_eq!(
            segment_dest_range_at(&bases, 1, 1, 2, 1, 2).unwrap(),
            202..206
        );
        // A job naming a base outside the table is rejected exactly like
        // the contiguous variant rejects an out-of-table entry.
        assert!(matches!(
            segment_dest_range_at(&bases, 2, 0, 1, 2, 3),
            Err(GpuError::LengthMismatch {
                name: "packed segment job",
                expected: 2,
                actual: 2,
            })
        ));
        // Checked overflow on absurd inputs.
        assert!(matches!(
            segment_dest_range_at(&[usize::MAX], 0, 1, 1, 2, 2),
            Err(GpuError::DimensionOverflow)
        ));
    }

    /// Canonical staged words must stream to the *absolute* per-job bases:
    /// jobs landing at sparse offsets inside a larger poisoned arena write
    /// exactly their query-major runs and leave every other word untouched.
    #[test]
    fn reconstruct_packed_into_at_streams_to_absolute_bases() {
        let field = PrimeField::<MODULUS>::new();
        // Same shapes and golden words as the contiguous fixture: job 0 has
        // two queries of 2 words, job 1 one query of 2 words.
        let shapes = [
            Shape {
                n: 2,
                b: 2,
                rows: 2,
                s: 1,
            },
            Shape {
                n: 2,
                b: 2,
                rows: 1,
                s: 2,
            },
        ];
        let golden = [
            142_606_263_u32,
            796_917_593,
            956_301_214,
            33_554_204,
            209_715_183,
            721_420_250,
        ];
        let staged = golden
            .iter()
            .flat_map(|word| word.to_le_bytes())
            .collect::<Vec<u8>>();
        // Absolute bases straddling a poisoned gap: job 0 at word 8, job 1
        // at word 100. The arena is 110 words.
        let bases = [8, 100];
        let sentinel = field.element_u32(MODULUS - 1);
        let mut dest = vec![sentinel; 110];
        let plan = chunk(vec![seg(0, 0, 2, 0, 4), seg(1, 0, 1, 4, 2)], 6);
        reconstruct_packed_into_at::<MODULUS>(
            std::slice::from_ref(&plan),
            &[&staged],
            &bases,
            &shapes,
            &mut dest,
        )
        .unwrap();
        // Job 0's query-major run landed at 8..12, job 1's at 100..102.
        for (index, &word) in [17_u32, 39, 23, 53].iter().enumerate() {
            assert_eq!(dest[8 + index], field.element_u32(word));
        }
        assert_eq!(dest[100], field.element_u32(4));
        assert_eq!(dest[101], field.element_u32(9));
        // Every word outside the two runs keeps the poison sentinel,
        // including the gap between the jobs.
        assert!(dest[..8].iter().all(|slot| *slot == sentinel));
        assert!(dest[12..100].iter().all(|slot| *slot == sentinel));
        assert!(dest[102..].iter().all(|slot| *slot == sentinel));

        // The same golden words through the packed-subset-contiguous bases
        // ([0, 4], the prefix sums of entry words [4, 2]) must reproduce
        // the contiguous layout.
        let contiguous = [0, 4];
        let mut flat_dest = vec![sentinel; 6];
        reconstruct_packed_into_at::<MODULUS>(
            std::slice::from_ref(&plan),
            &[&staged],
            &contiguous,
            &shapes,
            &mut flat_dest,
        )
        .unwrap();
        let expected: Vec<_> = [17_u32, 39, 23, 53, 4, 9]
            .iter()
            .map(|&sum| field.element_u32(sum))
            .collect();
        assert_eq!(flat_dest, expected);

        // A job whose absolute run escapes the destination is rejected
        // fail-closed. Job 0's earlier segment in the same chunk had
        // already streamed (its run is written), but nothing beyond it is
        // touched: the caller treats the whole arena as failed.
        let mut dest = vec![sentinel; 110];
        assert!(matches!(
            reconstruct_packed_into_at::<MODULUS>(
                std::slice::from_ref(&plan),
                &[&staged],
                &[8, 109],
                &shapes,
                &mut dest,
            ),
            Err(GpuError::LengthMismatch {
                name: "packed answer arena",
                ..
            })
        ));
        assert!(dest[12..].iter().all(|slot| *slot == sentinel));
        assert_eq!(dest[8], field.element_u32(17));
    }

    /// A plan's segment coverage must account exactly for each job's
    /// `queries * rows * s` arena words, summed over multi-chunk
    /// continuations: this is the word accounting the engine's entry
    /// offsets and the into-write's per-segment destinations rely on.
    #[test]
    fn packed_plan_segment_coverage_accounts_for_every_job() {
        let shapes = [
            Shape {
                n: 16,
                b: 2,
                rows: 3,
                s: 8,
            },
            Shape {
                n: 8,
                b: 2,
                rows: 2,
                s: 4,
            },
        ];
        let chunks = plan_chunks(MODULUS, &shapes, &[9, 3], 256, 96, 16).unwrap();
        let mut totals = vec![0_usize; shapes.len()];
        for segment in chunks.iter().flat_map(|chunk| &chunk.segments) {
            let shape = &shapes[segment.job];
            totals[segment.job] += segment.query_count * shape.rows * shape.s;
        }
        // Job 0: 9 queries * 3 rows * 8 blocks; job 1: 3 * 2 * 4.
        assert_eq!(totals, vec![216, 24]);

        // A job whose queries continue across chunks accumulates both
        // segments' query counts: rows 2 * s 2 = 4 words per query, split
        // 5 + 5 queries across two chunks by a 32-word buffer cap.
        let one_job = [Shape {
            n: 4,
            b: 2,
            rows: 2,
            s: 2,
        }];
        let split = plan_chunks(MODULUS, &one_job, &[10], 32, 96, 8).unwrap();
        assert_eq!(split.iter().flat_map(|chunk| &chunk.segments).count(), 2);
        let mut split_total = 0_usize;
        for segment in split.iter().flat_map(|chunk| &chunk.segments) {
            split_total += segment.query_count * 2 * 2;
        }
        assert_eq!(split_total, 40);
    }

    /// The planner's per-job segment counts must agree with the planned
    /// dispatch segments, so callers can report per-entry segment counts
    /// before any device work.
    #[test]
    fn packed_plan_segment_counts_match_dispatch_segments() {
        let shapes = [
            Shape {
                n: 16,
                b: 2,
                rows: 3,
                s: 8,
            },
            Shape {
                n: 8,
                b: 2,
                rows: 2,
                s: 4,
            },
        ];
        let chunks = plan_chunks(MODULUS, &shapes, &[9, 3], 256, 96, 16).unwrap();
        let mut plan = PackedPlan {
            chunks,
            uniform_alignment: 64,
            stats: super::PackedStats::default(),
        };
        let expected: Vec<(usize, usize)> = (0..shapes.len())
            .map(|job| {
                (
                    job,
                    plan.chunks
                        .iter()
                        .flat_map(|chunk| &chunk.segments)
                        .filter(|segment| segment.job == job)
                        .count(),
                )
            })
            .collect();
        assert_eq!(plan.segment_counts(), expected);
        // A split plan reports two segments for the continued job.
        let one_job = [Shape {
            n: 4,
            b: 2,
            rows: 2,
            s: 2,
        }];
        plan.chunks = plan_chunks(MODULUS, &one_job, &[10], 32, 96, 8).unwrap();
        assert_eq!(plan.segment_counts(), vec![(0, 2)]);
    }
}
