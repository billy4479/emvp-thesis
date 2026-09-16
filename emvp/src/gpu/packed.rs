//! Packed multi-matrix execution used by [`crate::engine::AnswerEngine`].

use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use prime_field_layer::FieldElement;

use super::{
    DIMS_UNIFORM_BYTES, GpuAnswerer, GpuEncryptedMatrix, GpuError, MAX_ANSWER_WORDS,
    create_buffer_checked, dims_uniform_bytes, dispatch_grid, lock_recovered, staged_answer_word,
};
use crate::{AnswerMatrix, EncryptedQuery};

const WORD_BYTES: u64 = 4;

/// One complete engine job selected for device execution.
pub struct PackedJob<'a, const MODULUS: u32> {
    pub entry: usize,
    pub matrix: &'a GpuEncryptedMatrix<MODULUS>,
    pub queries: &'a [EncryptedQuery<MODULUS>],
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
struct Segment {
    job: usize,
    query_start: usize,
    query_count: usize,
    query_offset_words: usize,
    answer_offset_words: usize,
    query_words: usize,
    answer_words: usize,
    workgroups_x: u32,
    workgroups_y: u32,
    uniform: [u8; DIMS_UNIFORM_BYTES as usize],
}

#[derive(Clone, Debug, Default, Eq, PartialEq)]
struct ChunkPlan {
    segments: Vec<Segment>,
    query_words: usize,
    answer_words: usize,
}

#[derive(Clone, Copy)]
struct Shape {
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

pub struct PackedFlight<const MODULUS: u32> {
    scratch: Option<PackedScratch>,
    chunks: Vec<FlightChunk>,
    jobs: Vec<JobResultMeta>,
    submission: wgpu::SubmissionIndex,
    stats: PackedStats,
    timings: PackedTimings,
}

impl<const MODULUS: u32> PackedFlight<MODULUS> {
    pub fn segment_counts(&self) -> Vec<(usize, usize)> {
        let mut counts = vec![0; self.jobs.len()];
        for segment in self.chunks.iter().flat_map(|chunk| &chunk.plan.segments) {
            counts[segment.job] += 1;
        }
        self.jobs
            .iter()
            .zip(counts)
            .map(|(job, count)| (job.entry, count))
            .collect()
    }
}

impl<const MODULUS: u32> Drop for PackedFlight<MODULUS> {
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

pub type PackedAnswers<const MODULUS: u32> = Vec<(usize, Vec<AnswerMatrix<MODULUS>>)>;
pub type PackedOutcome<const MODULUS: u32> =
    Result<(PackedAnswers<MODULUS>, PackedStats, PackedTimings), GpuError>;

struct JobResultMeta {
    entry: usize,
    instance_id: u128,
    rows: usize,
    blocks: usize,
    query_ids: Vec<u64>,
}

fn align_up(value: usize, alignment: usize) -> Result<usize, GpuError> {
    value
        .checked_add(alignment - 1)
        .map(|sum| sum / alignment * alignment)
        .ok_or(GpuError::DimensionOverflow)
}

fn plan_chunks(
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
                    let (workgroups_x, workgroups_y) = dispatch_grid(answer_words)?;
                    let uniform = dims_uniform_bytes(
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
    pub fn plan_packed<const MODULUS: u32>(
        &self,
        jobs: &[PackedJob<'_, MODULUS>],
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

    /// Allocates or leases scratch, encodes uploads, and submits a preflighted plan.
    ///
    /// # Errors
    ///
    /// Returns allocation, upload, encoding, or submission failures.
    pub fn begin_packed<const MODULUS: u32>(
        &self,
        jobs: &[PackedJob<'_, MODULUS>],
        plan: PackedPlan,
    ) -> Result<PackedFlight<MODULUS>, GpuError> {
        let mut timings = PackedTimings::default();
        let prepare_start = Instant::now();
        let mut scratch = lock_recovered(&self.packed_scratch_pool)
            .pop()
            .unwrap_or(PackedScratch { chunks: Vec::new() });
        scratch.prepare(&self.device, &plan.chunks, plan.uniform_alignment)?;
        let pipeline = self.answer_pipeline::<MODULUS>();
        timings.prepare_buffers = prepare_start.elapsed();

        let upload_start = Instant::now();
        self.upload_packed_inputs(jobs, &plan, &scratch)?;
        timings.encode_upload_queries = upload_start.elapsed();

        let submit_start = Instant::now();
        let encoder = self.encode_packed_commands(jobs, &plan, &scratch, &pipeline)?;
        let submission = self.queue.submit([encoder.finish()]);
        timings.dispatch_submit = submit_start.elapsed();

        let mut flight_chunks = Vec::with_capacity(plan.chunks.len());
        for (chunk, buffers) in plan.chunks.into_iter().zip(&scratch.chunks) {
            let (sender, receiver) = std::sync::mpsc::channel();
            buffers
                .staging
                .slice(0..chunk.answer_words as u64 * WORD_BYTES)
                .map_async(wgpu::MapMode::Read, move |result| {
                    drop(sender.send(result.map_err(|error| error.to_string())));
                });
            flight_chunks.push(FlightChunk {
                plan: chunk,
                receiver,
            });
        }
        let jobs = jobs
            .iter()
            .map(|job| {
                Ok(JobResultMeta {
                    entry: job.entry,
                    instance_id: job.matrix.instance_id,
                    rows: job.matrix.rows,
                    blocks: job.matrix.params.blocks()?,
                    query_ids: job.queries.iter().map(EncryptedQuery::query_id).collect(),
                })
            })
            .collect::<Result<_, GpuError>>()?;
        Ok(PackedFlight {
            scratch: Some(scratch),
            chunks: flight_chunks,
            jobs,
            submission,
            stats: plan.stats,
            timings,
        })
    }

    fn upload_packed_inputs<const MODULUS: u32>(
        &self,
        jobs: &[PackedJob<'_, MODULUS>],
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
                let (workgroups_x, _) = dispatch_grid(segment.answer_words)?;
                let uniform = dims_uniform_bytes(
                    job.matrix.columns,
                    job.matrix.params.block_size(),
                    job.matrix.params.blocks()?,
                    job.matrix.rows,
                    segment.query_count,
                    workgroups_x,
                )?;
                self.queue.write_buffer(
                    &buffers.uniforms,
                    segment_index as u64 * plan.uniform_alignment,
                    &uniform,
                );
            }
        }
        Ok(())
    }

    fn encode_packed_commands<const MODULUS: u32>(
        &self,
        jobs: &[PackedJob<'_, MODULUS>],
        plan: &PackedPlan,
        scratch: &PackedScratch,
        pipeline: &wgpu::ComputePipeline,
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
                    pipeline,
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

    fn encode_packed_segment<const MODULUS: u32>(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        job: &PackedJob<'_, MODULUS>,
        buffers: &ChunkScratch,
        segment: &Segment,
        uniform_offset: u64,
        pipeline: &wgpu::ComputePipeline,
    ) -> Result<(), GpuError> {
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
        let (x, y) = dispatch_grid(segment.answer_words)?;
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("emvp-packed-answer-pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(x, y, 1);
        Ok(())
    }

    /// Waits for and reconstructs a submitted packed operation.
    ///
    /// # Errors
    ///
    /// Returns polling, mapping, or noncanonical readback failures. Failed
    /// flights are unmapped and discarded instead of entering the pool.
    pub fn finish_packed<const MODULUS: u32>(
        &self,
        mut flight: PackedFlight<MODULUS>,
    ) -> PackedOutcome<MODULUS> {
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
        flight.timings.wait_readback = wait_start.elapsed();

        let reconstruct_start = Instant::now();
        let answers = reconstruct_flight(&flight)?;
        flight.timings.reconstruct = reconstruct_start.elapsed();
        let Some(scratch) = flight.scratch.take() else {
            return Err(GpuError::Map("packed scratch unavailable".into()));
        };
        lock_recovered(&self.packed_scratch_pool).push(scratch);
        Ok((answers, flight.stats, flight.timings))
    }
}

fn write_query_segment<const MODULUS: u32>(
    queue: &wgpu::Queue,
    buffer: &wgpu::Buffer,
    offset: u64,
    words: usize,
    queries: &[EncryptedQuery<MODULUS>],
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
            .flat_map(EncryptedQuery::values)
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

fn reconstruct_flight<const MODULUS: u32>(
    flight: &PackedFlight<MODULUS>,
) -> Result<PackedAnswers<MODULUS>, GpuError> {
    let Some(scratch) = &flight.scratch else {
        return Err(GpuError::Map("packed scratch unavailable".into()));
    };
    let mut values: Vec<Vec<Vec<FieldElement<MODULUS>>>> = flight
        .jobs
        .iter()
        .map(|job| vec![Vec::new(); job.query_ids.len()])
        .collect();
    for (flight_chunk, buffers) in flight.chunks.iter().zip(&scratch.chunks) {
        let data = buffers
            .staging
            .slice(0..flight_chunk.plan.answer_words as u64 * WORD_BYTES)
            .get_mapped_range()
            .map_err(|error| GpuError::Map(error.to_string()))?;
        let result = reconstruct_chunk(&flight_chunk.plan, &data, &mut values);
        drop(data);
        buffers.staging.unmap();
        result?;
    }
    Ok(flight
        .jobs
        .iter()
        .zip(values)
        .map(|(job, values)| {
            let answers = job
                .query_ids
                .iter()
                .zip(values)
                .map(|(&query_id, values)| {
                    AnswerMatrix::from_parts(
                        job.instance_id,
                        query_id,
                        values,
                        job.rows,
                        job.blocks,
                    )
                })
                .collect();
            (job.entry, answers)
        })
        .collect())
}

fn reconstruct_chunk<const MODULUS: u32>(
    plan: &ChunkPlan,
    data: &[u8],
    values: &mut [Vec<Vec<FieldElement<MODULUS>>>],
) -> Result<(), GpuError> {
    for segment in &plan.segments {
        let start = segment.answer_offset_words * WORD_BYTES as usize;
        let end = start + segment.answer_words * WORD_BYTES as usize;
        let words_per_query = segment.answer_words / segment.query_count;
        for (query_offset, query_bytes) in data[start..end]
            .chunks_exact(words_per_query * WORD_BYTES as usize)
            .enumerate()
        {
            values[segment.job][segment.query_start + query_offset] = query_bytes
                .chunks_exact(WORD_BYTES as usize)
                .map(staged_answer_word)
                .collect::<Result<Vec<_>, _>>()?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{Shape, plan_chunks};

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
        let chunks = plan_chunks(&shapes, &[9, 3], 256, 96, 16).unwrap();
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
}
