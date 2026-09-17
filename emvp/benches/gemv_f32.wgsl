// Tuned float32 matrix-vector product: the cleartext baseline an unencrypted
// f32 deployment would run on the same device as the EMVP answer kernel.
//
// For each output `q * rows + r` (query-major, matching the EMVP answer
// arena) the kernel computes
//
//     output[q * rows + r] = sum_{t < ell} A[r][t] * X[q][t]      in f32,
//
// where `X` holds the `batch` query vectors (`batch x ell`, row-major) and
// `A` is the `rows x ell` matrix stored COLUMN-MAJOR on the device, i.e.
// `matrix_words[t * rows + r] = A[r][t]`. All indices stay inside u32
// because the host caps `rows * ell` and `batch * ell` below 2^32 (at the
// benchmark's largest shape, rows * ell = 16384 * 4096 ~ 2^26).
//
// # Thread-per-row streaming design
//
// This is the same access pattern as the EMVP answer kernel
// (`src/gpu/answer.wgsl`): one thread per (query tile, row), adjacent lanes
// on adjacent rows, so at one loop step `t` a warp reads 128 consecutive
// bytes of the column-major matrix and every lane reads the SAME query word,
// which the hardware serves as a single broadcast load. The matrix is the
// only per-lane stream; the query costs one broadcast per step regardless of
// the workgroup width. Each thread's four loads per step are four
// consecutive columns, four independent sequential readers.
//
// A workgroup-per-output cooperative reduction (the previous baseline) was
// measured slower on the reference GTX 1060: it re-read the query vector per
// lane (five loads per four MACs instead of one plus a broadcast), spread
// each scalar load across four cache lines, and paid a seven-round barrier
// reduction after only `ell / 512` loop iterations. The thread-per-row form
// has no barriers and no shared memory; occupancy comes from
// `ceil(batch / tile) * rows` threads, which the tiled entry point keeps
// above 10^5 at every benchmarked shape.
//
// `main_q4` tiles four queries per thread so the matrix streams
// `ceil(batch / 4)` times instead of once per query; tail queries are
// clamped to the last query and their stores individually guarded,
// mirroring the answer kernel's tiled entry point.
//
// Summation order differs from a sequential CPU accumulation, so results
// are not bit-identical to a naive CPU GEMV; the benchmark pins them with a
// loose tolerance instead. This kernel is a performance baseline only - no
// cryptographic or constant-time property is claimed.

// Threads per workgroup. An override, not a const: the WGSL spec allows a
// pipeline-overridable constant in `@workgroup_size`, and the bench
// supplies its `F32_WORKGROUP_SIZE` here, so sweeping workgroup sizes is a
// host-side constant change with no shader edit.
override WORKGROUP_SIZE: u32 = 256u;

// Per-call dimensions, little-endian u32 words in a 16-byte uniform buffer.
struct Dims {
    rows: u32,
    batch: u32,
    ell: u32,
    workgroups_x: u32,
}

@group(0) @binding(0) var<storage, read> matrix_words: array<f32>;
@group(0) @binding(1) var<storage, read> query_words: array<f32>;
@group(0) @binding(2) var<storage, read_write> output_words: array<f32>;
@group(0) @binding(3) var<uniform> dims: Dims;

// Linear thread index over the host's 2D dispatch grid.
fn thread_index(local_id: vec3<u32>, workgroup: vec3<u32>) -> u32 {
    return ((workgroup.y * dims.workgroups_x) + workgroup.x) * WORKGROUP_SIZE + local_id.x;
}

@compute
@workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup: vec3<u32>,
) {
    let index = thread_index(local_id, workgroup);
    if (index >= dims.batch * dims.rows) {
        return;
    }
    let row = index % dims.rows;
    let query = index / dims.rows;
    let query_base = query * dims.ell;

    // Four-wide main loop over the multiples of four, then a scalar tail
    // for `ell mod 4` columns. Two accumulator pairs break the serial
    // dependency on every add.
    let rows1 = dims.rows;
    let rows2 = 2u * dims.rows;
    let rows3 = 3u * dims.rows;
    let rows4 = 4u * dims.rows;
    let vector_end = dims.ell - (dims.ell % 4u);
    var acc_a = 0.0;
    var acc_b = 0.0;
    var column = row;
    var offset = 0u;
    while (offset < vector_end) {
        let m0 = matrix_words[column];
        let m1 = matrix_words[column + rows1];
        let m2 = matrix_words[column + rows2];
        let m3 = matrix_words[column + rows3];
        let q0 = query_words[query_base + offset];
        let q1 = query_words[query_base + offset + 1u];
        let q2 = query_words[query_base + offset + 2u];
        let q3 = query_words[query_base + offset + 3u];
        acc_a = acc_a + (m0 * q0 + m1 * q1);
        acc_b = acc_b + (m2 * q2 + m3 * q3);
        column = column + rows4;
        offset = offset + 4u;
    }
    while (offset < dims.ell) {
        acc_a = acc_a + matrix_words[column] * query_words[query_base + offset];
        column = column + rows1;
        offset = offset + 1u;
    }
    output_words[query * dims.rows + row] = acc_a + acc_b;
}

// Four-query tiled variant of `main`: one thread computes the same row's
// output for a tile of four consecutive queries, so every matrix word load
// feeds four accumulators and the matrix is streamed `ceil(batch / 4)`
// times instead of once per query. Query words stay broadcast loads: all
// lanes of a warp share the tile.
@compute
@workgroup_size(WORKGROUP_SIZE)
fn main_q4(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup: vec3<u32>,
) {
    let index = thread_index(local_id, workgroup);
    let total = ((dims.batch + 3u) / 4u) * dims.rows;
    if (index >= total) {
        return;
    }
    let row = index % dims.rows;
    let first_query = (index / dims.rows) * 4u;
    let last = dims.batch - 1u;
    let query_base0 = min(first_query, last) * dims.ell;
    let query_base1 = min(first_query + 1u, last) * dims.ell;
    let query_base2 = min(first_query + 2u, last) * dims.ell;
    let query_base3 = min(first_query + 3u, last) * dims.ell;

    let rows1 = dims.rows;
    let rows2 = 2u * dims.rows;
    let rows3 = 3u * dims.rows;
    let rows4 = 4u * dims.rows;
    let vector_end = dims.ell - (dims.ell % 4u);
    var acc = vec4<f32>(0.0, 0.0, 0.0, 0.0);
    var column = row;
    var offset = 0u;
    while (offset < vector_end) {
        let m0 = matrix_words[column];
        let m1 = matrix_words[column + rows1];
        let m2 = matrix_words[column + rows2];
        let m3 = matrix_words[column + rows3];
        // One vec4 per query: its four columns of the tile's query vector.
        let x0 = vec4<f32>(
            query_words[query_base0 + offset],
            query_words[query_base0 + offset + 1u],
            query_words[query_base0 + offset + 2u],
            query_words[query_base0 + offset + 3u],
        );
        let x1 = vec4<f32>(
            query_words[query_base1 + offset],
            query_words[query_base1 + offset + 1u],
            query_words[query_base1 + offset + 2u],
            query_words[query_base1 + offset + 3u],
        );
        let x2 = vec4<f32>(
            query_words[query_base2 + offset],
            query_words[query_base2 + offset + 1u],
            query_words[query_base2 + offset + 2u],
            query_words[query_base2 + offset + 3u],
        );
        let x3 = vec4<f32>(
            query_words[query_base3 + offset],
            query_words[query_base3 + offset + 1u],
            query_words[query_base3 + offset + 2u],
            query_words[query_base3 + offset + 3u],
        );
        let m = vec4<f32>(m0, m1, m2, m3);
        acc = acc + vec4<f32>(dot(m, x0), dot(m, x1), dot(m, x2), dot(m, x3));
        column = column + rows4;
        offset = offset + 4u;
    }
    while (offset < dims.ell) {
        let m = matrix_words[column];
        acc = acc + m * vec4<f32>(
            query_words[query_base0 + offset],
            query_words[query_base1 + offset],
            query_words[query_base2 + offset],
            query_words[query_base3 + offset],
        );
        column = column + rows1;
        offset = offset + 1u;
    }

    let out0 = first_query * dims.rows + row;
    output_words[out0] = acc.x;
    if (first_query + 1u < dims.batch) {
        output_words[out0 + dims.rows] = acc.y;
    }
    if (first_query + 2u < dims.batch) {
        output_words[out0 + 2u * dims.rows] = acc.z;
    }
    if (first_query + 3u < dims.batch) {
        output_words[out0 + 3u * dims.rows] = acc.w;
    }
}
