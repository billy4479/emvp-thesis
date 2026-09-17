// Tuned float32 matrix-vector product: the cleartext baseline an unencrypted
// f32 deployment would run on the same device as the EMVP answer kernel.
//
// For each output pair `g = q * rows + r` (query-major, matching the EMVP
// answer arena) the kernel computes
//
//     output[g] = sum_{t < ell} A[r][t] * X[q][t]      in f32,
//
// where `A` is the row-major matrix (`rows x ell`), `X` holds the `batch`
// query vectors (`batch x ell`), and all indices stay inside u32 because the
// host caps `rows * ell` and `batch * ell` below 2^32 (at the benchmark's
// largest shape, rows * ell = 16384 * 4096 ~ 2^26).
//
// # Workgroup-per-row cooperative design
//
// One whole workgroup computes one (query, row) output pair, so the thread
// count is `batch * rows * WORKGROUP_SIZE` instead of the `batch * rows` a
// thread-per-output kernel would launch. At rows = 4096 and batch = 1 a
// thread-per-row kernel would issue only 4096 threads, far below every
// discrete GPU's occupancy, and would flatter the EMVP comparison at small
// shapes; the cooperative layout keeps ~10^6 threads in flight at every
// benchmarked shape.
//
// Each thread accumulates a private f32 partial over its share of the row:
// the main loop strides through the row in chunks of `4 * WORKGROUP_SIZE`
// elements with thread `t` loading the four consecutive words at
// `base + 4 * t` .. `+ 3`, so every iteration step reads
// `4 * WORKGROUP_SIZE` consecutive f32 words per workgroup and the driver
// can merge each thread's four loads into single wide loads. A scalar
// strided tail covers `ell mod (4 * WORKGROUP_SIZE)` elements. The
// workgroup then tree-reduces its partials through workgroup-shared memory
// (`log2(256)` rounds, barrier after each) and thread 0 writes the result.
//
// Summation order differs from a sequential CPU accumulation, so results
// are not bit-identical to a naive CPU GEMV; the benchmark pins them with a
// loose tolerance instead. This kernel is a performance baseline only - no
// cryptographic or constant-time property is claimed.

// Must match the host WORKGROUP_SIZE. A module const (not an override) keeps
// the workgroup size a compile-time constant as required.
const WORKGROUP_SIZE: u32 = 256u;

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

// Per-thread partials reduced across the workgroup after the load loop.
var<workgroup> partials: array<f32, WORKGROUP_SIZE>;

@compute
@workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup: vec3<u32>,
) {
    // The host dispatches workgroups_x x workgroups_y workgroups over a
    // linear work count of `batch * rows`; every thread reconstructs its
    // output pair the same way the EMVP answer kernel reconstructs its
    // answer index.
    let pair = (workgroup.y * dims.workgroups_x) + workgroup.x;
    if (pair >= dims.batch * dims.rows) {
        return;
    }
    let row = pair % dims.rows;
    let query = pair / dims.rows;

    let matrix_base = row * dims.ell;
    let query_base = query * dims.ell;
    var accumulator = 0.0;
    // Four-wide main loop: thread `t` covers indices
    // `base + 4 * t .. base + 4 * t + 3` of one 4 * WORKGROUP_SIZE chunk, so
    // each thread's four loads per operand are consecutive words at a
    // 16-byte-aligned offset and merge into wide load instructions.
    let chunk_span = 4u * WORKGROUP_SIZE;
    let chunk_end = dims.ell - (dims.ell % chunk_span);
    var base = 0u;
    while (base < chunk_end) {
        let index = base + local_id.x * 4u;
        let m0 = matrix_words[matrix_base + index];
        let q0 = query_words[query_base + index];
        let m1 = matrix_words[matrix_base + index + 1u];
        let q1 = query_words[query_base + index + 1u];
        let m2 = matrix_words[matrix_base + index + 2u];
        let q2 = query_words[query_base + index + 2u];
        let m3 = matrix_words[matrix_base + index + 3u];
        let q3 = query_words[query_base + index + 3u];
        accumulator = accumulator + ((m0 * q0 + m1 * q1) + (m2 * q2 + m3 * q3));
        base = base + chunk_span;
    }
    // Scalar tail for `ell mod (4 * WORKGROUP_SIZE)` elements, strided by
    // the workgroup so the remaining loads stay coalesced.
    var tail = chunk_end + local_id.x;
    while (tail < dims.ell) {
        accumulator = accumulator
            + matrix_words[matrix_base + tail] * query_words[query_base + tail];
        tail = tail + WORKGROUP_SIZE;
    }

    // Tree reduction: every thread publishes its partial, then rounds of
    // half-span pairwise adds with a barrier between rounds. All threads
    // reach every barrier because the loop bounds are workgroup-uniform and
    // the conditional add is inside the barrier points.
    partials[local_id.x] = accumulator;
    workgroupBarrier();
    var span = WORKGROUP_SIZE / 2u;
    while (span > 0u) {
        if (local_id.x < span) {
            partials[local_id.x] = partials[local_id.x] + partials[local_id.x + span];
        }
        workgroupBarrier();
        span = span / 2u;
    }
    if (local_id.x == 0u) {
        output_words[pair] = partials[0];
    }
}
