// EMVP server answer kernel over the raw Montgomery words of the protocol
// field.
//
// Every buffer word is a canonical Montgomery residue `a * 2^32 mod p`
// with `0 < p < 2^31` (`p = 998244353` in the protocol deployment), exactly
// the representation `prime_field_layer::FieldElement` stores, so words
// stream to and from the device without conversion and results are
// bit-identical to the CPU `answer_batch` in `emvp::protocol`, which remains
// the reference implementation.
//
// For each output index `g = (q * rows + r) * s + j` (query-major, matching
// the CPU answer arena) the kernel computes
//
//     answer[g] = sum_{t < b} M[r][j * b + t] * Q[q][j * b + t]   in F_p,
//
// where `M` is the row-major encrypted matrix (`rows x n` words), `Q` holds
// the `batch` encrypted queries (`batch x n` words), `n = b * s` is the
// codeword length, and `b` is the protocol block size. Host code must keep
// `rows * n <= 2^32` and `batch * n <= 2^32` so every index below fits u32
// without overflow; the output count is additionally capped below
// `2^32 - 65535 * 256` by the host (see `MAX_ANSWER_WORDS` in gpu/mod.rs) so
// the reconstructed thread index, which includes workgroup padding, also
// stays inside u32.
//
// All protocol data here is public (encrypted matrix, encrypted queries,
// answers), so no constant-time discipline is required on the GPU side.
//
// WGSL constructs used: module-scope `override` pipeline constants, one
// `struct` of scalar `u32` members in the uniform address space, runtime-sized
// `array<u32>` storage bindings, `@workgroup_size` with a module `const`,
// builtins `@local_invocation_id` and `@workgroup_id`, and `select`. No
// extensions, no workgroup arrays, no f64.

// MODULUS is the prime p. NEG_INV is -p^{-1} mod 2^32, the REDC constant
// PrimeField::montgomery_neg_inv() returns. R2 = 2^64 mod p is deliberately
// NOT passed: these buffers already hold Montgomery residues, and R2 is only
// needed to enter Montgomery form, which the CPU did when building the
// elements. The defaults are never used because the host always supplies
// both constants.
override MODULUS: u32 = 0u;
override NEG_INV: u32 = 0u;

// Must match the host WORKGROUP_SIZE. A module const (not an override) keeps
// the workgroup size a compile-time constant as required.
const WORKGROUP_SIZE: u32 = 256u;

// Per-call dimensions. All words are little-endian u32; the two pads keep the
// struct at a clean 32 bytes for the uniform binding.
struct Dims {
    n: u32,
    b: u32,
    s: u32,
    rows: u32,
    batch: u32,
    workgroups_x: u32,
    pad_0: u32,
    pad_1: u32,
}

@group(0) @binding(0) var<storage, read> matrix_words: array<u32>;
@group(0) @binding(1) var<storage, read> query_words: array<u32>;
@group(0) @binding(2) var<storage, read_write> answer_words: array<u32>;
@group(0) @binding(3) var<uniform> dims: Dims;

// Schoolbook 32x32 -> 64-bit multiplication via 16-bit halves, returning
// vec2<u32>(low, high). WGSL has no 64-bit scalars and no mul_hi, so the
// product is assembled from four 16x16 partial products, each of which fits
// u32 exactly: (2^16 - 1)^2 = 2^32 - 2^17 + 1 < 2^32.
//
// With a = a1 * 2^16 + a0 and b = b1 * 2^16 + b0:
//   a * b = hh * 2^32 + mid * 2^16 + ll,  mid = lh + hl,
// where ll = a0*b0, lh = a0*b1, hl = a1*b0, hh = a1*b1. `mid` is up to
// 2*(2^32 - 2^17 + 1) < 2^33, so its bit 32 cannot stay in the wrapping u32
// sum and is captured explicitly as `mid_carry`. Similarly
// ll + (mid << 16) can wrap and leaves `lo_carry`. The high word is then
// exactly (a * b) >> 32 = hh + (mid >> 16) + mid_carry * 2^16 + lo_carry,
// which is at most 2^32 - 2 for u32 inputs, so the monotone partial sums
// never wrap either.
//
// Consecutive threads touch consecutive 16-bit halves of consecutive words,
// so the shifts and masks stay fully in registers.
fn mul_32x32(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0xFFFFu;
    let a1 = a >> 16u;
    let b0 = b & 0xFFFFu;
    let b1 = b >> 16u;
    let ll = a0 * b0;
    let lh = a0 * b1;
    let hl = a1 * b0;
    let hh = a1 * b1;
    let mid = lh + hl;
    let mid_carry = select(0u, 1u, mid < lh);
    let lo = ll + (mid << 16u);
    let lo_carry = select(0u, 1u, lo < ll);
    let hi = hh + (mid >> 16u) + (mid_carry << 16u) + lo_carry;
    return vec2<u32>(lo, hi);
}

// Montgomery multiplication (REDC) of two canonical residues a, b < p,
// returning the canonical Montgomery product a * b * 2^-32 mod p.
//
// This is the same computation as PrimeField::montgomery_mul_scalar with the
// u64 arithmetic carried in two u32 words, so results are bit-identical to
// the CPU kernel:
//   T = a * b < 2^62            (both operands are canonical, p < 2^31)
//   m = (T mod 2^32) * NEG_INV  mod 2^32  (wrapping u32 multiply)
//   U = m * p < 2^63            (m < 2^32, p < 2^31)
//   T + U < 2^62 + 2^63 < 2^64  (no 64-bit overflow, so unlike the CPU
//                                kernel no bit-63 carry correction exists)
//   result = (T + U) >> 32, exact because m makes T + U divisible by 2^32
// and, since T < p * 2^32 and U < 2^32 * p, result < 2p, so the single
// conditional subtraction restores 0 <= result < p.
fn fmul(a: u32, b: u32) -> u32 {
    let t = mul_32x32(a, b);
    let m = t.x * NEG_INV;
    let mp = mul_32x32(m, MODULUS);
    let lo = t.x + mp.x;
    let carry = select(0u, 1u, lo < t.x);
    let reduced = t.y + mp.y + carry;
    return select(reduced, reduced - MODULUS, reduced >= MODULUS);
}

// Field addition of canonical residues. a + b < 2p < 2^32 cannot overflow.
// This is a branchless `select` on public data; constant-time discipline is
// unnecessary here and also unnecessary on the CPU reference.
fn fadd(a: u32, b: u32) -> u32 {
    let s = a + b;
    return select(s, s - MODULUS, s >= MODULUS);
}

@compute
@workgroup_size(WORKGROUP_SIZE)
fn main(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup: vec3<u32>,
) {
    // The host dispatches workgroups_x x workgroups_y workgroups over a
    // linear work count of ceil(total / WORKGROUP_SIZE). All index math is
    // u32 and cannot wrap for any accepted output count: the host's
    // `MAX_ANSWER_WORDS` cap keeps the padded thread count below 2^32 (see
    // the module comment).
    let index =
        ((workgroup.y * dims.workgroups_x) + workgroup.x) * WORKGROUP_SIZE + local_id.x;
    let total = dims.batch * dims.rows * dims.s;
    if index >= total {
        return;
    }
    // Query-major grid matching the CPU answer arena: answers for query 0
    // first, then query 1, and so on. Consecutive threads cover consecutive
    // blocks of one (query, row) pair, so each iteration step reads one word
    // per thread at a stride of `b` words (one block); a warp therefore
    // touches up to `b` distinct cache lines per step, which is fully
    // coalesced for small `b`. The `offset` loop revisits the same lines, and
    // neighbouring workgroups reuse the same matrix row across queries, so
    // the traffic stays L1/L2-resident.
    let block = index % dims.s;
    let query_row = index / dims.s;
    let row = query_row % dims.rows;
    let query = query_row / dims.rows;

    let start = block * dims.b;
    // The Montgomery residue of zero is zero, matching the CPU accumulator's
    // canonical zero seed.
    var accumulator: u32 = 0u;
    for (var offset: u32 = 0u; offset < dims.b; offset = offset + 1u) {
        let matrix_word = matrix_words[row * dims.n + start + offset];
        let query_word = query_words[query * dims.n + start + offset];
        accumulator = fadd(accumulator, fmul(matrix_word, query_word));
    }
    answer_words[index] = accumulator;
}
