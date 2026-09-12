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
// # Lazy reduction
//
// The inner loop does not reduce per product. Each thread accumulates the
// raw 64-bit products of the Montgomery words into a 96-bit accumulator
// (three u32 words) and applies one two-step Montgomery fold at the end.
// The accumulator is exact for every shape the host accepts: each product
// of canonical residues is below `p^2 < 2^62`, and `b <= n <= 2^32` keeps
// the sum below `2^32 * 2^62 = 2^94 < 2^96`. This halves the multiply
// count of the multiply-dominated inner loop (one wide multiply per
// element instead of REDC's two) at a fixed cost of four wide multiplies
// per output element. Each fold divides by `2^32`, so the folded value is
// congruent to `A * 2^-64 mod p`; one REDC multiply by `R2 = 2^64 mod p`
// cancels the extra factor and returns the canonical word
// `A * 2^-32 = sum_t m_t q_t * 2^-32`, exactly the sum of canonical
// Montgomery products the CPU accumulates, so results stay bit-identical
// to the CPU reference.
//
// The four-wide main loop issues one thread's four consecutive loads
// together so the driver can merge them into single wide loads, and drops
// the loop trip count fourfold; a scalar tail covers `b` values that are
// not multiples of four.
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
// PrimeField::montgomery_neg_inv() returns. R2 = 2^64 mod p is
// PrimeField::montgomery_r2(): the buffers already hold Montgomery
// residues, so R2 is not needed to enter Montgomery form — it exists to
// cancel the lazy accumulator's double `2^-32` fold (see "Lazy reduction"
// above). The defaults are never used because the host always supplies all
// three constants.
override MODULUS: u32 = 0u;
override NEG_INV: u32 = 0u;
override R2: u32 = 0u;

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
// u64 arithmetic carried in two u32 words:
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

// Adds one raw 64-bit product `lo + hi * 2^32` into the 96-bit accumulator
// `acc`. `hi` is below `2^30` because both product operands are canonical
// residues below `p < 2^31`, so `hi + carry_low` cannot wrap; the carries
// into the middle and top words are captured explicitly. The top word stays
// below `2^32` for every accepted shape (the whole accumulator is below
// `2^94`, see the module comment), so the accumulation is exact.
fn accumulate(acc: vec3<u32>, lo: u32, hi: u32) -> vec3<u32> {
    let low = acc.x + lo;
    let carry_low = select(0u, 1u, low < acc.x);
    let mid_add = hi + carry_low;
    let mid = acc.y + mid_add;
    let carry_mid = select(0u, 1u, mid < acc.y);
    return vec3<u32>(low, mid, acc.z + carry_mid);
}

// Reduces the 96-bit accumulator `A = acc.x + acc.y * 2^32 + acc.z * 2^64`
// to the canonical Montgomery sum `A * 2^-32 mod p`: two REDC steps (one
// per 32-bit word) fold `A` to a single word congruent to `A * 2^-64`, and
// a final REDC multiply by `R2 = 2^64 mod p` cancels the extra `2^-64`
// factor (see the module comment).
//
// Bounds, with `b <= n <= 2^32` and `p < 2^31`:
//   A < b * 2^62 <= 2^94, so three words hold it exactly.
//   Step one: `m = acc.x * NEG_INV mod 2^32` makes `A + m * p` divisible
//   by `2^32`, and `A + m * p < 2^94 + 2^63 < 2^95`, so
//   `S1 = (A + m * p) / 2^32` fits two words with `S1`'s high word below
//   `2^31`.
//   Step two: `m2 = (S1 mod 2^32) * NEG_INV mod 2^32` makes
//   `S1 + m2 * p` divisible by `2^32`, and `S1 + m2 * p < 2^63 + 2^63`, so
//   `S2 = (S1 + m2 * p) / 2^32 < 2^32` fits one word, and every addition
//   along the way stays inside u32.
//   `S2` is congruent to `A * 2^-64 mod p` but can exceed `p` (up to
//   roughly `2p + b/4`), so the canonicalization loop subtracts `p` until
//   the value is canonical; on protocol moduli it runs at most twice, and
//   the data is public so the data-dependent trip count needs no
//   constant-time discipline.
fn fold(acc: vec3<u32>) -> u32 {
    let m = acc.x * NEG_INV;
    let mp = mul_32x32(m, MODULUS);
    // A + mp is divisible by 2^32, so the low word of the sum is zero and
    // only the carries propagate upward.
    let low = acc.x + mp.x;
    let carry_low = select(0u, 1u, low < acc.x);
    let mid = acc.y + mp.y;
    let carry_mid = select(0u, 1u, mid < acc.y);
    let s1_low = mid + carry_low;
    let carry_s1 = select(0u, 1u, s1_low < mid);
    let s1_high = acc.z + carry_mid + carry_s1;

    let m2 = s1_low * NEG_INV;
    let mp2 = mul_32x32(m2, MODULUS);
    let low2 = s1_low + mp2.x;
    let carry_low2 = select(0u, 1u, low2 < s1_low);
    var reduced = s1_high + mp2.y + carry_low2;
    while (reduced >= MODULUS) {
        reduced = reduced - MODULUS;
    }
    return fmul(reduced, R2);
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
    // blocks of one (query, row) pair, so each iteration step reads four
    // consecutive words per thread at a stride of `b` words (one block); the
    // four-wide loop keeps the per-thread reads contiguous so they merge
    // into wide load instructions. The loop revisits the same cache lines,
    // and neighbouring workgroups reuse the same matrix row across queries,
    // so the traffic stays L1/L2-resident.
    let block = index % dims.s;
    let query_row = index / dims.s;
    let row = query_row % dims.rows;
    let query = query_row / dims.rows;

    let matrix_base = row * dims.n + block * dims.b;
    let query_base = query * dims.n + block * dims.b;
    // The accumulator starts at exact integer zero: it holds raw products,
    // not residues, so no Montgomery seed is involved.
    var accumulator = vec3<u32>(0u, 0u, 0u);
    // Four-wide main loop over the multiples of four, then a scalar tail
    // for the remaining `b mod 4` elements. For `b` a multiple of four the
    // thread's four reads per operand are consecutive words starting at a
    // 16-byte-aligned offset (`b | n` keeps every block base aligned), so
    // the driver can merge them into single wide loads.
    let vector_end = dims.b - (dims.b % 4u);
    var offset: u32 = 0u;
    while (offset < vector_end) {
        let m0 = matrix_words[matrix_base + offset];
        let q0 = query_words[query_base + offset];
        let m1 = matrix_words[matrix_base + offset + 1u];
        let q1 = query_words[query_base + offset + 1u];
        let m2 = matrix_words[matrix_base + offset + 2u];
        let q2 = query_words[query_base + offset + 2u];
        let m3 = matrix_words[matrix_base + offset + 3u];
        let q3 = query_words[query_base + offset + 3u];
        let p0 = mul_32x32(m0, q0);
        let p1 = mul_32x32(m1, q1);
        let p2 = mul_32x32(m2, q2);
        let p3 = mul_32x32(m3, q3);
        accumulator = accumulate(accumulator, p0.x, p0.y);
        accumulator = accumulate(accumulator, p1.x, p1.y);
        accumulator = accumulate(accumulator, p2.x, p2.y);
        accumulator = accumulate(accumulator, p3.x, p3.y);
        offset = offset + 4u;
    }
    while (offset < dims.b) {
        let product = mul_32x32(
            matrix_words[matrix_base + offset],
            query_words[query_base + offset],
        );
        accumulator = accumulate(accumulator, product.x, product.y);
        offset = offset + 1u;
    }
    answer_words[index] = fold(accumulator);
}
