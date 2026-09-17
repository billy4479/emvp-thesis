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
// where `M` is the encrypted matrix (`rows x n` words), `Q` holds the
// `batch` encrypted queries (`batch x n` words), `n = b * s` is the
// codeword length, and `b` is the protocol block size. Host code must keep
// `rows * n <= 2^32` and `batch * n <= 2^32` so every index below fits u32
// without overflow; the output count is additionally capped below
// `2^32 - 65535 * WORKGROUP_SIZE` by the host (see `MAX_ANSWER_WORDS` in
// gpu/mod.rs) so the reconstructed thread index, which includes workgroup
// padding, also stays inside u32.
//
// # Device matrix layout
//
// The wire layout of `M` is row-major (`r * n + j * b + t`, the CPU answer
// arena's order), but the host uploads it permuted into the device layout
// `matrix[j][t][r]` (word `(j * b + t) * rows + r`), and adjacent threads
// cover adjacent rows `r` of one (query, block) pair. At one loop step the
// warp's 32 matrix loads are then 128 contiguous bytes consumed in full,
// and its query loads all broadcast the same word; nothing depends on L1
// retaining strided reuse across iterations. The kernel's answers are
// unchanged — the permutation is invisible to the arithmetic — so results
// stay bit-identical to the CPU reference.
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
// element instead of REDC's two).
//
// # Narrow-modulus hot loop
//
// When `MODULUS < 2^30` — the deployment prime `998244353` and the NTT
// prime `1073479681` included — every hot-loop operand is a canonical
// residue below `2^30`, so the wide product is computed by the 15-bit
// Karatsuba split `mul_30x30`: three native multiplies instead of the four
// of the schoolbook `mul_32x32` (the sums of the 15-bit halves are below
// `2^16`, so the crossing term still fits u32 exactly). The same bound
// makes four consecutive product-high words sum exactly into one u32 word
// (`each < 2^28`), so the four-wide loop first folds its four products into
// a single exact 64-bit chunk and pays one accumulator carry chain per
// four elements. Moduli of `2^30` and above keep the general `mul_32x32`
// loop; the branch condition is a pipeline-overridable constant, uniform
// across the dispatch and folded at pipeline specialization.
//
// # Final fold
//
// Each fold divides by `2^32`, so after the first REDC step the 64-bit
// value `S1` is congruent to `A * 2^-32 mod p` — exactly the
// representation the canonical output needs, still spread over two words.
// The host picks per call between two exact reductions of `S1`:
//
// 1. Direct (`Dims.fast_fold = 1`): write `S1 = s1_low + s1_high * 2^32`
//    and fold the high word through `R1 = 2^32 mod p` (override `R1`) in a
//    carry cascade whose first pass the host guards: it derives the exact
//    worst case `s1_high <= ((b * (p-1)^2 + (2^32-1) * p) / 2^64)` from the
//    block size and enables the path only when `s1_high * R1 < 2^32` holds
//    unconditionally. Every later cascade pass has `s1_high <= 1`, so
//    `s1_high * R1 < 2^32` holds on its own and the cascaded value strictly
//    shrinks; one conditional-subtraction loop canonicalizes. For the
//    deployment prime this covers `b <= 273`, for the NTT prime all
//    benchmarked block sizes.
// 2. Generic: a second REDC step collapses `S1` to one word congruent to
//    `A * 2^-64`, and a REDC multiply by `R2 = 2^64 mod p` cancels the
//    extra `2^-32` factor (see `fold`).
//
// Both return the canonical word `A * 2^-32 = sum_t m_t q_t * 2^-32`,
// exactly the sum of canonical Montgomery products the CPU accumulates, so
// results stay bit-identical to the CPU reference under either path.
//
// The four-wide main loop issues one thread's four loads together so the
// driver can merge them into single wide loads, and drops the loop trip
// count fourfold; a scalar tail covers `b` values that are not multiples
// of four.
//
// All protocol data here is public (encrypted matrix, encrypted queries,
// answers), so no constant-time discipline is required on the GPU side.
//
// WGSL constructs used: module-scope `override` pipeline constants (one of
// them in `@workgroup_size`, as the WGSL spec's own pipeline-overridable
// workgroup-size example), one `struct` of scalar `u32` members in the
// uniform address space, runtime-sized `array<u32>` storage bindings,
// builtins `@local_invocation_id` and `@workgroup_id`, and `select`. No
// extensions, no workgroup arrays, no f64.

// MODULUS is the prime p. NEG_INV is -p^{-1} mod 2^32, the REDC constant
// PrimeField::montgomery_neg_inv() returns. R1 = 2^32 mod p is
// PrimeField::montgomery_r(): the direct final fold folds each 2^32 weight
// through it (see "Final fold" above). R2 = 2^64 mod p is
// PrimeField::montgomery_r2(): in the generic final fold it cancels the
// lazy accumulator's extra `2^-32` fold. The defaults are never used
// because the host always supplies all four constants.
override MODULUS: u32 = 0u;
override NEG_INV: u32 = 0u;
override R1: u32 = 0u;
override R2: u32 = 0u;

// Threads per workgroup. An override, not a const: the WGSL spec allows a
// pipeline-overridable constant in `@workgroup_size`, and the host supplies
// its `WORKGROUP_SIZE` here, so sweeping workgroup sizes is a host-side
// constant change with no shader edit.
override WORKGROUP_SIZE: u32 = 256u;

// Per-call dimensions. All words are little-endian u32. `fast_fold` is the
// host-computed final-fold path selector (1 = direct `S1` reduction is
// exact for this call's block size, see "Final fold" above); the last word
// keeps the struct at a clean 32 bytes for the uniform binding.
struct Dims {
    n: u32,
    b: u32,
    s: u32,
    rows: u32,
    batch: u32,
    workgroups_x: u32,
    fast_fold: u32,
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

// Karatsuba 30x30 -> 64-bit multiplication via 15-bit halves, returning
// vec2<u32>(low, high). Exact for `a, b < 2^30`, which every canonical
// residue of a `MODULUS < 2^30` satisfies, so the four-wide hot loop can
// use it in place of `mul_32x32` and save one of its four native
// multiplies per product.
//
// With a = a1 * 2^15 + a0 and b = b1 * 2^15 + b0 (all halves < 2^15):
//   a * b = ll + cross * 2^15 + hh * 2^30,
// where ll = a0*b0, hh = a1*b1, and Karatsuba's crossing term
// cross = a0*b1 + a1*b0 = (a0 + a1) * (b0 + b1) - ll - hh. The half sums
// are below 2^16, so their product fits u32 exactly and `cross < 2^31`;
// `cross * 2^15` splits into (cross >> 17) * 2^32 plus its wrapped low
// word, and `hh * 2^30` into (hh >> 2) * 2^32 plus its own, so
//   lo  = ll + (cross << 15) + (hh << 30)        (wrapping, carries kept)
//   hi  = (cross >> 17) + (hh >> 2) + carry1 + carry2
// with carry1, carry2 the two wrap bits of the low sum. The high word is
// at most 2^14 + 2^28 + 2, far inside u32.
fn mul_30x30(a: u32, b: u32) -> vec2<u32> {
    let a0 = a & 0x7FFFu;
    let a1 = a >> 15u;
    let b0 = b & 0x7FFFu;
    let b1 = b >> 15u;
    let ll = a0 * b0;
    let hh = a1 * b1;
    let cross = (a0 + a1) * (b0 + b1) - ll - hh;
    let lo1 = ll + (cross << 15u);
    let carry1 = select(0u, 1u, lo1 < ll);
    let lo = lo1 + (hh << 30u);
    let carry2 = select(0u, 1u, lo < lo1);
    let hi = (cross >> 17u) + (hh >> 2u) + carry1 + carry2;
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

// Combines four raw 64-bit products into one exact 64-bit chunk `lo4 +
// hi4 * 2^32` and adds it to `acc` with a single `accumulate` carry chain,
// instead of paying four serialized chains. Exact only for narrow moduli:
// each product high word is below `2^28` (both operands below `2^30`), so
// the four high words plus the three wrap carries stay below `2^30` and
// `hi4` never wraps. Used by the narrow four-wide loops of both entry
// points.
fn accumulate4(acc: vec3<u32>, p0: vec2<u32>, p1: vec2<u32>, p2: vec2<u32>, p3: vec2<u32>) -> vec3<u32> {
    let lo01 = p0.x + p1.x;
    let carry01 = select(0u, 1u, lo01 < p0.x);
    let lo23 = p2.x + p3.x;
    let carry23 = select(0u, 1u, lo23 < p2.x);
    let lo4 = lo01 + lo23;
    let carry4 = select(0u, 1u, lo4 < lo01);
    let hi4 = p0.y + p1.y + p2.y + p3.y + carry01 + carry23 + carry4;
    return accumulate(acc, lo4, hi4);
}

// Reduces the 96-bit accumulator `A = acc.x + acc.y * 2^32 + acc.z * 2^64`
// to the canonical Montgomery sum `A * 2^-32 mod p`.
//
// Step one is a REDC fold of the low word, shared by both paths: it turns
// `A` into the two-word `S1 = (A + m * p) / 2^32`, which is already
// congruent to `A * 2^-32 mod p` — the representation the canonical output
// needs — and merely has to be narrowed to one canonical word (bounds:
// `A < b * 2^62 <= 2^94` and `m * p < 2^63`, so `S1 < 2^63` fits two words
// with `s1_high < 2^31`).
//
// Direct path (`dims.fast_fold` set by the host): `S1 mod p` =
// `s1_low + s1_high * R1 mod p` with `R1 = 2^32 mod p`. The cascade keeps
// the invariant `x + h * 2^32 = S1 (mod p)` while replacing the `2^32`
// weight by `R1`; the host enables the path only when the exact worst case
// `s1_high * R1 < 2^32` holds for the call's `b`, so the first pass's
// multiply is exact, every later pass has `h <= 1` (a single wrap carry)
// and the multiplied value strictly shrinks, and the loop terminates after
// at most a few passes. One conditional-subtraction loop canonicalizes
// `x < 2^32` (data-dependent trip count, but the data is public).
//
// Generic path: a second REDC fold on `s1_low` collapses `S1` to one word
//   `m2 = (S1 mod 2^32) * NEG_INV mod 2^32` makes `S1 + m2 * p` divisible
//   by `2^32`, and `S1 + m2 * p < 2^63 + 2^63`, so
//   `S2 = (S1 + m2 * p) / 2^32 < 2^32` fits one word, and every addition
//   along the way stays inside u32.
// `S2` is congruent to `A * 2^-64 mod p` but can exceed `p` (up to
// roughly `2p + b/4`), so the canonicalization loop subtracts `p` until
// the value is canonical; on protocol moduli it runs at most twice, and
// the data is public so the data-dependent trip count needs no
// constant-time discipline. The REDC multiply by `R2 = 2^64 mod p` then
// returns `S2 * 2^32 = A * 2^-32`, canonical.
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

    if (dims.fast_fold != 0u) {
        var x = s1_low;
        var h = s1_high;
        loop {
            if h == 0u {
                break;
            }
            // Exact: the host guarantees s1_high * R1 < 2^32 for the first
            // pass, and h <= 1 afterwards.
            let sum = x + h * R1;
            h = select(0u, 1u, sum < x);
            x = sum;
        }
        while (x >= MODULUS) {
            x = x - MODULUS;
        }
        return x;
    }

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
    // first, then query 1, and so on. Adjacent threads cover adjacent
    // matrix ROWS of one (query, block) pair — the encrypted matrix is
    // stored permuted as `matrix[block][t][row]` (see the module comment),
    // so at one loop step `t` the warp's 32 lanes read 128 consecutive
    // bytes of the matrix stream and all broadcast the same query word;
    // every cache line the kernel touches is consumed in full the moment
    // it arrives. Each thread's four loads per step span four consecutive
    // `t` rows of the permuted layout, so the per-thread stream is four
    // independent sequential readers instead of one strided one. Writes
    // scatter across one answer word per lane at a stride of `s`, which is
    // one word per thread per kernel and negligible.
    let row = index % dims.rows;
    let rest = index / dims.rows;
    let block = rest % dims.s;
    let query = rest / dims.s;

    // Permuted matrix base `block * b * rows + row`, advanced by `rows`
    // words per `t` step; the invariant offsets are hoisted.
    let matrix_base = block * dims.b * dims.rows + row;
    let query_base = query * dims.n + block * dims.b;
    // The accumulator starts at exact integer zero: it holds raw products,
    // not residues, so no Montgomery seed is involved.
    var accumulator = vec3<u32>(0u, 0u, 0u);
    // Four-wide main loop over the multiples of four, then a scalar tail
    // for the remaining `b mod 4` elements. The thread's four matrix reads
    // per step sit `rows` words apart (consecutive `t` rows of the permuted
    // layout) and its four query reads are consecutive words; the driver
    // can merge each group into wide loads.
    //
    // For `MODULUS < 2^30` the narrow branch multiplies with `mul_30x30`
    // (three native multiplies per product) and folds each group of four
    // products into one exact 64-bit chunk before a single accumulator
    // update: each product high word is below 2^28, so the four high words
    // plus the three wrap carries stay below 2^30. Otherwise the general
    // `mul_32x32` loop applies. The branch condition is the specialized
    // pipeline constant, uniform over the dispatch.
    let narrow = MODULUS < 0x40000000u;
    let vector_end = dims.b - (dims.b % 4u);
    let row1 = dims.rows;
    let row2 = 2u * dims.rows;
    let row3 = 3u * dims.rows;
    let row4 = 4u * dims.rows;
    var mrow = matrix_base;
    var offset: u32 = 0u;
    if (narrow) {
        while (offset < vector_end) {
            let m0 = matrix_words[mrow];
            let q0 = query_words[query_base + offset];
            let m1 = matrix_words[mrow + row1];
            let q1 = query_words[query_base + offset + 1u];
            let m2 = matrix_words[mrow + row2];
            let q2 = query_words[query_base + offset + 2u];
            let m3 = matrix_words[mrow + row3];
            let q3 = query_words[query_base + offset + 3u];
            let p0 = mul_30x30(m0, q0);
            let p1 = mul_30x30(m1, q1);
            let p2 = mul_30x30(m2, q2);
            let p3 = mul_30x30(m3, q3);
            accumulator = accumulate4(accumulator, p0, p1, p2, p3);
            mrow = mrow + row4;
            offset = offset + 4u;
        }
    } else {
        while (offset < vector_end) {
            let m0 = matrix_words[mrow];
            let q0 = query_words[query_base + offset];
            let m1 = matrix_words[mrow + row1];
            let q1 = query_words[query_base + offset + 1u];
            let m2 = matrix_words[mrow + row2];
            let q2 = query_words[query_base + offset + 2u];
            let m3 = matrix_words[mrow + row3];
            let q3 = query_words[query_base + offset + 3u];
            let p0 = mul_32x32(m0, q0);
            let p1 = mul_32x32(m1, q1);
            let p2 = mul_32x32(m2, q2);
            let p3 = mul_32x32(m3, q3);
            accumulator = accumulate(accumulator, p0.x, p0.y);
            accumulator = accumulate(accumulator, p1.x, p1.y);
            accumulator = accumulate(accumulator, p2.x, p2.y);
            accumulator = accumulate(accumulator, p3.x, p3.y);
            mrow = mrow + row4;
            offset = offset + 4u;
        }
    }
    // Scalar tail: at most three elements, so the general multiply costs
    // nothing measurable.
    while (offset < dims.b) {
        let product = mul_32x32(
            matrix_words[mrow],
            query_words[query_base + offset],
        );
        accumulator = accumulate(accumulator, product.x, product.y);
        mrow = mrow + row1;
        offset = offset + 1u;
    }
    // The arena is query-major, then row-major, then block-major — the
    // thread index is (query, block, row)-major after the row-dispatched
    // lane swap, so the store slot is computed explicitly rather than
    // taken from `index`.
    let out = query * dims.rows * dims.s + row * dims.s + block;
    answer_words[out] = fold(accumulator);
}

// Four-query tiled variant of `main`, for `MODULUS < 2^30` batches.
//
// One invocation computes the same `(row, block)` dot product for a tile of
// four consecutive queries — the block products `M_hat_j q_q` of the
// protocol's Fig. 1 are independent per block, so grouping queries changes
// no output values, only which matrix loads feed how many dot products.
// Every matrix word load now feeds four accumulators instead of one, which
// cuts the matrix traffic (and its load instructions) fourfold for
// `batch >= 4`; query traffic is unchanged. The price is four 96-bit
// accumulators of register state per thread.
//
// Dispatch geometry: the host launches `ceil(batch / 4) * rows * s`
// threads, tile-group-major; `index / (rows * s)` selects the query tile,
// and within one tile adjacent threads cover adjacent matrix rows (see
// `main`). Queries past the end of the batch are clamped to the last
// query: their lanes load, multiply, and fold duplicate data (keeping the
// loop branch-free and every address in bounds) but their stores are
// individually guarded, so the answers for queries `batch..` are never
// written.
@compute
@workgroup_size(WORKGROUP_SIZE)
fn main_q4(
    @builtin(local_invocation_id) local_id: vec3<u32>,
    @builtin(workgroup_id) workgroup: vec3<u32>,
) {
    let tiles = dims.rows * dims.s;
    let index =
        ((workgroup.y * dims.workgroups_x) + workgroup.x) * WORKGROUP_SIZE + local_id.x;
    let total = ((dims.batch + 3u) / 4u) * tiles;
    if index >= total {
        return;
    }
    // Row-dispatched lanes over the permuted matrix (see `main`): adjacent
    // threads cover adjacent rows of one (tile-group, block), so the
    // warp's matrix reads are contiguous and its query reads broadcast.
    // `group` selects the query tile, `out` is the per-query answer offset
    // inside one query's `tiles`-word arena run.
    let row = index % dims.rows;
    let rest = index / dims.rows;
    let block = rest % dims.s;
    let group = rest / dims.s;
    let first_query = group * 4u;
    let out = row * dims.s + block;

    let matrix_base = block * dims.b * dims.rows + row;
    // Clamp the tile's query indices into the batch; guarded stores below
    // decide which of the four results are real.
    let query0 = min(first_query, dims.batch - 1u);
    let query1 = min(first_query + 1u, dims.batch - 1u);
    let query2 = min(first_query + 2u, dims.batch - 1u);
    let query3 = min(first_query + 3u, dims.batch - 1u);
    let query_base0 = query0 * dims.n + block * dims.b;
    let query_base1 = query1 * dims.n + block * dims.b;
    let query_base2 = query2 * dims.n + block * dims.b;
    let query_base3 = query3 * dims.n + block * dims.b;

    var acc0 = vec3<u32>(0u, 0u, 0u);
    var acc1 = vec3<u32>(0u, 0u, 0u);
    var acc2 = vec3<u32>(0u, 0u, 0u);
    var acc3 = vec3<u32>(0u, 0u, 0u);
    let vector_end = dims.b - (dims.b % 4u);
    let row1 = dims.rows;
    let row2 = 2u * dims.rows;
    let row3 = 3u * dims.rows;
    let row4 = 4u * dims.rows;
    var mrow = matrix_base;
    var offset: u32 = 0u;
    while (offset < vector_end) {
        let m0 = matrix_words[mrow];
        let m1 = matrix_words[mrow + row1];
        let m2 = matrix_words[mrow + row2];
        let m3 = matrix_words[mrow + row3];

        let a00 = query_words[query_base0 + offset];
        let a01 = query_words[query_base0 + offset + 1u];
        let a02 = query_words[query_base0 + offset + 2u];
        let a03 = query_words[query_base0 + offset + 3u];
        let b00 = mul_30x30(m0, a00);
        let b01 = mul_30x30(m1, a01);
        let b02 = mul_30x30(m2, a02);
        let b03 = mul_30x30(m3, a03);
        acc0 = accumulate4(acc0, b00, b01, b02, b03);

        let a10 = query_words[query_base1 + offset];
        let a11 = query_words[query_base1 + offset + 1u];
        let a12 = query_words[query_base1 + offset + 2u];
        let a13 = query_words[query_base1 + offset + 3u];
        let b10 = mul_30x30(m0, a10);
        let b11 = mul_30x30(m1, a11);
        let b12 = mul_30x30(m2, a12);
        let b13 = mul_30x30(m3, a13);
        acc1 = accumulate4(acc1, b10, b11, b12, b13);

        let a20 = query_words[query_base2 + offset];
        let a21 = query_words[query_base2 + offset + 1u];
        let a22 = query_words[query_base2 + offset + 2u];
        let a23 = query_words[query_base2 + offset + 3u];
        let b20 = mul_30x30(m0, a20);
        let b21 = mul_30x30(m1, a21);
        let b22 = mul_30x30(m2, a22);
        let b23 = mul_30x30(m3, a23);
        acc2 = accumulate4(acc2, b20, b21, b22, b23);

        let a30 = query_words[query_base3 + offset];
        let a31 = query_words[query_base3 + offset + 1u];
        let a32 = query_words[query_base3 + offset + 2u];
        let a33 = query_words[query_base3 + offset + 3u];
        let b30 = mul_30x30(m0, a30);
        let b31 = mul_30x30(m1, a31);
        let b32 = mul_30x30(m2, a32);
        let b33 = mul_30x30(m3, a33);
        acc3 = accumulate4(acc3, b30, b31, b32, b33);
        mrow = mrow + row4;
        offset = offset + 4u;
    }
    while (offset < dims.b) {
        let m = matrix_words[mrow];
        let c0 = mul_30x30(m, query_words[query_base0 + offset]);
        acc0 = accumulate(acc0, c0.x, c0.y);
        let c1 = mul_30x30(m, query_words[query_base1 + offset]);
        acc1 = accumulate(acc1, c1.x, c1.y);
        let c2 = mul_30x30(m, query_words[query_base2 + offset]);
        acc2 = accumulate(acc2, c2.x, c2.y);
        let c3 = mul_30x30(m, query_words[query_base3 + offset]);
        acc3 = accumulate(acc3, c3.x, c3.y);
        mrow = mrow + row1;
        offset = offset + 1u;
    }

    let out0 = first_query * tiles + out;
    answer_words[out0] = fold(acc0);
    if (first_query + 1u < dims.batch) {
        answer_words[out0 + tiles] = fold(acc1);
    }
    if (first_query + 2u < dims.batch) {
        answer_words[out0 + 2u * tiles] = fold(acc2);
    }
    if (first_query + 3u < dims.batch) {
        answer_words[out0 + 3u * tiles] = fold(acc3);
    }
}
