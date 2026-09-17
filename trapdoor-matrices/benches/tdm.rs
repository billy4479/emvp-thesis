#![expect(
    clippy::unwrap_used,
    reason = "benchmarks use fixed valid dimensions and keep setup beside measurements"
)]

//! Trapdoor-matrix (TDM) construction, application, and materialization
//! benchmarks.
//!
//! Construction is split into two groups per mask construction so unlike
//! work is never plotted as comparable:
//!
//! - `full_sampling`: the fail-closed production sampling entry point
//!   (security assessment, secret sampling, and plan construction), timed
//!   end to end. Ring-LPN full sampling is only defined for assessed
//!   parameters, which start at the degree floor `RING_DEGREE_FLOOR =
//!   2048`; below that the assessment reports `Broken` and sampling fails
//!   closed, so the Ring-LPN full-sampling group never runs at the small
//!   construction sizes.
//! - `from_parts`: construction from pre-sampled secrets, timed. Only the
//!   RNG draws that produce the secrets happen during setup; all plan and
//!   table construction stays inside the timing.
//!
//! The two group kinds count different work, and the three constructions
//! retain differently shaped secrets, so no construction group reports a
//! `Throughput::Elements` metric: a shared elements-per-second column would
//! plot words of different meaning against each other. Apply, sensitivity,
//! and materialization cases count their true output elements (`k` outputs
//! per apply, `rows * k` or `k * k` for materialization), which is
//! consistent within and across constructions.
//!
//! Apply sensitivity cases sweep the parameters that dominate each apply:
//! the Ring-LPN secret column weight (on the deterministic fixed-weight
//! fixture builder, which is not the security-sampling path) and the RAA
//! repetition factor `c` (on the production sampler). The
//! `toeplitz_materialize_top_rows` group exercises the public bounded
//! materialization API. All setup happens before timing.

use std::{hint::black_box, time::Duration};

use bench_common as common;
use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use prime_field_layer::{FieldElement, PrimeField};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use trapdoor_matrices::{
    IrreducibleRingLpn, Permutation, RaaWeightedProduct, SparseMatrix, ToeplitzFastProduct,
    ToeplitzMap,
};

const MODULUS: u32 = 1_073_479_681;
const RAA_C: usize = 4;
// Column weight `t` of the secret sparse matrix `E` (the paper's expected
// nonzero count per column, measured exactly instead of in expectation),
// sized to the project policy floor `POLICY_WEIGHT_FLOOR`.
const TARGET_COLUMN_WEIGHT: usize = 192;
// Lowest degree the security assessment accepts for Ring-LPN full sampling
// (`RING_DEGREE_FLOOR`); anything smaller fails closed.
const FULL_SAMPLING_DEGREE_FLOOR: usize = 2_048;

fn seeded_rng(domain: u8, size: usize) -> ChaCha20Rng {
    let mut seed = [domain; 32];
    for (slot, byte) in seed.iter_mut().zip(size.to_le_bytes()) {
        *slot ^= byte;
    }
    ChaCha20Rng::from_seed(seed)
}

fn field_values(length: usize, domain: u8) -> Vec<FieldElement<MODULUS>> {
    let field = PrimeField::<MODULUS>::new();
    let mut values = vec![field.element_u32(0); length];
    field.fill_uniform(&mut seeded_rng(domain, length), &mut values);
    values
}

fn elements(count: usize) -> Throughput {
    Throughput::Elements(u64::try_from(count).unwrap())
}

// ---------------------------------------------------------------------------
// Ring-LPN
// ---------------------------------------------------------------------------

fn ring_inputs(k: usize) -> (Box<[u32]>, Box<[u32]>, SparseMatrix<MODULUS>) {
    // Not `IrreducibleRingLpn::sample`: the from-parts benchmark needs the
    // raw parts to time `new_unchecked_irreducible` alone, and the sparse
    // matrix must have an exact column weight so the measured work does not
    // depend on Bernoulli sampling luck.
    let (modulus, multiplier) = ring_public_parts(k);
    let target_weight = k.min(TARGET_COLUMN_WEIGHT);
    let sparse = trapdoor_matrices::testing::fixed_weight_sparse::<MODULUS, _>(
        k,
        target_weight,
        &mut seeded_rng(0x31, k),
    )
    .unwrap();

    (modulus, multiplier, sparse)
}

fn ring_public_parts(k: usize) -> (Box<[u32]>, Box<[u32]>) {
    // One RNG drawn down once, so the multiplier really is a uniform
    // polynomial. A separate domain from sparse fixtures keeps them
    // independent. The automatic binomial is irreducible by construction.
    let field = PrimeField::<MODULUS>::new();
    let mut multiplier_rng = seeded_rng(0x30, k);
    let multiplier = (0..k)
        .map(|_| field.sample_uniform(&mut multiplier_rng).value())
        .collect::<Box<[u32]>>();
    let modulus = trapdoor_matrices::automatic_ring_modulus::<MODULUS>(k).unwrap();
    (modulus, multiplier)
}

fn ring_instance(k: usize) -> IrreducibleRingLpn<MODULUS> {
    let (modulus, multiplier, sparse) = ring_inputs(k);
    IrreducibleRingLpn::new_unchecked_irreducible(k, &modulus, &multiplier, sparse).unwrap()
}

fn ring_instance_with_weight(k: usize, weight: usize) -> IrreducibleRingLpn<MODULUS> {
    let (modulus, multiplier) = ring_public_parts(k);
    let sparse = trapdoor_matrices::testing::fixed_weight_sparse::<MODULUS, _>(
        k,
        weight,
        &mut seeded_rng(0x34, weight),
    )
    .unwrap();
    IrreducibleRingLpn::new_unchecked_irreducible(k, &modulus, &multiplier, sparse).unwrap()
}

/// Times the fail-closed production sampling entry point at assessed
/// parameters: assessment, secret sampling, and construction together.
fn ring_full_sampling_for(group: &mut BenchmarkGroup<'_, WallTime>, k: usize) {
    group.bench_function(BenchmarkId::from_parameter(k), |b| {
        b.iter_batched(
            || seeded_rng(0x35, k),
            |mut rng| {
                black_box(
                    IrreducibleRingLpn::<MODULUS>::sample(
                        black_box(k),
                        black_box(TARGET_COLUMN_WEIGHT),
                        black_box(&mut rng),
                    )
                    .unwrap(),
                )
            },
            BatchSize::SmallInput,
        );
    });
}

/// Times construction from pre-sampled secrets: plan and ring-table
/// building only.
fn ring_from_parts_for(group: &mut BenchmarkGroup<'_, WallTime>, k: usize) {
    let (modulus, multiplier, sparse) = ring_inputs(k);
    group.bench_function(BenchmarkId::from_parameter(k), |b| {
        b.iter_batched(
            || (&multiplier, sparse.clone()),
            |(multiplier, sparse)| {
                black_box(
                    IrreducibleRingLpn::<MODULUS>::new_unchecked_irreducible(
                        black_box(k),
                        black_box(&modulus),
                        black_box(multiplier),
                        black_box(sparse),
                    )
                    .unwrap(),
                )
            },
            BatchSize::SmallInput,
        );
    });
}

fn ring_apply_for(group: &mut BenchmarkGroup<'_, WallTime>, k: usize) {
    let map = ring_instance(k);
    let input = field_values(k, 0x32);
    let mut output = field_values(k, 0x33);
    let mut scratch = map.scratch();
    group.throughput(elements(k));
    group.bench_function(BenchmarkId::from_parameter(k), |b| {
        b.iter(|| {
            black_box(&map)
                .apply(
                    black_box(&input),
                    black_box(&mut output),
                    black_box(&mut scratch),
                )
                .unwrap();
        });
    });
}

/// Apply cost sensitivity to the secret column weight, on the fixed-weight
/// fixture builder (not the security-sampling path).
fn ring_apply_weight_for(group: &mut BenchmarkGroup<'_, WallTime>, k: usize, weight: usize) {
    let map = ring_instance_with_weight(k, weight);
    let input = field_values(k, 0x32);
    let mut output = field_values(k, 0x33);
    let mut scratch = map.scratch();
    group.throughput(elements(k));
    group.bench_function(BenchmarkId::new(format!("weight{weight}"), k), |b| {
        b.iter(|| {
            black_box(&map)
                .apply(
                    black_box(&input),
                    black_box(&mut output),
                    black_box(&mut scratch),
                )
                .unwrap();
        });
    });
}

fn ring_materialize_for(group: &mut BenchmarkGroup<'_, WallTime>, k: usize) {
    let map = ring_instance(k);
    group.throughput(elements(k * k));
    group.bench_function(BenchmarkId::from_parameter(k), |b| {
        b.iter(|| black_box(black_box(&map).materialize().unwrap()));
    });
}

// ---------------------------------------------------------------------------
// Toeplitz
// ---------------------------------------------------------------------------

fn toeplitz_instance(k: usize) -> ToeplitzFastProduct<MODULUS> {
    ToeplitzFastProduct::sample(k, &mut seeded_rng(0x41, k)).unwrap()
}

/// Times the fail-closed production sampling entry point.
fn toeplitz_full_sampling_for(group: &mut BenchmarkGroup<'_, WallTime>, k: usize) {
    group.bench_function(BenchmarkId::from_parameter(k), |b| {
        b.iter_batched(
            || seeded_rng(0x41, k),
            |mut rng| {
                black_box(
                    ToeplitzFastProduct::<MODULUS>::sample(black_box(k), black_box(&mut rng))
                        .unwrap(),
                )
            },
            BatchSize::SmallInput,
        );
    });
}

/// Times construction from pre-sampled secrets: the three Toeplitz-map
/// NTT plans and the assembled fast product.
fn toeplitz_from_parts_for(group: &mut BenchmarkGroup<'_, WallTime>, k: usize) {
    // The same shapes and fill pattern `ToeplitzFastProduct::sample` uses,
    // so the from-parts construction sees representative data.
    let expanded = 2 * k;
    let right_count = expanded + k - 1;
    let middle_count = 2 * expanded - 1;
    let left_count = k + expanded - 1;
    let right_diagonals = field_values(right_count, 0x47);
    let middle_diagonals = field_values(middle_count, 0x48);
    let left_diagonals = field_values(left_count, 0x49);
    let pi_right = Permutation::sample(expanded, &mut seeded_rng(0x4a, k)).unwrap();
    let pi_left = Permutation::sample(expanded, &mut seeded_rng(0x4b, k)).unwrap();
    group.bench_function(BenchmarkId::from_parameter(k), |b| {
        b.iter_batched(
            || {
                (
                    right_diagonals.clone(),
                    middle_diagonals.clone(),
                    left_diagonals.clone(),
                )
            },
            |(right_diagonals, middle_diagonals, left_diagonals)| {
                let s_right = ToeplitzMap::new(expanded, k, right_diagonals).unwrap();
                let middle = ToeplitzMap::new(expanded, expanded, middle_diagonals).unwrap();
                let s_left = ToeplitzMap::new(k, expanded, left_diagonals).unwrap();
                black_box(
                    ToeplitzFastProduct::<MODULUS>::new(
                        black_box(k),
                        black_box(s_right),
                        black_box(pi_right.clone()),
                        black_box(middle),
                        black_box(pi_left.clone()),
                        black_box(s_left),
                    )
                    .unwrap(),
                )
            },
            BatchSize::SmallInput,
        );
    });
}

fn toeplitz_apply_for(group: &mut BenchmarkGroup<'_, WallTime>, k: usize) {
    let map = toeplitz_instance(k);
    let input = field_values(k, 0x42);
    let mut output = field_values(k, 0x43);
    let mut scratch = map.scratch();
    group.throughput(elements(k));
    group.bench_function(BenchmarkId::from_parameter(k), |b| {
        b.iter(|| {
            black_box(&map)
                .apply(
                    black_box(&input),
                    black_box(&mut output),
                    black_box(&mut scratch),
                )
                .unwrap();
        });
    });
}

fn toeplitz_materialize_for(group: &mut BenchmarkGroup<'_, WallTime>, k: usize) {
    let map = toeplitz_instance(k);
    group.throughput(elements(k * k));
    group.bench_function(BenchmarkId::from_parameter(k), |b| {
        b.iter(|| black_box(black_box(&map).materialize().unwrap()));
    });
}

/// Bounded materialization through the public `materialize_top_rows` API.
fn toeplitz_materialize_top_rows_for(
    group: &mut BenchmarkGroup<'_, WallTime>,
    k: usize,
    rows: usize,
) {
    let map = toeplitz_instance(k);
    group.throughput(elements(rows * k));
    group.bench_function(BenchmarkId::new(format!("rows{rows}"), k), |b| {
        b.iter(|| {
            black_box(
                black_box(&map)
                    .materialize_top_rows(black_box(rows))
                    .unwrap(),
            )
        });
    });
}

// ---------------------------------------------------------------------------
// RAA
// ---------------------------------------------------------------------------

fn raa_instance(k: usize) -> RaaWeightedProduct<MODULUS> {
    RaaWeightedProduct::sample_nonzero(k, RAA_C, &mut seeded_rng(0x51, k)).unwrap()
}

/// Times the fail-closed production sampling entry point.
fn raa_full_sampling_for(group: &mut BenchmarkGroup<'_, WallTime>, k: usize) {
    group.bench_function(BenchmarkId::new(format!("c{RAA_C}"), k), |b| {
        b.iter_batched(
            || seeded_rng(0x51, k),
            |mut rng| {
                black_box(
                    RaaWeightedProduct::<MODULUS>::sample_nonzero(
                        black_box(k),
                        black_box(RAA_C),
                        black_box(&mut rng),
                    )
                    .unwrap(),
                )
            },
            BatchSize::SmallInput,
        );
    });
}

/// Times construction from pre-sampled secrets: the checked reassembly and
/// gather-index building.
fn raa_from_parts_for(group: &mut BenchmarkGroup<'_, WallTime>, k: usize) {
    // Sample the same parts `sample_nonzero` would, during setup.
    let expanded = k * RAA_C;
    let first_permutation = Permutation::sample(expanded, &mut seeded_rng(0x54, k)).unwrap();
    let second_permutation = Permutation::sample(expanded, &mut seeded_rng(0x55, k)).unwrap();
    let third_permutation = Permutation::sample(expanded, &mut seeded_rng(0x56, k)).unwrap();
    let fourth_permutation = Permutation::sample(expanded, &mut seeded_rng(0x57, k)).unwrap();
    let field = PrimeField::<MODULUS>::new();
    let mut first_weights = Vec::with_capacity(expanded);
    let mut second_weights = Vec::with_capacity(expanded);
    let mut third_weights = Vec::with_capacity(expanded);
    let mut weight_rng = seeded_rng(0x58, k);
    for _ in 0..expanded {
        first_weights.push(field.sample_uniform_nonzero(&mut weight_rng));
        second_weights.push(field.sample_uniform_nonzero(&mut weight_rng));
        third_weights.push(field.sample_uniform_nonzero(&mut weight_rng));
    }
    group.bench_function(BenchmarkId::new(format!("c{RAA_C}"), k), |b| {
        b.iter_batched(
            || {
                (
                    first_weights.clone(),
                    second_weights.clone(),
                    third_weights.clone(),
                )
            },
            |(first_weights, second_weights, third_weights)| {
                black_box(
                    RaaWeightedProduct::<MODULUS>::new(
                        black_box(k),
                        black_box(RAA_C),
                        black_box(first_weights),
                        black_box(second_weights),
                        black_box(third_weights),
                        black_box(first_permutation.clone()),
                        black_box(second_permutation.clone()),
                        black_box(third_permutation.clone()),
                        black_box(fourth_permutation.clone()),
                    )
                    .unwrap(),
                )
            },
            BatchSize::SmallInput,
        );
    });
}

fn raa_apply_for(group: &mut BenchmarkGroup<'_, WallTime>, k: usize) {
    let map = raa_instance(k);
    let input = field_values(k, 0x52);
    let mut output = field_values(k, 0x53);
    let mut scratch = map.scratch();
    group.throughput(elements(k));
    group.bench_function(BenchmarkId::new("structured_c4", k), |b| {
        b.iter(|| {
            black_box(&map)
                .apply(
                    black_box(&input),
                    black_box(&mut output),
                    black_box(&mut scratch),
                )
                .unwrap();
        });
    });
}

/// Apply cost sensitivity to the repetition factor `c`, on the production
/// sampler.
fn raa_apply_c_for(group: &mut BenchmarkGroup<'_, WallTime>, k: usize, c: usize) {
    let map = RaaWeightedProduct::sample_nonzero(k, c, &mut seeded_rng(0x5c, k)).unwrap();
    let input = field_values(k, 0x52);
    let mut output = field_values(k, 0x53);
    let mut scratch = map.scratch();
    group.throughput(elements(k));
    group.bench_function(BenchmarkId::new(format!("c{c}"), k), |b| {
        b.iter(|| {
            black_box(&map)
                .apply(
                    black_box(&input),
                    black_box(&mut output),
                    black_box(&mut scratch),
                )
                .unwrap();
        });
    });
}

fn raa_materialize_for(group: &mut BenchmarkGroup<'_, WallTime>, k: usize) {
    let map = raa_instance(k);
    group.throughput(elements(k * k));
    group.bench_function(BenchmarkId::new("materialize_c4", k), |b| {
        b.iter(|| black_box(black_box(&map).materialize().unwrap()));
    });
}
// Suite wiring
// ---------------------------------------------------------------------------

macro_rules! bench_group {
    ($criterion:expr, $tdm_name:literal, $fn_name:literal, $func:expr, $($n:expr),* $(,)?) => {{
        let mut group =
            $criterion.benchmark_group(format!("tdm_{}_{}_p{}", $tdm_name, $fn_name, MODULUS));
        $(
            ($func)(&mut group, $n);
        )*
        group.finish();
    }};
}

macro_rules! all_materialize {
    ($criterion:expr, $($n:expr),* $(,)?) => {
        bench_group!($criterion, "ring_lpn", "materialize", ring_materialize_for, $($n),*);
        bench_group!($criterion, "toeplitz", "materialize", toeplitz_materialize_for, $($n),*);
        bench_group!($criterion, "raa", "materialize", raa_materialize_for, $($n),*);
    };
}

// Construction (from parts), apply, and sensitivity groups at the quick
// sizes; Ring-LPN full sampling starts at the assessment degree floor,
// where the production sampler is defined.
fn quick_tdms<const MODULUS: u32>(c: &mut Criterion) {
    bench_group!(
        c,
        "ring_lpn",
        "full_sampling",
        ring_full_sampling_for,
        FULL_SAMPLING_DEGREE_FLOOR
    );
    bench_group!(
        c,
        "toeplitz",
        "full_sampling",
        toeplitz_full_sampling_for,
        256,
        1024
    );
    bench_group!(c, "raa", "full_sampling", raa_full_sampling_for, 256, 1024);

    bench_group!(c, "ring_lpn", "from_parts", ring_from_parts_for, 256, 1024);
    bench_group!(
        c,
        "toeplitz",
        "from_parts",
        toeplitz_from_parts_for,
        256,
        1024
    );
    bench_group!(c, "raa", "from_parts", raa_from_parts_for, 256, 1024);

    bench_group!(c, "ring_lpn", "apply", ring_apply_for, 256, 1024);
    bench_group!(c, "toeplitz", "apply", toeplitz_apply_for, 256, 1024);
    bench_group!(c, "raa", "apply", raa_apply_for, 256, 1024);

    bench_group!(
        c,
        "ring_lpn",
        "apply_weight",
        |group: &mut BenchmarkGroup<'_, WallTime>, k: usize| {
            for weight in [64, 128, 192] {
                ring_apply_weight_for(group, k, weight);
            }
        },
        256
    );
    bench_group!(
        c,
        "raa",
        "apply_c",
        |group: &mut BenchmarkGroup<'_, WallTime>, k: usize| {
            for c in [2, 3, 4] {
                raa_apply_c_for(group, k, c);
            }
        },
        256
    );
}

// Construction (from parts), apply, and sensitivity groups at the
// LLM-scale block dimensions (the protocol's mask blocks at the LLM record
// lengths have n = 8192/16384), with 2048/4096 as intermediate sweep
// points between the quick sizes and the record lengths.
fn default_tdms<const MODULUS: u32>(c: &mut Criterion) {
    bench_group!(
        c,
        "ring_lpn",
        "full_sampling",
        ring_full_sampling_for,
        2_048,
        4_096,
        8_192,
        16_384
    );
    bench_group!(
        c,
        "toeplitz",
        "full_sampling",
        toeplitz_full_sampling_for,
        2_048,
        4_096,
        8_192,
        16_384
    );
    bench_group!(
        c,
        "raa",
        "full_sampling",
        raa_full_sampling_for,
        2_048,
        4_096,
        8_192,
        16_384
    );

    bench_group!(
        c,
        "ring_lpn",
        "from_parts",
        ring_from_parts_for,
        2_048,
        4_096,
        8_192,
        16_384
    );
    bench_group!(
        c,
        "toeplitz",
        "from_parts",
        toeplitz_from_parts_for,
        2_048,
        4_096,
        8_192,
        16_384
    );
    bench_group!(c, "raa", "from_parts", raa_from_parts_for, 2_048, 4_096, 8_192, 16_384);

    bench_group!(c, "ring_lpn", "apply", ring_apply_for, 2_048, 4_096, 8_192, 16_384);
    bench_group!(c, "toeplitz", "apply", toeplitz_apply_for, 2_048, 4_096, 8_192, 16_384);
    bench_group!(c, "raa", "apply", raa_apply_for, 2_048, 4_096, 8_192, 16_384);

    bench_group!(
        c,
        "ring_lpn",
        "apply_weight",
        |group: &mut BenchmarkGroup<'_, WallTime>, k: usize| {
            for weight in [192, 384, 768] {
                ring_apply_weight_for(group, k, weight);
            }
        },
        8_192
    );
    bench_group!(
        c,
        "raa",
        "apply_c",
        |group: &mut BenchmarkGroup<'_, WallTime>, k: usize| {
            for c in [2, 3, 4] {
                raa_apply_c_for(group, k, c);
            }
        },
        8_192
    );
}

fn all_tdms<const MODULUS: u32>(c: &mut Criterion) {
    if common::is_quick() {
        quick_tdms::<MODULUS>(c);
    } else {
        default_tdms::<MODULUS>(c);
    }

    // Materialize is the O(k^2) dense reference path the protocol never
    // runs (a 16384 x 16384 output costs ~2 min per iteration), so it
    // always stays at the small reference sizes. `materialize_top_rows`
    // exercises the bounded public API on one instance of the same size.
    all_materialize!(c, 256, 1024);
    bench_group!(
        c,
        "toeplitz",
        "materialize_top_rows",
        |group: &mut BenchmarkGroup<'_, WallTime>, k: usize| {
            for rows in [64, 256] {
                toeplitz_materialize_top_rows_for(group, k, rows);
            }
        },
        1024
    );
}

criterion_group! {
    name = benches;
    config = common::criterion_tuned(20, Duration::from_secs(1), Duration::from_secs(2));
    targets = all_tdms<MODULUS>
}
criterion_main!(benches);
