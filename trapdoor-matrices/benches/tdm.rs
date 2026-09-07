#![expect(
    clippy::unwrap_used,
    reason = "benchmarks use fixed valid dimensions and keep setup beside measurements"
)]

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
    IrreducibleRingLpn, RaaWeightedProduct, SparseMatrix, ToeplitzFastProduct,
};

const MODULUS: u32 = 1_073_479_681;
const RAA_C: usize = 4;
// Column weight `t` of the secret sparse matrix `E` (the paper's expected
// nonzero count per column, measured exactly instead of in expectation),
// sized to the project policy floor `POLICY_WEIGHT_FLOOR`.
const TARGET_COLUMN_WEIGHT: usize = 192;

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

fn ring_inputs(k: usize) -> (Box<[u32]>, Box<[u32]>, SparseMatrix<MODULUS>) {
    // Not `IrreducibleRingLpn::sample`: the construction benchmark needs the
    // raw parts to time `new_unchecked_irreducible` alone, and the sparse
    // matrix must have an exact column weight so the measured work does not
    // depend on Bernoulli sampling luck.
    let field = PrimeField::<MODULUS>::new();
    let multiplier = (0..k)
        .map(|_| field.sample_uniform(&mut seeded_rng(0x31, k)).value())
        .collect::<Box<[u32]>>();
    let target_weight = k.min(TARGET_COLUMN_WEIGHT);
    let sparse = trapdoor_matrices::testing::fixed_weight_sparse::<MODULUS, _>(
        k,
        target_weight,
        &mut seeded_rng(0x31, k),
    )
    .unwrap();

    // The automatic binomial modulus is irreducible by the binomial
    // criterion, so the unchecked constructor is sound here and isolates
    // construction cost from Rabin's irreducibility search.
    let modulus = trapdoor_matrices::automatic_ring_modulus::<MODULUS>(k).unwrap();
    (modulus, multiplier, sparse)
}

fn ring_instance(k: usize) -> IrreducibleRingLpn<MODULUS> {
    let (modulus, multiplier, sparse) = ring_inputs(k);
    IrreducibleRingLpn::new_unchecked_irreducible(k, &modulus, &multiplier, sparse).unwrap()
}

fn ring_construction_for(group: &mut BenchmarkGroup<'_, WallTime>, k: usize) {
    let (modulus, multiplier, sparse) = ring_inputs(k);
    let target_weight = k.min(TARGET_COLUMN_WEIGHT);
    group.throughput(elements(k + 1 + k + k * target_weight));
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

fn ring_materialize_for(group: &mut BenchmarkGroup<'_, WallTime>, k: usize) {
    let map = ring_instance(k);
    group.throughput(elements(k * k));
    group.bench_function(BenchmarkId::from_parameter(k), |b| {
        b.iter(|| black_box(black_box(&map).materialize().unwrap()));
    });
}

fn toeplitz_instance(k: usize) -> ToeplitzFastProduct<MODULUS> {
    ToeplitzFastProduct::sample(k, &mut seeded_rng(0x41, k)).unwrap()
}

fn toeplitz_construction_for(group: &mut BenchmarkGroup<'_, WallTime>, k: usize) {
    group.throughput(elements(10 * k - 3));
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

fn raa_instance(k: usize) -> RaaWeightedProduct<MODULUS> {
    RaaWeightedProduct::sample_nonzero(k, RAA_C, &mut seeded_rng(0x51, k)).unwrap()
}

fn raa_construction_for(group: &mut BenchmarkGroup<'_, WallTime>, k: usize) {
    group.throughput(elements(3 * k * RAA_C));
    group.bench_function(BenchmarkId::new("sample_nonzero_c4", k), |b| {
        b.iter_batched(
            || seeded_rng(0x51, k),
            |mut rng| {
                black_box(
                    RaaWeightedProduct::<MODULUS>::sample_nonzero(
                        black_box(k),
                        RAA_C,
                        black_box(&mut rng),
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

fn raa_materialize_for(group: &mut BenchmarkGroup<'_, WallTime>, k: usize) {
    let map = raa_instance(k);
    group.throughput(elements(k * k));
    group.bench_function(BenchmarkId::new("materialize_c4", k), |b| {
        b.iter(|| black_box(black_box(&map).materialize().unwrap()));
    });
}

fn all_tdms<const MODULUS: u32>(c: &mut Criterion) {
    macro_rules! fn_for_each {
        ($func:ident, $group:ident, $($n:expr),* $(,)?) => {
            $(
                $func(&mut $group, $n);
            )*
        };
    }

    macro_rules! bench_group {
        ($tdm_name:literal, $fn_name:literal, $func:ident, $($n:expr),* $(,)?) => {{
            let mut group =
                c.benchmark_group(format!("tdm_{}_{}_p{}", $tdm_name, $fn_name, MODULUS));

            fn_for_each!($func, group, $($n),*);

            group.finish();
        }};
    }

    macro_rules! all {
        ($($n:expr),* $(,)?) => {
            bench_group!("ring_lpn", "construction", ring_construction_for, $($n),*);
            bench_group!("toeplitz", "construction", toeplitz_construction_for, $($n),*);
            bench_group!("raa", "construction", raa_construction_for, $($n),*);

            bench_group!("ring_lpn", "apply", ring_apply_for, $($n),*);
            bench_group!("toeplitz", "apply", toeplitz_apply_for, $($n),*);
            bench_group!("raa", "apply", raa_apply_for, $($n),*);
        };
    }

    macro_rules! all_materialize {
        ($($n:expr),* $(,)?) => {
            bench_group!("ring_lpn", "materialize", ring_materialize_for, $($n),*);
            bench_group!("toeplitz", "materialize", toeplitz_materialize_for, $($n),*);
            bench_group!("raa", "materialize", raa_materialize_for, $($n),*);
        };
    }

    // Construction and apply are the protocol-relevant paths; the default
    // suite uses the LLM-scale block dimensions (the protocol's mask blocks
    // at the LLM record lengths have n = 8192/16384).
    if common::is_quick() {
        all!(256, 1024);
    } else {
        all!(8_192, 16_384);
    }
    // Materialize is the O(k^2) dense reference path the protocol never
    // runs (a 16384 x 16384 output costs ~2 min per iteration), so it
    // always stays at the small reference sizes.
    all_materialize!(256, 1024);
}

criterion_group! {
    name = benches;
    config = common::criterion_tuned(20, Duration::from_secs(1), Duration::from_secs(2));
    targets = all_tdms<MODULUS>
}
criterion_main!(benches);
