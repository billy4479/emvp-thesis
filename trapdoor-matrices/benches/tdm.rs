#![expect(
    clippy::unwrap_used,
    reason = "benchmarks use fixed valid dimensions and keep setup beside measurements"
)]

use std::{hint::black_box, time::Duration};

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

const MODULUS: u32 = 998_244_353;
const RAA_C: usize = 4;
const TARGET_COLUMN_WEIGHT: usize = 16;

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
    let field = PrimeField::<MODULUS>::new();
    let mut rng = seeded_rng(0x31, k);
    let multiplier = (0..k)
        .map(|_| field.sample_uniform(&mut rng).value())
        .collect::<Box<[u32]>>();
    let rows = 2 * k;
    let target_weight = k.min(TARGET_COLUMN_WEIGHT);
    let mut offsets = Vec::with_capacity(k + 1);
    let mut row_indices = Vec::with_capacity(k * target_weight);
    let mut values = Vec::with_capacity(k * target_weight);
    offsets.push(0);
    for column in 0..k {
        for entry in 0..target_weight {
            row_indices.push((17 * column + entry) % rows);
            values.push(field.sample_uniform_nonzero(&mut rng));
        }
        offsets.push(row_indices.len());
    }
    let sparse = SparseMatrix::new(rows, k, offsets, row_indices, values).unwrap();

    // This monic polynomial is benchmark input, not an irreducibility claim.
    // The unchecked API isolates construction cost from an impractical search
    // for checked degree-64 and degree-256 modulus polynomials.
    let mut modulus = vec![0; k + 1].into_boxed_slice();
    modulus[0] = MODULUS - 3;
    modulus[k] = 1;
    (modulus, multiplier, sparse)
}

fn ring_instance(k: usize) -> IrreducibleRingLpn<MODULUS> {
    let (modulus, multiplier, sparse) = ring_inputs(k);
    IrreducibleRingLpn::new_unchecked_irreducible(k, &modulus, &multiplier, sparse).unwrap()
}

fn ring_label(k: usize) -> String {
    let rows = 2 * k;
    let target_weight = k.min(TARGET_COLUMN_WEIGHT);
    let extension = if k == 16 {
        String::from("extension_schoolbook")
    } else {
        format!("extension_ntt_len{}", (2 * k - 1).next_power_of_two())
    };
    format!(
        "{extension}_target_expected_per_column_t{target_weight}_p{target_weight}_over_{rows}_e_rows{rows}"
    )
}

fn ring_construction_for(group: &mut BenchmarkGroup<'_, WallTime>, k: usize) {
    let (modulus, multiplier, sparse) = ring_inputs(k);
    let target_weight = k.min(TARGET_COLUMN_WEIGHT);
    group.throughput(elements(k + 1 + k + k * target_weight));
    let label = format!("new_unchecked_irreducible_{}", ring_label(k));
    group.bench_function(BenchmarkId::new(label, k), |b| {
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
    group.bench_function(BenchmarkId::new(ring_label(k), k), |b| {
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
    group.bench_function(BenchmarkId::new(ring_label(k), k), |b| {
        b.iter(|| black_box(black_box(&map).materialize().unwrap()));
    });
}

fn toeplitz_instance(k: usize) -> ToeplitzFastProduct<MODULUS> {
    ToeplitzFastProduct::sample(k, &mut seeded_rng(0x41, k)).unwrap()
}

fn toeplitz_label(map: &ToeplitzFastProduct<MODULUS>, operation: &str) -> String {
    format!(
        "{operation}_{:?}_ntt_len{}",
        map.middle().backend(),
        map.middle().transform_length()
    )
}

fn toeplitz_construction_for(group: &mut BenchmarkGroup<'_, WallTime>, k: usize) {
    let map = toeplitz_instance(k);
    let label = toeplitz_label(&map, "sample_and_cache");
    group.throughput(elements(10 * k - 3));
    group.bench_function(BenchmarkId::new(label, k), |b| {
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
    group.bench_function(
        BenchmarkId::new(toeplitz_label(&map, "structured"), k),
        |b| {
            b.iter(|| {
                black_box(&map)
                    .apply(
                        black_box(&input),
                        black_box(&mut output),
                        black_box(&mut scratch),
                    )
                    .unwrap();
            });
        },
    );

    if k <= 64 {
        let dense = map.materialize().unwrap();
        let mut dense_output = field_values(k, 0x44);
        group.bench_function(BenchmarkId::new("direct_dense_crossover", k), |b| {
            b.iter(|| {
                black_box(&dense)
                    .apply(black_box(&input), black_box(&mut dense_output))
                    .unwrap();
            });
        });
    }
}

fn toeplitz_materialize_for(group: &mut BenchmarkGroup<'_, WallTime>, k: usize) {
    let map = toeplitz_instance(k);
    group.throughput(elements(k * k));
    group.bench_function(
        BenchmarkId::new(toeplitz_label(&map, "materialize"), k),
        |b| b.iter(|| black_box(black_box(&map).materialize().unwrap())),
    );
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

    if k <= 64 {
        let dense = map.materialize().unwrap();
        let mut dense_output = field_values(k, 0x54);
        group.bench_function(BenchmarkId::new("direct_dense_crossover", k), |b| {
            b.iter(|| {
                black_box(&dense)
                    .apply(black_box(&input), black_box(&mut dense_output))
                    .unwrap();
            });
        });
    }
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
            bench_group!("ring_lpn", "materialize", ring_materialize_for, $($n),*);
            bench_group!("ring_lpn", "apply", ring_apply_for, $($n),*);

            bench_group!("toeplitz", "construction", toeplitz_construction_for, $($n),*);
            bench_group!("toeplitz", "materialize", toeplitz_materialize_for, $($n),*);
            bench_group!("toeplitz", "apply", toeplitz_apply_for, $($n),*);

            bench_group!("raa", "construction", raa_construction_for, $($n),*);
            bench_group!("raa", "materialize", raa_materialize_for, $($n),*);
            bench_group!("raa", "apply", raa_apply_for, $($n),*);
        };
    }

    #[cfg(feature = "bench-quick")]
    all!(16, 256);

    #[cfg(not(feature = "bench-quick"))]
    all!(16, 64, 256, 1024);
}

fn criterion_config() -> Criterion {
    if cfg!(feature = "bench-quick") {
        Criterion::default()
            .sample_size(10)
            .warm_up_time(Duration::from_millis(250))
            .measurement_time(Duration::from_secs(1))
    } else {
        Criterion::default()
            .sample_size(20)
            .warm_up_time(Duration::from_secs(1))
            .measurement_time(Duration::from_secs(2))
    }
}

criterion_group! {
    name = benches;
    config = criterion_config();
    targets = all_tdms<MODULUS>
}
criterion_main!(benches);
