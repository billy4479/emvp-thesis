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

const Q: u32 = 998_244_353;
const RAA_C: usize = 4;
const TARGET_COLUMN_WEIGHT: usize = 16;

fn seeded_rng(domain: u8, size: usize) -> ChaCha20Rng {
    let mut seed = [domain; 32];
    for (slot, byte) in seed.iter_mut().zip(size.to_le_bytes()) {
        *slot ^= byte;
    }
    ChaCha20Rng::from_seed(seed)
}

fn field_values(length: usize, domain: u8) -> Vec<FieldElement<Q>> {
    let field = PrimeField::<Q>::new();
    let mut values = vec![field.element_u32(0); length];
    field.fill_uniform(&mut seeded_rng(domain, length), &mut values);
    values
}

fn elements(count: usize) -> Throughput {
    Throughput::Elements(u64::try_from(count).unwrap())
}

fn ring_inputs<const K: usize>() -> (Vec<u32>, [u32; K], SparseMatrix<Q>) {
    let field = PrimeField::<Q>::new();
    let mut rng = seeded_rng(0x31, K);
    let multiplier = std::array::from_fn(|_| field.sample_uniform(&mut rng).value());
    let rows = 2 * K;
    let target_weight = K.min(TARGET_COLUMN_WEIGHT);
    let mut offsets = Vec::with_capacity(K + 1);
    let mut row_indices = Vec::with_capacity(K * target_weight);
    let mut values = Vec::with_capacity(K * target_weight);
    offsets.push(0);
    for column in 0..K {
        for entry in 0..target_weight {
            row_indices.push((17 * column + entry) % rows);
            values.push(field.sample_uniform_nonzero(&mut rng));
        }
        offsets.push(row_indices.len());
    }
    let sparse = SparseMatrix::new(rows, K, offsets, row_indices, values).unwrap();

    // This monic polynomial is benchmark input, not an irreducibility claim.
    // The unchecked API isolates construction cost from an impractical search
    // for checked degree-64 and degree-256 modulus polynomials.
    let mut modulus = vec![0; K + 1];
    modulus[0] = Q - 3;
    modulus[K] = 1;
    (modulus, multiplier, sparse)
}

fn ring_instance<const K: usize>() -> IrreducibleRingLpn<Q, K> {
    let (modulus, multiplier, sparse) = ring_inputs::<K>();
    IrreducibleRingLpn::new_unchecked_irreducible(&modulus, multiplier, sparse).unwrap()
}

fn ring_label<const K: usize>() -> String {
    let rows = 2 * K;
    let target_weight = K.min(TARGET_COLUMN_WEIGHT);
    let extension = if K == 16 {
        String::from("extension_schoolbook")
    } else {
        format!("extension_ntt_len{}", (2 * K - 1).next_power_of_two())
    };
    format!(
        "{extension}_target_expected_per_column_t{target_weight}_p{target_weight}_over_{rows}_e_rows{rows}"
    )
}

fn ring_construction_for<const K: usize>(group: &mut BenchmarkGroup<'_, WallTime>) {
    let (modulus, multiplier, sparse) = ring_inputs::<K>();
    let target_weight = K.min(TARGET_COLUMN_WEIGHT);
    group.throughput(elements(K + 1 + K + K * target_weight));
    let label = format!("new_unchecked_irreducible_{}", ring_label::<K>());
    group.bench_function(BenchmarkId::new(label, K), |b| {
        b.iter_batched(
            || (multiplier, sparse.clone()),
            |(multiplier, sparse)| {
                black_box(
                    IrreducibleRingLpn::<Q, K>::new_unchecked_irreducible(
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

fn ring_apply_for<const K: usize>(group: &mut BenchmarkGroup<'_, WallTime>) {
    let map = ring_instance::<K>();
    let input = field_values(K, 0x32);
    let mut output = field_values(K, 0x33);
    let mut scratch = map.scratch();
    group.throughput(elements(K));
    group.bench_function(BenchmarkId::new(ring_label::<K>(), K), |b| {
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

fn ring_materialize_for<const K: usize>(group: &mut BenchmarkGroup<'_, WallTime>) {
    let map = ring_instance::<K>();
    group.throughput(elements(K * K));
    group.bench_function(BenchmarkId::new(ring_label::<K>(), K), |b| {
        b.iter(|| black_box(black_box(&map).materialize().unwrap()));
    });
}

fn ring_lpn_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("tdm_ring_lpn_construction_p998244353");
    ring_construction_for::<16>(&mut group);
    ring_construction_for::<64>(&mut group);
    if !cfg!(feature = "bench-quick") {
        ring_construction_for::<256>(&mut group);
    }
    group.finish();
}

fn ring_lpn_warmed_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("tdm_ring_lpn_warmed_apply_p998244353");
    ring_apply_for::<16>(&mut group);
    ring_apply_for::<64>(&mut group);
    if !cfg!(feature = "bench-quick") {
        ring_apply_for::<256>(&mut group);
    }
    group.finish();
}

fn ring_lpn_materialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("tdm_ring_lpn_materialize_p998244353");
    ring_materialize_for::<16>(&mut group);
    ring_materialize_for::<64>(&mut group);
    if !cfg!(feature = "bench-quick") {
        ring_materialize_for::<256>(&mut group);
    }
    group.finish();
}

fn toeplitz_instance<const K: usize>() -> ToeplitzFastProduct<Q, K> {
    ToeplitzFastProduct::sample(&mut seeded_rng(0x41, K)).unwrap()
}

fn toeplitz_label<const K: usize>(map: &ToeplitzFastProduct<Q, K>, operation: &str) -> String {
    format!(
        "{operation}_{:?}_ntt_len{}",
        map.middle().backend(),
        map.middle().transform_length()
    )
}

fn toeplitz_construction_for<const K: usize>(group: &mut BenchmarkGroup<'_, WallTime>) {
    let map = toeplitz_instance::<K>();
    let label = toeplitz_label(&map, "sample_and_cache");
    group.throughput(elements(10 * K - 3));
    group.bench_function(BenchmarkId::new(label, K), |b| {
        b.iter_batched(
            || seeded_rng(0x41, K),
            |mut rng| black_box(ToeplitzFastProduct::<Q, K>::sample(black_box(&mut rng)).unwrap()),
            BatchSize::SmallInput,
        );
    });
}

fn toeplitz_apply_for<const K: usize>(group: &mut BenchmarkGroup<'_, WallTime>) {
    let map = toeplitz_instance::<K>();
    let input = field_values(K, 0x42);
    let mut output = field_values(K, 0x43);
    let mut scratch = map.scratch();
    group.throughput(elements(K));
    group.bench_function(
        BenchmarkId::new(toeplitz_label(&map, "structured"), K),
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

    if K <= 64 {
        let dense = map.materialize().unwrap();
        let mut dense_output = field_values(K, 0x44);
        group.bench_function(BenchmarkId::new("direct_dense_crossover", K), |b| {
            b.iter(|| {
                black_box(&dense)
                    .apply(black_box(&input), black_box(&mut dense_output))
                    .unwrap();
            });
        });
    }
}

fn toeplitz_materialize_for<const K: usize>(group: &mut BenchmarkGroup<'_, WallTime>) {
    let map = toeplitz_instance::<K>();
    group.throughput(elements(K * K));
    group.bench_function(
        BenchmarkId::new(toeplitz_label(&map, "materialize"), K),
        |b| b.iter(|| black_box(black_box(&map).materialize().unwrap())),
    );
}

fn toeplitz_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("tdm_toeplitz_construction_p998244353");
    toeplitz_construction_for::<16>(&mut group);
    toeplitz_construction_for::<64>(&mut group);
    if !cfg!(feature = "bench-quick") {
        toeplitz_construction_for::<256>(&mut group);
    }
    group.finish();
}

fn toeplitz_warmed_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("tdm_toeplitz_warmed_apply_p998244353");
    toeplitz_apply_for::<16>(&mut group);
    toeplitz_apply_for::<64>(&mut group);
    if !cfg!(feature = "bench-quick") {
        toeplitz_apply_for::<256>(&mut group);
    }
    group.finish();
}

fn toeplitz_materialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("tdm_toeplitz_materialize_p998244353");
    toeplitz_materialize_for::<16>(&mut group);
    toeplitz_materialize_for::<64>(&mut group);
    if !cfg!(feature = "bench-quick") {
        toeplitz_materialize_for::<256>(&mut group);
    }
    group.finish();
}

fn raa_instance(size: usize) -> RaaWeightedProduct<Q> {
    RaaWeightedProduct::sample_nonzero(size, RAA_C, &mut seeded_rng(0x51, size)).unwrap()
}

fn raa_construction_for(group: &mut BenchmarkGroup<'_, WallTime>, size: usize) {
    group.throughput(elements(3 * size * RAA_C));
    group.bench_function(BenchmarkId::new("sample_nonzero_c4", size), |b| {
        b.iter_batched(
            || seeded_rng(0x51, size),
            |mut rng| {
                black_box(
                    RaaWeightedProduct::<Q>::sample_nonzero(
                        black_box(size),
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

fn raa_apply_for(group: &mut BenchmarkGroup<'_, WallTime>, size: usize) {
    let map = raa_instance(size);
    let input = field_values(size, 0x52);
    let mut output = field_values(size, 0x53);
    let mut scratch = map.scratch();
    group.throughput(elements(size));
    group.bench_function(BenchmarkId::new("structured_c4", size), |b| {
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

    if size <= 64 {
        let dense = map.materialize().unwrap();
        let mut dense_output = field_values(size, 0x54);
        group.bench_function(BenchmarkId::new("direct_dense_crossover", size), |b| {
            b.iter(|| {
                black_box(&dense)
                    .apply(black_box(&input), black_box(&mut dense_output))
                    .unwrap();
            });
        });
    }
}

fn raa_materialize_for(group: &mut BenchmarkGroup<'_, WallTime>, size: usize) {
    let map = raa_instance(size);
    group.throughput(elements(size * size));
    group.bench_function(BenchmarkId::new("materialize_c4", size), |b| {
        b.iter(|| black_box(black_box(&map).materialize().unwrap()));
    });
}

fn raa_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("tdm_raa_construction_p998244353");
    raa_construction_for(&mut group, 16);
    raa_construction_for(&mut group, 64);
    if !cfg!(feature = "bench-quick") {
        raa_construction_for(&mut group, 256);
    }
    group.finish();
}

fn raa_warmed_apply(c: &mut Criterion) {
    let mut group = c.benchmark_group("tdm_raa_warmed_apply_p998244353");
    raa_apply_for(&mut group, 16);
    raa_apply_for(&mut group, 64);
    if !cfg!(feature = "bench-quick") {
        raa_apply_for(&mut group, 256);
    }
    group.finish();
}

fn raa_materialize(c: &mut Criterion) {
    let mut group = c.benchmark_group("tdm_raa_materialize_p998244353");
    raa_materialize_for(&mut group, 16);
    raa_materialize_for(&mut group, 64);
    if !cfg!(feature = "bench-quick") {
        raa_materialize_for(&mut group, 256);
    }
    group.finish();
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
    targets =
        ring_lpn_construction,
        ring_lpn_warmed_apply,
        ring_lpn_materialize,
        toeplitz_construction,
        toeplitz_warmed_apply,
        toeplitz_materialize,
        raa_construction,
        raa_warmed_apply,
        raa_materialize
}
criterion_main!(benches);
