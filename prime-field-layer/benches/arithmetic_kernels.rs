#![expect(
    clippy::unwrap_used,
    reason = "benchmark inputs establish that these operations must succeed"
)]

use std::hint::black_box;

use bench_common as common;
use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use prime_field_layer::{
    FieldElement, IndexedValue, PrimeField, batched_weighted_inclusive_scan_assign,
    sparse_accumulate, weighted_inclusive_scan_assign,
};

fn field_elements<const MODULUS: u32>(length: usize, offset: u64) -> Vec<FieldElement<MODULUS>> {
    let field = PrimeField::<MODULUS>::new();
    common::values(length, MODULUS, offset)
        .into_iter()
        .map(|value| field.element_u32(value))
        .collect()
}

fn bench_weighted_scan<const MODULUS: u32>(group: &mut BenchmarkGroup<'_, WallTime>) {
    // The default path keeps only the LLM-scale vector length; smaller
    // sizes are noisy and live in the quick mode instead.
    let lengths: &[usize] = if common::is_quick() {
        &[256, 4_096]
    } else {
        &[65_536]
    };

    for &length in lengths {
        group.throughput(Throughput::Elements(length as u64));
        let values = field_elements::<MODULUS>(length, 12_345);
        let weights = field_elements::<MODULUS>(length, 97);
        group.bench_function(
            BenchmarkId::new("weighted_scan", format!("p={MODULUS}/n={length}")),
            |b| {
                b.iter_batched_ref(
                    || values.clone(),
                    |output| {
                        weighted_inclusive_scan_assign(black_box(output), black_box(&weights))
                            .unwrap();
                    },
                    BatchSize::SmallInput,
                );
            },
        );
    }
}

fn weighted_scan(c: &mut Criterion) {
    let mut group = c.benchmark_group("arithmetic_kernels/weighted_scan");
    bench_weighted_scan::<1_073_479_681>(&mut group);
    bench_weighted_scan::<2_013_265_921>(&mut group);
    let positions = 512;
    let width = 8;
    let values = field_elements::<1_073_479_681>(positions * width, 12_345);
    let weights = field_elements::<1_073_479_681>(positions, 97);
    group.throughput(Throughput::Elements((positions * width) as u64));
    group.bench_function(
        BenchmarkId::new(
            "batched_weighted_scan",
            format!("p=1073479681/positions={positions}/width={width}"),
        ),
        |b| {
            b.iter_batched_ref(
                || values.clone(),
                |output| {
                    batched_weighted_inclusive_scan_assign(
                        black_box(output),
                        black_box(&weights),
                        black_box(width),
                    )
                    .unwrap();
                },
                BatchSize::SmallInput,
            );
        },
    );
    group.finish();
}

fn bench_sparse_accumulation<const MODULUS: u32>(group: &mut BenchmarkGroup<'_, WallTime>) {
    let output_len = 4_096;
    let counts: &[usize] = if common::is_quick() {
        &[64, 1_024]
    } else {
        &[16_384]
    };

    for &count in counts {
        group.throughput(Throughput::Elements(count as u64));
        let output = field_elements::<MODULUS>(output_len, 12_345);
        let entry_values = field_elements::<MODULUS>(count, 97);
        let entries: Vec<_> = entry_values
            .into_iter()
            .enumerate()
            .map(|(position, value)| {
                IndexedValue::new(position.wrapping_mul(2_654_435_761) % output_len, value)
            })
            .collect();
        group.bench_function(
            BenchmarkId::new(
                "sparse_accumulate",
                format!("p={MODULUS}/output={output_len}/entries={count}"),
            ),
            |b| {
                b.iter_batched_ref(
                    || output.clone(),
                    |output| sparse_accumulate(black_box(output), black_box(&entries)).unwrap(),
                    BatchSize::SmallInput,
                );
            },
        );
    }
}

fn sparse_accumulation(c: &mut Criterion) {
    let mut group = c.benchmark_group("arithmetic_kernels/sparse_accumulation");
    bench_sparse_accumulation::<1_073_479_681>(&mut group);
    bench_sparse_accumulation::<2_013_265_921>(&mut group);
    group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion_default();
    targets = weighted_scan, sparse_accumulation
}
criterion_main!(benches);
