#![expect(
    clippy::unwrap_used,
    reason = "benchmark inputs are fixed to valid field operations"
)]

use std::hint::black_box;

use bench_common as common;
use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use prime_field_layer::PrimeField;

fn bench_construction<const MODULUS: u32>(group: &mut BenchmarkGroup<'_, WallTime>) {
    group.bench_function(BenchmarkId::from_parameter(MODULUS), |b| {
        b.iter(|| black_box(PrimeField::<MODULUS>::new()));
    });
}

fn construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction");
    bench_construction::<65_537>(&mut group);
    bench_construction::<998_244_353>(&mut group);
    bench_construction::<4_294_967_291>(&mut group);
    group.finish();
}

fn bench_scalar<const MODULUS: u32>(group: &mut BenchmarkGroup<'_, WallTime>) {
    let field = PrimeField::<MODULUS>::new();
    let lhs = MODULUS - 2;
    let rhs = MODULUS - 1;

    group.bench_function(BenchmarkId::new("add", MODULUS), |b| {
        b.iter(|| field.add_canonical(black_box(lhs), black_box(rhs)));
    });
    group.bench_function(BenchmarkId::new("mul", MODULUS), |b| {
        b.iter(|| field.mul(black_box(lhs), black_box(rhs)));
    });
    group.bench_function(BenchmarkId::new("element_mul", MODULUS), |b| {
        let lhs = field.element(u64::from(lhs));
        let rhs = field.element(u64::from(rhs));
        b.iter(|| black_box(lhs) * black_box(rhs));
    });
    group.bench_function(BenchmarkId::new("square", MODULUS), |b| {
        b.iter(|| field.square(black_box(lhs)));
    });
    group.bench_function(BenchmarkId::new("pow_u32", MODULUS), |b| {
        b.iter(|| field.pow(black_box(lhs), black_box(u64::from(u32::MAX))));
    });
    group.bench_function(BenchmarkId::new("inv", MODULUS), |b| {
        b.iter(|| field.inv(black_box(lhs)).unwrap());
    });
}

fn scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar");
    bench_scalar::<65_537>(&mut group);
    bench_scalar::<4_294_967_291>(&mut group);
    group.finish();
}

fn bench_bulk<const MODULUS: u32>(group: &mut BenchmarkGroup<'_, WallTime>) {
    let field = PrimeField::<MODULUS>::new();
    let lengths: &[usize] = if common::is_quick() {
        &[256, 4_096]
    } else {
        &[256, 4_096, 65_536]
    };
    for &length in lengths {
        group.throughput(Throughput::Elements(length as u64));
        let lhs = common::values(length, MODULUS, 12_345);
        let rhs = common::values(length, MODULUS, 97);
        let lhs_elements: Vec<_> = lhs
            .iter()
            .map(|&value| field.element(u64::from(value)))
            .collect();
        let rhs_elements: Vec<_> = rhs
            .iter()
            .map(|&value| field.element(u64::from(value)))
            .collect();
        let parameter = format!("p={MODULUS}/n={length}");

        group.bench_function(BenchmarkId::new("add_assign", &parameter), |b| {
            b.iter_batched_ref(
                || lhs.clone(),
                |output| {
                    field
                        .add_assign_canonical(black_box(output), black_box(&rhs))
                        .unwrap();
                },
                BatchSize::SmallInput,
            );
        });
        group.bench_function(BenchmarkId::new("mul_assign", &parameter), |b| {
            b.iter_batched_ref(
                || lhs.clone(),
                |output| {
                    field
                        .mul_assign(black_box(output), black_box(&rhs))
                        .unwrap();
                },
                BatchSize::SmallInput,
            );
        });
        group.bench_function(BenchmarkId::new("montgomery_mul_assign", &parameter), |b| {
            b.iter_batched_ref(
                || lhs_elements.clone(),
                |output| {
                    field
                        .mul_elements_assign(black_box(output), black_box(&rhs_elements))
                        .unwrap();
                },
                BatchSize::SmallInput,
            );
        });
        group.bench_function(
            BenchmarkId::new("montgomery_scalar_mul_assign", &parameter),
            |b| {
                let scalar = field.element(u64::from(MODULUS - 1));
                b.iter_batched_ref(
                    || lhs_elements.clone(),
                    |output| field.scalar_mul_elements_assign(black_box(output), black_box(scalar)),
                    BatchSize::SmallInput,
                );
            },
        );
        group.bench_function(BenchmarkId::new("scalar_mul_assign", &parameter), |b| {
            b.iter_batched_ref(
                || lhs.clone(),
                |output| {
                    field.scalar_mul_assign_canonical(black_box(output), black_box(MODULUS - 1));
                },
                BatchSize::SmallInput,
            );
        });
        group.bench_function(BenchmarkId::new("dot", &parameter), |b| {
            b.iter(|| {
                field
                    .dot_canonical(black_box(&lhs), black_box(&rhs))
                    .unwrap()
            });
        });
    }
}

fn bulk(c: &mut Criterion) {
    let mut group = c.benchmark_group("bulk");
    bench_bulk::<65_537>(&mut group);
    bench_bulk::<4_294_967_291>(&mut group);
    group.finish();
}

fn batch_inversion(c: &mut Criterion) {
    let mut group = c.benchmark_group("batch_inversion");
    let field = PrimeField::<998_244_353>::new();
    let lengths: &[usize] = if common::is_quick() {
        &[4_096]
    } else {
        &[16, 64, 256, 4_096]
    };
    for &length in lengths {
        group.throughput(Throughput::Elements(length as u64));
        let input = common::values(length, field.modulus(), 1);
        group.bench_with_input(BenchmarkId::new("batch", length), &length, |b, _| {
            b.iter_batched_ref(
                || input.clone(),
                |values| field.batch_inv_assign(black_box(values)).unwrap(),
                BatchSize::SmallInput,
            );
        });

        let element_input: Vec<_> = input
            .iter()
            .map(|&value| field.element_u32(value))
            .collect();
        let mut element_output = vec![field.element_u32(0); length];
        group.bench_with_input(BenchmarkId::new("elements", length), &length, |b, _| {
            b.iter(|| {
                field
                    .batch_inv_elements(black_box(&element_input), black_box(&mut element_output))
                    .unwrap();
                black_box(&element_output);
            });
        });
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion_default();
    targets = construction, scalar, bulk, batch_inversion
}
criterion_main!(benches);
