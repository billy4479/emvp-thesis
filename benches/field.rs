use std::hint::black_box;

use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use prime_field_layer::PrimeField;

const MODULI: [u32; 2] = [65_537, 4_294_967_291];
const BULK_LENGTHS: [usize; 3] = [256, 4_096, 65_536];

fn values(length: usize, modulus: u32, offset: u64) -> Vec<u32> {
    (0..length)
        .map(|index| ((index as u64 * 2_654_435_761 + offset) % modulus as u64) as u32)
        .collect()
}

fn construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("construction");
    for modulus in [65_537, 998_244_353, 4_294_967_291] {
        group.bench_with_input(
            BenchmarkId::from_parameter(modulus),
            &modulus,
            |b, &modulus| {
                b.iter(|| PrimeField::new(black_box(modulus)).unwrap());
            },
        );
    }
    group.finish();
}

fn scalar(c: &mut Criterion) {
    let mut group = c.benchmark_group("scalar");
    for modulus in MODULI {
        let field = PrimeField::new(modulus).unwrap();
        let lhs = modulus - 2;
        let rhs = modulus - 1;

        group.bench_with_input(BenchmarkId::new("add", modulus), &modulus, |b, _| {
            b.iter(|| field.add(black_box(lhs), black_box(rhs)));
        });
        group.bench_with_input(BenchmarkId::new("mul", modulus), &modulus, |b, _| {
            b.iter(|| field.mul(black_box(lhs), black_box(rhs)));
        });
        group.bench_with_input(BenchmarkId::new("square", modulus), &modulus, |b, _| {
            b.iter(|| field.square(black_box(lhs)));
        });
        group.bench_with_input(BenchmarkId::new("pow_u32", modulus), &modulus, |b, _| {
            b.iter(|| field.pow(black_box(lhs), black_box(u32::MAX as u64)));
        });
        group.bench_with_input(BenchmarkId::new("inv", modulus), &modulus, |b, _| {
            b.iter(|| field.inv(black_box(lhs)).unwrap());
        });
    }
    group.finish();
}

fn bulk(c: &mut Criterion) {
    let mut group = c.benchmark_group("bulk");

    for modulus in MODULI {
        let field = PrimeField::new(modulus).unwrap();
        for length in BULK_LENGTHS {
            group.throughput(Throughput::Elements(length as u64));
            let lhs = values(length, modulus, 12_345);
            let rhs = values(length, modulus, 97);
            let parameter = format!("p={modulus}/n={length}");

            group.bench_with_input(
                BenchmarkId::new("add_assign", &parameter),
                &length,
                |b, _| {
                    b.iter_batched_ref(
                        || lhs.clone(),
                        |output| {
                            field
                                .add_assign(black_box(output), black_box(&rhs))
                                .unwrap()
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
            group.bench_with_input(
                BenchmarkId::new("mul_assign", &parameter),
                &length,
                |b, _| {
                    b.iter_batched_ref(
                        || lhs.clone(),
                        |output| {
                            field
                                .mul_assign(black_box(output), black_box(&rhs))
                                .unwrap()
                        },
                        BatchSize::SmallInput,
                    );
                },
            );
            group.bench_with_input(
                BenchmarkId::new("scalar_mul_assign", &parameter),
                &length,
                |b, _| {
                    b.iter_batched_ref(
                        || lhs.clone(),
                        |output| field.scalar_mul_assign(black_box(output), black_box(modulus - 1)),
                        BatchSize::SmallInput,
                    );
                },
            );
            group.bench_with_input(BenchmarkId::new("dot", &parameter), &length, |b, _| {
                b.iter(|| field.dot(black_box(&lhs), black_box(&rhs)).unwrap());
            });
        }
    }
    group.finish();
}

criterion_group!(benches, construction, scalar, bulk);
criterion_main!(benches);
