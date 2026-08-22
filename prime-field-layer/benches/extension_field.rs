#![expect(
    clippy::unwrap_used,
    reason = "benchmarks use fixed valid fields and modulus polynomials"
)]

use std::hint::black_box;
use std::time::Duration;

use bench_common as common;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use prime_field_layer::{ExtensionField, PolynomialReductionPlan};

const MODULUS: u32 = 998_244_353;

fn irreducible_binomial(degree: usize) -> Vec<u32> {
    let mut polynomial = vec![0; degree + 1];
    polynomial[0] = MODULUS - 3;
    polynomial[degree] = 1;
    polynomial
}

fn multiplication_for_degree<const K: usize>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
) {
    let modulus = irreducible_binomial(K);
    let extension = ExtensionField::<MODULUS, K>::new_unchecked_irreducible(&modulus).unwrap();
    let lhs = common::array_values::<MODULUS, K>(97);
    let rhs = common::array_values::<MODULUS, K>(12_345);
    let mut output = [0; K];
    let mut scratch = extension.scratch();
    group.throughput(Throughput::Elements(K as u64));
    group.bench_function(
        BenchmarkId::new(format!("{:?}", extension.algorithm()), K),
        |b| {
            b.iter(|| {
                extension
                    .mul(
                        black_box(&lhs),
                        black_box(&rhs),
                        black_box(&mut output),
                        black_box(&mut scratch),
                    )
                    .unwrap();
            });
        },
    );
}

fn extension_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("extension_multiplication_p998244353");
    common::tune_group(
        &mut group,
        20,
        Duration::from_secs(1),
        Duration::from_secs(2),
    );
    multiplication_for_degree::<16>(&mut group);
    multiplication_for_degree::<24>(&mut group);
    multiplication_for_degree::<256>(&mut group);
    group.finish();
}

fn reduction_for_degree<const K: usize>(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
) {
    let modulus = irreducible_binomial(K);
    let reduction = PolynomialReductionPlan::<MODULUS, K>::new(&modulus).unwrap();
    let input = common::values(2 * K - 1, MODULUS, 97);
    let mut output = [0; K];
    let mut scratch = reduction.scratch();
    group.throughput(Throughput::Elements(K as u64));
    group.bench_function(
        BenchmarkId::new(format!("{:?}", reduction.algorithm()), K),
        |b| {
            b.iter(|| {
                reduction
                    .reduce(
                        black_box(&input),
                        black_box(&mut output),
                        black_box(&mut scratch),
                    )
                    .unwrap();
            });
        },
    );
}

fn fixed_monic_reduction(c: &mut Criterion) {
    let mut group = c.benchmark_group("fixed_monic_reduction_p998244353");
    common::tune_group(
        &mut group,
        20,
        Duration::from_secs(1),
        Duration::from_secs(2),
    );
    reduction_for_degree::<16>(&mut group);
    reduction_for_degree::<24>(&mut group);
    reduction_for_degree::<256>(&mut group);
    group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion_tuned(20, Duration::from_secs(1), Duration::from_secs(2));
    targets = extension_multiplication, fixed_monic_reduction
}
criterion_main!(benches);
