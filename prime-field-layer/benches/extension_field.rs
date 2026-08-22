#![expect(
    clippy::unwrap_used,
    clippy::significant_drop_tightening,
    reason = "benchmarks use fixed valid fields and keep paired measurements together"
)]

use std::hint::black_box;
use std::time::Duration;

use bench_common as common;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use prime_field_layer::{
    ExtensionField, PolynomialReductionPlan, StaticExtensionField, StaticPolynomialReductionPlan,
};

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

fn compare_static_dynamic_for_degree<const K: usize, const N: usize>(c: &mut Criterion) {
    let modulus = irreducible_binomial(K);
    let lhs = common::array_values::<MODULUS, K>(97);
    let rhs = common::array_values::<MODULUS, K>(12_345);

    let dynamic = ExtensionField::<MODULUS, K>::new_unchecked_irreducible(&modulus).unwrap();
    let static_field =
        StaticExtensionField::<MODULUS, K, N>::new_unchecked_irreducible(&modulus).unwrap();
    let mut dynamic_output = [0; K];
    let mut static_output = [0; K];
    let mut dynamic_scratch = dynamic.scratch();
    let mut static_scratch = static_field.scratch();
    let mut multiplication = c.benchmark_group(format!(
        "extension_static_dynamic_multiplication_p{MODULUS}_k{K}"
    ));
    common::tune_group(
        &mut multiplication,
        20,
        Duration::from_secs(1),
        Duration::from_secs(2),
    );
    multiplication.throughput(Throughput::Elements(K as u64));
    multiplication.bench_function("dynamic", |b| {
        b.iter(|| {
            dynamic
                .mul(
                    black_box(&lhs),
                    black_box(&rhs),
                    black_box(&mut dynamic_output),
                    black_box(&mut dynamic_scratch),
                )
                .unwrap();
        });
    });
    multiplication.bench_function("static", |b| {
        b.iter(|| {
            static_field
                .mul(
                    black_box(&lhs),
                    black_box(&rhs),
                    black_box(&mut static_output),
                    black_box(&mut static_scratch),
                )
                .unwrap();
        });
    });
    multiplication.finish();

    let input = common::values(2 * K - 1, MODULUS, 97);
    let dynamic = PolynomialReductionPlan::<MODULUS, K>::new(&modulus).unwrap();
    let static_plan = StaticPolynomialReductionPlan::<MODULUS, K, N>::new(&modulus).unwrap();
    let mut dynamic_output = [0; K];
    let mut static_output = [0; K];
    let mut dynamic_scratch = dynamic.scratch();
    let mut static_scratch = static_plan.scratch();
    let mut reduction = c.benchmark_group(format!(
        "extension_static_dynamic_reduction_p{MODULUS}_k{K}"
    ));
    common::tune_group(
        &mut reduction,
        20,
        Duration::from_secs(1),
        Duration::from_secs(2),
    );
    reduction.throughput(Throughput::Elements(K as u64));
    reduction.bench_function("dynamic", |b| {
        b.iter(|| {
            dynamic
                .reduce(
                    black_box(&input),
                    black_box(&mut dynamic_output),
                    black_box(&mut dynamic_scratch),
                )
                .unwrap();
        });
    });
    reduction.bench_function("static", |b| {
        b.iter(|| {
            static_plan
                .reduce(
                    black_box(&input),
                    black_box(&mut static_output),
                    black_box(&mut static_scratch),
                )
                .unwrap();
        });
    });
    reduction.finish();
}

fn static_dynamic_extension(c: &mut Criterion) {
    compare_static_dynamic_for_degree::<24, 64>(c);
    compare_static_dynamic_for_degree::<256, 512>(c);
}

criterion_group! {
    name = benches;
    config = common::criterion_tuned(20, Duration::from_secs(1), Duration::from_secs(2));
    targets = extension_multiplication, fixed_monic_reduction, static_dynamic_extension
}
criterion_main!(benches);
