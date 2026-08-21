#![expect(
    clippy::unwrap_used,
    reason = "benchmarks keep setup beside measurements and use fixed valid inputs"
)]

use std::hint::black_box;
use std::time::Duration;

use bench_common as common;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use prime_field_layer::{NegacyclicPlan, NttPlan};

fn transforms_for_modulus<const MODULUS: u32>(c: &mut Criterion, length: usize) {
    let mut group = c.benchmark_group(format!("ntt_cached_p{MODULUS}"));
    common::tune_group(
        &mut group,
        10,
        Duration::from_millis(500),
        Duration::from_secs(1),
    );
    group.throughput(Throughput::Elements(length as u64));
    let auto = NttPlan::<MODULUS>::new(length).unwrap();
    let scalar = NttPlan::<MODULUS>::new_scalar(length).unwrap();
    let avx2 = NttPlan::<MODULUS>::new_avx2(length).ok();
    let compare_scalar = auto.backend() != scalar.backend();
    let compare_avx2 = avx2
        .as_ref()
        .is_some_and(|avx2| auto.backend() != avx2.backend());
    let input = common::values(length, MODULUS, 97);
    let scalar_elements = scalar.elements(&input);

    if compare_scalar {
        group.bench_function(BenchmarkId::new("forward_forced_scalar", length), |b| {
            b.iter_batched_ref(
                || scalar_elements.clone(),
                |values| scalar.forward(black_box(values)).unwrap(),
                BatchSize::SmallInput,
            );
        });
    }
    if compare_avx2 && let Some(avx2) = &avx2 {
        let avx2_elements = avx2.elements(&input);
        group.bench_function(
            BenchmarkId::new("forward_explicit_avx2_butterflies", length),
            |b| {
                b.iter_batched_ref(
                    || avx2_elements.clone(),
                    |values| avx2.forward(black_box(values)).unwrap(),
                    BatchSize::SmallInput,
                );
            },
        );
    }

    if compare_scalar {
        let mut scalar_transformed = scalar_elements.clone();
        scalar.forward(&mut scalar_transformed).unwrap();
        group.bench_function(BenchmarkId::new("inverse_forced_scalar", length), |b| {
            b.iter_batched_ref(
                || scalar_transformed.clone(),
                |values| scalar.inverse(black_box(values)).unwrap(),
                BatchSize::SmallInput,
            );
        });
        group.bench_function(BenchmarkId::new("pointwise_forced_scalar", length), |b| {
            b.iter_batched_ref(
                || scalar_transformed.clone(),
                |values| {
                    scalar
                        .pointwise_mul_assign(black_box(values), black_box(&scalar_transformed))
                        .unwrap();
                },
                BatchSize::SmallInput,
            );
        });
    }
    if compare_avx2 && let Some(avx2) = &avx2 {
        let mut transformed = avx2.elements(&input);
        avx2.forward(&mut transformed).unwrap();
        group.bench_function(
            BenchmarkId::new("inverse_explicit_avx2_butterflies", length),
            |b| {
                b.iter_batched_ref(
                    || transformed.clone(),
                    |values| avx2.inverse(black_box(values)).unwrap(),
                    BatchSize::SmallInput,
                );
            },
        );
    }

    let rhs = common::values(length, MODULUS, 12_345);
    if compare_scalar {
        group.bench_function(BenchmarkId::new("cyclic_forced_scalar", length), |b| {
            b.iter(|| {
                scalar
                    .cyclic_convolution(black_box(&input), black_box(&rhs))
                    .unwrap()
            });
        });
    }
    if compare_avx2 && let Some(avx2) = &avx2 {
        group.bench_function(
            BenchmarkId::new("cyclic_explicit_avx2_butterflies", length),
            |b| {
                b.iter(|| {
                    avx2.cyclic_convolution(black_box(&input), black_box(&rhs))
                        .unwrap()
                });
            },
        );
    }
    group.finish();
}

fn transforms(c: &mut Criterion) {
    for length in [1_024, 4_096, 16_384, 65_536] {
        transforms_for_modulus::<998_244_353>(c, length);
    }
    transforms_for_modulus::<2_013_265_921>(c, 4_096);
    transforms_for_modulus::<2_281_701_377>(c, 4_096);
}

fn convolutions(c: &mut Criterion) {
    let mut group = c.benchmark_group("convolution_p998244353");
    for transform_length in [16, 64, 128, 256, 4_096] {
        let linear_lhs = common::values(transform_length / 2, 998_244_353, 97);
        let linear_rhs = common::values(transform_length / 2, 998_244_353, 12_345);
        let output_length = linear_lhs.len() + linear_rhs.len() - 1;
        let auto = NttPlan::<998_244_353>::new(transform_length).unwrap();
        let scalar = NttPlan::<998_244_353>::new_scalar(transform_length).unwrap();
        let compare_scalar = auto.backend() != scalar.backend();
        if compare_scalar {
            group.throughput(Throughput::Elements(output_length as u64));
            group.bench_function(
                BenchmarkId::new("linear_cached_forced_scalar", transform_length),
                |b| {
                    b.iter(|| {
                        scalar
                            .linear_convolution(black_box(&linear_lhs), black_box(&linear_rhs))
                            .unwrap()
                    });
                },
            );
        }

        let lhs = common::values(transform_length, 998_244_353, 97);
        let rhs = common::values(transform_length, 998_244_353, 12_345);
        let scalar_negacyclic =
            NegacyclicPlan::<998_244_353>::new_scalar(transform_length).unwrap();
        group.throughput(Throughput::Elements(transform_length as u64));
        if compare_scalar {
            group.bench_function(
                BenchmarkId::new("cyclic_cached_forced_scalar", transform_length),
                |b| {
                    b.iter(|| {
                        scalar
                            .cyclic_convolution(black_box(&lhs), black_box(&rhs))
                            .unwrap()
                    });
                },
            );
            group.bench_function(
                BenchmarkId::new("negacyclic_cached_forced_scalar", transform_length),
                |b| {
                    b.iter(|| {
                        scalar_negacyclic
                            .convolution(black_box(&lhs), black_box(&rhs))
                            .unwrap()
                    });
                },
            );
        }
    }
    group.finish();
}

fn backends(c: &mut Criterion) {
    if common::skip_in_quick_mode("backends") {
        return;
    }
    transforms(c);
    convolutions(c);
}

criterion_group! {
    name = benches;
    config = common::criterion_tuned(20, Duration::from_secs(1), Duration::from_secs(2));
    targets = backends
}
criterion_main!(benches);
