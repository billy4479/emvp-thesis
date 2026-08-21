#![expect(
    clippy::unwrap_used,
    reason = "benchmarks keep setup beside measurements and use fixed valid inputs"
)]

use std::hint::black_box;
use std::time::Duration;

use bench_common as common;
use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use prime_field_layer::{NegacyclicPlan, NttPlan, linear_convolution};

fn plan_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt_plan_setup_inclusive");
    for length in [256, 4_096] {
        group.bench_function(BenchmarkId::new("p998244353", length), |b| {
            b.iter(|| NttPlan::<998_244_353>::new(black_box(length)).unwrap());
        });
        if !common::is_quick() {
            group.bench_function(BenchmarkId::new("p2281701377", length), |b| {
                b.iter(|| NttPlan::<2_281_701_377>::new(black_box(length)).unwrap());
            });
        }
    }
    group.finish();
}

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
    let input = common::values(length, MODULUS, 97);
    let auto_elements = auto.elements(&input);

    group.bench_function(
        BenchmarkId::new(format!("forward_auto_{:?}", auto.backend()), length),
        |b| {
            b.iter_batched_ref(
                || auto_elements.clone(),
                |values| auto.forward(black_box(values)).unwrap(),
                BatchSize::SmallInput,
            );
        },
    );

    let mut auto_transformed = auto_elements.clone();
    auto.forward(&mut auto_transformed).unwrap();
    group.bench_function(
        BenchmarkId::new(format!("inverse_auto_{:?}", auto.backend()), length),
        |b| {
            b.iter_batched_ref(
                || auto_transformed.clone(),
                |values| auto.inverse(black_box(values)).unwrap(),
                BatchSize::SmallInput,
            );
        },
    );
    group.bench_function(
        BenchmarkId::new(format!("pointwise_auto_{:?}", auto.backend()), length),
        |b| {
            b.iter_batched_ref(
                || auto_transformed.clone(),
                |values| {
                    auto.pointwise_mul_assign(black_box(values), black_box(&auto_transformed))
                        .unwrap();
                },
                BatchSize::SmallInput,
            );
        },
    );

    let rhs = common::values(length, MODULUS, 12_345);
    group.bench_function(
        BenchmarkId::new(format!("cyclic_auto_{:?}", auto.backend()), length),
        |b| {
            b.iter(|| {
                auto.cyclic_convolution(black_box(&input), black_box(&rhs))
                    .unwrap()
            });
        },
    );
    group.finish();
}

fn transforms(c: &mut Criterion) {
    for length in [1_024, 4_096, 16_384, 65_536] {
        transforms_for_modulus::<998_244_353>(c, length);
    }
    if !common::is_quick() {
        transforms_for_modulus::<2_013_265_921>(c, 4_096);
    }
    transforms_for_modulus::<2_281_701_377>(c, 4_096);
}

fn bench_linear(
    group: &mut BenchmarkGroup<'_, WallTime>,
    transform_length: usize,
    lhs: &[u32],
    rhs: &[u32],
) {
    let output_length = lhs.len() + rhs.len() - 1;
    let auto = NttPlan::<998_244_353>::new(transform_length).unwrap();
    group.throughput(Throughput::Elements(output_length as u64));
    group.bench_function(
        BenchmarkId::new("linear_free_auto_dispatch", transform_length),
        |b| b.iter(|| linear_convolution::<998_244_353>(black_box(lhs), black_box(rhs)).unwrap()),
    );
    group.bench_function(
        BenchmarkId::new("linear_ntt_setup_inclusive", transform_length),
        |b| {
            b.iter(|| {
                NttPlan::<998_244_353>::new(transform_length)
                    .unwrap()
                    .linear_convolution(black_box(lhs), black_box(rhs))
                    .unwrap()
            });
        },
    );
    group.bench_function(
        BenchmarkId::new(
            format!("linear_cached_auto_{:?}", auto.backend()),
            transform_length,
        ),
        |b| {
            b.iter(|| {
                auto.linear_convolution(black_box(lhs), black_box(rhs))
                    .unwrap()
            });
        },
    );
}

fn convolutions(c: &mut Criterion) {
    let mut group = c.benchmark_group("convolution_p998244353");
    // Small cases expose the schoolbook/NTT crossover; 4096 remains a
    // representative production-sized transform without an O(N^2) baseline.
    let lengths: &[usize] = if common::is_quick() {
        &[64, 256, 4_096]
    } else {
        &[16, 64, 128, 256, 4_096]
    };
    for &transform_length in lengths {
        let linear_lhs = common::values(transform_length / 2, 998_244_353, 97);
        let linear_rhs = common::values(transform_length / 2, 998_244_353, 12_345);
        bench_linear(&mut group, transform_length, &linear_lhs, &linear_rhs);

        let lhs = common::values(transform_length, 998_244_353, 97);
        let rhs = common::values(transform_length, 998_244_353, 12_345);
        let auto = NttPlan::<998_244_353>::new(transform_length).unwrap();
        let negacyclic = NegacyclicPlan::<998_244_353>::new(transform_length).unwrap();
        group.throughput(Throughput::Elements(transform_length as u64));
        group.bench_function(
            BenchmarkId::new(
                format!("cyclic_cached_auto_{:?}", auto.backend()),
                transform_length,
            ),
            |b| {
                b.iter(|| {
                    auto.cyclic_convolution(black_box(&lhs), black_box(&rhs))
                        .unwrap()
                });
            },
        );
        group.bench_function(
            BenchmarkId::new(
                format!("negacyclic_cached_auto_{:?}", negacyclic.backend()),
                transform_length,
            ),
            |b| {
                b.iter(|| {
                    negacyclic
                        .convolution(black_box(&lhs), black_box(&rhs))
                        .unwrap()
                });
            },
        );
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion_tuned(20, Duration::from_secs(1), Duration::from_secs(2));
    targets = plan_construction, transforms, convolutions
}
criterion_main!(benches);
