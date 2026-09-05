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
        group.bench_function(BenchmarkId::new("p1073479681", length), |b| {
            b.iter(|| NttPlan::<1_073_479_681>::new(black_box(length)).unwrap());
        });
        group.bench_function(BenchmarkId::new("p2013265921", length), |b| {
            b.iter(|| NttPlan::<2_013_265_921>::new(black_box(length)).unwrap());
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
        transforms_for_modulus::<1_073_479_681>(c, length);
    }
    transforms_for_modulus::<2_013_265_921>(c, 4_096);
    transforms_for_modulus::<2_281_701_377>(c, 4_096);
}

fn bench_linear(
    group: &mut BenchmarkGroup<'_, WallTime>,
    transform_length: usize,
    lhs: &[u32],
    rhs: &[u32],
) {
    let output_length = lhs.len() + rhs.len() - 1;
    let auto = NttPlan::<1_073_479_681>::new(transform_length).unwrap();
    group.throughput(Throughput::Elements(output_length as u64));
    group.bench_function(
        BenchmarkId::new("linear_free_auto_dispatch", transform_length),
        |b| b.iter(|| linear_convolution::<1_073_479_681>(black_box(lhs), black_box(rhs)).unwrap()),
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
    let mut group = c.benchmark_group("convolution_p1073479681");
    // Small cases expose the schoolbook/NTT crossover; 4096 remains a
    // representative production-sized transform without an O(N^2) baseline.
    let lengths: &[usize] = if common::is_quick() {
        &[64, 256, 4_096]
    } else {
        &[16, 64, 128, 256, 4_096]
    };
    for &transform_length in lengths {
        let linear_lhs = common::values(transform_length / 2, 1_073_479_681, 97);
        let linear_rhs = common::values(transform_length / 2, 1_073_479_681, 12_345);
        bench_linear(&mut group, transform_length, &linear_lhs, &linear_rhs);

        let lhs = common::values(transform_length, 1_073_479_681, 97);
        let rhs = common::values(transform_length, 1_073_479_681, 12_345);
        let auto = NttPlan::<1_073_479_681>::new(transform_length).unwrap();
        let negacyclic = NegacyclicPlan::<1_073_479_681>::new(transform_length).unwrap();
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

fn fixed_operand_convolutions_for_modulus<const MODULUS: u32>(c: &mut Criterion) {
    // A fixed Toeplitz matrix reduces to convolution with fixed diagonal data;
    // repeated calls vary only the input vector and reuse the transformed data.
    let mut group = c.benchmark_group(format!("fixed_operand_linear_convolution_p{MODULUS}"));
    common::tune_group(
        &mut group,
        20,
        Duration::from_secs(1),
        Duration::from_secs(2),
    );
    let lengths: &[(usize, usize)] = if common::is_quick() {
        &[(257, 256), (2_049, 2_048)]
    } else {
        &[(65, 64), (257, 256), (1_025, 1_024), (4_097, 4_096)]
    };

    for &(fixed_length, input_length) in lengths {
        let output_length = fixed_length + input_length - 1;
        let transform_length = output_length.next_power_of_two();
        let fixed = common::values(fixed_length, MODULUS, 97);
        let input = common::values(input_length, MODULUS, 12_345);
        let plan = NttPlan::<MODULUS>::new(transform_length).unwrap();
        let prepared = plan.pretransform_linear_operand(&fixed).unwrap();
        let mut workspace = prepared.workspace();
        let mut output = vec![0; output_length];
        group.throughput(Throughput::Elements(output_length as u64));

        group.bench_function(
            BenchmarkId::new("convenience_allocating", transform_length),
            |b| {
                b.iter(|| {
                    plan.linear_convolution(black_box(&fixed), black_box(&input))
                        .unwrap()
                });
            },
        );
        group.bench_function(
            BenchmarkId::new("pretransformed_allocation_free", transform_length),
            |b| {
                b.iter(|| {
                    prepared
                        .convolve(
                            black_box(&input),
                            black_box(&mut output),
                            black_box(&mut workspace),
                        )
                        .unwrap();
                });
            },
        );
    }
    group.finish();
}

fn fixed_operand_convolutions(c: &mut Criterion) {
    fixed_operand_convolutions_for_modulus::<1_073_479_681>(c);
    fixed_operand_convolutions_for_modulus::<2_013_265_921>(c);
}

criterion_group! {
    name = benches;
    config = common::criterion_tuned(20, Duration::from_secs(1), Duration::from_secs(2));
    targets = plan_construction, transforms, convolutions, fixed_operand_convolutions
}
criterion_main!(benches);
