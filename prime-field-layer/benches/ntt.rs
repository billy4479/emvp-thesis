#![expect(
    clippy::too_many_lines,
    clippy::unwrap_used,
    reason = "benchmarks keep setup beside measurements and use fixed valid inputs"
)]

use std::hint::black_box;
use std::time::Duration;

use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use prime_field_layer::{NegacyclicPlan, NttPlan, PrimeField, linear_convolution};

fn values(length: usize, modulus: u32, offset: u64) -> Vec<u32> {
    (0..length)
        .map(|index| ((index as u64 * 2_654_435_761 + offset) % u64::from(modulus)) as u32)
        .collect()
}

fn field_element_schoolbook_linear<const MODULUS: u32>(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    let field = PrimeField::<MODULUS>::new();
    let lhs: Vec<_> = lhs.iter().map(|&value| field.element_u32(value)).collect();
    let rhs: Vec<_> = rhs.iter().map(|&value| field.element_u32(value)).collect();
    let mut result = vec![field.element_u32(0); lhs.len() + rhs.len() - 1];
    for (lhs_index, &lhs) in lhs.iter().enumerate() {
        for (rhs_index, &rhs) in rhs.iter().enumerate() {
            result[lhs_index + rhs_index] += lhs * rhs;
        }
    }
    result
        .into_iter()
        .map(prime_field_layer::FieldElement::value)
        .collect()
}

fn u64_mod_schoolbook_cyclic<const MODULUS: u32>(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    let mut result = vec![0; lhs.len()];
    for (lhs_index, &lhs_value) in lhs.iter().enumerate() {
        for (rhs_index, &rhs_value) in rhs.iter().enumerate() {
            let index = (lhs_index + rhs_index) % lhs.len();
            result[index] = ((u64::from(result[index])
                + u64::from(lhs_value) * u64::from(rhs_value))
                % u64::from(MODULUS)) as u32;
        }
    }
    result
}

fn u64_mod_schoolbook_negacyclic<const MODULUS: u32>(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    let mut result = vec![0; lhs.len()];
    for (lhs_index, &lhs_value) in lhs.iter().enumerate() {
        for (rhs_index, &rhs_value) in rhs.iter().enumerate() {
            let degree = lhs_index + rhs_index;
            let product = u64::from(lhs_value) * u64::from(rhs_value) % u64::from(MODULUS);
            let (index, contribution) = if degree < lhs.len() {
                (degree, product)
            } else {
                (degree - lhs.len(), u64::from(MODULUS) - product)
            };
            result[index] = ((u64::from(result[index]) + contribution) % u64::from(MODULUS)) as u32;
        }
    }
    result
}

fn plan_construction(c: &mut Criterion) {
    let mut group = c.benchmark_group("ntt_plan_setup_inclusive");
    for length in [256, 4_096] {
        group.bench_function(BenchmarkId::new("p998244353", length), |b| {
            b.iter(|| NttPlan::<998_244_353>::new(black_box(length)).unwrap());
        });
        group.bench_function(BenchmarkId::new("p2281701377", length), |b| {
            b.iter(|| NttPlan::<2_281_701_377>::new(black_box(length)).unwrap());
        });
    }
    group.finish();
}

fn transforms_for_modulus<const MODULUS: u32>(c: &mut Criterion, length: usize) {
    let mut group = c.benchmark_group(format!("ntt_cached_p{MODULUS}"));
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(1));
    group.throughput(Throughput::Elements(length as u64));
    let auto = NttPlan::<MODULUS>::new(length).unwrap();
    let scalar = NttPlan::<MODULUS>::new_scalar(length).unwrap();
    let avx2 = NttPlan::<MODULUS>::new_avx2(length).ok();
    let compare_scalar = auto.backend() != scalar.backend();
    let compare_avx2 = avx2
        .as_ref()
        .is_some_and(|avx2| auto.backend() != avx2.backend());
    let input = values(length, MODULUS, 97);
    let auto_elements = auto.elements(&input);
    let scalar_elements = scalar.elements(&input);

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

    let rhs = values(length, MODULUS, 12_345);
    group.bench_function(
        BenchmarkId::new(format!("cyclic_auto_{:?}", auto.backend()), length),
        |b| {
            b.iter(|| {
                auto.cyclic_convolution(black_box(&input), black_box(&rhs))
                    .unwrap()
            });
        },
    );
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

fn bench_linear(
    group: &mut BenchmarkGroup<'_, WallTime>,
    transform_length: usize,
    lhs: &[u32],
    rhs: &[u32],
) {
    let output_length = lhs.len() + rhs.len() - 1;
    let auto = NttPlan::<998_244_353>::new(transform_length).unwrap();
    let scalar = NttPlan::<998_244_353>::new_scalar(transform_length).unwrap();
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
    if auto.backend() != scalar.backend() {
        group.bench_function(
            BenchmarkId::new("linear_cached_forced_scalar", transform_length),
            |b| {
                b.iter(|| {
                    scalar
                        .linear_convolution(black_box(lhs), black_box(rhs))
                        .unwrap()
                });
            },
        );
    }
    if transform_length <= 256 {
        group.bench_function(
            BenchmarkId::new(
                "linear_field_element_montgomery_schoolbook",
                transform_length,
            ),
            |b| {
                b.iter(|| {
                    field_element_schoolbook_linear::<998_244_353>(black_box(lhs), black_box(rhs))
                });
            },
        );
    }
}

fn convolutions(c: &mut Criterion) {
    let mut group = c.benchmark_group("convolution_p998244353");
    // Small cases expose the schoolbook/NTT crossover; 4096 remains a
    // representative production-sized transform without an O(N^2) baseline.
    for transform_length in [16, 64, 128, 256, 4_096] {
        let linear_lhs = values(transform_length / 2, 998_244_353, 97);
        let linear_rhs = values(transform_length / 2, 998_244_353, 12_345);
        bench_linear(&mut group, transform_length, &linear_lhs, &linear_rhs);

        let lhs = values(transform_length, 998_244_353, 97);
        let rhs = values(transform_length, 998_244_353, 12_345);
        let auto = NttPlan::<998_244_353>::new(transform_length).unwrap();
        let scalar = NttPlan::<998_244_353>::new_scalar(transform_length).unwrap();
        let negacyclic = NegacyclicPlan::<998_244_353>::new(transform_length).unwrap();
        let scalar_negacyclic =
            NegacyclicPlan::<998_244_353>::new_scalar(transform_length).unwrap();
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
        if auto.backend() != scalar.backend() {
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
        if transform_length <= 256 {
            group.bench_function(
                BenchmarkId::new("cyclic_u64_mod_schoolbook_baseline", transform_length),
                |b| {
                    b.iter(|| {
                        u64_mod_schoolbook_cyclic::<998_244_353>(black_box(&lhs), black_box(&rhs))
                    });
                },
            );
            group.bench_function(
                BenchmarkId::new("negacyclic_u64_mod_schoolbook_baseline", transform_length),
                |b| {
                    b.iter(|| {
                        u64_mod_schoolbook_negacyclic::<998_244_353>(
                            black_box(&lhs),
                            black_box(&rhs),
                        )
                    });
                },
            );
        }
    }
    group.finish();
}

fn linear_dispatch_for_modulus<const MODULUS: u32>(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("linear_free_dispatch_p{MODULUS}"));
    group.sample_size(10);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(1));
    for (lhs_length, rhs_length) in [
        (64, 64),
        (65, 65),
        (72, 72),
        (80, 80),
        (3, 1_728),
        (3, 1_729),
        (1, 8_192),
        (2, 4_097),
    ] {
        let lhs = values(lhs_length, MODULUS, 97);
        let rhs = values(rhs_length, MODULUS, 12_345);
        let result_length = lhs_length + rhs_length - 1;
        let transform_length = result_length.next_power_of_two();
        let plan = NttPlan::<MODULUS>::new_scalar(transform_length).unwrap();
        let parameter = format!("{lhs_length}x{rhs_length}");
        group.throughput(Throughput::Elements(result_length as u64));
        group.bench_function(BenchmarkId::new("free_auto_dispatch", &parameter), |b| {
            b.iter(|| linear_convolution::<MODULUS>(black_box(&lhs), black_box(&rhs)).unwrap());
        });
        group.bench_function(
            BenchmarkId::new("field_element_montgomery_schoolbook", &parameter),
            |b| {
                b.iter(|| {
                    field_element_schoolbook_linear::<MODULUS>(black_box(&lhs), black_box(&rhs))
                });
            },
        );
        group.bench_function(BenchmarkId::new("cached_scalar_ntt", &parameter), |b| {
            b.iter(|| {
                plan.linear_convolution(black_box(&lhs), black_box(&rhs))
                    .unwrap()
            });
        });
    }
    group.finish();
}

fn linear_dispatch(c: &mut Criterion) {
    linear_dispatch_for_modulus::<998_244_353>(c);
    linear_dispatch_for_modulus::<2_281_701_377>(c);
}

criterion_group! {
    name = benches;
    config = Criterion::default()
        .sample_size(20)
        .warm_up_time(Duration::from_secs(1))
        .measurement_time(Duration::from_secs(2));
    targets = plan_construction, transforms, convolutions, linear_dispatch
}
criterion_main!(benches);
