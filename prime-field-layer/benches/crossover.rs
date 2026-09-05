#![expect(
    clippy::unwrap_used,
    reason = "benchmarks keep setup beside measurements and use fixed valid inputs"
)]

use std::hint::black_box;
use std::time::Duration;

use bench_common as common;
use criterion::{BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use prime_field_layer::{FieldElement, NttPlan, PrimeField, linear_convolution};

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
    result.into_iter().map(FieldElement::value).collect()
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

fn convolutions(c: &mut Criterion) {
    // O(N^2) schoolbook baselines calibrated the schoolbook/NTT dispatch
    // threshold; re-run them only when revisiting that choice.
    if common::skip_calibration("schoolbook convolution baselines") {
        return;
    }
    let mut group = c.benchmark_group("convolution_p998244353");
    // Only the lengths where an O(N^2) baseline is still measurable; the
    // production-sized 4096 case lives in the auto/backend convolution groups.
    for transform_length in [16, 64, 128, 256] {
        let linear_lhs = common::values(transform_length / 2, 998_244_353, 97);
        let linear_rhs = common::values(transform_length / 2, 998_244_353, 12_345);
        let output_length = linear_lhs.len() + linear_rhs.len() - 1;
        group.throughput(Throughput::Elements(output_length as u64));
        group.bench_function(
            BenchmarkId::new(
                "linear_field_element_montgomery_schoolbook",
                transform_length,
            ),
            |b| {
                b.iter(|| {
                    field_element_schoolbook_linear::<998_244_353>(
                        black_box(&linear_lhs),
                        black_box(&linear_rhs),
                    )
                });
            },
        );

        let lhs = common::values(transform_length, 998_244_353, 97);
        let rhs = common::values(transform_length, 998_244_353, 12_345);
        group.throughput(Throughput::Elements(transform_length as u64));
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
                    u64_mod_schoolbook_negacyclic::<998_244_353>(black_box(&lhs), black_box(&rhs))
                });
            },
        );
    }
    group.finish();
}

fn linear_dispatch_for_modulus<const MODULUS: u32>(c: &mut Criterion) {
    let mut group = c.benchmark_group(format!("linear_free_dispatch_p{MODULUS}"));
    common::tune_group(
        &mut group,
        10,
        Duration::from_millis(500),
        Duration::from_secs(1),
    );
    // Square shapes near the schoolbook/NTT dispatch threshold; degenerate
    // aspect ratios (1x8192, 3x1728, ...) were one-time dispatch probes.
    for (lhs_length, rhs_length) in [(64, 64), (65, 65), (72, 72), (80, 80)] {
        let lhs = common::values(lhs_length, MODULUS, 97);
        let rhs = common::values(rhs_length, MODULUS, 12_345);
        let result_length = lhs_length + rhs_length - 1;
        let transform_length = result_length.next_power_of_two();
        let plan = NttPlan::<MODULUS>::new_scalar(transform_length).unwrap();
        let parameter = format!("{lhs_length}x{rhs_length}");
        group.throughput(Throughput::Elements(result_length as u64));
        group.bench_function(BenchmarkId::new("free_auto_dispatch", &parameter), |b| {
            b.iter(|| linear_convolution::<MODULUS>(black_box(&lhs), black_box(&rhs)).unwrap());
        });
        if common::calibration_enabled() {
            group.bench_function(
                BenchmarkId::new("field_element_montgomery_schoolbook", &parameter),
                |b| {
                    b.iter(|| {
                        field_element_schoolbook_linear::<MODULUS>(black_box(&lhs), black_box(&rhs))
                    });
                },
            );
        }
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
}

fn crossover(c: &mut Criterion) {
    if common::skip_in_quick_mode("crossover") {
        return;
    }
    convolutions(c);
    linear_dispatch(c);
}

criterion_group! {
    name = benches;
    config = common::criterion_tuned(20, Duration::from_secs(1), Duration::from_secs(2));
    targets = crossover
}
criterion_main!(benches);
