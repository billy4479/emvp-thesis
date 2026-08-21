#![expect(
    clippy::unwrap_used,
    clippy::significant_drop_tightening,
    reason = "benchmarks use fixed valid sizes and keep paired measurements together"
)]

use std::hint::black_box;
use std::time::Duration;

use bench_common as common;
use criterion::{BatchSize, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main};
use prime_field_layer::StaticNttPlan;

fn static_case<const MODULUS: u32, const N: usize>(c: &mut Criterion) {
    let plan = StaticNttPlan::<MODULUS, N>::new().unwrap();
    let backend = format!("{:?}", plan.backend());
    let input = common::array_values::<MODULUS, N>(97);
    let rhs = plan.elements(&common::array_values::<MODULUS, N>(193));
    let elements = plan.elements(&input);

    let mut setup = c.benchmark_group(format!("static_plan_setup_p{MODULUS}_n{N}"));
    common::tune_group(
        &mut setup,
        20,
        Duration::from_secs(3),
        Duration::from_secs(2),
    );
    setup.bench_function("new", |b| {
        b.iter(|| black_box(StaticNttPlan::<MODULUS, N>::new().unwrap()));
    });
    setup.finish();

    let mut transforms = c.benchmark_group(format!("static_plan_transform_p{MODULUS}_n{N}"));
    common::tune_group(
        &mut transforms,
        20,
        Duration::from_secs(1),
        Duration::from_secs(3),
    );
    transforms.throughput(Throughput::Elements(N as u64));
    transforms.bench_function(BenchmarkId::new("forward_auto", &backend), |b| {
        b.iter_batched_ref(
            || elements,
            |values| plan.forward(black_box(values)),
            BatchSize::SmallInput,
        );
    });
    let mut transformed = elements;
    plan.forward(&mut transformed);
    let mut rhs_transformed = rhs;
    plan.forward(&mut rhs_transformed);
    transforms.bench_function(BenchmarkId::new("inverse_auto", &backend), |b| {
        b.iter_batched_ref(
            || transformed,
            |values| plan.inverse(black_box(values)),
            BatchSize::SmallInput,
        );
    });
    transforms.bench_function(BenchmarkId::new("pointwise_auto", &backend), |b| {
        b.iter_batched_ref(
            || (transformed, rhs_transformed),
            |(lhs, rhs)| {
                plan.pointwise_mul_assign(black_box(lhs), black_box(rhs));
            },
            BatchSize::SmallInput,
        );
    });
    transforms.finish();

    let mut convolution = c.benchmark_group(format!("static_plan_convolution_p{MODULUS}_n{N}"));
    common::tune_group(
        &mut convolution,
        20,
        Duration::from_secs(1),
        Duration::from_secs(3),
    );
    convolution.throughput(Throughput::Elements(N as u64));
    convolution.bench_function(BenchmarkId::new("convolve_elements_auto", &backend), |b| {
        b.iter_batched_ref(
            || (elements, plan.elements(&input)),
            |(lhs, rhs)| plan.convolve_elements(black_box(lhs), black_box(rhs)),
            BatchSize::SmallInput,
        );
    });
    convolution.finish();
}

fn static_plans(c: &mut Criterion) {
    if common::is_quick() {
        static_case::<998_244_353, 256>(c);
        static_case::<998_244_353, 4_096>(c);
    } else {
        static_case::<998_244_353, 64>(c);
        static_case::<998_244_353, 256>(c);
        static_case::<998_244_353, 1_024>(c);
        static_case::<998_244_353, 4_096>(c);
        static_case::<998_244_353, 16_384>(c);
        static_case::<2_013_265_921, 256>(c);
        static_case::<2_013_265_921, 4_096>(c);
        static_case::<2_281_701_377, 256>(c);
        static_case::<2_281_701_377, 4_096>(c);
    }
}

criterion_group! {
    name = benches;
    config = common::criterion_default();
    targets = static_plans
}
criterion_main!(benches);
