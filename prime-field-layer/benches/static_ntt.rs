#![expect(
    clippy::too_many_lines,
    clippy::unwrap_used,
    clippy::significant_drop_tightening,
    reason = "benchmarks use fixed valid sizes and keep paired measurements together"
)]

use std::{hint::black_box, time::Duration};

use criterion::{
    BatchSize, BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use prime_field_layer::{NttPlan, StaticNttPlan};

fn values<const MODULUS: u32, const N: usize>(offset: u64) -> [u32; N] {
    std::array::from_fn(|index| {
        ((index as u64 * 2_654_435_761 + offset) % u64::from(MODULUS)) as u32
    })
}

fn array_mut<T, const N: usize>(values: &mut [T]) -> &mut [T; N] {
    values.try_into().unwrap()
}

fn array_ref<T, const N: usize>(values: &[T]) -> &[T; N] {
    values.try_into().unwrap()
}

fn compare_case<const MODULUS: u32, const N: usize>(c: &mut Criterion) {
    let mut setup = c.benchmark_group(format!("static_ntt_setup_p{MODULUS}_n{N}"));
    setup.sample_size(20);
    setup.measurement_time(Duration::from_secs(2));
    setup.bench_function("dynamic", |b| {
        b.iter(|| NttPlan::<MODULUS>::new(black_box(N)).unwrap());
    });
    setup.bench_function("static", |b| {
        b.iter(|| black_box(StaticNttPlan::<MODULUS, N>::new().unwrap()));
    });
    setup.finish();

    let dynamic = NttPlan::<MODULUS>::new(N).unwrap();
    let static_plan = StaticNttPlan::<MODULUS, N>::new().unwrap();
    let dynamic_scalar = NttPlan::<MODULUS>::new_scalar(N).unwrap();
    let static_scalar = StaticNttPlan::<MODULUS, N>::new_scalar().unwrap();
    let input = values::<MODULUS, N>(97);
    let elements = dynamic.elements(&input);

    let mut transforms = c.benchmark_group(format!("static_ntt_transform_p{MODULUS}_n{N}"));
    configure(&mut transforms, N);
    transforms.bench_function(
        BenchmarkId::new("dynamic_forward_auto", format!("{:?}", dynamic.backend())),
        |b| {
            b.iter_batched_ref(
                || elements.clone(),
                |values| dynamic.forward(black_box(values)).unwrap(),
                BatchSize::SmallInput,
            );
        },
    );
    transforms.bench_function(
        BenchmarkId::new(
            "static_forward_auto",
            format!("{:?}", static_plan.backend()),
        ),
        |b| {
            b.iter_batched_ref(
                || elements.clone(),
                |values| static_plan.forward(black_box(array_mut::<_, N>(values))),
                BatchSize::SmallInput,
            );
        },
    );
    transforms.bench_function("dynamic_forward_scalar", |b| {
        b.iter_batched_ref(
            || elements.clone(),
            |values| dynamic_scalar.forward(black_box(values)).unwrap(),
            BatchSize::SmallInput,
        );
    });
    transforms.bench_function("static_forward_scalar", |b| {
        b.iter_batched_ref(
            || elements.clone(),
            |values| static_scalar.forward(black_box(array_mut::<_, N>(values))),
            BatchSize::SmallInput,
        );
    });

    let mut transformed = elements.clone();
    dynamic.forward(&mut transformed).unwrap();
    transforms.bench_function(
        BenchmarkId::new("dynamic_inverse_auto", format!("{:?}", dynamic.backend())),
        |b| {
            b.iter_batched_ref(
                || transformed.clone(),
                |values| dynamic.inverse(black_box(values)).unwrap(),
                BatchSize::SmallInput,
            );
        },
    );
    transforms.bench_function(
        BenchmarkId::new(
            "static_inverse_auto",
            format!("{:?}", static_plan.backend()),
        ),
        |b| {
            b.iter_batched_ref(
                || transformed.clone(),
                |values| static_plan.inverse(black_box(array_mut::<_, N>(values))),
                BatchSize::SmallInput,
            );
        },
    );
    let mut scalar_transformed = elements.clone();
    dynamic_scalar.forward(&mut scalar_transformed).unwrap();
    transforms.bench_function("dynamic_inverse_scalar", |b| {
        b.iter_batched_ref(
            || scalar_transformed.clone(),
            |values| dynamic_scalar.inverse(black_box(values)).unwrap(),
            BatchSize::SmallInput,
        );
    });
    transforms.bench_function("static_inverse_scalar", |b| {
        b.iter_batched_ref(
            || scalar_transformed.clone(),
            |values| static_scalar.inverse(black_box(array_mut::<_, N>(values))),
            BatchSize::SmallInput,
        );
    });
    if let (Ok(dynamic_avx2), Ok(static_avx2)) = (
        NttPlan::<MODULUS>::new_avx2(N),
        StaticNttPlan::<MODULUS, N>::new_avx2(),
    ) {
        transforms.bench_function("dynamic_forward_avx2", |b| {
            b.iter_batched_ref(
                || elements.clone(),
                |values| dynamic_avx2.forward(black_box(values)).unwrap(),
                BatchSize::SmallInput,
            );
        });
        transforms.bench_function("static_forward_avx2", |b| {
            b.iter_batched_ref(
                || elements.clone(),
                |values| static_avx2.forward(black_box(array_mut::<_, N>(values))),
                BatchSize::SmallInput,
            );
        });
        let mut avx2_transformed = elements.clone();
        dynamic_avx2.forward(&mut avx2_transformed).unwrap();
        transforms.bench_function("dynamic_inverse_avx2", |b| {
            b.iter_batched_ref(
                || avx2_transformed.clone(),
                |values| dynamic_avx2.inverse(black_box(values)).unwrap(),
                BatchSize::SmallInput,
            );
        });
        transforms.bench_function("static_inverse_avx2", |b| {
            b.iter_batched_ref(
                || avx2_transformed.clone(),
                |values| static_avx2.inverse(black_box(array_mut::<_, N>(values))),
                BatchSize::SmallInput,
            );
        });
    }
    transforms.finish();

    let rhs = dynamic.elements(&values::<MODULUS, N>(193));
    let mut convolution = c.benchmark_group(format!("static_ntt_convolution_p{MODULUS}_n{N}"));
    configure(&mut convolution, N);
    convolution.bench_function("dynamic_auto", |b| {
        b.iter_batched_ref(
            || (elements.clone(), rhs.clone()),
            |(lhs, rhs)| {
                dynamic.forward(lhs).unwrap();
                dynamic.forward(rhs).unwrap();
                dynamic.pointwise_mul_assign(lhs, rhs).unwrap();
                dynamic.inverse(black_box(lhs)).unwrap();
            },
            BatchSize::SmallInput,
        );
    });
    convolution.bench_function("static_auto", |b| {
        b.iter_batched_ref(
            || (elements.clone(), rhs.clone()),
            |(lhs, rhs)| {
                static_plan.forward(array_mut::<_, N>(lhs));
                static_plan.forward(array_mut::<_, N>(rhs));
                static_plan.pointwise_mul_assign(array_mut::<_, N>(lhs), array_ref::<_, N>(rhs));
                static_plan.inverse(black_box(array_mut::<_, N>(lhs)));
            },
            BatchSize::SmallInput,
        );
    });
    convolution.finish();
}

fn configure(group: &mut BenchmarkGroup<'_, WallTime>, length: usize) {
    group.sample_size(20);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    group.throughput(Throughput::Elements(length as u64));
}

fn static_ntt(c: &mut Criterion) {
    compare_case::<998_244_353, 8>(c);
    compare_case::<998_244_353, 16>(c);
    compare_case::<998_244_353, 32>(c);
    compare_case::<998_244_353, 64>(c);
    compare_case::<998_244_353, 128>(c);
    compare_case::<998_244_353, 256>(c);
    compare_case::<998_244_353, 1_024>(c);
    compare_case::<998_244_353, 4_096>(c);
    compare_case::<998_244_353, 16_384>(c);
    compare_case::<998_244_353, 65_536>(c);
    compare_case::<2_013_265_921, 256>(c);
    compare_case::<2_013_265_921, 4_096>(c);
    compare_case::<2_013_265_921, 16_384>(c);
    compare_case::<2_281_701_377, 256>(c);
    compare_case::<2_281_701_377, 4_096>(c);
    compare_case::<2_281_701_377, 16_384>(c);
}

criterion_group!(benches, static_ntt);
criterion_main!(benches);
