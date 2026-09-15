#![expect(
    clippy::unwrap_used,
    reason = "benchmarks keep setup beside measurements and use fixed valid inputs"
)]

//! One-shot linear-convolution dispatch study around the production
//! schoolbook/NTT cutoff.
//!
//! The production policy (`prime_field_layer::linear_convolution`) is one
//! global cutoff: products with `m * n <= 5184` use schoolbook, everything
//! else builds a one-shot NTT. This target measures that boundary on
//! representative odd-prime modulus/backend tiers, at exact cutoff
//! boundaries (`m * n == 5184`) and one product past them, in square and
//! highly rectangular shapes.
//!
//! Every shape runs four clearly separated cases:
//!
//! - `auto_oneshot_dispatch`: the production free `linear_convolution`
//!   (dispatch + everything for one call).
//! - `schoolbook_baseline`: a benchmark-local Montgomery-domain schoolbook
//!   kernel, independent of the production dispatch.
//! - `ntt_oneshot_setup_inclusive`: `NttPlan::new` plus the plan product,
//!   all inside the timing — the true cost of a one-shot NTT.
//! - `ntt_cached_plan`: a plan built during setup and reused; the
//!   amortized NTT cost a repeated caller sees.
//!
//! No production forcing API is used or added: the schoolbook baseline is
//! benchmark-local and the NTT cases go through the public plan API.

use std::hint::black_box;
use std::time::Duration;

use bench_common as common;
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
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
    let mut group = c.benchmark_group("convolution_p1073479681");
    // Only the lengths where an O(N^2) baseline is still measurable; the
    // production-sized 4096 case lives in the auto/backend convolution groups.
    for transform_length in [16, 64, 128, 256] {
        let linear_lhs = common::values(transform_length / 2, 1_073_479_681, 97);
        let linear_rhs = common::values(transform_length / 2, 1_073_479_681, 12_345);
        let output_length = linear_lhs.len() + linear_rhs.len() - 1;
        group.throughput(Throughput::Elements(output_length as u64));
        group.bench_function(
            BenchmarkId::new(
                "linear_field_element_montgomery_schoolbook",
                transform_length,
            ),
            |b| {
                b.iter(|| {
                    field_element_schoolbook_linear::<1_073_479_681>(
                        black_box(&linear_lhs),
                        black_box(&linear_rhs),
                    )
                });
            },
        );

        let lhs = common::values(transform_length, 1_073_479_681, 97);
        let rhs = common::values(transform_length, 1_073_479_681, 12_345);
        group.throughput(Throughput::Elements(transform_length as u64));
        group.bench_function(
            BenchmarkId::new("cyclic_u64_mod_schoolbook_baseline", transform_length),
            |b| {
                b.iter(|| {
                    u64_mod_schoolbook_cyclic::<1_073_479_681>(black_box(&lhs), black_box(&rhs))
                });
            },
        );
        group.bench_function(
            BenchmarkId::new("negacyclic_u64_mod_schoolbook_baseline", transform_length),
            |b| {
                b.iter(|| {
                    u64_mod_schoolbook_negacyclic::<1_073_479_681>(black_box(&lhs), black_box(&rhs))
                });
            },
        );
    }
    group.finish();
}

// Shapes at and just past the exact `m * n == 5184` cutoff boundary, in
// square and increasingly rectangular aspect ratios, plus two larger
// squares that bracket the suspected one-shot (setup-inclusive) crossover:
//
// | shape      | m * n | side of cutoff | aspect |
// |------------|-------|----------------|--------|
// | (72, 72)   | 5184  | exact (<=)     | 1:1    |
// | (73, 73)   | 5329  | first past     | 1:1    |
// | (80, 80)   | 6400  | well past      | 1:1    |
// | (96, 96)   | 9216  | well past      | 1:1    |
// | (112, 112) | 12544 | well past      | 1:1    |
// | (64, 81)   | 5184  | exact (<=)     | 1:1.27 |
// | (65, 81)   | 5265  | first past     | 1:1.25 |
// | (32, 162)  | 5184  | exact (<=)     | 1:5    |
// | (33, 162)  | 5346  | first past     | 1:5    |
const BOUNDARY_SHAPES: [(usize, usize); 9] = [
    (72, 72),
    (73, 73),
    (80, 80),
    (96, 96),
    (112, 112),
    (64, 81),
    (65, 81),
    (32, 162),
    (33, 162),
];

// The two boundary pairs for the additional modulus tiers (square and
// highly rectangular keep the tier comparison focused).
const TIER_SHAPES: [(usize, usize); 4] = [(72, 72), (73, 73), (32, 162), (33, 162)];

fn dispatch_shape<const MODULUS: u32>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    (lhs_length, rhs_length): (usize, usize),
) {
    let lhs = common::values(lhs_length, MODULUS, 97);
    let rhs = common::values(rhs_length, MODULUS, 12_345);
    let result_length = lhs_length + rhs_length - 1;
    let transform_length = result_length.next_power_of_two();
    // Setup-only cached plan for the `ntt_cached_plan` case.
    let plan = NttPlan::<MODULUS>::new(transform_length).unwrap();
    let parameter = format!("{lhs_length}x{rhs_length}");
    group.throughput(Throughput::Elements(result_length as u64));
    group.bench_function(BenchmarkId::new("auto_oneshot_dispatch", &parameter), |b| {
        b.iter(|| linear_convolution::<MODULUS>(black_box(&lhs), black_box(&rhs)).unwrap());
    });
    group.bench_function(BenchmarkId::new("schoolbook_baseline", &parameter), |b| {
        b.iter(|| field_element_schoolbook_linear::<MODULUS>(black_box(&lhs), black_box(&rhs)));
    });
    group.bench_function(
        BenchmarkId::new("ntt_oneshot_setup_inclusive", &parameter),
        |b| {
            b.iter(|| {
                // The whole one-shot NTT cost: plan construction (twiddle
                // tables) plus the transform-based product.
                let plan = NttPlan::<MODULUS>::new(transform_length).unwrap();
                plan.linear_convolution(black_box(&lhs), black_box(&rhs))
                    .unwrap()
            });
        },
    );
    group.bench_function(BenchmarkId::new("ntt_cached_plan", &parameter), |b| {
        b.iter(|| {
            plan.linear_convolution(black_box(&lhs), black_box(&rhs))
                .unwrap()
        });
    });
}

fn linear_dispatch_for_modulus<const MODULUS: u32>(c: &mut Criterion, shapes: &[(usize, usize)]) {
    let mut group = c.benchmark_group(format!("linear_free_dispatch_p{MODULUS}"));
    common::tune_group(
        &mut group,
        10,
        Duration::from_millis(500),
        Duration::from_secs(1),
    );
    for &shape in shapes {
        dispatch_shape::<MODULUS>(&mut group, shape);
    }
    group.finish();
}

fn linear_dispatch(c: &mut Criterion) {
    // Representative odd-prime modulus/backend tiers, matching the backend
    // selection in `prime_field_layer`: below 2^30 picks the lazy-Shoup
    // butterflies, [2^30, 2^31) the reduced-Shoup butterflies, and at or
    // above 2^31 the Montgomery fallback.
    linear_dispatch_for_modulus::<998_244_353>(c, &TIER_SHAPES);
    linear_dispatch_for_modulus::<1_073_479_681>(c, &BOUNDARY_SHAPES);
    linear_dispatch_for_modulus::<2_281_701_377>(c, &TIER_SHAPES);
}

fn crossover(c: &mut Criterion) {
    // The whole target is a one-time threshold study at sizes far below the
    // LLM scale, so it only runs under EMVP_BENCH_CALIBRATION=1.
    if common::skip_calibration("crossover dispatch-threshold studies") {
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
