#![expect(
    clippy::unwrap_used,
    reason = "benchmarks use fixed valid fields and keep paired measurements together"
)]

//! Extension-field multiplication and reduction around the production
//! dispatch boundary.
//!
//! Production dispatch is one global cutoff
//! (`SCHOOLBOOK_EXTENSION_DEGREE = 23`): degrees up to 23 use schoolbook
//! code, and degrees 24 and above build an NTT reduction plan. This bench measures
//! both sides of that boundary instead of only the selected production
//! path, so the crossover stays evidenced rather than assumed:
//!
//! - `schoolbook_reference`: a benchmark-local schoolbook kernel
//!   (long-division reduction, or schoolbook multiplication plus that
//!   reduction). No plan, no cached tables, no construction of any kind.
//! - `production_schoolbook_cached`: the production schoolbook path with
//!   its negated-modulus table built during setup (setup excluded).
//! - `ntt_cached_plan`: the production NTT path (`PolynomialReductionPlan`
//!   or `ExtensionField`, whose NTT plan and transformed tables are built
//!   during setup, excluded from timing). Only the cached kernel call is
//!   timed.
//! - `ntt_setup_inclusive`: the full one-shot cost — plan and table
//!   construction, scratch allocation, and the kernel call, all inside the
//!   timing.
//!
//! The production NTT cases exist only at degrees where the production
//! dispatch selects NTT (the API has a single global cutoff and no forcing
//! switch, deliberately); the schoolbook reference covers the counterfactual
//! on both sides. Degrees 23 and 24 straddle the production boundary, and
//! degree 40 is a dense non-power-of-two degree: its modulus is a dense
//! irreducible polynomial (verified with the crate's Rabin test) rather
//! than a binomial; its reduction transform rounds up to length 128.
//! Setup-inclusive cases are intentionally limited to these bounded crossover
//! degrees. Production protocols retain their plans, while the protocol derive
//! benchmarks cover setup at production dimensions.

use std::hint::black_box;
use std::time::Duration;

use bench_common as common;
use criterion::{
    BenchmarkGroup, BenchmarkId, Criterion, Throughput, criterion_group, criterion_main,
    measurement::WallTime,
};
use prime_field_layer::{ExtensionField, FieldElement, PolynomialReductionPlan, PrimeField};

const MODULUS: u32 = 1_073_479_681;

// `X^k - 11` is irreducible over this field for every benched binomial
// degree: 11 is a primitive root, so the binomial irreducibility criterion
// holds for every `k` whose prime factors divide `MODULUS - 1` (Lidl &
// Niederreiter, *Finite Fields*, Thm 3.75). The previous generator
// `X^k - 3` was reducible for all benched degrees: 3 has order
// `2^16 * 273`, so `gcd(k, (MODULUS - 1)/ord(3)) > 1`.
fn irreducible_binomial(degree: usize) -> Vec<u32> {
    let mut polynomial = vec![0; degree + 1];
    polynomial[0] = MODULUS - 11;
    polynomial[degree] = 1;
    polynomial
}

// A dense irreducible degree-40 modulus over this field, found by a fixed
// deterministic search over dense monic polynomials and verified with the
// crate's Rabin irreducibility test (`ExtensionField::new`). It represents
// the dense-modulus shape the binomial family hides, at a degree whose
// reduction transform length (128) is not a power-of-two multiple of the
// degree.
const DENSE_DEGREE_40_MODULUS: [u32; 41] = [
    405_199_464,
    573_581_451,
    1_072_716_366,
    975_652_481,
    584_305_635,
    445_832_531,
    1_018_190_857,
    1_008_470_224,
    650_971_965,
    423_766_759,
    735_899_282,
    763_554_044,
    892_542_205,
    606_174_058,
    831_509_854,
    416_204_829,
    391_755_663,
    223_753_497,
    525_148_197,
    220_574_150,
    306_569_258,
    547_145_261,
    445_324_440,
    107_199_573,
    46_726_320,
    322_591_927,
    136_617_913,
    599_908_536,
    297_769_817,
    612_109_743,
    143_124_541,
    793_179_793,
    204_251_691,
    846_372_122,
    20_302_546,
    996_619_968,
    469_640_106,
    69_777_270,
    885_683_029,
    171_542_183,
    1,
];

fn modulus_for_degree(degree: usize) -> Vec<u32> {
    if degree == 40 {
        assert!(
            ExtensionField::<MODULUS>::new(40, &DENSE_DEGREE_40_MODULUS).is_ok(),
            "the benchmark's dense degree-40 modulus must remain irreducible"
        );
        DENSE_DEGREE_40_MODULUS.to_vec()
    } else {
        irreducible_binomial(degree)
    }
}

// Benchmark-local schoolbook reduction: monic long division with the
// negated lower modulus coefficients, exactly the arithmetic the production
// schoolbook path performs, but written here so the reference never
// depends on the production dispatch.
fn schoolbook_reduce_reference<const MODULUS: u32>(
    modulus: &[u32],
    product: &[u32],
    output: &mut [u32],
) {
    let field = PrimeField::<MODULUS>::new();
    let k = modulus.len() - 1;
    let mut values: Vec<_> = product
        .iter()
        .map(|&value| field.element_u32(value))
        .collect();
    values.resize(2 * k - 1, field.element_u32(0));
    let negatives: Vec<_> = modulus[..k]
        .iter()
        .map(|&coefficient| field.element_u32(field.neg_canonical(coefficient)))
        .collect();
    for degree in (k..values.len()).rev() {
        let high = values[degree];
        for (index, &negative) in negatives.iter().enumerate() {
            values[degree - k + index] += high * negative;
        }
    }
    for (slot, value) in output.iter_mut().zip(&values) {
        *slot = value.value();
    }
}

// Benchmark-local schoolbook multiplication: the O(K^2) product followed by
// the schoolbook reduction reference above. No plans anywhere.
fn schoolbook_mul_reference<const MODULUS: u32>(
    modulus: &[u32],
    lhs: &[u32],
    rhs: &[u32],
    output: &mut [u32],
) {
    let field = PrimeField::<MODULUS>::new();
    let k = modulus.len() - 1;
    let lhs: Vec<_> = lhs.iter().map(|&value| field.element_u32(value)).collect();
    let rhs: Vec<_> = rhs.iter().map(|&value| field.element_u32(value)).collect();
    let mut product = vec![field.element_u32(0); 2 * k - 1];
    for (lhs_index, &lhs) in lhs.iter().enumerate() {
        for (rhs_index, &rhs) in rhs.iter().enumerate() {
            product[lhs_index + rhs_index] += lhs * rhs;
        }
    }
    let product: Vec<_> = product.into_iter().map(FieldElement::value).collect();
    schoolbook_reduce_reference::<MODULUS>(modulus, &product, output);
}

fn reduction_cases_for_degree<const K: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    quick_mode: bool,
) {
    let modulus = modulus_for_degree(K);
    let input = common::values(2 * K - 1, MODULUS, 97);
    let mut output = [0; K];
    group.throughput(Throughput::Elements(K as u64));

    // No plan, no cached tables, no construction: the bare schoolbook
    // long division.
    group.bench_function(BenchmarkId::new("schoolbook_reference", K), |b| {
        b.iter(|| {
            schoolbook_reduce_reference::<MODULUS>(
                black_box(&modulus),
                black_box(&input),
                black_box(&mut output),
            );
        });
    });

    // The production path at this degree, with its plan and tables built
    // during setup (excluded from timing). Below the production boundary
    // this selects the schoolbook kernel; above it, NTT.
    let production_is_ntt = K > 23;
    let plan = PolynomialReductionPlan::<MODULUS>::new(K, &modulus).unwrap();
    let mut scratch = plan.scratch();
    let label = if production_is_ntt {
        "ntt_cached_plan"
    } else {
        "production_schoolbook_cached"
    };
    group.bench_function(BenchmarkId::new(label, K), |b| {
        b.iter(|| {
            plan.reduce(
                black_box(&input),
                black_box(&mut output),
                black_box(&mut scratch),
            )
            .unwrap();
        });
    });

    // The full one-shot NTT cost — plan and transformed-table construction,
    // scratch allocation, and the reduction — all inside the timing. Only
    // meaningful where the production dispatch offers NTT at all.
    if production_is_ntt && quick_mode {
        group.bench_function(BenchmarkId::new("ntt_setup_inclusive", K), |b| {
            b.iter(|| {
                let plan = PolynomialReductionPlan::<MODULUS>::new(K, black_box(&modulus)).unwrap();
                let mut scratch = plan.scratch();
                plan.reduce(black_box(&input), black_box(&mut output), &mut scratch)
                    .unwrap();
            });
        });
    }
}

fn fixed_monic_reduction(c: &mut Criterion) {
    let mut group = c.benchmark_group("fixed_monic_reduction_p1073479681");
    common::tune_group(
        &mut group,
        20,
        Duration::from_secs(1),
        Duration::from_secs(2),
    );
    let quick_mode = common::is_quick();
    if quick_mode {
        reduction_cases_for_degree::<16>(&mut group, quick_mode);
        reduction_cases_for_degree::<23>(&mut group, quick_mode);
        reduction_cases_for_degree::<24>(&mut group, quick_mode);
        reduction_cases_for_degree::<40>(&mut group, quick_mode);
    } else {
        reduction_cases_for_degree::<8_192>(&mut group, quick_mode);
        reduction_cases_for_degree::<16_384>(&mut group, quick_mode);
    }
    group.finish();
}

fn multiplication_cases_for_degree<const K: usize>(
    group: &mut BenchmarkGroup<'_, WallTime>,
    quick_mode: bool,
) {
    let modulus = modulus_for_degree(K);
    let lhs = common::array_values::<MODULUS, K>(97);
    let rhs = common::array_values::<MODULUS, K>(12_345);
    let mut output = [0; K];
    group.throughput(Throughput::Elements(K as u64));

    // No plan, no cached tables: schoolbook product plus schoolbook
    // reduction, all benchmark-local.
    group.bench_function(BenchmarkId::new("schoolbook_reference", K), |b| {
        b.iter(|| {
            schoolbook_mul_reference::<MODULUS>(
                black_box(&modulus),
                black_box(&lhs),
                black_box(&rhs),
                black_box(&mut output),
            );
        });
    });

    // The production path at this degree, with the extension and its
    // scratch built during setup (excluded from timing).
    let production_is_ntt = K > 23;
    let extension = ExtensionField::<MODULUS>::new_unchecked_irreducible(K, &modulus).unwrap();
    let mut scratch = extension.scratch();
    let label = if production_is_ntt {
        "ntt_cached_plan"
    } else {
        "production_schoolbook_cached"
    };
    group.bench_function(BenchmarkId::new(label, K), |b| {
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
    });

    // The full one-shot NTT cost — extension construction (which builds the
    // NTT reduction plan and transformed tables), scratch allocation, and
    // the multiplication — all inside the timing.
    if production_is_ntt && quick_mode {
        group.bench_function(BenchmarkId::new("ntt_setup_inclusive", K), |b| {
            b.iter(|| {
                let extension =
                    ExtensionField::<MODULUS>::new_unchecked_irreducible(K, black_box(&modulus))
                        .unwrap();
                let mut scratch = extension.scratch();
                extension
                    .mul(
                        black_box(&lhs),
                        black_box(&rhs),
                        black_box(&mut output),
                        &mut scratch,
                    )
                    .unwrap();
            });
        });
    }
}

fn extension_multiplication(c: &mut Criterion) {
    let mut group = c.benchmark_group("extension_multiplication_p1073479681");
    common::tune_group(
        &mut group,
        20,
        Duration::from_secs(1),
        Duration::from_secs(2),
    );
    let quick_mode = common::is_quick();
    if quick_mode {
        multiplication_cases_for_degree::<16>(&mut group, quick_mode);
        multiplication_cases_for_degree::<23>(&mut group, quick_mode);
        multiplication_cases_for_degree::<24>(&mut group, quick_mode);
        multiplication_cases_for_degree::<40>(&mut group, quick_mode);
    } else {
        multiplication_cases_for_degree::<8_192>(&mut group, quick_mode);
        multiplication_cases_for_degree::<16_384>(&mut group, quick_mode);
    }
    group.finish();
}

criterion_group! {
    name = benches;
    config = common::criterion_tuned(20, Duration::from_secs(1), Duration::from_secs(2));
    targets = extension_multiplication, fixed_monic_reduction
}
criterion_main!(benches);
