use std::time::Duration;

use criterion::{BenchmarkGroup, Criterion, measurement::WallTime};

const QUICK_SAMPLE_SIZE: usize = 10;
const QUICK_WARM_UP_TIME: Duration = Duration::from_millis(250);
const QUICK_MEASUREMENT_TIME: Duration = Duration::from_secs(1);

pub fn is_quick() -> bool {
    cfg!(feature = "quick")
}

/// One-time threshold studies (schoolbook/dense crossovers, thread-count
/// boundaries) that do not need re-running on every suite invocation.
const CALIBRATION_ENV: &str = "EMVP_BENCH_CALIBRATION";

#[must_use]
pub fn calibration_enabled() -> bool {
    std::env::var_os(CALIBRATION_ENV).is_some_and(|value| value == "1")
}

pub fn skip_calibration(target: &str) -> bool {
    if calibration_enabled() {
        false
    } else {
        println!("skipping `{target}` calibration (set {CALIBRATION_ENV}=1 to run)");
        true
    }
}

pub fn values(length: usize, modulus: u32, offset: u64) -> Vec<u32> {
    (0..length)
        .map(|index| ((index as u64 * 2_654_435_761 + offset) % u64::from(modulus)) as u32)
        .collect()
}

pub fn array_values<const MODULUS: u32, const N: usize>(offset: u64) -> [u32; N] {
    std::array::from_fn(|index| {
        ((index as u64 * 2_654_435_761 + offset) % u64::from(MODULUS)) as u32
    })
}

pub fn criterion_default() -> Criterion {
    if is_quick() {
        quick_criterion()
    } else {
        Criterion::default()
    }
}

pub fn criterion_tuned(
    sample_size: usize,
    warm_up_time: Duration,
    measurement_time: Duration,
) -> Criterion {
    if is_quick() {
        quick_criterion()
    } else {
        Criterion::default()
            .sample_size(sample_size)
            .warm_up_time(warm_up_time)
            .measurement_time(measurement_time)
    }
}

fn quick_criterion() -> Criterion {
    Criterion::default()
        .sample_size(QUICK_SAMPLE_SIZE)
        .warm_up_time(QUICK_WARM_UP_TIME)
        .measurement_time(QUICK_MEASUREMENT_TIME)
}

pub fn tune_group(
    group: &mut BenchmarkGroup<'_, WallTime>,
    sample_size: usize,
    warm_up_time: Duration,
    measurement_time: Duration,
) {
    if is_quick() {
        group.sample_size(QUICK_SAMPLE_SIZE);
        group.warm_up_time(QUICK_WARM_UP_TIME);
        group.measurement_time(QUICK_MEASUREMENT_TIME);
    } else {
        group.sample_size(sample_size);
        group.warm_up_time(warm_up_time);
        group.measurement_time(measurement_time);
    }
}
