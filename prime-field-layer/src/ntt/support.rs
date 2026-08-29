use super::{
    FieldElement, NttBackend, NttPerformanceWarning, PrimeField, Twiddle, reduce_once_u64,
};

#[derive(Clone, Copy)]
pub(super) enum BackendPreference {
    Auto,
    Scalar,
}
pub(super) fn select_backend<const MODULUS: u32>(
    preference: BackendPreference,
) -> (NttBackend, Option<NttPerformanceWarning>) {
    if MODULUS >= 1 << 31 {
        return (
            NttBackend::ScalarMontgomery,
            Some(NttPerformanceWarning::MontgomeryFallback),
        );
    }
    let backend = if MODULUS < 1 << 30 {
        NttBackend::ScalarShoupLazy
    } else {
        NttBackend::ScalarShoup
    };
    let warning = match preference {
        BackendPreference::Auto => None,
        BackendPreference::Scalar => Some(NttPerformanceWarning::ScalarRequested),
    };
    (backend, warning)
}
pub(super) fn make_twiddle<const MODULUS: u32>(
    field: PrimeField<MODULUS>,
    canonical: u32,
) -> Twiddle {
    Twiddle {
        canonical,
        shoup: ((u64::from(canonical) << 32) / u64::from(MODULUS)) as u32,
        montgomery: field.element(u64::from(canonical)).montgomery(),
    }
}

pub(super) fn twiddle_powers<const MODULUS: u32>(
    field: PrimeField<MODULUS>,
    root: u32,
    count: usize,
) -> Vec<Twiddle> {
    let mut canonical = 1;
    (0..count)
        .map(|_| {
            let twiddle = make_twiddle(field, canonical);
            canonical = field.mul(canonical, root);
            twiddle
        })
        .collect()
}

pub(super) fn powers<const MODULUS: u32>(
    field: PrimeField<MODULUS>,
    base: u32,
    length: usize,
) -> Vec<FieldElement<MODULUS>> {
    let base = field.element(u64::from(base));
    let mut power = field.element(1);
    (0..length)
        .map(|_| {
            let result = power;
            power *= base;
            result
        })
        .collect()
}

#[inline(always)]
pub(super) fn shoup_mul<const MODULUS: u32>(value: u32, twiddle: Twiddle) -> u32 {
    reduce_once(shoup_mul_lazy_for::<MODULUS>(value, twiddle), MODULUS)
}

/// Lazy Shoup product of any `u32` word with a twiddle constant.
///
/// This is the wide-input form of Shoup's multiplication (Harvey, J. Symbolic
/// Comput. 60 (2014), section 3; also Bradbury et al., ePrint 2021/1396,
/// Theorem 2, for the 32-bit SIMD lanes used here). For a twiddle `w` in
/// `[0, p)` with precomputed `w' = floor(w * 2^32 / p)`, the quotient
/// `q = (z * w') >> 32` and product `t = z * w - q * p` satisfy
/// `0 <= t < 2p` for every input `z` in `[0, 2^32)`, not only reduced inputs:
/// from `w' <= w * 2^32 / p < w' + 1` follows `q <= z * w / p`, hence `t >= 0`,
/// and `q > z * w / p - z / 2^32 - 1`, hence
/// `t < (z / 2^32 + 1) * p < 2p`. The lazy callers have `p < 2^30` and
/// arbitrary `u32` inputs; the reduced callers have `p < 2^31` and inputs
/// below `p`. Thus `z * w` and `q * p` are below `2^62`. The quotient product
/// `z * w'` is below `2^64` for lazy callers and `2^63` for reduced callers,
/// so every intermediate fits `u64`; `t < 2p < 2^32` fits a `u32` lane.
///
/// This bound is what lets the lazy butterflies feed unreduced lazy words
/// straight into the multiplication: a difference planted with a `+2p` offset
/// stays below `4p <= 2^32` for `p < 2^30`, and the quotient estimate in the
/// high half of the product absorbs the planted offset without any
/// conditional correction of the product.
#[inline(always)]
pub(super) fn shoup_mul_lazy_for<const MODULUS: u32>(value: u32, twiddle: Twiddle) -> u32 {
    let quotient = (u64::from(value) * u64::from(twiddle.shoup)) >> 32;
    (u64::from(value) * u64::from(twiddle.canonical) - quotient * u64::from(MODULUS)) as u32
}

/// Halves the lazy interval `[0, 4p)` to `[0, 2p)` with one branchless minimum.
///
/// For `value` in `[0, 4p)` the wrapping difference `value - two_p` lies below
/// `value` exactly when no borrow occurs, i.e. when `value >= two_p`; when
/// `value < two_p` it wraps far above `value`. The `u32::min` selection
/// therefore returns `value - two_p` on `[2p, 4p)` and `value` unchanged on
/// `[0, 2p)`, with no coefficient-dependent branch. Vector code lowers it to a
/// single `vpminud` beside the subtraction.
#[inline(always)]
pub(super) fn halve_interval(value: u32, two_p: u32) -> u32 {
    value.min(value.wrapping_sub(two_p))
}

/// Subtracts `modulus` from a value in `[0, 2 * modulus)` with one mask.
///
/// The wrapping difference `value - modulus` underflows exactly when
/// `value < modulus`; `0u32.wrapping_sub(borrow)` then materializes either
/// all ones or zero, so the expression returns `value - modulus` or `value`
/// unchanged with no coefficient-dependent branch. This is deliberately
/// portable Rust rather than the inline assembly of `constant_time`:
/// assembly is opaque to LLVM and blocks loop vectorization, while this form
/// lowers to the same `sub`/`cmov` sequence in scalar code and to vector
/// compares and selects inside auto-vectorized loops.
#[inline(always)]
pub(super) const fn reduce_once(value: u32, modulus: u32) -> u32 {
    let (reduced, borrow) = value.overflowing_sub(modulus);
    reduced.wrapping_add(modulus & 0u32.wrapping_sub(borrow as u32))
}

#[inline(always)]
pub(super) fn add_mod<const MODULUS: u32>(lhs: u32, rhs: u32) -> u32 {
    reduce_once_u64(u64::from(lhs) + u64::from(rhs), u64::from(MODULUS)) as u32
}

#[inline(always)]
pub(super) fn sub_mod<const MODULUS: u32>(lhs: u32, rhs: u32) -> u32 {
    reduce_once_u64(
        u64::from(lhs) + u64::from(MODULUS) - u64::from(rhs),
        u64::from(MODULUS),
    ) as u32
}

/// Restores canonical `[0, p)` Montgomery words from lazy stage residues.
///
/// Forward lazy stages emit values in `[0, 4p)` and inverse lazy stages values
/// in `[0, 2p)`; halving by `2p` (a no-op for the inverse interval) followed
/// by one `p` correction covers both.
pub(super) fn normalize<const MODULUS: u32>(values: &mut [FieldElement<MODULUS>]) {
    let two_p = MODULUS * 2;
    for value in values {
        let halved = halve_interval(value.montgomery(), two_p);
        value.set_montgomery(reduce_once(halved, MODULUS));
    }
}
