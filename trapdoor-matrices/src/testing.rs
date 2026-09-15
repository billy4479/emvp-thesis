//! Deterministic fixed-weight builders shared by tests and benchmarks.
//!
//! These helpers exist so that tests and benchmarks construct identical
//! Ring-LPN blocks with a fixed, reproducible sparsity layout, which keeps
//! saved benchmark baselines comparable. They deliberately bypass the
//! sampling API of [`crate::IrreducibleRingLpn::sample`]: the sparse matrix
//! uses a seeded pseudorandom fixed-weight support instead of Bernoulli
//! noise, and the construction skips the security-parameter assessment. Do
//! not use them for anything security-sensitive.

use prime_field_layer::{FieldElement, PrimeField};
use rand_core::CryptoRng;

use crate::TdmError;
use crate::ring_lpn::SparseMatrix;
use crate::{IrreducibleRingLpn, automatic_ring_modulus};

/// Draws one uniform index in `[0, bound)` from `rng`.
///
/// Rejection sampling with a 64-bit draw: the acceptance space is the
/// largest multiple of `bound` below `2^64`, so every index is exactly
/// equally likely. `bound` must be positive.
fn uniform_below(bound: u64, rng: &mut (impl CryptoRng + ?Sized)) -> u64 {
    debug_assert!(bound > 0);
    let threshold = bound.wrapping_neg() % bound;
    loop {
        let draw = rng.next_u64();
        if draw >= threshold {
            return draw % bound;
        }
    }
}

/// Samples `count` distinct indices from `[0, population)` without
/// replacement into `out` using Floyd's algorithm.
///
/// Every size-`count` subset is equally likely, the draw count is exactly
/// `count`, and the `population`-bit bitmap is reused across calls. `out`
/// must have exactly `count` slots; `count` may be zero or equal the whole
/// population.
fn sample_without_replacement<R: CryptoRng + ?Sized>(
    population: usize,
    count: usize,
    rng: &mut R,
    bitmap: &mut [u64],
    out: &mut [usize],
) {
    debug_assert_eq!(out.len(), count);
    debug_assert_eq!(bitmap.len(), population.div_ceil(u64::BITS as usize));
    for slot in bitmap.iter_mut() {
        *slot = 0;
    }
    // Floyd's iteration: for `t` from `population - count` up, map a draw in
    // `[0, t]` through the already-chosen set. Every element of
    // `[0, population)` ends up chosen exactly once.
    for (cursor, t) in (population - count..population).enumerate() {
        let candidate = uniform_below(t as u64 + 1, rng) as usize;
        let chosen = if bitmap[candidate / 64] >> (candidate % 64) & 1 == 1 {
            t
        } else {
            candidate
        };
        bitmap[chosen / 64] |= 1 << (chosen % 64);
        out[cursor] = chosen;
    }
}

/// Builds the deterministic fixed-weight sparse matrix `E`.
///
/// Column `c` stores `weight` uniform-nonzero entries at `weight` distinct
/// rows sampled pseudorandomly without replacement from the `2k` rows, using
/// only the passed RNG as the deterministic seed. Each column's rows are
/// sorted ascending, so CSC traversal sees the same cache-friendly layout a
/// sorted sampled support would have, and repeated runs (and saved benchmark
/// baselines) exercise identical memory access patterns.
///
/// # Errors
///
/// Returns an error for a zero degree, dimension overflow, or malformed
/// sampling bounds. A weight above `k` is not an error: it is silently
/// clamped to `k`, matching the per-column capacity bounds of the sparse
/// layout.
pub fn fixed_weight_sparse<const MODULUS: u32, R: CryptoRng + ?Sized>(
    k: usize,
    weight: usize,
    rng: &mut R,
) -> Result<SparseMatrix<MODULUS>, TdmError> {
    let rows = k.checked_mul(2).ok_or(TdmError::DimensionOverflow)?;
    let offsets_capacity = k.checked_add(1).ok_or(TdmError::DimensionOverflow)?;
    let stored_weight = weight.min(k);
    let field = PrimeField::<MODULUS>::new();

    let mut offsets = Vec::with_capacity(offsets_capacity);
    let mut row_indices = Vec::with_capacity(k * stored_weight);
    let mut values = Vec::with_capacity(k * stored_weight);
    // One reusable selection bitmap plus one per-column scratch, allocated
    // once instead of per column.
    let bitmap_words = rows.div_ceil(u64::BITS as usize);
    let mut bitmap = vec![0_u64; bitmap_words];
    let mut selected = vec![0_usize; stored_weight];
    offsets.push(0);
    for _column in 0..k {
        sample_without_replacement(rows, stored_weight, rng, &mut bitmap, &mut selected);
        selected.sort_unstable();
        row_indices.extend_from_slice(&selected);
        for _ in 0..stored_weight {
            values.push(field.sample_uniform_nonzero(rng));
        }
        offsets.push(row_indices.len());
    }
    SparseMatrix::new(rows, k, offsets, row_indices, values)
}

/// Builds one square `k x k` Ring-LPN mask block with fixed-weight secret.
///
/// The modulus is the automatic irreducible binomial for the field and
/// degree (see [`crate::automatic_ring_modulus`]), the multiplier is
/// uniform, and the secret `E` comes from [`fixed_weight_sparse`]. The
/// construction skips the irreducibility test and the security assessment.
///
/// # Errors
///
/// Returns an error under the same conditions as the two builders it
/// composes, or if the extension-ring construction fails.
pub fn ring_block<const MODULUS: u32, R: CryptoRng + ?Sized>(
    k: usize,
    weight: usize,
    rng: &mut R,
) -> Result<IrreducibleRingLpn<MODULUS>, TdmError> {
    let modulus = automatic_ring_modulus::<MODULUS>(k)?;
    let field = PrimeField::<MODULUS>::new();
    let multiplier: Box<[u32]> = (0..k).map(|_| field.sample_uniform(rng).value()).collect();
    let sparse = fixed_weight_sparse::<MODULUS, R>(k, weight, rng)?;
    IrreducibleRingLpn::new_unchecked_irreducible(k, &modulus, &multiplier, sparse)
}

/// Uniform field-element fixture helper shared by downstream test suites.
#[doc(hidden)]
pub fn field_values<const MODULUS: u32, R: CryptoRng + ?Sized>(
    length: usize,
    rng: &mut R,
) -> Vec<FieldElement<MODULUS>> {
    let field = PrimeField::<MODULUS>::new();
    (0..length).map(|_| field.sample_uniform(rng)).collect()
}

#[cfg(test)]
mod tests {
    use rand_chacha::ChaCha20Rng;
    use rand_core::SeedableRng;

    use super::{fixed_weight_sparse, sample_without_replacement, uniform_below};

    #[test]
    fn uniform_below_is_in_range() {
        let mut rng = ChaCha20Rng::from_seed([7; 32]);
        for bound in [1_u64, 2, 3, 63, 64, 65, 1000, u64::from(u32::MAX)] {
            for _ in 0..1000 {
                assert!(uniform_below(bound, &mut rng) < bound);
            }
        }
    }

    #[test]
    fn selection_is_without_replacement_and_deterministic() {
        let population: usize = 10_000;
        let count: usize = 192;
        let mut bitmap = vec![0_u64; population.div_ceil(64)];
        let mut out = vec![0_usize; count];

        let mut rng = ChaCha20Rng::from_seed([9; 32]);
        sample_without_replacement(population, count, &mut rng, &mut bitmap, &mut out);
        let mut sorted = out.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), count, "rows must be distinct");
        assert!(out.iter().all(|&row| row < population));

        let mut rng = ChaCha20Rng::from_seed([9; 32]);
        let mut replay = vec![0_usize; count];
        sample_without_replacement(population, count, &mut rng, &mut bitmap, &mut replay);
        assert_eq!(out, replay, "selection must be seed-deterministic");
    }

    #[test]
    fn selection_edge_cases() {
        let mut bitmap = vec![0_u64; 0];
        let mut out = vec![0_usize; 0];
        let mut rng = ChaCha20Rng::from_seed([3; 32]);
        sample_without_replacement(0, 0, &mut rng, &mut bitmap, &mut out);

        let mut bitmap = vec![0_u64; 1];
        let mut out = vec![0_usize; 8];
        sample_without_replacement(8, 8, &mut rng, &mut bitmap, &mut out);
        out.sort_unstable();
        assert_eq!(out, (0..8).collect::<Vec<_>>());
    }

    #[test]
    fn sparse_fixture_has_exact_weight_and_distinct_sorted_rows() {
        let mut rng = ChaCha20Rng::from_seed([11; 32]);
        let k = 256;
        let weight = 192;
        let sparse = fixed_weight_sparse::<1_073_479_681, _>(k, weight, &mut rng).unwrap();
        assert_eq!(sparse.rows(), 2 * k);
        assert_eq!(sparse.columns(), k);
        for column in 0..k {
            let start = sparse.offsets()[column];
            let end = sparse.offsets()[column + 1];
            assert_eq!(end - start, weight, "column weight must be exact");
            let rows = &sparse.row_indices()[start..end];
            let mut sorted = rows.to_vec();
            sorted.sort_unstable();
            assert_eq!(rows, sorted.as_slice(), "rows must be sorted per column");
            sorted.dedup();
            assert_eq!(sorted.len(), weight, "rows must be distinct");
        }
        // Weight above k clamps to k without error.
        let clamped = fixed_weight_sparse::<1_073_479_681, _>(4, 100, &mut rng).unwrap();
        for column in 0..4 {
            let start = clamped.offsets()[column];
            let end = clamped.offsets()[column + 1];
            assert_eq!(end - start, 4);
        }
    }

    #[test]
    fn sparse_fixture_is_seed_deterministic() {
        let mut rng = ChaCha20Rng::from_seed([13; 32]);
        let first = fixed_weight_sparse::<1_073_479_681, _>(64, 16, &mut rng).unwrap();
        let mut rng = ChaCha20Rng::from_seed([13; 32]);
        let second = fixed_weight_sparse::<1_073_479_681, _>(64, 16, &mut rng).unwrap();
        assert_eq!(first, second);
    }
}
