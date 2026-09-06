//! Deterministic fixed-weight builders shared by tests and benchmarks.
//!
//! These helpers exist so that tests and benchmarks construct identical
//! Ring-LPN blocks with a fixed, reproducible sparsity layout, which keeps
//! saved benchmark baselines comparable. They deliberately bypass the
//! sampling API of [`crate::IrreducibleRingLpn::sample`]: the sparse matrix
//! uses a deterministic row layout instead of Bernoulli noise, and the
//! construction skips the security-parameter assessment. Do not use them
//! for anything security-sensitive.

use prime_field_layer::{FieldElement, PrimeField};
use rand_core::CryptoRng;

use crate::TdmError;
use crate::ring_lpn::SparseMatrix;
use crate::{IrreducibleRingLpn, automatic_ring_modulus};

/// Builds the deterministic fixed-weight sparse matrix `E`.
///
/// Column `c` stores its `weight` entries at rows `(17 c + j) mod 2k`, and
/// every stored entry is uniform nonzero. The row layout is fixed so that
/// repeated runs (and saved benchmark baselines) exercise identical memory
/// access patterns.
///
/// # Errors
///
/// Returns an error for a zero degree, an oversized weight, dimension
/// overflow, or malformed sampling bounds.
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
    offsets.push(0);
    for column in 0..k {
        for entry in 0..stored_weight {
            row_indices.push((17 * column + entry) % rows);
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

/// Re-exports the field element helper for downstream test fixtures.
#[doc(hidden)]
pub fn field_values<const MODULUS: u32, R: CryptoRng + ?Sized>(
    length: usize,
    rng: &mut R,
) -> Vec<FieldElement<MODULUS>> {
    let field = PrimeField::<MODULUS>::new();
    (0..length).map(|_| field.sample_uniform(rng)).collect()
}
