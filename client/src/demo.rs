//! Deterministic demo inputs: seed mixing, parameter selection, and the
//! compiled construction parameters.
//!
//! Every matrix, key, and query of one invocation derives from the single
//! `--seed` argument through domain-separated mixing, so runs with equal
//! arguments are bit-for-bit reproducible.

use emvp::{EmvpParams, search};
use prime_field_layer::{FieldElement, PrimeField};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;

use crate::error::RunError;

/// The wire and arithmetic field modulus; matches the handshake.
pub const PROTOCOL_MODULUS: u32 = emvp_network::PROTOCOL_MODULUS;

/// The security level every parameter search targets.
pub const SECURITY_LAMBDA: u32 = 128;

/// The compiled Ring-LPN secret weight.
pub const RING_LPN_WEIGHT: usize = 192;

/// The compiled RAA repetition factor.
pub const RAA_FACTOR: usize = 3;

/// Domain separators for the per-matrix PRF-style seed derivation.
const DOMAIN_KEY: u64 = 0x6B_45_4D_56_50_5F_4B_45; // "kEMVP_KE"
const DOMAIN_DERIVE: u64 = 0x64_45_4D_56_50_5F_44_52; // "dEMVP_DR"
const DOMAIN_PLAINTEXT: u64 = 0x70_45_4D_56_50_5F_50_4C; // "pEMVP_PL"
const DOMAIN_QUERIES: u64 = 0x71_45_4D_56_50_5F_51_52; // "qEMVP_QR"

/// The protocol field type.
pub type Field = FieldElement<PROTOCOL_MODULUS>;

/// Splitmix64 finalizer: turns counter-like inputs into well-mixed seeds.
const fn mix(input: u64) -> u64 {
    let mixed = (input ^ (input >> 30)).wrapping_mul(0xBF_58_47_6D_1C_E4_E5_B9);
    let mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94_D0_49_BB_13_31_11_EB);
    mixed ^ (mixed >> 31)
}

/// Derives the deterministic [`ChaCha20Rng`] of one domain and matrix index.
#[must_use]
pub fn seed_rng(base_seed: u64, matrix_index: usize, domain: u64) -> ChaCha20Rng {
    let mixed = mix(mix(base_seed ^ domain) ^ matrix_index as u64);
    ChaCha20Rng::seed_from_u64(mixed)
}

/// Uniform plaintext values of one domain and matrix index.
#[must_use]
pub fn plaintext_values(count: usize, base_seed: u64, matrix_index: usize) -> Vec<Field> {
    let field = PrimeField::<PROTOCOL_MODULUS>::new();
    let mut rng = seed_rng(base_seed, matrix_index, DOMAIN_PLAINTEXT);
    (0..count).map(|_| field.sample_uniform(&mut rng)).collect()
}

/// The deterministic 32-byte client key of one matrix.
#[must_use]
pub fn matrix_key(base_seed: u64, matrix_index: usize) -> [u8; 32] {
    let mut rng = seed_rng(base_seed, matrix_index, DOMAIN_KEY);
    let mut key = [0_u8; 32];
    rand_core::Rng::fill_bytes(&mut rng, &mut key);
    key
}

/// The deterministic derive-phase RNG of one matrix.
#[must_use]
pub fn derive_rng(base_seed: u64, matrix_index: usize) -> ChaCha20Rng {
    seed_rng(base_seed, matrix_index, DOMAIN_DERIVE)
}

/// The deterministic query-generation RNG of one matrix.
#[must_use]
pub fn query_rng(base_seed: u64, matrix_index: usize) -> ChaCha20Rng {
    seed_rng(base_seed, matrix_index, DOMAIN_QUERIES)
}

/// Searches the protocol parameters for one vector width.
///
/// # Errors
///
/// Returns the search failure for a zero width or an exhausted scan.
pub fn select_params(width: usize) -> Result<EmvpParams, RunError> {
    if width == 0 {
        return Err(RunError::Width);
    }
    Ok(search(width, SECURITY_LAMBDA)?)
}

/// Rejects parameter sets the compiled Ring-LPN weight cannot support.
///
/// # Errors
///
/// Returns [`RunError::WidthTooSmall`] when the searched code width `n`
/// is below [`RING_LPN_WEIGHT`].
pub fn check_ring_lpn_width(params: &EmvpParams) -> Result<(), RunError> {
    let width = params.n().map_err(RunError::Params)?;
    if width < RING_LPN_WEIGHT {
        return Err(RunError::WidthTooSmall {
            width,
            weight: RING_LPN_WEIGHT,
        });
    }
    Ok(())
}

/// Validates the demo's row-count list.
///
/// # Errors
///
/// Returns [`RunError::Rows`] for an empty list or a zero row count.
pub fn check_rows(rows: &[usize]) -> Result<(), RunError> {
    if rows.is_empty() || rows.contains(&0) {
        return Err(RunError::Rows);
    }
    Ok(())
}

/// Validates the per-matrix query count.
///
/// # Errors
///
/// Returns [`RunError::Queries`] for a zero count.
pub const fn check_queries(queries: usize) -> Result<(), RunError> {
    if queries == 0 {
        return Err(RunError::Queries);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{
        check_queries, check_ring_lpn_width, check_rows, mix, plaintext_values, seed_rng,
        select_params,
    };
    use crate::error::RunError;

    #[test]
    fn seeding_is_deterministic_and_domain_separated() {
        use rand_core::Rng;
        let mut first = seed_rng(7, 0, 1);
        let mut second = seed_rng(7, 0, 1);
        assert_eq!(first.clone().next_u64(), second.clone().next_u64());
        let mut other_domain = seed_rng(7, 0, 2);
        assert_ne!(first.next_u64(), other_domain.next_u64());
        let mut other_index = seed_rng(7, 1, 1);
        assert_ne!(second.next_u64(), other_index.next_u64());
    }

    #[test]
    fn plaintext_values_are_reproducible() {
        let first = plaintext_values(64, 9, 0);
        let second = plaintext_values(64, 9, 0);
        assert_eq!(first, second);
        assert_eq!(first.len(), 64);
    }

    #[test]
    fn mixing_is_constant_for_constant_input() {
        assert_eq!(mix(0), mix(0));
        assert_eq!(mix(0x1234_5678_9ABC_DEF0), mix(0x1234_5678_9ABC_DEF0));
    }

    #[test]
    fn width_selection_searches_validated_parameters() {
        let params = select_params(512).unwrap();
        assert_eq!(params.ell, 512);
        assert_eq!(params.lambda, super::SECURITY_LAMBDA);
        params.validate().unwrap();
    }

    #[test]
    fn zero_width_is_rejected() {
        assert!(matches!(select_params(0), Err(RunError::Width)));
    }

    #[test]
    fn ring_lpn_needs_the_compiled_weight() {
        let params = select_params(512).unwrap();
        check_ring_lpn_width(&params).unwrap();
        // The rejection is about the code width alone: any parameter set
        // with `n = 2k` below the compiled weight is refused.
        let narrow = emvp::EmvpParams {
            k: 64,
            ell: 8,
            b: 2,
            lambda: super::SECURITY_LAMBDA,
        };
        assert!(matches!(
            check_ring_lpn_width(&narrow),
            Err(RunError::WidthTooSmall {
                width: 128,
                weight: 192
            })
        ));
    }

    #[test]
    fn row_and_query_counts_are_validated() {
        check_rows(&[1, 2]).unwrap();
        assert!(matches!(check_rows(&[]), Err(RunError::Rows)));
        assert!(matches!(check_rows(&[4, 0]), Err(RunError::Rows)));
        check_queries(1).unwrap();
        assert!(matches!(check_queries(0), Err(RunError::Queries)));
    }
}
