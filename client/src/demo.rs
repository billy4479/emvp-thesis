//! Demo inputs: master-seed derivation, parameter selection, and the
//! compiled construction parameters.
//!
//! One invocation derives everything from a single 256-bit master seed:
//! either explicitly supplied with `--seed` (reproducible, insecure) or
//! freshly drawn from the operating system. Every matrix, key, and query
//! derives from the master seed through domain-separated PRF expansion
//! ([`Prf`] with one context tag per component), so runs with equal
//! arguments and the same explicit seed are bit-for-bit reproducible.
//! The 256-bit master seed is never printed.

use emvp::{EmvpParams, Prf, PrfError, search};
use prime_field_layer::{FieldElement, PrimeField};
use rand_chacha::ChaCha20Rng;
use trapdoor_matrices::{SecurityAssessment, SecurityLevel};

use crate::error::RunError;

/// The wire and arithmetic field modulus; matches the handshake.
pub const PROTOCOL_MODULUS: u32 = emvp_network::PROTOCOL_MODULUS;

/// The security level every parameter search targets.
pub const SECURITY_LAMBDA: u32 = 128;

/// The compiled Ring-LPN secret weight.
pub const RING_LPN_WEIGHT: usize = 192;

/// The compiled RAA repetition factor.
pub const RAA_FACTOR: usize = 3;

/// The one purpose tag every demo component's PRF stream addresses. The
/// components are separated by their context tags, so one purpose per
/// domain suffices.
const DEMO_PURPOSE: u32 = 0;

/// Context tags of the per-matrix PRF-style derivation. Each occupies its
/// own `ChaCha20` stream of the master-seed PRF, so the components cannot
/// collide with each other or with the protocol's own derivation (which is
/// keyed by the per-matrix keys below, never by the master seed).
const DOMAIN_KEY: u128 = 0x6B_45_4D_56_50_5F_4B_45; // "kEMVP_KE"
const DOMAIN_DERIVE: u128 = 0x64_45_4D_56_50_5F_44_52; // "dEMVP_DR"
const DOMAIN_PLAINTEXT: u128 = 0x70_45_4D_56_50_5F_50_4C; // "pEMVP_PL"
const DOMAIN_QUERIES: u128 = 0x71_45_4D_56_50_5F_51_52; // "qEMVP_QR"

/// Stable mask-suite context identifier bases, one per construction.
///
/// The compiled configuration (RAA factor, Ring-LPN weight) is mixed into
/// the identifier, so a changed configuration changes the context
/// identifier and with it every derived instance.
const CONTEXT_TOEPLITZ_BASE: u128 = 0x54_4F_45_50_00_00_00_00; // "TOEP"
const CONTEXT_RAA_BASE: u128 = 0x52_41_41_41_00_00_00_00; // "RAAA"
const CONTEXT_RING_BASE: u128 = 0x52_49_4E_47_00_00_00_00; // "RING"

/// The stable context identifier of the Toeplitz suite.
pub const CONTEXT_TOEPLITZ: emvp::MaskContextId = emvp::MaskContextId::new(CONTEXT_TOEPLITZ_BASE);

/// The stable context identifier of the weighted-RAA suite; it changes
/// whenever [`RAA_FACTOR`] changes.
pub const CONTEXT_RAA: emvp::MaskContextId =
    emvp::MaskContextId::new(CONTEXT_RAA_BASE | RAA_FACTOR as u128);

/// The stable context identifier of the Ring-LPN suite; it changes
/// whenever [`RING_LPN_WEIGHT`] changes.
pub const CONTEXT_RING: emvp::MaskContextId =
    emvp::MaskContextId::new(CONTEXT_RING_BASE | RING_LPN_WEIGHT as u128);

/// The 32-byte master seed every key, matrix, and query derives from.
pub type MasterSeed = [u8; 32];

/// The protocol field type.
pub type Field = FieldElement<PROTOCOL_MODULUS>;

/// Draws a fresh 256-bit master seed from the operating system.
///
/// # Errors
///
/// Returns the OS entropy-source failure.
pub fn random_master_seed() -> Result<MasterSeed, getrandom::Error> {
    let mut seed = [0_u8; 32];
    getrandom::fill(&mut seed)?;
    Ok(seed)
}

/// Expands an explicit `--seed` argument into the 32-byte master seed.
///
/// The 64-bit argument carries at most 64 bits of entropy, so runs using
/// it are reproducible by construction and secure only against observers
/// who do not know it. Deterministic for a given argument.
#[must_use]
pub fn master_seed_from_u64(seed: u64) -> MasterSeed {
    let mut master = [0_u8; 32];
    for (chunk, round) in master.chunks_exact_mut(8).zip(0_u64..4) {
        chunk.copy_from_slice(
            &mix(seed ^ round.wrapping_mul(0x9E_37_79_B9_7F_4A_7C_15)).to_le_bytes(),
        );
    }
    master
}

/// Splitmix64 finalizer: turns counter-like inputs into well-mixed words.
const fn mix(input: u64) -> u64 {
    let mixed = (input ^ (input >> 30)).wrapping_mul(0xBF_58_47_6D_1C_E4_E5_B9);
    let mixed = (mixed ^ (mixed >> 27)).wrapping_mul(0x94_D0_49_BB_13_31_11_EB);
    mixed ^ (mixed >> 31)
}

/// The PRF-derived generator of one domain and matrix index.
///
/// Domain separation: the master seed keys a [`Prf`], the component's
/// domain tag selects a `ChaCha20` stream, and the matrix index selects one
/// 32-byte slot inside it. Derivation order does not matter.
fn domain_rng(
    master: &MasterSeed,
    domain: u128,
    matrix_index: usize,
) -> Result<ChaCha20Rng, PrfError> {
    Prf::new(*master).derive_context(domain).stream(
        DEMO_PURPOSE,
        u64::try_from(matrix_index).unwrap_or(u64::MAX),
    )
}

/// Uniform plaintext values of one matrix index.
///
/// # Errors
///
/// Returns the PRF index failure for an out-of-range matrix index.
pub fn plaintext_values(
    count: usize,
    master: &MasterSeed,
    matrix_index: usize,
) -> Result<Vec<Field>, PrfError> {
    let field = PrimeField::<PROTOCOL_MODULUS>::new();
    let mut rng = domain_rng(master, DOMAIN_PLAINTEXT, matrix_index)?;
    Ok((0..count).map(|_| field.sample_uniform(&mut rng)).collect())
}

/// The deterministic 32-byte client key of one matrix.
///
/// # Errors
///
/// Returns the PRF index failure for an out-of-range matrix index.
pub fn matrix_key(master: &MasterSeed, matrix_index: usize) -> Result<[u8; 32], PrfError> {
    let mut rng = domain_rng(master, DOMAIN_KEY, matrix_index)?;
    let mut key = [0_u8; 32];
    rand_core::Rng::fill_bytes(&mut rng, &mut key);
    Ok(key)
}

/// The deterministic derive-phase generator of one matrix.
///
/// # Errors
///
/// Returns the PRF index failure for an out-of-range matrix index.
pub fn derive_rng(master: &MasterSeed, matrix_index: usize) -> Result<ChaCha20Rng, PrfError> {
    domain_rng(master, DOMAIN_DERIVE, matrix_index)
}

/// The deterministic query-generation generator of one matrix.
///
/// # Errors
///
/// Returns the PRF index failure for an out-of-range matrix index.
pub fn query_rng(master: &MasterSeed, matrix_index: usize) -> Result<ChaCha20Rng, PrfError> {
    domain_rng(master, DOMAIN_QUERIES, matrix_index)
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

/// Rejects Ring-LPN configurations the compiled weight cannot support or
/// the parameter policy assesses as broken.
///
/// # Errors
///
/// Returns [`RunError::WidthTooSmall`] when the searched code width `n` is
/// below [`RING_LPN_WEIGHT`], [`RunError::RingWidthNotPowerOfTwo`] when the
/// automatic irreducible modulus cannot be selected, and
/// [`RunError::InsecureRing`] when the assessment of `(n, weight)` fails
/// closed.
pub fn check_ring_lpn_configuration(params: &EmvpParams) -> Result<(), RunError> {
    let width = params.n().map_err(RunError::Params)?;
    if width < RING_LPN_WEIGHT {
        return Err(RunError::WidthTooSmall {
            width,
            weight: RING_LPN_WEIGHT,
        });
    }
    if !width.is_power_of_two() {
        return Err(RunError::RingWidthNotPowerOfTwo { width });
    }
    let assessment = assess_ring_lpn(width);
    if assessment.level == SecurityLevel::Broken {
        return Err(RunError::InsecureRing {
            degree: width,
            weight: RING_LPN_WEIGHT,
            reasons: assessment.broken_reasons(),
        });
    }
    Ok(())
}

/// The security assessment of the compiled Ring-LPN configuration at one
/// ring degree.
#[must_use]
pub fn assess_ring_lpn(degree: usize) -> SecurityAssessment {
    trapdoor_matrices::assess::<PROTOCOL_MODULUS>(degree, RING_LPN_WEIGHT)
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

/// The plaintext length of one matrix, rejecting overflow.
///
/// # Errors
///
/// Returns [`RunError::RowOverflow`] when `rows * ell` overflows.
pub const fn plaintext_len(rows: usize, ell: usize) -> Result<usize, RunError> {
    match rows.checked_mul(ell) {
        Some(len) => Ok(len),
        None => Err(RunError::RowOverflow { rows, ell }),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CONTEXT_RAA, CONTEXT_RING, CONTEXT_TOEPLITZ, check_queries, check_ring_lpn_configuration,
        check_rows, master_seed_from_u64, plaintext_len, random_master_seed, select_params,
    };
    use crate::error::RunError;
    use trapdoor_matrices::SecurityLevel;

    #[test]
    fn master_seed_from_u64_is_deterministic() {
        let first = master_seed_from_u64(7);
        let second = master_seed_from_u64(7);
        assert_eq!(first, second);
        let other = master_seed_from_u64(8);
        assert_ne!(first, other);
    }

    #[test]
    fn master_seed_from_u64_mixes_across_the_whole_seed() {
        let seed = master_seed_from_u64(0);
        // Every 8-byte lane is derived from a distinct mixer round, so the
        // 32 bytes do not collapse into a repeated pattern.
        let lanes: Vec<[u8; 8]> = seed
            .chunks_exact(8)
            .map(|lane| lane.try_into().unwrap())
            .collect();
        assert_eq!(lanes.len(), 4);
        assert_ne!(lanes[0], lanes[1]);
        assert_ne!(lanes[1], lanes[2]);
        assert_ne!(lanes[2], lanes[3]);
    }

    #[test]
    fn random_master_seed_is_structurally_sound() {
        let first = random_master_seed().unwrap();
        let second = random_master_seed().unwrap();
        // Structural only: two fresh draws are independent by construction,
        // and no probabilistic equality is asserted.
        assert_eq!(first.len(), 32);
        assert_eq!(second.len(), 32);
        // The derivation pipeline accepts a fresh seed.
        let _ = super::matrix_key(&first, 0).unwrap();
        let _ = super::matrix_key(&second, 0).unwrap();
    }

    #[test]
    fn derivation_is_deterministic_and_domain_separated() {
        use rand_core::Rng;
        let master = master_seed_from_u64(9);
        let mut first = super::derive_rng(&master, 0).unwrap();
        let mut second = super::derive_rng(&master, 0).unwrap();
        assert_eq!(first.clone().next_u64(), second.clone().next_u64());
        let mut other_index = super::derive_rng(&master, 1).unwrap();
        assert_ne!(first.next_u64(), other_index.next_u64());
        let mut other_domain = super::query_rng(&master, 0).unwrap();
        assert_ne!(second.next_u64(), other_domain.next_u64());
    }

    #[test]
    fn matrix_keys_are_deterministic_per_matrix() {
        let master = master_seed_from_u64(9);
        let first = super::matrix_key(&master, 0).unwrap();
        let second = super::matrix_key(&master, 0).unwrap();
        assert_eq!(first, second);
        let other_index = super::matrix_key(&master, 1).unwrap();
        assert_ne!(first, other_index);
        let plaintext = super::plaintext_values(64, &master, 0).unwrap();
        let again = super::plaintext_values(64, &master, 0).unwrap();
        assert_eq!(plaintext, again);
        assert_eq!(plaintext.len(), 64);
    }

    #[test]
    fn mask_contexts_are_distinct_and_track_their_configuration() {
        assert_ne!(CONTEXT_TOEPLITZ, CONTEXT_RAA);
        assert_ne!(CONTEXT_RAA, CONTEXT_RING);
        assert_ne!(CONTEXT_TOEPLITZ, CONTEXT_RING);
        // The compiled configuration is part of each identifier.
        assert_eq!(
            CONTEXT_RAA.get() & (super::RAA_FACTOR as u128),
            super::RAA_FACTOR as u128
        );
        assert_eq!(
            CONTEXT_RING.get() & (super::RING_LPN_WEIGHT as u128),
            super::RING_LPN_WEIGHT as u128
        );
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
    fn ring_lpn_rejects_narrow_or_broken_configurations() {
        // The default searched width is assessed broken: its ring degree
        // is far below the construction's floor.
        let narrow = emvp::EmvpParams {
            k: 64,
            ell: 8,
            b: 2,
            lambda: super::SECURITY_LAMBDA,
        };
        assert!(matches!(
            check_ring_lpn_configuration(&narrow),
            Err(RunError::WidthTooSmall {
                width: 128,
                weight: 192
            })
        ));

        let default_params = select_params(512).unwrap();
        assert!(matches!(
            check_ring_lpn_configuration(&default_params),
            Err(RunError::InsecureRing {
                degree: 1024,
                weight: 192,
                ..
            })
        ));

        // A non-power-of-two degree cannot get an automatic modulus.
        let odd = emvp::EmvpParams {
            k: 3000,
            ell: 8,
            b: 2,
            lambda: super::SECURITY_LAMBDA,
        };
        assert!(matches!(
            check_ring_lpn_configuration(&odd),
            Err(RunError::RingWidthNotPowerOfTwo { width: 6000 })
        ));
    }

    #[test]
    fn ring_assessment_tracks_the_policy_levels() {
        assert_eq!(super::assess_ring_lpn(1024).level, SecurityLevel::Broken);
        assert_eq!(super::assess_ring_lpn(2048).level, SecurityLevel::Sound);
    }

    #[test]
    fn row_and_query_counts_are_validated() {
        check_rows(&[1, 2]).unwrap();
        assert!(matches!(check_rows(&[]), Err(RunError::Rows)));
        assert!(matches!(check_rows(&[4, 0]), Err(RunError::Rows)));
        check_queries(1).unwrap();
        assert!(matches!(check_queries(0), Err(RunError::Queries)));
        assert!(matches!(
            plaintext_len(usize::MAX, 2),
            Err(RunError::RowOverflow { .. })
        ));
        assert_eq!(plaintext_len(3, 4).unwrap(), 12);
    }
}
