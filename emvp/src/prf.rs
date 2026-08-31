use std::fmt;

use rand_chacha::ChaCha20Rng;
use rand_core::{Rng, SeedableRng};

/// Named purpose tags for [`Prf::stream`].
///
/// Every protocol component must derive its randomness under its own tag:
/// the tag selects an exclusive `2^32`-word window of the `ChaCha20` stream, so
/// index spaces are per-tag and independent. Reusing a tag for two components
/// would hand both the same derived streams; adding a component means adding
/// a fresh tag, never renumbering an existing one.
pub mod purpose {
    /// Seeds the secret code multiplier of the cyclic dual code.
    pub const CODE_MULTIPLIER: u32 = 1;
    /// Seeds the public random permutation that decorrelates the 1D-SLSN
    /// block structure.
    pub const CODE_PERMUTATION: u32 = 2;
    /// Seeds the stack of trapdoored matrices.
    pub const TDM: u32 = 3;
    /// Seeds the nonzero entries of client queries.
    pub const QUERY_NONZERO: u32 = 4;
    /// Seeds the random codeword of each client query.
    pub const QUERY_CODEWORD: u32 = 5;
    /// Derives the public identifier that binds artifacts to a key and matrix.
    pub const INSTANCE_ID: u32 = 6;
}

/// Each indexed slot contains the 8-word (32-byte) seed of a downstream
/// generator.
const SLOT_WORDS: u128 = 8;

/// Largest stream index whose 8-word slot stays inside the purpose window:
/// `index * 8 + 8 <= 2^32`.
const MAX_INDEX: u64 = (1 << (u32::BITS - 3)) - 1;

/// A purpose-indexed deterministic PRF keyed by a 32-byte secret.
///
/// All protocol randomness is derived from one short key: [`Prf::stream`]
/// reads one 32-byte seed at a counter offset computed from a purpose tag and
/// index, then returns a new [`ChaCha20Rng`] initialized from that seed. This
/// is a PRF under the standard assumption that `ChaCha20` in counter mode
/// keyed with a uniform secret key is a secure stream cipher. The key must
/// therefore be a uniform 32-byte secret.
///
/// The stream is partitioned so that derived streams cannot collide:
///
/// - every purpose tag occupies its own window of `2^32` `ChaCha` words;
/// - each index addresses an 8-word slot inside that window (32 bytes, one
///   downstream seed);
/// - indices are bounded by `MAX_INDEX`, so a slot never spills into the
///   next purpose's window.
pub struct Prf([u8; 32]);

impl Prf {
    /// Wraps a uniform 32-byte secret key.
    #[must_use]
    pub const fn new(key: [u8; 32]) -> Self {
        Self(key)
    }

    /// Derives a PRF for one public 128-bit protocol instance identifier.
    ///
    /// The high 64 bits select a `ChaCha20` stream and the low 64 bits select
    /// a non-overlapping 32-byte seed slot in that stream. Distinct instance
    /// identifiers therefore domain-separate all downstream protocol state.
    #[must_use]
    pub fn derive_context(&self, context: u128) -> Self {
        let mut root = ChaCha20Rng::from_seed(self.0);
        root.set_stream((context >> u64::BITS) as u64);
        root.set_word_pos(u128::from(context as u64) * SLOT_WORDS);
        let mut seed = [0_u8; 32];
        root.fill_bytes(&mut seed);
        Self(seed)
    }

    /// Returns a downstream `ChaCha20` generator seeded by the slot selected
    /// by `purpose` and `index`.
    ///
    /// The counter position, in 32-bit words, is
    /// `(u128::from(purpose) << 32) + index * 8`. The stream position is the
    /// only root-stream state, so derivation order does not matter: streams for
    /// `(purpose A, index i)` and `(purpose B, index j)` yield identical bytes
    /// regardless of which is derived first.
    ///
    /// # Errors
    ///
    /// Returns [`PrfError::IndexOutOfRange`] if `index` exceeds
    /// `MAX_INDEX`, since such a slot would overlap the next purpose's
    /// window.
    pub fn stream(&self, purpose: u32, index: u64) -> Result<ChaCha20Rng, PrfError> {
        Self::check_index(index)?;
        let position = (u128::from(purpose) << u32::BITS) | (u128::from(index) * SLOT_WORDS);
        let mut root = ChaCha20Rng::from_seed(self.0);
        root.set_word_pos(position);
        let mut seed = [0_u8; 32];
        root.fill_bytes(&mut seed);
        Ok(ChaCha20Rng::from_seed(seed))
    }

    pub(crate) const fn check_index(index: u64) -> Result<(), PrfError> {
        if index > MAX_INDEX {
            Err(PrfError::IndexOutOfRange { index })
        } else {
            Ok(())
        }
    }
}

impl fmt::Debug for Prf {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter.debug_struct("Prf").finish_non_exhaustive()
    }
}

/// A rejected PRF derivation.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum PrfError {
    /// A stream index would overlap the next purpose's window.
    IndexOutOfRange {
        /// The rejected index.
        index: u64,
    },
}

impl fmt::Display for PrfError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::IndexOutOfRange { index } => write!(
                formatter,
                "stream index {index} exceeds the {SLOT_WORDS}-word slot budget of a purpose window"
            ),
        }
    }
}

impl std::error::Error for PrfError {}
