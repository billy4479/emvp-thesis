//! The 1D-SLSN-based EMVP protocol (ePrint 2025/858, Fig. 1), cyclic variant.
//!
//! The client holds a matrix `M in F^(m x ell)` and a short secret key. The
//! online phase produces compact shares of `M q` for any query vector `q`:
//! the server returns an `m x s` answer matrix while the client keeps an
//! `s`-vector and an `m`-vector such that `M q = M' p' - r'`.
//!
//! The key derivation expands the short key into the long-term secrets via
//! the [`Prf`]: the cyclic dual-code multiplier `g`, the public permutation
//! `Pi` of length `n = 2k`, and the stack of trapdoored mask blocks. All of
//! them are bound to the full public reconstruction context (the caller's
//! [`MaskContextId`], every [`EmvpParams`] field, the row count, and the
//! instance nonce) through the hierarchical derivation of
//! [`SecretKey::derive`]. Query randomness (the codeword and the nonzero
//! block scalars) is derived under a monotonic query index. Callers persist
//! the next index when reconstructing state after a restart.
//!
//! The data flow, matching the conventions of [`crate::code`] and the
//! `trapdoor_matrices` mask module:
//!
//! - encrypt: each matrix row is dual-encoded to `(-conv(g, m_pad) | m_pad)`
//!   (row times the dual generator `D = [-M_g^T | I_k]`), masked by adding
//!   the materialized mask row `R_row`, and permuted through the gather
//!   `Pi`, giving the encrypted matrix `M_hat = permute(M D + R)`.
//! - query: the padded query is placed in the systematic half,
//!   `q_tilde = (0^k | q_pad) + c` for a fresh codeword `c`, so
//!   `D q_tilde = q_pad`. The same gather as the matrix columns aligns the
//!   query with them, `q_pi[i] = q_tilde[Pi[i]]`. The client computes the
//!   mask share `r' = R q_tilde` through the trapdoor
//!   and hides each block of size `b` behind a fresh nonzero scalar:
//!   `q_hat = (alpha_0 q_pi[0..b] | ... | alpha_{s-1} q_pi[(s-1)b..n])`,
//!   keeping `p' = (alpha_0^{-1}, ..., alpha_{s-1}^{-1})` secret.
//! - answer: the server multiplies each column block of `M_hat` with the
//!   matching block of `q_hat`, returning `M' in F^(m x s)`.
//! - decode: `a = M' p' - r'`, which equals `M q` because
//!   `M' p' = M_hat q_pi = M D q_tilde + R q_tilde = M q_pad + r'`.
//!
//! The permutation decorrelates the cyclic code structure from the fixed
//! block grid (paper Section 3.1, the `P = X^k - 1` variant); it is public
//! and derived deterministically from the key.
//!
//! Public instance and query identifiers reject accidental artifact mixups;
//! they do not authenticate data received from a malicious transport.
//!
//! This is experimental cryptography: 1D-SLSN has no settled security
//! parameters, secret state is not zeroized, per-query timing and memory
//! access are not constant-time, and [`encrypt`] materializes the mask.

use std::fmt;

use prime_field_layer::{FieldElement, FieldError, PrimeField};
use rand_chacha::ChaCha20Rng;
use rand_core::{CryptoRng, Rng};
use rayon::prelude::*;
use trapdoor_matrices::{DenseMatrix, Permutation, RowStackMask, TdmError, TdmMask};

use crate::code::{CodeError, CyclicCodeScratch, CyclicDualCode};
use crate::params::{EmvpParams, ParamsError};
use crate::prf::{Prf, PrfError, purpose};
use crate::view::{
    AnswerRef, AnswerValues, EncryptedMatrixRef, EncryptedQueryRef, MatrixValues, QueryValues,
};

/// Caller-chosen identifier of the mask suite and its configuration.
///
/// Every [`SecretKey::derive`] and [`SecretKey::restore`] call requires one
/// explicitly, so a caller cannot silently re-derive state under a different
/// trapdoored-matrix construction, a different builder configuration, or
/// different protocol parameters while reusing persisted artifacts. The
/// value is opaque to the protocol: pick any stable `u128` (for example a
/// random 128-bit tag fixed at deployment time), and change it whenever the
/// mask construction, its parameters, or anything else about the
/// `build_block` closure's output changes. It must be persisted alongside
/// the instance nonce and the next query index, because
/// [`SecretKey::restore`] accepts it explicitly and reconstruction only
/// succeeds for the exact context the original derivation used.
///
/// The identifier is a domain-separation input, not a secret: distinct
/// context identifiers under the same root key yield statistically
/// independent protocol state.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct MaskContextId(u128);

impl MaskContextId {
    /// Wraps a caller-chosen stable identifier.
    #[must_use]
    pub const fn new(value: u128) -> Self {
        Self(value)
    }

    /// Convenience constructor for context identifiers that fit a `u64`.
    #[must_use]
    pub const fn from_u64(value: u64) -> Self {
        Self(value as u128)
    }

    /// Returns the wrapped stable identifier.
    #[must_use]
    pub const fn get(self) -> u128 {
        self.0
    }
}

/// Domain and stage constants of the hierarchical instance binding.
///
/// [`SecretKey::derive`] expands the root key into the instance PRF with a
/// fixed-order chain of [`Prf::derive_context`] steps: the domain tag, then
/// alternating stage tags and context fields. Every stage tag has its high
/// bit set, so a step never collides with the `stream(purpose, index)`
/// layout of the PRF windows (which always select stream zero), and every
/// field occupies exactly one `u128` step, so neither the order nor the
/// field boundaries can be confused.
mod binding {
    /// High bit shared by every stage tag, keeping the chain off the
    /// stream-zero windows that [`Prf::stream`] addresses.
    const STAGE_FLAG: u128 = 1_u128 << 127;

    /// Root domain tag: only EMVP instance reconstruction chains pass
    /// through here.
    pub const DOMAIN: u128 = STAGE_FLAG | 1;
    /// Announces the caller's [`MaskContextId`](crate::protocol::MaskContextId)
    /// step.
    pub const STAGE_MASK_CONTEXT: u128 = STAGE_FLAG | 2;
    /// Announces the four [`crate::params::EmvpParams`] field steps.
    pub const STAGE_PARAMS: u128 = STAGE_FLAG | 3;
    /// Announces the matrix row count step.
    pub const STAGE_ROWS: u128 = STAGE_FLAG | 4;
    /// Announces the instance nonce step, the last field of the chain.
    pub const STAGE_INSTANCE_NONCE: u128 = STAGE_FLAG | 5;
}

/// A rejected protocol operation.
#[derive(Clone, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ProtocolError {
    /// A slice did not have the required length.
    LengthMismatch {
        /// The rejected slice.
        name: &'static str,
        /// The required length.
        expected: usize,
        /// The observed length.
        actual: usize,
    },
    /// Dimension arithmetic overflowed.
    DimensionOverflow,
    /// Key-derivation randomness indexing failed.
    Prf(PrfError),
    /// Code construction, encoding, or codeword sampling failed.
    Code(CodeError),
    /// Mask construction, evaluation, or materialization failed.
    Mask(TdmError),
    /// Field arithmetic failed.
    Field(FieldError),
    /// Protocol parameters were malformed or did not meet the requested
    /// security level.
    Params(ParamsError),
    /// 1D-SLSN is not meaningful over a field with fewer than three elements.
    UnsupportedField {
        /// The rejected field modulus.
        modulus: u32,
    },
    /// The state's deterministic matrix mask was already consumed.
    AlreadyEncrypted,
    /// Two protocol artifacts belong to different matrix instances.
    InstanceMismatch {
        /// The rejected artifact.
        name: &'static str,
        /// The required public instance identifier.
        expected: u128,
        /// The observed public instance identifier.
        actual: u128,
    },
    /// Two protocol artifacts belong to different queries.
    QueryMismatch {
        /// The rejected artifact.
        name: &'static str,
        /// The required public query identifier.
        expected: u64,
        /// The observed public query identifier.
        actual: u64,
    },
    /// A caller-owned answer workspace lacked the arena capacity an
    /// operation required.
    Capacity {
        /// The arena word count the operation required.
        required: usize,
        /// The arena word count the workspace offered.
        available: usize,
    },
}

impl fmt::Display for ProtocolError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LengthMismatch {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "{name} length mismatch: expected {expected}, got {actual}"
            ),
            Self::DimensionOverflow => formatter.write_str("dimension arithmetic overflowed"),
            Self::Prf(error) => error.fmt(formatter),
            Self::Code(error) => error.fmt(formatter),
            Self::Mask(error) => error.fmt(formatter),
            Self::Field(error) => error.fmt(formatter),
            Self::Params(error) => error.fmt(formatter),
            Self::UnsupportedField { modulus } => {
                write!(
                    formatter,
                    "1D-SLSN requires a field larger than F_2, got modulus {modulus}"
                )
            }
            Self::AlreadyEncrypted => formatter.write_str(
                "derived state has already encrypted a matrix; use a new instance identifier for another matrix",
            ),
            Self::InstanceMismatch {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "{name} instance mismatch: expected {expected}, got {actual}"
            ),
            Self::QueryMismatch {
                name,
                expected,
                actual,
            } => write!(
                formatter,
                "{name} query mismatch: expected {expected}, got {actual}"
            ),
            Self::Capacity { required, available } => write!(
                formatter,
                "answer arena capacity mismatch: required {required} words, available {available}"
            ),
        }
    }
}

impl std::error::Error for ProtocolError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Prf(error) => Some(error),
            Self::Code(error) => Some(error),
            Self::Mask(error) => Some(error),
            Self::Field(error) => Some(error),
            Self::Params(error) => Some(error),
            _ => None,
        }
    }
}

impl From<PrfError> for ProtocolError {
    fn from(error: PrfError) -> Self {
        Self::Prf(error)
    }
}

impl From<CodeError> for ProtocolError {
    fn from(error: CodeError) -> Self {
        Self::Code(error)
    }
}

impl From<TdmError> for ProtocolError {
    fn from(error: TdmError) -> Self {
        Self::Mask(error)
    }
}

impl From<FieldError> for ProtocolError {
    fn from(error: FieldError) -> Self {
        Self::Field(error)
    }
}

impl From<ParamsError> for ProtocolError {
    fn from(error: ParamsError) -> Self {
        Self::Params(error)
    }
}

pub(crate) const fn check_len(
    name: &'static str,
    expected: usize,
    actual: usize,
) -> Result<(), ProtocolError> {
    if expected == actual {
        Ok(())
    } else {
        Err(ProtocolError::LengthMismatch {
            name,
            expected,
            actual,
        })
    }
}

/// The single-query answer row kernel: one row of `s` column-block dot
/// products.
///
/// Shared by every CPU answer path (the one-shot batch, the dispatcher's
/// CPU tier, and the plan-reserve-execute arena path), so all of them
/// produce bit-identical answer rows by construction.
pub(crate) fn fill_answer_row<const MODULUS: u32>(
    matrix_row: &[FieldElement<MODULUS>],
    query: &[FieldElement<MODULUS>],
    block_len: usize,
    answer_row: &mut [FieldElement<MODULUS>],
) {
    let zero = PrimeField::<MODULUS>::new().element_u32(0);
    for (block, slot) in answer_row.iter_mut().enumerate() {
        let start = block * block_len;
        // Slicing each block once removes the per-element bounds checks the
        // compiler cannot elide across the two operand slices, and the four
        // independent accumulators break the serial multiply-add dependency
        // chain (`dot_canonical` uses the same lane layout).
        let matrix_block = &matrix_row[start..start + block_len];
        let query_block = &query[start..start + block_len];
        let mut sums = [zero; 4];
        for (matrix, query) in matrix_block
            .chunks_exact(4)
            .zip(query_block.chunks_exact(4))
        {
            sums[0] += matrix[0] * query[0];
            sums[1] += matrix[1] * query[1];
            sums[2] += matrix[2] * query[2];
            sums[3] += matrix[3] * query[3];
        }
        let matrix_tail = matrix_block.chunks_exact(4).remainder();
        let query_tail = query_block.chunks_exact(4).remainder();
        for (&matrix, &query) in matrix_tail.iter().zip(query_tail) {
            sums[0] += matrix * query;
        }
        *slot = sums.into_iter().fold(zero, |total, sum| total + sum);
    }
}

/// The short client secret key.
///
/// It holds only the protocol parameters and the 32-byte PRF key; every
/// long-term secret is re-derived from it. The PRF key is secret and the
/// struct is not zeroized on drop.
pub struct SecretKey<const MODULUS: u32> {
    params: EmvpParams,
    prf: Prf,
}

impl<const MODULUS: u32> SecretKey<MODULUS> {
    /// Constructs a key from secure parameters and a uniform 32-byte key.
    ///
    /// # Errors
    ///
    /// Returns an error if parameter validation fails or `MODULUS <= 2`.
    pub fn new(params: EmvpParams, key: [u8; 32]) -> Result<Self, ProtocolError> {
        params.validate()?;
        Self::new_with_dimensions(params, key)
    }

    /// Constructs a key for a deliberately insecure research instance.
    ///
    /// Structural dimensions and the field requirement are still enforced.
    /// This bypasses only concrete attack-cost validation.
    ///
    /// # Errors
    ///
    /// Returns an error if dimensions are malformed or `MODULUS <= 2`.
    pub fn new_insecure(params: EmvpParams, key: [u8; 32]) -> Result<Self, ProtocolError> {
        params.validate_dimensions()?;
        Self::new_with_dimensions(params, key)
    }

    const fn new_with_dimensions(params: EmvpParams, key: [u8; 32]) -> Result<Self, ProtocolError> {
        if MODULUS <= 2 {
            return Err(ProtocolError::UnsupportedField { modulus: MODULUS });
        }
        Ok(Self {
            params,
            prf: Prf::new(key),
        })
    }

    /// Returns the protocol parameters.
    #[must_use]
    pub const fn params(&self) -> EmvpParams {
        self.params
    }

    /// Expands the short key into fresh long-term secrets for one matrix.
    ///
    /// A fresh 128-bit matrix nonce is sampled from `rng`. The RNG state must
    /// never be replayed under the same root key. Persist the nonce through
    /// [`DerivedState::instance_nonce`] together with the next query index
    /// *and* the [`MaskContextId`] passed here: [`Self::restore`] requires
    /// the exact same context, so losing it makes the instance
    /// unreconstructable.
    ///
    /// All deterministic instance state (the instance identifier, the code,
    /// the permutation, the mask streams, and every query's randomness) is
    /// derived from a PRF bound to the complete public reconstruction
    /// context: `context`, every [`EmvpParams`] field, `rows`, and the
    /// instance nonce, chained through [`Self::bound_instance_prf`] in a
    /// fixed order. Changing any of them therefore yields an independent
    /// instance even under the same root key and nonce.
    ///
    /// `rows` is the number of matrix rows to support; it determines how
    /// many square mask blocks are stacked. `build_block` constructs one
    /// mask block from the block's PRF stream and index, which is how the
    /// caller picks a trapdoored-matrix construction and its parameters.
    /// Whenever that construction or its configuration changes, the caller
    /// must change the [`MaskContextId`] it passes. Stacks of two or more
    /// blocks construct them across rayon workers, so the closure must be
    /// callable from several threads (`Fn` plus `Sync`).
    ///
    /// # Errors
    ///
    /// Returns errors from the PRF, the code and permutation sampling, the
    /// mask construction, or the closure itself.
    pub fn derive<M, F, R>(
        self,
        context: MaskContextId,
        rows: usize,
        rng: &mut R,
        build_block: F,
    ) -> Result<DerivedState<MODULUS, M>, ProtocolError>
    where
        M: TdmMask<MODULUS>,
        F: Fn(&mut ChaCha20Rng, usize) -> Result<M, ProtocolError> + Sync,
        R: CryptoRng + ?Sized,
    {
        let mut nonce = [0_u8; 16];
        rng.fill_bytes(&mut nonce);
        self.derive_inner(
            context,
            u128::from_le_bytes(nonce),
            0,
            rows,
            false,
            &build_block,
        )
    }

    /// Restores query state for an already encrypted matrix.
    ///
    /// `context`, `instance_nonce`, and `next_query_index` must be the
    /// durably persisted values from the original state. The context
    /// identifier identifies the mask suite and configuration the original
    /// derivation used; restoring under any other value yields a different
    /// (wrong) instance rather than the original state. Restored state
    /// cannot encrypt another matrix, which prevents deterministic mask
    /// reuse after restart.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::derive`].
    pub fn restore<M, F>(
        self,
        context: MaskContextId,
        instance_nonce: u128,
        next_query_index: u64,
        rows: usize,
        build_block: F,
    ) -> Result<DerivedState<MODULUS, M>, ProtocolError>
    where
        M: TdmMask<MODULUS>,
        F: Fn(&mut ChaCha20Rng, usize) -> Result<M, ProtocolError> + Sync,
    {
        self.derive_inner(
            context,
            instance_nonce,
            next_query_index,
            rows,
            true,
            &build_block,
        )
    }

    /// The PRF all of one instance's deterministic state derives from.
    ///
    /// The chain of [`Prf::derive_context`] steps binds the caller's
    /// [`MaskContextId`], every [`EmvpParams`] field, the row count, and the
    /// instance nonce to the root key, in the fixed order documented by the
    /// `binding` stage constants. Exposed so oracle tests can replicate the
    /// exact derivation; protocol users obtain the same PRF indirectly through
    /// [`Self::derive`] and [`Self::restore`].
    ///
    /// # Security
    ///
    /// The returned PRF exposes every secret derived for this instance,
    /// including mask and query randomness. It must be protected exactly like
    /// the root key and must never be shared with the server or another party.
    #[doc(hidden)]
    #[must_use]
    pub fn bound_instance_prf(
        &self,
        context: MaskContextId,
        rows: usize,
        instance_nonce: u128,
    ) -> Prf {
        let params = self.params;
        self.prf
            .derive_context(binding::DOMAIN)
            .derive_context(binding::STAGE_MASK_CONTEXT)
            .derive_context(context.get())
            .derive_context(binding::STAGE_PARAMS)
            .derive_context(params.k as u128)
            .derive_context(params.ell as u128)
            .derive_context(params.b as u128)
            .derive_context(u128::from(params.lambda))
            .derive_context(binding::STAGE_ROWS)
            .derive_context(rows as u128)
            .derive_context(binding::STAGE_INSTANCE_NONCE)
            .derive_context(instance_nonce)
    }

    fn derive_inner<M, F>(
        self,
        context: MaskContextId,
        instance_nonce: u128,
        next_query_index: u64,
        rows: usize,
        encrypted: bool,
        build_block: &F,
    ) -> Result<DerivedState<MODULUS, M>, ProtocolError>
    where
        M: TdmMask<MODULUS>,
        F: Fn(&mut ChaCha20Rng, usize) -> Result<M, ProtocolError> + Sync,
    {
        let k = self.params.k;
        let n = self.params.n()?;
        if rows == 0 {
            return Err(ProtocolError::LengthMismatch {
                name: "matrix rows",
                expected: 1,
                actual: 0,
            });
        }

        let prf = self.bound_instance_prf(context, rows, instance_nonce);
        let mut identifier_stream = prf.stream(purpose::INSTANCE_ID, 0)?;
        let mut identifier = [0_u8; 16];
        identifier_stream.fill_bytes(&mut identifier);
        let instance_id = u128::from_le_bytes(identifier);
        let mut code_stream = prf.stream(purpose::CODE_MULTIPLIER, 0)?;
        let code = CyclicDualCode::sample(k, &mut code_stream)?;

        let mut permutation_stream = prf.stream(purpose::CODE_PERMUTATION, 0)?;
        let permutation = Permutation::sample(n, &mut permutation_stream)?;

        let block_count = rows.div_ceil(n);
        let last_block = u64::try_from(block_count - 1)
            .map_err(|_conversion_error| ProtocolError::DimensionOverflow)?;
        Prf::check_index(last_block)?;
        // Every block seeds its own PRF stream, so the constructions are
        // independent and an indexed parallel collect preserves block order;
        // the result is identical to the serial loop. Block construction
        // always seeds a CSPRNG and builds trapdoor state (cached NTT plans,
        // spectra, or sparse secrets), comfortably above rayon scheduling
        // overhead, so any stack of two or more blocks parallelizes.
        let threads = rayon::current_num_threads();
        let blocks: Vec<M> = if threads > 1 && block_count >= 2 {
            (0..block_count)
                .into_par_iter()
                .map(|block_index| {
                    let mut stream = prf.stream(purpose::TDM, block_index as u64)?;
                    build_block(&mut stream, block_index)
                })
                .collect::<Result<Vec<_>, ProtocolError>>()?
        } else {
            let mut blocks = Vec::with_capacity(block_count);
            for block_index in 0..block_count {
                let mut stream = prf.stream(purpose::TDM, block_index as u64)?;
                blocks.push(build_block(&mut stream, block_index)?);
            }
            blocks
        };
        let mask = RowStackMask::new(blocks, rows)?;
        check_len("mask columns", n, mask.dims().1)?;
        let online_scratch = QueryScratch {
            code: code.scratch(),
            mask: mask.scratch(),
            q_tilde: vec![PrimeField::<MODULUS>::new().element_u32(0); n],
        };

        Ok(DerivedState {
            params: self.params,
            instance_nonce,
            instance_id,
            next_query_index,
            prf,
            code,
            permutation,
            mask,
            online_scratch,
            encrypted,
        })
    }
}

/// Caller-owned reusable storage for concurrent query generation.
///
/// Construct it with [`DerivedState::query_scratch`]. Buffers retain secret
/// intermediate values and are not zeroized on drop.
pub struct QueryScratch<const MODULUS: u32, M: TdmMask<MODULUS>> {
    code: CyclicCodeScratch<MODULUS>,
    mask: <RowStackMask<M, MODULUS> as TdmMask<MODULUS>>::Scratch,
    q_tilde: Vec<FieldElement<MODULUS>>,
}

/// One non-reusable query identifier reserved from a [`DerivedState`].
///
/// The token is intentionally neither `Clone` nor `Copy`; consuming it in
/// [`query_with_scratch`] prevents accidental in-process randomness reuse.
pub struct QueryReservation {
    instance_id: u128,
    query_id: u64,
}

impl QueryReservation {
    /// Returns the public matrix-instance identifier the token belongs to.
    #[must_use]
    pub const fn instance_id(&self) -> u128 {
        self.instance_id
    }

    /// Returns the reserved public query identifier.
    #[must_use]
    pub const fn query_id(&self) -> u64 {
        self.query_id
    }
}

/// Iterator over query reservations allocated in one counter update.
pub struct QueryReservations {
    instance_id: u128,
    range: std::ops::Range<u64>,
}

impl Iterator for QueryReservations {
    type Item = QueryReservation;

    fn next(&mut self) -> Option<Self::Item> {
        self.range.next().map(|query_id| QueryReservation {
            instance_id: self.instance_id,
            query_id,
        })
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        self.range.size_hint()
    }
}

impl ExactSizeIterator for QueryReservations {}

/// The expanded long-term secrets for one matrix encryption and many queries.
///
/// The code multiplier, its cached transform, and the mask trapdoors are
/// secret; none of them are zeroized on drop. The permutation is public.
pub struct DerivedState<const MODULUS: u32, M: TdmMask<MODULUS>> {
    params: EmvpParams,
    instance_nonce: u128,
    instance_id: u128,
    next_query_index: u64,
    prf: Prf,
    code: CyclicDualCode<MODULUS>,
    permutation: Permutation,
    mask: RowStackMask<M, MODULUS>,
    online_scratch: QueryScratch<MODULUS, M>,
    encrypted: bool,
}

impl<const MODULUS: u32, M: TdmMask<MODULUS>> DerivedState<MODULUS, M> {
    /// Returns the protocol parameters.
    #[must_use]
    pub const fn params(&self) -> EmvpParams {
        self.params
    }

    /// Returns the public identifier of the encrypted matrix instance.
    #[must_use]
    pub const fn instance_id(&self) -> u128 {
        self.instance_id
    }

    /// Returns the matrix nonce that must be persisted for reconstruction.
    #[must_use]
    pub const fn instance_nonce(&self) -> u128 {
        self.instance_nonce
    }

    /// Returns the query index that must be persisted for state reconstruction.
    #[must_use]
    pub const fn next_query_index(&self) -> u64 {
        self.next_query_index
    }

    /// Allocates independent scratch storage for [`query_with_scratch`].
    #[must_use]
    pub fn query_scratch(&self) -> QueryScratch<MODULUS, M> {
        QueryScratch {
            code: self.code.scratch(),
            mask: self.mask.scratch(),
            q_tilde: vec![
                PrimeField::<MODULUS>::new().element_u32(0);
                self.online_scratch.q_tilde.len()
            ],
        }
    }

    /// Reserves unique query identifiers for concurrent query generation.
    ///
    /// Persist [`Self::next_query_index`] before reservations leave the process,
    /// then move each returned token to [`query_with_scratch`].
    ///
    /// # Errors
    ///
    /// Returns an error if the counter overflows or exceeds the PRF index
    /// budget. The counter remains unchanged on error.
    pub fn reserve_query_ids(&mut self, count: u64) -> Result<QueryReservations, ProtocolError> {
        let start = self.next_query_index;
        let end = start
            .checked_add(count)
            .ok_or(ProtocolError::DimensionOverflow)?;
        if count > 0 {
            Prf::check_index(end - 1)?;
        }
        self.next_query_index = end;
        Ok(QueryReservations {
            instance_id: self.instance_id,
            range: start..end,
        })
    }

    /// Returns the secret cyclic dual code.
    #[must_use]
    pub const fn code(&self) -> &CyclicDualCode<MODULUS> {
        &self.code
    }

    /// Returns the public permutation as gather indices.
    #[must_use]
    pub fn permutation_indices(&self) -> &[usize] {
        self.permutation.indices()
    }

    /// Returns the mask stack.
    #[must_use]
    pub const fn mask(&self) -> &RowStackMask<M, MODULUS> {
        &self.mask
    }
}

/// The encrypted matrix `M_hat = permute(M D + R)` held by the server.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncryptedMatrix<const MODULUS: u32> {
    matrix: DenseMatrix<MODULUS>,
    instance_id: u128,
}

impl<const MODULUS: u32> EncryptedMatrix<MODULUS> {
    /// Returns the encrypted matrix rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.matrix.rows()
    }

    /// Returns the encrypted matrix columns (`n = 2k`).
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.matrix.columns()
    }

    /// Returns the row-major encrypted entries.
    #[must_use]
    pub fn values(&self) -> &[FieldElement<MODULUS>] {
        self.matrix.values()
    }

    /// Returns the underlying dense matrix.
    #[must_use]
    pub const fn dense(&self) -> &DenseMatrix<MODULUS> {
        &self.matrix
    }

    /// Returns the public matrix-instance identifier.
    #[must_use]
    pub const fn instance_id(&self) -> u128 {
        self.instance_id
    }

    /// Returns a borrowed view over this encrypted matrix.
    ///
    /// The view borrows the ciphertext slice; it holds no storage of its
    /// own and exposes the same shape and identifier accessors.
    #[must_use]
    pub fn as_ref(&self) -> EncryptedMatrixRef<'_, MODULUS> {
        EncryptedMatrixRef::new_unchecked(
            self.instance_id,
            self.matrix.rows(),
            self.matrix.columns(),
            self.matrix.values(),
        )
    }

    /// Rebuilds an encrypted matrix from its parts.
    ///
    /// # Errors
    ///
    /// Returns an error if the value length does not match the dimensions.
    pub fn from_parts(
        instance_id: u128,
        rows: usize,
        columns: usize,
        values: Vec<FieldElement<MODULUS>>,
    ) -> Result<Self, ProtocolError> {
        Ok(Self {
            matrix: DenseMatrix::new(rows, columns, values)?,
            instance_id,
        })
    }
}

/// The encrypted query `q_hat` sent to the server.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncryptedQuery<const MODULUS: u32> {
    values: Vec<FieldElement<MODULUS>>,
    instance_id: u128,
    query_id: u64,
}

impl<const MODULUS: u32> EncryptedQuery<MODULUS> {
    /// Returns the encrypted query coordinates (length `n`).
    #[must_use]
    pub fn values(&self) -> &[FieldElement<MODULUS>] {
        &self.values
    }

    /// Returns the public matrix-instance identifier.
    #[must_use]
    pub const fn instance_id(&self) -> u128 {
        self.instance_id
    }

    /// Returns the public query identifier.
    #[must_use]
    pub const fn query_id(&self) -> u64 {
        self.query_id
    }

    /// Returns a borrowed view over this encrypted query.
    ///
    /// The view borrows the coordinate slice; it holds no storage of its
    /// own and exposes the same identifier accessors.
    #[must_use]
    pub fn as_ref(&self) -> EncryptedQueryRef<'_, MODULUS> {
        EncryptedQueryRef::new_unchecked(self.instance_id, self.query_id, &self.values)
    }

    /// Rebuilds an encrypted query from its coordinates.
    #[must_use]
    pub const fn from_parts(
        instance_id: u128,
        query_id: u64,
        values: Vec<FieldElement<MODULUS>>,
    ) -> Self {
        Self {
            values,
            instance_id,
            query_id,
        }
    }
}

/// The client's decoding information `q' = (p', r')`.
#[derive(Clone, Eq, PartialEq)]
pub struct DecodingKey<const MODULUS: u32> {
    p_prime: Vec<FieldElement<MODULUS>>,
    r_prime: Vec<FieldElement<MODULUS>>,
    instance_id: u128,
    query_id: u64,
}

impl<const MODULUS: u32> fmt::Debug for DecodingKey<MODULUS> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DecodingKey")
            .field("blocks", &self.p_prime.len())
            .field("rows", &self.r_prime.len())
            .field("instance_id", &self.instance_id)
            .field("query_id", &self.query_id)
            .finish_non_exhaustive()
    }
}

impl<const MODULUS: u32> DecodingKey<MODULUS> {
    /// Rebuilds a decoding key from its parts.
    #[must_use]
    pub const fn from_parts(
        instance_id: u128,
        query_id: u64,
        p_prime: Vec<FieldElement<MODULUS>>,
        r_prime: Vec<FieldElement<MODULUS>>,
    ) -> Self {
        Self {
            p_prime,
            r_prime,
            instance_id,
            query_id,
        }
    }

    /// Returns the block coefficients `p'` (length `s`).
    #[must_use]
    pub fn p_prime(&self) -> &[FieldElement<MODULUS>] {
        &self.p_prime
    }

    /// Returns the mask share `r'` (length `m`).
    #[must_use]
    pub fn r_prime(&self) -> &[FieldElement<MODULUS>] {
        &self.r_prime
    }

    /// Returns the public matrix-instance identifier.
    #[must_use]
    pub const fn instance_id(&self) -> u128 {
        self.instance_id
    }

    /// Returns the public query identifier.
    #[must_use]
    pub const fn query_id(&self) -> u64 {
        self.query_id
    }
}

/// The server's answer `M' in F^(m x s)`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnswerMatrix<const MODULUS: u32> {
    values: Vec<FieldElement<MODULUS>>,
    rows: usize,
    blocks: usize,
    instance_id: u128,
    query_id: u64,
}

impl<const MODULUS: u32> AnswerMatrix<MODULUS> {
    /// Rebuilds an answer matrix from its parts.
    #[must_use]
    pub const fn from_parts(
        instance_id: u128,
        query_id: u64,
        values: Vec<FieldElement<MODULUS>>,
        rows: usize,
        blocks: usize,
    ) -> Self {
        Self {
            values,
            rows,
            blocks,
            instance_id,
            query_id,
        }
    }

    /// Returns the answer row count (`m`).
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the answer column count (`s`).
    #[must_use]
    pub const fn blocks(&self) -> usize {
        self.blocks
    }

    /// Returns the row-major answer entries.
    #[must_use]
    pub fn values(&self) -> &[FieldElement<MODULUS>] {
        &self.values
    }

    /// Returns the public matrix-instance identifier.
    #[must_use]
    pub const fn instance_id(&self) -> u128 {
        self.instance_id
    }

    /// Returns the public query identifier.
    #[must_use]
    pub const fn query_id(&self) -> u64 {
        self.query_id
    }

    /// Returns a borrowed view over this answer matrix.
    ///
    /// The view borrows the answer slice; it holds no storage of its own
    /// and exposes the same shape and identifier accessors.
    #[must_use]
    pub fn as_ref(&self) -> AnswerRef<'_, MODULUS> {
        AnswerRef::new_unchecked(
            self.instance_id,
            self.query_id,
            self.rows,
            self.blocks,
            &self.values,
        )
    }
}

/// Encrypts a row-major `rows x ell` matrix for the server.
///
/// The mask is materialized into the `rows x n` ciphertext allocation and
/// overwritten row by row after encoding and permutation. Rows are
/// independent, so large matrices run the encode-mask-permute pipeline
/// across rayon workers, each owning fresh code scratch and a row buffer;
/// the ciphertext is identical to the serial row loop. This is the offline
/// phase of the protocol. A derived state can encrypt exactly one matrix.
///
/// # Errors
///
/// Returns an error before any output is produced if the state already
/// encrypted a matrix or the matrix length differs from `rows * ell`; those
/// checks run before the ciphertext is touched. A code or mask failure after
/// that point is not possible for validated parameters and scratch, but the
/// ciphertext allocation may already hold partially written rows if one
/// occurred.
pub fn encrypt<const MODULUS: u32, M: TdmMask<MODULUS>>(
    state: &mut DerivedState<MODULUS, M>,
    matrix: &[FieldElement<MODULUS>],
) -> Result<EncryptedMatrix<MODULUS>, ProtocolError> {
    if state.encrypted {
        return Err(ProtocolError::AlreadyEncrypted);
    }
    let ell = state.params.ell;
    let n = state.params.n()?;
    let rows = state.mask.dims().0;
    let expected = rows
        .checked_mul(ell)
        .ok_or(ProtocolError::DimensionOverflow)?;
    check_len("input matrix", expected, matrix.len())?;

    let mut encoded = state.mask.materialize()?.into_values();
    fill_encrypted_rows(
        &state.code,
        &state.permutation,
        matrix,
        &mut encoded,
        ell,
        n,
        rows,
    )?;
    let encrypted = EncryptedMatrix {
        matrix: DenseMatrix::new(rows, n, encoded)?,
        instance_id: state.instance_id,
    };
    state.encrypted = true;
    Ok(encrypted)
}

/// Fills the ciphertext rows: dual-encode each plaintext row, add the
/// materialized mask row, and apply the public gather permutation.
///
/// One row costs at least one `n`-element convolution, mask addition, and
/// gather, so `rows * n` is a conservative multiplication estimate for the
/// parallel guard. Workers initialize one code scratch and one row buffer
/// per thread, and every row writes a disjoint ciphertext slice.
fn fill_encrypted_rows<const MODULUS: u32>(
    code: &CyclicDualCode<MODULUS>,
    permutation: &Permutation,
    matrix: &[FieldElement<MODULUS>],
    encoded: &mut [FieldElement<MODULUS>],
    ell: usize,
    n: usize,
    rows: usize,
) -> Result<(), ProtocolError> {
    let zero = PrimeField::<MODULUS>::new().element_u32(0);
    let threads = rayon::current_num_threads();
    let work = rows.saturating_mul(n);
    if crate::dispatch::is_parallel_work(work, rows, threads) {
        encoded
            .par_chunks_mut(n)
            .zip(matrix.par_chunks(ell))
            .map_init(
                || (code.scratch(), vec![zero; n]),
                |(scratch, row), (output_row, matrix_row)| {
                    encode_masked_row(code, permutation, matrix_row, row, output_row, scratch)
                },
            )
            .try_for_each(|result| result)
    } else {
        let mut scratch = code.scratch();
        let mut row = vec![zero; n];
        for (output_row, matrix_row) in encoded.chunks_mut(n).zip(matrix.chunks(ell)) {
            encode_masked_row(
                code,
                permutation,
                matrix_row,
                &mut row,
                output_row,
                &mut scratch,
            )?;
        }
        Ok(())
    }
}

/// Encodes one plaintext row, masks it with the materialized row, and
/// gathers it through the public permutation into `output_row`.
fn encode_masked_row<const MODULUS: u32>(
    code: &CyclicDualCode<MODULUS>,
    permutation: &Permutation,
    matrix_row: &[FieldElement<MODULUS>],
    row: &mut [FieldElement<MODULUS>],
    output_row: &mut [FieldElement<MODULUS>],
    scratch: &mut CyclicCodeScratch<MODULUS>,
) -> Result<(), ProtocolError> {
    code.dual_encode_row(matrix_row, row, scratch)?;
    for (slot, &masked) in row.iter_mut().zip(output_row.iter()) {
        *slot += masked;
    }
    Ok(permutation.apply(row, output_row)?)
}

/// Generates an encrypted query for `q` plus the client's decoding key.
///
/// The codeword and nonzero block scalars are derived from independent PRF
/// streams under the state's next query index. The mask share `r'` is
/// evaluated through the trapdoor without allocating temporary storage.
/// Persist [`DerivedState::next_query_index`] before releasing the resulting
/// artifacts when queries must survive process crashes.
///
/// # Errors
///
/// Returns an error without consuming an identifier if `q` has the wrong
/// length. A later code, mask, or field failure consumes the reserved
/// identifier so retrying cannot reuse its randomness.
pub fn query<const MODULUS: u32, M: TdmMask<MODULUS>>(
    state: &mut DerivedState<MODULUS, M>,
    q: &[FieldElement<MODULUS>],
) -> Result<(EncryptedQuery<MODULUS>, DecodingKey<MODULUS>), ProtocolError> {
    check_len("query vector", state.params.ell, q.len())?;
    let Some(reservation) = state.reserve_query_ids(1)?.next() else {
        // Defensive: `reserve_query_ids(1)` always yields exactly one token,
        // so this arm is unreachable; matching keeps the reservation protocol
        // explicit.
        return Err(ProtocolError::DimensionOverflow);
    };
    query_core(
        state.params,
        state.instance_id,
        &state.prf,
        &state.code,
        &state.permutation,
        &state.mask,
        q,
        reservation.query_id,
        &mut state.online_scratch,
    )
}

/// Generates a query with a pre-reserved identifier and caller-owned scratch.
///
/// Callers first reserve tokens through [`DerivedState::reserve_query_ids`]
/// and give each concurrent worker its own [`QueryScratch`].
///
/// # Errors
///
/// Returns an error for an out-of-range identifier, malformed query, or a
/// failed code, mask, permutation, or field operation.
#[expect(
    clippy::needless_pass_by_value,
    reason = "consuming the non-cloneable reservation prevents identifier reuse"
)]
pub fn query_with_scratch<const MODULUS: u32, M: TdmMask<MODULUS>>(
    state: &DerivedState<MODULUS, M>,
    reservation: QueryReservation,
    q: &[FieldElement<MODULUS>],
    scratch: &mut QueryScratch<MODULUS, M>,
) -> Result<(EncryptedQuery<MODULUS>, DecodingKey<MODULUS>), ProtocolError> {
    if reservation.instance_id != state.instance_id {
        return Err(ProtocolError::InstanceMismatch {
            name: "query reservation",
            expected: state.instance_id,
            actual: reservation.instance_id,
        });
    }
    query_core(
        state.params,
        state.instance_id,
        &state.prf,
        &state.code,
        &state.permutation,
        &state.mask,
        q,
        reservation.query_id,
        scratch,
    )
}

/// Generates encrypted queries for a batch of query vectors plus the
/// client's decoding keys.
///
/// Every query is generated exactly as [`query`] generates it alone: the
/// batch reserves one contiguous identifier range in a single counter
/// update, and query `i` carries the identifier the serial loop would have
/// assigned it, so its artifacts are bit-for-bit identical to the serial
/// loop's. Generation parallelizes across rayon workers, each owning a
/// [`QueryScratch`] built once per worker and reused for every query it
/// receives.
///
/// # Errors
///
/// Validation is all-or-nothing, matching [`answer_batch`]: every query
/// vector must have length `ell` before any identifier is reserved, so a
/// malformed batch consumes nothing. An empty batch is rejected with a
/// `queries` length mismatch. A later code, mask, or field failure for one
/// query leaves earlier identifiers reserved; persist
/// [`DerivedState::next_query_index`] only after the batch succeeds.
pub fn query_batch<const MODULUS: u32, M: TdmMask<MODULUS>>(
    state: &mut DerivedState<MODULUS, M>,
    queries: &[&[FieldElement<MODULUS>]],
) -> Result<Vec<(EncryptedQuery<MODULUS>, DecodingKey<MODULUS>)>, ProtocolError> {
    let Some(first) = queries.first() else {
        return Err(ProtocolError::LengthMismatch {
            name: "queries",
            expected: 1,
            actual: 0,
        });
    };
    check_len("query vector", state.params.ell, first.len())?;
    for q in &queries[1..] {
        check_len("query vector", state.params.ell, q.len())?;
    }
    let count = u64::try_from(queries.len())
        .map_err(|_conversion_error| ProtocolError::DimensionOverflow)?;
    let width = state.params.n()?;
    let reservations: Vec<QueryReservation> = state.reserve_query_ids(count)?.collect();
    let borrowed = &*state;
    let params = borrowed.params;
    let instance_id = borrowed.instance_id;
    let prf = &borrowed.prf;
    let code = &borrowed.code;
    let permutation = &borrowed.permutation;
    let mask = &borrowed.mask;
    let zero = PrimeField::<MODULUS>::new().element_u32(0);
    queries
        .par_iter()
        .zip(reservations)
        .map_init(
            || QueryScratch {
                code: code.scratch(),
                mask: mask.scratch(),
                q_tilde: vec![zero; width],
            },
            |scratch, (q, reservation)| {
                query_core(
                    params,
                    instance_id,
                    prf,
                    code,
                    permutation,
                    mask,
                    q,
                    reservation.query_id,
                    scratch,
                )
            },
        )
        .collect()
}

#[expect(
    clippy::too_many_arguments,
    reason = "the helper separates immutable long-term state from caller-owned scratch"
)]
fn query_core<const MODULUS: u32, M: TdmMask<MODULUS>>(
    params: EmvpParams,
    instance_id: u128,
    prf: &Prf,
    code: &CyclicDualCode<MODULUS>,
    permutation: &Permutation,
    mask: &RowStackMask<M, MODULUS>,
    q: &[FieldElement<MODULUS>],
    query_id: u64,
    scratch: &mut QueryScratch<MODULUS, M>,
) -> Result<(EncryptedQuery<MODULUS>, DecodingKey<MODULUS>), ProtocolError> {
    let code_dim = params.k;
    let ell = params.ell;
    let width = params.n()?;
    let block_len = params.block_size();
    let block_count = params.blocks()?;
    let rows = mask.dims().0;
    check_len("query vector", ell, q.len())?;
    check_len("query scratch vector", width, scratch.q_tilde.len())?;
    let mut codeword_stream = prf.stream(purpose::QUERY_CODEWORD, query_id)?;
    let mut nonzero_stream = prf.stream(purpose::QUERY_NONZERO, query_id)?;

    let zero = PrimeField::<MODULUS>::new().element_u32(0);
    code.sample_codeword(
        &mut codeword_stream,
        &mut scratch.q_tilde,
        &mut scratch.code,
    )?;
    // Add the padded query in the systematic (second) half.
    for (slot, &coefficient) in scratch.q_tilde[code_dim..].iter_mut().zip(q) {
        *slot += coefficient;
    }

    // Mask share through the trapdoor, on the unpermuted query: the server
    // answer unwinds to X q_tilde = M D q_tilde + R q_tilde = M q_pad + R q_tilde.
    let mut r_prime = vec![zero; rows];
    mask.apply(&scratch.q_tilde, &mut r_prime, &mut scratch.mask)?;

    // Gather through the permutation. The encrypted matrix columns are the
    // same gather of the encoded columns, M_hat[i] = X[Pi[i]], so the answer
    // matches X q_tilde exactly when the query is gathered too.
    let mut q_hat = vec![zero; width];
    permutation.apply(&scratch.q_tilde, &mut q_hat)?;

    // Hide each block behind a fresh nonzero scalar. At this point `q_tilde`
    // is dead — both consumers below (`r'` above and the gather above) have
    // read it — so its head doubles as staging for the alphas, letting the
    // batch inversion below run without a fresh allocation. Reordering any
    // consumer after this loop would silently corrupt the query.
    let field = PrimeField::<MODULUS>::new();
    let mut p_prime = vec![zero; block_count];
    for block in 0..block_count {
        let alpha = field.sample_uniform_nonzero(&mut nonzero_stream);
        scratch.q_tilde[block] = alpha;
        for slot in &mut q_hat[block * block_len..(block + 1) * block_len] {
            *slot *= alpha;
        }
    }
    field.batch_inv_elements(&scratch.q_tilde[..block_count], &mut p_prime)?;

    Ok((
        EncryptedQuery {
            values: q_hat,
            instance_id,
            query_id,
        },
        DecodingKey {
            p_prime,
            r_prime,
            instance_id,
            query_id,
        },
    ))
}

/// Answers an encrypted query into caller-owned row-major storage.
///
/// This is the server's online work: `s` column-block matrix-vector
/// products, `m * n` field multiplications in total. `output` must contain
/// `matrix.rows() * params.blocks()` elements. This variant allows servers
/// to reuse allocations across queries.
///
/// # Errors
///
/// Returns an error before output mutation for malformed dimensions, context
/// mismatches, or an incorrectly sized output.
pub fn answer_into<const MODULUS: u32>(
    params: &EmvpParams,
    matrix: &EncryptedMatrix<MODULUS>,
    query: &EncryptedQuery<MODULUS>,
    output: &mut [FieldElement<MODULUS>],
) -> Result<(), ProtocolError> {
    let (n, b, s, rows) = validate_answer(params, matrix, query)?;
    let output_len = rows
        .checked_mul(s)
        .ok_or(ProtocolError::DimensionOverflow)?;
    check_len("answer output", output_len, output.len())?;
    fill_answer(matrix, query, output, n, b, s, rows);
    Ok(())
}

/// Validates the parameters and one query against the encrypted matrix and
/// returns `(n, b, s, rows)`.
fn validate_answer<const MODULUS: u32>(
    params: &EmvpParams,
    matrix: &EncryptedMatrix<MODULUS>,
    query: &EncryptedQuery<MODULUS>,
) -> Result<(usize, usize, usize, usize), ProtocolError> {
    let n = validate_matrix_shape(params, matrix)?;
    validate_query_against_matrix(n, matrix.instance_id(), query)?;
    let b = params.block_size();
    let s = params.blocks()?;
    let rows = matrix.rows();
    Ok((n, b, s, rows))
}

/// Validates one encrypted matrix's shape against `params` and returns the
/// codeword length `n = 2k`.
///
/// Accepts any borrowed or owned matrix representation implementing
/// [`MatrixValues`], so the server-side paths can validate wire workspaces
/// without copying them into [`EncryptedMatrix`].
///
/// This is the shape half of the CPU answer-path validation
/// ([`validate_answer`]) and the single validation entry point shared by the
/// dispatcher's fast-fail construction and the GPU answerer's upload, so
/// CPU, dispatch, and GPU reject the same matrix shapes before any work.
pub(crate) fn validate_matrix_shape<const MODULUS: u32, V: MatrixValues<MODULUS>>(
    params: &EmvpParams,
    matrix: &V,
) -> Result<usize, ProtocolError> {
    params.validate_dimensions()?;
    let n = params.n()?;
    check_len("encrypted matrix columns", n, matrix.columns())?;
    let rows = matrix.rows();
    if rows == 0 {
        return Err(ProtocolError::LengthMismatch {
            name: "matrix rows",
            expected: 1,
            actual: 0,
        });
    }
    let words = rows
        .checked_mul(n)
        .ok_or(ProtocolError::DimensionOverflow)?;
    check_len("encrypted matrix values", words, matrix.values().len())?;
    Ok(n)
}

/// Checks one query's length and instance identifier against the matrix
/// instance.
///
/// Accepts any borrowed or owned query representation implementing
/// [`QueryValues`], mirroring [`validate_matrix_shape`].
///
/// Exposed next to [`validate_matrix_shape`] so the server-side dispatch and
/// device paths can reuse the exact CPU validation; callers pass the
/// instance identifier of the matrix they answer against.
pub(crate) fn validate_query_against_matrix<const MODULUS: u32, Q: QueryValues<MODULUS>>(
    n: usize,
    matrix_instance_id: u128,
    query: &Q,
) -> Result<(), ProtocolError> {
    check_len("encrypted query", n, query.values().len())?;
    if query.instance_id() != matrix_instance_id {
        return Err(ProtocolError::InstanceMismatch {
            name: "encrypted query",
            expected: matrix_instance_id,
            actual: query.instance_id(),
        });
    }
    Ok(())
}

fn fill_answer<const MODULUS: u32>(
    matrix: &EncryptedMatrix<MODULUS>,
    query: &EncryptedQuery<MODULUS>,
    output: &mut [FieldElement<MODULUS>],
    n: usize,
    b: usize,
    s: usize,
    rows: usize,
) {
    let threads = rayon::current_num_threads();
    let work = rows.saturating_mul(n);
    if crate::dispatch::is_parallel_work(work, rows, threads) {
        output
            .par_chunks_mut(s)
            .zip(matrix.values().par_chunks(n))
            .for_each(|(output_row, matrix_row)| {
                fill_answer_row(matrix_row, &query.values, b, output_row);
            });
    } else {
        for (output_row, matrix_row) in output.chunks_exact_mut(s).zip(matrix.values().chunks(n)) {
            fill_answer_row(matrix_row, &query.values, b, output_row);
        }
    }
}

/// Answers a batch of encrypted queries against one encrypted matrix.
///
/// Every query is answered exactly as [`answer_into`] answers it alone:
/// the server performs `s` column-block matrix-vector products per row,
/// for `queries.len() * m * n` field multiplications in total. The answers
/// are written into one query-major arena whose flattened (query, row) grid
/// runs across rayon workers when it clears the crate's parallel-work
/// threshold; smaller batches stay on the serial row loop.
///
/// # Errors
///
/// Validation is all-or-nothing, matching [`answer_into`]: parameters are
/// checked once, and every query must have length `n` and carry the
/// matrix's instance identifier before any output is produced. An empty
/// batch is rejected with a `queries` length mismatch, consistent with how
/// the crate treats empty block lists and row counts elsewhere.
pub fn answer_batch<const MODULUS: u32>(
    params: &EmvpParams,
    matrix: &EncryptedMatrix<MODULUS>,
    queries: &[EncryptedQuery<MODULUS>],
) -> Result<Vec<AnswerMatrix<MODULUS>>, ProtocolError> {
    let Some(first) = queries.first() else {
        return Err(ProtocolError::LengthMismatch {
            name: "queries",
            expected: 1,
            actual: 0,
        });
    };
    let (n, b, s, rows) = validate_answer(params, matrix, first)?;
    for query in &queries[1..] {
        validate_query_against_matrix(n, matrix.instance_id(), query)?;
    }
    let answer_len = rows
        .checked_mul(s)
        .ok_or(ProtocolError::DimensionOverflow)?;
    let arena_len = queries
        .len()
        .checked_mul(answer_len)
        .ok_or(ProtocolError::DimensionOverflow)?;
    let zero = PrimeField::<MODULUS>::new().element_u32(0);
    let mut arena = vec![zero; arena_len];
    fill_answer_batch(matrix, queries, &mut arena, n, b, s, rows);
    Ok(arena
        .chunks_exact(answer_len)
        .zip(queries.iter())
        .map(|(values, query)| {
            AnswerMatrix::from_parts(
                matrix.instance_id(),
                query.query_id(),
                values.to_vec(),
                rows,
                s,
            )
        })
        .collect())
}

/// Fills a query-major arena holding each batch answer back to back.
///
/// The arena splits into `queries.len() * rows` disjoint answer rows, so
/// the flattened grid parallelizes without nested pools and every row
/// reuses the single-query row kernel. The batch is generic over the query
/// representation ([`QueryValues`], which must be [`Sync`] because the
/// parallel tier reads it across rayon workers), so the owned and borrowed
/// answer paths share one kernel. The parallel tier is selected by the
/// shared dispatch policy ([`crate::dispatch`]); smaller batches stay on
/// the serial row loop.
fn fill_answer_batch<const MODULUS: u32, Q: QueryValues<MODULUS> + Sync>(
    matrix: &EncryptedMatrix<MODULUS>,
    queries: &[Q],
    arena: &mut [FieldElement<MODULUS>],
    n: usize,
    b: usize,
    s: usize,
    rows: usize,
) {
    let threads = rayon::current_num_threads();
    let grid = queries.len().saturating_mul(rows);
    let work = grid.saturating_mul(n);
    if crate::dispatch::is_parallel_work(work, grid, threads) {
        arena
            .par_chunks_mut(s)
            .enumerate()
            .for_each(|(flat_row, answer_row)| {
                let query = &queries[flat_row / rows];
                let matrix_row_index = flat_row % rows;
                let matrix_row = &matrix.values()[matrix_row_index * n..(matrix_row_index + 1) * n];
                fill_answer_row(matrix_row, query.values(), b, answer_row);
            });
    } else {
        fill_answer_batch_serial(matrix, queries, arena, n, b, s, rows);
    }
}

/// Fills a query-major arena on the serial row loop.
///
/// The serial tier shared by [`fill_answer_batch`] and the plan-reserve-
/// execute path ([`crate::answer::execute_answer_batch`]): every answer row
/// is computed by the same row kernel in the same order, so both paths
/// produce bit-identical arenas.
pub(crate) fn fill_answer_batch_serial<const MODULUS: u32, Q: QueryValues<MODULUS>>(
    matrix: &EncryptedMatrix<MODULUS>,
    queries: &[Q],
    arena: &mut [FieldElement<MODULUS>],
    n: usize,
    b: usize,
    s: usize,
    rows: usize,
) {
    for (flat_row, answer_row) in arena.chunks_mut(s).enumerate() {
        let query = &queries[flat_row / rows];
        let matrix_row_index = flat_row % rows;
        let matrix_row = &matrix.values()[matrix_row_index * n..(matrix_row_index + 1) * n];
        fill_answer_row(matrix_row, query.values(), b, answer_row);
    }
}

/// Decodes a matrix-vector product into caller-owned storage.
///
/// Accepts any borrowed or owned answer representation implementing
/// [`AnswerValues`], so a client can decode straight from a wire workspace
/// without copying it into [`AnswerMatrix`]. The [`Sync`] bound lets large
/// answers decode across rayon workers; every view and owned answer is a
/// plain shared-slice or `Vec` aggregate, so the bound costs nothing.
///
/// # Errors
///
/// Returns an error before output mutation if dimensions or public protocol
/// identifiers do not match.
pub fn decode_into<const MODULUS: u32, A: AnswerValues<MODULUS> + Sync>(
    answer: &A,
    key: &DecodingKey<MODULUS>,
    output: &mut [FieldElement<MODULUS>],
) -> Result<(), ProtocolError> {
    let (rows, s) = validate_decode(answer, key)?;
    check_len("decoding output", rows, output.len())?;
    fill_decoded(answer, key, output, s);
    Ok(())
}

fn validate_decode<const MODULUS: u32, A: AnswerValues<MODULUS>>(
    answer: &A,
    key: &DecodingKey<MODULUS>,
) -> Result<(usize, usize), ProtocolError> {
    let rows = answer.rows();
    let s = answer.blocks();
    check_len("decoding p'", s, key.p_prime.len())?;
    check_len("decoding r'", rows, key.r_prime.len())?;
    let expected = rows
        .checked_mul(s)
        .ok_or(ProtocolError::DimensionOverflow)?;
    check_len("answer matrix", expected, answer.values().len())?;
    if key.instance_id != answer.instance_id() {
        return Err(ProtocolError::InstanceMismatch {
            name: "decoding key",
            expected: answer.instance_id(),
            actual: key.instance_id,
        });
    }
    if key.query_id != answer.query_id() {
        return Err(ProtocolError::QueryMismatch {
            name: "decoding key",
            expected: answer.query_id(),
            actual: key.query_id,
        });
    }
    Ok((rows, s))
}

/// Writes each decoded product `M' p' - r'` into `output`.
///
/// Rows are independent dot products of length `s`, so large answers
/// distribute them across rayon workers under the shared dispatch policy
/// ([`crate::dispatch`]); the output is identical to the serial row loop.
/// Smaller answers stay serial.
fn fill_decoded<const MODULUS: u32, A: AnswerValues<MODULUS> + Sync>(
    answer: &A,
    key: &DecodingKey<MODULUS>,
    output: &mut [FieldElement<MODULUS>],
    s: usize,
) {
    let threads = rayon::current_num_threads();
    let rows = output.len();
    let work = rows.saturating_mul(s);
    if crate::dispatch::is_parallel_work(work, rows, threads) {
        output
            .par_iter_mut()
            .enumerate()
            .for_each(|(row, slot)| *slot = decode_row(answer, key, row, s));
    } else {
        for (row, slot) in output.iter_mut().enumerate() {
            *slot = decode_row(answer, key, row, s);
        }
    }
}

/// Evaluates one decoded product `M' p' - r'` for answer row `row`.
fn decode_row<const MODULUS: u32, A: AnswerValues<MODULUS>>(
    answer: &A,
    key: &DecodingKey<MODULUS>,
    row: usize,
    s: usize,
) -> FieldElement<MODULUS> {
    let zero = PrimeField::<MODULUS>::new().element_u32(0);
    let mut accumulator = zero;
    let answer_values = answer.values();
    for (block, &p) in key.p_prime.iter().enumerate() {
        accumulator += answer_values[row * s + block] * p;
    }
    accumulator - key.r_prime[row]
}
