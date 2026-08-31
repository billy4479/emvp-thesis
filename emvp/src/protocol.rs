//! The 1D-SLSN-based EMVP protocol (ePrint 2025/858, Fig. 1), cyclic variant.
//!
//! The client holds a matrix `M in F^(m x ell)` and a short secret key. The
//! online phase produces compact shares of `M q` for any query vector `q`:
//! the server returns an `m x s` answer matrix while the client keeps an
//! `s`-vector and an `m`-vector such that `M q = M' p' - r'`.
//!
//! The key derivation expands the short key into the long-term secrets via
//! the [`Prf`]: the cyclic dual-code multiplier `g`, the public permutation
//! `Pi` of length `n = 2k`, and the stack of trapdoored mask blocks. Query
//! randomness (the codeword and the nonzero block scalars) is derived under a
//! monotonic query index. Callers persist the next index when reconstructing
//! state after a restart.
//!
//! The data flow, matching the conventions of [`crate::code`] and
//! [`crate::mask`]:
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
use trapdoor_matrices::{DenseMatrix, Permutation};

use crate::code::{CodeError, CyclicCodeScratch, CyclicDualCode};
use crate::mask::{MaskError, RowStackMask, TdmMask};
use crate::params::{EmvpParams, ParamsError};
use crate::prf::{Prf, PrfError, purpose};

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
    Mask(MaskError),
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

impl From<MaskError> for ProtocolError {
    fn from(error: MaskError) -> Self {
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

impl From<trapdoor_matrices::TdmError> for ProtocolError {
    fn from(error: trapdoor_matrices::TdmError) -> Self {
        Self::Mask(error.into())
    }
}

const fn check_len(
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

fn fill_answer_row<const MODULUS: u32>(
    matrix_row: &[FieldElement<MODULUS>],
    query: &[FieldElement<MODULUS>],
    block_len: usize,
    answer_row: &mut [FieldElement<MODULUS>],
) {
    let zero = PrimeField::<MODULUS>::new().element_u32(0);
    for (block, slot) in answer_row.iter_mut().enumerate() {
        let start = block * block_len;
        let mut accumulator = zero;
        for offset in 0..block_len {
            accumulator += matrix_row[start + offset] * query[start + offset];
        }
        *slot = accumulator;
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
    /// [`DerivedState::instance_nonce`] together with the next query index.
    ///
    /// `rows` is the number of matrix rows to support; it determines how
    /// many square mask blocks are stacked. `build_block` constructs one
    /// mask block from the block's PRF stream and index, which is how the
    /// caller picks a trapdoored-matrix construction and its parameters.
    ///
    /// # Errors
    ///
    /// Returns errors from the PRF, the code and permutation sampling, the
    /// mask construction, or the closure itself.
    pub fn derive<M, F, R>(
        self,
        rows: usize,
        rng: &mut R,
        mut build_block: F,
    ) -> Result<DerivedState<MODULUS, M>, ProtocolError>
    where
        M: TdmMask<MODULUS>,
        F: FnMut(&mut ChaCha20Rng, usize) -> Result<M, ProtocolError>,
        R: CryptoRng + ?Sized,
    {
        let mut nonce = [0_u8; 16];
        rng.fill_bytes(&mut nonce);
        self.derive_inner(u128::from_le_bytes(nonce), 0, rows, false, &mut build_block)
    }

    /// Restores query state for an already encrypted matrix.
    ///
    /// `instance_nonce` and `next_query_index` must be the durably persisted
    /// values from the original state. Restored state cannot encrypt another
    /// matrix, which prevents deterministic mask reuse after restart.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::derive`].
    pub fn restore<M, F>(
        self,
        instance_nonce: u128,
        next_query_index: u64,
        rows: usize,
        mut build_block: F,
    ) -> Result<DerivedState<MODULUS, M>, ProtocolError>
    where
        M: TdmMask<MODULUS>,
        F: FnMut(&mut ChaCha20Rng, usize) -> Result<M, ProtocolError>,
    {
        self.derive_inner(
            instance_nonce,
            next_query_index,
            rows,
            true,
            &mut build_block,
        )
    }

    fn derive_inner<M, F>(
        self,
        instance_nonce: u128,
        next_query_index: u64,
        rows: usize,
        encrypted: bool,
        build_block: &mut F,
    ) -> Result<DerivedState<MODULUS, M>, ProtocolError>
    where
        M: TdmMask<MODULUS>,
        F: FnMut(&mut ChaCha20Rng, usize) -> Result<M, ProtocolError>,
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

        let prf = self.prf.derive_context(instance_nonce);
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
        let mut blocks = Vec::with_capacity(block_count);
        for block_index in 0..block_count {
            let mut stream = prf.stream(purpose::TDM, block_index as u64)?;
            blocks.push(build_block(&mut stream, block_index)?);
        }
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
}

/// Encrypts a row-major `rows x ell` matrix for the server.
///
/// The mask is materialized into the `rows x n` ciphertext allocation and
/// overwritten row by row after encoding and permutation. This is the offline
/// phase of the protocol. A derived state can encrypt exactly one matrix.
///
/// # Errors
///
/// Returns an error before any output is produced if the state already
/// encrypted a matrix, the matrix length differs from `rows * ell`, or a code
/// or mask operation fails.
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
    let zero = PrimeField::<MODULUS>::new().element_u32(0);
    let mut row = vec![zero; n];
    for row_index in 0..rows {
        state.code.dual_encode_row(
            &matrix[row_index * ell..(row_index + 1) * ell],
            &mut row,
            &mut state.online_scratch.code,
        )?;
        let output_row = &mut encoded[row_index * n..(row_index + 1) * n];
        for (slot, &masked) in row.iter_mut().zip(output_row.iter()) {
            *slot += masked;
        }
        state.permutation.apply(&row, output_row)?;
    }
    let encrypted = EncryptedMatrix {
        matrix: DenseMatrix::new(rows, n, encoded)?,
        instance_id: state.instance_id,
    };
    state.encrypted = true;
    Ok(encrypted)
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

    // Hide each block behind a fresh nonzero scalar.
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

/// Answers an encrypted query against an encrypted matrix.
///
/// This is the server's online work: `s` column-block matrix-vector
/// products, `m * n` field multiplications in total.
///
/// # Errors
///
/// Returns an error before producing output if parameters are malformed, the
/// matrix columns differ from `n`, or the query length differs from `n`.
pub fn answer<const MODULUS: u32>(
    params: &EmvpParams,
    matrix: &EncryptedMatrix<MODULUS>,
    query: &EncryptedQuery<MODULUS>,
) -> Result<AnswerMatrix<MODULUS>, ProtocolError> {
    let (n, b, s, rows) = validate_answer(params, matrix, query)?;
    let zero = PrimeField::<MODULUS>::new().element_u32(0);
    let mut values = vec![
        zero;
        rows.checked_mul(s)
            .ok_or(ProtocolError::DimensionOverflow)?
    ];
    fill_answer(matrix, query, &mut values, n, b, s, rows);
    Ok(AnswerMatrix {
        values,
        rows,
        blocks: s,
        instance_id: matrix.instance_id,
        query_id: query.query_id,
    })
}

/// Answers an encrypted query into caller-owned row-major storage.
///
/// `output` must contain `matrix.rows() * params.blocks()` elements. This
/// variant allows servers to reuse allocations across queries.
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

fn validate_answer<const MODULUS: u32>(
    params: &EmvpParams,
    matrix: &EncryptedMatrix<MODULUS>,
    query: &EncryptedQuery<MODULUS>,
) -> Result<(usize, usize, usize, usize), ProtocolError> {
    params.validate_dimensions()?;
    let n = params.n()?;
    let b = params.block_size();
    let s = params.blocks()?;
    let rows = matrix.rows();
    check_len("encrypted matrix columns", n, matrix.columns())?;
    check_len("encrypted query", n, query.values.len())?;
    if query.instance_id != matrix.instance_id {
        return Err(ProtocolError::InstanceMismatch {
            name: "encrypted query",
            expected: matrix.instance_id,
            actual: query.instance_id,
        });
    }
    Ok((n, b, s, rows))
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
    const MIN_PARALLEL_MULTIPLICATIONS: usize = 32 * 1024;
    let threads = rayon::current_num_threads();
    let work = rows.saturating_mul(n);
    if threads > 1 && rows >= threads.saturating_mul(2) && work >= MIN_PARALLEL_MULTIPLICATIONS {
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

/// Decodes the client share of the matrix-vector product.
///
/// Returns `a = M' p' - r'`, which equals `M q` for a matching protocol run.
///
/// # Errors
///
/// Returns an error if the decoding-key lengths do not match the answer
/// dimensions.
pub fn decode<const MODULUS: u32>(
    answer: &AnswerMatrix<MODULUS>,
    key: &DecodingKey<MODULUS>,
) -> Result<Vec<FieldElement<MODULUS>>, ProtocolError> {
    let (rows, s) = validate_decode(answer, key)?;
    let zero = PrimeField::<MODULUS>::new().element_u32(0);
    let mut output = vec![zero; rows];
    fill_decoded(answer, key, &mut output, s);
    Ok(output)
}

/// Decodes a matrix-vector product into caller-owned storage.
///
/// # Errors
///
/// Returns an error before output mutation if dimensions or public protocol
/// identifiers do not match.
pub fn decode_into<const MODULUS: u32>(
    answer: &AnswerMatrix<MODULUS>,
    key: &DecodingKey<MODULUS>,
    output: &mut [FieldElement<MODULUS>],
) -> Result<(), ProtocolError> {
    let (rows, s) = validate_decode(answer, key)?;
    check_len("decoding output", rows, output.len())?;
    fill_decoded(answer, key, output, s);
    Ok(())
}

fn validate_decode<const MODULUS: u32>(
    answer: &AnswerMatrix<MODULUS>,
    key: &DecodingKey<MODULUS>,
) -> Result<(usize, usize), ProtocolError> {
    let rows = answer.rows;
    let s = answer.blocks;
    check_len("decoding p'", s, key.p_prime.len())?;
    check_len("decoding r'", rows, key.r_prime.len())?;
    let expected = rows
        .checked_mul(s)
        .ok_or(ProtocolError::DimensionOverflow)?;
    check_len("answer matrix", expected, answer.values.len())?;
    if key.instance_id != answer.instance_id {
        return Err(ProtocolError::InstanceMismatch {
            name: "decoding key",
            expected: answer.instance_id,
            actual: key.instance_id,
        });
    }
    if key.query_id != answer.query_id {
        return Err(ProtocolError::QueryMismatch {
            name: "decoding key",
            expected: answer.query_id,
            actual: key.query_id,
        });
    }
    Ok((rows, s))
}

fn fill_decoded<const MODULUS: u32>(
    answer: &AnswerMatrix<MODULUS>,
    key: &DecodingKey<MODULUS>,
    output: &mut [FieldElement<MODULUS>],
    s: usize,
) {
    let zero = PrimeField::<MODULUS>::new().element_u32(0);
    for (row, slot) in output.iter_mut().enumerate() {
        let mut accumulator = zero;
        for (block, &p) in key.p_prime.iter().enumerate() {
            accumulator += answer.values[row * s + block] * p;
        }
        *slot = accumulator - key.r_prime[row];
    }
}
