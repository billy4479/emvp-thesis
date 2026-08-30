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
//! randomness (the codeword and the nonzero block scalars) is fresh per
//! query and comes from the caller's random number generator.
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
//!   `M' p' = M_hat q_pi = M D q_tilde + R q_pi = M q_pad + r'`.
//!
//! The permutation decorrelates the cyclic code structure from the fixed
//! block grid (paper Section 3.1, the `P = X^k - 1` variant); it is public
//! and derived deterministically from the key.
//!
//! This is experimental cryptography: 1D-SLSN has no settled security
//! parameters, secret state is not zeroized, per-query timing and memory
//! access are not constant-time, and [`encrypt`] materializes the mask.

use std::fmt;

use prime_field_layer::{FieldElement, FieldError, PrimeField};
use rand_chacha::ChaCha20Rng;
use rand_core::CryptoRng;
use trapdoor_matrices::{DenseMatrix, Permutation};

use crate::code::{CodeError, CyclicCodeScratch, CyclicDualCode};
use crate::mask::{MaskError, RowStackMask, TdmMask};
use crate::params::{EmvpParams, ParamsError};
use crate::prf::{Prf, PrfError, purpose};

/// A rejected protocol operation.
#[derive(Clone, Debug, Eq, PartialEq)]
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
                "derived state has already encrypted a matrix; derive a fresh key state",
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

    /// Expands the short key into the long-term secrets.
    ///
    /// This consumes the matrix key so one key cannot accidentally derive two
    /// states with the same deterministic mask.
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
    pub fn derive<M, F>(
        self,
        rows: usize,
        mut build_block: F,
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

        let mut code_stream = self.prf.stream(purpose::CODE_MULTIPLIER, 0)?;
        let code = CyclicDualCode::sample(k, &mut code_stream)?;

        let mut permutation_stream = self.prf.stream(purpose::CODE_PERMUTATION, 0)?;
        let permutation = Permutation::sample(n, &mut permutation_stream)?;

        let block_count = rows.div_ceil(n);
        let mut blocks = Vec::with_capacity(block_count);
        for block_index in 0..block_count {
            let mut stream = self.prf.stream(purpose::TDM, block_index as u64)?;
            blocks.push(build_block(&mut stream, block_index)?);
        }
        let mask = RowStackMask::new(blocks, rows)?;
        check_len("mask columns", n, mask.dims().1)?;
        let code_scratch = code.scratch();
        let mask_scratch = mask.scratch();

        Ok(DerivedState {
            params: self.params,
            code,
            permutation,
            mask,
            code_scratch,
            mask_scratch,
            encrypted: false,
        })
    }
}

/// The expanded long-term secrets for one matrix encryption and many queries.
///
/// The code multiplier, its cached transform, the permutation, and the mask
/// trapdoors are secret; none of them are zeroized on drop.
pub struct DerivedState<const MODULUS: u32, M: TdmMask<MODULUS>> {
    params: EmvpParams,
    code: CyclicDualCode<MODULUS>,
    permutation: Permutation,
    mask: RowStackMask<M, MODULUS>,
    code_scratch: CyclicCodeScratch<MODULUS>,
    mask_scratch: <RowStackMask<M, MODULUS> as TdmMask<MODULUS>>::Scratch,
    encrypted: bool,
}

impl<const MODULUS: u32, M: TdmMask<MODULUS>> DerivedState<MODULUS, M> {
    /// Returns the protocol parameters.
    #[must_use]
    pub const fn params(&self) -> EmvpParams {
        self.params
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

    /// Rebuilds an encrypted matrix from its parts.
    ///
    /// # Errors
    ///
    /// Returns an error if the value length does not match the dimensions.
    pub fn from_parts(
        rows: usize,
        columns: usize,
        values: Vec<FieldElement<MODULUS>>,
    ) -> Result<Self, ProtocolError> {
        Ok(Self {
            matrix: DenseMatrix::new(rows, columns, values)?,
        })
    }
}

/// The encrypted query `q_hat` sent to the server.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct EncryptedQuery<const MODULUS: u32> {
    values: Vec<FieldElement<MODULUS>>,
}

impl<const MODULUS: u32> EncryptedQuery<MODULUS> {
    /// Returns the encrypted query coordinates (length `n`).
    #[must_use]
    pub fn values(&self) -> &[FieldElement<MODULUS>] {
        &self.values
    }

    /// Rebuilds an encrypted query from its coordinates.
    #[must_use]
    pub const fn from_parts(values: Vec<FieldElement<MODULUS>>) -> Self {
        Self { values }
    }
}

/// The client's decoding information `q' = (p', r')`.
#[derive(Clone, Eq, PartialEq)]
pub struct DecodingKey<const MODULUS: u32> {
    p_prime: Vec<FieldElement<MODULUS>>,
    r_prime: Vec<FieldElement<MODULUS>>,
}

impl<const MODULUS: u32> fmt::Debug for DecodingKey<MODULUS> {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DecodingKey")
            .field("blocks", &self.p_prime.len())
            .field("rows", &self.r_prime.len())
            .finish_non_exhaustive()
    }
}

impl<const MODULUS: u32> DecodingKey<MODULUS> {
    /// Rebuilds a decoding key from its parts.
    #[must_use]
    pub const fn from_parts(
        p_prime: Vec<FieldElement<MODULUS>>,
        r_prime: Vec<FieldElement<MODULUS>>,
    ) -> Self {
        Self { p_prime, r_prime }
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
}

/// The server's answer `M' in F^(m x s)`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AnswerMatrix<const MODULUS: u32> {
    values: Vec<FieldElement<MODULUS>>,
    rows: usize,
    blocks: usize,
}

impl<const MODULUS: u32> AnswerMatrix<MODULUS> {
    /// Rebuilds an answer matrix from its parts.
    #[must_use]
    pub const fn from_parts(
        values: Vec<FieldElement<MODULUS>>,
        rows: usize,
        blocks: usize,
    ) -> Self {
        Self {
            values,
            rows,
            blocks,
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
            &mut state.code_scratch,
        )?;
        let output_row = &mut encoded[row_index * n..(row_index + 1) * n];
        for (slot, &masked) in row.iter_mut().zip(output_row.iter()) {
            *slot += masked;
        }
        state.permutation.apply(&row, output_row)?;
    }
    let encrypted = EncryptedMatrix {
        matrix: DenseMatrix::new(rows, n, encoded)?,
    };
    state.encrypted = true;
    Ok(encrypted)
}

/// Generates an encrypted query for `q` plus the client's decoding key.
///
/// The codeword and the nonzero block scalars are sampled from `rng`, so
/// every query uses fresh randomness. The mask share `r'` is evaluated
/// through the trapdoor without allocating.
///
/// # Errors
///
/// Returns an error before mutation if `q` does not have the record length
/// `ell`, or if code, mask, or field operations fail.
pub fn query<const MODULUS: u32, M: TdmMask<MODULUS>, R: CryptoRng + ?Sized>(
    state: &mut DerivedState<MODULUS, M>,
    q: &[FieldElement<MODULUS>],
    rng: &mut R,
) -> Result<(EncryptedQuery<MODULUS>, DecodingKey<MODULUS>), ProtocolError> {
    let code_dim = state.params.k;
    let ell = state.params.ell;
    let width = state.params.n()?;
    let block_len = state.params.block_size();
    let block_count = state.params.blocks()?;
    let rows = state.mask.dims().0;
    check_len("query vector", ell, q.len())?;

    let zero = PrimeField::<MODULUS>::new().element_u32(0);
    let mut q_tilde = vec![zero; width];
    state
        .code
        .sample_codeword(rng, &mut q_tilde, &mut state.code_scratch)?;
    // Add the padded query in the systematic (second) half.
    for (slot, &coefficient) in q_tilde[code_dim..].iter_mut().zip(q) {
        *slot += coefficient;
    }

    // Mask share through the trapdoor, on the unpermuted query: the server
    // answer unwinds to X q_tilde = M D q_tilde + R q_tilde = M q_pad + R q_tilde.
    let mut r_prime = vec![zero; rows];
    state
        .mask
        .apply(&q_tilde, &mut r_prime, &mut state.mask_scratch)?;

    // Gather through the permutation. The encrypted matrix columns are the
    // same gather of the encoded columns, M_hat[i] = X[Pi[i]], so the answer
    // matches X q_tilde exactly when the query is gathered too.
    let mut q_pi = vec![zero; width];
    state.permutation.apply(&q_tilde, &mut q_pi)?;

    // Hide each block behind a fresh nonzero scalar.
    let field = PrimeField::<MODULUS>::new();
    let mut q_hat = q_pi;
    let mut p_prime = vec![zero; block_count];
    for block in 0..block_count {
        let alpha = field.sample_uniform_nonzero(rng);
        q_tilde[block] = alpha;
        for slot in &mut q_hat[block * block_len..(block + 1) * block_len] {
            *slot *= alpha;
        }
    }
    field.batch_inv_elements(&q_tilde[..block_count], &mut p_prime)?;

    Ok((
        EncryptedQuery { values: q_hat },
        DecodingKey { p_prime, r_prime },
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
    params.validate_dimensions()?;
    let n = params.n()?;
    let b = params.block_size();
    let s = params.blocks()?;
    let rows = matrix.rows();
    check_len("encrypted matrix columns", n, matrix.columns())?;
    check_len("encrypted query", n, query.values.len())?;

    let zero = PrimeField::<MODULUS>::new().element_u32(0);
    let mut values = vec![
        zero;
        rows.checked_mul(s)
            .ok_or(ProtocolError::DimensionOverflow)?
    ];
    for row in 0..rows {
        let matrix_row = &matrix.values()[row * n..(row + 1) * n];
        let answer_row = &mut values[row * s..(row + 1) * s];
        for (block, slot) in answer_row.iter_mut().enumerate() {
            let start = block * b;
            let mut accumulator = zero;
            for offset in 0..b {
                accumulator += matrix_row[start + offset] * query.values[start + offset];
            }
            *slot = accumulator;
        }
    }
    Ok(AnswerMatrix {
        values,
        rows,
        blocks: s,
    })
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
    let rows = answer.rows;
    let s = answer.blocks;
    check_len("decoding p'", s, key.p_prime.len())?;
    check_len("decoding r'", rows, key.r_prime.len())?;
    let expected = rows
        .checked_mul(s)
        .ok_or(ProtocolError::DimensionOverflow)?;
    check_len("answer matrix", expected, answer.values.len())?;

    let zero = PrimeField::<MODULUS>::new().element_u32(0);
    let mut output = vec![zero; rows];
    for (row, slot) in output.iter_mut().enumerate() {
        let mut accumulator = zero;
        for (block, &p) in key.p_prime.iter().enumerate() {
            accumulator += answer.values[row * s + block] * p;
        }
        *slot = accumulator - key.r_prime[row];
    }
    Ok(output)
}
