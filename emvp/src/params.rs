//! Parameter selection and validation for the 1D-SLSN protocol.
//!
//! A protocol instance splits each query codeword of length `n = 2k` into
//! `s = n / b` blocks of size `b`, scaling each block by a secret nonzero
//! scalar. Concrete parameters must raise the cost of the known attacks
//! above the target work `2^lambda`:
//!
//! - Algebraic annihilating-polynomial attack (ePrint 2025/858, Sections
//!   7.1.2 and 7.4.1). The public permutation is sampled once, so it defines a
//!   fixed partition across queries. For `d = ceil(k / (b - 1))`, the attack
//!   costs `b^(d - 1) * min(b, d)`, which must be at least `2^lambda`. The
//!   stronger `(k + 1)^d` bound requires a fresh partition for every query.
//! - Inclusion/exclusion attack (Section 7.3.2): `(n / b + 1) * k > n +
//!   lambda` keeps the space-unions attack infeasible.
//! - Structural constraints of the cyclic instantiation: records of length
//!   `ell <= k` are zero-padded, blocks must divide `n`, and `b >= 2` so
//!   that each block carries more than one element (`s = n / b` may be 1;
//!   `b = n` is the supported single-block shape).
//! - The protocol needs a noticeable padding slack (`n - k >= lambda^Omega(1)`,
//!   Fig. 1). With `n = 2k` we take `k >= ceil(lambda / 4)` as a concrete
//!   stand-in; the paper's own tables keep `k` of the same order as
//!   `lambda`.
//!
//! The field size itself is chosen at the type level by the caller; local
//! distinguishers (Section 7.3.3) shrink on large fields, so the caller is
//! responsible for picking a modulus consistent with the claimed security
//! level.
//!
//! This is experimental cryptography: no parameter set for 1D-SLSN is
//! settled, and the constraints here only raise the cost of the attacks
//! analyzed in the paper.

use std::fmt;

/// Largest security level [`pow_ge_pow2`] compares exactly.
pub const POW_MAX_LAMBDA: u32 = 4096;

/// Largest protocol security level supported by the 256-bit PRF key.
pub const PROTOCOL_MAX_LAMBDA: u32 = 256;

/// Ranks scanned by [`search`] before it gives up.
const SEARCH_RANK_BUDGET: usize = 10_000_000;

/// An exact integer comparison `(k + 1)^d >= 2^lambda` for `lambda <= 4096`.
///
/// Returns `true` exactly when `base^exp >= 2^lambda`. No floating point:
/// the power is accumulated in a small exact big integer with just enough
/// limbs to represent `2^lambda`, so every bit that could decide a boundary
/// case is present. Two rigorous early-exit guards keep the loop short when
/// the answer is far from the boundary: after `remaining` further factors of
/// `base` the value lies in `[2^(lo + (bitlen(base)-1) * remaining),
/// 2^(hi + bitlen(base) * remaining))`, where `[lo, hi)` is the current
/// exact bit-length interval, and either guard decides as soon as its bound
/// crosses the target. Overflowing the limb capacity proves the value is at
/// least `2^(lambda + 64)` and decides immediately.
///
/// `lambda > 4096` returns `false` by contract; [`EmvpParams::validate`]
/// rejects such security levels before calling this function.
///
/// Parameter validation uses the same exact comparison for its
/// algebraic-attack bound; rounded comparisons could silently move the
/// claimed security level.
#[must_use]
pub fn pow_ge_pow2(base: u64, exp: u64, lambda: u32) -> bool {
    scaled_pow_ge_pow2(base, exp, 1, lambda)
}

/// An exact integer comparison `factor * base^exp >= 2^lambda`.
fn scaled_pow_ge_pow2(base: u64, exp: u64, factor: u64, lambda: u32) -> bool {
    const MAX_LAMBDA: u32 = 4096;
    if lambda == 0 {
        return true;
    }
    if factor == 0 || lambda > MAX_LAMBDA {
        return false;
    }
    if base < 2 {
        if base == 0 && exp > 0 {
            return false;
        }
        return u64::from(u64::BITS - factor.leading_zeros()) > u64::from(lambda);
    }
    let lambda_u = u64::from(lambda);
    // Little-endian limbs with capacity for one bit beyond 2^lambda.
    let limbs = lambda as usize / 64 + 2;
    let mut value = vec![0_u64; limbs];
    value[0] = factor;
    let mut significant = 1_usize;
    let bitlen_base = u64::from(64 - base.leading_zeros());
    let base_lo = bitlen_base - 1; // base >= 2^base_lo
    let base_hi = bitlen_base; // base < 2^base_hi
    let mut consumed: u64 = 0;
    loop {
        let remaining = exp - consumed;
        if remaining == 0 {
            break;
        }
        let lo = exact_bit_length(&value[..significant]) - 1;
        // The final power is at least 2^(lo + base_lo * remaining).
        if lo.saturating_add(base_lo.saturating_mul(remaining)) >= lambda_u {
            return true;
        }
        // It is at most 2^(lo + 1 + base_hi * remaining) - 1.
        if (lo + 1).saturating_add(base_hi.saturating_mul(remaining)) <= lambda_u {
            return false;
        }
        // Multiply the exact value by base.
        let mut carry = 0_u128;
        for limb in &mut value[..significant] {
            let product = u128::from(*limb) * u128::from(base) + carry;
            *limb = product as u64;
            carry = product >> 64;
        }
        if carry > 0 {
            if significant == limbs {
                // The value reached 2^(64 * limbs) >= 2^(lambda + 64).
                return true;
            }
            value[significant] = carry as u64;
            significant += 1;
        }
        consumed += 1;
    }
    exact_bit_length(&value[..significant]) > lambda_u
}

/// The exact bit length of a little-endian nonzero limb slice.
const fn exact_bit_length(limbs: &[u64]) -> u64 {
    let top = limbs[limbs.len() - 1];
    let limb_bits = u64::BITS as u64;
    (limbs.len() as u64 - 1) * limb_bits + limb_bits - top.leading_zeros() as u64
}

/// A rejected or unfeasible parameter set.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum ParamsError {
    /// The record length was zero.
    ZeroEll,
    /// The security level exceeded the 256-bit protocol key strength.
    LambdaOutOfScope {
        /// The rejected security parameter.
        lambda: u32,
    },
    /// The record length exceeded the code rank.
    EllExceedsRank {
        /// The rejected record length.
        ell: usize,
        /// The code rank.
        k: usize,
    },
    /// The rank fell below the `ceil(lambda / 4)` padding-slack floor.
    RankBelowSecurityFloor {
        /// The rejected rank.
        k: usize,
        /// The requested security parameter.
        lambda: u32,
    },
    /// The block size was below two, leaving at most one block.
    BlockTooSmall {
        /// The rejected block size.
        b: usize,
    },
    /// The block size did not divide the codeword length.
    BlockDoesNotDivideLength {
        /// The rejected block size.
        b: usize,
        /// The codeword length `n`.
        n: usize,
    },
    /// The algebraic annihilating-polynomial attack stays below the target
    /// work.
    InsecureAgainstAlgebraicAttack {
        /// The code rank.
        k: usize,
        /// The block size.
        b: usize,
        /// The attack degree `ceil(k / (b - 1))`.
        d: u64,
        /// The requested security parameter.
        lambda: u32,
    },
    /// The inclusion/exclusion space-union attack stays feasible.
    InsecureAgainstInclusionExclusion {
        /// The codeword length `n`.
        n: usize,
        /// The code rank.
        k: usize,
        /// The block size.
        b: usize,
        /// The requested security parameter.
        lambda: u32,
    },
    /// The parameter search hit its iteration cap without success.
    SearchExhausted {
        /// The smallest rank the search was allowed to consider.
        floor_rank: usize,
    },
    /// Dimension arithmetic overflowed.
    DimensionOverflow,
}

impl fmt::Display for ParamsError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::ZeroEll => formatter.write_str("record length must be nonzero"),
            Self::LambdaOutOfScope { lambda } => write!(
                formatter,
                "security level lambda = {lambda} exceeds the 256-bit protocol key strength"
            ),
            Self::EllExceedsRank { ell, k } => {
                write!(formatter, "record length {ell} exceeds the rank {k}")
            }
            Self::RankBelowSecurityFloor { k, lambda } => write!(
                formatter,
                "rank {k} is below the ceil(lambda/4) floor for lambda = {lambda}"
            ),
            Self::BlockTooSmall { b } => {
                write!(formatter, "block size {b} must be at least 2")
            }
            Self::BlockDoesNotDivideLength { b, n } => {
                write!(
                    formatter,
                    "block size {b} does not divide the codeword length {n}"
                )
            }
            Self::InsecureAgainstAlgebraicAttack { k, b, d, lambda } => write!(
                formatter,
                "b^(d-1) * min(b,d) with k = {k}, d = {d}, b = {b} stays below 2^{lambda}"
            ),
            Self::InsecureAgainstInclusionExclusion { n, k, b, lambda } => write!(
                formatter,
                "inclusion/exclusion attack stays feasible for n = {n}, k = {k}, b = {b}, lambda = {lambda}"
            ),
            Self::SearchExhausted { floor_rank } => write!(
                formatter,
                "parameter search exhausted its rank budget from floor rank {floor_rank}"
            ),
            Self::DimensionOverflow => formatter.write_str("dimension arithmetic overflowed"),
        }
    }
}

impl std::error::Error for ParamsError {}

impl From<std::num::TryFromIntError> for ParamsError {
    fn from(_conversion: std::num::TryFromIntError) -> Self {
        Self::DimensionOverflow
    }
}

/// Concrete parameters of one protocol instance.
///
/// The field modulus is a compile-time constant of the arithmetic crates and
/// is deliberately not part of this structure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EmvpParams {
    /// The code rank; codewords have length `n = 2k`.
    pub k: usize,
    /// The record length; records are zero-padded up to `k`.
    pub ell: usize,
    /// The 1D-SLSN block size; it divides `n` and yields `s = n / b` blocks.
    pub b: usize,
    /// The security parameter; the attack constraints target work `2^lambda`.
    pub lambda: u32,
}

impl EmvpParams {
    /// The codeword length `n = 2k`.
    ///
    /// # Errors
    ///
    /// Returns [`ParamsError::DimensionOverflow`] if `2k` does not fit in
    /// `usize`.
    pub const fn n(&self) -> Result<usize, ParamsError> {
        match self.k.checked_mul(2) {
            Some(n) => Ok(n),
            None => Err(ParamsError::DimensionOverflow),
        }
    }

    /// The number of blocks `s = n / b`.
    ///
    /// # Errors
    ///
    /// Returns an error if the dimensions overflow, `b < 2`, or `b` does not
    /// divide `n`.
    pub fn blocks(&self) -> Result<usize, ParamsError> {
        if self.b < 2 {
            return Err(ParamsError::BlockTooSmall { b: self.b });
        }
        let n = self.n()?;
        if !n.is_multiple_of(self.b) {
            return Err(ParamsError::BlockDoesNotDivideLength { b: self.b, n });
        }
        Ok(n / self.b)
    }

    /// The block size `b`.
    #[must_use]
    pub const fn block_size(&self) -> usize {
        self.b
    }

    /// Checks every constraint of the module documentation.
    ///
    /// # Errors
    ///
    /// Returns the first violated constraint as a [`ParamsError`].
    pub fn validate(&self) -> Result<(), ParamsError> {
        let Self { k, b, lambda, .. } = *self;
        if lambda > PROTOCOL_MAX_LAMBDA {
            return Err(ParamsError::LambdaOutOfScope { lambda });
        }
        self.validate_dimensions()?;
        let floor = Self::rank_floor(lambda);
        if k < floor {
            return Err(ParamsError::RankBelowSecurityFloor { k, lambda });
        }
        let n = self.n()?;
        // d = ceil(k / (b - 1)); b >= 2 so the divisor is positive.
        let b_minus_one = u64::try_from(b - 1).map_err(ParamsError::from)?;
        let k_u64 = u64::try_from(k).map_err(ParamsError::from)?;
        let d = k_u64
            .checked_add(b_minus_one - 1)
            .ok_or(ParamsError::DimensionOverflow)?
            / b_minus_one;
        let b_u64 = u64::try_from(b).map_err(ParamsError::from)?;
        if !scaled_pow_ge_pow2(b_u64, d - 1, u64::min(b_u64, d), lambda) {
            return Err(ParamsError::InsecureAgainstAlgebraicAttack { k, b, d, lambda });
        }
        // (n / b + 1) * k > n + lambda, computed without truncation in u128.
        let blocks = u128::try_from(n / b).map_err(ParamsError::from)?;
        let k_u128 = u128::try_from(k).map_err(ParamsError::from)?;
        let n_u128 = u128::try_from(n).map_err(ParamsError::from)?;
        let lambda_u128 = u128::from(lambda);
        let left = blocks
            .checked_mul(k_u128)
            .and_then(|product| product.checked_add(k_u128))
            .ok_or(ParamsError::DimensionOverflow)?;
        let right = n_u128
            .checked_add(lambda_u128)
            .ok_or(ParamsError::DimensionOverflow)?;
        if left <= right {
            return Err(ParamsError::InsecureAgainstInclusionExclusion { n, k, b, lambda });
        }
        Ok(())
    }

    /// Checks the dimensions required for protocol correctness without
    /// claiming a concrete security level.
    ///
    /// This is intended for deliberately tiny research and test instances.
    /// Production callers should use [`Self::validate`].
    ///
    /// # Errors
    ///
    /// Returns the first malformed dimension as a [`ParamsError`].
    pub fn validate_dimensions(&self) -> Result<(), ParamsError> {
        if self.ell == 0 {
            return Err(ParamsError::ZeroEll);
        }
        if self.ell > self.k {
            return Err(ParamsError::EllExceedsRank {
                ell: self.ell,
                k: self.k,
            });
        }
        if self.b < 2 {
            return Err(ParamsError::BlockTooSmall { b: self.b });
        }
        let n = self.n()?;
        if !n.is_multiple_of(self.b) {
            return Err(ParamsError::BlockDoesNotDivideLength { b: self.b, n });
        }
        Ok(())
    }

    /// The smallest rank acceptable for a security parameter:
    /// `ceil(lambda / 4)`.
    ///
    /// The narrowing `lambda as usize` truncates only on platforms with a
    /// 16-bit `usize`; every caller first rejects `lambda` beyond
    /// [`PROTOCOL_MAX_LAMBDA`] (256), which fits any `usize` width.
    #[must_use]
    pub const fn rank_floor(lambda: u32) -> usize {
        (lambda as usize).div_ceil(4)
    }
}

/// Searches for concrete parameters at a given record length and security
/// level.
///
/// The search scans ranks `k` upward from `max(ell, ceil(lambda / 4))` and,
/// for each rank, tries the divisors `b >= 2` of `n = 2k` in descending
/// order, returning the first feasible `(k, b)`: the smallest rank wins, and
/// within a rank the largest block size wins, which maximizes the download
/// compression ratio `b / f` for the server overhead `f = n / ell`. Cyclic
/// ranks whose double is a power of two admit no divisor beyond small powers
/// of two, so the scan naturally settles on ranks with a smooth `2k`.
///
/// # Errors
///
/// Returns [`ParamsError::ZeroEll`] for a zero record length,
/// [`ParamsError::LambdaOutOfScope`] for security levels beyond
/// [`PROTOCOL_MAX_LAMBDA`], and [`ParamsError::SearchExhausted`] after scanning
/// the rank budget without success.
pub fn search(ell: usize, lambda: u32) -> Result<EmvpParams, ParamsError> {
    if ell == 0 {
        return Err(ParamsError::ZeroEll);
    }
    if lambda > PROTOCOL_MAX_LAMBDA {
        return Err(ParamsError::LambdaOutOfScope { lambda });
    }
    let floor = usize::max(ell, EmvpParams::rank_floor(lambda));
    let last = floor
        .checked_add(SEARCH_RANK_BUDGET)
        .ok_or(ParamsError::DimensionOverflow)?;
    for k in floor..last {
        let Some(n) = k.checked_mul(2) else {
            return Err(ParamsError::DimensionOverflow);
        };
        // Divisors of n in descending order, tried largest first.
        let mut divisor: usize = 1;
        let mut divisors = Vec::new();
        while divisor
            .checked_mul(divisor)
            .is_some_and(|square| square <= n)
        {
            if n % divisor == 0 {
                divisors.push(divisor);
                if divisor != n / divisor {
                    divisors.push(n / divisor);
                }
            }
            divisor += 1;
        }
        divisors.sort_unstable();
        for &b in divisors.iter().rev().filter(|divisor| **divisor >= 2) {
            let params = EmvpParams { k, ell, b, lambda };
            if params.validate().is_ok() {
                return Ok(params);
            }
        }
    }
    Err(ParamsError::SearchExhausted { floor_rank: floor })
}
