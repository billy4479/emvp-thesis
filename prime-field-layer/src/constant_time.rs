//! Small branchless reduction primitives shared by scalar hot paths.

/// Subtracts `modulus` when `value >= modulus`.
///
/// Callers ensure `value < 2 * modulus`. The selection is expressed with
/// wrapping integer arithmetic rather than a comparison or `select`, so the
/// emitted instruction sequence is data-independent by construction: there is
/// no branch or select in the IR that LLVM could turn into
/// coefficient-dependent control flow, and auto-vectorized loops can lower the
/// whole expression with vector compares and masks. This is a code-generation
/// safeguard, not a claim of a formally verified side-channel implementation.
#[inline(always)]
#[expect(
    clippy::redundant_pub_crate,
    reason = "pub(super) keeps this private if the module visibility changes"
)]
pub(super) const fn reduce_once_u64(value: u64, modulus: u64) -> u64 {
    let (reduced, borrow) = value.overflowing_sub(modulus);
    reduced.wrapping_add(modulus & 0u64.wrapping_sub(borrow as u64))
}
