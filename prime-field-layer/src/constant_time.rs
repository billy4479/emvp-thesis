//! Small branchless reduction primitives shared by scalar hot paths.

/// Subtracts `modulus` when `value >= modulus`.
///
/// Callers ensure `value < 2 * modulus`. Inline assembly pins the operation to
/// `sub`/`cmovb` on x86-64 and `subs`/`csel` on aarch64, preventing LLVM from
/// replacing the select with a coefficient-dependent branch on those targets.
/// Other targets use the same mask-based operation in portable Rust. This is a
/// code-generation safeguard, not a claim of a formally verified side-channel
/// implementation.
#[inline(always)]
#[expect(
    clippy::redundant_pub_crate,
    reason = "pub(super) keeps this private if the module visibility changes"
)]
pub(super) fn reduce_once_u64(value: u64, modulus: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        let reduced: u64;
        // SAFETY: this uses only integer registers, does not access memory, and
        // returns either `value - modulus` or the original `value`.
        unsafe {
            std::arch::asm!(
                "mov {reduced}, {value}",
                "sub {reduced}, {modulus}",
                "cmovb {reduced}, {value}",
                value = in(reg) value,
                modulus = in(reg) modulus,
                reduced = out(reg) reduced,
                options(pure, nomem, nostack),
            );
        }
        reduced
    }

    #[cfg(target_arch = "aarch64")]
    {
        let reduced: u64;
        // SAFETY: this uses only integer registers and selects the original
        // value when `subs` reports an unsigned borrow.
        unsafe {
            std::arch::asm!(
                "subs {reduced}, {value}, {modulus}",
                "csel {reduced}, {value}, {reduced}, lo",
                value = in(reg) value,
                modulus = in(reg) modulus,
                reduced = out(reg) reduced,
                options(pure, nomem, nostack),
            );
        }
        reduced
    }

    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        let (reduced, borrow) = value.overflowing_sub(modulus);
        let keep_original = 0u64.wrapping_sub(borrow as u64);
        (reduced & !keep_original) | (value & keep_original)
    }
}

/// Returns `(lhs + rhs) >> 32`, retaining a carry out of bit 63 as bit 32.
#[inline(always)]
#[expect(
    clippy::redundant_pub_crate,
    reason = "pub(super) keeps this private if the module visibility changes"
)]
pub(super) fn add_with_carry_shr_32(lhs: u64, rhs: u64) -> u64 {
    #[cfg(target_arch = "x86_64")]
    {
        let mut sum = lhs;
        let carry_mask: u64;
        // SAFETY: the assembly performs an integer add and materializes its
        // carry flag as either zero or all ones; it has no memory effects.
        unsafe {
            std::arch::asm!(
                "add {sum}, {rhs}",
                "sbb {carry_mask}, {carry_mask}",
                sum = inout(reg) sum,
                rhs = in(reg) rhs,
                carry_mask = lateout(reg) carry_mask,
                options(pure, nomem, nostack),
            );
        }
        (sum >> 32) | (carry_mask & (1u64 << 32))
    }

    #[cfg(not(target_arch = "x86_64"))]
    {
        let (sum, carry) = lhs.overflowing_add(rhs);
        (sum >> 32) | ((carry as u64) << 32)
    }
}
