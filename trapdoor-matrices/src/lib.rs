//! Research implementations of three trapdoored-matrix candidates.
//!
//! The crate implements the irreducible-extension Ring-LPN construction
//! `R = [I | M_a] E`, the rectangular-Toeplitz product
//! `R = S_L Pi_L S Pi_R S_R`, and the RAA-style weighted product
//! `R = D^T Pi_4 S_3 Pi_3 S_2 Pi_2 S_1 Pi_1 D`.
//!
//! The [`TdmMask`] trait abstracts one square construction as an EMVP mask
//! block, and [`RowStackMask`] stacks square blocks into the rectangular
//! masks the EMVP protocol hides its encodings behind.
//!
//! These are experimental constructions, not production cryptographic
//! primitives. The Ring-LPN construction relies on structured dual-LPN and has
//! no settled parameter set. [`IrreducibleRingLpn::sample`] therefore sizes
//! every request against the best-known-attack estimates of [`assess`] and
//! fails closed: a [`SecurityLevel::Broken`] assessment aborts sampling with
//! [`TdmError::InsecureParameters`], while sound and marginal requests
//! construct the instance and return the outcome as warnings.
//! The Toeplitz and RAA constructions rely on new, speculative
//! structured-cubic pseudorandomness assumptions. The quasi-cyclic product
//! of the original EMVP proposal was rejected in favor of the constructions
//! implemented here and does not appear in this crate.
//!
//! Evaluation allocates no memory when the caller reuses the construction's
//! scratch object. Sampling uses rejection and has variable RNG consumption.
//! Secret state is not zeroized on drop. The arithmetic has not received a
//! formal constant-time audit.
//!
//! The deterministic fixed-weight fixture builders used by tests and
//! benchmarks live in the `testing` module, which is only compiled with the
//! `test-utils` feature and must not be used for anything security-sensitive.

mod error;
mod mask;
mod matrix;
mod parameters;
mod permutation;
mod raa;
mod ring_lpn;
#[cfg(feature = "test-utils")]
pub mod testing;
mod toeplitz;

pub use error::TdmError;
pub use mask::{RowStackMask, TdmMask};
pub use matrix::DenseMatrix;
pub use parameters::{
    ParameterWarning, SecurityAssessment, SecurityLevel, SecurityWarningKind, TARGET_SECURITY_BITS,
    assess, automatic_ring_modulus,
};
pub use permutation::Permutation;
pub use raa::{RaaScratch, RaaWeightedProduct};
pub use ring_lpn::{IrreducibleRingLpn, RingLpnScratch, SampledIrreducibleRingLpn, SparseMatrix};
pub use toeplitz::{ToeplitzFastProduct, ToeplitzMap, ToeplitzScratch};
