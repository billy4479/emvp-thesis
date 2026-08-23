//! Research implementations of three trapdoored-matrix candidates.
//!
//! The crate implements the irreducible-extension Ring-LPN construction
//! `R = [I | M_a] E`, the rectangular-Toeplitz product
//! `R = S_L Pi_L S Pi_R S_R`, and the RAA-style weighted product
//! `R = D^T Pi_4 S_3 Pi_3 S_2 Pi_2 S_1 Pi_1 D`.
//!
//! These are experimental constructions, not production cryptographic
//! primitives. The Ring-LPN construction relies on structured dual-LPN and has
//! no settled parameter set. The Toeplitz and RAA constructions rely on new,
//! speculative structured-cubic pseudorandomness assumptions. The rejected
//! quasi-cyclic product from an older EMVP implementation is neither the
//! Ring-LPN construction nor the rectangular-Toeplitz construction here.
//!
//! Evaluation allocates no memory when the caller reuses the construction's
//! scratch object. Sampling uses rejection and has variable RNG consumption.
//! Secret state is not zeroized on drop. The arithmetic has not received a
//! formal constant-time audit.

mod error;
mod matrix;
mod permutation;
mod raa;
mod ring_lpn;
mod toeplitz;

pub use error::TdmError;
pub use matrix::DenseMatrix;
pub use permutation::Permutation;
pub use raa::{RaaScratch, RaaWeightedProduct};
pub use ring_lpn::{IrreducibleRingLpn, RingLpnScratch, SparseMatrix};
pub use toeplitz::{ToeplitzFastProduct, ToeplitzMap, ToeplitzScratch};
