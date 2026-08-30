//! The 1D-SLSN-based EMVP protocol.
//!
//! This crate implements the encrypted matrix-vector product (EMVP) protocol
//! of Fig. 1 in "Encrypted Matrix-Vector Products from Secret Dual Codes"
//! (IACR ePrint 2025/858) over the cyclic dual-code variant of that paper's
//! Section 3.1, where the parity-check polynomial ring uses `P = X^k - 1`. The
//! public random permutation from the protocol decorrelates the 1D-SLSN block
//! structure of the cyclic code. Encodings are masked by a stack of
//! trapdoored matrices from the [`trapdoor_matrices`] crate.
//!
//! These are experimental constructions, not production cryptographic
//! primitives. The security relies on the 1D-SLSN conjecture for the cyclic
//! dual code, which has no settled parameter set. Secret state is not
//! zeroized on drop. The arithmetic has not received a formal constant-time
//! audit.
//!
//! [`trapdoor_matrices`]: trapdoor_matrices

pub mod prf;

pub use prf::{purpose, Prf, PrfError};
