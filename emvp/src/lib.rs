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
//! All deterministic instance state is bound to a public reconstruction
//! context: a caller-chosen [`MaskContextId`] identifying the mask suite
//! and its configuration, the [`EmvpParams`] fields, the matrix row count,
//! and the instance nonce. The context identifier, the instance nonce, and
//! the next query index must be persisted together for
//! [`SecretKey::restore`] to reconstruct the exact state.
//!
//! These are experimental constructions, not production cryptographic
//! primitives. The security relies on the 1D-SLSN conjecture for the cyclic
//! dual code, which has no settled parameter set. Secret state is not
//! zeroized on drop. The arithmetic has not received a formal constant-time
//! audit.
//!
//! [`trapdoor_matrices`]: trapdoor_matrices

pub mod code;
pub mod dispatch;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod params;
pub mod prf;
pub mod protocol;

pub use code::{CodeError, CyclicCodeScratch, CyclicDualCode};
pub use dispatch::{AnswerBackend, MIN_PARALLEL_MULTIPLICATIONS, select_answer_backend};
#[cfg(feature = "gpu")]
pub use dispatch::{AnswerDispatchError, AnswerDispatcher, MIN_GPU_MULTIPLICATIONS};
#[cfg(feature = "gpu")]
pub use gpu::{GpuAnswerer, GpuEncryptedMatrix, GpuError, PhaseTimings};
pub use params::{
    EmvpParams, POW_MAX_LAMBDA, PROTOCOL_MAX_LAMBDA, ParamsError, pow_ge_pow2, search,
};
pub use prf::{Prf, PrfError, purpose};
pub use protocol::{
    AnswerMatrix, DecodingKey, DerivedState, EncryptedMatrix, EncryptedQuery, MaskContextId,
    ProtocolError, QueryReservation, QueryReservations, QueryScratch, SecretKey, answer_batch,
    answer_into, decode_into, encrypt, query, query_batch, query_with_scratch,
};
pub use trapdoor_matrices::{RowStackMask, TdmMask};
