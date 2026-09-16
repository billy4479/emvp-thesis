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
//! # Answer production: plan → reserve → execute
//!
//! Every answer path — the CPU batch kernels, the GPU single-batch and
//! packed flights, and the mixed [`engine`](answer_engine) surface —
//! follows the same three-step contract instead of returning freshly
//! allocated answers:
//!
//! 1. **Plan**: validates the whole job set all-or-nothing and binds the
//!    exact inputs by borrowing, so executing a plan against different
//!    inputs is unrepresentable.
//! 2. **Reserve**: the caller's workspace grows once to the plan's peak
//!    (checked, initialized storage; the only fallible growth point) and
//!    then serves every same-shape call unchanged.
//! 3. **Execute**: fills disjoint ranges of the caller's arena and returns
//!    borrowed answer views. It allocates nothing project-owned, never
//!    resizes, and returns no views on error; once execution has begun,
//!    failed output contents are unspecified but the workspace stays
//!    memory-safe and reusable.
//!
//! The contract ends where project code ends: `wgpu` allocates per-submit
//! state (command encoders, bind groups, map callbacks) and drivers
//! allocate outside the Rust allocator entirely, so the guarantee is zero
//! *project-controlled* allocations after reserve, not a silent process.
//! It is enforced by counting-allocator tests on the CPU and codec paths
//! and observed end to end by the `gpu_phase_sweep` bench's host RSS,
//! high-water, and minor-fault telemetry.
//!
//! [`answer_engine`]: engine::AnswerEngine
//!
//! [`trapdoor_matrices`]: trapdoor_matrices

pub mod answer;
pub mod code;
pub mod dispatch;
pub mod engine;
#[cfg(feature = "gpu")]
pub mod gpu;
pub mod params;
pub mod prf;
pub mod protocol;
pub mod view;

pub use answer::{AnswerPlan, AnswerShape, AnswerWorkspace, Answers, execute_answer_batch};
pub use code::{CodeError, CyclicCodeScratch, CyclicDualCode};
#[cfg(feature = "gpu")]
pub use dispatch::MIN_GPU_MULTIPLICATIONS;
pub use dispatch::{
    AnswerBackend, MIN_PARALLEL_MULTIPLICATIONS, select_answer_backend, select_cpu_backend,
};
pub use engine::{
    AnswerEngine, AnswerEngineError, AnswerJob, EngineAnswers, EnginePlan, EngineReport,
    EngineWorkspace, EntryPlan, PrepareBatchError, PrepareEntry, PrepareError, PrepareMatrixError,
    PreparePlan, PrepareWorkspace, PreparedMatrix, UploadRef,
};
#[cfg(feature = "gpu")]
pub use gpu::{GpuAnswerer, GpuEncryptedMatrix, GpuError, PhaseTimings};
pub use params::{
    EmvpParams, POW_MAX_LAMBDA, PROTOCOL_MAX_LAMBDA, ParamsError, pow_ge_pow2, search,
};
pub use prf::{Prf, PrfError, purpose};
pub use protocol::{
    DecodingKey, DerivedState, EncryptedMatrix, EncryptedQuery, MaskContextId, ProtocolError,
    QueryReservation, QueryReservations, QueryScratch, SecretKey, answer_into, decode_into,
    encrypt, query, query_batch, query_with_scratch,
};
pub use trapdoor_matrices::{RowStackMask, TdmMask};
pub use view::{
    AnswerRef, AnswerValues, EncryptedMatrixRef, EncryptedQueryRef, MatrixValues, QueryValues,
};
