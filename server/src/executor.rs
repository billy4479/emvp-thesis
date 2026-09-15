//! The coarse matrix-store interface the session is written against.
//!
//! The session must be testable without a compute device, so it talks to
//! a [`MatrixExecutor`] instead of the coordinator directly. The
//! trait deliberately mirrors only what one session needs: an atomic
//! whole-set upload, one answer batch per stored matrix, and the byte
//! release of its reservation. Device details (the shared lock, the
//! answerer, the phase timings) stay inside the coordinator.
//!
//! [`MatrixExecutor`]: crate::executor::MatrixExecutor
//! [`GpuCoordinator`]: crate::coordinator::GpuCoordinator

use emvp::{AnswerMatrix, EncryptedQuery, GpuError, ProtocolError};
use emvp_network::{MatrixUpload, PROTOCOL_MODULUS};

/// The session-side identity of one stored encrypted matrix.
#[derive(Clone, Copy, Debug)]
pub struct MatrixInfo {
    /// The public instance identifier the matrix was encrypted with.
    pub instance_id: u128,
    /// The matrix width (`n = 2k`), which every query must match.
    pub width: usize,
}

/// A failed matrix-store operation.
#[derive(Debug)]
#[non_exhaustive]
pub enum StoreError {
    /// A whole-set upload exceeds the remaining residency budget.
    ///
    /// Nothing was uploaded and nothing is reserved.
    BudgetExceeded {
        /// The byte count the set needs.
        requested_bytes: u64,
    },
    /// A device operation failed. The store released every partial
    /// effect before returning.
    Device(GpuError),
    /// A CPU-side (test) store failed protocol validation.
    ///
    /// Only the deterministic test store constructs this, so the normal
    /// build legitimately sees it as dead code.
    #[cfg_attr(
        not(test),
        expect(dead_code, reason = "constructed by the test store only")
    )]
    Protocol(ProtocolError),
}

impl std::fmt::Display for StoreError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::BudgetExceeded { requested_bytes } => write!(
                formatter,
                "the matrix set needs {requested_bytes} bytes and exceeds the residency budget"
            ),
            Self::Device(error) => error.fmt(formatter),
            Self::Protocol(error) => error.fmt(formatter),
        }
    }
}

impl std::error::Error for StoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Device(error) => Some(error),
            Self::Protocol(error) => Some(error),
            Self::BudgetExceeded { .. } => None,
        }
    }
}

/// The coarse matrix-store operations one session needs.
///
/// Deliberately narrower than the [`GpuCoordinator`]: the session sees
/// whole-set upload, per-matrix answering, and byte release, but never the
/// device lock, the answerer, or the phase timings.
pub trait MatrixExecutor {
    /// The handle one accepted upload produced.
    type Matrix;

    /// Uploads one whole matrix set atomically.
    ///
    /// On success the store keeps every uploaded matrix and `total_bytes`
    /// of device residency; the handles are in upload order and map to
    /// the one-based identifiers the session assigns. On any failure the
    /// store has released every partial effect (no handles, no
    /// reservation) before returning.
    ///
    /// # Errors
    ///
    /// Returns [`StoreError::BudgetExceeded`] when `total_bytes` exceeds
    /// the remaining budget, and the store failure otherwise.
    fn upload_set(
        &self,
        total_bytes: u64,
        uploads: &[MatrixUpload],
    ) -> Result<Vec<Self::Matrix>, StoreError>;

    /// The stored matrix's public identity.
    fn info(&self, matrix: &Self::Matrix) -> Option<MatrixInfo>;

    /// Answers one batch of encrypted queries against `matrix`.
    ///
    /// `matrix_id` is the session-assigned identifier, for logging only.
    ///
    /// # Errors
    ///
    /// Returns the store failure; the answers are all-or-nothing.
    fn answer_batch(
        &self,
        matrix_id: u64,
        matrix: &Self::Matrix,
        queries: &[EncryptedQuery<PROTOCOL_MODULUS>],
    ) -> Result<Vec<AnswerMatrix<PROTOCOL_MODULUS>>, StoreError>;

    /// Releases a prior successful upload's `bytes` of residency.
    fn release(&self, bytes: u64);

    /// The store's currently reserved byte count.
    fn reserved_bytes(&self) -> u64;
}
