//! The process-wide GPU coordinator: one device, one operation at a time,
//! and a byte budget for long-lived matrix buffers.

use std::sync::{Mutex, MutexGuard};

use emvp::{AnswerMatrix, EncryptedQuery, GpuAnswerer, GpuEncryptedMatrix, GpuError};
use emvp_network::{MatrixUpload, PROTOCOL_MODULUS};

use crate::executor::{MatrixExecutor, MatrixInfo, StoreError};

/// Recovering lock acquisition: the guarded values are an idempotent
/// counter and an opaque lock, so a panicking peer cannot have corrupted
/// them.
fn lock_recovered<T>(mutex: &Mutex<T>) -> MutexGuard<'_, T> {
    match mutex.lock() {
        Ok(guard) => guard,
        Err(poisoned) => poisoned.into_inner(),
    }
}

/// The GPU-residency accounting for stored encrypted matrices.
///
/// The budget tracks only the long-lived device buffers of loaded
/// matrices; the transient query and answer buffers of an answer batch
/// live outside it and are bounded by the adapter's own limits.
#[derive(Debug)]
pub struct ResidencyBudget {
    max_bytes: u64,
    reserved_bytes: u64,
}

impl ResidencyBudget {
    /// Creates a budget of `max_bytes` bytes with nothing reserved.
    #[must_use]
    pub const fn new(max_bytes: u64) -> Self {
        Self {
            max_bytes,
            reserved_bytes: 0,
        }
    }

    /// Reserves `bytes` bytes, returning `false` if the budget cannot
    /// cover them. The reservation is all-or-nothing.
    pub const fn reserve(&mut self, bytes: u64) -> bool {
        let Some(projected) = self.reserved_bytes.checked_add(bytes) else {
            return false;
        };
        if projected > self.max_bytes {
            return false;
        }
        self.reserved_bytes = projected;
        true
    }

    /// Releases a prior reservation of `bytes` bytes.
    pub const fn release(&mut self, bytes: u64) {
        self.reserved_bytes = self.reserved_bytes.saturating_sub(bytes);
    }

    /// The currently reserved byte count.
    #[must_use]
    pub const fn reserved_bytes(&self) -> u64 {
        self.reserved_bytes
    }
}

/// The shared GPU resources of one server process.
///
/// The answerer is used under [`Self::lock_gpu`], serializing every device
/// operation across all connection threads; the residency budget caps the
/// device memory bound up in stored matrices across all sessions.
pub struct GpuCoordinator {
    answerer: GpuAnswerer,
    gpu_lock: Mutex<()>,
    budget: Mutex<ResidencyBudget>,
}

impl GpuCoordinator {
    /// Initializes the compute device with a matrix-residency budget.
    ///
    /// # Errors
    ///
    /// Returns [`GpuError::NoAdapter`] when no suitable compute device is
    /// available, and the device-request error otherwise.
    pub fn new(max_gpu_bytes: u64) -> Result<Self, GpuError> {
        Ok(Self {
            answerer: GpuAnswerer::new()?,
            gpu_lock: Mutex::new(()),
            budget: Mutex::new(ResidencyBudget::new(max_gpu_bytes)),
        })
    }

    /// The shared answerer; hold [`Self::lock_gpu`] while using it.
    #[must_use]
    pub const fn answerer(&self) -> &GpuAnswerer {
        &self.answerer
    }

    /// Serializes GPU operations across all connection threads.
    pub fn lock_gpu(&self) -> MutexGuard<'_, ()> {
        lock_recovered(&self.gpu_lock)
    }

    /// Reserves `bytes` bytes of the shared residency budget.
    ///
    /// Returns `false` when the reservation exceeds the remaining budget.
    pub fn reserve(&self, bytes: u64) -> bool {
        lock_recovered(&self.budget).reserve(bytes)
    }

    /// Releases a prior reservation of `bytes` bytes.
    pub fn release(&self, bytes: u64) {
        lock_recovered(&self.budget).release(bytes);
    }

    /// The currently reserved byte count across all sessions.
    #[must_use]
    pub fn reserved_bytes(&self) -> u64 {
        lock_recovered(&self.budget).reserved_bytes()
    }
}

impl MatrixExecutor for GpuCoordinator {
    type Matrix = GpuEncryptedMatrix<PROTOCOL_MODULUS>;

    fn upload_set(
        &self,
        total_bytes: u64,
        uploads: &[MatrixUpload],
    ) -> Result<Vec<Self::Matrix>, StoreError> {
        if !self.reserve(total_bytes) {
            return Err(StoreError::BudgetExceeded {
                requested_bytes: total_bytes,
            });
        }
        // The whole set uploads under one device lock; a failure
        // anywhere releases the handles, the lock, and the reservation,
        // so the upload stays atomic.
        let gpu_guard = self.lock_gpu();
        let mut handles = Vec::with_capacity(uploads.len());
        for upload in uploads {
            match self
                .answerer()
                .upload_matrix(&upload.params, &upload.matrix)
            {
                Ok(handle) => handles.push(handle),
                Err(error) => {
                    drop(handles);
                    drop(gpu_guard);
                    Self::release(self, total_bytes);
                    return Err(StoreError::Device(error));
                }
            }
        }
        drop(gpu_guard);
        Ok(handles)
    }

    fn info(&self, matrix: &Self::Matrix) -> Option<MatrixInfo> {
        Some(MatrixInfo {
            instance_id: matrix.instance_id(),
            width: matrix.columns(),
        })
    }

    fn answer_batch(
        &self,
        matrix_id: u64,
        matrix: &Self::Matrix,
        queries: &[EncryptedQuery<PROTOCOL_MODULUS>],
    ) -> Result<Vec<AnswerMatrix<PROTOCOL_MODULUS>>, StoreError> {
        let gpu_guard = self.lock_gpu();
        let outcome = self
            .answerer()
            .answer_batch_with_timings(matrix, queries)
            .map(|(answers, timings)| {
                eprintln!(
                    "server: matrix {matrix_id}: answered {} queries in {} ms",
                    queries.len(),
                    timings.total().as_millis()
                );
                answers
            });
        drop(gpu_guard);
        outcome.map_err(StoreError::Device)
    }

    fn release(&self, bytes: u64) {
        Self::release(self, bytes);
    }

    fn reserved_bytes(&self) -> u64 {
        Self::reserved_bytes(self)
    }
}

#[cfg(test)]
mod tests {
    use super::{GpuCoordinator, MatrixExecutor, ResidencyBudget};
    use emvp_network::MatrixUpload;

    #[test]
    fn reservations_are_all_or_nothing() {
        let mut budget = ResidencyBudget::new(100);
        assert!(budget.reserve(60));
        assert_eq!(budget.reserved_bytes(), 60);
        assert!(!budget.reserve(60));
        assert_eq!(budget.reserved_bytes(), 60);
        assert!(budget.reserve(40));
        assert_eq!(budget.reserved_bytes(), 100);
    }

    #[test]
    fn releases_free_budget_for_later_reservations() {
        let mut budget = ResidencyBudget::new(100);
        assert!(budget.reserve(100));
        assert!(!budget.reserve(1));
        budget.release(100);
        assert_eq!(budget.reserved_bytes(), 0);
        assert!(budget.reserve(1));
    }

    #[test]
    fn overflowing_reservations_are_rejected() {
        let mut budget = ResidencyBudget::new(u64::MAX);
        assert!(budget.reserve(u64::MAX));
        assert!(!budget.reserve(1));
    }

    #[test]
    fn releasing_more_than_reserved_is_clamped() {
        let mut budget = ResidencyBudget::new(100);
        budget.release(50);
        assert_eq!(budget.reserved_bytes(), 0);
        assert!(budget.reserve(100));
    }

    /// A genuine device exercise: the coordinator's executor surface
    /// uploads a real encrypted matrix and releases its reservation.
    /// Ignored because it probes for a compute adapter.
    #[test]
    #[ignore = "requires a compute adapter"]
    fn coordinator_uploads_a_set_and_releases_its_reservation() {
        use crate::executor::StoreError;
        use emvp::{MaskContextId, SecretKey, encrypt};
        use emvp_network::PROTOCOL_MODULUS as MODULUS;
        use prime_field_layer::PrimeField;
        use rand_chacha::ChaCha20Rng;
        use rand_core::SeedableRng;
        use trapdoor_matrices::ToeplitzFastProduct;

        const PARAMS: emvp::EmvpParams = emvp::EmvpParams {
            k: 8,
            ell: 8,
            b: 2,
            lambda: 7,
        };

        let coordinator = GpuCoordinator::new(1 << 30).unwrap();
        let field = PrimeField::<MODULUS>::new();
        let mut rng = ChaCha20Rng::seed_from_u64(9);
        let mut state = SecretKey::<MODULUS>::new_insecure(PARAMS, [9; 32])
            .unwrap()
            .derive(
                MaskContextId::from_u64(0x0001),
                3,
                &mut rng,
                |stream, _index| {
                    ToeplitzFastProduct::sample(2 * PARAMS.k, stream)
                        .map_err(emvp::ProtocolError::Mask)
                },
            )
            .unwrap();
        let plaintext: Vec<_> = (0..3 * PARAMS.ell)
            .map(|index| field.element_u32((index % 11 + 1) as u32))
            .collect();
        let matrix = encrypt(&mut state, &plaintext).unwrap();
        let upload = MatrixUpload {
            params: PARAMS,
            matrix,
        };

        // A budget the set does not fit is refused before any upload.
        let bytes = u64::try_from(3 * PARAMS.n().unwrap() * 4).unwrap();
        let tight = GpuCoordinator::new(bytes - 1).unwrap();
        assert!(matches!(
            MatrixExecutor::upload_set(&tight, bytes, std::slice::from_ref(&upload)),
            Err(StoreError::BudgetExceeded {
                requested_bytes
            }) if requested_bytes == bytes
        ));

        let handles = MatrixExecutor::upload_set(&coordinator, bytes, &[upload]).unwrap();
        assert_eq!(handles.len(), 1);
        assert_eq!(coordinator.reserved_bytes(), bytes);
        let info = MatrixExecutor::info(&coordinator, &handles[0]).unwrap();
        assert_eq!(info.width, 16);
        drop(handles);
        coordinator.release(bytes);
        assert_eq!(coordinator.reserved_bytes(), 0);
    }
}
