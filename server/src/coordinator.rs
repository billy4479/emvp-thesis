//! The process-wide GPU coordinator: one device, one operation at a time,
//! and a byte budget for long-lived matrix buffers.

use std::sync::{Mutex, MutexGuard};

use emvp::{GpuAnswerer, GpuError};

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
            answerer: GpuAnswerer::new_sync()?,
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

#[cfg(test)]
mod tests {
    use super::ResidencyBudget;

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
}
