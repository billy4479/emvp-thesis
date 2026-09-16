//! Preallocated decode workspaces for protocol v2 payloads.
//!
//! A workspace owns the field-element arena the value region streams into
//! plus the plain-scalar descriptor metadata the plan validated. No
//! borrowed view is ever stored inside a workspace: views construct
//! [`EncryptedQueryRef`], [`AnswerRef`], and matrix views lazily on
//! access from the scalars and arena subslices, so a workspace never
//! holds self-references. [`Workspace::reserve`] is the pipeline's only
//! growth point; it is monotone, and reusing a workspace across frames of
//! the same or smaller shape reuses every allocation.

use prime_field_layer::PrimeField;

use crate::error::CodecError;
use crate::frame::{Field, PROTOCOL_MODULUS};
use crate::v2::plan::{
    EvaluateEntryMeta, EvaluatePlan, EvaluateQueryMeta, ProductsEntryMeta, ProductsPlan,
    UploadAcceptedPlan, UploadMatrixMeta, UploadPlan,
};
use crate::v2::views::{EvaluateViews, ProductsViews, UploadAcceptedIds, UploadViews};

/// The additive zero of the protocol field, the arena's fill value.
fn zero_field() -> Field {
    PrimeField::<PROTOCOL_MODULUS>::new().element_u32(0)
}

/// Grows a vector's length and capacity to `len` if smaller, filling new
/// slots with `fill`, and never shrinking.
fn grow<T: Clone>(vector: &mut Vec<T>, len: usize, fill: impl Fn() -> T) -> Result<(), CodecError> {
    if vector.len() < len {
        vector
            .try_reserve_exact(len - vector.len())
            .map_err(|_reserve| CodecError::AllocationFailed)?;
        vector.resize(len, fill());
    }
    Ok(())
}

/// Grows a vector's capacity to hold `len` elements if smaller, without
/// touching its length, and never shrinking.
fn grow_capacity<T>(vector: &mut Vec<T>, len: usize) -> Result<(), CodecError> {
    if vector.capacity() < len {
        vector
            .try_reserve_exact(len - vector.capacity())
            .map_err(|_reserve| CodecError::AllocationFailed)?;
    }
    Ok(())
}

/// A decode workspace for `UploadMatrices` payloads: the field-element
/// arena plus one scalar metadata record per uploaded matrix.
#[derive(Debug, Default)]
pub struct UploadWorkspace {
    pub(crate) arena: Vec<Field>,
    pub(crate) matrices: Vec<UploadMatrixMeta>,
}

impl UploadWorkspace {
    /// Creates an empty workspace with no capacity.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Grows the workspace to hold `plan`'s decoded form.
    ///
    /// The arena grows to at least `plan.value_fields` elements and the
    /// metadata table to `plan.matrices.len()` records. Growth is
    /// monotone: reserving for a plan of the same or smaller shape
    /// reuses the existing allocation.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::AllocationFailed`] when a reservation is
    /// refused.
    pub fn reserve(&mut self, plan: &UploadPlan) -> Result<(), CodecError> {
        grow(&mut self.arena, plan.value_fields, zero_field)?;
        grow_capacity(&mut self.matrices, plan.matrices.len())?;
        self.matrices.clear();
        self.matrices.extend_from_slice(&plan.matrices);
        Ok(())
    }

    /// The arena's capacity in field elements.
    #[must_use]
    pub const fn capacity_fields(&self) -> usize {
        self.arena.capacity()
    }

    /// Drops the decoded contents, keeping every allocation.
    pub fn clear(&mut self) {
        self.arena.clear();
        self.matrices.clear();
    }

    /// Releases the workspace, handing back the field-element arena for
    /// reuse elsewhere.
    #[must_use]
    pub fn release(self) -> Vec<Field> {
        self.arena
    }

    /// Borrows the decoded upload as views over the workspace.
    #[must_use]
    pub const fn views(&self) -> UploadViews<'_> {
        UploadViews::new(self)
    }
}

/// A decode workspace for `UploadAccepted` payloads: the accepted
/// identifier list.
#[derive(Debug, Default)]
pub struct UploadAcceptedWorkspace {
    pub(crate) identifiers: Vec<u64>,
}

impl UploadAcceptedWorkspace {
    /// Creates an empty workspace with no capacity.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Grows the workspace to hold `plan`'s identifier list.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::AllocationFailed`] when the reservation is
    /// refused.
    pub fn reserve(&mut self, plan: &UploadAcceptedPlan) -> Result<(), CodecError> {
        grow_capacity(&mut self.identifiers, plan.identifiers.len())?;
        self.identifiers.clear();
        self.identifiers.extend_from_slice(&plan.identifiers);
        Ok(())
    }

    /// The identifier buffer's capacity in slots.
    #[must_use]
    pub const fn capacity_fields(&self) -> usize {
        self.identifiers.capacity()
    }

    /// Drops the decoded contents, keeping every allocation.
    pub fn clear(&mut self) {
        self.identifiers.clear();
    }

    /// Releases the workspace, handing back the identifier buffer.
    #[must_use]
    pub fn release(self) -> Vec<u64> {
        self.identifiers
    }

    /// Borrows the decoded identifiers.
    #[must_use]
    pub fn identifiers(&self) -> UploadAcceptedIds<'_> {
        &self.identifiers
    }
}

/// A decode workspace for `Evaluate` payloads: the field-element arena
/// plus the entry and query descriptor metadata.
#[derive(Debug, Default)]
pub struct EvaluateWorkspace {
    pub(crate) arena: Vec<Field>,
    pub(crate) entries: Vec<EvaluateEntryMeta>,
    pub(crate) queries: Vec<EvaluateQueryMeta>,
}

impl EvaluateWorkspace {
    /// Creates an empty workspace with no capacity.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Grows the workspace to hold `plan`'s decoded form.
    ///
    /// Growth is monotone: reserving for a plan of the same or smaller
    /// shape reuses the existing allocations.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::AllocationFailed`] when a reservation is
    /// refused.
    pub fn reserve(&mut self, plan: &EvaluatePlan) -> Result<(), CodecError> {
        grow(&mut self.arena, plan.value_fields, zero_field)?;
        grow_capacity(&mut self.entries, plan.entries.len())?;
        self.entries.clear();
        self.entries.extend_from_slice(&plan.entries);
        grow_capacity(&mut self.queries, plan.queries.len())?;
        self.queries.clear();
        self.queries.extend_from_slice(&plan.queries);
        Ok(())
    }

    /// The arena's capacity in field elements.
    #[must_use]
    pub const fn capacity_fields(&self) -> usize {
        self.arena.capacity()
    }

    /// Drops the decoded contents, keeping every allocation.
    pub fn clear(&mut self) {
        self.arena.clear();
        self.entries.clear();
        self.queries.clear();
    }

    /// Releases the workspace, handing back the field-element arena for
    /// reuse elsewhere.
    #[must_use]
    pub fn release(self) -> Vec<Field> {
        self.arena
    }

    /// Borrows the decoded evaluation as views over the workspace.
    #[must_use]
    pub const fn views(&self) -> EvaluateViews<'_> {
        EvaluateViews::new(self)
    }
}

/// A decode workspace for `Products` payloads: the field-element arena
/// plus the entry descriptor and query-identifier metadata.
#[derive(Debug, Default)]
pub struct ProductsWorkspace {
    pub(crate) arena: Vec<Field>,
    pub(crate) entries: Vec<ProductsEntryMeta>,
    pub(crate) query_ids: Vec<u64>,
}

impl ProductsWorkspace {
    /// Creates an empty workspace with no capacity.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Grows the workspace to hold `plan`'s decoded form.
    ///
    /// Growth is monotone: reserving for a plan of the same or smaller
    /// shape reuses the existing allocations.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::AllocationFailed`] when a reservation is
    /// refused.
    pub fn reserve(&mut self, plan: &ProductsPlan) -> Result<(), CodecError> {
        grow(&mut self.arena, plan.value_fields, zero_field)?;
        grow_capacity(&mut self.entries, plan.entries.len())?;
        self.entries.clear();
        self.entries.extend_from_slice(&plan.entries);
        grow_capacity(&mut self.query_ids, plan.query_ids.len())?;
        self.query_ids.clear();
        self.query_ids.extend_from_slice(&plan.query_ids);
        Ok(())
    }

    /// The arena's capacity in field elements.
    #[must_use]
    pub const fn capacity_fields(&self) -> usize {
        self.arena.capacity()
    }

    /// Drops the decoded contents, keeping every allocation.
    pub fn clear(&mut self) {
        self.arena.clear();
        self.entries.clear();
        self.query_ids.clear();
    }

    /// Releases the workspace, handing back the field-element arena for
    /// reuse elsewhere.
    #[must_use]
    pub fn release(self) -> Vec<Field> {
        self.arena
    }

    /// Borrows the decoded products as views over the workspace.
    #[must_use]
    pub const fn views(&self) -> ProductsViews<'_> {
        ProductsViews::new(self)
    }
}
