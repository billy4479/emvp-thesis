//! Borrowed encode inputs and decode outputs over protocol v2 payloads.
//!
//! The same borrowed shapes serve both directions. An
//! [`UploadMatrixView`] is what the encoder streams and what decoding an
//! upload yields; [`EncryptedQueryRef`] and [`AnswerRef`] slices are what
//! the encoder consumes, and the entry views hand out lazily constructed
//! refs over the workspace arena. Views hold only borrowed element slices
//! and scalar metadata, so constructing one is pure pointer work and the
//! encoder or consumer never copies field elements.

use emvp::{AnswerRef, EmvpParams, EncryptedMatrixRef, EncryptedQueryRef, ProtocolError};

use crate::frame::{Field, PROTOCOL_MODULUS};
use crate::v2::plan::{EvaluateEntryMeta, EvaluateQueryMeta, ProductsEntryMeta, UploadMatrixMeta};
use crate::v2::workspace::{EvaluateWorkspace, ProductsWorkspace, UploadWorkspace};

/// One uploaded encrypted matrix, viewed generically.
///
/// Holds the parameters the matrix was encrypted under plus the shape,
/// identifier, and value slice of the ciphertext. This is both the
/// encoder's input record and the item yielded by iterating
/// [`UploadViews`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct UploadMatrixView<'a> {
    /// The protocol parameters the matrix was encrypted under.
    pub params: EmvpParams,
    /// The public matrix-instance identifier.
    pub instance_id: u128,
    /// The encrypted matrix rows.
    pub rows: usize,
    /// The encrypted matrix columns (`n = 2k`).
    pub columns: usize,
    /// The row-major encrypted entries.
    pub values: &'a [Field],
}

impl<'a> UploadMatrixView<'a> {
    /// Returns a borrowed [`EncryptedMatrixRef`] over this record.
    ///
    /// # Errors
    ///
    /// Returns the [`EncryptedMatrixRef::new`] errors, which a view built
    /// by a validated decode never triggers.
    pub fn matrix(&self) -> Result<EncryptedMatrixRef<'a, PROTOCOL_MODULUS>, ProtocolError> {
        EncryptedMatrixRef::new(self.instance_id, self.rows, self.columns, self.values)
    }
}

/// One evaluation entry as the encoder consumes it.
///
/// A matrix identifier plus the entry's encrypted queries as borrowed
/// refs. Every query of an entry must share one coordinate count, the
/// entry's wire `query_width`.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EvaluateEntryInput<'a> {
    /// The server identifier of the targeted matrix.
    pub matrix_id: u64,
    /// The encrypted queries, in order.
    pub queries: &'a [EncryptedQueryRef<'a, PROTOCOL_MODULUS>],
}

/// One products entry as the encoder consumes it: the answered matrix's
/// identifiers and shape plus its answers as borrowed refs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProductEntryInput<'a> {
    /// The server identifier of the answered matrix.
    pub matrix_id: u64,
    /// The public matrix-instance identifier written in the entry
    /// descriptor.
    pub instance_id: u128,
    /// The row count of every answer.
    pub rows: usize,
    /// The block count of every answer.
    pub blocks: usize,
    /// The encrypted answers, in the entry's query order.
    pub answers: &'a [AnswerRef<'a, PROTOCOL_MODULUS>],
}

/// The decoded matrix-set upload of one frame, borrowing the
/// [`UploadWorkspace`] the values were streamed into.
#[derive(Clone, Copy, Debug)]
pub struct UploadViews<'a> {
    workspace: &'a UploadWorkspace,
}

impl<'a> UploadViews<'a> {
    pub(crate) const fn new(workspace: &'a UploadWorkspace) -> Self {
        Self { workspace }
    }

    /// The number of uploaded matrices.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.workspace.matrices.len()
    }

    /// Whether the upload carried no matrices.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.workspace.matrices.is_empty()
    }

    /// Returns the matrix at `index`, or `None` out of range.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<UploadMatrixView<'a>> {
        let meta = self.workspace.matrices.get(index)?;
        matrix_view(meta, &self.workspace.arena)
    }

    /// Iterates the uploaded matrices in wire order.
    #[must_use]
    pub const fn iter(&self) -> UploadMatrixViewIter<'a> {
        UploadMatrixViewIter {
            views: *self,
            index: 0,
        }
    }
}

impl<'a> IntoIterator for UploadViews<'a> {
    type Item = UploadMatrixView<'a>;
    type IntoIter = UploadMatrixViewIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &UploadViews<'a> {
    type Item = UploadMatrixView<'a>;
    type IntoIter = UploadMatrixViewIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Builds the public matrix view from one validated metadata record and
/// the workspace arena.
fn matrix_view<'a>(meta: &UploadMatrixMeta, arena: &'a [Field]) -> Option<UploadMatrixView<'a>> {
    let end = meta
        .value_offset
        .checked_add(meta.rows.checked_mul(meta.columns)?)?;
    Some(UploadMatrixView {
        params: meta.params,
        instance_id: meta.instance_id,
        rows: meta.rows,
        columns: meta.columns,
        values: arena.get(meta.value_offset..end)?,
    })
}

/// An owning iterator over [`UploadViews`].
#[derive(Clone, Debug)]
pub struct UploadMatrixViewIter<'a> {
    views: UploadViews<'a>,
    index: usize,
}

impl<'a> Iterator for UploadMatrixViewIter<'a> {
    type Item = UploadMatrixView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.views.get(self.index)?;
        self.index += 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.views.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for UploadMatrixViewIter<'_> {}

/// The decoded evaluation request of one frame, borrowing the
/// [`EvaluateWorkspace`] the values were streamed into.
#[derive(Clone, Copy, Debug)]
pub struct EvaluateViews<'a> {
    workspace: &'a EvaluateWorkspace,
}

impl<'a> EvaluateViews<'a> {
    pub(crate) const fn new(workspace: &'a EvaluateWorkspace) -> Self {
        Self { workspace }
    }

    /// The number of evaluation entries.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.workspace.entries.len()
    }

    /// Whether the request carried no entries.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.workspace.entries.is_empty()
    }

    /// Returns the entry at `index`, or `None` out of range.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<EvaluateEntryView<'a>> {
        let meta = self.workspace.entries.get(index)?;
        let end = meta
            .value_offset
            .checked_add(meta.query_count.checked_mul(meta.query_width)?)?;
        Some(EvaluateEntryView {
            meta,
            queries: self.workspace.queries.get(meta.query_offset..)?,
            arena: self.workspace.arena.get(meta.value_offset..end)?,
        })
    }

    /// Iterates the entries in wire order.
    #[must_use]
    pub const fn iter(&self) -> EvaluateEntryIter<'a> {
        EvaluateEntryIter {
            views: *self,
            index: 0,
        }
    }
}

impl<'a> IntoIterator for EvaluateViews<'a> {
    type Item = EvaluateEntryView<'a>;
    type IntoIter = EvaluateEntryIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &EvaluateViews<'a> {
    type Item = EvaluateEntryView<'a>;
    type IntoIter = EvaluateEntryIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// An owning iterator over [`EvaluateViews`].
#[derive(Clone, Debug)]
pub struct EvaluateEntryIter<'a> {
    views: EvaluateViews<'a>,
    index: usize,
}

impl<'a> Iterator for EvaluateEntryIter<'a> {
    type Item = EvaluateEntryView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.views.get(self.index)?;
        self.index += 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.views.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for EvaluateEntryIter<'_> {}

/// One decoded evaluation entry, borrowing the workspace arena. Query
/// refs are constructed lazily on access from the scalar metadata and
/// arena subslices.
#[derive(Clone, Copy, Debug)]
pub struct EvaluateEntryView<'a> {
    meta: &'a EvaluateEntryMeta,
    queries: &'a [EvaluateQueryMeta],
    arena: &'a [Field],
}

impl<'a> EvaluateEntryView<'a> {
    /// The server identifier of the targeted matrix.
    #[must_use]
    pub const fn matrix_id(&self) -> u64 {
        self.meta.matrix_id
    }

    /// The coordinate count shared by every query of the entry.
    #[must_use]
    pub const fn query_width(&self) -> usize {
        self.meta.query_width
    }

    /// The number of encrypted queries in the entry.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.meta.query_count
    }

    /// Whether the entry carried no queries.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.meta.query_count == 0
    }

    /// Returns the query at `index` as a borrowed ref, or `None` out of
    /// range.
    #[must_use]
    pub fn query(&self, index: usize) -> Option<EncryptedQueryRef<'a, PROTOCOL_MODULUS>> {
        if index >= self.meta.query_count {
            return None;
        }
        // `self.queries` already starts at the entry's `query_offset`, and
        // `self.arena` at the entry's `value_offset`, so both index
        // relative to the entry.
        let query = self.queries.get(index)?;
        let start = index.checked_mul(self.meta.query_width)?;
        let values = self
            .arena
            .get(start..start.checked_add(self.meta.query_width)?)?;
        query_ref(query, values)
    }

    /// Iterates the entry's queries as borrowed refs, in wire order.
    #[must_use]
    pub const fn iter(&self) -> EvaluateQueryIter<'a> {
        EvaluateQueryIter {
            view: *self,
            index: 0,
        }
    }
}

impl<'a> IntoIterator for &'a EvaluateEntryView<'a> {
    type Item = EncryptedQueryRef<'a, PROTOCOL_MODULUS>;
    type IntoIter = EvaluateQueryIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// Lazily builds a query ref from scalar metadata and an arena subslice.
fn query_ref<'a>(
    query: &EvaluateQueryMeta,
    values: &'a [Field],
) -> Option<EncryptedQueryRef<'a, PROTOCOL_MODULUS>> {
    // `EncryptedQueryRef::new` imposes no invariant of its own, so this
    // never returns `None` for a validated plan.
    EncryptedQueryRef::new(query.instance_id, query.query_id, values).ok()
}

/// An owning iterator over one entry's queries as borrowed refs.
#[derive(Clone, Debug)]
pub struct EvaluateQueryIter<'a> {
    view: EvaluateEntryView<'a>,
    index: usize,
}

impl<'a> Iterator for EvaluateQueryIter<'a> {
    type Item = EncryptedQueryRef<'a, PROTOCOL_MODULUS>;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.view.query(self.index)?;
        self.index += 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.view.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for EvaluateQueryIter<'_> {}

/// The decoded products of one frame, borrowing the
/// [`ProductsWorkspace`] the values were streamed into.
#[derive(Clone, Copy, Debug)]
pub struct ProductsViews<'a> {
    workspace: &'a ProductsWorkspace,
}

impl<'a> ProductsViews<'a> {
    pub(crate) const fn new(workspace: &'a ProductsWorkspace) -> Self {
        Self { workspace }
    }

    /// The number of product entries.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.workspace.entries.len()
    }

    /// Whether the response carried no entries.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.workspace.entries.is_empty()
    }

    /// Returns the entry at `index`, or `None` out of range.
    #[must_use]
    pub fn get(&self, index: usize) -> Option<ProductEntryView<'a>> {
        let meta = self.workspace.entries.get(index)?;
        let words = meta.rows.checked_mul(meta.blocks)?;
        let end = meta
            .value_offset
            .checked_add(words.checked_mul(meta.answer_count)?)?;
        Some(ProductEntryView {
            meta,
            query_ids: self.workspace.query_ids.get(meta.answer_offset..)?,
            arena: self.workspace.arena.get(meta.value_offset..end)?,
        })
    }

    /// Iterates the entries in wire order.
    #[must_use]
    pub const fn iter(&self) -> ProductsEntryIter<'a> {
        ProductsEntryIter {
            views: *self,
            index: 0,
        }
    }
}

impl<'a> IntoIterator for ProductsViews<'a> {
    type Item = ProductEntryView<'a>;
    type IntoIter = ProductsEntryIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

impl<'a> IntoIterator for &ProductsViews<'a> {
    type Item = ProductEntryView<'a>;
    type IntoIter = ProductsEntryIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// An owning iterator over [`ProductsViews`].
#[derive(Clone, Debug)]
pub struct ProductsEntryIter<'a> {
    views: ProductsViews<'a>,
    index: usize,
}

impl<'a> Iterator for ProductsEntryIter<'a> {
    type Item = ProductEntryView<'a>;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.views.get(self.index)?;
        self.index += 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.views.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ProductsEntryIter<'_> {}

/// One decoded product entry, borrowing the workspace arena. Answer refs
/// are constructed lazily on access and paired positionally with the
/// entry's query-id descriptors.
#[derive(Clone, Copy, Debug)]
pub struct ProductEntryView<'a> {
    meta: &'a ProductsEntryMeta,
    query_ids: &'a [u64],
    arena: &'a [Field],
}

impl<'a> ProductEntryView<'a> {
    /// The server identifier of the answered matrix.
    #[must_use]
    pub const fn matrix_id(&self) -> u64 {
        self.meta.matrix_id
    }

    /// The public matrix-instance identifier.
    #[must_use]
    pub const fn instance_id(&self) -> u128 {
        self.meta.instance_id
    }

    /// The row count of every answer of the entry.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.meta.rows
    }

    /// The block count of every answer of the entry.
    #[must_use]
    pub const fn blocks(&self) -> usize {
        self.meta.blocks
    }

    /// The number of answers in the entry.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.meta.answer_count
    }

    /// Whether the entry carried no answers.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.meta.answer_count == 0
    }

    /// Returns the answer at `index` as a borrowed ref, or `None` out of
    /// range.
    #[must_use]
    pub fn answer(&self, index: usize) -> Option<AnswerRef<'a, PROTOCOL_MODULUS>> {
        if index >= self.meta.answer_count {
            return None;
        }
        // `self.query_ids` already starts at the entry's `answer_offset`,
        // and `self.arena` at the entry's `value_offset`, so both index
        // relative to the entry.
        let query_id = *self.query_ids.get(index)?;
        let words = self.meta.rows.checked_mul(self.meta.blocks)?;
        let start = index.checked_mul(words)?;
        let values = self.arena.get(start..start.checked_add(words)?)?;
        // `AnswerRef::new` re-checks the plan-validated shape, so this
        // never returns `None` for a decoded workspace.
        AnswerRef::new(
            self.meta.instance_id,
            query_id,
            values,
            self.meta.rows,
            self.meta.blocks,
        )
        .ok()
    }

    /// Iterates the entry's answers as borrowed refs, in wire order.
    #[must_use]
    pub const fn iter(&self) -> ProductAnswerIter<'a> {
        ProductAnswerIter {
            view: *self,
            index: 0,
        }
    }
}

impl<'a> IntoIterator for &'a ProductEntryView<'a> {
    type Item = AnswerRef<'a, PROTOCOL_MODULUS>;
    type IntoIter = ProductAnswerIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        self.iter()
    }
}

/// An owning iterator over one entry's answers as borrowed refs.
#[derive(Clone, Debug)]
pub struct ProductAnswerIter<'a> {
    view: ProductEntryView<'a>,
    index: usize,
}

impl<'a> Iterator for ProductAnswerIter<'a> {
    type Item = AnswerRef<'a, PROTOCOL_MODULUS>;

    fn next(&mut self) -> Option<Self::Item> {
        let item = self.view.answer(self.index)?;
        self.index += 1;
        Some(item)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let remaining = self.view.len().saturating_sub(self.index);
        (remaining, Some(remaining))
    }
}

impl ExactSizeIterator for ProductAnswerIter<'_> {}

/// The decoded upload acknowledgment: the accepted matrix identifiers, in
/// upload order, borrowing the [`crate::v2::UploadAcceptedWorkspace`].
pub type UploadAcceptedIds<'a> = &'a [u64];
