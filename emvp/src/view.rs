//! Borrowed protocol views over wire workspaces.
//!
//! The protocol artifacts [`EncryptedMatrix`], [`EncryptedQuery`], and
//! [`AnswerMatrix`] own their field-element storage. Servers and decoders
//! that receive those artifacts over a transport, however, hold them as
//! borrowed byte- or element-slices, and copying them into the owned types
//! doubles peak memory for no benefit. This module defines Copy view
//! structs over borrowed element slices plus the [`QueryValues`],
//! [`MatrixValues`], and [`AnswerValues`] traits that abstract over both
//! representations, so pure consumers such as
//! [`crate::protocol::decode_into`] accept wire workspaces without copies.
//!
//! Views are constructed with the same checked invariants the owned
//! constructors enforce: positive dimensions and a value count that exactly
//! covers the row-major shape. The identifiers stay part of the view, so
//! the public-identifier cross-checks of the protocol run unchanged over
//! borrowed data.

use prime_field_layer::FieldElement;

use crate::protocol::{AnswerMatrix, EncryptedMatrix, EncryptedQuery, ProtocolError, check_len};

/// The encrypted query side of the protocol, viewed generically.
///
/// Implemented by the owned [`EncryptedQuery`] and by the borrowed
/// [`EncryptedQueryRef`], letting consumers read query coordinates and the
/// public identifiers without naming a concrete storage type.
pub trait QueryValues<const MODULUS: u32> {
    /// Returns the encrypted query coordinates (length `n`).
    #[must_use]
    fn values(&self) -> &[FieldElement<MODULUS>];

    /// Returns the public matrix-instance identifier.
    #[must_use]
    fn instance_id(&self) -> u128;

    /// Returns the public query identifier.
    #[must_use]
    fn query_id(&self) -> u64;
}

/// The encrypted matrix side of the protocol, viewed generically.
///
/// Implemented by the owned [`EncryptedMatrix`] and by the borrowed
/// [`EncryptedMatrixRef`], letting consumers read the row-major ciphertext
/// and its shape without naming a concrete storage type.
pub trait MatrixValues<const MODULUS: u32> {
    /// Returns the row-major encrypted entries.
    #[must_use]
    fn values(&self) -> &[FieldElement<MODULUS>];

    /// Returns the encrypted matrix rows.
    #[must_use]
    fn rows(&self) -> usize;

    /// Returns the encrypted matrix columns (`n = 2k`).
    #[must_use]
    fn columns(&self) -> usize;

    /// Returns the public matrix-instance identifier.
    #[must_use]
    fn instance_id(&self) -> u128;
}

/// The server answer side of the protocol, viewed generically.
///
/// Implemented by the owned [`AnswerMatrix`] and by the borrowed
/// [`AnswerRef`], letting decoders such as
/// [`crate::protocol::decode_into`] work over either representation.
pub trait AnswerValues<const MODULUS: u32> {
    /// Returns the row-major answer entries.
    #[must_use]
    fn values(&self) -> &[FieldElement<MODULUS>];

    /// Returns the answer row count (`m`).
    #[must_use]
    fn rows(&self) -> usize;

    /// Returns the answer column count (`s`).
    #[must_use]
    fn blocks(&self) -> usize;

    /// Returns the public matrix-instance identifier.
    #[must_use]
    fn instance_id(&self) -> u128;

    /// Returns the public query identifier.
    #[must_use]
    fn query_id(&self) -> u64;
}

/// A borrowed view of [`EncryptedMatrix`]: shape, identifiers, and the
/// row-major ciphertext slice, with no owned storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncryptedMatrixRef<'a, const MODULUS: u32> {
    rows: usize,
    columns: usize,
    instance_id: u128,
    values: &'a [FieldElement<MODULUS>],
}

impl<'a, const MODULUS: u32> EncryptedMatrixRef<'a, MODULUS> {
    /// Builds a borrowed encrypted matrix from its parts.
    ///
    /// The checks mirror [`EncryptedMatrix::from_parts`] (positive
    /// dimensions and a value count of exactly `rows * columns`), so a view
    /// and its owned counterpart reject the same malformed buffers.
    ///
    /// # Errors
    ///
    /// Returns an error for a zero row or column count, a value count that
    /// overflows `rows * columns`, or a slice whose length differs from
    /// `rows * columns`.
    pub fn new(
        instance_id: u128,
        rows: usize,
        columns: usize,
        values: &'a [FieldElement<MODULUS>],
    ) -> Result<Self, ProtocolError> {
        if rows == 0 {
            return Err(ProtocolError::LengthMismatch {
                name: "matrix rows",
                expected: 1,
                actual: 0,
            });
        }
        if columns == 0 {
            return Err(ProtocolError::LengthMismatch {
                name: "matrix columns",
                expected: 1,
                actual: 0,
            });
        }
        let expected = rows
            .checked_mul(columns)
            .ok_or(ProtocolError::DimensionOverflow)?;
        check_len("matrix values", expected, values.len())?;
        Ok(Self {
            rows,
            columns,
            instance_id,
            values,
        })
    }

    /// Returns the encrypted matrix rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the encrypted matrix columns (`n = 2k`).
    #[must_use]
    pub const fn columns(&self) -> usize {
        self.columns
    }

    /// Returns the public matrix-instance identifier.
    #[must_use]
    pub const fn instance_id(&self) -> u128 {
        self.instance_id
    }

    /// Returns the row-major encrypted entries.
    #[must_use]
    pub const fn values(&self) -> &'a [FieldElement<MODULUS>] {
        self.values
    }

    /// Builds a view directly from parts without re-validating.
    ///
    /// Crate-internal fast path behind [`EncryptedMatrix::as_ref`]: the
    /// owned matrix already validated its buffer in
    /// [`crate::protocol::EncryptedMatrix::from_parts`], so re-checking
    /// would duplicate work on a hot conversion.
    #[must_use]
    pub(crate) const fn new_unchecked(
        instance_id: u128,
        rows: usize,
        columns: usize,
        values: &'a [FieldElement<MODULUS>],
    ) -> Self {
        Self {
            rows,
            columns,
            instance_id,
            values,
        }
    }
}

/// A borrowed view of [`EncryptedQuery`]: the identifiers and the encrypted
/// query coordinate slice, with no owned storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct EncryptedQueryRef<'a, const MODULUS: u32> {
    instance_id: u128,
    query_id: u64,
    values: &'a [FieldElement<MODULUS>],
}

impl<'a, const MODULUS: u32> EncryptedQueryRef<'a, MODULUS> {
    /// Builds a borrowed encrypted query from its parts.
    ///
    /// Like [`EncryptedQuery::from_parts`], this constructor imposes no
    /// structural invariant of its own: the coordinate count is a property
    /// of the protocol parameters (`n = 2k`), which a view cannot know, so
    /// the length check happens where the parameters are available, in
    /// [`crate::protocol::validate_query_against_matrix`]. The constructor
    /// therefore always succeeds; it returns a `Result` for uniformity with
    /// the other view constructors.
    ///
    /// # Errors
    ///
    /// Never returns an error; see above for why the signature is fallible.
    pub const fn new(
        instance_id: u128,
        query_id: u64,
        values: &'a [FieldElement<MODULUS>],
    ) -> Result<Self, ProtocolError> {
        Ok(Self {
            instance_id,
            query_id,
            values,
        })
    }

    /// Returns the public matrix-instance identifier.
    #[must_use]
    pub const fn instance_id(&self) -> u128 {
        self.instance_id
    }

    /// Returns the public query identifier.
    #[must_use]
    pub const fn query_id(&self) -> u64 {
        self.query_id
    }

    /// Returns the encrypted query coordinates (length `n`).
    #[must_use]
    pub const fn values(&self) -> &'a [FieldElement<MODULUS>] {
        self.values
    }

    /// Builds a view directly from parts without validating.
    ///
    /// Crate-internal fast path behind
    /// [`EncryptedQuery::as_ref`](crate::protocol::EncryptedQuery::as_ref),
    /// mirroring the owned type's unchecked `from_parts`.
    #[must_use]
    pub(crate) const fn new_unchecked(
        instance_id: u128,
        query_id: u64,
        values: &'a [FieldElement<MODULUS>],
    ) -> Self {
        Self {
            instance_id,
            query_id,
            values,
        }
    }
}

/// A borrowed view of [`AnswerMatrix`]: shape, identifiers, and the
/// row-major answer slice, with no owned storage.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct AnswerRef<'a, const MODULUS: u32> {
    rows: usize,
    blocks: usize,
    instance_id: u128,
    query_id: u64,
    values: &'a [FieldElement<MODULUS>],
}

impl<'a, const MODULUS: u32> AnswerRef<'a, MODULUS> {
    /// Builds a borrowed answer matrix from its parts.
    ///
    /// The checks match what [`crate::protocol::decode_into`] validates for
    /// the owned type (positive dimensions and a value count of exactly
    /// `rows * blocks`), so malformed wire buffers are rejected at
    /// construction instead of at every consumer call.
    ///
    /// # Errors
    ///
    /// Returns an error for a zero row or block count, a value count that
    /// overflows `rows * blocks`, or a slice whose length differs from
    /// `rows * blocks`.
    pub fn new(
        instance_id: u128,
        query_id: u64,
        values: &'a [FieldElement<MODULUS>],
        rows: usize,
        blocks: usize,
    ) -> Result<Self, ProtocolError> {
        if rows == 0 {
            return Err(ProtocolError::LengthMismatch {
                name: "answer rows",
                expected: 1,
                actual: 0,
            });
        }
        if blocks == 0 {
            return Err(ProtocolError::LengthMismatch {
                name: "answer blocks",
                expected: 1,
                actual: 0,
            });
        }
        let expected = rows
            .checked_mul(blocks)
            .ok_or(ProtocolError::DimensionOverflow)?;
        check_len("answer matrix", expected, values.len())?;
        Ok(Self {
            rows,
            blocks,
            instance_id,
            query_id,
            values,
        })
    }

    /// Returns the answer row count (`m`).
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Returns the answer column count (`s`).
    #[must_use]
    pub const fn blocks(&self) -> usize {
        self.blocks
    }

    /// Returns the public matrix-instance identifier.
    #[must_use]
    pub const fn instance_id(&self) -> u128 {
        self.instance_id
    }

    /// Returns the public query identifier.
    #[must_use]
    pub const fn query_id(&self) -> u64 {
        self.query_id
    }

    /// Returns the row-major answer entries.
    #[must_use]
    pub const fn values(&self) -> &'a [FieldElement<MODULUS>] {
        self.values
    }

    /// Builds a view directly from parts without validating.
    ///
    /// Crate-internal fast path behind
    /// [`AnswerMatrix::as_ref`](crate::protocol::AnswerMatrix::as_ref),
    /// mirroring the owned type's unchecked `from_parts`; consumers such as
    /// [`crate::protocol::decode_into`] validate the shape themselves.
    #[must_use]
    pub(crate) const fn new_unchecked(
        instance_id: u128,
        query_id: u64,
        rows: usize,
        blocks: usize,
        values: &'a [FieldElement<MODULUS>],
    ) -> Self {
        Self {
            rows,
            blocks,
            instance_id,
            query_id,
            values,
        }
    }
}

impl<const MODULUS: u32> QueryValues<MODULUS> for EncryptedQueryRef<'_, MODULUS> {
    fn values(&self) -> &[FieldElement<MODULUS>] {
        self.values
    }

    fn instance_id(&self) -> u128 {
        self.instance_id
    }

    fn query_id(&self) -> u64 {
        self.query_id
    }
}

impl<const MODULUS: u32> MatrixValues<MODULUS> for EncryptedMatrixRef<'_, MODULUS> {
    fn values(&self) -> &[FieldElement<MODULUS>] {
        self.values
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn columns(&self) -> usize {
        self.columns
    }

    fn instance_id(&self) -> u128 {
        self.instance_id
    }
}

impl<const MODULUS: u32> AnswerValues<MODULUS> for AnswerRef<'_, MODULUS> {
    fn values(&self) -> &[FieldElement<MODULUS>] {
        self.values
    }

    fn rows(&self) -> usize {
        self.rows
    }

    fn blocks(&self) -> usize {
        self.blocks
    }

    fn instance_id(&self) -> u128 {
        self.instance_id
    }

    fn query_id(&self) -> u64 {
        self.query_id
    }
}

impl<const MODULUS: u32> QueryValues<MODULUS> for EncryptedQuery<MODULUS> {
    fn values(&self) -> &[FieldElement<MODULUS>] {
        Self::values(self)
    }

    fn instance_id(&self) -> u128 {
        Self::instance_id(self)
    }

    fn query_id(&self) -> u64 {
        Self::query_id(self)
    }
}

impl<const MODULUS: u32> MatrixValues<MODULUS> for EncryptedMatrix<MODULUS> {
    fn values(&self) -> &[FieldElement<MODULUS>] {
        Self::values(self)
    }

    fn rows(&self) -> usize {
        Self::rows(self)
    }

    fn columns(&self) -> usize {
        Self::columns(self)
    }

    fn instance_id(&self) -> u128 {
        Self::instance_id(self)
    }
}

impl<const MODULUS: u32> AnswerValues<MODULUS> for AnswerMatrix<MODULUS> {
    fn values(&self) -> &[FieldElement<MODULUS>] {
        Self::values(self)
    }

    fn rows(&self) -> usize {
        Self::rows(self)
    }

    fn blocks(&self) -> usize {
        Self::blocks(self)
    }

    fn instance_id(&self) -> u128 {
        Self::instance_id(self)
    }

    fn query_id(&self) -> u64 {
        Self::query_id(self)
    }
}

impl<'a, const MODULUS: u32> From<&'a EncryptedMatrix<MODULUS>>
    for EncryptedMatrixRef<'a, MODULUS>
{
    fn from(matrix: &'a EncryptedMatrix<MODULUS>) -> Self {
        Self {
            rows: matrix.rows(),
            columns: matrix.columns(),
            instance_id: matrix.instance_id(),
            values: matrix.values(),
        }
    }
}

impl<'a, const MODULUS: u32> From<&'a EncryptedQuery<MODULUS>> for EncryptedQueryRef<'a, MODULUS> {
    fn from(query: &'a EncryptedQuery<MODULUS>) -> Self {
        Self {
            instance_id: query.instance_id(),
            query_id: query.query_id(),
            values: query.values(),
        }
    }
}

impl<'a, const MODULUS: u32> From<&'a AnswerMatrix<MODULUS>> for AnswerRef<'a, MODULUS> {
    fn from(answer: &'a AnswerMatrix<MODULUS>) -> Self {
        Self {
            rows: answer.rows(),
            blocks: answer.blocks(),
            instance_id: answer.instance_id(),
            query_id: answer.query_id(),
            values: answer.values(),
        }
    }
}
