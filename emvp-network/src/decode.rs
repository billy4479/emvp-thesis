//! Value-region streaming into preallocated workspaces.
//!
//! Each decode function pairs with a plan: it streams the value region
//! described by the plan into a workspace previously grown by that
//! workspace's `reserve`, proves exact payload consumption with
//! [`FrameReader::finish`], and returns borrowed views over the
//! workspace. Given a reserved workspace, decoding performs zero managed
//! allocations.

use std::io::Read;

use crate::error::CodecError;
use crate::frame::FrameReader;
use crate::plan::{EvaluatePlan, ProductsPlan, UploadAcceptedPlan, UploadPlan};
use crate::views::{EvaluateViews, ProductsViews, UploadAcceptedIds, UploadViews};
use crate::workspace::{
    EvaluateWorkspace, ProductsWorkspace, UploadAcceptedWorkspace, UploadWorkspace,
};

/// Streams one upload's value region into `workspace` and returns views
/// over the decoded matrices.
///
/// `workspace` must have been reserved with [`UploadWorkspace::reserve`]
/// for this `plan`; a workspace too small is rejected instead of
/// panicking.
///
/// # Errors
///
/// Returns [`CodecError::AllocationFailed`] when the workspace was not
/// reserved for this plan, [`CodecError::NonCanonicalField`] for an
/// encoded integer at or above the modulus, [`CodecError::TruncatedFrame`]
/// or [`CodecError::TrailingFrameBytes`] when the payload does not end
/// exactly at the value region, and the framing errors of
/// [`FrameReader`].
pub fn decode_upload<'ws, R: Read>(
    frame: &mut FrameReader<'_, R>,
    plan: &UploadPlan,
    workspace: &'ws mut UploadWorkspace,
) -> Result<UploadViews<'ws>, CodecError> {
    for meta in &plan.matrices {
        let fields = meta
            .rows
            .checked_mul(meta.columns)
            .ok_or(CodecError::DimensionOverflow)?;
        let end = meta
            .value_offset
            .checked_add(fields)
            .ok_or(CodecError::DimensionOverflow)?;
        let slice = workspace
            .arena
            .get_mut(meta.value_offset..end)
            .ok_or(CodecError::AllocationFailed)?;
        frame.read_field_slice_into(slice)?;
    }
    frame.finish()?;
    Ok(workspace.views())
}

/// Copies one accepted upload's identifiers into `workspace` and returns
/// the borrowed list.
///
/// `workspace` must have been reserved with
/// [`UploadAcceptedWorkspace::reserve`] for this `plan`.
///
/// # Errors
///
/// Returns [`CodecError::AllocationFailed`] when the workspace was not
/// reserved for this plan, and [`CodecError::TrailingFrameBytes`] when
/// the payload does not end exactly at the identifier list.
pub fn decode_upload_accepted<'ws, R: Read>(
    frame: &mut FrameReader<'_, R>,
    plan: &UploadAcceptedPlan,
    workspace: &'ws mut UploadAcceptedWorkspace,
) -> Result<UploadAcceptedIds<'ws>, CodecError> {
    if workspace.identifiers.capacity() < plan.identifiers.len() {
        return Err(CodecError::AllocationFailed);
    }
    workspace.identifiers.clear();
    workspace.identifiers.extend_from_slice(&plan.identifiers);
    frame.finish()?;
    Ok(workspace.identifiers())
}

/// Streams one evaluation's value region into `workspace` and returns
/// views over the decoded entries.
///
/// `workspace` must have been reserved with
/// [`EvaluateWorkspace::reserve`] for this `plan`.
///
/// # Errors
///
/// Returns the same errors as [`decode_upload`].
pub fn decode_evaluate<'ws, R: Read>(
    frame: &mut FrameReader<'_, R>,
    plan: &EvaluatePlan,
    workspace: &'ws mut EvaluateWorkspace,
) -> Result<EvaluateViews<'ws>, CodecError> {
    for meta in &plan.entries {
        let fields = meta
            .query_count
            .checked_mul(meta.query_width)
            .ok_or(CodecError::DimensionOverflow)?;
        let end = meta
            .value_offset
            .checked_add(fields)
            .ok_or(CodecError::DimensionOverflow)?;
        let slice = workspace
            .arena
            .get_mut(meta.value_offset..end)
            .ok_or(CodecError::AllocationFailed)?;
        frame.read_field_slice_into(slice)?;
    }
    frame.finish()?;
    Ok(workspace.views())
}

/// Streams one products response's value region into `workspace` and
/// returns views over the decoded entries.
///
/// `workspace` must have been reserved with
/// [`ProductsWorkspace::reserve`] for this `plan`.
///
/// # Errors
///
/// Returns the same errors as [`decode_upload`].
pub fn decode_products<'ws, R: Read>(
    frame: &mut FrameReader<'_, R>,
    plan: &ProductsPlan,
    workspace: &'ws mut ProductsWorkspace,
) -> Result<ProductsViews<'ws>, CodecError> {
    for meta in &plan.entries {
        let words = meta
            .rows
            .checked_mul(meta.blocks)
            .ok_or(CodecError::DimensionOverflow)?;
        let fields = meta
            .answer_count
            .checked_mul(words)
            .ok_or(CodecError::DimensionOverflow)?;
        let end = meta
            .value_offset
            .checked_add(fields)
            .ok_or(CodecError::DimensionOverflow)?;
        let slice = workspace
            .arena
            .get_mut(meta.value_offset..end)
            .ok_or(CodecError::AllocationFailed)?;
        frame.read_field_slice_into(slice)?;
    }
    frame.finish()?;
    Ok(workspace.views())
}
