//! Frame framing, bounded payload reading, and canonical field-element
//! encoding.
//!
//! Every message after the handshake travels in one frame: a one-byte
//! [`FrameKind`] discriminator, a little-endian `u64` payload length, and
//! exactly that many payload bytes. Readers parse payloads through a
//! [`FrameReader`], which refuses to read past the frame boundary and
//! detects payloads that were shorter than their header declared, so a
//! truncated or padded stream never decodes into a silently wrong message.

use std::fmt;
use std::io::{Read, Write};

use prime_field_layer::{FieldElement, PrimeField, encoding::FIELD_ELEMENT_ENCODED_SIZE};

use crate::error::CodecError;

/// The protocol field type: every wire artifact uses this modulus.
pub type Field = FieldElement<PROTOCOL_MODULUS>;

/// The single field modulus both binaries compile with and the handshake
/// verifies: `998244353` is prime, NTT-friendly, and supported by the GPU
/// answer kernel.
pub const PROTOCOL_MODULUS: u32 = 998_244_353;

/// The protocol version the handshake must match exactly.
///
/// Version 2 replaces the v1 per-record fixed prefixes with
/// descriptor-table-first layouts; there is no v1 compatibility.
pub const PROTOCOL_VERSION: u16 = 2;

/// The handshake magic preceding every client hello.
pub const MAGIC: [u8; 4] = *b"EMVP";

/// Bytes in one frame header: the kind byte plus the `u64` payload length.
pub const HEADER_BYTES: u64 = 1 + size_of::<u64>() as u64;

/// Field elements encoded per chunk in the bulk slice codecs; 16 KiB.
const CHUNK_FIELDS: usize = 4096;

/// Bytes in one frame payload chunk.
const CHUNK_BYTES: usize = CHUNK_FIELDS * FIELD_ELEMENT_ENCODED_SIZE;

/// The kind of one frame.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[repr(u8)]
pub enum FrameKind {
    /// Client to server: the one matrix-set upload.
    UploadMatrices = 1,
    /// Server to client: the identifiers of an accepted upload.
    UploadAccepted = 2,
    /// Client to server: one ordered evaluation request.
    Evaluate = 3,
    /// Server to client: the encrypted products of one evaluation.
    Products = 4,
    /// Server to client: a rejected request.
    Error = 5,
}

impl FrameKind {
    /// The wire byte of this frame kind.
    #[must_use]
    pub const fn to_u8(self) -> u8 {
        self as u8
    }

    /// The frame kind with the given wire byte, if any.
    #[must_use]
    pub const fn from_u8(byte: u8) -> Option<Self> {
        match byte {
            1 => Some(Self::UploadMatrices),
            2 => Some(Self::UploadAccepted),
            3 => Some(Self::Evaluate),
            4 => Some(Self::Products),
            5 => Some(Self::Error),
            _ => None,
        }
    }
}

impl fmt::Display for FrameKind {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        let text = match self {
            Self::UploadMatrices => "UploadMatrices",
            Self::UploadAccepted => "UploadAccepted",
            Self::Evaluate => "Evaluate",
            Self::Products => "Products",
            Self::Error => "Error",
        };
        formatter.write_str(text)
    }
}

/// The kind and payload length read from a frame header.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FrameHeader {
    /// The frame's message kind.
    pub kind: FrameKind,
    /// The declared payload length in bytes.
    pub payload_len: u64,
}

/// Reads a frame header from the stream.
///
/// Returns `Ok(None)` on a clean end of stream before any byte of the
/// header, which is how a peer signals a finished session. A header that
/// starts but does not complete is a truncation error.
///
/// # Errors
///
/// Returns [`CodecError::UnknownFrameKind`] for a kind byte outside the
/// protocol, [`CodecError::TruncatedFrame`] for a partially received
/// header, and [`CodecError::Io`] for transport failures.
pub fn read_frame_header(reader: &mut impl Read) -> Result<Option<FrameHeader>, CodecError> {
    let mut kind = [0_u8; 1];
    let read = reader.read(&mut kind).map_err(CodecError::Io)?;
    if read == 0 {
        return Ok(None);
    }
    let mut length = [0_u8; 8];
    reader.read_exact(&mut length).map_err(map_read_error)?;
    let Some(kind) = FrameKind::from_u8(kind[0]) else {
        return Err(CodecError::UnknownFrameKind { kind: kind[0] });
    };
    Ok(Some(FrameHeader {
        kind,
        payload_len: u64::from_le_bytes(length),
    }))
}

/// Writes a frame header to the stream.
///
/// # Errors
///
/// Returns [`CodecError::Io`] if the transport fails.
pub fn write_frame_header(
    writer: &mut impl Write,
    kind: FrameKind,
    payload_len: u64,
) -> Result<(), CodecError> {
    write_u8(writer, kind.to_u8())?;
    write_u64(writer, payload_len)
}

/// A reader confined to one frame payload.
///
/// Every read checks against the payload length the frame header declared,
/// so payload parsing cannot silently spill into the next frame, and a
/// payload shorter than declared surfaces as [`CodecError::TruncatedFrame`]
/// instead of blocking on the socket.
pub struct FrameReader<'a, R: Read> {
    inner: &'a mut R,
    remaining: u64,
}

impl<'a, R: Read> FrameReader<'a, R> {
    /// Bounds `reader` to the next `payload_len` bytes.
    #[must_use]
    pub const fn new(reader: &'a mut R, payload_len: u64) -> Self {
        Self {
            inner: reader,
            remaining: payload_len,
        }
    }

    /// The number of payload bytes not yet consumed.
    #[must_use]
    pub const fn remaining(&self) -> u64 {
        self.remaining
    }

    /// Reads exactly `buf.len()` payload bytes.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::TruncatedFrame`] when the frame or the stream
    /// runs out first, and [`CodecError::Io`] for transport failures.
    pub fn read_exact_checked(&mut self, buf: &mut [u8]) -> Result<(), CodecError> {
        let need = u64::try_from(buf.len()).map_err(|_conversion| CodecError::DimensionOverflow)?;
        if need > self.remaining {
            return Err(CodecError::TruncatedFrame);
        }
        self.inner.read_exact(buf).map_err(map_read_error)?;
        self.remaining -= need;
        Ok(())
    }

    /// Reads one payload byte.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::read_exact_checked`].
    pub fn read_u8(&mut self) -> Result<u8, CodecError> {
        let mut buffer = [0_u8; 1];
        self.read_exact_checked(&mut buffer)?;
        Ok(buffer[0])
    }

    /// Reads one little-endian `u16` payload value.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::read_exact_checked`].
    pub fn read_u16(&mut self) -> Result<u16, CodecError> {
        let mut buffer = [0_u8; 2];
        self.read_exact_checked(&mut buffer)?;
        Ok(u16::from_le_bytes(buffer))
    }

    /// Reads one little-endian `u32` payload value.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::read_exact_checked`].
    pub fn read_u32(&mut self) -> Result<u32, CodecError> {
        let mut buffer = [0_u8; 4];
        self.read_exact_checked(&mut buffer)?;
        Ok(u32::from_le_bytes(buffer))
    }

    /// Reads one little-endian `u64` payload value.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::read_exact_checked`].
    pub fn read_u64(&mut self) -> Result<u64, CodecError> {
        let mut buffer = [0_u8; 8];
        self.read_exact_checked(&mut buffer)?;
        Ok(u64::from_le_bytes(buffer))
    }

    /// Reads one little-endian `u128` payload value.
    ///
    /// # Errors
    ///
    /// Returns the same errors as [`Self::read_exact_checked`].
    pub fn read_u128(&mut self) -> Result<u128, CodecError> {
        let mut buffer = [0_u8; 16];
        self.read_exact_checked(&mut buffer)?;
        Ok(u128::from_le_bytes(buffer))
    }

    /// Reads `count` canonical field elements in bounded chunks.
    ///
    /// The frame's remaining length is checked before any allocation, so a
    /// declared element count the payload cannot hold is rejected without
    /// reserving memory; a count the payload claims to hold but the host
    /// cannot buffer surfaces as [`CodecError::AllocationFailed`].
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::TruncatedFrame`] when the payload cannot hold
    /// `count` elements, [`CodecError::AllocationFailed`] when the buffer
    /// cannot be reserved, [`CodecError::NonCanonicalField`] for an encoded
    /// integer at or above the modulus, and [`CodecError::Io`] for
    /// transport failures.
    pub fn read_field_slice(&mut self, count: u64) -> Result<Vec<Field>, CodecError> {
        let field_bytes = field_byte_len(count)?;
        if field_bytes > self.remaining {
            return Err(CodecError::TruncatedFrame);
        }
        let count = usize::try_from(count).map_err(|_conversion| CodecError::AllocationFailed)?;
        let mut values = Vec::new();
        values
            .try_reserve_exact(count)
            .map_err(|_reserve| CodecError::AllocationFailed)?;
        let zero = PrimeField::<PROTOCOL_MODULUS>::new().element_u32(0);
        values.resize(count, zero);
        self.read_field_slice_into(&mut values)?;
        Ok(values)
    }

    /// Reads exactly `values.len()` canonical field elements into `values`
    /// in bounded chunks.
    ///
    /// This is the allocation-free form of [`Self::read_field_slice`]: the
    /// destination is caller-provided, so decoding only stages bytes
    /// through a fixed stack chunk. The frame's remaining length is checked
    /// before any byte is read.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::TruncatedFrame`] when the payload cannot hold
    /// `values.len()` elements, [`CodecError::NonCanonicalField`] for an
    /// encoded integer at or above the modulus, and [`CodecError::Io`] for
    /// transport failures.
    pub fn read_field_slice_into(&mut self, values: &mut [Field]) -> Result<(), CodecError> {
        let count =
            u64::try_from(values.len()).map_err(|_conversion| CodecError::DimensionOverflow)?;
        let field_bytes = field_byte_len(count)?;
        if field_bytes > self.remaining {
            return Err(CodecError::TruncatedFrame);
        }
        let field = PrimeField::<PROTOCOL_MODULUS>::new();
        let mut buffer = [0_u8; CHUNK_BYTES];
        let mut done = 0_usize;
        while done < values.len() {
            let take = (values.len() - done).min(CHUNK_FIELDS);
            let take_bytes = take * FIELD_ELEMENT_ENCODED_SIZE;
            self.read_exact_checked(&mut buffer[..take_bytes])?;
            let (words, _tail) = buffer[..take_bytes].as_chunks::<FIELD_ELEMENT_ENCODED_SIZE>();
            for (slot, word) in values[done..done + take].iter_mut().zip(words) {
                *slot = field
                    .from_canonical_le_bytes(*word)
                    .map_err(|noncanonical| CodecError::NonCanonicalField {
                        value: noncanonical.value(),
                    })?;
            }
            done += take;
        }
        Ok(())
    }

    /// Asserts the whole payload was consumed by exactly one message.
    ///
    /// # Errors
    ///
    /// Returns [`CodecError::TrailingFrameBytes`] if payload bytes remain.
    pub const fn finish(&mut self) -> Result<(), CodecError> {
        if self.remaining == 0 {
            Ok(())
        } else {
            Err(CodecError::TrailingFrameBytes {
                extra: self.remaining,
            })
        }
    }
}

impl<R: Read> Read for FrameReader<'_, R> {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let limit = usize::try_from(self.remaining)
            .unwrap_or(usize::MAX)
            .min(buf.len());
        if limit == 0 {
            return Ok(0);
        }
        let read = self.inner.read(&mut buf[..limit])?;
        self.remaining -= u64::try_from(read).unwrap_or(0);
        Ok(read)
    }
}

/// Writes one byte.
///
/// # Errors
///
/// Returns [`CodecError::Io`] if the transport fails.
pub fn write_u8(writer: &mut impl Write, value: u8) -> Result<(), CodecError> {
    writer.write_all(&[value]).map_err(CodecError::Io)
}

/// Writes one little-endian `u16`.
///
/// # Errors
///
/// Returns [`CodecError::Io`] if the transport fails.
pub fn write_u16(writer: &mut impl Write, value: u16) -> Result<(), CodecError> {
    writer
        .write_all(&value.to_le_bytes())
        .map_err(CodecError::Io)
}

/// Writes one little-endian `u32`.
///
/// # Errors
///
/// Returns [`CodecError::Io`] if the transport fails.
pub fn write_u32(writer: &mut impl Write, value: u32) -> Result<(), CodecError> {
    writer
        .write_all(&value.to_le_bytes())
        .map_err(CodecError::Io)
}

/// Writes one little-endian `u64`.
///
/// # Errors
///
/// Returns [`CodecError::Io`] if the transport fails.
pub fn write_u64(writer: &mut impl Write, value: u64) -> Result<(), CodecError> {
    writer
        .write_all(&value.to_le_bytes())
        .map_err(CodecError::Io)
}

/// Writes one little-endian `u128`.
///
/// # Errors
///
/// Returns [`CodecError::Io`] if the transport fails.
pub fn write_u128(writer: &mut impl Write, value: u128) -> Result<(), CodecError> {
    writer
        .write_all(&value.to_le_bytes())
        .map_err(CodecError::Io)
}

/// Writes `values` as canonical little-endian field elements in bounded
/// chunks, so arbitrarily large slices stream without an intermediate copy
/// of the whole encoded form.
///
/// # Errors
///
/// Returns [`CodecError::Io`] if the transport fails.
pub fn write_field_slice(writer: &mut impl Write, values: &[Field]) -> Result<(), CodecError> {
    let mut buffer = [0_u8; CHUNK_BYTES];
    for chunk in values.chunks(CHUNK_FIELDS) {
        let (word_slots, _tail) = buffer.as_chunks_mut::<FIELD_ELEMENT_ENCODED_SIZE>();
        for (slot, element) in word_slots.iter_mut().zip(chunk.iter()) {
            *slot = element.to_canonical_le_bytes();
        }
        writer
            .write_all(&buffer[..chunk.len() * FIELD_ELEMENT_ENCODED_SIZE])
            .map_err(CodecError::Io)?;
    }
    Ok(())
}

/// The byte length of `count` encoded field elements.
pub(crate) const fn field_byte_len(count: u64) -> Result<u64, CodecError> {
    const FIELD_BYTES: u64 = FIELD_ELEMENT_ENCODED_SIZE as u64;
    match count.checked_mul(FIELD_BYTES) {
        Some(bytes) => Ok(bytes),
        None => Err(CodecError::DimensionOverflow),
    }
}

/// Converts a host length into its wire `u64` representation.
pub(crate) fn wire_len(value: usize) -> Result<u64, CodecError> {
    u64::try_from(value).map_err(|_conversion| CodecError::DimensionOverflow)
}

/// Converts a wire count into its host `usize` representation.
pub(crate) fn host_len(name: &'static str, value: u64) -> Result<usize, CodecError> {
    usize::try_from(value).map_err(|_conversion| CodecError::ValueOutOfRange { name, value })
}

/// Adds two payload size components, rejecting overflow.
pub(crate) const fn checked_size(total: u64, addition: u64) -> Result<u64, CodecError> {
    match total.checked_add(addition) {
        Some(sum) => Ok(sum),
        None => Err(CodecError::DimensionOverflow),
    }
}

/// Maps a transport read failure, distinguishing a short stream from other
/// transport errors.
pub(crate) fn map_read_error(error: std::io::Error) -> CodecError {
    if error.kind() == std::io::ErrorKind::UnexpectedEof {
        CodecError::TruncatedFrame
    } else {
        CodecError::Io(error)
    }
}
