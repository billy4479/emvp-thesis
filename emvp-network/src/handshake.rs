//! The connection handshake: an exact version and modulus check.
//!
//! The client opens every connection with a fixed ten-byte hello: the
//! [`MAGIC`] bytes, the protocol version, and the field modulus. The server
//! answers with a single status byte and closes the connection on any
//! mismatch, so no framed message ever crosses an unverified boundary.

use std::io::{Read, Write};

use crate::error::{CodecError, HandshakeError};
use crate::frame::{
    MAGIC, PROTOCOL_MODULUS, PROTOCOL_VERSION, map_read_error, write_u8, write_u16, write_u32,
};

/// Bytes in the client hello: magic, `u16` version, `u32` modulus.
pub const HELLO_BYTES: usize = MAGIC.len() + size_of::<u16>() + size_of::<u32>();

/// The client's hello: a protocol version and a field modulus.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ClientHello {
    /// The offered protocol version.
    pub version: u16,
    /// The offered field modulus.
    pub modulus: u32,
}

/// Writes a client hello.
///
/// # Errors
///
/// Returns [`CodecError::Io`] if the transport fails.
pub fn write_client_hello(
    writer: &mut impl Write,
    version: u16,
    modulus: u32,
) -> Result<(), CodecError> {
    writer.write_all(&MAGIC).map_err(CodecError::Io)?;
    write_u16(writer, version)?;
    write_u32(writer, modulus)
}

/// Reads a client hello and checks its magic.
///
/// # Errors
///
/// Returns [`CodecError::InvalidMagic`] for a foreign magic, the framing
/// errors for a short or failed read, and [`CodecError::Io`] for transport
/// failures.
pub fn read_client_hello(reader: &mut impl Read) -> Result<ClientHello, CodecError> {
    let mut hello = [0_u8; HELLO_BYTES];
    reader.read_exact(&mut hello).map_err(map_read_error)?;
    if hello[..MAGIC.len()] != MAGIC {
        return Err(CodecError::InvalidMagic);
    }
    let version = u16::from_le_bytes([hello[4], hello[5]]);
    let modulus = u32::from_le_bytes([hello[6], hello[7], hello[8], hello[9]]);
    Ok(ClientHello { version, modulus })
}

/// Writes the server's accept or reject status byte.
///
/// # Errors
///
/// Returns [`CodecError::Io`] if the transport fails.
pub fn write_server_hello(writer: &mut impl Write, accepted: bool) -> Result<(), CodecError> {
    write_u8(writer, u8::from(accepted))
}

/// Reads the server's status byte.
///
/// # Errors
///
/// Returns [`CodecError::InvalidHandshakeStatus`] for any byte other than
/// accept or reject, and [`CodecError::Io`] for transport failures.
pub fn read_server_hello(reader: &mut impl Read) -> Result<bool, CodecError> {
    let mut status = [0_u8; 1];
    reader.read_exact(&mut status).map_err(map_read_error)?;
    match status[0] {
        1 => Ok(true),
        0 => Ok(false),
        other => Err(CodecError::InvalidHandshakeStatus { status: other }),
    }
}

/// Performs the client side of the handshake on a fresh connection.
///
/// Writes this crate's protocol version and modulus and waits for the
/// server's verdict.
///
/// # Errors
///
/// Returns [`HandshakeError::Rejected`] when the server refuses the hello,
/// and the underlying codec errors otherwise.
pub fn client_handshake<S: Read + Write>(stream: &mut S) -> Result<(), HandshakeError> {
    write_client_hello(stream, PROTOCOL_VERSION, PROTOCOL_MODULUS)?;
    if read_server_hello(stream)? {
        Ok(())
    } else {
        Err(HandshakeError::Rejected)
    }
}

/// Performs the server side of the handshake on a fresh connection.
///
/// The connection is usable for framed messages only after this returns
/// `Ok`; on any mismatch the server has already written its rejection and
/// the caller must close the stream.
///
/// # Errors
///
/// Returns [`HandshakeError::UnsupportedVersion`] or
/// [`HandshakeError::UnsupportedModulus`] for a foreign hello, and the
/// underlying codec errors otherwise.
pub fn server_handshake<S: Read + Write>(stream: &mut S) -> Result<(), HandshakeError> {
    let hello = read_client_hello(stream)?;
    if hello.version != PROTOCOL_VERSION {
        write_server_hello(stream, false)?;
        return Err(HandshakeError::UnsupportedVersion {
            received: hello.version,
        });
    }
    if hello.modulus != PROTOCOL_MODULUS {
        write_server_hello(stream, false)?;
        return Err(HandshakeError::UnsupportedModulus {
            received: hello.modulus,
        });
    }
    write_server_hello(stream, true)?;
    Ok(())
}
