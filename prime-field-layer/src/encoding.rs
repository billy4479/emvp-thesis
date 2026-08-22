//! Canonical field-element encoding and cryptographic uniform sampling.
//!
//! The wire encoding is always four bytes. It is the canonical residue in
//! little-endian order, not the internal Montgomery representation. Decoding
//! rejects integers greater than or equal to the field modulus rather than
//! reducing them.
//!
//! Sampling uses caller-owned [`rand_core::CryptoRng`] state. This module does
//! not choose an RNG, hold global RNG state, or derive protocol seeds. For a
//! target set of size `n`, it accepts a random 32-bit word below
//! `floor(2^32 / n) * n` and returns its remainder modulo `n`. The accepted
//! interval contains every possible remainder the same number of times.
//!
//! # Side-channel model
//!
//! Rejection sampling takes a variable number of iterations. Its control flow
//! can reveal how many random words were rejected, so these functions are not
//! constant-time. They are intended for fresh sampling from a cryptographic RNG
//! whose state and raw output remain secret. The modulus and output length may
//! be public. The accepted field element is statistically independent of the
//! rejection count, but callers must not use these APIs where timing must be
//! independent of RNG consumption or where observing consumption can expose RNG
//! state.

use std::fmt;

use rand_core::CryptoRng;

use crate::{FieldElement, PrimeField};

/// Number of bytes in the canonical wire encoding of a field element.
pub const FIELD_ELEMENT_ENCODED_SIZE: usize = size_of::<u32>();

/// Error returned when an encoded integer is not a canonical field element.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct NonCanonicalEncoding {
    value: u32,
    modulus: u32,
}

impl NonCanonicalEncoding {
    /// Returns the noncanonical integer read from the encoding.
    #[must_use]
    pub const fn value(self) -> u32 {
        self.value
    }

    /// Returns the field modulus that the integer must be below.
    #[must_use]
    pub const fn modulus(self) -> u32 {
        self.modulus
    }
}

impl fmt::Display for NonCanonicalEncoding {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            formatter,
            "encoded field element {} is not below modulus {}",
            self.value, self.modulus
        )
    }
}

impl std::error::Error for NonCanonicalEncoding {}

impl<const MODULUS: u32> FieldElement<MODULUS> {
    /// Encodes this element as its canonical four-byte little-endian residue.
    ///
    /// The result is independent of the internal Montgomery representation.
    /// Values smaller than `2^24` have zeroes in the high bytes; the encoding
    /// is never shortened.
    #[must_use]
    pub fn to_canonical_le_bytes(self) -> [u8; FIELD_ELEMENT_ENCODED_SIZE] {
        self.value().to_le_bytes()
    }
}

impl<const MODULUS: u32> PrimeField<MODULUS> {
    /// Decodes a canonical four-byte little-endian field element.
    ///
    /// Unlike [`Self::element_u32`], this wire-format boundary does not reduce
    /// its input.
    ///
    /// # Errors
    ///
    /// Returns [`NonCanonicalEncoding`] if the encoded integer is greater than
    /// or equal to `MODULUS`.
    pub fn from_canonical_le_bytes(
        &self,
        encoding: [u8; FIELD_ELEMENT_ENCODED_SIZE],
    ) -> Result<FieldElement<MODULUS>, NonCanonicalEncoding> {
        let value = u32::from_le_bytes(encoding);
        if value >= MODULUS {
            return Err(NonCanonicalEncoding {
                value,
                modulus: MODULUS,
            });
        }
        Ok(FieldElement::from_u32(value))
    }

    /// Samples a uniformly distributed element of `F_q` from `rng`.
    ///
    /// The caller must provide and seed the cryptographic RNG. Rejection makes
    /// the iteration count variable as described in the [`crate::encoding`]
    /// module's side-channel model.
    #[must_use]
    pub fn sample_uniform<R: CryptoRng + ?Sized>(&self, rng: &mut R) -> FieldElement<MODULUS> {
        FieldElement::from_u32(sample_below(rng, MODULUS))
    }

    /// Samples a uniformly distributed nonzero element of `F_q` from `rng`.
    ///
    /// This samples `[0, q - 1)` without bias and adds one, so zero cannot be
    /// produced. Rejection makes the iteration count variable as described in
    /// the [`crate::encoding`] module's side-channel model.
    #[must_use]
    pub fn sample_uniform_nonzero<R: CryptoRng + ?Sized>(
        &self,
        rng: &mut R,
    ) -> FieldElement<MODULUS> {
        FieldElement::from_u32(sample_below(rng, MODULUS - 1) + 1)
    }

    /// Fills `output` with independent uniform elements of `F_q`.
    ///
    /// This allocates no memory. The caller owns the RNG and output storage.
    /// Rejection makes the total RNG consumption and running time variable.
    pub fn fill_uniform<R: CryptoRng + ?Sized>(
        &self,
        rng: &mut R,
        output: &mut [FieldElement<MODULUS>],
    ) {
        for value in output {
            *value = self.sample_uniform(rng);
        }
    }

    /// Fills `output` with independent uniform elements of `F_q*`.
    ///
    /// This allocates no memory and never writes zero. The caller owns the RNG
    /// and output storage. Rejection makes the total RNG consumption and
    /// running time variable.
    pub fn fill_uniform_nonzero<R: CryptoRng + ?Sized>(
        &self,
        rng: &mut R,
        output: &mut [FieldElement<MODULUS>],
    ) {
        for value in output {
            *value = self.sample_uniform_nonzero(rng);
        }
    }
}

#[inline]
fn sample_below<R: CryptoRng + ?Sized>(rng: &mut R, cardinality: u32) -> u32 {
    let cardinality = u64::from(cardinality);
    let source_cardinality = 1u64 << u32::BITS;
    let acceptance_bound = source_cardinality / cardinality * cardinality;

    loop {
        let candidate = u64::from(rng.next_u32());
        if candidate < acceptance_bound {
            return (candidate % cardinality) as u32;
        }
    }
}
