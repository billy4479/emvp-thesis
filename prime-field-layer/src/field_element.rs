use std::ops::{Add, AddAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::{FieldError, PrimeField};

mod bulk;

/// A field element stored as a four-byte Montgomery residue.
///
/// For a canonical value `a`, the private word represents `a * 2^32 mod
/// MODULUS`, not `a` itself. Arithmetic keeps this representation so repeated
/// multiplication, including NTT pointwise multiplication, does not repeatedly
/// convert at API boundaries. Use [`Self::value`] to obtain a canonical `u32`.
/// The modulus is part of the type, and an element stores no field pointer.
/// Hot add, subtract, and Montgomery multiplication kernels are designed without
/// value-dependent branches, but the crate has not been formally audited as a
/// constant-time implementation. Exponentiation has separate caveats documented
/// on [`Self::pow`].
#[repr(transparent)]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct FieldElement<const MODULUS: u32> {
    montgomery: u32,
}

impl<const MODULUS: u32> FieldElement<MODULUS> {
    /// Reduces `value` modulo `MODULUS` and converts it to Montgomery form.
    ///
    /// This takes constant space and performs one `u64` remainder plus a
    /// Montgomery conversion. For a `u32` input, [`PrimeField::element_u32`]
    /// avoids the separate remainder. `field` establishes the matching
    /// compile-time modulus; it is not retained by the element.
    #[must_use]
    pub fn new(field: &PrimeField<MODULUS>, value: u64) -> Self {
        Self::from_montgomery(PrimeField::<MODULUS>::to_montgomery(
            field.reduce_u64(value),
        ))
    }

    /// Returns this element as its canonical residue in `0..MODULUS`.
    ///
    /// This performs one Montgomery reduction in constant space. Prefer
    /// keeping intermediate values as [`FieldElement`]s and converting only at
    /// an external boundary or after an inverse NTT.
    #[must_use]
    pub fn value(self) -> u32 {
        PrimeField::<MODULUS>::from_montgomery(self.montgomery)
    }

    /// Returns the raw Montgomery residue stored in this element.
    ///
    /// The returned word is `a * 2^32 mod MODULUS` for the canonical value
    /// `a`, which is exactly the representation [`FieldElement`] keeps
    /// internally. This exposes that word without conversion so callers can
    /// stream field elements into external buffers, such as GPU uploads, in
    /// the native representation and read them back with [`Self::try_from_raw`].
    /// The word is always a canonical residue, so equality on raw words
    /// matches equality on elements.
    #[inline(always)]
    #[must_use]
    pub const fn to_raw(self) -> u32 {
        self.montgomery
    }

    /// Wraps a raw Montgomery residue into an element, checking canonicality.
    ///
    /// `raw` must be the canonical Montgomery residue `a * 2^32 mod MODULUS`
    /// of some canonical `a`, that is, a word in `0..MODULUS` as produced by
    /// [`Self::to_raw`]. This is the checked inverse of [`Self::to_raw`]: one
    /// comparison keeps the zero-copy interchange safe without canonicalizing
    /// or reducing the word.
    ///
    /// Unsupported moduli are rejected during compilation, so `F_2` cannot
    /// enter through the raw-word interchange either:
    ///
    /// ```compile_fail
    /// let _ = prime_field_layer::FieldElement::<2>::try_from_raw(1);
    /// ```
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::NonCanonicalRaw`] when `raw >= MODULUS`. Like
    /// every other construction path, this refuses to compile for moduli the
    /// crate does not support, such as `F_2`.
    #[inline(always)]
    pub const fn try_from_raw(raw: u32) -> Result<Self, FieldError> {
        let () = PrimeField::<MODULUS>::VALID_MODULUS;
        if raw < MODULUS {
            Ok(Self::from_raw_unchecked(raw))
        } else {
            Err(FieldError::NonCanonicalRaw {
                raw,
                modulus: MODULUS,
            })
        }
    }

    /// Wraps a raw word into an element without checking canonicality.
    ///
    /// Crate-internal fast path for words whose canonicality the crate has
    /// already established, such as the output of [`Self::try_from_raw`]'s
    /// bound check.
    #[inline(always)]
    pub(crate) const fn from_raw_unchecked(raw: u32) -> Self {
        debug_assert!(
            raw < MODULUS,
            "raw word must be a canonical Montgomery residue"
        );
        Self::from_montgomery(raw)
    }

    /// Returns the zero-sized field value for this element's modulus.
    ///
    /// This is an `O(1)` type-level association and neither inspects nor
    /// converts the element. It routes through [`PrimeField::new`], so the
    /// odd-prime modulus validation is identical and cannot be bypassed.
    #[must_use]
    pub const fn field(self) -> PrimeField<MODULUS> {
        PrimeField::<MODULUS>::new()
    }

    /// Squares this element while retaining Montgomery representation.
    ///
    /// This is one Montgomery multiplication, takes constant space, and is
    /// equivalent to `self * self`.
    #[must_use]
    #[inline(always)]
    pub fn square(self) -> Self {
        Self::from_montgomery(PrimeField::<MODULUS>::montgomery_mul_scalar(
            self.montgomery,
            self.montgomery,
        ))
    }

    /// Raises this element to `exponent` in the field.
    ///
    /// Binary exponentiation takes `O(log(exponent + 1))` Montgomery
    /// multiplications and allocates nothing. An exponent of zero returns one,
    /// including for a zero base. Control flow depends on the exponent bits, so
    /// this method is not intended to hide a secret exponent.
    #[must_use]
    pub fn pow(self, exponent: u64) -> Self {
        Self::from_montgomery(PrimeField::<MODULUS>::pow_montgomery(
            self.montgomery,
            exponent,
        ))
    }

    /// Returns the multiplicative inverse of this element.
    ///
    /// This uses Fermat exponentiation by the public, compile-time exponent
    /// `MODULUS - 2`, takes `O(log MODULUS)` operations, and allocates nothing.
    /// It returns [`FieldError::DivisionByZero`] for zero.
    ///
    /// # Errors
    ///
    /// Returns [`FieldError::DivisionByZero`] when this element is zero.
    pub fn inv(self) -> Result<Self, FieldError> {
        if self.montgomery == 0 {
            return Err(FieldError::DivisionByZero);
        }
        Ok(self.pow(u64::from(MODULUS - 2)))
    }

    #[inline(always)]
    pub(super) const fn from_montgomery(montgomery: u32) -> Self {
        Self { montgomery }
    }

    #[inline(always)]
    pub(super) fn from_u32(value: u32) -> Self {
        Self::from_montgomery(PrimeField::<MODULUS>::to_montgomery(value))
    }

    #[inline(always)]
    pub(super) const fn montgomery(self) -> u32 {
        self.montgomery
    }

    #[inline(always)]
    pub(super) const fn set_montgomery(&mut self, montgomery: u32) {
        self.montgomery = montgomery;
    }
}

impl<const MODULUS: u32> PrimeField<MODULUS> {
    /// Reduces a `u64` and constructs a Montgomery-form [`FieldElement`].
    ///
    /// This is the convenient general constructor and is equivalent to
    /// [`FieldElement::new`]. For `u32` coefficients, especially when preparing
    /// NTT buffers, [`Self::element_u32`] avoids a separate remainder.
    #[must_use]
    pub fn element(&self, value: u64) -> FieldElement<MODULUS> {
        FieldElement::new(self, value)
    }

    /// Reduces any `u32` and constructs a Montgomery-form [`FieldElement`].
    ///
    /// The result represents `value mod MODULUS`, even when `value` is not
    /// canonical. Montgomery reduction accepts the full-width input directly,
    /// avoiding the `u64` remainder performed by [`Self::element`]. This is an
    /// `O(1)`, allocation-free conversion suited to coefficient buffers.
    #[must_use]
    pub fn element_u32(&self, value: u32) -> FieldElement<MODULUS> {
        FieldElement::from_u32(value)
    }
}

impl<const MODULUS: u32> Add for FieldElement<MODULUS> {
    type Output = Self;

    #[inline(always)]
    fn add(self, rhs: Self) -> Self::Output {
        let field = self.field();
        Self::from_montgomery(field.add_canonical(self.montgomery, rhs.montgomery))
    }
}

impl<const MODULUS: u32> AddAssign for FieldElement<MODULUS> {
    #[inline(always)]
    fn add_assign(&mut self, rhs: Self) {
        *self = *self + rhs;
    }
}

impl<const MODULUS: u32> Sub for FieldElement<MODULUS> {
    type Output = Self;

    #[inline(always)]
    fn sub(self, rhs: Self) -> Self::Output {
        let field = self.field();
        Self::from_montgomery(field.sub_canonical(self.montgomery, rhs.montgomery))
    }
}

impl<const MODULUS: u32> SubAssign for FieldElement<MODULUS> {
    #[inline(always)]
    fn sub_assign(&mut self, rhs: Self) {
        *self = *self - rhs;
    }
}

impl<const MODULUS: u32> Mul for FieldElement<MODULUS> {
    type Output = Self;

    #[inline(always)]
    fn mul(self, rhs: Self) -> Self::Output {
        Self::from_montgomery(PrimeField::<MODULUS>::montgomery_mul_scalar(
            self.montgomery,
            rhs.montgomery,
        ))
    }
}

impl<const MODULUS: u32> MulAssign for FieldElement<MODULUS> {
    #[inline(always)]
    fn mul_assign(&mut self, rhs: Self) {
        *self = *self * rhs;
    }
}

impl<const MODULUS: u32> Neg for FieldElement<MODULUS> {
    type Output = Self;

    #[inline(always)]
    fn neg(self) -> Self::Output {
        let field = self.field();
        Self::from_montgomery(field.neg_canonical(self.montgomery))
    }
}

impl<const MODULUS: u32> std::fmt::Display for FieldElement<MODULUS> {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(formatter, "{}", self.value())
    }
}
