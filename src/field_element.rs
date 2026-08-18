use std::ops::{Add, Mul, Neg, Sub};

use crate::{FieldError, PrimeField};

/// A canonical residue associated with a prime field.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FieldElement<'a> {
    value: u32,
    field: &'a PrimeField,
}

impl<'a> FieldElement<'a> {
    /// Constructs an element by reducing `value` into `field`.
    pub fn new(field: &'a PrimeField, value: u64) -> Self {
        Self {
            value: field.reduce_u64(value),
            field,
        }
    }

    pub fn value(self) -> u32 {
        self.value
    }

    pub fn field(self) -> &'a PrimeField {
        self.field
    }

    pub fn square(self) -> Self {
        Self::from_canonical(self.field, self.field.square(self.value))
    }

    pub fn pow(self, exponent: u64) -> Self {
        Self::from_canonical(self.field, self.field.pow(self.value, exponent))
    }

    pub fn inv(self) -> Result<Self, FieldError> {
        self.field
            .inv(self.value)
            .map(|value| Self::from_canonical(self.field, value))
    }

    fn from_canonical(field: &'a PrimeField, value: u32) -> Self {
        Self { value, field }
    }

    fn assert_same_field(self, rhs: Self) {
        assert_eq!(
            self.field, rhs.field,
            "cannot operate on elements from different fields"
        );
    }
}

impl<'a> PrimeField {
    pub fn element(&'a self, value: u64) -> FieldElement<'a> {
        FieldElement::new(self, value)
    }
}

impl Add for FieldElement<'_> {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        self.assert_same_field(rhs);
        Self::from_canonical(self.field, self.field.add(self.value, rhs.value))
    }
}

impl Sub for FieldElement<'_> {
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output {
        self.assert_same_field(rhs);
        Self::from_canonical(self.field, self.field.sub(self.value, rhs.value))
    }
}

impl Mul for FieldElement<'_> {
    type Output = Self;

    fn mul(self, rhs: Self) -> Self::Output {
        self.assert_same_field(rhs);
        Self::from_canonical(self.field, self.field.mul(self.value, rhs.value))
    }
}

impl Neg for FieldElement<'_> {
    type Output = Self;

    fn neg(self) -> Self::Output {
        Self::from_canonical(self.field, self.field.neg(self.value))
    }
}

impl std::fmt::Display for FieldElement<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.value)
    }
}
