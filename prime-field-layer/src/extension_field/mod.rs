//! Fixed monic polynomial reduction and extension-field arithmetic.
//!
//! [`ExtensionField`] represents `F_q[X]/(f)` for one fixed monic, irreducible
//! polynomial `f` of degree `K`. The modulus polynomial is used only for ordinary
//! polynomial remainder computation. When `K` is large, an NTT computes
//! zero-padded linear products before reduction; it never changes the quotient
//! to `X^K - 1` or any other NTT-friendly polynomial.
//!
//! Construction canonicalizes the modulus coefficients and precomputes the data
//! needed by repeated reduction. [`ExtensionField::new`] also runs Rabin's
//! deterministic Frobenius/GCD irreducibility test. Use
//! [`ExtensionField::new_unchecked_irreducible`] only when irreducibility was
//! established elsewhere. Both constructors validate the coefficient count and
//! monicity.
//!
//! Multiplication and squaring write into caller-provided arrays and use
//! [`ExtensionFieldScratch`], whose allocation is reusable. Addition and
//! subtraction need no scratch. Every public coefficient input may be any
//! `u32`; every public output and stored modulus coefficient is canonical.
//!
//! [`StaticExtensionField`] uses [`StaticNttPlan`](crate::StaticNttPlan) when
//! both the extension degree and transform length are protocol constants. The
//! dynamic type remains the default because static specialization is
//! size-dependent and adds transform tables to the binary.

mod error;
mod field;
mod irreducibility;
mod reduction;

pub use error::ExtensionFieldError;
pub use field::{ExtensionField, ExtensionFieldScratch, StaticExtensionField};
#[doc(hidden)]
pub use reduction::{DynamicReductionNtt, ReductionNtt, StaticReductionNtt};
pub use reduction::{
    PolynomialAlgorithm, PolynomialReductionPlan, PolynomialReductionScratch,
    SCHOOLBOOK_EXTENSION_DEGREE, StaticPolynomialReductionPlan,
};
