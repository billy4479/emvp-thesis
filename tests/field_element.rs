use std::mem::size_of;

use prime_field_layer::{FieldElement, PrimeField};

#[test]
fn construction_reduces_to_a_canonical_residue() {
    let field = PrimeField::<17>::new();

    let element = FieldElement::new(&field, 20);
    assert_eq!(element.value(), 3);
    assert_eq!(element.field(), field);
    assert_eq!(field.element(35).value(), 1);
}

#[test]
fn arithmetic_uses_the_static_field() {
    let field = PrimeField::<17>::new();
    let lhs = field.element(15);
    let rhs = field.element(5);

    assert_eq!((lhs + rhs).value(), field.add(15, 5));
    assert_eq!((lhs - rhs).value(), field.sub(15, 5));
    assert_eq!((lhs * rhs).value(), field.mul(15, 5));
    assert_eq!((-lhs).value(), field.neg(15));
    assert_eq!(lhs.square().value(), field.square(15));
    assert_eq!(lhs.pow(13).value(), field.pow(15, 13));
    assert_eq!(lhs.inv().unwrap().value(), field.inv(15).unwrap());
    assert!(field.element(0).inv().is_err());
}

#[test]
fn assignment_operators_use_montgomery_arithmetic() {
    let field = PrimeField::<998_244_353>::new();
    let mut value = field.element(123_456_789);

    value += field.element(17);
    value *= field.element(31);
    value -= field.element(9);

    let expected = field.sub(field.mul(field.add(123_456_789, 17), 31), 9);
    assert_eq!(value.value(), expected);
}

#[test]
fn element_has_no_runtime_field_pointer() {
    assert_eq!(size_of::<FieldElement<998_244_353>>(), size_of::<u32>());
}
