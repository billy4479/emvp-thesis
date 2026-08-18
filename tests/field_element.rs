use prime_field_layer::{FieldElement, PrimeField};

#[test]
fn construction_reduces_to_a_canonical_residue() {
    let field = PrimeField::new(17).unwrap();

    let element = FieldElement::new(&field, 20);

    assert_eq!(element.value(), 3);
    assert_eq!(element.field(), &field);
    assert_eq!(field.element(35).value(), 1);
}

#[test]
fn arithmetic_delegates_to_the_field() {
    let field = PrimeField::new(17).unwrap();
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
#[should_panic(expected = "cannot operate on elements from different fields")]
fn arithmetic_rejects_different_fields() {
    let field_17 = PrimeField::new(17).unwrap();
    let field_19 = PrimeField::new(19).unwrap();

    let _ = field_17.element(1) + field_19.element(1);
}
