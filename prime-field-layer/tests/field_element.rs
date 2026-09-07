use std::mem::size_of;

use prime_field_layer::{FieldElement, FieldError, PrimeField};

#[test]
fn construction_reduces_to_a_canonical_residue() {
    let field = PrimeField::<17>::new();

    let element = FieldElement::new(&field, 20);
    assert_eq!(element.value(), 3);
    assert_eq!(element.field(), field);
    assert_eq!(field.element(35).value(), 1);
}

fn check_element_u32_boundaries<const MODULUS: u32>() {
    let field = PrimeField::<MODULUS>::new();
    for value in [0, MODULUS - 1, MODULUS, u32::MAX] {
        let direct = field.element_u32(value);
        let generic = field.element(u64::from(value));
        let expected = (u64::from(value) % u64::from(MODULUS)) as u32;
        assert_eq!(direct, generic, "modulus {MODULUS}, input {value}");
        assert_eq!(direct.value(), expected, "modulus {MODULUS}, input {value}");
    }
}

#[test]
fn element_u32_reduces_boundaries_without_a_preliminary_remainder() {
    check_element_u32_boundaries::<2>();
    check_element_u32_boundaries::<17>();
    check_element_u32_boundaries::<4_294_967_291>();
}

#[test]
fn arithmetic_uses_the_static_field() {
    let field = PrimeField::<17>::new();
    let lhs = field.element(15);
    let rhs = field.element(5);

    assert_eq!((lhs + rhs).value(), field.add_canonical(15, 5));
    assert_eq!((lhs - rhs).value(), field.sub_canonical(15, 5));
    assert_eq!((lhs * rhs).value(), field.mul(15, 5));
    assert_eq!((-lhs).value(), field.neg_canonical(15));
    assert_eq!(lhs.square().value(), field.square(15));
    assert_eq!(lhs.pow(13).value(), field.pow(15, 13));
    assert_eq!(lhs.inv().unwrap().value(), field.inv(15).unwrap());
    field.element(0).inv().unwrap_err();
}

#[test]
fn assignment_operators_use_montgomery_arithmetic() {
    let field = PrimeField::<1_073_479_681>::new();
    let mut value = field.element(123_456_789);

    value += field.element(17);
    value *= field.element(31);
    value -= field.element(9);

    let expected = field.sub_canonical(field.mul(field.add_canonical(123_456_789, 17), 31), 9);
    assert_eq!(value.value(), expected);
}

#[test]
fn element_has_no_runtime_field_pointer() {
    assert_eq!(size_of::<FieldElement<1_073_479_681>>(), size_of::<u32>());
}

#[test]
fn raw_words_round_trip_without_conversion() {
    let field = PrimeField::<998_244_353>::new();
    for value in [0_u32, 1, 2, 998_244_351, 998_244_352] {
        let element = field.element_u32(value);
        // The raw word is the Montgomery residue of the canonical value,
        // `a * 2^32 mod MODULUS`.
        assert_eq!(
            element.to_raw(),
            ((u64::from(value) << 32) % 998_244_353) as u32
        );
        // from_raw is the exact inverse, and arithmetic round-trips through it.
        assert_eq!(
            FieldElement::<998_244_353>::from_raw(element.to_raw()).value(),
            value
        );
        assert_eq!(
            FieldElement::<998_244_353>::from_raw(element.to_raw()),
            element
        );
    }
    // Products computed on raw words match the canonical product, so a GPU
    // kernel working on to_raw words and returning from_raw values is
    // indistinguishable from element arithmetic.
    let (a, b) = (
        field.element_u32(123_456_789),
        field.element_u32(987_654_321),
    );
    let product_from_raw = FieldElement::from_raw(a.to_raw()) * FieldElement::from_raw(b.to_raw());
    assert_eq!(product_from_raw, a * b);
    assert_eq!(
        product_from_raw.value(),
        field.mul(123_456_789, 987_654_321)
    );
}

#[test]
fn write_raw_words_copies_the_montgomery_words() {
    let field = PrimeField::<998_244_353>::new();
    let values: Vec<_> = [0_u32, 1, 500_000_000, 998_244_352]
        .iter()
        .map(|&value| field.element_u32(value))
        .collect();
    let mut output = vec![0_u32; values.len()];
    field.write_raw_words(&values, &mut output).unwrap();
    let expected: Vec<_> = values.iter().map(|element| element.to_raw()).collect();
    assert_eq!(output, expected);

    // Length mismatches are rejected before any word is written.
    let mut unchanged = vec![7_u32; 4];
    assert!(matches!(
        field.write_raw_words(&values[..2], &mut unchanged),
        Err(FieldError::LengthMismatch)
    ));
    assert_eq!(unchanged, vec![7; 4]);
}
