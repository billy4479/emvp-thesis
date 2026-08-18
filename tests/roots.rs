use prime_field_layer::PrimeField;

#[test]
fn roots_have_exact_requested_order() {
    for (modulus, max_length) in [(17, 16), (65_537, 65_536), (998_244_353, 1 << 23)] {
        let field = PrimeField::new(modulus).unwrap();
        let mut length = 1;
        while length <= max_length {
            let root = field.root_of_unity(length).unwrap();
            assert!(root < modulus);
            assert_eq!(field.pow(root, length as u64), 1);
            if length > 1 {
                assert_ne!(field.pow(root, (length / 2) as u64), 1);
            } else {
                assert_eq!(root, 1);
            }
            length *= 2;
        }
    }
}

#[test]
fn rejects_non_power_of_two_and_unsupported_lengths() {
    let field = PrimeField::new(65_537).unwrap();

    for length in [0, 3, 6, 65_535, 131_072] {
        assert!(
            field.root_of_unity(length).is_err(),
            "accepted transform length {length}"
        );
    }
}
