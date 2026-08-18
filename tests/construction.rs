use prime_field_layer::PrimeField;

#[test]
fn accepts_representative_primes() {
    let primes = [
        2,
        3,
        5,
        17,
        257,
        65_537,
        998_244_353,
        2_147_483_647,
        4_294_967_291,
    ];

    for modulus in primes {
        let field = PrimeField::new(modulus)
            .unwrap_or_else(|error| panic!("rejected prime {modulus}: {error}"));
        assert_eq!(field.modulus(), modulus);
    }
}

#[test]
fn rejects_non_primes_including_pseudoprimes() {
    let non_primes = [
        0,
        1,
        4,
        6,
        9,
        15,
        341,
        561,
        1_105,
        1_729,
        2_465,
        2_821,
        6_601,
        3_215_031_751,
        4_294_967_293,
        u32::MAX,
    ];

    for modulus in non_primes {
        assert!(
            PrimeField::new(modulus).is_err(),
            "accepted composite modulus {modulus}"
        );
    }
}

#[test]
fn reports_two_adicity() {
    let cases = [(2, 0), (3, 1), (17, 4), (65_537, 16), (998_244_353, 23)];

    for (modulus, expected) in cases {
        let field = PrimeField::new(modulus).unwrap();
        assert_eq!(field.two_adicity(), expected, "modulus {modulus}");
    }
}
