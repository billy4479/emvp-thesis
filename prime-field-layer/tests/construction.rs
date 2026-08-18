use prime_field_layer::PrimeField;

fn accepts<const MODULUS: u32>() {
    let field = PrimeField::<MODULUS>::new();
    assert_eq!(field.modulus(), MODULUS);
}

#[test]
fn accepts_representative_primes() {
    accepts::<2>();
    accepts::<3>();
    accepts::<5>();
    accepts::<17>();
    accepts::<257>();
    accepts::<65_537>();
    accepts::<998_244_353>();
    accepts::<2_147_483_647>();
    accepts::<4_294_967_291>();
}

fn check_two_adicity<const MODULUS: u32>(expected: u32) {
    let field = PrimeField::<MODULUS>::new();
    assert_eq!(field.two_adicity(), expected, "modulus {MODULUS}");
}

#[test]
fn reports_two_adicity() {
    check_two_adicity::<2>(0);
    check_two_adicity::<3>(1);
    check_two_adicity::<17>(4);
    check_two_adicity::<65_537>(16);
    check_two_adicity::<998_244_353>(23);
}
