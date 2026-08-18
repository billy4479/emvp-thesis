use prime_field_layer::PrimeField;

fn main() {
    let field = PrimeField::<2_147_352_577>::new();

    let lol = field.element(u64::MAX);
    println!("{lol}");
}
