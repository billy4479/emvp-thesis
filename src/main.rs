use std::error::Error;

use prime_field_layer::PrimeField;

fn main() -> Result<(), Box<dyn Error>> {
    let field = PrimeField::new(2147352577)?;

    let lol = field.reduce_u64(u64::MAX);
    println!("{lol}");

    Ok(())
}
