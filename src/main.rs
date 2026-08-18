use std::error::Error;

use prime_field_layer::{FieldElement, PrimeField};

fn main() -> Result<(), Box<dyn Error>> {
    let field = PrimeField::new(2147352577)?;

    let lol = FieldElement::new(&field, u64::MAX);
    println!("{lol}");

    Ok(())
}
