use crate::PrimeField;

/// Rabin's deterministic irreducibility criterion.
///
/// For every prime divisor `r` of `n`, an irreducible degree-`n` polynomial `f`
/// satisfies `gcd(f, X^(q^(n/r)) - X) = 1`, and it also satisfies
/// `X^(q^n) - X = 0 mod f`. Together these conditions are necessary and
/// sufficient. See Rabin, SIAM J. Comput. 9.2 (1980), 273-280.
pub(super) fn is_irreducible<const MODULUS: u32>(modulus: &[u32]) -> bool {
    let degree = modulus.len() - 1;
    let field = PrimeField::<MODULUS>::new();
    let x = remainder(field, vec![0, 1], modulus);
    let checkpoints: Vec<_> = prime_divisors(degree)
        .into_iter()
        .map(|divisor| degree / divisor)
        .collect();
    let mut frobenius = x.clone();

    for iteration in 1..=degree {
        frobenius = pow_mod(field, &frobenius, u64::from(MODULUS), modulus);
        if checkpoints.contains(&iteration) {
            let difference = sub(field, &frobenius, &x);
            if gcd(field, modulus.to_vec(), difference).len() != 1 {
                return false;
            }
        }
    }
    trim(frobenius) == trim(x)
}

fn prime_divisors(mut value: usize) -> Vec<usize> {
    let mut divisors = Vec::new();
    let mut candidate = 2;
    while candidate <= value / candidate {
        if value.is_multiple_of(candidate) {
            divisors.push(candidate);
            while value.is_multiple_of(candidate) {
                value /= candidate;
            }
        }
        candidate += usize::from(candidate == 2) + 2 * usize::from(candidate != 2);
    }
    if value > 1 {
        divisors.push(value);
    }
    divisors
}

fn pow_mod<const MODULUS: u32>(
    field: PrimeField<MODULUS>,
    base: &[u32],
    mut exponent: u64,
    modulus: &[u32],
) -> Vec<u32> {
    let mut result = vec![1];
    let mut base = base.to_vec();
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = mul_mod(field, &result, &base, modulus);
        }
        exponent >>= 1;
        if exponent != 0 {
            base = mul_mod(field, &base, &base, modulus);
        }
    }
    result.resize(modulus.len() - 1, 0);
    result
}

fn mul_mod<const MODULUS: u32>(
    field: PrimeField<MODULUS>,
    lhs: &[u32],
    rhs: &[u32],
    modulus: &[u32],
) -> Vec<u32> {
    let mut product = vec![0; lhs.len() + rhs.len() - 1];
    for (lhs_index, &lhs) in lhs.iter().enumerate() {
        for (rhs_index, &rhs) in rhs.iter().enumerate() {
            let index = lhs_index + rhs_index;
            product[index] = field.add_canonical(product[index], field.mul(lhs, rhs));
        }
    }
    remainder(field, product, modulus)
}

fn remainder<const MODULUS: u32>(
    field: PrimeField<MODULUS>,
    mut dividend: Vec<u32>,
    divisor: &[u32],
) -> Vec<u32> {
    trim_in_place(&mut dividend);
    let divisor_degree = divisor.len() - 1;
    while dividend.len() > divisor_degree {
        let degree = dividend.len() - divisor.len();
        let factor = *dividend.last().unwrap_or(&0);
        for (index, &coefficient) in divisor.iter().enumerate() {
            let target = degree + index;
            dividend[target] =
                field.sub_canonical(dividend[target], field.mul(factor, coefficient));
        }
        trim_in_place(&mut dividend);
    }
    dividend.resize(divisor_degree, 0);
    dividend
}

fn gcd<const MODULUS: u32>(
    field: PrimeField<MODULUS>,
    mut lhs: Vec<u32>,
    mut rhs: Vec<u32>,
) -> Vec<u32> {
    trim_in_place(&mut lhs);
    trim_in_place(&mut rhs);
    while !rhs.is_empty() {
        let remainder = general_remainder(field, lhs, &rhs);
        lhs = rhs;
        rhs = remainder;
    }
    if let Some(&leading) = lhs.last() {
        let inverse = field.inv(leading).unwrap_or(0);
        for coefficient in &mut lhs {
            *coefficient = field.mul(*coefficient, inverse);
        }
    }
    trim(lhs)
}

fn general_remainder<const MODULUS: u32>(
    field: PrimeField<MODULUS>,
    mut dividend: Vec<u32>,
    divisor: &[u32],
) -> Vec<u32> {
    trim_in_place(&mut dividend);
    let inverse_leading = field.inv(*divisor.last().unwrap_or(&0)).unwrap_or(0);
    while dividend.len() >= divisor.len() {
        let degree = dividend.len() - divisor.len();
        let factor = field.mul(*dividend.last().unwrap_or(&0), inverse_leading);
        for (index, &coefficient) in divisor.iter().enumerate() {
            let target = degree + index;
            dividend[target] =
                field.sub_canonical(dividend[target], field.mul(factor, coefficient));
        }
        trim_in_place(&mut dividend);
    }
    dividend
}

fn sub<const MODULUS: u32>(field: PrimeField<MODULUS>, lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    let length = lhs.len().max(rhs.len());
    let mut result = vec![0; length];
    for (index, output) in result.iter_mut().enumerate() {
        *output = field.sub_canonical(
            lhs.get(index).copied().unwrap_or(0),
            rhs.get(index).copied().unwrap_or(0),
        );
    }
    trim(result)
}

fn trim(mut polynomial: Vec<u32>) -> Vec<u32> {
    trim_in_place(&mut polynomial);
    polynomial
}

fn trim_in_place(polynomial: &mut Vec<u32>) {
    while polynomial.last() == Some(&0) {
        let _ = polynomial.pop();
    }
}
