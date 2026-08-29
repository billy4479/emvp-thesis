use prime_field_layer::{NegacyclicPlan, NttPlan, linear_convolution};

const fn oracle_pow(base: u32, mut exponent: usize, modulus: u32) -> u32 {
    let mut base = base as u64;
    let mut result = 1u64;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = result * base % modulus as u64;
        }
        base = base * base % modulus as u64;
        exponent >>= 1;
    }
    result as u32
}

pub fn oracle_dft<const MODULUS: u32>(input: &[u32], root: u32) -> Vec<u32> {
    (0..input.len())
        .map(|frequency| {
            input.iter().enumerate().fold(0u64, |sum, (index, &value)| {
                (sum + u64::from(value) * u64::from(oracle_pow(root, index * frequency, MODULUS)))
                    % u64::from(MODULUS)
            }) as u32
        })
        .collect()
}

pub const fn bit_reverse(value: usize, bits: u32) -> usize {
    if bits == 0 {
        0
    } else {
        value.reverse_bits() >> (usize::BITS - bits)
    }
}

pub fn oracle_linear<const MODULUS: u32>(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    if lhs.is_empty() || rhs.is_empty() {
        return Vec::new();
    }
    let mut result = vec![0u32; lhs.len() + rhs.len() - 1];
    for (lhs_index, &lhs) in lhs.iter().enumerate() {
        for (rhs_index, &rhs) in rhs.iter().enumerate() {
            let index = lhs_index + rhs_index;
            result[index] = ((u128::from(result[index]) + u128::from(lhs) * u128::from(rhs))
                % u128::from(MODULUS)) as u32;
        }
    }
    result
}

pub fn oracle_cyclic<const MODULUS: u32>(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    let mut result = vec![0u32; lhs.len()];
    for (lhs_index, &lhs_value) in lhs.iter().enumerate() {
        for (rhs_index, &rhs_value) in rhs.iter().enumerate() {
            let index = (lhs_index + rhs_index) % lhs.len();
            result[index] = ((u64::from(result[index])
                + u64::from(lhs_value) * u64::from(rhs_value))
                % u64::from(MODULUS)) as u32;
        }
    }
    result
}

pub fn oracle_negacyclic<const MODULUS: u32>(lhs: &[u32], rhs: &[u32]) -> Vec<u32> {
    let mut result = vec![0i128; lhs.len()];
    for (lhs_index, &lhs_value) in lhs.iter().enumerate() {
        for (rhs_index, &rhs_value) in rhs.iter().enumerate() {
            let degree = lhs_index + rhs_index;
            let product = i128::from(lhs_value) * i128::from(rhs_value);
            if degree < lhs.len() {
                result[degree] += product;
            } else {
                result[degree - lhs.len()] -= product;
            }
        }
    }
    result
        .into_iter()
        .map(|value| value.rem_euclid(i128::from(MODULUS)) as u32)
        .collect()
}

pub fn check_round_trip<const MODULUS: u32>(length: usize) {
    let plan = NttPlan::<MODULUS>::new(length).unwrap();
    let input: Vec<_> = (0..length)
        .map(|index| ((index as u64 * 2_654_435_761 + 97) % u64::from(MODULUS)) as u32)
        .collect();
    let mut values = plan.elements(&input);
    plan.forward(&mut values).unwrap();
    plan.inverse(&mut values).unwrap();
    assert_eq!(
        values
            .into_iter()
            .map(prime_field_layer::FieldElement::value)
            .collect::<Vec<_>>(),
        input
    );
}

pub fn check_convolutions<const MODULUS: u32>() {
    let boundaries = [0, 1, MODULUS / 2, MODULUS - 2, MODULUS - 1, 7, 11, 3];
    let rhs = [MODULUS - 1, 0, 2, MODULUS / 2, 5, 1, 9, 4];

    let cyclic = NttPlan::<MODULUS>::new(8).unwrap();
    assert_eq!(
        cyclic.cyclic_convolution(&boundaries, &rhs).unwrap(),
        oracle_cyclic::<MODULUS>(&boundaries, &rhs)
    );

    let negacyclic = NegacyclicPlan::<MODULUS>::new(8).unwrap();
    assert_eq!(
        negacyclic.convolution(&boundaries, &rhs).unwrap(),
        oracle_negacyclic::<MODULUS>(&boundaries, &rhs)
    );

    let lhs = &boundaries[..5];
    let rhs = &rhs[..4];
    assert_eq!(
        linear_convolution::<MODULUS>(lhs, rhs).unwrap(),
        oracle_linear::<MODULUS>(lhs, rhs)
    );
}
