use super::{Twiddle, stage_twiddle_index};

pub(super) const fn pow_mod(base: u32, mut exponent: u64, modulus: u32) -> u32 {
    let mut result = 1u64;
    let mut base_u64 = base as u64;
    while exponent != 0 {
        if exponent & 1 == 1 {
            result = result * base_u64 % modulus as u64;
        }
        exponent >>= 1;
        if exponent != 0 {
            base_u64 = base_u64 * base_u64 % modulus as u64;
        }
    }
    result as u32
}

const fn two_adic_root<const MODULUS: u32>() -> u32 {
    if MODULUS == 2 {
        return 1;
    }
    if (MODULUS - 1).trailing_zeros() == 1 {
        return MODULUS - 1;
    }
    let mut non_residue = 2;
    while pow_mod(non_residue, ((MODULUS - 1) / 2) as u64, MODULUS) == 1 {
        non_residue += 1;
    }
    pow_mod(
        non_residue,
        ((MODULUS - 1) >> (MODULUS - 1).trailing_zeros()) as u64,
        MODULUS,
    )
}

pub(super) const fn root_of_unity<const MODULUS: u32, const N: usize>() -> u32 {
    let mut root = two_adic_root::<MODULUS>();
    let mut shifts = N.trailing_zeros();
    while shifts < (MODULUS - 1).trailing_zeros() {
        root = (root as u64 * root as u64 % MODULUS as u64) as u32;
        shifts += 1;
    }
    root
}

pub(super) const fn to_montgomery_const<const MODULUS: u32>(value: u32) -> u32 {
    if MODULUS == 2 {
        return value & 1;
    }
    ((value as u128 * (1u128 << 32)) % MODULUS as u128) as u32
}

const fn twiddle_const<const MODULUS: u32>(canonical: u32) -> Twiddle {
    Twiddle {
        canonical,
        shoup: ((canonical as u64) << 32).div_euclid(MODULUS as u64) as u32,
        montgomery: to_montgomery_const::<MODULUS>(canonical),
    }
}

pub(super) const fn stage_twiddles<const MODULUS: u32, const N: usize>(root: u32) -> [Twiddle; N] {
    let one = twiddle_const::<MODULUS>(1);
    let mut powers = [one; N];
    let mut canonical = 1;
    let mut index = 0;
    while index < N {
        powers[index] = twiddle_const::<MODULUS>(canonical);
        canonical = (canonical as u64 * root as u64 % MODULUS as u64) as u32;
        index += 1;
    }

    let mut result = [one; N];
    let mut distance = N / 2;
    while distance != 0 {
        let blocks = N / (2 * distance);
        let bits = blocks.trailing_zeros();
        let mut block = 0;
        while block < blocks {
            let reversed = if bits == 0 {
                0
            } else {
                block.reverse_bits() >> (usize::BITS - bits)
            };
            result[stage_twiddle_index(blocks, block)] = powers[reversed * distance];
            block += 1;
        }
        distance /= 2;
    }
    result
}
