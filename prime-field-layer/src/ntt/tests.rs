use super::*;

#[test]
fn lazy_boundaries_round_trip_and_normalize_output() {
    const MODULUS: u32 = 1_073_479_681;

    let plan = NttPlan::<MODULUS>::new_scalar(256).unwrap();
    // Every endpoint of the [0, 4p) forward lazy interval, plus the
    // interior boundaries at p and 3p.
    let boundaries = [
        0,
        1,
        MODULUS - 1,
        MODULUS,
        2 * MODULUS - 2,
        2 * MODULUS - 1,
        2 * MODULUS,
        3 * MODULUS - 1,
        4 * MODULUS - 2,
        4 * MODULUS - 1,
    ];
    let mut state = 0xd1b5_4a32_d192_ed03u64;
    let mut values = vec![plan.field.element(0); 256];
    for (index, value) in values.iter_mut().enumerate() {
        state ^= state << 7;
        state ^= state >> 9;
        let raw = if index < boundaries.len() {
            boundaries[index]
        } else {
            (state % u64::from(4 * MODULUS)) as u32
        };
        value.set_montgomery(raw);
    }
    // forward∘inverse is the identity on field elements, so a planted raw
    // Montgomery word `w` (representing `w * R^-1 mod p`) comes back as the
    // canonical word `w mod p`.
    let expected: Vec<_> = values.iter().map(|v| v.montgomery() % MODULUS).collect();
    plan.forward(&mut values).unwrap();
    assert!(values.iter().all(|value| value.montgomery() < MODULUS));
    plan.inverse(&mut values).unwrap();
    assert_eq!(
        values.iter().map(|v| v.montgomery()).collect::<Vec<_>>(),
        expected
    );
}

#[test]
fn lazy_inverse_boundaries_round_trip_and_normalize_output() {
    const MODULUS: u32 = 1_073_479_681;

    let plan = NttPlan::<MODULUS>::new_scalar(256).unwrap();
    // Endpoints of the [0, 2p) inverse lazy interval, seeded directly
    // into the inverse transform without a prior forward pass.
    let boundaries = [0, 1, MODULUS - 1, MODULUS, 2 * MODULUS - 2, 2 * MODULUS - 1];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut values = vec![plan.field.element(0); 256];
    for (index, value) in values.iter_mut().enumerate() {
        state ^= state << 7;
        state ^= state >> 9;
        state ^= state << 8;
        let raw = if index < boundaries.len() {
            boundaries[index]
        } else {
            (state % u64::from(2 * MODULUS)) as u32
        };
        value.set_montgomery(raw);
    }
    // The first inverse stage consumes the planted lazy words directly from
    // `[0, 2p)`, then forward∘inverse returns the identity, so a raw
    // Montgomery word `w` (representing `w * R^-1 mod p`) comes back as the
    // canonical word `w mod p`.
    let expected: Vec<_> = values.iter().map(|v| v.montgomery() % MODULUS).collect();
    plan.inverse(&mut values).unwrap();
    assert!(values.iter().all(|value| value.montgomery() < MODULUS));
    plan.forward(&mut values).unwrap();
    assert_eq!(
        values.iter().map(|v| v.montgomery()).collect::<Vec<_>>(),
        expected
    );
}

#[test]
fn transform_outputs_have_canonical_montgomery_words() {
    fn check<const MODULUS: u32>() {
        let plan = NttPlan::<MODULUS>::new_scalar(256).unwrap();
        let input: Vec<_> = (0_u64..256)
            .map(|index| ((index * 1_103_515_245 + 12_345) % u64::from(MODULUS)) as u32)
            .collect();
        let mut values = plan.elements(&input);
        plan.forward(&mut values).unwrap();
        assert!(values.iter().all(|value| value.montgomery() < MODULUS));
        plan.inverse(&mut values).unwrap();
        assert!(values.iter().all(|value| value.montgomery() < MODULUS));
    }

    check::<1_073_479_681>();
    check::<2_013_265_921>();
    check::<2_281_701_377>();
}

#[test]
fn incremental_twiddle_table_preserves_stage_mapping() {
    let field = PrimeField::<65_537>::new();
    let plan = NttPlan::<65_537>::new_scalar(256).unwrap();
    let root = field.root_of_unity(256).unwrap();
    for stage in plan.stages.iter() {
        let bits = stage.forward.len().trailing_zeros();
        for (block, twiddle) in stage.forward.iter().enumerate() {
            let reversed = if bits == 0 {
                0
            } else {
                block.reverse_bits() >> (usize::BITS - bits)
            };
            assert_eq!(
                twiddle.canonical,
                field.pow(root, (reversed * stage.distance) as u64)
            );
        }
    }
}

#[test]
fn convolution_length_overflow_has_a_distinct_error() {
    assert_eq!(
        convolution::convolution_result_length(usize::MAX, 2),
        Err(FieldError::ConvolutionLengthOverflow)
    );
}
