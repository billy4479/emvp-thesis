use super::*;

#[test]
#[cfg(target_arch = "x86_64")]
fn forced_avx2_matches_forced_scalar() {
    fn check<const MODULUS: u32>() {
        let scalar = NttPlan::<MODULUS>::new_scalar(256).unwrap();
        let avx2 = NttPlan::<MODULUS>::new_avx2(256).unwrap();
        let boundaries = [0, 1, MODULUS / 2, MODULUS - 2, MODULUS - 1];
        let input: Vec<_> = (0_u64..256)
            .map(|index| {
                if index < boundaries.len() as u64 {
                    boundaries[index as usize]
                } else {
                    ((index * 2_654_435_761 + 97) % u64::from(MODULUS)) as u32
                }
            })
            .collect();
        let mut scalar_values = scalar.elements(&input);
        let mut avx2_values = avx2.elements(&input);
        scalar.forward(&mut scalar_values).unwrap();
        avx2.forward(&mut avx2_values).unwrap();
        assert_eq!(avx2_values, scalar_values);

        let scalar_rhs = scalar_values.clone();
        let avx2_rhs = avx2_values.clone();
        scalar
            .pointwise_mul_assign(&mut scalar_values, &scalar_rhs)
            .unwrap();
        avx2.pointwise_mul_assign(&mut avx2_values, &avx2_rhs)
            .unwrap();
        assert_eq!(avx2_values, scalar_values);
        scalar.inverse(&mut scalar_values).unwrap();
        avx2.inverse(&mut avx2_values).unwrap();
        assert_eq!(avx2_values, scalar_values);
    }

    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }
    check::<998_244_353>();
    check::<2_013_265_921>();
}

#[test]
#[cfg(target_arch = "x86_64")]
fn avx2_lazy_boundaries_match_scalar_and_normalize_output() {
    const MODULUS: u32 = 998_244_353;

    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }
    let scalar = NttPlan::<MODULUS>::new_scalar(256).unwrap();
    let avx2 = NttPlan::<MODULUS>::new_avx2(256).unwrap();
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
    let mut scalar_values = vec![scalar.field.element(0); 256];
    for (index, value) in scalar_values.iter_mut().enumerate() {
        state ^= state << 7;
        state ^= state >> 9;
        let raw = if index < boundaries.len() {
            boundaries[index]
        } else {
            (state % u64::from(4 * MODULUS)) as u32
        };
        value.set_montgomery(raw);
    }
    let mut avx2_values = scalar_values.clone();
    scalar.forward(&mut scalar_values).unwrap();
    avx2.forward(&mut avx2_values).unwrap();
    assert_eq!(avx2_values, scalar_values);
    assert!(avx2_values.iter().all(|value| value.montgomery() < MODULUS));
    scalar.inverse(&mut scalar_values).unwrap();
    avx2.inverse(&mut avx2_values).unwrap();
    assert_eq!(avx2_values, scalar_values);
    assert!(avx2_values.iter().all(|value| value.montgomery() < MODULUS));
}

#[test]
#[cfg(target_arch = "x86_64")]
fn avx2_lazy_inverse_boundaries_match_scalar_and_normalize_output() {
    const MODULUS: u32 = 998_244_353;

    if !std::arch::is_x86_feature_detected!("avx2") {
        return;
    }
    let scalar = NttPlan::<MODULUS>::new_scalar(256).unwrap();
    let avx2 = NttPlan::<MODULUS>::new_avx2(256).unwrap();
    // Endpoints of the [0, 2p) inverse lazy interval, seeded directly
    // into the inverse transform without a prior forward pass.
    let boundaries = [0, 1, MODULUS - 1, MODULUS, 2 * MODULUS - 2, 2 * MODULUS - 1];
    let mut state = 0x9e37_79b9_7f4a_7c15u64;
    let mut scalar_values = vec![scalar.field.element(0); 256];
    for (index, value) in scalar_values.iter_mut().enumerate() {
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
    let mut avx2_values = scalar_values.clone();
    scalar.inverse(&mut scalar_values).unwrap();
    avx2.inverse(&mut avx2_values).unwrap();
    assert_eq!(avx2_values, scalar_values);
    assert!(avx2_values.iter().all(|value| value.montgomery() < MODULUS));
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

    check::<998_244_353>();
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
