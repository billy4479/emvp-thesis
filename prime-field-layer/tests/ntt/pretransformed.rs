use prime_field_layer::{FieldElement, FieldError, NttPlan, PrimeField};

use crate::support::{oracle_cyclic, oracle_linear};

fn values<const MODULUS: u32>(length: usize, offset: u64) -> Vec<u32> {
    (0..length)
        .map(|index| ((index as u64 * 2_654_435_761 + offset) % u64::from(MODULUS)) as u32)
        .collect()
}

// Converts coefficients to Montgomery form, zero-padded to the plan length.
fn elements<const MODULUS: u32>(
    plan: &NttPlan<MODULUS>,
    coefficients: &[u32],
) -> Vec<FieldElement<MODULUS>> {
    let field = PrimeField::<MODULUS>::new();
    let mut padded = vec![field.element_u32(0); plan.len()];
    for (slot, &value) in padded.iter_mut().zip(coefficients) {
        *slot = field.element_u32(value);
    }
    padded
}

#[test]
fn pretransformed_linear_operand_matches_convenience_api_and_reuses_scratch() {
    let plan = NttPlan::<1_073_479_681>::new(16).unwrap();
    let fixed = [7, 11, 13, 17, 19];
    let prepared = plan.pretransform_linear_operand(&fixed).unwrap();
    let mut workspace = prepared.workspace();

    let first = [23, 29, 31, 37];
    let mut first_output = vec![0; fixed.len() + first.len() - 1];
    prepared
        .convolve(&first, &mut first_output, &mut workspace)
        .unwrap();
    assert_eq!(
        first_output,
        plan.linear_convolution(&fixed, &first).unwrap()
    );

    let second = [41, 43];
    let mut second_output = vec![0; fixed.len() + second.len() - 1];
    prepared
        .convolve(&second, &mut second_output, &mut workspace)
        .unwrap();
    assert_eq!(second_output, oracle_linear::<1_073_479_681>(&fixed, &second));
}

#[test]
fn pretransformed_linear_operand_handles_rectangles_and_boundary_coefficients() {
    const MODULUS: u32 = 2_281_701_377;
    let plan = NttPlan::<MODULUS>::new(16).unwrap();
    let fixed = [MODULUS - 1, 1, 0, u32::MAX, MODULUS, MODULUS + 1, 7];
    let input = [u32::MAX, 0, MODULUS - 1];
    let prepared = plan.pretransform_linear_operand(&fixed).unwrap();
    let mut workspace = prepared.workspace();
    let mut output = [0; 9];

    prepared
        .convolve(&input, &mut output, &mut workspace)
        .unwrap();
    assert_eq!(output.as_slice(), oracle_linear::<MODULUS>(&fixed, &input));
    assert_eq!(
        output[0],
        (u64::from(fixed[0]) * u64::from(input[0]) % u64::from(MODULUS)) as u32
    );
    assert_eq!(
        output[output.len() - 1],
        (u64::from(fixed[fixed.len() - 1]) * u64::from(input[input.len() - 1]) % u64::from(MODULUS))
            as u32
    );
}

#[test]
fn pretransformed_linear_operand_preserves_empty_semantics_and_reports_errors() {
    let plan = NttPlan::<17>::new(8).unwrap();
    let prepared = plan.pretransform_linear_operand(&[1, 2, 3, 4, 5]).unwrap();
    let mut workspace = prepared.workspace();
    let mut wrong_output = [99; 5];
    assert_eq!(
        prepared.convolve(&[6, 7], &mut wrong_output, &mut workspace),
        Err(FieldError::LengthMismatch)
    );
    assert_eq!(wrong_output, [99; 5]);

    let mut too_large_output = [88; 9];
    assert_eq!(
        prepared.convolve(&[1; 5], &mut too_large_output, &mut workspace),
        Err(FieldError::PlanTooSmall {
            required: 16,
            available: 8,
        })
    );
    assert_eq!(too_large_output, [88; 9]);
    assert_eq!(
        prepared.output_len(usize::MAX),
        Err(FieldError::ConvolutionLengthOverflow)
    );
    assert_eq!(
        plan.pretransform_linear_operand(&[1; 9]).err(),
        Some(FieldError::PlanTooSmall {
            required: 16,
            available: 8,
        })
    );

    let larger_plan = NttPlan::<17>::new(16).unwrap();
    let larger_prepared = larger_plan.pretransform_linear_operand(&[1]).unwrap();
    let mut wrong_workspace = larger_prepared.workspace();
    let mut valid_output = [77; 6];
    assert_eq!(
        prepared.convolve(&[6, 7], &mut valid_output, &mut wrong_workspace),
        Err(FieldError::LengthMismatch)
    );
    assert_eq!(valid_output, [77; 6]);

    let empty = plan.pretransform_linear_operand(&[]).unwrap();
    let mut empty_workspace = empty.workspace();
    assert_eq!(empty.output_len(usize::MAX).unwrap(), 0);
    empty
        .convolve(&[1, 2, 3], &mut [], &mut empty_workspace)
        .unwrap();
    prepared.convolve(&[], &mut [], &mut workspace).unwrap();
}

#[test]
fn pretransformed_kernel_computes_linear_products_and_zeroes_stale_tail() {
    const MODULUS: u32 = 1_073_479_681;
    let plan = NttPlan::<MODULUS>::new(16).unwrap();
    let field = PrimeField::<MODULUS>::new();
    let fixed = values::<MODULUS>(7, 97);
    let input = values::<MODULUS>(5, 12_345);

    let mut spectrum = elements(&plan, &fixed);
    plan.forward(&mut spectrum).unwrap();

    // Stale garbage beyond the input must never reach the transform.
    let mut values = elements(&plan, &input);
    for slot in &mut values[input.len()..] {
        *slot = field.element_u32(u32::MAX);
    }
    plan.convolve_pretransformed_assign(&spectrum, input.len(), &mut values)
        .unwrap();

    let expected = oracle_linear::<MODULUS>(&fixed, &input);
    assert_eq!(
        values[..expected.len()]
            .iter()
            .map(|&value| value.value())
            .collect::<Vec<_>>(),
        expected
    );
}

#[test]
fn pretransformed_kernel_matches_cyclic_semantics_for_full_length_inputs() {
    const MODULUS: u32 = 2_281_701_377;
    let plan = NttPlan::<MODULUS>::new(8).unwrap();
    let fixed = values::<MODULUS>(8, 97);
    let input = values::<MODULUS>(8, 12_345);
    let mut spectrum = elements(&plan, &fixed);
    plan.forward(&mut spectrum).unwrap();
    let mut values = elements(&plan, &input);
    plan.convolve_pretransformed_assign(&spectrum, input.len(), &mut values)
        .unwrap();
    let observed = values
        .iter()
        .map(|&value| value.value())
        .collect::<Vec<_>>();
    assert_eq!(observed, oracle_cyclic::<MODULUS>(&fixed, &input));
}

#[test]
fn pretransformed_kernel_supports_wrapped_toeplitz_packing() {
    const MODULUS: u32 = 1_073_479_681;
    let (rows, columns) = (5_usize, 4_usize);
    let diagonals = values::<MODULUS>(rows + columns - 1, 97);
    let input = values::<MODULUS>(columns, 12_345);
    let transform_length = 8;
    let plan = NttPlan::<MODULUS>::new(transform_length).unwrap();

    // The fast Toeplitz packing: reversed head diagonals, then the negative
    // diagonals wrapped onto the tail so the cyclic product folds them back.
    let mut packed = vec![0; transform_length];
    for offset in 0..rows {
        packed[offset] = diagonals[rows - 1 - offset];
    }
    for offset in 0..columns - 1 {
        packed[transform_length - 1 - offset] = diagonals[rows + offset];
    }
    let mut spectrum = elements(&plan, &packed);
    plan.forward(&mut spectrum).unwrap();

    let mut values = elements(&plan, &input);
    plan.convolve_pretransformed_assign(&spectrum, columns, &mut values)
        .unwrap();

    let field = PrimeField::<MODULUS>::new();
    for row in 0..rows {
        let mut expected = field.element_u32(0);
        for (column, &value) in input.iter().enumerate() {
            expected +=
                field.element_u32(diagonals[rows - 1 - row + column]) * field.element_u32(value);
        }
        assert_eq!(values[row].value(), expected.value(), "row {row}");
    }
}

#[test]
fn pretransformed_kernel_rejects_mismatched_slices_before_mutation() {
    const MODULUS: u32 = 17;
    let plan = NttPlan::<MODULUS>::new(8).unwrap();
    let field = PrimeField::<MODULUS>::new();
    let good_spectrum = vec![field.element_u32(1); 8];
    let mut values = vec![field.element_u32(3); 8];

    let short_spectrum = vec![field.element_u32(1); 7];
    assert_eq!(
        plan.convolve_pretransformed_assign(&short_spectrum, 4, &mut values),
        Err(FieldError::LengthMismatch)
    );
    assert_eq!(
        plan.convolve_pretransformed_assign(&good_spectrum, 9, &mut values),
        Err(FieldError::LengthMismatch)
    );
    let short_values = &mut values[..7];
    assert_eq!(
        plan.convolve_pretransformed_assign(&good_spectrum, 4, short_values),
        Err(FieldError::LengthMismatch)
    );
    assert!(values.iter().all(|&value| value == field.element_u32(3)));
}
