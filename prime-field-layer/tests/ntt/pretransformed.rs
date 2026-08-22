use prime_field_layer::{FieldError, NttPlan};

use crate::support::oracle_linear;

#[test]
fn pretransformed_linear_operand_matches_convenience_api_and_reuses_scratch() {
    let plan = NttPlan::<998_244_353>::new(16).unwrap();
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
    assert_eq!(second_output, oracle_linear::<998_244_353>(&fixed, &second));
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
