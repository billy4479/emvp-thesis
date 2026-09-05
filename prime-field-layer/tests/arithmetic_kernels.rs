#![expect(
    clippy::unwrap_used,
    reason = "test inputs establish that these operations must succeed"
)]

use prime_field_layer::{
    ArithmeticKernelError, FieldElement, IndexedValue, PrimeField, axpy_assign,
    batched_weighted_inclusive_scan_assign, dot_product, sparse_accumulate,
    weighted_inclusive_scan_assign,
};

const LENGTHS: [usize; 9] = [0, 1, 2, 7, 8, 9, 31, 64, 67];

fn canonical_inputs<const MODULUS: u32>(length: usize, offset: u64) -> Vec<u32> {
    let boundaries = [0, 1, MODULUS / 2, MODULUS.saturating_sub(2), MODULUS - 1];
    (0..length)
        .map(|index| {
            if index < boundaries.len() {
                boundaries[index]
            } else {
                ((index as u64 * 2_654_435_761 + offset) % u64::from(MODULUS)) as u32
            }
        })
        .collect()
}

fn elements<const MODULUS: u32>(values: &[u32]) -> Vec<FieldElement<MODULUS>> {
    let field = PrimeField::<MODULUS>::new();
    values
        .iter()
        .map(|&value| field.element_u32(value))
        .collect()
}

fn reference_add<const MODULUS: u32>(lhs: u32, rhs: u32) -> u32 {
    ((u64::from(lhs) + u64::from(rhs)) % u64::from(MODULUS)) as u32
}

fn reference_mul<const MODULUS: u32>(lhs: u32, rhs: u32) -> u32 {
    (u64::from(lhs) * u64::from(rhs) % u64::from(MODULUS)) as u32
}

fn check_dense_kernels<const MODULUS: u32>() {
    for length in LENGTHS {
        let lhs = canonical_inputs::<MODULUS>(length, 11);
        let rhs = canonical_inputs::<MODULUS>(length, 97);
        let lhs_elements = elements::<MODULUS>(&lhs);
        let rhs_elements = elements::<MODULUS>(&rhs);

        let expected_dot = lhs.iter().zip(&rhs).fold(0, |sum, (&lhs, &rhs)| {
            reference_add::<MODULUS>(sum, reference_mul::<MODULUS>(lhs, rhs))
        });
        assert_eq!(
            dot_product(&lhs_elements, &rhs_elements).unwrap().value(),
            expected_dot,
            "dot, modulus {MODULUS}, length {length}"
        );

        for scalar in [0, 1, MODULUS / 2, MODULUS - 1] {
            let mut actual = lhs_elements.clone();
            axpy_assign(
                &mut actual,
                PrimeField::<MODULUS>::new().element_u32(scalar),
                &rhs_elements,
            )
            .unwrap();
            let expected: Vec<_> = lhs
                .iter()
                .zip(&rhs)
                .map(|(&lhs, &rhs)| {
                    reference_add::<MODULUS>(lhs, reference_mul::<MODULUS>(scalar, rhs))
                })
                .collect();
            assert_eq!(
                actual
                    .iter()
                    .copied()
                    .map(FieldElement::value)
                    .collect::<Vec<_>>(),
                expected,
                "axpy, modulus {MODULUS}, scalar {scalar}, length {length}"
            );
        }

        let mut actual = lhs_elements;
        weighted_inclusive_scan_assign(&mut actual, &rhs_elements).unwrap();
        let mut sum = 0;
        let expected: Vec<_> = lhs
            .iter()
            .zip(&rhs)
            .map(|(&value, &weight)| {
                sum = reference_add::<MODULUS>(sum, reference_mul::<MODULUS>(value, weight));
                sum
            })
            .collect();
        assert_eq!(
            actual
                .iter()
                .copied()
                .map(FieldElement::value)
                .collect::<Vec<_>>(),
            expected,
            "scan, modulus {MODULUS}, length {length}"
        );
    }
}

#[test]
fn dense_kernels_match_canonical_references_at_boundaries_and_tails() {
    check_dense_kernels::<2>();
    check_dense_kernels::<1_073_479_681>();
    check_dense_kernels::<2_013_265_921>();
}

fn check_batched_scan<const MODULUS: u32>(positions: usize, width: usize) {
    let input = canonical_inputs::<MODULUS>(positions * width, 31);
    let weights = canonical_inputs::<MODULUS>(positions, 73);
    let mut actual = elements::<MODULUS>(&input);

    batched_weighted_inclusive_scan_assign(&mut actual, &elements(&weights), width).unwrap();

    let mut expected = vec![0; input.len()];
    for lane in 0..width {
        let mut sum = 0;
        for (position, &weight) in weights.iter().enumerate() {
            let index = position * width + lane;
            sum = reference_add::<MODULUS>(sum, reference_mul::<MODULUS>(input[index], weight));
            expected[index] = sum;
        }
    }
    assert_eq!(
        actual
            .iter()
            .copied()
            .map(FieldElement::value)
            .collect::<Vec<_>>(),
        expected,
        "batched scan, modulus {MODULUS}, positions {positions}, width {width}"
    );
}

#[test]
fn batched_scan_matches_independent_lane_references_at_boundaries() {
    check_batched_scan::<2>(7, 3);
    check_batched_scan::<1_073_479_681>(7, 3);
    check_batched_scan::<2_013_265_921>(7, 3);
}

#[test]
fn batched_scan_supports_one_lane() {
    check_batched_scan::<1_073_479_681>(9, 1);
}

#[test]
fn batched_scan_accepts_empty_positions_for_positive_width() {
    let mut values = [];
    let weights = [];

    batched_weighted_inclusive_scan_assign::<2>(&mut values, &weights, 4).unwrap();
}

#[test]
fn batched_scan_rejects_zero_width_without_mutation() {
    let mut empty = [];
    assert_eq!(
        batched_weighted_inclusive_scan_assign::<2>(&mut empty, &[], 0),
        Err(ArithmeticKernelError::ZeroBatchWidth)
    );

    let original = elements::<1_073_479_681>(&[1, 2]);
    let mut values = original.clone();

    assert_eq!(
        batched_weighted_inclusive_scan_assign(&mut values, &[], 0),
        Err(ArithmeticKernelError::ZeroBatchWidth)
    );
    assert_eq!(values, original);
}

#[test]
fn batched_scan_shape_errors_do_not_mutate_values() {
    let original = elements::<2_013_265_921>(&[1, 2, 3, 4, 5]);
    let weights = elements::<2_013_265_921>(&[6, 7]);
    let mut values = original.clone();
    assert_eq!(
        batched_weighted_inclusive_scan_assign(&mut values, &weights, 3),
        Err(ArithmeticKernelError::LengthMismatch {
            expected: 6,
            actual: 5,
        })
    );
    assert_eq!(values, original);

    let mut values = original.clone();
    assert_eq!(
        batched_weighted_inclusive_scan_assign(&mut values, &weights, usize::MAX),
        Err(ArithmeticKernelError::ShapeOverflow {
            positions: 2,
            width: usize::MAX,
        })
    );
    assert_eq!(values, original);
}

fn check_sparse<const MODULUS: u32>() {
    let field = PrimeField::<MODULUS>::new();
    let initial = canonical_inputs::<MODULUS>(9, 42);
    let mut actual = elements::<MODULUS>(&initial);
    let raw_entries = [
        (8, 0),
        (0, 1),
        (4, MODULUS / 2),
        (8, MODULUS - 1),
        (4, MODULUS - 1),
        (0, MODULUS.saturating_sub(2)),
    ];
    let entries: Vec<_> = raw_entries
        .iter()
        .map(|&(index, value)| IndexedValue::new(index, field.element_u32(value)))
        .collect();

    sparse_accumulate(&mut actual, &entries).unwrap();
    let mut expected = initial;
    for &(index, value) in &raw_entries {
        expected[index] = reference_add::<MODULUS>(expected[index], value);
    }
    assert_eq!(
        actual
            .iter()
            .copied()
            .map(FieldElement::value)
            .collect::<Vec<_>>(),
        expected
    );

    sparse_accumulate(&mut actual, &[]).unwrap();
}

#[test]
fn sparse_accumulation_matches_reference_with_duplicates_and_boundary_indices() {
    check_sparse::<2>();
    check_sparse::<1_073_479_681>();
    check_sparse::<2_013_265_921>();
}

#[test]
fn length_errors_do_not_mutate_outputs() {
    let field = PrimeField::<1_073_479_681>::new();
    let original = elements::<1_073_479_681>(&[1, 2, 3]);
    let short = elements::<1_073_479_681>(&[4, 5]);
    let scalar = field.element_u32(7);

    assert_eq!(
        dot_product(&original, &short),
        Err(ArithmeticKernelError::LengthMismatch {
            expected: 3,
            actual: 2,
        })
    );

    let mut output = original.clone();
    assert_eq!(
        axpy_assign(&mut output, scalar, &short),
        Err(ArithmeticKernelError::LengthMismatch {
            expected: 3,
            actual: 2,
        })
    );
    assert_eq!(output, original);

    let mut output = original.clone();
    assert_eq!(
        weighted_inclusive_scan_assign(&mut output, &short),
        Err(ArithmeticKernelError::LengthMismatch {
            expected: 3,
            actual: 2,
        })
    );
    assert_eq!(output, original);
}

#[test]
fn invalid_sparse_index_reports_entry_and_does_not_mutate_output() {
    let field = PrimeField::<2_013_265_921>::new();
    let original = elements::<2_013_265_921>(&[3, 5, 7]);
    let entries = [
        IndexedValue::new(1, field.element_u32(11)),
        IndexedValue::new(3, field.element_u32(13)),
        IndexedValue::new(0, field.element_u32(17)),
    ];
    let mut output = original.clone();

    assert_eq!(
        sparse_accumulate(&mut output, &entries),
        Err(ArithmeticKernelError::IndexOutOfBounds {
            entry: 1,
            index: 3,
            output_len: 3,
        })
    );
    assert_eq!(output, original);
}

#[test]
fn sparse_entries_are_rejected_for_empty_output_before_mutation() {
    let field = PrimeField::<2>::new();
    let entries = [IndexedValue::new(0, field.element_u32(1))];
    let mut output = [];

    assert_eq!(
        sparse_accumulate(&mut output, &entries),
        Err(ArithmeticKernelError::IndexOutOfBounds {
            entry: 0,
            index: 0,
            output_len: 0,
        })
    );
}
