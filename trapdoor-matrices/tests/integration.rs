#![expect(
    clippy::unwrap_used,
    reason = "fixed test fixtures establish that construction and evaluation must succeed"
)]

use prime_field_layer::{
    ExtensionField, ExtensionFieldError, FieldElement, FieldError, PrimeField,
};
use proptest::prelude::*;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use trapdoor_matrices::{
    DenseMatrix, IrreducibleRingLpn, ParameterWarning, Permutation, RaaWeightedProduct,
    SparseMatrix, TdmError, ToeplitzFastProduct, ToeplitzMap, automatic_ring_modulus,
};

fn elements<const MODULUS: u32>(values: &[u32]) -> Vec<FieldElement<MODULUS>> {
    let field = PrimeField::<MODULUS>::new();
    values
        .iter()
        .map(|&value| field.element_u32(value))
        .collect()
}

fn values<const MODULUS: u32>(elements: &[FieldElement<MODULUS>]) -> Vec<u32> {
    elements.iter().map(|element| element.value()).collect()
}

fn add_mod(lhs: u32, rhs: u32, modulus: u32) -> u32 {
    u32::try_from((u64::from(lhs) + u64::from(rhs)) % u64::from(modulus)).unwrap()
}

fn mul_mod(lhs: u32, rhs: u32, modulus: u32) -> u32 {
    u32::try_from(u64::from(lhs) * u64::from(rhs) % u64::from(modulus)).unwrap()
}

fn dense_apply(
    rows: usize,
    columns: usize,
    matrix: &[u32],
    input: &[u32],
    modulus: u32,
) -> Vec<u32> {
    assert_eq!(matrix.len(), rows * columns);
    assert_eq!(input.len(), columns);
    matrix
        .chunks_exact(columns)
        .map(|row| {
            row.iter()
                .zip(input)
                .fold(0, |sum, (&coefficient, &value)| {
                    add_mod(
                        sum,
                        mul_mod(coefficient % modulus, value % modulus, modulus),
                        modulus,
                    )
                })
        })
        .collect()
}

fn toeplitz_oracle(
    rows: usize,
    columns: usize,
    diagonals: &[u32],
    input: &[u32],
    modulus: u32,
) -> Vec<u32> {
    assert_eq!(diagonals.len(), rows + columns - 1);
    assert_eq!(input.len(), columns);
    (0..rows)
        .map(|row| {
            (0..columns).fold(0, |sum, column| {
                let diagonal = diagonals[rows - 1 + column - row] % modulus;
                add_mod(
                    sum,
                    mul_mod(diagonal, input[column] % modulus, modulus),
                    modulus,
                )
            })
        })
        .collect()
}

fn gather(input: &[u32], indices: &[usize]) -> Vec<u32> {
    indices.iter().map(|&index| input[index]).collect()
}

fn materialize_oracle(
    dimension: usize,
    modulus: u32,
    apply: impl Fn(&[u32]) -> Vec<u32>,
) -> Vec<u32> {
    let mut matrix = vec![0; dimension * dimension];
    for column in 0..dimension {
        let mut basis = vec![0; dimension];
        basis[column] = 1;
        let image = apply(&basis);
        for (row, &entry) in image.iter().enumerate() {
            matrix[row * dimension + column] = entry % modulus;
        }
    }
    matrix
}

#[test]
fn dense_matrix_checks_layout_application_and_error_atomicity() {
    let matrix = DenseMatrix::<17>::new(2, 3, elements(&[1, 2, 3, 4, 5, 6])).unwrap();
    assert_eq!(matrix.rows(), 2);
    assert_eq!(matrix.columns(), 3);
    assert_eq!(values(matrix.values()), [1, 2, 3, 4, 5, 6]);

    let mut output = elements(&[99, 99]);
    matrix.apply(&elements(&[7, 8, 9]), &mut output).unwrap();
    assert_eq!(values(&output), [16, 3]);

    let sentinel = elements::<17>(&[11, 12]);
    let mut wrong_input_output = sentinel.clone();
    assert!(matches!(
        matrix.apply(&elements(&[1, 2]), &mut wrong_input_output),
        Err(TdmError::LengthMismatch { name: "input", .. })
    ));
    assert_eq!(wrong_input_output, sentinel);

    let mut wrong_output = elements::<17>(&[13]);
    let wrong_output_before = wrong_output.clone();
    assert!(matches!(
        matrix.apply(&elements(&[1, 2, 3]), &mut wrong_output),
        Err(TdmError::LengthMismatch { name: "output", .. })
    ));
    assert_eq!(wrong_output, wrong_output_before);
    assert!(matches!(
        DenseMatrix::<17>::new(2, 3, elements(&[1, 2])),
        Err(TdmError::LengthMismatch {
            name: "matrix values",
            expected: 6,
            actual: 2
        })
    ));
    assert_eq!(
        DenseMatrix::<17>::new(3, 0, Vec::new()),
        Err(TdmError::ZeroDimension("matrix columns"))
    );
    assert_eq!(
        DenseMatrix::<17>::new(0, 3, Vec::new()),
        Err(TdmError::ZeroDimension("matrix rows"))
    );
}

#[test]
fn permutation_uses_gather_convention_and_rejects_invalid_indices() {
    let permutation = Permutation::new(vec![2, 0, 3, 1]).unwrap();
    let mut output = [0; 4];
    permutation.apply(&[10, 20, 30, 40], &mut output).unwrap();
    assert_eq!(output, [30, 10, 40, 20]);
    assert_eq!(permutation.indices(), [2, 0, 3, 1]);
    assert_eq!(permutation.len(), 4);
    assert!(!permutation.is_empty());

    assert_eq!(
        Permutation::new(vec![0, 1, 1]),
        Err(TdmError::InvalidPermutation {
            position: 2,
            index: 1
        })
    );
    assert_eq!(
        Permutation::new(vec![0, 3, 1]),
        Err(TdmError::InvalidPermutation {
            position: 1,
            index: 3
        })
    );

    let sentinel = [91, 92, 93, 94];
    let mut wrong_input_output = sentinel;
    assert!(
        permutation
            .apply(&[1, 2, 3], &mut wrong_input_output)
            .is_err()
    );
    assert_eq!(wrong_input_output, sentinel);
    let mut short_output = [81, 82, 83];
    let short_output_before = short_output;
    assert!(permutation.apply(&[1, 2, 3, 4], &mut short_output).is_err());
    assert_eq!(short_output, short_output_before);

    let empty = Permutation::new(Vec::new()).unwrap();
    assert!(empty.is_empty());
    empty.apply::<u32>(&[], &mut []).unwrap();
}

#[test]
fn permutation_sampling_is_seeded_deterministic_and_valid() {
    let mut first_rng = ChaCha20Rng::from_seed([7; 32]);
    let mut second_rng = ChaCha20Rng::from_seed([7; 32]);
    let first = Permutation::sample(257, &mut first_rng).unwrap();
    let second = Permutation::sample(257, &mut second_rng).unwrap();
    assert_eq!(first, second);

    let mut sorted = first.indices().to_vec();
    sorted.sort_unstable();
    assert_eq!(sorted, (0..257).collect::<Vec<_>>());
}

#[test]
fn toeplitz_two_by_three_embedding_known_answer() {
    let diagonals = [2, 3, 5, 7];
    let map = ToeplitzMap::<17>::new(2, 3, elements(&diagonals)).unwrap();
    let mut output = elements(&[0, 0]);
    let mut scratch = map.scratch();
    map.apply(&elements(&[11, 13, 17]), &mut output, &mut scratch)
        .unwrap();

    // T = [[d1, d2, d3], [d0, d1, d2]].
    assert_eq!(values(&output), [13, 10]);
    assert_eq!(
        values(&output),
        toeplitz_oracle(2, 3, &diagonals, &[11, 13, 17], 17)
    );
    assert_eq!(map.rows(), 2);
    assert_eq!(map.columns(), 3);
    assert_eq!(values(map.diagonals()), diagonals);
    assert_eq!(map.transform_length(), 4);
}

fn check_rectangular_toeplitz<const MODULUS: u32>(
    rows: usize,
    columns: usize,
    diagonals: &[u32],
    input: &[u32],
) {
    let map = ToeplitzMap::<MODULUS>::new(rows, columns, elements(diagonals)).unwrap();
    let mut output = elements(&vec![0; rows]);
    let mut scratch = map.scratch();
    map.apply(&elements(input), &mut output, &mut scratch)
        .unwrap();
    assert_eq!(
        values(&output),
        toeplitz_oracle(rows, columns, diagonals, input, MODULUS)
    );
}

#[test]
fn awkward_rectangular_toeplitz_matches_dense_oracle_in_supported_fields() {
    let wide_diagonals = [4, 9, 16, 8, 15, 7, 3, 14, 2, 11, 6];
    let wide_input = [5, 1, 13, 7, 4];
    check_rectangular_toeplitz::<17>(7, 5, &wide_diagonals, &wide_input);
    check_rectangular_toeplitz::<1_073_479_681>(7, 5, &wide_diagonals, &wide_input);

    let tall_diagonals = [1, 8, 6, 14, 3, 12, 5, 9, 4, 16, 7];
    let tall_input = [
        123_456_789,
        1_073_479_680,
        42,
        765_432_100,
        19,
        81,
        2,
        700_000_003,
        11,
    ];
    check_rectangular_toeplitz::<1_073_479_681>(3, 9, &tall_diagonals, &tall_input);
}

#[test]
fn toeplitz_validation_and_length_errors_leave_output_unchanged() {
    assert!(matches!(
        ToeplitzMap::<17>::new(0, 2, elements(&[1])),
        Err(TdmError::ZeroDimension("Toeplitz rows"))
    ));
    assert!(matches!(
        ToeplitzMap::<17>::new(2, 0, elements(&[1])),
        Err(TdmError::ZeroDimension("Toeplitz columns"))
    ));
    assert!(matches!(
        ToeplitzMap::<17>::new(2, 3, elements(&[1, 2, 3])),
        Err(TdmError::LengthMismatch {
            name: "Toeplitz diagonals",
            expected: 4,
            actual: 3
        })
    ));
    assert!(matches!(
        ToeplitzMap::<17>::new(9, 9, elements(&[1; 17])),
        Err(TdmError::Field(FieldError::UnsupportedTransformLength(32)))
    ));

    let map = ToeplitzMap::<17>::new(3, 2, elements(&[1, 2, 3, 4])).unwrap();
    let mut scratch = map.scratch();
    let sentinel = elements::<17>(&[8, 9, 10]);
    let mut output = sentinel.clone();
    assert!(
        map.apply(&elements(&[1]), &mut output, &mut scratch)
            .is_err()
    );
    assert_eq!(output, sentinel);
    let mut short_output = elements::<17>(&[8, 9]);
    let short_output_before = short_output.clone();
    assert!(
        map.apply(&elements(&[1, 2]), &mut short_output, &mut scratch)
            .is_err()
    );
    assert_eq!(short_output, short_output_before);
}

fn explicit_fast_product() -> ToeplitzFastProduct<1_073_479_681> {
    let k = 3;
    let right = ToeplitzMap::new(6, 3, elements(&[2, 7, 1, 8, 2, 8, 1, 8])).unwrap();
    let pi_right = Permutation::new(vec![2, 5, 0, 4, 1, 3]).unwrap();
    let middle = ToeplitzMap::new(6, 6, elements(&[3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5])).unwrap();
    let pi_left = Permutation::new(vec![4, 0, 5, 2, 1, 3]).unwrap();
    let left = ToeplitzMap::new(3, 6, elements(&[8, 9, 7, 9, 3, 2, 3, 8])).unwrap();
    ToeplitzFastProduct::new(k, right, pi_right, middle, pi_left, left).unwrap()
}

fn fast_product_oracle(product: &ToeplitzFastProduct<1_073_479_681>, input: &[u32]) -> Vec<u32> {
    let right_diagonals = values(product.s_right().diagonals());
    let middle_diagonals = values(product.middle().diagonals());
    let left_diagonals = values(product.s_left().diagonals());
    let right = toeplitz_oracle(6, 3, &right_diagonals, input, 1_073_479_681);
    let right_gather = gather(&right, product.pi_right().indices());
    let middle = toeplitz_oracle(6, 6, &middle_diagonals, &right_gather, 1_073_479_681);
    let left_gather = gather(&middle, product.pi_left().indices());
    toeplitz_oracle(3, 6, &left_diagonals, &left_gather, 1_073_479_681)
}

#[test]
fn full_toeplitz_product_matches_hand_composition_materialization_and_linearity() {
    let product = explicit_fast_product();
    let input = [17, 23, 42];
    let expected = fast_product_oracle(&product, &input);
    let mut actual = elements(&[0; 3]);
    let mut scratch = product.scratch();
    product
        .apply(&elements(&input), &mut actual, &mut scratch)
        .unwrap();
    assert_eq!(values(&actual), expected);
    assert_eq!(scratch.stage_length(), 6);
    assert_eq!(scratch.transform_length(), 16);

    let materialized = product.materialize().unwrap();
    let expected_matrix = materialize_oracle(3, 1_073_479_681, |basis| {
        fast_product_oracle(&product, basis)
    });
    assert_eq!(values(materialized.values()), expected_matrix);
    let mut dense_output = elements(&[0; 3]);
    materialized
        .apply(&elements(&input), &mut dense_output)
        .unwrap();
    assert_eq!(dense_output, actual);

    let x = [10, 20, 30];
    let y = [7, 11, 13];
    let x_plus_y: [u32; 3] =
        std::array::from_fn(|index| add_mod(x[index], y[index], 1_073_479_681));
    let rx = fast_product_oracle(&product, &x);
    let ry = fast_product_oracle(&product, &y);
    let expected_sum: Vec<_> = rx
        .iter()
        .zip(&ry)
        .map(|(&lhs, &rhs)| add_mod(lhs, rhs, 1_073_479_681))
        .collect();
    assert_eq!(fast_product_oracle(&product, &x_plus_y), expected_sum);
}

#[test]
fn partial_toeplitz_materialization_matches_full_row_prefix() {
    let product = explicit_fast_product();
    let full = product.materialize().unwrap();
    for rows in 1..=3 {
        let partial = product.materialize_top_rows(rows).unwrap();
        assert_eq!((partial.rows(), partial.columns()), (rows, 3));
        assert_eq!(partial.values(), &full.values()[..rows * 3]);
    }
    product.materialize_top_rows(0).unwrap_err();
    product.materialize_top_rows(4).unwrap_err();
}

#[test]
fn parallel_toeplitz_top_rows_matches_serial_and_the_full_prefix() {
    // k = 64: 8 * 64 * 64 = 32768 estimated multiplications clear the
    // parallel-work threshold and 8 >= 2 * 4 rows satisfy the thread guard,
    // so the four-thread run forces the parallel branch while the
    // single-thread run takes the serial reference path.
    let mut rng = ChaCha20Rng::seed_from_u64(0x7a00);
    let product = ToeplitzFastProduct::<1_073_479_681>::sample(64, &mut rng).unwrap();

    let run = |threads: usize| {
        rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap()
            .install(|| product.materialize_top_rows(8))
            .unwrap()
    };
    let parallel = run(4);
    let serial = run(1);
    assert_eq!(parallel, serial);

    let full = product.materialize().unwrap();
    assert_eq!(parallel.values(), &full.values()[..8 * 64]);
}

#[test]
fn full_toeplitz_product_length_errors_leave_output_unchanged() {
    let product = explicit_fast_product();
    let mut scratch = product.scratch();
    let sentinel = elements::<1_073_479_681>(&[101, 102, 103]);
    let mut output = sentinel.clone();
    assert!(
        product
            .apply(&elements(&[1, 2]), &mut output, &mut scratch)
            .is_err()
    );
    assert_eq!(output, sentinel);

    let mut short_output = elements::<1_073_479_681>(&[201, 202]);
    let short_output_before = short_output.clone();
    assert!(
        product
            .apply(&elements(&[1, 2, 3]), &mut short_output, &mut scratch)
            .is_err()
    );
    assert_eq!(short_output, short_output_before);
}

#[test]
fn sampled_toeplitz_product_is_reproducible_from_seed() {
    let mut first_rng = ChaCha20Rng::from_seed([19; 32]);
    let mut second_rng = ChaCha20Rng::from_seed([19; 32]);
    let first = ToeplitzFastProduct::<1_073_479_681>::sample(5, &mut first_rng).unwrap();
    let second = ToeplitzFastProduct::<1_073_479_681>::sample(5, &mut second_rng).unwrap();

    assert_eq!(first.s_right().diagonals(), second.s_right().diagonals());
    assert_eq!(first.pi_right(), second.pi_right());
    assert_eq!(first.middle().diagonals(), second.middle().diagonals());
    assert_eq!(first.pi_left(), second.pi_left());
    assert_eq!(first.s_left().diagonals(), second.s_left().diagonals());
}

fn polynomial_mul_reduce<const K: usize>(
    lhs: &[u32; K],
    rhs: &[u32; K],
    modulus_polynomial: &[u32],
    modulus: u32,
) -> [u32; K] {
    assert!(K > 0);
    assert_eq!(modulus_polynomial.len(), K + 1);
    assert_eq!(modulus_polynomial[K] % modulus, 1);
    let mut product = vec![0; 2 * K - 1];
    for (lhs_degree, &lhs_coefficient) in lhs.iter().enumerate() {
        for (rhs_degree, &rhs_coefficient) in rhs.iter().enumerate() {
            let degree = lhs_degree + rhs_degree;
            product[degree] = add_mod(
                product[degree],
                mul_mod(
                    lhs_coefficient % modulus,
                    rhs_coefficient % modulus,
                    modulus,
                ),
                modulus,
            );
        }
    }

    // Long division by the monic modulus, from the leading term down.
    for degree in (K..product.len()).rev() {
        let factor = product[degree];
        for (modulus_degree, &modulus_coefficient) in modulus_polynomial.iter().take(K).enumerate()
        {
            let target = degree - K + modulus_degree;
            let subtract = mul_mod(factor, modulus_coefficient % modulus, modulus);
            product[target] = u32::try_from(
                (u64::from(product[target]) + u64::from(modulus) - u64::from(subtract))
                    % u64::from(modulus),
            )
            .unwrap();
        }
        product[degree] = 0;
    }
    std::array::from_fn(|index| product[index])
}

fn ring_sparse_dense() -> Vec<u32> {
    vec![
        2, 0, 0, // row 0
        0, 3, 0, // row 1
        0, 0, 7, // row 2
        5, 4, 0, // row 3
        0, 6, 2, // row 4
        1, 0, 8, // row 5
    ]
}

fn explicit_ring_lpn() -> IrreducibleRingLpn<17> {
    let k = 3;
    let sparse = SparseMatrix::new(
        6,
        3,
        vec![0, 3, 6, 9],
        vec![0, 3, 5, 1, 3, 4, 2, 4, 5],
        elements(&[2, 5, 1, 3, 4, 6, 7, 2, 8]),
    )
    .unwrap();
    // x^3 + 3x + 1 has no root in F_17, so the cubic is irreducible.
    IrreducibleRingLpn::new(k, &[1, 3, 0, 1], &[2, 1, 3], sparse).unwrap()
}

fn ring_lpn_oracle(input: &[u32]) -> Vec<u32> {
    let sparse_product = dense_apply(6, 3, &ring_sparse_dense(), input, 17);
    let u0: [u32; 3] = sparse_product[..3].try_into().unwrap();
    let u1: [u32; 3] = sparse_product[3..].try_into().unwrap();
    let multiplied = polynomial_mul_reduce(&[2, 1, 3], &u1, &[1, 3, 0, 1], 17);
    u0.iter()
        .zip(multiplied)
        .map(|(&identity, product)| add_mod(identity, product, 17))
        .collect()
}

#[test]
fn ring_lpn_matches_independent_sparse_and_polynomial_oracles() {
    let instance = explicit_ring_lpn();
    let input = [2, 3, 4];
    let sparse_product = dense_apply(6, 3, &ring_sparse_dense(), &input, 17);
    assert_eq!(sparse_product, [4, 9, 11, 5, 9, 0]);

    let expected = ring_lpn_oracle(&input);
    let mut actual = elements(&[0; 3]);
    let mut scratch = instance.scratch();
    instance
        .apply(&elements(&input), &mut actual, &mut scratch)
        .unwrap();
    assert_eq!(values(&actual), expected);
    assert_eq!(instance.multiplier(), &[2, 1, 3]);
    assert_eq!(instance.modulus(), &[1, 3, 0, 1]);
    assert_eq!(instance.nnz(), 9);

    let materialized = instance.materialize().unwrap();
    let expected_matrix = materialize_oracle(3, 17, ring_lpn_oracle);
    assert_eq!(values(materialized.values()), expected_matrix);
    let mut dense_output = elements(&[0; 3]);
    materialized
        .apply(&elements(&input), &mut dense_output)
        .unwrap();
    assert_eq!(dense_output, actual);

    let x = [1, 5, 9];
    let y = [4, 8, 16];
    let sum: [u32; 3] = std::array::from_fn(|index| add_mod(x[index], y[index], 17));
    let expected_sum: Vec<_> = ring_lpn_oracle(&x)
        .into_iter()
        .zip(ring_lpn_oracle(&y))
        .map(|(lhs, rhs)| add_mod(lhs, rhs, 17))
        .collect();
    assert_eq!(ring_lpn_oracle(&sum), expected_sum);
}

#[test]
fn ring_lpn_rejects_reducible_modulus_and_malformed_csc() {
    let k = 3;
    let valid_sparse = SparseMatrix::new(6, 3, vec![0, 0, 0, 0], Vec::new(), Vec::new()).unwrap();
    assert!(matches!(
        IrreducibleRingLpn::<17>::new(k, &[0, 0, 0, 1], &[1, 2, 3], valid_sparse),
        Err(TdmError::ExtensionField(
            ExtensionFieldError::ReducibleModulus
        ))
    ));

    assert!(matches!(
        SparseMatrix::<17>::new(3, 2, vec![0, 0], Vec::new(), Vec::new()),
        Err(TdmError::LengthMismatch {
            name: "sparse offsets",
            ..
        })
    ));
    for offsets in [vec![1, 1, 1], vec![0, 2, 1], vec![0, 0, 1]] {
        assert_eq!(
            SparseMatrix::<17>::new(3, 2, offsets, Vec::new(), Vec::new()),
            Err(TdmError::InvalidSparseOffsets)
        );
    }
    assert_eq!(
        SparseMatrix::<17>::new(3, 1, vec![0, 1], vec![3], elements(&[1])),
        Err(TdmError::SparseRowOutOfBounds {
            entry: 0,
            row: 3,
            rows: 3
        })
    );
    assert_eq!(
        SparseMatrix::<17>::new(3, 1, vec![0, 1], vec![1], Vec::new()),
        Err(TdmError::InvalidSparseOffsets)
    );
}

#[test]
fn ring_lpn_duplicate_sparse_rows_accumulate_independently() {
    let sparse = SparseMatrix::new(2, 1, vec![0, 3], vec![0, 0, 1], elements(&[2, 3, 4])).unwrap();
    let k = 1;
    let instance = IrreducibleRingLpn::<17>::new(k, &[1, 1], &[5], sparse).unwrap();
    let mut output = elements(&[0]);
    instance
        .apply(&elements(&[2]), &mut output, &mut instance.scratch())
        .unwrap();
    assert_eq!(values(&output), [16]);
}

#[test]
fn ring_lpn_sampling_is_reproducible_and_respects_weight_bounds() {
    let modulus = [1, 3, 0, 1];
    let k = 3;

    let mut first_rng = ChaCha20Rng::from_seed([41; 32]);
    let mut second_rng = ChaCha20Rng::from_seed([41; 32]);
    let first =
        IrreducibleRingLpn::<17>::sample_with_modulus(k, 2, &modulus, &mut first_rng).unwrap();
    let second =
        IrreducibleRingLpn::<17>::sample_with_modulus(k, 2, &modulus, &mut second_rng).unwrap();
    assert_eq!(
        first.instance().multiplier(),
        second.instance().multiplier()
    );
    assert_eq!(
        first.instance().sparse_matrix(),
        second.instance().sparse_matrix()
    );

    let mut zero_rng = ChaCha20Rng::from_seed([31; 32]);
    assert!(matches!(
        IrreducibleRingLpn::<17>::sample_with_modulus(k, 0, &modulus, &mut zero_rng),
        Err(TdmError::ZeroDimension("column weight"))
    ));

    let mut oversized_rng = ChaCha20Rng::from_seed([43; 32]);
    assert!(matches!(
        IrreducibleRingLpn::<17>::sample_with_modulus(k, 4, &modulus, &mut oversized_rng),
        Err(TdmError::WeightExceedsColumns {
            weight: 4,
            maximum: 3
        })
    ));
}

#[test]
fn ring_lpn_sampling_never_produces_empty_columns() {
    let modulus = automatic_ring_modulus::<17>(8).unwrap();
    let k = 8;
    // With weight one over sixteen rows, a Bernoulli column is empty with
    // probability about (15/16)^16 ~= 0.36, so this exercises resampling.
    for domain in 1..8u8 {
        let mut rng = ChaCha20Rng::from_seed([domain; 32]);
        let sampled =
            IrreducibleRingLpn::<17>::sample_with_modulus(k, 1, &modulus, &mut rng).unwrap();
        let offsets = sampled.instance().sparse_matrix().column_offsets();
        assert!(offsets.windows(2).all(|pair| pair[0] < pair[1]));
        assert!(
            sampled
                .instance()
                .sparse_matrix()
                .values()
                .iter()
                .all(|value| value.value() != 0)
        );
    }
}

#[test]
fn ring_lpn_automatic_sampling_proves_its_modulus_and_reports_assessment() {
    // F_17 has two-adicity four, so k = 4 gets an irreducible binomial.
    let mut rng = ChaCha20Rng::from_seed([47; 32]);
    let sampled = IrreducibleRingLpn::<17>::sample(4, 2, &mut rng).unwrap();
    let modulus = sampled.instance().modulus();
    assert_eq!(modulus.len(), 5);
    assert_eq!(modulus[4], 1);
    assert!(modulus[1..4].iter().all(|&coefficient| coefficient == 0));
    assert!(modulus[0] > 0 && modulus[0] < 17);
    // The analytic proof is confirmed here by Rabin's independent test.
    ExtensionField::<17>::new(4, modulus).unwrap();
    // Tiny parameters still construct but are reported as broken: the ring
    // degree, the plausibility weight, the decoding estimate, and the
    // enumeration estimate all fail their floors at k = 4.
    assert_eq!(sampled.warnings().len(), 4);
    assert!(matches!(
        sampled.warnings()[0],
        ParameterWarning::RingDegreeBelowFloor { .. }
    ));

    // Determinism: the same seed yields the same instance.
    let mut first_rng = ChaCha20Rng::from_seed([47; 32]);
    let first = IrreducibleRingLpn::<17>::sample(4, 2, &mut first_rng).unwrap();
    assert_eq!(
        first.instance().sparse_matrix(),
        sampled.instance().sparse_matrix()
    );
    assert_eq!(
        first.instance().multiplier(),
        sampled.instance().multiplier()
    );
    assert_eq!(first.warnings(), sampled.warnings());
}

#[test]
fn ring_lpn_automatic_sampling_rejects_unsupported_shapes() {
    let mut rng = ChaCha20Rng::from_seed([53; 32]);
    // Non-power-of-two degree has no automatic binomial.
    assert!(matches!(
        IrreducibleRingLpn::<17>::sample(3, 1, &mut rng),
        Err(TdmError::AutomaticModulusUnsupported {
            modulus: 17,
            degree: 3
        })
    ));
    // F_2 has two-adicity zero.
    assert!(matches!(
        IrreducibleRingLpn::<2>::sample(4, 1, &mut rng),
        Err(TdmError::AutomaticModulusUnsupported {
            modulus: 2,
            degree: 4
        })
    ));
}

#[test]
fn ring_lpn_degree_one_edge_has_expected_split_orientation() {
    let k = 1;
    let sparse = SparseMatrix::new(2, 1, vec![0, 2], vec![0, 1], elements(&[3, 4])).unwrap();
    let instance = IrreducibleRingLpn::<17>::new(k, &[1, 1], &[5], sparse).unwrap();
    let mut output = elements(&[0]);
    let mut scratch = instance.scratch();
    instance
        .apply(&elements(&[2]), &mut output, &mut scratch)
        .unwrap();

    // E*x = [6, 8], then [I | M_a] gives 6 + 5*8 = 12 mod 17.
    assert_eq!(values(&output), [12]);
    assert_eq!(values(instance.materialize().unwrap().values()), [6]);
}

#[test]
fn ring_lpn_length_errors_leave_output_unchanged() {
    let instance = explicit_ring_lpn();
    let mut scratch = instance.scratch();
    let sentinel = elements::<17>(&[12, 13, 14]);
    let mut output = sentinel.clone();
    assert!(
        instance
            .apply(&elements(&[1, 2]), &mut output, &mut scratch)
            .is_err()
    );
    assert_eq!(output, sentinel);

    let mut short_output = elements::<17>(&[15, 16]);
    let short_output_before = short_output.clone();
    assert!(
        instance
            .apply(&elements(&[1, 2, 3]), &mut short_output, &mut scratch)
            .is_err()
    );
    assert_eq!(short_output, short_output_before);
}

fn weighted_scan(input: &[u32], weights: &[u32], modulus: u32) -> Vec<u32> {
    let mut sum = 0;
    input
        .iter()
        .zip(weights)
        .map(|(&value, &weight)| {
            sum = add_mod(
                sum,
                mul_mod(value % modulus, weight % modulus, modulus),
                modulus,
            );
            sum
        })
        .collect()
}

fn raa_oracle(
    input: &[u32],
    c: usize,
    weights: [&[u32]; 3],
    permutations: [&[usize]; 4],
    modulus: u32,
) -> Vec<u32> {
    let repeated: Vec<_> = input
        .iter()
        .flat_map(|&value| std::iter::repeat_n(value, c))
        .collect();
    let first_gather = gather(&repeated, permutations[0]);
    let first_scan = weighted_scan(&first_gather, weights[0], modulus);
    let second_gather = gather(&first_scan, permutations[1]);
    let second_scan = weighted_scan(&second_gather, weights[1], modulus);
    let third_gather = gather(&second_scan, permutations[2]);
    let third_scan = weighted_scan(&third_gather, weights[2], modulus);
    let fourth_gather = gather(&third_scan, permutations[3]);
    fourth_gather
        .chunks_exact(c)
        .map(|block| {
            block
                .iter()
                .fold(0, |sum, &value| add_mod(sum, value, modulus))
        })
        .collect()
}

const RAA_W1: [u32; 6] = [2, 0, 3, 4, 1, 5];
const RAA_W2: [u32; 6] = [1, 3, 2, 0, 4, 2];
const RAA_W3: [u32; 6] = [5, 1, 0, 3, 2, 4];
const RAA_P1: [usize; 6] = [2, 5, 0, 3, 1, 4];
const RAA_P2: [usize; 6] = [4, 1, 5, 0, 3, 2];
const RAA_P3: [usize; 6] = [1, 3, 0, 5, 2, 4];
const RAA_P4: [usize; 6] = [5, 2, 4, 1, 0, 3];

fn explicit_raa() -> RaaWeightedProduct<17> {
    RaaWeightedProduct::new(
        3,
        2,
        elements(&RAA_W1),
        elements(&RAA_W2),
        elements(&RAA_W3),
        Permutation::new(RAA_P1.to_vec()).unwrap(),
        Permutation::new(RAA_P2.to_vec()).unwrap(),
        Permutation::new(RAA_P3.to_vec()).unwrap(),
        Permutation::new(RAA_P4.to_vec()).unwrap(),
    )
    .unwrap()
}

fn explicit_raa_oracle(input: &[u32]) -> Vec<u32> {
    raa_oracle(
        input,
        2,
        [&RAA_W1, &RAA_W2, &RAA_W3],
        [&RAA_P1, &RAA_P2, &RAA_P3, &RAA_P4],
        17,
    )
}

#[test]
fn raa_order_orientation_zero_weights_and_dense_forms_match_hand_oracle() {
    let product = explicit_raa();
    let input = [2, 5, 7];
    let expected = explicit_raa_oracle(&input);
    assert_eq!(expected, [7, 3, 7]);

    let mut actual = elements(&[0; 3]);
    let mut scratch = product.scratch();
    product
        .apply(&elements(&input), &mut actual, &mut scratch)
        .unwrap();
    assert_eq!(values(&actual), expected);
    assert_eq!(product.k(), 3);
    assert_eq!(product.c(), 2);
    assert_eq!(product.n(), 6);
    assert_eq!(scratch.len(), 6);
    assert!(!scratch.is_empty());
    assert_eq!(values(product.first_weights()), RAA_W1);
    assert_eq!(product.first_permutation().indices(), RAA_P1);
    assert_eq!(product.second_permutation().indices(), RAA_P2);
    assert_eq!(product.third_permutation().indices(), RAA_P3);
    assert_eq!(product.fourth_permutation().indices(), RAA_P4);

    let materialized = product.materialize().unwrap();
    let expected_matrix = materialize_oracle(3, 17, explicit_raa_oracle);
    assert_eq!(values(materialized.values()), expected_matrix);
    let mut dense_output = elements(&[0; 3]);
    materialized
        .apply(&elements(&input), &mut dense_output)
        .unwrap();
    assert_eq!(dense_output, actual);

    let x = [1, 4, 8];
    let y = [3, 5, 9];
    let sum: [u32; 3] = std::array::from_fn(|index| add_mod(x[index], y[index], 17));
    let expected_sum: Vec<_> = explicit_raa_oracle(&x)
        .into_iter()
        .zip(explicit_raa_oracle(&y))
        .map(|(lhs, rhs)| add_mod(lhs, rhs, 17))
        .collect();
    assert_eq!(explicit_raa_oracle(&sum), expected_sum);
}

#[test]
fn raa_c_one_edge_and_all_zero_weights_are_supported() {
    let identity = Permutation::new(vec![0, 1, 2]).unwrap();
    let product = RaaWeightedProduct::<17>::new(
        3,
        1,
        elements(&[2, 3, 4]),
        elements(&[5, 6, 7]),
        elements(&[8, 9, 10]),
        identity.clone(),
        identity.clone(),
        identity.clone(),
        identity,
    )
    .unwrap();
    let expected = raa_oracle(
        &[3, 5, 7],
        1,
        [&[2, 3, 4], &[5, 6, 7], &[8, 9, 10]],
        [&[0, 1, 2], &[0, 1, 2], &[0, 1, 2], &[0, 1, 2]],
        17,
    );
    let mut output = elements(&[0; 3]);
    product
        .apply(&elements(&[3, 5, 7]), &mut output, &mut product.scratch())
        .unwrap();
    assert_eq!(values(&output), expected);

    let identity = Permutation::new(vec![0, 1, 2, 3]).unwrap();
    let zero_product = RaaWeightedProduct::<17>::new(
        2,
        2,
        elements(&[0; 4]),
        elements(&[1; 4]),
        elements(&[1; 4]),
        identity.clone(),
        identity.clone(),
        identity.clone(),
        identity,
    )
    .unwrap();
    let mut zero_output = elements(&[9, 9]);
    zero_product
        .apply(
            &elements(&[4, 6]),
            &mut zero_output,
            &mut zero_product.scratch(),
        )
        .unwrap();
    assert_eq!(values(&zero_output), [0, 0]);
}

#[test]
fn raa_validation_and_length_errors_leave_output_unchanged() {
    let empty = Permutation::new(Vec::new()).unwrap();
    assert!(matches!(
        RaaWeightedProduct::<17>::new(
            0,
            2,
            Vec::new(),
            Vec::new(),
            Vec::new(),
            empty.clone(),
            empty.clone(),
            empty.clone(),
            empty
        ),
        Err(TdmError::ZeroDimension("k"))
    ));

    let identity = Permutation::new(vec![0, 1, 2, 3]).unwrap();
    assert!(matches!(
        RaaWeightedProduct::<17>::new(
            2,
            2,
            elements(&[1; 3]),
            elements(&[1; 4]),
            elements(&[1; 4]),
            identity.clone(),
            identity.clone(),
            identity.clone(),
            identity
        ),
        Err(TdmError::LengthMismatch {
            name: "first weights",
            ..
        })
    ));

    let product = explicit_raa();
    let mut scratch = product.scratch();
    let sentinel = elements::<17>(&[11, 12, 13]);
    let mut output = sentinel.clone();
    assert!(
        product
            .apply(&elements(&[1, 2]), &mut output, &mut scratch)
            .is_err()
    );
    assert_eq!(output, sentinel);
    let mut short_output = elements::<17>(&[14, 15]);
    let short_output_before = short_output.clone();
    assert!(
        product
            .apply(&elements(&[1, 2, 3]), &mut short_output, &mut scratch)
            .is_err()
    );
    assert_eq!(short_output, short_output_before);
}

#[test]
fn sampled_raa_is_reproducible_and_has_nonzero_weights() {
    let mut first_rng = ChaCha20Rng::from_seed([53; 32]);
    let mut second_rng = ChaCha20Rng::from_seed([53; 32]);
    let first = RaaWeightedProduct::<17>::sample_nonzero(5, 3, &mut first_rng).unwrap();
    let second = RaaWeightedProduct::<17>::sample_nonzero(5, 3, &mut second_rng).unwrap();
    assert_eq!(first, second);
    assert!(first.first_weights().iter().all(|value| value.value() != 0));
    assert!(
        first
            .second_weights()
            .iter()
            .all(|value| value.value() != 0)
    );
    assert!(first.third_weights().iter().all(|value| value.value() != 0));
}

#[test]
fn toeplitz_product_k_one_matches_materialized_matrix() {
    let mut rng = ChaCha20Rng::from_seed([61; 32]);
    let product = ToeplitzFastProduct::<17>::sample(1, &mut rng).unwrap();
    let input = elements(&[7]);
    let mut structured = elements(&[0]);
    product
        .apply(&input, &mut structured, &mut product.scratch())
        .unwrap();
    let mut dense = elements(&[0]);
    product
        .materialize()
        .unwrap()
        .apply(&input, &mut dense)
        .unwrap();
    assert_eq!(structured, dense);
}

#[test]
fn warmed_apply_paths_allocate_nothing() {
    let ring = explicit_ring_lpn();
    let ring_input = elements(&[1, 2, 3]);
    let mut ring_output = elements(&[0; 3]);
    let mut ring_scratch = ring.scratch();

    let toeplitz = explicit_fast_product();
    let toeplitz_input = elements(&[1, 2, 3]);
    let mut toeplitz_output = elements(&[0; 3]);
    let mut toeplitz_scratch = toeplitz.scratch();

    let raa = explicit_raa();
    let raa_input = elements(&[1, 2, 3]);
    let mut raa_output = elements(&[0; 3]);
    let mut raa_scratch = raa.scratch();

    ring.apply(&ring_input, &mut ring_output, &mut ring_scratch)
        .unwrap();
    toeplitz
        .apply(&toeplitz_input, &mut toeplitz_output, &mut toeplitz_scratch)
        .unwrap();
    raa.apply(&raa_input, &mut raa_output, &mut raa_scratch)
        .unwrap();

    let allocations = allocation_counter::measure(|| {
        for _ in 0..8 {
            ring.apply(&ring_input, &mut ring_output, &mut ring_scratch)
                .unwrap();
            toeplitz
                .apply(&toeplitz_input, &mut toeplitz_output, &mut toeplitz_scratch)
                .unwrap();
            raa.apply(&raa_input, &mut raa_output, &mut raa_scratch)
                .unwrap();
        }
    });
    assert_eq!(allocations.count_total, 0);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn toeplitz_fast_path_matches_formula_for_arbitrary_entries(
        diagonals in prop::collection::vec(any::<u32>(), 11),
        input in prop::collection::vec(any::<u32>(), 7),
    ) {
        check_rectangular_toeplitz::<1_073_479_681>(5, 7, &diagonals, &input);
    }

    #[test]
    fn ring_lpn_matches_independent_long_division_for_arbitrary_input(
        input in prop::array::uniform3(any::<u32>()),
    ) {
        let instance = explicit_ring_lpn();
        let mut output = elements(&[0; 3]);
        instance
            .apply(&elements(&input), &mut output, &mut instance.scratch())
            .unwrap();
        prop_assert_eq!(values(&output), ring_lpn_oracle(&input));
    }

    #[test]
    fn raa_matches_hand_pipeline_for_arbitrary_input(
        input in prop::array::uniform3(any::<u32>()),
    ) {
        let product = explicit_raa();
        let mut output = elements(&[0; 3]);
        product
            .apply(&elements(&input), &mut output, &mut product.scratch())
            .unwrap();
        prop_assert_eq!(values(&output), explicit_raa_oracle(&input));
    }
}
