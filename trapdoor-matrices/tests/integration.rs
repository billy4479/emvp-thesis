#![expect(
    clippy::expect_used,
    clippy::unwrap_used,
    reason = "fixed test fixtures establish that construction and evaluation must succeed"
)]

use std::collections::VecDeque;
use std::convert::Infallible;

use prime_field_layer::{
    ExtensionField, ExtensionFieldError, FieldElement, FieldError, PolynomialAlgorithm, PrimeField,
};
use proptest::prelude::*;
use rand_chacha::ChaCha20Rng;
use rand_core::{SeedableRng, TryCryptoRng, TryRng};
use trapdoor_matrices::{
    DenseMatrix, IrreducibleRingLpn, Permutation, RaaWeightedProduct, SecurityWarningKind,
    SparseMatrix, TdmError, ToeplitzFastProduct, ToeplitzMap, automatic_ring_modulus,
};

/// NTT-friendly prime with two-adicity 18: every transform length used by the
/// policy-grade sampling tests below is supported.
const FIELD: u32 = 1_073_479_681;

/// The smallest `(degree, weight)` pair the assessment rates sound: the
/// ring-degree floor of 2048 together with the project policy weight floor of
/// 192. Every test that must reach the sampler works at this pair, because
/// sampling now refuses broken assessments outright.
const SOUND_DEGREE: usize = 2048;
const SOUND_WEIGHT: usize = 192;

/// A deterministic `CryptoRng` replaying a run-length-coded word script.
///
/// Each entry yields the same 32-bit word for the given number of draws, so
/// tests can script the millions of Bernoulli draws of a policy-grade sample
/// compactly and exactly. Panics when the script runs dry: a sampler whose
/// draw pattern deviates from the scripted one exhausts the script and fails
/// the test loudly instead of silently.
struct ScriptedRng {
    runs: VecDeque<(u32, u64)>,
    current: u32,
    remaining: u64,
    draws: u64,
}

impl ScriptedRng {
    /// Builds a scripted RNG whose `draws` counter starts at zero.
    fn new(runs: &[(u32, u64)]) -> Self {
        let mut runs: VecDeque<(u32, u64)> = runs.iter().copied().collect();
        let (current, remaining) = runs.pop_front().unwrap_or((0, u64::MAX));
        Self {
            runs,
            current,
            remaining,
            draws: 0,
        }
    }

    fn next_word(&mut self) -> u32 {
        self.draws += 1;
        if self.remaining == 0 {
            let (word, length) = self.runs.pop_front().expect("scripted RNG exhausted");
            self.current = word;
            self.remaining = length;
        }
        self.remaining -= 1;
        self.current
    }
}

impl TryRng for ScriptedRng {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Self::Error> {
        Ok(self.next_word())
    }

    fn try_next_u64(&mut self) -> Result<u64, Self::Error> {
        let low = u64::from(self.next_word());
        let high = u64::from(self.next_word());
        Ok((high << 32) | low)
    }

    fn try_fill_bytes(&mut self, destination: &mut [u8]) -> Result<(), Self::Error> {
        for chunk in destination.chunks_mut(4) {
            let word = self.next_word().to_le_bytes();
            chunk.copy_from_slice(&word[..chunk.len()]);
        }
        Ok(())
    }
}

impl TryCryptoRng for ScriptedRng {}

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
    let mut rx = elements(&[0; 3]);
    let mut ry = elements(&[0; 3]);
    let mut rxy = elements(&[0; 3]);
    product.apply(&elements(&x), &mut rx, &mut scratch).unwrap();
    product.apply(&elements(&y), &mut ry, &mut scratch).unwrap();
    product
        .apply(&elements(&x_plus_y), &mut rxy, &mut scratch)
        .unwrap();
    let expected_sum: Vec<_> = values(&rx)
        .iter()
        .zip(values(&ry))
        .map(|(&lhs, rhs)| add_mod(lhs, rhs, 1_073_479_681))
        .collect();
    assert_eq!(values(&rxy), expected_sum);
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
    let mut rx = elements(&[0; 3]);
    let mut ry = elements(&[0; 3]);
    let mut rxy = elements(&[0; 3]);
    let mut scratch = instance.scratch();
    instance
        .apply(&elements(&x), &mut rx, &mut scratch)
        .unwrap();
    instance
        .apply(&elements(&y), &mut ry, &mut scratch)
        .unwrap();
    instance
        .apply(&elements(&sum), &mut rxy, &mut scratch)
        .unwrap();
    let expected_sum: Vec<_> = values(&rx)
        .iter()
        .zip(values(&ry))
        .map(|(&lhs, rhs)| add_mod(lhs, rhs, 17))
        .collect();
    assert_eq!(values(&rxy), expected_sum);
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
    // The smallest policy-grade pair: sound with an empty warning list, so
    // the sampler actually constructs the instance. Rabin's irreducibility
    // test would be far too slow at this degree, so the happy-path runs use
    // the automatic modulus, whose irreducibility is proven analytically;
    // the explicit-modulus entry point shares everything downstream of the
    // assessment and is covered by the fail-closed and validation tests.
    let modulus = automatic_ring_modulus::<FIELD>(SOUND_DEGREE).unwrap();

    let mut first_rng = ChaCha20Rng::from_seed([41; 32]);
    let mut second_rng = ChaCha20Rng::from_seed([41; 32]);
    let first =
        IrreducibleRingLpn::<FIELD>::sample(SOUND_DEGREE, SOUND_WEIGHT, &mut first_rng).unwrap();
    let second =
        IrreducibleRingLpn::<FIELD>::sample(SOUND_DEGREE, SOUND_WEIGHT, &mut second_rng).unwrap();
    assert_eq!(
        first.instance().multiplier(),
        second.instance().multiplier()
    );
    assert_eq!(
        first.instance().sparse_matrix(),
        second.instance().sparse_matrix()
    );
    assert_eq!(first.warnings(), second.warnings());
    assert!(first.warnings().is_empty());

    // Columns are conditioned to be nonempty, so every stored column has at
    // least one entry and every value is a nonzero field element.
    let offsets = first.instance().sparse_matrix().offsets();
    assert!(offsets.windows(2).all(|pair| pair[0] < pair[1]));
    assert!(
        first
            .instance()
            .sparse_matrix()
            .values()
            .iter()
            .all(|value| value.value() != 0)
    );

    // Shape validation still precedes the assessment, keeping its precise
    // typed errors (and running before any expensive modulus work).
    let mut zero_rng = ChaCha20Rng::from_seed([31; 32]);
    assert!(matches!(
        IrreducibleRingLpn::<FIELD>::sample_with_modulus(SOUND_DEGREE, 0, &modulus, &mut zero_rng),
        Err(TdmError::ZeroDimension("column weight"))
    ));

    let mut oversized_rng = ChaCha20Rng::from_seed([43; 32]);
    assert!(matches!(
        IrreducibleRingLpn::<FIELD>::sample_with_modulus(
            SOUND_DEGREE,
            SOUND_DEGREE + 1,
            &modulus,
            &mut oversized_rng
        ),
        Err(TdmError::WeightExceedsColumns { weight, maximum })
            if weight == SOUND_DEGREE + 1 && maximum == SOUND_DEGREE
    ));
}

#[test]
fn ring_lpn_sampling_never_produces_empty_columns() {
    // At the policy-grade pair a Bernoulli column draw is empty with
    // probability about e^{-192}; conditioning makes every stored column
    // nonempty by construction, which this checks across seeds.
    for domain in 1..3u8 {
        let mut rng = ChaCha20Rng::from_seed([domain; 32]);
        let sampled =
            IrreducibleRingLpn::<FIELD>::sample(SOUND_DEGREE, SOUND_WEIGHT, &mut rng).unwrap();
        let offsets = sampled.instance().sparse_matrix().offsets();
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
fn ring_lpn_sampling_fails_closed_on_broken_assessments() {
    // A degree below the ring-degree floor is broken, so automatic sampling
    // refuses before drawing anything: the scripted RNG records zero draws.
    let mut scripted = ScriptedRng::new(&[(3, u64::MAX)]);
    assert!(matches!(
        IrreducibleRingLpn::<FIELD>::sample(1024, 256, &mut scripted),
        Err(TdmError::InsecureParameters {
            degree: 1024,
            weight: 256,
            reasons
        }) if reasons == vec![SecurityWarningKind::RingDegreeBelowFloor]
    ));
    assert_eq!(scripted.draws, 0);

    // The explicit-modulus entry point fails closed the same way, before the
    // (potentially expensive) irreducibility test would run.
    let modulus = automatic_ring_modulus::<FIELD>(1024).unwrap();
    let mut scripted = ScriptedRng::new(&[(3, u64::MAX)]);
    assert!(matches!(
        IrreducibleRingLpn::<FIELD>::sample_with_modulus(1024, 256, &modulus, &mut scripted),
        Err(TdmError::InsecureParameters {
            degree: 1024,
            weight: 256,
            ..
        })
    ));
    assert_eq!(scripted.draws, 0);

    // A tiny weight at a policy-grade degree is broken through the
    // plausibility floor and the decoding estimate; the structured reasons
    // name every failed check without floating-point payloads.
    let mut scripted = ScriptedRng::new(&[(3, u64::MAX)]);
    assert!(matches!(
        IrreducibleRingLpn::<FIELD>::sample(8192, 16, &mut scripted),
        Err(TdmError::InsecureParameters {
            degree: 8192,
            weight: 16,
            reasons
        }) if reasons == vec![
            SecurityWarningKind::WeightBelowPlausibilityFloor,
            SecurityWarningKind::DecodingCostBelowTarget
        ]
    ));
    assert_eq!(scripted.draws, 0);

    // Tiny parameters were previously sampled with warnings; they are now
    // refused with the full ordered list of broken reasons.
    let mut rng = ChaCha20Rng::from_seed([47; 32]);
    assert!(matches!(
        IrreducibleRingLpn::<17>::sample(4, 2, &mut rng),
        Err(TdmError::InsecureParameters {
            degree: 4,
            weight: 2,
            reasons
        }) if reasons == vec![
            SecurityWarningKind::RingDegreeBelowFloor,
            SecurityWarningKind::WeightBelowPlausibilityFloor,
            SecurityWarningKind::DecodingCostBelowTarget,
            SecurityWarningKind::EnumerationCostBelowTarget
        ]
    ));
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
    // F_2 is unsupported by the field layer itself (odd-prime invariant), so
    // it cannot instantiate this construction at all; its automatic-modulus
    // rejection is covered by the parameter tests, which never touch a field
    // type.
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

#[test]
fn ring_lpn_constructors_reject_wrong_multiplier_lengths() {
    let valid_sparse =
        SparseMatrix::new(6, 3, vec![0, 1, 2, 3], vec![0, 1, 2], elements(&[3, 4, 5])).unwrap();

    for multiplier in [Vec::new(), vec![2, 1], vec![2, 1, 3, 4]] {
        assert!(matches!(
            IrreducibleRingLpn::<17>::new(3, &[1, 3, 0, 1], &multiplier, valid_sparse.clone()),
            Err(TdmError::LengthMismatch {
                name: "multiplier",
                expected: 3,
                ..
            })
        ));
        assert!(matches!(
            IrreducibleRingLpn::<17>::new_unchecked_irreducible(
                3,
                &[1, 3, 0, 1],
                &multiplier,
                valid_sparse.clone()
            ),
            Err(TdmError::LengthMismatch {
                name: "multiplier",
                expected: 3,
                ..
            })
        ));
    }

    // The exact-length multiplier is accepted by both constructors.
    let instance = IrreducibleRingLpn::<17>::new(
        3,
        &[1, 3, 0, 1],
        &[2, 1, 3],
        SparseMatrix::new(
            6,
            3,
            vec![0, 3, 6, 9],
            vec![0, 3, 5, 1, 3, 4, 2, 4, 5],
            elements(&[2, 5, 1, 3, 4, 6, 7, 2, 8]),
        )
        .unwrap(),
    )
    .unwrap();
    assert_eq!(instance.multiplier(), &[2, 1, 3]);
}

/// Builds a deterministic instance of an arbitrary degree over `F_17`, with a
/// provably irreducible automatic binomial modulus.
fn explicit_ring_lpn_with_degree(k: usize) -> IrreducibleRingLpn<17> {
    let rows = k * 2;
    let mut row_indices = Vec::new();
    let mut offsets = vec![0usize];
    let mut entry_values = Vec::new();
    for column in 0..k {
        for entry in 0..2usize {
            row_indices.push((3 * column + entry) % rows);
            entry_values.push(((column * 7 + entry * 5) % 16 + 1) as u32);
        }
        offsets.push(row_indices.len());
    }
    let sparse =
        SparseMatrix::new(rows, k, offsets, row_indices, elements::<17>(&entry_values)).unwrap();
    let modulus = automatic_ring_modulus::<17>(k).unwrap();
    let multiplier: Vec<u32> = (0..k).map(|index| (index * 5 + 3) as u32).collect();
    IrreducibleRingLpn::new_unchecked_irreducible(k, &modulus, &multiplier, sparse).unwrap()
}

#[test]
fn ring_lpn_rejects_cross_instance_scratch_and_leaves_output_unchanged() {
    let small = explicit_ring_lpn();
    let large = explicit_ring_lpn_with_degree(4);

    // A scratch from the smaller instance is too short for the larger one.
    let mut small_scratch = small.scratch();
    let large_sentinel = elements::<17>(&[41, 42, 43, 44]);
    let mut large_output = large_sentinel.clone();
    assert!(matches!(
        large.apply(
            &elements(&[5, 6, 7, 8]),
            &mut large_output,
            &mut small_scratch
        ),
        Err(TdmError::LengthMismatch {
            name: "scratch sparse product",
            expected: 8,
            actual: 6
        })
    ));
    assert_eq!(large_output, large_sentinel);

    // A scratch from the larger instance is longer than the smaller one
    // needs; it must still be rejected rather than silently applied.
    let mut large_scratch = large.scratch();
    let small_sentinel = elements::<17>(&[31, 32, 33]);
    let mut small_output = small_sentinel.clone();
    assert!(matches!(
        small.apply(&elements(&[2, 3, 4]), &mut small_output, &mut large_scratch),
        Err(TdmError::LengthMismatch {
            name: "scratch sparse product",
            expected: 6,
            actual: 8
        })
    ));
    assert_eq!(small_output, small_sentinel);
}

#[test]
fn ring_lpn_per_column_retry_redraws_only_the_empty_column() {
    // Script a policy-grade sample so the first draw of column 0 comes out
    // empty and every other column succeeds on its first attempt:
    //
    // - 2048 uniform multiplier draws of the word 3;
    // - column 0, attempt 1: 4096 row draws of 4095 (>= weight 192, empty);
    // - then one successful attempt per column: row 0 hits (word 0 < 192),
    //   the nonzero entry is drawn from word 1000, and the remaining 4095
    //   rows draw 4095 and miss.
    let rows = SOUND_DEGREE * 2;
    let mut runs = vec![(3, SOUND_DEGREE as u64), (4095, rows as u64)];
    for _column in 0..SOUND_DEGREE {
        runs.extend([(0, 1), (1000, 1), (4095, (rows - 1) as u64)]);
    }
    let mut scripted = ScriptedRng::new(&runs);

    let sampled =
        IrreducibleRingLpn::<FIELD>::sample(SOUND_DEGREE, SOUND_WEIGHT, &mut scripted).unwrap();
    let sparse = sampled.instance().sparse_matrix();

    // Exactly the scripted words were consumed: 2048 for the multiplier, one
    // wasted 4096-word attempt for column 0, and 4097 words per column. A
    // whole-matrix retry would consume exactly twice as much.
    let expected_draws =
        SOUND_DEGREE as u64 + rows as u64 + SOUND_DEGREE as u64 * (rows as u64 + 1);
    assert_eq!(scripted.draws, expected_draws);
    assert_eq!(sampled.instance().multiplier(), vec![3; SOUND_DEGREE]);

    // Every column holds exactly the scripted support: one entry at row 0
    // with value 1000 % (p - 1) + 1 = 1001.
    assert_eq!(sparse.offsets(), &(0..=SOUND_DEGREE).collect::<Vec<_>>());
    assert_eq!(sparse.row_indices(), &vec![0; SOUND_DEGREE]);
    assert_eq!(
        sparse
            .values()
            .iter()
            .map(|value| value.value())
            .collect::<Vec<_>>(),
        vec![1001; SOUND_DEGREE]
    );

    // The seeded real RNG stays deterministic: replaying the sample with the
    // same ChaCha seed reproduces the instance exactly.
    let mut first_rng = ChaCha20Rng::from_seed([71; 32]);
    let mut second_rng = ChaCha20Rng::from_seed([71; 32]);
    let first =
        IrreducibleRingLpn::<FIELD>::sample(SOUND_DEGREE, SOUND_WEIGHT, &mut first_rng).unwrap();
    let second =
        IrreducibleRingLpn::<FIELD>::sample(SOUND_DEGREE, SOUND_WEIGHT, &mut second_rng).unwrap();
    assert_eq!(
        first.instance().multiplier(),
        second.instance().multiplier()
    );
    assert_eq!(
        first.instance().sparse_matrix(),
        second.instance().sparse_matrix()
    );
}

#[test]
fn ring_lpn_empty_column_retry_budget_is_bounded() {
    // The constant word 255 maps to row 255 for every row draw, which is
    // never below the weight 192, so every attempt of every column comes out
    // empty and the per-column budget must stop the loop deterministically.
    let mut scripted = ScriptedRng::new(&[(255, u64::MAX)]);
    assert!(matches!(
        IrreducibleRingLpn::<FIELD>::sample(SOUND_DEGREE, SOUND_WEIGHT, &mut scripted),
        Err(TdmError::SamplingRetryBudgetExhausted { retries: 1024 })
    ));
    // 2048 multiplier draws, then 1024 attempts of 4096 row draws on the
    // first column.
    assert_eq!(
        scripted.draws,
        SOUND_DEGREE as u64 + 1024 * (SOUND_DEGREE * 2) as u64
    );
}

#[test]
fn ring_lpn_ntt_reduction_path_matches_oracles_at_degree_32() {
    let k = 32;
    let rows = 64;
    // K = 32 exceeds the schoolbook cutoff, so the extension multiplication
    // runs the NTT path with a length-64 transform; confirm the kernel
    // before checking the arithmetic against independent oracles.
    let modulus = automatic_ring_modulus::<FIELD>(k).unwrap();
    assert_eq!(
        ExtensionField::<FIELD>::new_unchecked_irreducible(k, &modulus)
            .unwrap()
            .algorithm(),
        PolynomialAlgorithm::Ntt {
            transform_length: 64
        }
    );

    let field = PrimeField::<FIELD>::new();
    let mut rng = ChaCha20Rng::seed_from_u64(0x4e00);
    let multiplier: Vec<u32> = (0..k)
        .map(|_| field.sample_uniform(&mut rng).value())
        .collect();
    let mut row_indices = Vec::new();
    let mut offsets = vec![0usize];
    let mut entry_values = Vec::new();
    for column in 0..k {
        for entry in 0..4usize {
            row_indices.push((5 * column + 3 * entry + 1) % rows);
            entry_values.push(u32::try_from((17 * column + 11 * entry) % 1_000).unwrap() + 1);
        }
        offsets.push(row_indices.len());
    }
    let sparse = SparseMatrix::new(
        rows,
        k,
        offsets,
        row_indices,
        elements::<FIELD>(&entry_values),
    )
    .unwrap();
    let instance =
        IrreducibleRingLpn::<FIELD>::new_unchecked_irreducible(k, &modulus, &multiplier, sparse)
            .unwrap();

    // Independent dense oracle for the secret sparse matrix E.
    let secret = instance.sparse_matrix();
    let mut dense = vec![0u32; rows * k];
    for column in 0..k {
        for entry in secret.offsets()[column]..secret.offsets()[column + 1] {
            dense[secret.row_indices()[entry] * k + column] = secret.values()[entry].value();
        }
    }

    // Independent polynomial oracle: H E x = u0 + a * u1 mod f. The instance
    // keeps its own canonicalized copy of the multiplier, so the fixture
    // vector can move into the fixed-size array.
    let multiplier_array: [u32; 32] = multiplier.try_into().unwrap();
    let ring_oracle = |input: &[u32]| -> Vec<u32> {
        let sparse_product = dense_apply(rows, k, &dense, input, FIELD);
        let u0: [u32; 32] = sparse_product[..k].try_into().unwrap();
        let u1: [u32; 32] = sparse_product[k..].try_into().unwrap();
        let reduced = polynomial_mul_reduce::<32>(&multiplier_array, &u1, &modulus, FIELD);
        u0.iter()
            .zip(reduced)
            .map(|(&identity, product)| add_mod(identity, product, FIELD))
            .collect()
    };

    let input: Vec<u32> = (0..k)
        .map(|index| (index as u32 * 123_457 + 9_876_543) % FIELD)
        .collect();
    let mut actual = elements::<FIELD>(&vec![0; k]);
    instance
        .apply(
            &elements::<FIELD>(&input),
            &mut actual,
            &mut instance.scratch(),
        )
        .unwrap();
    assert_eq!(values(&actual), ring_oracle(&input));

    // The dense materialization agrees with the oracle column by column, so
    // the whole NTT-mediated map is checked, not just one application.
    let materialized = instance.materialize().unwrap();
    let expected_matrix = materialize_oracle(k, FIELD, ring_oracle);
    assert_eq!(values(materialized.values()), expected_matrix);

    // Linearity of the implementation at NTT degree.
    let x: Vec<u32> = (0..k)
        .map(|index| (index as u32 * 31 + 5) % FIELD)
        .collect();
    let y: Vec<u32> = (0..k)
        .map(|index| (index as u32 * 17 + 800) % FIELD)
        .collect();
    let sum: Vec<u32> = x
        .iter()
        .zip(&y)
        .map(|(&lhs, &rhs)| add_mod(lhs, rhs, FIELD))
        .collect();
    let mut rx = elements::<FIELD>(&vec![0; k]);
    let mut ry = elements::<FIELD>(&vec![0; k]);
    let mut rxy = elements::<FIELD>(&vec![0; k]);
    let mut scratch = instance.scratch();
    instance
        .apply(&elements::<FIELD>(&x), &mut rx, &mut scratch)
        .unwrap();
    instance
        .apply(&elements::<FIELD>(&y), &mut ry, &mut scratch)
        .unwrap();
    instance
        .apply(&elements::<FIELD>(&sum), &mut rxy, &mut scratch)
        .unwrap();
    let expected_sum: Vec<_> = values(&rx)
        .iter()
        .zip(values(&ry))
        .map(|(&lhs, rhs)| add_mod(lhs, rhs, FIELD))
        .collect();
    assert_eq!(values(&rxy), expected_sum);
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
    let mut rx = elements(&[0; 3]);
    let mut ry = elements(&[0; 3]);
    let mut rxy = elements(&[0; 3]);
    let mut scratch = product.scratch();
    product.apply(&elements(&x), &mut rx, &mut scratch).unwrap();
    product.apply(&elements(&y), &mut ry, &mut scratch).unwrap();
    product
        .apply(&elements(&sum), &mut rxy, &mut scratch)
        .unwrap();
    let expected_sum: Vec<_> = values(&rx)
        .iter()
        .zip(values(&ry))
        .map(|(&lhs, rhs)| add_mod(lhs, rhs, 17))
        .collect();
    assert_eq!(values(&rxy), expected_sum);
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
