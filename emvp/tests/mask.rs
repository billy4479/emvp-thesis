#![expect(
    clippy::unwrap_used,
    reason = "fixed test fixtures establish that construction and evaluation must succeed"
)]

use emvp::{MaskError, RowStackMask, TdmMask};
use prime_field_layer::{FieldElement, PrimeField};
use proptest::prelude::*;
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use trapdoor_matrices::{
    DenseMatrix, IrreducibleRingLpn, RaaWeightedProduct, SparseMatrix, TdmError,
    ToeplitzFastProduct,
};

// NTT-friendly prime for the Toeplitz and RAA constructions: 998_244_353 - 1
// is divisible by 2^23, so every transform length below is supported.
const MODULUS: u32 = 998_244_353;

// Small prime for the Ring-LPN adapter, with the irreducible cubic
// x^3 + 3x + 1 copied from the trapdoor-matrices integration tests.
const SMALL_MODULUS: u32 = 17;
const RING_MODULUS_POLY: [u32; 4] = [1, 3, 0, 1];

fn elements<const MODULUS: u32>(values: &[u32]) -> Vec<FieldElement<MODULUS>> {
    let field = PrimeField::<MODULUS>::new();
    values
        .iter()
        .map(|&value| field.element_u32(value))
        .collect()
}

fn field_values<const MODULUS: u32>(elements: &[FieldElement<MODULUS>]) -> Vec<u32> {
    elements.iter().map(|element| element.value()).collect()
}

fn zeros<const MODULUS: u32>(length: usize) -> Vec<FieldElement<MODULUS>> {
    vec![PrimeField::<MODULUS>::new().element_u32(0); length]
}

fn sentinel_values<const MODULUS: u32>(length: usize, value: u32) -> Vec<FieldElement<MODULUS>> {
    vec![PrimeField::<MODULUS>::new().element_u32(value); length]
}

fn random_vector<const MODULUS: u32>(
    length: usize,
    rng: &mut ChaCha20Rng,
) -> Vec<FieldElement<MODULUS>> {
    let field = PrimeField::<MODULUS>::new();
    (0..length).map(|_| field.sample_uniform(rng)).collect()
}

fn sample_toeplitz(k: usize, seed: u64) -> ToeplitzFastProduct<MODULUS> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    ToeplitzFastProduct::sample(k, &mut rng).unwrap()
}

fn toeplitz_stack(
    blocks: usize,
    total_rows: usize,
    seed: u64,
) -> RowStackMask<ToeplitzFastProduct<MODULUS>, MODULUS> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    let stacked: Vec<_> = (0..blocks)
        .map(|_| ToeplitzFastProduct::sample(5, &mut rng).unwrap())
        .collect();
    RowStackMask::new(stacked, total_rows).unwrap()
}

/// Independently rebuilds the dense stack from the individual blocks,
/// truncating to the mask's row count.
fn stacked_dense<M: TdmMask<MODULUS>>(stack: &RowStackMask<M, MODULUS>) -> DenseMatrix<MODULUS> {
    let (total_rows, columns) = stack.dims();
    let mut values = Vec::new();
    for block in stack.blocks() {
        values.extend_from_slice(block.materialize().unwrap().values());
    }
    values.truncate(total_rows * columns);
    DenseMatrix::new(total_rows, columns, values).unwrap()
}

fn assert_apply_matches_materialize<const MODULUS: u32, M: TdmMask<MODULUS>>(
    mask: &M,
    input: &[FieldElement<MODULUS>],
) {
    let (rows, columns) = mask.dims();
    assert_eq!(input.len(), columns);
    let dense = mask.materialize().unwrap();
    assert_eq!((dense.rows(), dense.columns()), (rows, columns));

    let mut structured = zeros::<MODULUS>(rows);
    let mut scratch = mask.scratch();
    mask.apply(input, &mut structured, &mut scratch).unwrap();

    let mut expected = zeros::<MODULUS>(rows);
    dense.apply(input, &mut expected).unwrap();
    assert_eq!(structured, expected);
}

/// Copied from the trapdoor-matrices integration tests: `K = 3` over
/// `F_17` with modulus polynomial x^3 + 3x + 1.
fn explicit_ring_lpn() -> IrreducibleRingLpn<SMALL_MODULUS> {
    let k = 3;
    let sparse = SparseMatrix::new(
        6,
        3,
        vec![0, 3, 6, 9],
        vec![0, 3, 5, 1, 3, 4, 2, 4, 5],
        elements::<SMALL_MODULUS>(&[2, 5, 1, 3, 4, 6, 7, 2, 8]),
    )
    .unwrap();
    IrreducibleRingLpn::new(k, &RING_MODULUS_POLY, &[2, 1, 3], sparse).unwrap()
}

/// A minimal test double with caller-chosen rectangular dimensions.
struct FixedBlock {
    rows: usize,
    columns: usize,
}

impl TdmMask<MODULUS> for FixedBlock {
    type Scratch = ();

    fn dims(&self) -> (usize, usize) {
        (self.rows, self.columns)
    }

    fn scratch(&self) {}

    fn apply(
        &self,
        input: &[FieldElement<MODULUS>],
        output: &mut [FieldElement<MODULUS>],
        _scratch: &mut Self::Scratch,
    ) -> Result<(), MaskError> {
        if input.len() != self.columns {
            return Err(MaskError::LengthMismatch {
                name: "fixed block input",
                expected: self.columns,
                actual: input.len(),
            });
        }
        if output.len() != self.rows {
            return Err(MaskError::LengthMismatch {
                name: "fixed block output",
                expected: self.rows,
                actual: output.len(),
            });
        }
        output.fill(PrimeField::<MODULUS>::new().element_u32(0));
        Ok(())
    }

    fn materialize(&self) -> Result<DenseMatrix<MODULUS>, MaskError> {
        let length = self
            .rows
            .checked_mul(self.columns)
            .ok_or(MaskError::DimensionOverflow)?;
        Ok(DenseMatrix::new(
            self.rows,
            self.columns,
            zeros::<MODULUS>(length),
        )?)
    }
}

struct BadMaterialization;

impl TdmMask<MODULUS> for BadMaterialization {
    type Scratch = ();

    fn dims(&self) -> (usize, usize) {
        (2, 2)
    }

    fn scratch(&self) {}

    fn apply(
        &self,
        input: &[FieldElement<MODULUS>],
        output: &mut [FieldElement<MODULUS>],
        _scratch: &mut Self::Scratch,
    ) -> Result<(), MaskError> {
        if input.len() != 2 || output.len() != 2 {
            return Err(MaskError::LengthMismatch {
                name: "bad materialization apply",
                expected: 2,
                actual: usize::min(input.len(), output.len()),
            });
        }
        output.copy_from_slice(input);
        Ok(())
    }

    fn materialize(&self) -> Result<DenseMatrix<MODULUS>, MaskError> {
        Ok(DenseMatrix::new(1, 1, zeros::<MODULUS>(1))?)
    }
}

#[test]
fn ring_lpn_adapter_matches_its_dense_materialization() {
    let instance = explicit_ring_lpn();
    assert_eq!(TdmMask::dims(&instance), (3, 3));

    let mut rng = ChaCha20Rng::seed_from_u64(0x6300);
    for _ in 0..4 {
        let input = random_vector::<SMALL_MODULUS>(3, &mut rng);
        assert_apply_matches_materialize(&instance, &input);
    }
}

#[test]
fn toeplitz_adapter_matches_its_dense_materialization() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x6400);
    let product = ToeplitzFastProduct::sample(5, &mut rng).unwrap();
    assert_eq!(TdmMask::dims(&product), (5, 5));

    for _ in 0..4 {
        let input = random_vector::<MODULUS>(5, &mut rng);
        assert_apply_matches_materialize(&product, &input);
    }
}

#[test]
fn raa_adapter_matches_its_dense_materialization() {
    let mut rng = ChaCha20Rng::seed_from_u64(0x6500);
    let product = RaaWeightedProduct::sample_nonzero(4, 3, &mut rng).unwrap();
    assert_eq!(TdmMask::dims(&product), (4, 4));

    for _ in 0..4 {
        let input = random_vector::<MODULUS>(4, &mut rng);
        assert_apply_matches_materialize(&product, &input);
    }
}

#[test]
fn exact_row_stack_matches_the_stacked_dense_blocks() {
    let stack = toeplitz_stack(3, 15, 0x6100);
    assert_eq!(stack.dims(), (15, 5));
    assert_eq!(stack.block_count(), 3);
    assert_eq!(stack.blocks().len(), 3);
    for block in stack.blocks() {
        assert_eq!(block.dims(), (5, 5));
    }

    let oracle = stacked_dense(&stack);
    assert_eq!(stack.materialize().unwrap(), oracle);

    let mut rng = ChaCha20Rng::seed_from_u64(0x6101);
    for _ in 0..4 {
        let input = random_vector::<MODULUS>(5, &mut rng);
        let mut structured = sentinel_values::<MODULUS>(15, 1);
        let mut scratch = stack.scratch();
        stack.apply(&input, &mut structured, &mut scratch).unwrap();

        let mut expected = zeros::<MODULUS>(15);
        oracle.apply(&input, &mut expected).unwrap();
        assert_eq!(field_values(&structured), field_values(&expected));
    }
}

#[test]
fn remainder_row_stack_truncates_the_last_block() {
    let stack = toeplitz_stack(3, 13, 0x6200);
    assert_eq!(stack.dims(), (13, 5));
    assert_eq!(stack.block_count(), 3);

    let oracle = stacked_dense(&stack);
    assert_eq!(stack.materialize().unwrap(), oracle);

    let mut rng = ChaCha20Rng::seed_from_u64(0x6201);
    let input = random_vector::<MODULUS>(5, &mut rng);
    let mut structured = sentinel_values::<MODULUS>(13, 2);
    let mut scratch = stack.scratch();
    stack.apply(&input, &mut structured, &mut scratch).unwrap();

    let mut expected = zeros::<MODULUS>(13);
    oracle.apply(&input, &mut expected).unwrap();
    assert_eq!(field_values(&structured), field_values(&expected));

    // The output tail is the top of the last block's product; its surplus
    // rows never appear.
    let last = &stack.blocks()[2];
    let mut last_output = zeros::<MODULUS>(5);
    let mut last_scratch = last.scratch();
    last.apply(&input, &mut last_output, &mut last_scratch)
        .unwrap();
    assert_eq!(
        field_values(&structured[10..13]),
        field_values(&last_output[..3])
    );
}

#[test]
fn row_stack_rejects_bad_block_descriptors() {
    assert!(matches!(
        RowStackMask::<ToeplitzFastProduct<MODULUS>, MODULUS>::new(Vec::new(), 5),
        Err(MaskError::LengthMismatch {
            name: "mask blocks",
            expected: 1,
            actual: 0
        })
    ));
    assert!(matches!(
        RowStackMask::new(
            vec![FixedBlock {
                rows: 2,
                columns: 3
            }],
            2
        ),
        Err(MaskError::LengthMismatch {
            name: "first mask block rows",
            expected: 3,
            actual: 2
        })
    ));
    assert!(matches!(
        RowStackMask::new(
            vec![FixedBlock {
                rows: 0,
                columns: 0
            }],
            1
        ),
        Err(MaskError::LengthMismatch {
            name: "mask block rows",
            expected: 1,
            actual: 0
        })
    ));
    assert!(matches!(
        RowStackMask::new(
            vec![
                FixedBlock {
                    rows: 5,
                    columns: 5
                },
                FixedBlock {
                    rows: 4,
                    columns: 4
                }
            ],
            5
        ),
        Err(MaskError::LengthMismatch {
            name: "mask block rows",
            expected: 5,
            actual: 4
        })
    ));
    assert!(matches!(
        RowStackMask::new(
            vec![
                FixedBlock {
                    rows: 5,
                    columns: 5
                },
                FixedBlock {
                    rows: 5,
                    columns: 4
                }
            ],
            5
        ),
        Err(MaskError::LengthMismatch {
            name: "mask block columns",
            expected: 5,
            actual: 4
        })
    ));
}

#[test]
fn row_stack_rejects_materialization_that_disagrees_with_dimensions() {
    let stack = RowStackMask::new(vec![BadMaterialization], 2).unwrap();
    assert!(matches!(
        stack.materialize(),
        Err(MaskError::LengthMismatch {
            name: "materialized tail rows",
            expected: 2,
            actual: 1,
        })
    ));
}

#[test]
fn row_stack_rejects_invalid_constructions() {
    // Real constructions with different block sizes are rejected too.
    let large = sample_toeplitz(5, 0x6601);
    let small = sample_toeplitz(3, 0x6602);
    assert!(matches!(
        RowStackMask::new(vec![large, small], 5),
        Err(MaskError::LengthMismatch {
            name: "mask block rows",
            expected: 5,
            actual: 3
        })
    ));

    let first = sample_toeplitz(5, 0x6603);
    let second = sample_toeplitz(5, 0x6604);
    assert!(matches!(
        RowStackMask::new(vec![first, second], 11),
        Err(MaskError::LengthMismatch {
            name: "total mask rows",
            expected: 10,
            actual: 11
        })
    ));
    assert!(matches!(
        RowStackMask::new(vec![sample_toeplitz(5, 0x6605)], 0),
        Err(MaskError::LengthMismatch {
            name: "total mask rows",
            expected: 1,
            actual: 0
        })
    ));
}

#[test]
fn fixed_block_double_stays_consistent_with_its_dense_form() {
    let block = FixedBlock {
        rows: 2,
        columns: 3,
    };
    let mut output = zeros::<MODULUS>(2);
    block
        .apply(&elements::<MODULUS>(&[1, 2, 3]), &mut output, &mut ())
        .unwrap();
    assert_eq!(field_values(&output), vec![0, 0]);
    assert_eq!(
        field_values(block.materialize().unwrap().values()),
        vec![0; 6]
    );
}

#[test]
fn length_errors_leave_outputs_unchanged() {
    let stack = toeplitz_stack(3, 15, 0x6800);
    let mut scratch = stack.scratch();
    let sentinel = sentinel_values::<MODULUS>(15, 9);
    let mut output = sentinel.clone();

    let wrong_input = zeros::<MODULUS>(4);
    assert!(matches!(
        stack.apply(&wrong_input, &mut output, &mut scratch),
        Err(MaskError::LengthMismatch {
            name: "mask input",
            expected: 5,
            actual: 4
        })
    ));
    assert_eq!(output, sentinel);

    let mut short_output = zeros::<MODULUS>(14);
    let input = zeros::<MODULUS>(5);
    assert!(matches!(
        stack.apply(&input, &mut short_output, &mut scratch),
        Err(MaskError::LengthMismatch {
            name: "mask output",
            expected: 15,
            actual: 14
        })
    ));
    assert_eq!(field_values(&short_output), vec![0; 14]);

    let mut short_scratch = stack.scratch();
    short_scratch.0.pop();
    assert!(matches!(
        stack.apply(&input, &mut output, &mut short_scratch),
        Err(MaskError::LengthMismatch {
            name: "mask scratch blocks",
            expected: 3,
            actual: 2
        })
    ));
    assert_eq!(output, sentinel);

    // The adapter layer translates its own length failures through
    // MaskError::Tdm without touching the output.
    let product = sample_toeplitz(5, 0x6801);
    let mut product_scratch = product.scratch();
    let adapter_sentinel = sentinel_values::<MODULUS>(5, 7);
    let mut adapter_output = adapter_sentinel.clone();
    assert!(matches!(
        TdmMask::apply(
            &product,
            &zeros::<MODULUS>(4),
            &mut adapter_output,
            &mut product_scratch
        ),
        Err(MaskError::Tdm(TdmError::LengthMismatch {
            name: "fast-product input",
            expected: 5,
            actual: 4
        }))
    ));
    assert_eq!(adapter_output, adapter_sentinel);

    let mut short_adapter_output = zeros::<MODULUS>(4);
    let adapter_input = zeros::<MODULUS>(5);
    assert!(matches!(
        TdmMask::apply(
            &product,
            &adapter_input,
            &mut short_adapter_output,
            &mut product_scratch
        ),
        Err(MaskError::Tdm(TdmError::LengthMismatch {
            name: "fast-product output",
            expected: 5,
            actual: 4
        }))
    ));
    assert_eq!(field_values(&short_adapter_output), vec![0; 4]);
}

#[test]
fn partial_row_stack_apply_allocates_nothing_after_scratch_creation() {
    let stack = toeplitz_stack(3, 13, 0x6810);
    let input = elements::<MODULUS>(&[1, 2, 3, 4, 5]);
    let mut output = zeros::<MODULUS>(13);
    let mut scratch = stack.scratch();
    stack.apply(&input, &mut output, &mut scratch).unwrap();

    let allocations = allocation_counter::measure(|| {
        for _ in 0..8 {
            stack.apply(&input, &mut output, &mut scratch).unwrap();
        }
    });
    assert_eq!(allocations.count_total, 0);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(32))]

    #[test]
    fn row_stack_matches_dense_oracle_for_arbitrary_input(
        input in prop::collection::vec(any::<u32>(), 5),
    ) {
        let values = elements::<MODULUS>(&input);
        for total_rows in [15, 13] {
            let stack = toeplitz_stack(3, total_rows, 0x6900 + total_rows as u64);
            let dense = stacked_dense(&stack);

            let mut structured = zeros::<MODULUS>(total_rows);
            let mut scratch = stack.scratch();
            stack.apply(&values, &mut structured, &mut scratch).unwrap();

            let mut expected = zeros::<MODULUS>(total_rows);
            dense.apply(&values, &mut expected).unwrap();
            prop_assert_eq!(field_values(&structured), field_values(&expected));
        }
    }
}
