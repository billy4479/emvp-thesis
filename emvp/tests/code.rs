#![expect(
    clippy::unwrap_used,
    reason = "fixed test fixtures establish that construction and encoding must succeed"
)]

use emvp::{CodeError, CyclicDualCode};
use prime_field_layer::{FieldElement, PrimeField};
use proptest::prelude::*;
use rand_chacha::ChaCha20Rng;
use rand_core::{Rng, SeedableRng};

// 1_073_479_681 - 1 = 2^18 * 3^2 * 5 * 7 * 13, so the field supports the
// power-of-two transform length next_pow2(2k - 1) of every `k` exercised
// below; the largest is k = 17, needing a length-64 plan.
const MODULUS: u32 = 1_073_479_681;

fn elements(values: &[u32]) -> Vec<FieldElement<MODULUS>> {
    let field = PrimeField::<MODULUS>::new();
    values
        .iter()
        .map(|&value| field.element_u32(value))
        .collect()
}

fn values(elements: &[FieldElement<MODULUS>]) -> Vec<u32> {
    elements.iter().map(|element| element.value()).collect()
}

fn add_mod(lhs: u32, rhs: u32) -> u32 {
    u32::try_from((u64::from(lhs) + u64::from(rhs)) % u64::from(MODULUS)).unwrap()
}

fn mul_mod(lhs: u32, rhs: u32) -> u32 {
    u32::try_from(u64::from(lhs) * u64::from(rhs) % u64::from(MODULUS)).unwrap()
}

const fn neg_mod(value: u32) -> u32 {
    (MODULUS - value % MODULUS) % MODULUS
}

fn reverse(values: &[u32]) -> Vec<u32> {
    values.iter().rev().copied().collect()
}

/// `(M_g v)[i] = sum_j g[(i - j) mod k] * v[j]`, the forward cyclic
/// convolution with the circulant convention `(M_g)[i][j] = g[(i - j) mod k]`.
fn cyclic_convolution_oracle(g: &[u32], v: &[u32]) -> Vec<u32> {
    let k = g.len();
    (0..k)
        .map(|row| {
            (0..k)
                .map(|column| mul_mod(g[(row + k - column) % k], v[column]))
                .fold(0, add_mod)
        })
        .collect()
}

/// `(M_g^T r)[i] = sum_j g[(j - i) mod k] * r[j]`, the actual transpose with
/// the `(M_g^T)[i][j] = M_g[j][i]` convention.
fn transposed_convolution_oracle(g: &[u32], r: &[u32]) -> Vec<u32> {
    let k = g.len();
    (0..k)
        .map(|row| {
            (0..k)
                .map(|column| mul_mod(g[(column + k - row) % k], r[column]))
                .fold(0, add_mod)
        })
        .collect()
}

fn sample_code(k: usize, seed: u64) -> CyclicDualCode<MODULUS> {
    let mut rng = ChaCha20Rng::seed_from_u64(seed);
    CyclicDualCode::sample(k, &mut rng).unwrap()
}

fn zeros(length: usize) -> Vec<FieldElement<MODULUS>> {
    vec![PrimeField::<MODULUS>::new().element_u32(0); length]
}

#[test]
fn direct_cyclic_ntt_works_when_linear_fallback_is_unsupported() {
    let field = PrimeField::<17>::new();
    let zero = field.element_u32(0);
    let mut multiplier = vec![zero; 16];
    multiplier[0] = field.element_u32(1);
    let code = CyclicDualCode::<17>::new(16, multiplier).unwrap();
    let row: Vec<_> = (1..=16).map(|value| field.element_u32(value)).collect();
    let mut output = vec![zero; 32];
    let mut scratch = code.scratch();

    code.dual_encode_row(&row, &mut output, &mut scratch)
        .unwrap();

    for index in 0..16 {
        assert_eq!(output[index], -row[index]);
    }
    assert_eq!(&output[16..], row);
}

#[test]
fn codeword_image_matches_both_transpose_conventions() {
    for k in [1, 2, 5, 8, 17] {
        let code = sample_code(k, 0x1000 + k as u64);
        let g = values(code.multiplier());
        let mut scratch = code.scratch();

        let mut rng = ChaCha20Rng::seed_from_u64(0x2000 + k as u64);
        let mut codeword = zeros(code.n());
        code.sample_codeword(&mut rng, &mut codeword, &mut scratch)
            .unwrap();
        let message = values(&codeword[..k]);
        let image = values(&codeword[k..]);

        // J (M_g (J r)), the reversal identity the module implements.
        let reversal_form = reverse(&cyclic_convolution_oracle(&g, &reverse(&message)));
        assert_eq!(image, reversal_form);
        // O^T r with the entrywise transpose convention (O^T)[i][j] = O[j][i].
        assert_eq!(image, transposed_convolution_oracle(&g, &message));
    }
}

#[test]
fn codewords_are_orthogonal_to_the_explicit_dual() {
    for k in [1, 2, 5, 8, 17] {
        let code = sample_code(k, 0x3000 + k as u64);
        let g = values(code.multiplier());
        let mut scratch = code.scratch();

        for trial in 0..3u64 {
            let mut rng = ChaCha20Rng::seed_from_u64(0x4000 + 8 * k as u64 + trial);
            let mut codeword = zeros(code.n());
            code.sample_codeword(&mut rng, &mut codeword, &mut scratch)
                .unwrap();
            let c = values(&codeword);

            // (D c)[i] = sum_{j < k} -M_g[j][i] * c[j] + c[k + i].
            for row in 0..k {
                let mut sum = (0..k)
                    .map(|column| mul_mod(neg_mod(g[(column + k - row) % k]), c[column]))
                    .fold(0, add_mod);
                sum = add_mod(sum, c[k + row]);
                assert_eq!(sum, 0, "dual check failed at k={k} trial={trial} row={row}");
            }
        }
    }
}

#[test]
fn dual_encoding_matches_the_explicit_matrix_product() {
    for k in [1, 2, 5, 8, 17] {
        let code = sample_code(k, 0x5000 + k as u64);
        let g = values(code.multiplier());
        let mut scratch = code.scratch();

        for columns in 0..=k {
            let mut rng = ChaCha20Rng::seed_from_u64(0x6000 + 64 * k as u64 + columns as u64);
            let row: Vec<u32> = (0..columns).map(|_| rng.next_u32()).collect();
            let mut out = zeros(code.n());
            code.dual_encode_row(&elements(&row), &mut out, &mut scratch)
                .unwrap();
            let encoded = values(&out);

            // (-m^T M_g^T)[j] = -sum_i m[i] * g[(j - i) mod k].
            for column in 0..k {
                let expected = (0..columns)
                    .map(|index| mul_mod(row[index], g[(column + k - index) % k]))
                    .fold(0, add_mod);
                assert_eq!(
                    encoded[column],
                    neg_mod(expected),
                    "dual encoding failed at k={k} columns={columns} column={column}"
                );
            }
            for index in 0..k {
                let expected = row.get(index).map_or(0, |&value| value % MODULUS);
                assert_eq!(encoded[k + index], expected);
            }
        }
    }
}

#[test]
fn repeated_calls_with_one_scratch_are_deterministic() {
    let code = sample_code(6, 0x7000);
    let mut scratch = code.scratch();

    let mut first = zeros(code.n());
    let mut second = zeros(code.n());
    let mut first_rng = ChaCha20Rng::seed_from_u64(0x7001);
    let mut second_rng = ChaCha20Rng::seed_from_u64(0x7001);
    code.sample_codeword(&mut first_rng, &mut first, &mut scratch)
        .unwrap();
    code.sample_codeword(&mut second_rng, &mut second, &mut scratch)
        .unwrap();
    assert_eq!(values(&first), values(&second));

    let field = PrimeField::<MODULUS>::new();
    let mut rng = ChaCha20Rng::seed_from_u64(0x7002);
    let row: Vec<_> = (0..code.k())
        .map(|_| field.sample_uniform(&mut rng))
        .collect();
    let mut encoded_first = zeros(code.n());
    let mut encoded_second = zeros(code.n());
    code.dual_encode_row(&row, &mut encoded_first, &mut scratch)
        .unwrap();
    code.dual_encode_row(&row, &mut encoded_second, &mut scratch)
        .unwrap();
    assert_eq!(values(&encoded_first), values(&encoded_second));

    // Interleaving both operations through one scratch stays correct.
    let mut interleaved = zeros(code.n());
    let mut interleaved_rng = ChaCha20Rng::seed_from_u64(0x7001);
    code.sample_codeword(&mut interleaved_rng, &mut interleaved, &mut scratch)
        .unwrap();
    assert_eq!(values(&interleaved), values(&first));
}

#[test]
fn invalid_inputs_are_rejected_without_mutating_outputs() {
    assert!(matches!(
        CyclicDualCode::<MODULUS>::new(0, Vec::new()),
        Err(CodeError::ZeroDimension("code dimension k"))
    ));
    assert!(matches!(
        CyclicDualCode::<MODULUS>::new(3, elements(&[1, 2])),
        Err(CodeError::LengthMismatch {
            name: "code multiplier",
            expected: 3,
            actual: 2
        })
    ));
    let mut oversized_rng = ChaCha20Rng::seed_from_u64(0x7fff);
    assert!(matches!(
        CyclicDualCode::<MODULUS>::sample(usize::MAX, &mut oversized_rng),
        Err(CodeError::DimensionOverflow)
    ));

    let code = sample_code(4, 0x8000);
    let mut scratch = code.scratch();

    let mut untouched = zeros(code.n());
    let long_row = zeros(5);
    assert!(matches!(
        code.dual_encode_row(&long_row, &mut untouched, &mut scratch),
        Err(CodeError::LengthMismatch {
            name: "dual-encoding row",
            expected: 4,
            actual: 5
        })
    ));
    assert_eq!(values(&untouched), vec![0; 8]);

    let mut short_output = zeros(7);
    let mut rng = ChaCha20Rng::seed_from_u64(0x8001);
    assert!(matches!(
        code.dual_encode_row(&zeros(4), &mut short_output, &mut scratch),
        Err(CodeError::LengthMismatch {
            name: "dual encoding output",
            expected: 8,
            actual: 7
        })
    ));
    assert!(matches!(
        code.sample_codeword(&mut rng, &mut short_output, &mut scratch),
        Err(CodeError::LengthMismatch {
            name: "codeword output",
            expected: 8,
            actual: 7
        })
    ));
    assert_eq!(values(&short_output), vec![0; 7]);
}

proptest! {
    #![proptest_config(ProptestConfig::with_cases(48))]

    #[test]
    fn codeword_image_matches_transpose_for_arbitrary_entries(
        g in prop::collection::vec(any::<u32>(), 5),
        r in prop::collection::vec(any::<u32>(), 5),
    ) {
        let code = CyclicDualCode::<MODULUS>::new(5, elements(&g)).unwrap();
        let mut scratch = code.scratch();
        let mut out = zeros(code.n());

        // Encoding the reversed message exposes conv(g, J r) as -left; the
        // codeword image of r is J conv(g, J r).
        code.dual_encode_row(&elements(&reverse(&r)), &mut out, &mut scratch)
            .unwrap();
        let image: Vec<u32> = values(&out[..5])
            .iter()
            .rev()
            .map(|&value| neg_mod(value))
            .collect();

        prop_assert_eq!(image.as_slice(), transposed_convolution_oracle(&g, &r));
        prop_assert_eq!(image, reverse(&cyclic_convolution_oracle(&g, &reverse(&r))));
    }
}
