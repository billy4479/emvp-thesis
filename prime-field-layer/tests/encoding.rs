use std::collections::VecDeque;

use prime_field_layer::{FIELD_ELEMENT_ENCODED_SIZE, FieldElement, PrimeField};
use rand_chacha::ChaCha8Rng;
use rand_core::{Infallible, SeedableRng, TryCryptoRng, TryRng};

#[derive(Debug)]
struct ScriptedRng {
    words: VecDeque<u32>,
    words_read: usize,
}

impl ScriptedRng {
    fn new(words: impl IntoIterator<Item = u32>) -> Self {
        Self {
            words: words.into_iter().collect(),
            words_read: 0,
        }
    }

    fn next_word(&mut self) -> u32 {
        self.words_read += 1;
        self.words.pop_front().unwrap_or_default()
    }
}

impl TryRng for ScriptedRng {
    type Error = Infallible;

    fn try_next_u32(&mut self) -> Result<u32, Infallible> {
        Ok(self.next_word())
    }

    fn try_next_u64(&mut self) -> Result<u64, Infallible> {
        Ok(u64::from(self.next_word()) | u64::from(self.next_word()) << 32)
    }

    fn try_fill_bytes(&mut self, destination: &mut [u8]) -> Result<(), Infallible> {
        for chunk in destination.chunks_mut(4) {
            let word = self.next_word().to_le_bytes();
            chunk.copy_from_slice(&word[..chunk.len()]);
        }
        Ok(())
    }
}

impl TryCryptoRng for ScriptedRng {}

fn element_values<const MODULUS: u32>(values: &[FieldElement<MODULUS>]) -> Vec<u32> {
    values.iter().map(|value| value.value()).collect()
}

#[test]
fn canonical_encoding_round_trips_boundaries() {
    let field = PrimeField::<998_244_353>::new();
    for expected in [0, 1, 255, 256, 65_535, 998_244_352] {
        let element = field.element_u32(expected);
        let encoding = element.to_canonical_le_bytes();
        assert_eq!(encoding.len(), FIELD_ELEMENT_ENCODED_SIZE);
        assert_eq!(encoding, expected.to_le_bytes());
        assert_eq!(field.from_canonical_le_bytes(encoding), Ok(element));
    }
}

#[test]
fn decoding_rejects_values_at_and_above_the_modulus() {
    let field = PrimeField::<17>::new();

    for value in [17, 18, u32::MAX] {
        let error = field
            .from_canonical_le_bytes(value.to_le_bytes())
            .unwrap_err();
        assert_eq!(error.value(), value);
        assert_eq!(error.modulus(), 17);
    }
}

#[test]
fn uniform_sampling_follows_forced_rejection_path() {
    let field = PrimeField::<17>::new();
    // floor(2^32 / 17) * 17 = 2^32 - 1, so u32::MAX is rejected.
    let mut rng = ScriptedRng::new([u32::MAX, 34]);

    assert_eq!(field.sample_uniform(&mut rng).value(), 0);
    assert_eq!(rng.words_read, 2);
}

#[test]
fn nonzero_sampling_follows_forced_rejection_path() {
    let field = PrimeField::<19>::new();
    // floor(2^32 / 18) * 18 = 2^32 - 4.
    let mut rng = ScriptedRng::new([u32::MAX, 19]);

    assert_eq!(field.sample_uniform_nonzero(&mut rng).value(), 2);
    assert_eq!(rng.words_read, 2);
}

#[test]
fn nonzero_sampling_never_returns_zero() {
    let field = PrimeField::<17>::new();
    let mut rng = ChaCha8Rng::from_seed([7; 32]);

    for _ in 0..10_000 {
        let value = field.sample_uniform_nonzero(&mut rng).value();
        assert!((1..17).contains(&value));
    }
}

#[test]
fn batch_fill_uses_caller_storage_and_handles_empty_slices() {
    let field = PrimeField::<17>::new();
    let mut uniform = [field.element(0); 5];
    let mut rng = ScriptedRng::new([0, 1, 16, 17, 18]);
    field.fill_uniform(&mut rng, &mut uniform);
    assert_eq!(element_values(&uniform), [0, 1, 16, 0, 1]);

    let mut nonzero = [field.element(0); 5];
    let mut rng = ScriptedRng::new([0, 1, 15, 16, 17]);
    field.fill_uniform_nonzero(&mut rng, &mut nonzero);
    assert_eq!(element_values(&nonzero), [1, 2, 16, 1, 2]);
    assert!(nonzero.iter().all(|value| value.value() != 0));

    let words_read = rng.words_read;
    field.fill_uniform(&mut rng, &mut []);
    field.fill_uniform_nonzero(&mut rng, &mut []);
    assert_eq!(rng.words_read, words_read);
}

#[test]
fn sampling_handles_boundary_moduli() {
    let binary = PrimeField::<2>::new();
    let mut rng = ScriptedRng::new([0, 1, u32::MAX]);
    assert_eq!(binary.sample_uniform(&mut rng).value(), 0);
    assert_eq!(binary.sample_uniform(&mut rng).value(), 1);
    assert_eq!(binary.sample_uniform_nonzero(&mut rng).value(), 1);

    let largest = PrimeField::<4_294_967_291>::new();
    let mut rng = ScriptedRng::new([u32::MAX, 4_294_967_291, 4_294_967_290]);
    assert_eq!(largest.sample_uniform(&mut rng).value(), 4_294_967_290);
    assert_eq!(rng.words_read, 3);

    let canonical_max = largest.element_u32(4_294_967_290);
    assert_eq!(
        largest.from_canonical_le_bytes(canonical_max.to_canonical_le_bytes()),
        Ok(canonical_max)
    );
    largest
        .from_canonical_le_bytes(4_294_967_291u32.to_le_bytes())
        .unwrap_err();
}

#[test]
fn deterministic_uniform_sample_has_reasonable_chi_squared() {
    const MODULUS: usize = 17;
    const EXPECTED_PER_BUCKET: u64 = 10_000;

    let field = PrimeField::<17>::new();
    let mut rng = ChaCha8Rng::from_seed([42; 32]);
    let mut counts = [0u64; MODULUS];
    for _ in 0..MODULUS as u64 * EXPECTED_PER_BUCKET {
        counts[field.sample_uniform(&mut rng).value() as usize] += 1;
    }

    // Pearson chi-squared with 16 degrees of freedom. The fixed seed makes the
    // test reproducible; 50 is a loose upper bound with p < 0.00003.
    let chi_squared_numerator: u64 = counts
        .into_iter()
        .map(|count| count.abs_diff(EXPECTED_PER_BUCKET).pow(2))
        .sum();
    assert!(chi_squared_numerator < 50 * EXPECTED_PER_BUCKET);
}
