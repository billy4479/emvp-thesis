use rand_core::CryptoRng;

use crate::TdmError;

/// A public permutation in gather form: `output[i] = input[indices[i]]`.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Permutation {
    indices: Vec<usize>,
}

impl Permutation {
    /// Validates and retains a gather permutation.
    ///
    /// # Errors
    ///
    /// Returns an error for a duplicate or out-of-range index.
    pub fn new(indices: Vec<usize>) -> Result<Self, TdmError> {
        let mut seen = vec![false; indices.len()];
        for (position, &index) in indices.iter().enumerate() {
            let Some(slot) = seen.get_mut(index) else {
                return Err(TdmError::InvalidPermutation { position, index });
            };
            if *slot {
                return Err(TdmError::InvalidPermutation { position, index });
            }
            *slot = true;
        }
        Ok(Self { indices })
    }

    /// Samples an unbiased Fisher-Yates permutation from a caller-owned CSPRNG.
    ///
    /// Sampling uses rejection and therefore consumes a variable number of RNG
    /// words. The maximum supported length is `2^32`.
    ///
    /// # Errors
    ///
    /// Returns an error if a shuffle bound cannot be represented by the sampler.
    pub fn sample<R: CryptoRng + ?Sized>(length: usize, rng: &mut R) -> Result<Self, TdmError> {
        if u64::try_from(length).map_or(true, |value| value > (1u64 << u32::BITS)) {
            return Err(TdmError::SamplingRangeTooLarge {
                upper_bound: length,
            });
        }
        let mut indices: Vec<_> = (0..length).collect();
        for upper in (2..=length).rev() {
            let selected = sample_below(rng, upper)?;
            indices.swap(upper - 1, selected);
        }
        Ok(Self { indices })
    }

    /// Returns the gather indices.
    #[must_use]
    pub fn indices(&self) -> &[usize] {
        &self.indices
    }

    /// Returns the permutation length.
    #[must_use]
    pub const fn len(&self) -> usize {
        self.indices.len()
    }

    /// Returns whether this is the empty permutation.
    #[must_use]
    pub const fn is_empty(&self) -> bool {
        self.indices.is_empty()
    }

    /// Applies the gather without allocation.
    ///
    /// # Errors
    ///
    /// Returns before mutation if either slice has the wrong length.
    pub fn apply<T: Copy>(&self, input: &[T], output: &mut [T]) -> Result<(), TdmError> {
        super::error::check_len("permutation input", self.len(), input.len())?;
        super::error::check_len("permutation output", self.len(), output.len())?;
        for (result, &index) in output.iter_mut().zip(&self.indices) {
            *result = input[index];
        }
        Ok(())
    }
}

pub fn sample_below<R: CryptoRng + ?Sized>(
    rng: &mut R,
    upper_bound: usize,
) -> Result<usize, TdmError> {
    let bound = u64::try_from(upper_bound)
        .map_err(|_conversion_error| TdmError::SamplingRangeTooLarge { upper_bound })?;
    let source_cardinality = 1u64 << u32::BITS;
    if bound == 0 || bound > source_cardinality {
        return Err(TdmError::SamplingRangeTooLarge { upper_bound });
    }
    let acceptance_bound = source_cardinality / bound * bound;
    loop {
        let candidate = u64::from(rng.next_u32());
        if candidate < acceptance_bound {
            return usize::try_from(candidate % bound)
                .map_err(|_conversion_error| TdmError::SamplingRangeTooLarge { upper_bound });
        }
    }
}
