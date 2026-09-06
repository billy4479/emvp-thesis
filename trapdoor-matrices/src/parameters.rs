//! Security-parameter assessment and automatic Ring-LPN modulus selection.
//!
//! The assessment implements the sizing analysis for the Ring-LPN TDM: the
//! decoding (information-set-decoding) estimate, the enumeration floor, the
//! statistical local-attack ceiling, and the ring-degree floor. It is a pure
//! function of `(k, weight)` and the compile-time field, so every threshold
//! is unit-testable.
//!
//! Nothing here is a proof of security. The construction has no settled
//! parameter set; the assessment encodes current best-known-attack
//! estimates against a target of [`TARGET_SECURITY_BITS`] bits.

use crate::TdmError;

/// The target security level for parameter assessment, in bits.
///
/// This matches the `lambda = 128` used by the EMVP protocol crate.
pub const TARGET_SECURITY_BITS: u32 = 128;

/// Minimum ring degree for automatic sampling.
///
/// Small extension rings are the regime where algebraic Ring-LPN attacks
/// (as experienced by early Lapin parameter sets) apply.
pub const RING_DEGREE_FLOOR: usize = 2048;

/// Plausibility floor for the expected per-column noise weight.
///
/// The dual-LPN pseudorandomness of `R = HE` requires superlogarithmic
/// column weight, and the EMVP paper (Section 5.2) considers the assumption
/// plausible only at around 100 nonzero entries per column or more.
pub const PLAUSIBILITY_WEIGHT_FLOOR: usize = 100;

/// Project-policy floor for the expected per-column noise weight.
///
/// Weights passing the generic-attack estimates but below this floor are
/// sound yet leave little headroom for structure-specific attacks, whose
/// cost has no reliable estimate. The project default is above this floor.
pub const POLICY_WEIGHT_FLOOR: usize = 192;

/// Pessimistic flat discount applied to the exponential part of the
/// information-set-decoding cost, standing in for refinements such as
/// Stern's algorithm and the representation technique (MMT, BJMM). Over
/// large fields these refinements gain less than over binary fields, so
/// the discount is deliberately conservative.
const ISD_REFINEMENT_DISCOUNT: f64 = 0.8;

/// The assessed soundness of a parameter pair.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SecurityLevel {
    /// At least one known attack family defeats the target.
    Broken,
    /// The undiscounted estimates clear the target but the pessimistic
    /// ISD-refinement discount does not.
    Marginal,
    /// All known-attack estimates clear the target, including the
    /// pessimistic discount.
    Sound,
}

/// One concrete reason a parameter pair fell short of the assessment policy.
#[derive(Clone, Debug, PartialEq)]
pub enum ParameterWarning {
    /// The ring degree is in the regime of known algebraic ring attacks.
    RingDegreeBelowFloor {
        /// Assessed extension degree.
        degree: usize,
        /// Minimum supported degree.
        floor: usize,
    },
    /// The expected column weight is below the paper's plausibility floor.
    WeightBelowPlausibilityFloor {
        /// Assessed expected column weight.
        weight: usize,
        /// Minimum plausible weight.
        floor: usize,
    },
    /// The undiscounted information-set-decoding estimate is below target.
    DecodingCostBelowTarget {
        /// Estimated cost in bits.
        log2_cost: f64,
        /// Target security in bits.
        target_bits: u32,
    },
    /// Exhaustive support enumeration is below target.
    EnumerationCostBelowTarget {
        /// Estimated cost in bits.
        log2_cost: f64,
        /// Target security in bits.
        target_bits: u32,
    },
    /// The NTT length needed by the extension multiplication exceeds the
    /// field's two-adicity, so evaluation would fail at run time.
    NttLengthUnsupported {
        /// Required transform length.
        ntt_length: usize,
        /// Largest power of two dividing `MODULUS - 1`.
        two_adicity: u32,
    },
    /// The pessimistic ISD-refinement estimate is below target.
    IsdRefinementMarginBelowTarget {
        /// Estimated cost in bits.
        log2_cost: f64,
        /// Target security in bits.
        target_bits: u32,
    },
    /// The weight clears every estimate but is below the project policy
    /// floor, leaving little headroom for unquantified structure attacks.
    WeightBelowPolicyDefault {
        /// Assessed expected column weight.
        weight: usize,
        /// Policy floor.
        floor: usize,
    },
    /// The degree arithmetic overflows the platform, so no assessment of
    /// the decoding regime is possible.
    DegreeBeyondPlatform {
        /// Assessed extension degree.
        degree: usize,
    },
    /// The expected weight exceeds the degree, which is outside the sparse
    /// noise regime this construction models.
    WeightOutsideSparseRegime {
        /// Assessed expected column weight.
        weight: usize,
        /// Maximum supported weight.
        maximum: usize,
    },
}

impl std::fmt::Display for ParameterWarning {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::RingDegreeBelowFloor { degree, floor } => {
                write!(formatter, "ring degree {degree} is below the floor {floor}")
            }
            Self::WeightBelowPlausibilityFloor { weight, floor } => {
                write!(
                    formatter,
                    "expected column weight {weight} is below the plausibility floor {floor}"
                )
            }
            Self::DecodingCostBelowTarget {
                log2_cost,
                target_bits,
            } => write!(
                formatter,
                "decoding estimate 2^{log2_cost:.1} is below the 2^{target_bits} target"
            ),
            Self::EnumerationCostBelowTarget {
                log2_cost,
                target_bits,
            } => write!(
                formatter,
                "enumeration estimate 2^{log2_cost:.1} is below the 2^{target_bits} target"
            ),
            Self::NttLengthUnsupported {
                ntt_length,
                two_adicity,
            } => write!(
                formatter,
                "NTT length {ntt_length} exceeds the field two-adicity 2^{two_adicity}"
            ),
            Self::IsdRefinementMarginBelowTarget {
                log2_cost,
                target_bits,
            } => write!(
                formatter,
                "ISD-refinement-discounted estimate 2^{log2_cost:.1} is below the \
                 2^{target_bits} target"
            ),
            Self::WeightBelowPolicyDefault { weight, floor } => write!(
                formatter,
                "expected column weight {weight} is below the policy floor {floor}; \
                 consider more headroom for structure attacks"
            ),
            Self::DegreeBeyondPlatform { degree } => {
                write!(formatter, "degree {degree} overflows platform arithmetic")
            }
            Self::WeightOutsideSparseRegime { weight, maximum } => write!(
                formatter,
                "expected column weight {weight} exceeds the sparse-regime maximum {maximum}"
            ),
        }
    }
}

/// The outcome of assessing one `(degree, weight)` pair.
#[derive(Clone, Debug)]
pub struct SecurityAssessment {
    /// The weakest level triggered by the assessment.
    pub level: SecurityLevel,
    /// Every triggered warning, in assessment order.
    pub warnings: Vec<ParameterWarning>,
}

/// Assesses a Ring-LPN parameter pair against the target security level.
///
/// The assessment models the attacker's problem as syndrome decoding for a
/// length-`2k` code of dimension `k` over the compile-time field with an
/// error of expected weight `weight`:
///
/// - *Decoding:* information-set decoding needs about
///   `prod_{i<weight} (2k-i)/(k-i)` iterations, each costing a Gaussian
///   elimination of roughly `2k^3` field operations.
/// - *Enumeration:* every weight-`w` support is
///   `binom(2k, weight)` candidates.
/// - *Structure:* degrees below [`RING_DEGREE_FLOOR`] and weights below
///   [`PLAUSIBILITY_WEIGHT_FLOOR`] are rejected outright.
/// - *Run time:* the NTT length `next_power_of_two(2k - 1)` must fit in the
///   field's two-adicity or evaluation fails during extension
///   multiplication.
///
/// Weights above the degree are outside the assessed regime and report
/// [`SecurityLevel::Broken`]. The assessment never panics for any input.
#[must_use]
pub fn assess<const MODULUS: u32>(k: usize, weight: usize) -> SecurityAssessment {
    let mut warnings = Vec::new();

    if k < RING_DEGREE_FLOOR {
        warnings.push(ParameterWarning::RingDegreeBelowFloor {
            degree: k,
            floor: RING_DEGREE_FLOOR,
        });
    }
    if weight < PLAUSIBILITY_WEIGHT_FLOOR {
        warnings.push(ParameterWarning::WeightBelowPlausibilityFloor {
            weight,
            floor: PLAUSIBILITY_WEIGHT_FLOOR,
        });
    }
    if weight > k {
        warnings.push(ParameterWarning::WeightOutsideSparseRegime { weight, maximum: k });
    }

    let doubled_rows = k.checked_mul(2);
    if doubled_rows.is_none() {
        warnings.push(ParameterWarning::DegreeBeyondPlatform { degree: k });
    }
    let rows = doubled_rows.filter(|_| k > 0 && weight > 0 && weight <= k);

    if let Some(rows) = rows {
        let decoding_iterations = information_set_decoding_iterations(k, weight);
        let elimination = 3.0f64.mul_add(bits(k).log2(), 1.0);
        let decoding_cost = decoding_iterations + elimination;
        if decoding_cost < f64::from(TARGET_SECURITY_BITS) {
            warnings.push(ParameterWarning::DecodingCostBelowTarget {
                log2_cost: decoding_cost,
                target_bits: TARGET_SECURITY_BITS,
            });
        }

        let enumeration_cost = enumeration_cost(rows, weight);
        if enumeration_cost < f64::from(TARGET_SECURITY_BITS) {
            warnings.push(ParameterWarning::EnumerationCostBelowTarget {
                log2_cost: enumeration_cost,
                target_bits: TARGET_SECURITY_BITS,
            });
        }

        let ntt_length = (rows - 1).next_power_of_two();
        let two_adicity = MODULUS.saturating_sub(1).trailing_zeros();
        if u64::from(u32::try_from(ntt_length).unwrap_or(u32::MAX)) > 1u64 << two_adicity {
            warnings.push(ParameterWarning::NttLengthUnsupported {
                ntt_length,
                two_adicity,
            });
        }

        // Refinement-discount and policy-floor checks only add signal on
        // otherwise-clean assessments; a structurally broken pair is already
        // reported at its strongest warning.
        if !warnings.iter().any(is_broken_warning) {
            let discounted_cost = ISD_REFINEMENT_DISCOUNT.mul_add(decoding_iterations, elimination);
            if discounted_cost < f64::from(TARGET_SECURITY_BITS) {
                warnings.push(ParameterWarning::IsdRefinementMarginBelowTarget {
                    log2_cost: discounted_cost,
                    target_bits: TARGET_SECURITY_BITS,
                });
            } else if weight < POLICY_WEIGHT_FLOOR {
                warnings.push(ParameterWarning::WeightBelowPolicyDefault {
                    weight,
                    floor: POLICY_WEIGHT_FLOOR,
                });
            }
        }
    }

    let level = if warnings.iter().any(is_broken_warning) {
        SecurityLevel::Broken
    } else if warnings.iter().any(|warning| {
        matches!(
            warning,
            ParameterWarning::IsdRefinementMarginBelowTarget { .. }
        )
    }) {
        SecurityLevel::Marginal
    } else {
        SecurityLevel::Sound
    };

    SecurityAssessment { level, warnings }
}

const fn is_broken_warning(warning: &ParameterWarning) -> bool {
    !matches!(
        warning,
        ParameterWarning::IsdRefinementMarginBelowTarget { .. }
            | ParameterWarning::WeightBelowPolicyDefault { .. }
    )
}

/// Exact log2 of the Prange iteration count for weight-`weight` errors.
///
/// The probability that a random size-`k` information set of the `2k` rows
/// avoids all `weight` error positions is `binom(2k-weight, k) / binom(2k, k)`,
/// so the iteration count is its reciprocal. The product form avoids gamma
/// functions and is exact up to floating-point rounding.
fn information_set_decoding_iterations(k: usize, weight: usize) -> f64 {
    let rows = bits(k) * 2.0;
    let degree = bits(k);
    let mut log2_iterations = 0.0;
    for offset in 0..weight {
        let offset_bits = bits(offset);
        log2_iterations += (rows - offset_bits).log2() - (degree - offset_bits).log2();
    }
    log2_iterations
}

/// Exact log2 of the number of weight-`weight` supports among `rows` rows.
fn enumeration_cost(rows: usize, weight: usize) -> f64 {
    let total = bits(rows);
    let weight_bits = bits(weight);
    let mut log2_count = 0.0;
    for offset in 0..weight {
        let offset_bits = bits(offset);
        log2_count += (total - offset_bits).log2() - (weight_bits - offset_bits).log2();
    }
    log2_count
}

fn bits(value: usize) -> f64 {
    f64::from(u32::try_from(value).unwrap_or(u32::MAX))
}

/// Constructs a provably irreducible binomial modulus `f(X) = X^k - c`.
///
/// The construction uses the binomial irreducibility criterion (Lidl &
/// Niederreiter, *Finite Fields*, Theorem 3.75): `X^n - a` is irreducible
/// over `F_q` exactly when `rad(n)` divides the multiplicative order of `a`,
/// `gcd(n, (q-1)/ord(a)) = 1`, and `4 | n` implies `q = 1 mod 4`.
///
/// For prime `MODULUS` with two-adicity at least two and a power-of-two
/// degree `k`, the field element `c` of multiplicative order `2^two_adicity`
/// satisfies all three conditions: `rad(k) = 2` divides `ord(c)`, the odd
/// cofactor `(q-1)/2^two_adicity` is coprime to the power-of-two degree, and
/// the two-adicity requirement is exactly the `q = 1 mod 4` condition. The
/// element is found deterministically from small bases, and its order is
/// verified by a single Fermat-style powering, so the result is irreducible
/// by proof rather than by a polynomial-time test.
///
/// # Errors
///
/// Returns [`TdmError::ModulusNotPrime`] for a composite base field,
/// [`TdmError::AutomaticModulusUnsupported`] when the field's two-adicity is
/// below two, the degree is zero, or the degree is not a power of two.
pub fn automatic_ring_modulus<const MODULUS: u32>(k: usize) -> Result<Box<[u32]>, TdmError> {
    if !is_prime_u32(MODULUS) {
        return Err(TdmError::ModulusNotPrime { modulus: MODULUS });
    }
    let two_adicity = MODULUS.saturating_sub(1).trailing_zeros();
    if k == 0 || two_adicity < 2 || !k.is_power_of_two() {
        return Err(TdmError::AutomaticModulusUnsupported {
            modulus: MODULUS,
            degree: k,
        });
    }

    let odd_cofactor = u64::from(MODULUS - 1) >> two_adicity;
    for base in 2..=64u64 {
        let candidate = mod_pow(base, odd_cofactor, MODULUS);
        if candidate <= 1 {
            continue;
        }
        let half_order = mod_pow(u64::from(candidate), 1u64 << (two_adicity - 1), MODULUS);
        if half_order == MODULUS - 1 {
            let mut polynomial = vec![0u32; k + 1];
            polynomial[0] = MODULUS - candidate;
            polynomial[k] = 1;
            return Ok(polynomial.into_boxed_slice());
        }
    }
    Err(TdmError::AutomaticModulusUnsupported {
        modulus: MODULUS,
        degree: k,
    })
}

/// Deterministic base-field modular exponentiation.
///
/// Products of residues below `2^32` fit in `u64`, so the schoolbook square-
/// and-multiply cannot overflow.
fn mod_pow(base: u64, exponent: u64, modulus: u32) -> u32 {
    let modulus = u64::from(modulus);
    let mut result = 1 % modulus;
    let mut base = base % modulus;
    let mut exponent = exponent;
    while exponent > 0 {
        if exponent & 1 == 1 {
            result = result * base % modulus;
        }
        exponent >>= 1;
        if exponent > 0 {
            base = base * base % modulus;
        }
    }
    // The result is below the modulus, which itself fits in `u32`.
    result as u32
}

/// Deterministic primality test for `u32` values.
///
/// Miller-Rabin with witnesses 2, 3, 5, 7, 11 is deterministic for all
/// inputs below 2,152,302,898,747, which covers the whole `u32` range.
fn is_prime_u32(value: u32) -> bool {
    if value < 2 {
        return false;
    }
    for prime in [2u32, 3, 5, 7, 11, 13] {
        if value.is_multiple_of(prime) {
            return value == prime;
        }
    }
    let twos = (value - 1).trailing_zeros();
    let odd = (value - 1) >> twos;
    'witnesses: for witness in [2u64, 3, 5, 7, 11] {
        let mut remainder = u64::from(mod_pow(witness, u64::from(odd), value));
        if remainder <= 1 || remainder == u64::from(value) - 1 {
            continue;
        }
        for _ in 1..twos {
            remainder = remainder * remainder % u64::from(value);
            if remainder == u64::from(value) - 1 {
                continue 'witnesses;
            }
        }
        return false;
    }
    true
}

#[cfg(test)]
mod tests {
    use prime_field_layer::ExtensionField;

    use super::{SecurityLevel, assess, automatic_ring_modulus, is_prime_u32, mod_pow};

    const FIELD: u32 = 1_073_479_681;

    #[test]
    fn automatic_modulus_is_irreducible_by_rabin_at_small_degrees() {
        for k in [1, 2, 4, 8] {
            let modulus = automatic_ring_modulus::<17>(k).unwrap();
            ExtensionField::<17>::new(k, &modulus).unwrap();
        }
        for k in [1, 2, 4, 8] {
            let modulus = automatic_ring_modulus::<FIELD>(k).unwrap();
            ExtensionField::<FIELD>::new(k, &modulus).unwrap();
        }
    }

    #[test]
    fn automatic_modulus_is_a_monic_binomial() {
        let modulus = automatic_ring_modulus::<17>(8).unwrap();
        assert_eq!(modulus.len(), 9);
        assert_eq!(modulus[8], 1);
        assert!(modulus[1..8].iter().all(|&coefficient| coefficient == 0));
        assert!(modulus[0] > 0 && modulus[0] < 17);
        // The same degree always yields the same polynomial.
        let again = automatic_ring_modulus::<17>(8).unwrap();
        assert_eq!(modulus, again);
    }

    #[test]
    fn automatic_modulus_rejects_unsupported_shapes() {
        // Degree zero and non-power-of-two degrees are unsupported.
        assert!(matches!(
            automatic_ring_modulus::<17>(0),
            Err(crate::TdmError::AutomaticModulusUnsupported {
                modulus: 17,
                degree: 0
            })
        ));
        assert!(matches!(
            automatic_ring_modulus::<17>(12),
            Err(crate::TdmError::AutomaticModulusUnsupported {
                modulus: 17,
                degree: 12
            })
        ));
        // F_2 has two-adicity zero.
        assert!(matches!(
            automatic_ring_modulus::<2>(8),
            Err(crate::TdmError::AutomaticModulusUnsupported {
                modulus: 2,
                degree: 8
            })
        ));
        // Composite base fields are rejected outright.
        assert!(matches!(
            automatic_ring_modulus::<15>(8),
            Err(crate::TdmError::ModulusNotPrime { modulus: 15 })
        ));
    }

    #[test]
    fn recommended_parameters_assess_sound_without_warnings() {
        let assessment = assess::<FIELD>(8192, 256);
        assert_eq!(assessment.level, SecurityLevel::Sound);
        assert!(assessment.warnings.is_empty());

        let assessment = assess::<FIELD>(8192, 192);
        assert_eq!(assessment.level, SecurityLevel::Sound);
        assert!(assessment.warnings.is_empty());
    }

    #[test]
    fn above_target_but_below_policy_weight_stays_sound_with_advisory() {
        let assessment = assess::<FIELD>(8192, 128);
        assert_eq!(assessment.level, SecurityLevel::Sound);
        assert_eq!(assessment.warnings.len(), 1);
        assert_eq!(
            assessment.warnings[0],
            super::ParameterWarning::WeightBelowPolicyDefault {
                weight: 128,
                floor: 192
            }
        );
    }

    #[test]
    fn plausibility_floor_weight_is_marginal_under_the_refinement_discount() {
        let assessment = assess::<FIELD>(8192, 100);
        assert_eq!(assessment.level, SecurityLevel::Marginal);
        assert!(matches!(
            assessment.warnings.as_slice(),
            [super::ParameterWarning::IsdRefinementMarginBelowTarget { .. }]
        ));
    }

    #[test]
    fn small_weight_breaks_the_decoding_estimate() {
        let assessment = assess::<FIELD>(8192, 64);
        assert_eq!(assessment.level, SecurityLevel::Broken);
        assert!(assessment.warnings.iter().any(|warning| matches!(
            warning,
            super::ParameterWarning::DecodingCostBelowTarget { .. }
        )));

        let assessment = assess::<FIELD>(8192, 16);
        assert_eq!(assessment.level, SecurityLevel::Broken);
        assert!(assessment.warnings.iter().any(|warning| matches!(
            warning,
            super::ParameterWarning::WeightBelowPlausibilityFloor { .. }
        )));
    }

    #[test]
    fn small_ring_degree_breaks_the_assessment() {
        let assessment = assess::<FIELD>(1024, 256);
        assert_eq!(assessment.level, SecurityLevel::Broken);
        assert!(matches!(
            assessment.warnings.as_slice(),
            [super::ParameterWarning::RingDegreeBelowFloor { .. }]
        ));
    }

    #[test]
    fn tiny_field_two_adicity_breaks_the_ntt_check() {
        // F_17 supports transform lengths up to 16, but k = 16 needs 31 -> 32.
        let assessment = assess::<17>(16, 4);
        assert_eq!(assessment.level, SecurityLevel::Broken);
        assert!(assessment.warnings.iter().any(|warning| matches!(
            warning,
            super::ParameterWarning::NttLengthUnsupported { .. }
        )));
        // k = 8 needs a length-16 transform, which F_17 supports.
        assert!(assess::<17>(8, 4).warnings.iter().all(|warning| !matches!(
            warning,
            super::ParameterWarning::NttLengthUnsupported { .. }
        )));
    }

    #[test]
    fn degenerate_assessment_inputs_report_broken_without_panicking() {
        for (k, weight) in [(0, 0), (4, 0), (4, 5), (usize::MAX, usize::MAX)] {
            assert_eq!(assess::<FIELD>(k, weight).level, SecurityLevel::Broken);
        }
    }

    #[test]
    fn primality_and_powering_helpers_match_known_values() {
        assert!(is_prime_u32(2));
        assert!(is_prime_u32(17));
        assert!(is_prime_u32(FIELD));
        assert!(!is_prime_u32(1));
        assert!(!is_prime_u32(15));
        assert!(!is_prime_u32(u32::MAX));
        // 2^8 = 256 = 1 mod 17 and 2^4 = 16 = -1 mod 17.
        assert_eq!(mod_pow(2, 8, 17), 1);
        assert_eq!(mod_pow(2, 4, 17), 16);
    }
}
