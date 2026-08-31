#![expect(clippy::unwrap_used, reason = "test fixtures must succeed")]

use emvp::{EmvpParams, PROTOCOL_MAX_LAMBDA, ParamsError, pow_ge_pow2};

const LAMBDA: u32 = 128;

const fn params(k: usize, ell: usize, b: usize) -> EmvpParams {
    EmvpParams {
        k,
        ell,
        b,
        lambda: LAMBDA,
    }
}

/// An independent big-integer power for cross-checking [`pow_ge_pow2`].
struct BigPower(Vec<u64>);

impl BigPower {
    fn new(base: u64) -> Self {
        Self(vec![base])
    }

    fn multiply_by(&mut self, factor: u64) {
        let mut carry = 0_u128;
        for limb in &mut self.0 {
            let product = u128::from(*limb) * u128::from(factor) + carry;
            *limb = product as u64;
            carry = product >> 64;
        }
        while carry > 0 {
            self.0.push(carry as u64);
            carry >>= 64;
        }
    }

    fn bit_length(&self) -> u64 {
        let top = *self.0.last().unwrap();
        let top_bits = u64::from(u64::BITS - top.leading_zeros());
        (self.0.len() as u64 - 1) * u64::from(u64::BITS) + top_bits
    }
}

/// Reference implementation: exact `base^exp >= 2^lambda` via big integers.
fn reference_pow_ge_pow2(base: u64, exp: u64, lambda: u32) -> bool {
    if lambda == 0 {
        return true;
    }
    if exp == 0 || base < 2 {
        return false;
    }
    let mut value = BigPower::new(base);
    for _ in 1..exp {
        value.multiply_by(base);
    }
    value.bit_length() > u64::from(lambda)
}

#[test]
fn pow_matches_bignum_reference() {
    for base in 2_u64..=300 {
        for exp in 1_u64..=40 {
            for &lambda in &[1_u32, 8, 64, 128, 129, 200] {
                let actual = pow_ge_pow2(base, exp, lambda);
                let expected = reference_pow_ge_pow2(base, exp, lambda);
                assert_eq!(actual, expected, "base {base}, exp {exp}, lambda {lambda}");
            }
        }
    }
    for &exp in &[5_u64, 100, 5_000] {
        for &lambda in &[1_u32, 128, 200, 1_000, 100_000] {
            let actual = pow_ge_pow2(30_000, exp, lambda);
            let expected = reference_pow_ge_pow2(30_000, exp, lambda);
            assert_eq!(actual, expected, "base 30000, exp {exp}, lambda {lambda}");
        }
    }
}

#[test]
fn validates_secure_parameter_sets() {
    // d = ceil(162 / 2) = 81, 3^81 > 2^128; (324/3 + 1) * 162 = 17658 > 452.
    params(162, 162, 3).validate().unwrap();
    // Records shorter than the rank are zero-padded.
    params(162, 100, 3).validate().unwrap();
    // At b = 2, the fixed-partition attack requires k >= lambda.
    params(128, 128, 2).validate().unwrap();
}

#[test]
fn rejects_insecure_or_malformed_parameter_sets() {
    assert_eq!(
        EmvpParams {
            k: usize::MAX,
            ell: 1,
            b: 2,
            lambda: LAMBDA,
        }
        .n(),
        Err(ParamsError::DimensionOverflow)
    );
    assert_eq!(
        params(8, 8, 0).blocks(),
        Err(ParamsError::BlockTooSmall { b: 0 })
    );
    assert_eq!(
        params(8, 8, 3).blocks(),
        Err(ParamsError::BlockDoesNotDivideLength { b: 3, n: 16 })
    );
    assert_eq!(params(162, 0, 3).validate(), Err(ParamsError::ZeroEll));
    assert_eq!(
        params(162, 163, 3).validate(),
        Err(ParamsError::EllExceedsRank { ell: 163, k: 162 })
    );
    assert_eq!(
        params(30, 30, 2).validate(),
        Err(ParamsError::RankBelowSecurityFloor {
            k: 30,
            lambda: LAMBDA
        })
    );
    assert_eq!(
        params(162, 162, 1).validate(),
        Err(ParamsError::BlockTooSmall { b: 1 })
    );
    assert_eq!(
        params(162, 162, 5).validate(),
        Err(ParamsError::BlockDoesNotDivideLength { b: 5, n: 324 })
    );
    // d = ceil(33 / 2) = 17 and 34^17 ~ 2^87 < 2^128.
    assert_eq!(
        params(33, 33, 3).validate(),
        Err(ParamsError::InsecureAgainstAlgebraicAttack {
            k: 33,
            b: 3,
            d: 17,
            lambda: LAMBDA
        })
    );
    // A static random permutation remains a fixed partition across queries.
    // The random-partition estimate 129^19 exceeds 2^128, but the applicable
    // fixed-partition cost is only 8^19 = 2^57.
    assert_eq!(
        params(128, 128, 8).validate(),
        Err(ParamsError::InsecureAgainstAlgebraicAttack {
            k: 128,
            b: 8,
            d: 19,
            lambda: LAMBDA,
        })
    );
    // (4/2 + 1) * 2 = 6 <= 4 + 2, while the fixed-partition cost is 2^2.
    assert_eq!(
        EmvpParams {
            k: 2,
            ell: 2,
            b: 2,
            lambda: 2
        }
        .validate(),
        Err(ParamsError::InsecureAgainstInclusionExclusion {
            n: 4,
            k: 2,
            b: 2,
            lambda: 2
        })
    );
    // Security levels beyond the root-key strength are rejected.
    assert_eq!(
        EmvpParams {
            k: 4096,
            ell: 4096,
            b: 2,
            lambda: PROTOCOL_MAX_LAMBDA + 1
        }
        .validate(),
        Err(ParamsError::LambdaOutOfScope {
            lambda: PROTOCOL_MAX_LAMBDA + 1
        })
    );
}

#[test]
fn search_finds_minimal_rank_and_largest_block() {
    let found = emvp::search(162, LAMBDA).unwrap();
    assert_eq!(found.k, 162, "search should settle on rank 162");
    // Within the winning rank the search must pick the largest feasible
    // divisor of 324; brute-force check both claims.
    let n = found.n().unwrap();
    let feasible: Vec<usize> = (2..=n)
        .filter(|b| {
            n.is_multiple_of(*b)
                && EmvpParams {
                    k: found.k,
                    ell: 162,
                    b: *b,
                    lambda: LAMBDA,
                }
                .validate()
                .is_ok()
        })
        .collect();
    assert_eq!(
        found.b,
        *feasible.last().unwrap(),
        "search must pick the largest feasible block size among {feasible:?}"
    );
    // Minimality: no smaller admissible rank admits any feasible divisor.
    let floor = usize::max(162, EmvpParams::rank_floor(LAMBDA));
    for k in floor..found.k {
        let n = 2 * k;
        for b in 2..=n {
            if n % b != 0 {
                continue;
            }
            let candidate = EmvpParams {
                k,
                ell: 162,
                b,
                lambda: LAMBDA,
            };
            assert!(
                candidate.validate().is_err(),
                "rank {k} with block size {b} should be infeasible"
            );
        }
    }
}

#[test]
fn search_respects_record_lengths() {
    for &ell in &[74_usize, 128, 512, 1024] {
        let found = emvp::search(ell, LAMBDA).unwrap();
        assert!(found.k >= ell);
        assert_eq!(found.ell, ell);
        found.validate().unwrap();
    }
}

#[test]
fn protocol_security_cannot_exceed_the_prf_key() {
    let params = EmvpParams {
        k: 257,
        ell: 257,
        b: 2,
        lambda: PROTOCOL_MAX_LAMBDA + 1,
    };
    assert_eq!(
        params.validate(),
        Err(ParamsError::LambdaOutOfScope {
            lambda: PROTOCOL_MAX_LAMBDA + 1,
        })
    );
    assert_eq!(
        emvp::search(257, PROTOCOL_MAX_LAMBDA + 1),
        Err(ParamsError::LambdaOutOfScope {
            lambda: PROTOCOL_MAX_LAMBDA + 1,
        })
    );
}

#[test]
fn search_budget_is_relative_to_the_starting_rank() {
    let found = emvp::search(10_000_000, LAMBDA).unwrap();
    assert_eq!(found.k, 10_000_000);
    assert_eq!(found.ell, 10_000_000);
    found.validate().unwrap();
}

#[test]
fn search_terminates_on_absurd_security_levels() {
    let result = emvp::search(4, u32::MAX);
    match result {
        Err(
            ParamsError::LambdaOutOfScope { .. }
            | ParamsError::RankBelowSecurityFloor { .. }
            | ParamsError::SearchExhausted { .. },
        ) => {}
        other => panic!("expected a rejection error, got {other:?}"),
    }
}
