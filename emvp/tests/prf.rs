#![expect(
    clippy::unwrap_used,
    reason = "fixed test fixtures establish that stream derivation must succeed"
)]

use emvp::prf::{Prf, PrfError, purpose};
use rand_core::Rng;

const KEY: [u8; 32] = [
    0x2b, 0x7e, 0x15, 0x16, 0x28, 0xae, 0xd2, 0xa6, 0xab, 0xf7, 0x15, 0x88, 0x09, 0xcf, 0x4f, 0x3c,
    0xef, 0x43, 0x59, 0x39, 0x85, 0x9c, 0xd1, 0xdd, 0x06, 0x91, 0x26, 0x09, 0x37, 0x0c, 0x21, 0xa5,
];

fn derived_bytes(key: [u8; 32], tag: u32, index: u64, count: usize) -> Vec<u8> {
    let mut stream = Prf::new(key).stream(tag, index).unwrap();
    let mut bytes = vec![0; count];
    stream.fill_bytes(&mut bytes);
    bytes
}

#[test]
fn same_key_purpose_and_index_give_identical_bytes() {
    let first = derived_bytes(KEY, purpose::TDM, 7, 64);
    let second = derived_bytes(KEY, purpose::TDM, 7, 64);
    assert_eq!(first, second);
}

#[test]
fn different_keys_give_different_bytes() {
    let mut other_key = KEY;
    other_key[0] ^= 1;
    assert_ne!(
        derived_bytes(KEY, purpose::TDM, 0, 32),
        derived_bytes(other_key, purpose::TDM, 0, 32)
    );
}

#[test]
fn different_purposes_give_different_bytes() {
    let tags = [
        purpose::CODE_MULTIPLIER,
        purpose::CODE_PERMUTATION,
        purpose::TDM,
        purpose::QUERY_NONZERO,
    ];
    let baseline = derived_bytes(KEY, purpose::CODE_MULTIPLIER, 0, 32);
    for &tag in &tags[1..] {
        assert_ne!(baseline, derived_bytes(KEY, tag, 0, 32));
    }
}

#[test]
fn different_indices_give_different_bytes() {
    let baseline = derived_bytes(KEY, purpose::QUERY_NONZERO, 0, 32);
    assert_ne!(baseline, derived_bytes(KEY, purpose::QUERY_NONZERO, 1, 32));
    assert_ne!(
        baseline,
        derived_bytes(KEY, purpose::QUERY_NONZERO, 1 << 20, 32)
    );
}

#[test]
fn index_boundary() {
    let last = derived_bytes(KEY, purpose::CODE_MULTIPLIER, (1 << 29) - 1, 32);
    assert!(!last.iter().all(|&byte| byte == 0));
    assert_eq!(
        Prf::new(KEY).stream(purpose::CODE_MULTIPLIER, 1 << 29),
        Err(PrfError::IndexOutOfRange { index: 1 << 29 })
    );
    assert_eq!(
        Prf::new(KEY).stream(purpose::CODE_MULTIPLIER, u64::MAX),
        Err(PrfError::IndexOutOfRange { index: u64::MAX })
    );
}

#[test]
fn derivation_order_does_not_matter() {
    let forward = {
        let multiplier = derived_bytes(KEY, purpose::CODE_MULTIPLIER, 0, 64);
        let permutation = derived_bytes(KEY, purpose::CODE_PERMUTATION, 0, 64);
        (multiplier, permutation)
    };
    let backward = {
        let permutation = derived_bytes(KEY, purpose::CODE_PERMUTATION, 0, 64);
        let multiplier = derived_bytes(KEY, purpose::CODE_MULTIPLIER, 0, 64);
        (multiplier, permutation)
    };
    assert_eq!(forward, backward);
}

#[test]
fn sibling_slots_are_disjoint() {
    for (low, high) in [(0, 1), (1, 2), (0, (1 << 29) - 1), (5, 6)] {
        assert_ne!(
            derived_bytes(KEY, purpose::TDM, low, 32),
            derived_bytes(KEY, purpose::TDM, high, 32)
        );
    }
}
