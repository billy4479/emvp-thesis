use std::alloc::{GlobalAlloc, Layout, System};
use std::cell::Cell;
use std::sync::atomic::{AtomicUsize, Ordering};

use prime_field_layer::{
    ExtensionField, ExtensionFieldError, PolynomialAlgorithm, PolynomialReductionPlan,
};

struct CountingAllocator;

thread_local! {
    static COUNT_ALLOCATIONS: Cell<bool> = const { Cell::new(false) };
}

static ALLOCATIONS: AtomicUsize = AtomicUsize::new(0);

// SAFETY: every allocation and deallocation is forwarded to `System` unchanged;
// the wrapper only increments a thread-local test counter before allocation.
unsafe impl GlobalAlloc for CountingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        COUNT_ALLOCATIONS.with(|enabled| {
            if enabled.get() {
                let _ = ALLOCATIONS.fetch_add(1, Ordering::Relaxed);
            }
        });
        // SAFETY: forwarding the allocator contract and layout unchanged.
        unsafe { System.alloc(layout) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        // SAFETY: the pointer came from `System` with this layout above.
        unsafe { System.dealloc(pointer, layout) };
    }
}

#[global_allocator]
static ALLOCATOR: CountingAllocator = CountingAllocator;

fn canonical(value: u32, modulus: u32) -> u32 {
    (u64::from(value) % u64::from(modulus)) as u32
}

fn oracle_reduce<const K: usize>(
    modulus: u32,
    modulus_polynomial: &[u32],
    product: &[u32],
) -> [u32; K] {
    let mut values = vec![0; 2 * K - 1];
    for (target, &coefficient) in values.iter_mut().zip(product) {
        *target = canonical(coefficient, modulus);
    }
    for degree in (K..values.len()).rev() {
        let factor = u64::from(values[degree]);
        for index in 0..K {
            let subtract = factor * u64::from(canonical(modulus_polynomial[index], modulus))
                % u64::from(modulus);
            values[degree - K + index] =
                ((u64::from(values[degree - K + index]) + u64::from(modulus) - subtract)
                    % u64::from(modulus)) as u32;
        }
    }
    std::array::from_fn(|index| values[index])
}

fn oracle_mul<const K: usize>(
    modulus: u32,
    modulus_polynomial: &[u32],
    lhs: &[u32; K],
    rhs: &[u32; K],
) -> [u32; K] {
    let mut product = vec![0; 2 * K - 1];
    for (lhs_index, &lhs) in lhs.iter().enumerate() {
        for (rhs_index, &rhs) in rhs.iter().enumerate() {
            let index = lhs_index + rhs_index;
            product[index] = ((u64::from(product[index])
                + u64::from(canonical(lhs, modulus)) * u64::from(canonical(rhs, modulus)))
                % u64::from(modulus)) as u32;
        }
    }
    oracle_reduce(modulus, modulus_polynomial, &product)
}

fn irreducible_binomial<const K: usize>(modulus: u32) -> Vec<u32> {
    let mut polynomial = vec![0; K + 1];
    polynomial[0] = modulus - 3;
    polynomial[K] = 1;
    polynomial
}

#[test]
fn schoolbook_reduction_matches_independent_long_division() {
    const MODULUS: u32 = 17;
    const K: usize = 4;
    let modulus = [14, 5, 0, 2, 1];
    let product = [u32::MAX, 17, 18, 16, 15, 14, 13];
    let plan = PolynomialReductionPlan::<MODULUS, K>::new(&modulus).unwrap();
    let mut scratch = plan.scratch();
    let mut actual = [0; K];
    plan.reduce(&product, &mut actual, &mut scratch).unwrap();
    assert_eq!(actual, oracle_reduce(MODULUS, &modulus, &product));
    assert_eq!(plan.algorithm(), PolynomialAlgorithm::Schoolbook);
}

#[test]
fn checked_extension_arithmetic_matches_independent_oracle() {
    const MODULUS: u32 = 17;
    const K: usize = 4;
    let modulus = [14, 0, 0, 0, 1];
    let extension = ExtensionField::<MODULUS, K>::new(&modulus).unwrap();
    let lhs = [u32::MAX, 17, 18, 16];
    let rhs = [16, 15, u32::MAX - 1, 34];
    let mut scratch = extension.scratch();
    let mut product = [0; K];
    let mut square = [0; K];

    extension
        .mul(&lhs, &rhs, &mut product, &mut scratch)
        .unwrap();
    extension.square(&lhs, &mut square, &mut scratch).unwrap();

    assert_eq!(product, oracle_mul(MODULUS, &modulus, &lhs, &rhs));
    assert_eq!(square, oracle_mul(MODULUS, &modulus, &lhs, &lhs));
    assert_eq!(extension.add(&lhs, &rhs), [16, 15, 0, 16]);
    assert_eq!(extension.sub(&lhs, &rhs), [1, 2, 2, 16]);
    assert!(product.iter().all(|&coefficient| coefficient < MODULUS));
    assert!(square.iter().all(|&coefficient| coefficient < MODULUS));
}

#[test]
fn validation_and_reducibility_errors_are_typed() {
    assert!(matches!(
        PolynomialReductionPlan::<17, 0>::new(&[1]),
        Err(ExtensionFieldError::ZeroDegree)
    ));
    assert!(matches!(
        PolynomialReductionPlan::<17, 4>::new(&[1, 2, 3]),
        Err(ExtensionFieldError::ModulusLength {
            expected: 5,
            actual: 3
        })
    ));
    assert!(matches!(
        PolynomialReductionPlan::<17, 2>::new(&[1, 0, 2]),
        Err(ExtensionFieldError::ModulusNotMonic)
    ));
    assert!(matches!(
        ExtensionField::<17, 4>::new(&[16, 0, 0, 0, 1]),
        Err(ExtensionFieldError::ReducibleModulus)
    ));

    let canonicalized = PolynomialReductionPlan::<17, 2>::new(&[u32::MAX, 34, 18]).unwrap();
    assert_eq!(canonicalized.modulus_polynomial(), &[0, 0, 1]);
    let mut scratch = canonicalized.scratch();
    let mut output = [99; 2];
    assert_eq!(
        canonicalized.reduce(&[1; 4], &mut output, &mut scratch),
        Err(ExtensionFieldError::ProductTooLong {
            maximum: 3,
            actual: 4
        })
    );
    assert_eq!(output, [99; 2]);
}

#[test]
fn ntt_reduction_multiplication_and_squaring_match_slow_oracle() {
    const MODULUS: u32 = 998_244_353;
    const K: usize = 128;
    let modulus = irreducible_binomial::<K>(MODULUS);
    let extension = ExtensionField::<MODULUS, K>::new_unchecked_irreducible(&modulus).unwrap();
    assert_eq!(
        extension.algorithm(),
        PolynomialAlgorithm::Ntt {
            transform_length: 256
        }
    );

    let lhs = std::array::from_fn(|index| {
        ((index as u64 * 2_654_435_761 + u64::from(u32::MAX)) % (1u64 << 32)) as u32
    });
    let rhs = std::array::from_fn(|index| {
        ((index as u64 * 1_103_515_245 + 12_345) % (1u64 << 32)) as u32
    });
    let mut scratch = extension.scratch();
    let mut product = [0; K];
    let mut square = [0; K];
    extension
        .mul(&lhs, &rhs, &mut product, &mut scratch)
        .unwrap();
    extension.square(&lhs, &mut square, &mut scratch).unwrap();
    assert_eq!(product, oracle_mul(MODULUS, &modulus, &lhs, &rhs));
    assert_eq!(square, oracle_mul(MODULUS, &modulus, &lhs, &lhs));

    let reduction = PolynomialReductionPlan::<MODULUS, K>::new(&modulus).unwrap();
    let input: Vec<_> = (0..2 * K - 1)
        .map(|index| (index as u32).wrapping_mul(2_654_435_761).wrapping_add(97))
        .collect();
    let mut reduction_scratch = reduction.scratch();
    let mut reduced = [0; K];
    reduction
        .reduce(&input, &mut reduced, &mut reduction_scratch)
        .unwrap();
    assert_eq!(reduced, oracle_reduce(MODULUS, &modulus, &input));
}

#[test]
fn threshold_behavior_is_degree_only() {
    let schoolbook_modulus = irreducible_binomial::<23>(998_244_353);
    let ntt_modulus = irreducible_binomial::<24>(998_244_353);
    assert_eq!(
        PolynomialReductionPlan::<998_244_353, 23>::new(&schoolbook_modulus)
            .unwrap()
            .algorithm(),
        PolynomialAlgorithm::Schoolbook
    );
    assert_eq!(
        PolynomialReductionPlan::<998_244_353, 24>::new(&ntt_modulus)
            .unwrap()
            .algorithm(),
        PolynomialAlgorithm::Ntt {
            transform_length: 64
        }
    );
}

#[test]
fn caller_scratch_reuses_all_allocations() {
    const MODULUS: u32 = 998_244_353;
    const K: usize = 128;
    let modulus = irreducible_binomial::<K>(MODULUS);
    let extension = ExtensionField::<MODULUS, K>::new_unchecked_irreducible(&modulus).unwrap();
    let lhs = std::array::from_fn(|index| index as u32 + 1);
    let rhs = std::array::from_fn(|index| index as u32 * 3 + 7);
    let mut output = [0; K];
    let mut scratch = extension.scratch();

    extension
        .mul(&lhs, &rhs, &mut output, &mut scratch)
        .unwrap();
    ALLOCATIONS.store(0, Ordering::Relaxed);
    COUNT_ALLOCATIONS.with(|enabled| enabled.set(true));
    for _ in 0..8 {
        extension
            .mul(&lhs, &rhs, &mut output, &mut scratch)
            .unwrap();
        extension.square(&lhs, &mut output, &mut scratch).unwrap();
    }
    COUNT_ALLOCATIONS.with(|enabled| enabled.set(false));
    assert_eq!(ALLOCATIONS.load(Ordering::Relaxed), 0);
}
