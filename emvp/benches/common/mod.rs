//! Fixtures and helpers shared by the `protocol` and `online` benchmark
//! targets.
//!
//! Both suites build their protocol artifacts from the same seeded builders,
//! so measurements taken by either target refer to identical fixtures.

use criterion::{BenchmarkGroup, Criterion, Throughput, measurement::WallTime};
use emvp::{DerivedState, EmvpParams, ProtocolError, SecretKey};
use prime_field_layer::{FieldElement, PrimeField};
use rand_chacha::ChaCha20Rng;
use rand_core::SeedableRng;
use trapdoor_matrices::{IrreducibleRingLpn, RaaWeightedProduct, TdmMask, ToeplitzFastProduct};

// NTT-friendly prime: 1_073_479_681 - 1 is divisible by 2^18.
pub const MODULUS: u32 = 1_073_479_681;

// Legacy small parameter set, kept for the bench-quick feedback suite. Its
// online client cases cross the n-row mask-block boundary at rows = n =
// 1024. The default suite only runs the LLM-scale parameter sets below.
pub const PARAMS: EmvpParams = EmvpParams {
    k: 512,
    ell: 512,
    b: 16,
    lambda: 128,
};

// LLM-scale record lengths: the model's hidden dimension, so `ell = 4096`
// matches 7B-class weight matrices and `ell = 8192` 70B-class ones. Each
// suite derives concrete (k, b) from the record length with the same
// parameter search a production deployment would run.
pub const LLM_RECORD_LENGTHS: [usize; 2] = [4096, 8192];
// LLM-scale matrix heights; 4096 x 4096 is one attention projection and
// 16384 x 8192 approaches a large FFN layer.
pub const LLM_ROW_COUNTS: [usize; 3] = [4096, 8192, 16384];
pub const LLM_LAMBDA: u32 = 128;

// Sparse column weight `t` of the Ring-LPN benchmark blocks' secret `E`,
// sized to the project policy floor `POLICY_WEIGHT_FLOOR`.
pub const TARGET_COLUMN_WEIGHT: usize = 192;

#[must_use]
pub fn seeded_rng(domain: u8, size: usize) -> ChaCha20Rng {
    let mut seed = [domain; 32];
    for (slot, byte) in seed.iter_mut().zip(size.to_le_bytes()) {
        *slot ^= byte;
    }
    ChaCha20Rng::from_seed(seed)
}

#[must_use]
pub fn field_values(length: usize, domain: u8) -> Vec<FieldElement<MODULUS>> {
    let field = PrimeField::<MODULUS>::new();
    let mut values = vec![field.element_u32(0); length];
    field.fill_uniform(&mut seeded_rng(domain, length), &mut values);
    values
}

pub fn elements(count: usize) -> Throughput {
    Throughput::Elements(u64::try_from(count).unwrap())
}

// Standard-suite benchmark IDs keep their historical shape so saved
// baselines stay comparable; other suites prefix the parameter.
#[must_use]
pub fn bench_parameter(tag: &str, rows: usize) -> String {
    if tag.is_empty() {
        rows.to_string()
    } else {
        format!("{tag}-rows{rows}")
    }
}

pub fn suite_group<'a>(
    criterion: &'a mut Criterion,
    name: &str,
    huge: bool,
) -> BenchmarkGroup<'a, WallTime> {
    let mut group = criterion.benchmark_group(name);
    if huge {
        // LLM-scale iterations cost seconds; fewer samples keep the suite
        // run time bounded.
        group.sample_size(10);
    }
    group
}

// One `n x n` Toeplitz mask block, matching the codeword length n = 2k.
pub fn toeplitz_block(
    params: EmvpParams,
    stream: &mut ChaCha20Rng,
    _index: usize,
) -> Result<ToeplitzFastProduct<MODULUS>, ProtocolError> {
    Ok(ToeplitzFastProduct::sample(params.n()?, stream)?)
}

// One `n x n` RAA mask block with three nonzero weights per factor.
pub fn raa_block(
    params: EmvpParams,
    stream: &mut ChaCha20Rng,
    _index: usize,
) -> Result<RaaWeightedProduct<MODULUS>, ProtocolError> {
    Ok(RaaWeightedProduct::sample_nonzero(params.n()?, 3, stream)?)
}

// One square `n x n` Ring-LPN mask block with a fixed-weight secret, built
// by the shared deterministic test/benchmark builder.
pub fn ring_block(
    params: EmvpParams,
    stream: &mut ChaCha20Rng,
    _index: usize,
) -> Result<IrreducibleRingLpn<MODULUS>, ProtocolError> {
    let n = params.n()?;
    Ok(trapdoor_matrices::testing::ring_block::<MODULUS, _>(
        n,
        TARGET_COLUMN_WEIGHT.min(n),
        stream,
    )?)
}

pub type BlockBuilder<M> = fn(EmvpParams, &mut ChaCha20Rng, usize) -> Result<M, ProtocolError>;

// The expanded long-term secrets for `rows` matrix rows.
pub fn derive_with<M: TdmMask<MODULUS>>(
    params: EmvpParams,
    rows: usize,
    domain: u8,
    build_block: BlockBuilder<M>,
) -> DerivedState<MODULUS, M> {
    let mut rng = seeded_rng(domain ^ 0x80, rows);
    SecretKey::<MODULUS>::new(params, [domain; 32])
        .unwrap()
        .derive(rows, &mut rng, |stream, index| {
            build_block(params, stream, index)
        })
        .unwrap()
}

// One client run producing the encrypted matrix plus `count` queries with
// decoding keys, shared by the answer and decode phases. Fixture building
// happens once per case, never per iteration. The answer and decode phases
// are mask-independent, so the toeplitz fixture stands in for every
// construction.
pub fn protocol_fixtures_batch(
    params: EmvpParams,
    rows: usize,
    count: usize,
) -> (
    emvp::EncryptedMatrix<MODULUS>,
    Vec<emvp::EncryptedQuery<MODULUS>>,
    Vec<emvp::DecodingKey<MODULUS>>,
) {
    let mut state = derive_with(params, rows, 0x06, toeplitz_block);
    let matrix = field_values(rows * params.ell, 0x07);
    let record = field_values(params.ell, 0x08);
    let encrypted = emvp::encrypt(&mut state, &matrix).unwrap();
    let mut queries = Vec::with_capacity(count);
    let mut decoding_keys = Vec::with_capacity(count);
    for _ in 0..count {
        let (encrypted_query, decoding_key) = emvp::query(&mut state, &record).unwrap();
        queries.push(encrypted_query);
        decoding_keys.push(decoding_key);
    }
    (encrypted, queries, decoding_keys)
}

#[must_use]
pub fn protocol_fixtures(
    params: EmvpParams,
    rows: usize,
) -> (
    emvp::EncryptedMatrix<MODULUS>,
    emvp::EncryptedQuery<MODULUS>,
    emvp::DecodingKey<MODULUS>,
) {
    let (encrypted, mut queries, mut decoding_keys) = protocol_fixtures_batch(params, rows, 1);
    (
        encrypted,
        queries.pop().unwrap(),
        decoding_keys.pop().unwrap(),
    )
}
