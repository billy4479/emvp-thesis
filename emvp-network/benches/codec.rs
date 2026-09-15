#![expect(
    clippy::unwrap_used,
    reason = "benchmarks use fixed valid fixtures and keep setup beside measurements"
)]

//! Canonical wire codec throughput at protocol-representative sizes.
//!
//! All fixtures share one `EmvpParams` so every count is semantically
//! consistent: the upload matrix is the encrypted `rows x n` matrix the
//! parameters produce (`columns = n = 2k`), the evaluation batch carries 64
//! encrypted queries of width `n`, and each products answer is the
//! `rows x blocks` matrix the server returns with `blocks = n / b = 64`.
//! The representative case is the 64-query evaluation answered by 64
//! answer matrices. Byte counts cover the full frame, header included.

use std::io::Cursor;
use std::time::Duration;

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use emvp::{AnswerMatrix, EmvpParams, EncryptedMatrix, EncryptedQuery};
use emvp_network::{
    EvaluateEntry, Field, MatrixUpload, PROTOCOL_MODULUS, ProductEntry, read_evaluate,
    read_products, read_upload_matrices, write_evaluate, write_products, write_upload_matrices,
};
use prime_field_layer::PrimeField;

/// Matrix shape for the upload fixture: `4096 x n` field elements, the
/// encrypted form of a 4096 x `ell` plaintext at the LLM-scale record
/// length.
const UPLOAD_ROWS: usize = 4096;
const UPLOAD_COLUMNS: usize = 1024;

/// Encrypted queries per evaluation entry.
const EVALUATE_QUERIES: usize = 64;

/// Query width `n` of the evaluation fixture (`n = 2k` of [`PARAMS`]).
const QUERY_WIDTH: usize = 1024;

/// Answered matrices per products entry: one answer per query in the
/// representative 64-query evaluation.
const PRODUCT_ENTRIES_ANSWERS: usize = 64;

/// Answer shape of the products fixture: `4096` rows x `blocks = n / b`
/// blocks, the answer shape [`PARAMS`] produces.
const ANSWER_ROWS: usize = 4096;
const ANSWER_BLOCKS: usize = 64;

const PARAMS: EmvpParams = EmvpParams {
    k: 512,
    ell: 512,
    b: 16,
    lambda: 128,
};

/// The field cardinality minus one, in host arithmetic, for value cycling.
const VALUE_CYCLE: usize = (PROTOCOL_MODULUS - 1) as usize;

fn field_values(count: usize) -> Vec<Field> {
    let field = PrimeField::<PROTOCOL_MODULUS>::new();
    (0..count)
        .map(|index| field.element_u32((index % VALUE_CYCLE) as u32))
        .collect()
}

fn upload_fixture() -> MatrixUpload {
    let values = field_values(UPLOAD_ROWS * UPLOAD_COLUMNS);
    let matrix = EncryptedMatrix::from_parts(
        0x1234_5678_9abc_def0_u128,
        UPLOAD_ROWS,
        UPLOAD_COLUMNS,
        values,
    )
    .unwrap();
    MatrixUpload {
        params: PARAMS,
        matrix,
    }
}

fn evaluate_fixture() -> Vec<EvaluateEntry> {
    let queries = (0..EVALUATE_QUERIES)
        .map(|index| EncryptedQuery::from_parts(1, index as u64, field_values(QUERY_WIDTH)))
        .collect();
    vec![EvaluateEntry {
        matrix_id: 1,
        queries,
    }]
}

fn products_fixture() -> Vec<ProductEntry> {
    let answers = (0..PRODUCT_ENTRIES_ANSWERS)
        .map(|index| {
            AnswerMatrix::from_parts(
                1,
                index as u64,
                field_values(ANSWER_ROWS * ANSWER_BLOCKS),
                ANSWER_ROWS,
                ANSWER_BLOCKS,
            )
        })
        .collect();
    vec![ProductEntry {
        matrix_id: 1,
        answers,
    }]
}

fn encoded_upload() -> Vec<u8> {
    let mut buffer = Vec::with_capacity(UPLOAD_ROWS * UPLOAD_COLUMNS * 4 + 128);
    write_upload_matrices(&mut buffer, &[upload_fixture()]).unwrap();
    buffer
}

fn encoded_evaluate() -> Vec<u8> {
    let mut buffer = Vec::with_capacity(EVALUATE_QUERIES * QUERY_WIDTH * 4 + 128);
    write_evaluate(&mut buffer, &evaluate_fixture()).unwrap();
    buffer
}

fn encoded_products() -> Vec<u8> {
    let mut buffer =
        Vec::with_capacity(PRODUCT_ENTRIES_ANSWERS * ANSWER_ROWS * ANSWER_BLOCKS * 4 + 128);
    write_products(&mut buffer, &products_fixture()).unwrap();
    buffer
}

fn bench_codec(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("codec");
    // The representative products frame carries 64 MiB of answers; fewer
    // samples keep the six-case group bounded without shrinking the
    // representative fixtures.
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    let upload = upload_fixture();
    let evaluate = evaluate_fixture();
    let products = products_fixture();
    let upload_bytes = encoded_upload();
    group.throughput(Throughput::Bytes(upload_bytes.len() as u64));
    group.bench_function("encode-matrix-upload", |bencher| {
        bencher.iter(|| {
            let mut sink = Vec::with_capacity(upload_bytes.len());
            write_upload_matrices(&mut sink, std::slice::from_ref(&upload)).unwrap();
            sink
        });
    });
    group.bench_function("decode-matrix-upload", |bencher| {
        bencher.iter(|| {
            let mut cursor = Cursor::new(upload_bytes.as_slice());
            read_upload_matrices(&mut cursor).unwrap()
        });
    });

    let evaluate_bytes = encoded_evaluate();
    group.throughput(Throughput::Bytes(evaluate_bytes.len() as u64));
    group.bench_function("encode-evaluate", |bencher| {
        bencher.iter(|| {
            let mut sink = Vec::with_capacity(evaluate_bytes.len());
            write_evaluate(&mut sink, &evaluate).unwrap();
            sink
        });
    });
    group.bench_function("decode-evaluate", |bencher| {
        bencher.iter(|| {
            let mut cursor = Cursor::new(evaluate_bytes.as_slice());
            read_evaluate(&mut cursor).unwrap()
        });
    });

    let products_bytes = encoded_products();
    group.throughput(Throughput::Bytes(products_bytes.len() as u64));
    group.bench_function("encode-products", |bencher| {
        bencher.iter(|| {
            let mut sink = Vec::with_capacity(products_bytes.len());
            write_products(&mut sink, &products).unwrap();
            sink
        });
    });
    group.bench_function("decode-products", |bencher| {
        bencher.iter(|| {
            let mut cursor = Cursor::new(products_bytes.as_slice());
            read_products(&mut cursor).unwrap()
        });
    });

    group.finish();
}

criterion_group!(benches, bench_codec);
criterion_main!(benches);
