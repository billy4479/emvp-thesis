#![expect(
    clippy::unwrap_used,
    reason = "benchmarks use fixed valid fixtures and keep setup beside measurements"
)]

//! Canonical wire codec throughput at protocol-representative sizes,
//! driven through the protocol v2 pipeline: encoders take borrowed views,
//! decoders run plan → reserve → decode into a reused workspace, so the
//! steady-state decode performs no allocations.
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
use emvp::{AnswerMatrix, AnswerRef, EmvpParams, EncryptedMatrix, EncryptedQuery, EncryptedQueryRef};
use emvp_network::v2::{
    EvaluateEntryInput, EvaluateWorkspace, ProductEntryInput, ProductsWorkspace,
    UploadMatrixView, UploadWorkspace, decode_evaluate, decode_products, decode_upload,
    plan_evaluate, plan_products, plan_upload, write_evaluate, write_products,
    write_upload_matrices,
};
use emvp_network::{FrameReader, HEADER_BYTES, PROTOCOL_MODULUS, read_frame_header};
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

fn field_values(count: usize) -> Vec<emvp_network::Field> {
    let field = PrimeField::<PROTOCOL_MODULUS>::new();
    (0..count)
        .map(|index| field.element_u32((index % VALUE_CYCLE) as u32))
        .collect()
}

fn upload_fixture() -> EncryptedMatrix<PROTOCOL_MODULUS> {
    let values = field_values(UPLOAD_ROWS * UPLOAD_COLUMNS);
    EncryptedMatrix::from_parts(
        0x1234_5678_9abc_def0_u128,
        UPLOAD_ROWS,
        UPLOAD_COLUMNS,
        values,
    )
    .unwrap()
}

fn evaluate_fixture() -> Vec<EncryptedQuery<PROTOCOL_MODULUS>> {
    (0..EVALUATE_QUERIES)
        .map(|index| EncryptedQuery::from_parts(1, index as u64, field_values(QUERY_WIDTH)))
        .collect()
}

fn products_fixture() -> Vec<AnswerMatrix<PROTOCOL_MODULUS>> {
    (0..PRODUCT_ENTRIES_ANSWERS)
        .map(|index| {
            AnswerMatrix::from_parts(
                1,
                index as u64,
                field_values(ANSWER_ROWS * ANSWER_BLOCKS),
                ANSWER_ROWS,
                ANSWER_BLOCKS,
            )
        })
        .collect()
}

fn bench_upload(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    upload_views: &[UploadMatrixView<'_>],
) {
    let bytes = {
        let mut buffer = Vec::with_capacity(UPLOAD_ROWS * UPLOAD_COLUMNS * 4 + 128);
        write_upload_matrices(&mut buffer, upload_views).unwrap();
        buffer
    };
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("encode-matrix-upload", |bencher| {
        bencher.iter(|| {
            let mut sink = Vec::with_capacity(bytes.len());
            write_upload_matrices(&mut sink, upload_views).unwrap();
            sink
        });
    });
    let mut workspace = UploadWorkspace::new();
    group.bench_function("decode-matrix-upload", |bencher| {
        bencher.iter(|| {
            let mut cursor = Cursor::new(bytes.as_slice());
            read_frame_header(&mut cursor).unwrap().unwrap();
            let mut frame = FrameReader::new(&mut cursor, bytes.len() as u64 - HEADER_BYTES);
            let plan = plan_upload(&mut frame).unwrap();
            workspace.reserve(&plan).unwrap();
            let views = decode_upload(&mut frame, &plan, &mut workspace).unwrap();
            std::hint::black_box(views.len())
        });
    });
}

fn bench_evaluate(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    evaluate_views: &[EvaluateEntryInput<'_>],
) {
    let bytes = {
        let mut buffer = Vec::with_capacity(EVALUATE_QUERIES * QUERY_WIDTH * 4 + 128);
        write_evaluate(&mut buffer, evaluate_views).unwrap();
        buffer
    };
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("encode-evaluate", |bencher| {
        bencher.iter(|| {
            let mut sink = Vec::with_capacity(bytes.len());
            write_evaluate(&mut sink, evaluate_views).unwrap();
            sink
        });
    });
    let mut workspace = EvaluateWorkspace::new();
    group.bench_function("decode-evaluate", |bencher| {
        bencher.iter(|| {
            let mut cursor = Cursor::new(bytes.as_slice());
            read_frame_header(&mut cursor).unwrap().unwrap();
            let mut frame = FrameReader::new(&mut cursor, bytes.len() as u64 - HEADER_BYTES);
            let plan = plan_evaluate(&mut frame).unwrap();
            workspace.reserve(&plan).unwrap();
            let views = decode_evaluate(&mut frame, &plan, &mut workspace).unwrap();
            std::hint::black_box(views.len())
        });
    });
}

fn bench_products(
    group: &mut criterion::BenchmarkGroup<'_, criterion::measurement::WallTime>,
    products_views: &[ProductEntryInput<'_>],
) {
    let bytes = {
        let mut buffer =
            Vec::with_capacity(PRODUCT_ENTRIES_ANSWERS * ANSWER_ROWS * ANSWER_BLOCKS * 4 + 128);
        write_products(&mut buffer, products_views).unwrap();
        buffer
    };
    group.throughput(Throughput::Bytes(bytes.len() as u64));
    group.bench_function("encode-products", |bencher| {
        bencher.iter(|| {
            let mut sink = Vec::with_capacity(bytes.len());
            write_products(&mut sink, products_views).unwrap();
            sink
        });
    });
    let mut workspace = ProductsWorkspace::new();
    group.bench_function("decode-products", |bencher| {
        bencher.iter(|| {
            let mut cursor = Cursor::new(bytes.as_slice());
            read_frame_header(&mut cursor).unwrap().unwrap();
            let mut frame = FrameReader::new(&mut cursor, bytes.len() as u64 - HEADER_BYTES);
            let plan = plan_products(&mut frame).unwrap();
            workspace.reserve(&plan).unwrap();
            let views = decode_products(&mut frame, &plan, &mut workspace).unwrap();
            std::hint::black_box(views.len())
        });
    });
}

fn bench_codec(criterion: &mut Criterion) {
    let mut group = criterion.benchmark_group("codec");
    // The representative products frame carries 64 MiB of answers; fewer
    // samples keep the six-case group bounded without shrinking the
    // representative fixtures.
    group.sample_size(20);
    group.warm_up_time(Duration::from_millis(500));
    group.measurement_time(Duration::from_secs(2));

    // Borrowed encode inputs over the owned fixtures, built once.
    let upload = upload_fixture();
    let upload_views = [UploadMatrixView {
        params: PARAMS,
        instance_id: upload.instance_id(),
        rows: upload.rows(),
        columns: upload.columns(),
        values: upload.values(),
    }];
    let queries = evaluate_fixture();
    let query_refs: Vec<EncryptedQueryRef<'_, PROTOCOL_MODULUS>> =
        queries.iter().map(EncryptedQueryRef::from).collect();
    let evaluate_views = [EvaluateEntryInput {
        matrix_id: 1,
        queries: &query_refs,
    }];
    let answers = products_fixture();
    let answer_refs: Vec<AnswerRef<'_, PROTOCOL_MODULUS>> =
        answers.iter().map(AnswerRef::from).collect();
    let products_views = [ProductEntryInput {
        matrix_id: 1,
        instance_id: 1,
        rows: ANSWER_ROWS,
        blocks: ANSWER_BLOCKS,
        answers: &answer_refs,
    }];

    bench_upload(&mut group, &upload_views);
    bench_evaluate(&mut group, &evaluate_views);
    bench_products(&mut group, &products_views);

    group.finish();
}

criterion_group!(benches, bench_codec);
criterion_main!(benches);
