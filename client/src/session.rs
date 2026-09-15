//! The client's connection flow: derive, encrypt, upload, evaluate,
//! decode, and verify.
//!
//! The flow is generic over the transport (any `Read + Write` pair, so
//! tests can use in-memory duplexes) and over the trapdoored mask
//! construction, which the caller picks before the first derivation. This
//! function owns the connection handshake: exactly one happens here,
//! before the first frame. The client keeps every secret locally:
//! plaintext matrices, plaintext queries, derived state, and decoding keys
//! never cross the stream.

use std::time::{Duration, Instant};

use emvp::{
    AnswerMatrix, DecodingKey, EncryptedMatrix, EncryptedQuery, MaskContextId, ProtocolError,
    SecretKey, TdmMask, decode_into, encrypt, query_batch,
};
use emvp_network::{
    EvaluateEntry, FrameKind, FrameReader, MatrixUpload, PROTOCOL_MODULUS, ProductEntry,
    client_handshake, read_error_payload, read_frame_header, read_products_payload,
    read_upload_accepted_payload, write_evaluate, write_upload_matrices,
};
use prime_field_layer::PrimeField;
use rand_chacha::ChaCha20Rng;

use crate::demo::{
    Field, MasterSeed, derive_rng, matrix_key, plaintext_len, plaintext_values, query_rng,
};
use crate::error::RunError;

/// Everything one demo invocation needs besides the transport.
#[derive(Clone)]
pub struct Config {
    /// One matrix per row count.
    pub rows: Vec<usize>,
    /// Encrypted queries generated per matrix.
    pub queries: usize,
    /// The 256-bit master seed all data and keys derive from.
    pub master_seed: MasterSeed,
    /// The searched protocol parameters.
    pub params: emvp::EmvpParams,
    /// The stable context identifier of the mask suite and configuration.
    pub mask_context: MaskContextId,
}

/// What one run did and how long it took.
#[derive(Debug)]
pub struct Report {
    /// The server identifiers of the uploaded matrices, in upload order.
    pub matrix_ids: Vec<u64>,
    /// Bytes written for the matrix upload, header included.
    pub upload_bytes: u64,
    /// Bytes written for the evaluation request, header included.
    pub evaluate_bytes: u64,
    /// Bytes read for the products response, header included.
    pub products_bytes: u64,
    /// The connection handshake.
    pub handshake_duration: Duration,
    /// Deriving and encrypting every matrix locally.
    pub derive_duration: Duration,
    /// Writing the upload and reading its acknowledgment.
    pub upload_duration: Duration,
    /// Writing the evaluation and reading the products.
    pub evaluate_duration: Duration,
    /// Decoding and verifying every product.
    pub verify_duration: Duration,
    /// Query products that decoded to the expected plaintext product.
    pub verified_queries: usize,
}

/// One derived, encrypted matrix and everything needed to verify its
/// products.
struct Instance {
    /// The row count the matrix was derived for.
    rows: usize,
    /// The plaintext matrix, row-major.
    plaintext: Vec<Field>,
    /// The plaintext query vectors, in generation order.
    plaintext_queries: Vec<Vec<Field>>,
    /// The encrypted queries sent to the server.
    queries: Vec<EncryptedQuery<PROTOCOL_MODULUS>>,
    /// The decoding keys paired with the queries by position.
    decoding_keys: Vec<DecodingKey<PROTOCOL_MODULUS>>,
    /// The encrypted matrix uploaded to the server.
    encrypted: EncryptedMatrix<PROTOCOL_MODULUS>,
}

/// Runs the whole demo against one connected stream.
///
/// Performs the connection's single handshake, then derives and encrypts
/// one matrix per configured row count, uploads the set once, evaluates
/// the configured number of encrypted queries per matrix, and verifies
/// every returned product against the plaintext matrix-vector product.
/// Returns the run's report on full verification.
///
/// # Errors
///
/// Returns any handshake, transport, server, protocol, or verification
/// failure; on a verification failure every earlier product was still
/// checked.
pub fn run<M, S, F>(stream: &mut S, config: &Config, build_block: F) -> Result<Report, RunError>
where
    M: TdmMask<PROTOCOL_MODULUS>,
    S: std::io::Read + std::io::Write,
    F: Fn(&mut ChaCha20Rng, usize) -> Result<M, ProtocolError> + Sync,
{
    crate::demo::check_rows(&config.rows)?;
    crate::demo::check_queries(config.queries)?;

    let derive_start = Instant::now();
    let mut instances = Vec::with_capacity(config.rows.len());
    for (index, &rows) in config.rows.iter().enumerate() {
        instances.push(derive_instance(config, index, rows, &build_block)?);
    }
    let derive_duration = derive_start.elapsed();

    let handshake_start = Instant::now();
    client_handshake(stream)?;
    let handshake_duration = handshake_start.elapsed();

    let upload_start = Instant::now();
    let uploads: Vec<MatrixUpload> = instances
        .iter()
        .map(|instance| MatrixUpload {
            params: config.params,
            matrix: instance.encrypted.clone(),
        })
        .collect();
    let upload_bytes = write_upload_matrices(stream, &uploads)?;
    let matrix_ids = read_upload_reply(stream)?;
    validate_upload_reply(&matrix_ids, instances.len())?;
    let upload_duration = upload_start.elapsed();

    let evaluate_start = Instant::now();
    let entries: Vec<EvaluateEntry> = instances
        .iter()
        .zip(&matrix_ids)
        .map(|(instance, matrix_id)| EvaluateEntry {
            matrix_id: *matrix_id,
            queries: instance.queries.clone(),
        })
        .collect();
    let evaluate_bytes = write_evaluate(stream, &entries)?;
    let (products, products_payload) = read_products_reply(stream)?;
    let products_bytes = products_payload + emvp_network::HEADER_BYTES;
    let evaluate_duration = evaluate_start.elapsed();

    let verify_start = Instant::now();
    if products.len() != instances.len() {
        return Err(RunError::ProductsShape {
            problem: "the entry count does not match the request",
        });
    }
    let mut verified_queries = 0_usize;
    for ((instance, matrix_id), product) in instances.iter().zip(&matrix_ids).zip(&products) {
        if product.matrix_id != *matrix_id {
            return Err(RunError::ProductsShape {
                problem: "an entry answers a different matrix identifier",
            });
        }
        if product.answers.len() != instance.decoding_keys.len() {
            return Err(RunError::ProductsShape {
                problem: "an entry's answer count does not match its query count",
            });
        }
        let pairs = product
            .answers
            .iter()
            .zip(&instance.decoding_keys)
            .zip(&instance.plaintext_queries);
        for ((answer, key), plaintext_query) in pairs {
            verify_answer(answer, key, plaintext_query, instance, *matrix_id)?;
            verified_queries += 1;
        }
    }
    let verify_duration = verify_start.elapsed();

    Ok(Report {
        matrix_ids,
        upload_bytes,
        evaluate_bytes,
        products_bytes,
        handshake_duration,
        derive_duration,
        upload_duration,
        evaluate_duration,
        verify_duration,
        verified_queries,
    })
}

/// Checks the upload acknowledgment against the upload: one identifier per
/// matrix, each nonzero and distinct.
///
/// # Errors
///
/// Returns the count mismatch or the identifier problem.
fn validate_upload_reply(identifiers: &[u64], uploaded: usize) -> Result<(), RunError> {
    if identifiers.len() != uploaded {
        return Err(RunError::UploadAcknowledge {
            uploaded,
            acknowledged: identifiers.len(),
        });
    }
    if identifiers.contains(&0) {
        return Err(RunError::UploadIdentifiers {
            problem: "an identifier is zero",
        });
    }
    let mut seen = std::collections::BTreeSet::new();
    if identifiers
        .iter()
        .any(|identifier| !seen.insert(*identifier))
    {
        return Err(RunError::UploadIdentifiers {
            problem: "an identifier appears twice",
        });
    }
    Ok(())
}

/// Reads the upload acknowledgment, mapping a rejection to its error.
fn read_upload_reply<S: std::io::Read>(stream: &mut S) -> Result<Vec<u64>, RunError> {
    let Some(header) = read_frame_header(stream)? else {
        return Err(RunError::ProductsShape {
            problem: "the connection closed before the upload acknowledgment",
        });
    };
    let mut frame = FrameReader::new(stream, header.payload_len);
    match header.kind {
        FrameKind::UploadAccepted => {
            let identifiers = read_upload_accepted_payload(&mut frame)?;
            frame.finish()?;
            Ok(identifiers)
        }
        FrameKind::Error => {
            let error = read_error_payload(&mut frame)?;
            frame.finish()?;
            Err(RunError::Server(error))
        }
        _ => Err(RunError::ProductsShape {
            problem: "the upload response was not an acknowledgment",
        }),
    }
}

/// Reads the products response, mapping a rejection to its error.
fn read_products_reply<S: std::io::Read>(
    stream: &mut S,
) -> Result<(Vec<ProductEntry>, u64), RunError> {
    let Some(header) = read_frame_header(stream)? else {
        return Err(RunError::ProductsShape {
            problem: "the connection closed before the products response",
        });
    };
    let mut frame = FrameReader::new(stream, header.payload_len);
    match header.kind {
        FrameKind::Products => {
            let products = read_products_payload(&mut frame)?;
            frame.finish()?;
            Ok((products, header.payload_len))
        }
        FrameKind::Error => {
            let error = read_error_payload(&mut frame)?;
            frame.finish()?;
            Err(RunError::Server(error))
        }
        _ => Err(RunError::ProductsShape {
            problem: "the evaluation response was not a products frame",
        }),
    }
}

/// Derives, encrypts, and prepares the queries of one matrix.
fn derive_instance<M, F>(
    config: &Config,
    index: usize,
    rows: usize,
    build_block: &F,
) -> Result<Instance, RunError>
where
    M: TdmMask<PROTOCOL_MODULUS>,
    F: Fn(&mut ChaCha20Rng, usize) -> Result<M, ProtocolError> + Sync,
{
    // User-controlled allocation sizes are checked before any buffer.
    let plaintext_len = plaintext_len(rows, config.params.ell)?;
    let key = matrix_key(&config.master_seed, index)?;
    let state = SecretKey::<PROTOCOL_MODULUS>::new(config.params, key)
        .map_err(RunError::Protocol)?
        .derive(
            config.mask_context,
            rows,
            &mut derive_rng(&config.master_seed, index)?,
            build_block,
        )
        .map_err(RunError::Protocol)?;
    let plaintext = plaintext_values(plaintext_len, &config.master_seed, index)?;
    let mut state = state;
    let encrypted = encrypt(&mut state, &plaintext)?;

    let field = PrimeField::<PROTOCOL_MODULUS>::new();
    let mut query_source = query_rng(&config.master_seed, index)?;
    let plaintext_queries: Vec<Vec<Field>> = (0..config.queries)
        .map(|_| {
            let mut query = vec![field.element_u32(0); config.params.ell];
            field.fill_uniform(&mut query_source, &mut query);
            query
        })
        .collect();
    let query_references: Vec<&[Field]> = plaintext_queries.iter().map(Vec::as_slice).collect();
    let artifacts = query_batch(&mut state, &query_references)?;
    let (queries, decoding_keys): (Vec<_>, Vec<_>) = artifacts.into_iter().unzip();

    Ok(Instance {
        rows,
        plaintext,
        plaintext_queries,
        queries,
        decoding_keys,
        encrypted,
    })
}

/// Decodes one answer and compares it with the plaintext product.
fn verify_answer(
    answer: &AnswerMatrix<PROTOCOL_MODULUS>,
    key: &DecodingKey<PROTOCOL_MODULUS>,
    plaintext_query: &[Field],
    instance: &Instance,
    matrix_id: u64,
) -> Result<(), RunError> {
    if answer.query_id() != key.query_id() {
        return Err(RunError::ProductsShape {
            problem: "an answer is paired with another query's decoding key",
        });
    }
    let mut decoded_product =
        vec![PrimeField::<PROTOCOL_MODULUS>::new().element_u32(0); instance.rows];
    decode_into(answer, key, &mut decoded_product)?;
    let expected = plaintext_product(&instance.plaintext, plaintext_query, instance.rows);
    if decoded_product != expected {
        return Err(RunError::Verification {
            matrix_id,
            query_id: key.query_id(),
        });
    }
    Ok(())
}

/// The plaintext matrix-vector product of one query.
fn plaintext_product(plaintext: &[Field], query: &[Field], rows: usize) -> Vec<Field> {
    let field = PrimeField::<PROTOCOL_MODULUS>::new();
    let ell = query.len();
    (0..rows)
        .map(|row| {
            let mut accumulator = field.element_u32(0);
            for (column, &coefficient) in query.iter().enumerate() {
                accumulator += plaintext[row * ell + column] * coefficient;
            }
            accumulator
        })
        .collect()
}
#[cfg(test)]
mod tests {
    use std::os::unix::net::UnixStream;
    use std::thread;

    use emvp::{EmvpParams, EncryptedMatrix, answer_batch};
    use emvp_network::{
        EvaluateEntry, FrameKind, FrameReader, MatrixUpload, ProductEntry, read_evaluate_payload,
        read_frame_header, read_upload_matrices_payload, server_handshake, write_products,
        write_upload_accepted,
    };
    use trapdoor_matrices::ToeplitzFastProduct;

    use super::{Config, run};
    use crate::demo::{
        CONTEXT_TOEPLITZ, PROTOCOL_MODULUS, master_seed_from_u64, random_master_seed,
    };
    use crate::error::RunError;

    fn params() -> EmvpParams {
        emvp::search(8, crate::demo::SECURITY_LAMBDA).unwrap()
    }

    fn toeplitz_builder(
        width: usize,
    ) -> impl Fn(
        &mut rand_chacha::ChaCha20Rng,
        usize,
    ) -> Result<ToeplitzFastProduct<PROTOCOL_MODULUS>, emvp::ProtocolError> {
        move |stream, _index| Ok(ToeplitzFastProduct::sample(width, stream)?)
    }

    /// One stored matrix of the fake server.
    struct StoredUpload {
        params: EmvpParams,
        matrix: EncryptedMatrix<PROTOCOL_MODULUS>,
    }

    /// Serves one session the way the server binary does, except products
    /// are computed on the CPU, `corrupt` flips one answer word of each
    /// matrix's first query, and `acknowledged` overrides the identifiers
    /// of the upload acknowledgment.
    fn fake_server(stream: &mut UnixStream, corrupt: bool, acknowledged: Option<Vec<u64>>) {
        server_handshake(stream).unwrap();
        let header = read_frame_header(stream).unwrap().unwrap();
        assert_eq!(header.kind, FrameKind::UploadMatrices);
        let mut frame = FrameReader::new(stream, header.payload_len);
        let uploads: Vec<MatrixUpload> = read_upload_matrices_payload(&mut frame).unwrap();
        frame.finish().unwrap();

        let stored: Vec<StoredUpload> = uploads
            .into_iter()
            .map(|upload| StoredUpload {
                params: upload.params,
                matrix: upload.matrix,
            })
            .collect();
        let identifiers = acknowledged.unwrap_or_else(|| (1..=stored.len() as u64).collect());
        write_upload_accepted(stream, &identifiers).unwrap();

        loop {
            let Some(header) = read_frame_header(stream).unwrap() else {
                return;
            };
            assert_eq!(header.kind, FrameKind::Evaluate);
            let mut frame = FrameReader::new(stream, header.payload_len);
            let entries: Vec<EvaluateEntry> = read_evaluate_payload(&mut frame).unwrap();
            frame.finish().unwrap();

            let mut products = Vec::with_capacity(entries.len());
            for entry in entries {
                let stored_matrix = &stored[(entry.matrix_id - 1) as usize];
                let mut answers =
                    answer_batch(&stored_matrix.params, &stored_matrix.matrix, &entry.queries)
                        .unwrap();
                if corrupt {
                    let answer = &answers[0];
                    let field = prime_field_layer::PrimeField::<PROTOCOL_MODULUS>::new();
                    let mut values = answer.values().to_vec();
                    values[0] -= field.element_u32(1);
                    answers[0] = emvp::AnswerMatrix::from_parts(
                        answer.instance_id(),
                        answer.query_id(),
                        values,
                        answer.rows(),
                        answer.blocks(),
                    );
                }
                products.push(ProductEntry {
                    matrix_id: entry.matrix_id,
                    answers,
                });
            }
            write_products(stream, &products).unwrap();
        }
    }

    fn config() -> Config {
        Config {
            rows: vec![2, 3],
            queries: 2,
            master_seed: master_seed_from_u64(7),
            params: params(),
            mask_context: CONTEXT_TOEPLITZ,
        }
    }

    #[test]
    fn client_verifies_products_end_to_end() {
        let (mut client_stream, server_stream) = UnixStream::pair().unwrap();
        let server = thread::spawn(move || {
            let mut server_stream = server_stream;
            fake_server(&mut server_stream, false, None);
        });
        let report = run::<ToeplitzFastProduct<PROTOCOL_MODULUS>, _, _>(
            &mut client_stream,
            &config(),
            toeplitz_builder(params().n().unwrap()),
        )
        .unwrap();

        assert_eq!(report.matrix_ids, vec![1, 2]);
        assert_eq!(report.verified_queries, 4);
        assert!(report.upload_bytes > 0);

        drop(client_stream);
        server.join().unwrap();
    }

    #[test]
    fn corrupted_products_fail_verification() {
        let (mut client_stream, server_stream) = UnixStream::pair().unwrap();
        let server = thread::spawn(move || {
            let mut server_stream = server_stream;
            fake_server(&mut server_stream, true, None);
        });
        let outcome = run::<ToeplitzFastProduct<PROTOCOL_MODULUS>, _, _>(
            &mut client_stream,
            &config(),
            toeplitz_builder(params().n().unwrap()),
        );

        assert!(matches!(
            outcome,
            Err(RunError::Verification {
                matrix_id: 1,
                query_id: 0
            })
        ));

        drop(client_stream);
        server.join().unwrap();
    }

    #[test]
    fn duplicate_acknowledged_ids_are_rejected() {
        let (mut client_stream, server_stream) = UnixStream::pair().unwrap();
        let server = thread::spawn(move || {
            let mut server_stream = server_stream;
            fake_server(&mut server_stream, false, Some(vec![1, 1]));
        });
        let outcome = run::<ToeplitzFastProduct<PROTOCOL_MODULUS>, _, _>(
            &mut client_stream,
            &config(),
            toeplitz_builder(params().n().unwrap()),
        );

        assert!(matches!(
            outcome,
            Err(RunError::UploadIdentifiers {
                problem: "an identifier appears twice"
            })
        ));

        drop(client_stream);
        server.join().unwrap();
    }

    /// A secure-mode structural test: the whole run completes from a
    /// fresh OS master seed. No equality is asserted between draws.
    #[test]
    fn a_fresh_os_master_seed_completes_the_session() {
        let (mut client_stream, server_stream) = UnixStream::pair().unwrap();
        let server = thread::spawn(move || {
            let mut server_stream = server_stream;
            fake_server(&mut server_stream, false, None);
        });
        let mut secure_config = config();
        secure_config.master_seed = random_master_seed().unwrap();
        let report = run::<ToeplitzFastProduct<PROTOCOL_MODULUS>, _, _>(
            &mut client_stream,
            &secure_config,
            toeplitz_builder(params().n().unwrap()),
        )
        .unwrap();

        assert_eq!(report.matrix_ids, vec![1, 2]);
        assert_eq!(report.verified_queries, 4);

        drop(client_stream);
        server.join().unwrap();
    }
}
