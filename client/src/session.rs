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
//!
//! The wire boundary is zero copy in both directions. Uploads and
//! evaluations stream from borrowed views over the owned artifacts, and
//! the acknowledgment and the products response decode into per-session
//! workspaces the next round reuses, so a steady-state round allocates
//! nothing on the send path or per verified query.

use std::time::{Duration, Instant};

use emvp::{
    AnswerRef, DecodingKey, EmvpParams, EncryptedMatrix, EncryptedQuery, EncryptedQueryRef,
    MaskContextId, ProtocolError, SecretKey, TdmMask, decode_into, encrypt, query_batch,
};
use emvp_network::v2::{
    EvaluateEntryInput, ProductsViews, ProductsWorkspace, UploadAcceptedWorkspace,
    UploadMatrixView, decode_products, decode_upload_accepted, plan_products, plan_upload_accepted,
    write_evaluate, write_upload_matrices,
};
use emvp_network::{
    CodecError, FrameKind, FrameReader, HEADER_BYTES, PROTOCOL_MODULUS, client_handshake,
    read_error_payload, read_frame_header,
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

/// What one evaluate-and-verify round did and how long it took.
#[derive(Clone, Copy, Debug)]
struct RoundOutcome {
    /// Bytes written for the evaluation request, header included.
    evaluate_bytes: u64,
    /// Bytes read for the products response, header included.
    products_bytes: u64,
    /// Writing the evaluation and reading the products.
    evaluate_duration: Duration,
    /// Decoding and verifying every product.
    verify_duration: Duration,
    /// Query products that decoded to the expected plaintext product.
    verified_queries: usize,
}

/// One connection's reusable client state.
///
/// The session owns every buffer the protocol rounds share: the codec
/// workspaces the upload acknowledgment and the products response decode
/// into, the borrowed-query scratch each evaluation request is built
/// from, and the two verification buffers every answer is decoded and
/// compared against. Each buffer grows only through an explicit reserve
/// step, is never shrunk, and carries nothing from one round to the next,
/// so after the first round has shaped them a round allocates nothing.
///
/// `'q` is the borrow the query-reference scratch points into (the
/// instances of the round), which a round supplies per call.
struct Session<'a, 'q, S> {
    /// The transport.
    stream: &'a mut S,
    /// The upload acknowledgment's identifiers, one list per upload.
    ack_ids: UploadAcceptedWorkspace,
    /// The products frame's decode arena, reused per evaluation round.
    products: ProductsWorkspace,
    /// Borrowed views of the owned queries, refilled per evaluation
    /// round.
    query_refs: Vec<EncryptedQueryRef<'q, PROTOCOL_MODULUS>>,
    /// The decoded product of the answer under verification.
    decoded: Vec<Field>,
    /// The expected plaintext product of the query under verification.
    expected: Vec<Field>,
}

impl<'a, 'q, S: std::io::Read + std::io::Write> Session<'a, 'q, S> {
    /// Performs the connection's single handshake and returns the session.
    ///
    /// # Errors
    ///
    /// Returns the handshake failure.
    fn new(stream: &'a mut S) -> Result<Self, RunError> {
        client_handshake(stream)?;
        Ok(Self {
            stream,
            ack_ids: UploadAcceptedWorkspace::new(),
            products: ProductsWorkspace::new(),
            query_refs: Vec::new(),
            decoded: Vec::new(),
            expected: Vec::new(),
        })
    }

    /// Uploads every instance's encrypted matrix and validates the
    /// acknowledgment, leaving the accepted identifiers in the session
    /// for the evaluation rounds.
    ///
    /// The upload frame streams straight from borrowed views over the
    /// encrypted matrices: no ciphertext element is copied.
    ///
    /// Returns the bytes written, header included.
    ///
    /// # Errors
    ///
    /// Returns the transport, codec, server, or acknowledgment failure.
    fn upload(&mut self, instances: &[Instance], params: EmvpParams) -> Result<u64, RunError> {
        let uploads: Vec<UploadMatrixView<'_>> = instances
            .iter()
            .map(|instance| UploadMatrixView {
                params,
                instance_id: instance.encrypted.instance_id(),
                rows: instance.encrypted.rows(),
                columns: instance.encrypted.columns(),
                values: instance.encrypted.values(),
            })
            .collect();
        let upload_bytes = write_upload_matrices(&mut self.stream, &uploads)?;
        let identifiers = self.read_upload_reply()?;
        validate_upload_reply(identifiers, instances.len())?;
        Ok(upload_bytes)
    }

    /// Sends one evaluation round and verifies the returned products.
    ///
    /// The request streams from borrowed views of the owned queries, the
    /// products frame decodes into the reused workspace, and every answer
    /// decodes into and compares against the reused verification buffers.
    ///
    /// # Errors
    ///
    /// Returns the transport, codec, server, shape, or verification
    /// failure.
    fn evaluate_and_verify<'inst>(
        &mut self,
        instances: &'inst [Instance],
        params: EmvpParams,
    ) -> Result<RoundOutcome, RunError>
    where
        'inst: 'q,
    {
        let matrix_ids = self.ack_ids.identifiers();
        if matrix_ids.len() != instances.len() {
            return Err(RunError::UploadAcknowledge {
                uploaded: instances.len(),
                acknowledged: matrix_ids.len(),
            });
        }

        let evaluate_start = Instant::now();
        let entries = evaluate_entries(instances, matrix_ids, &mut self.query_refs)?;
        let evaluate_bytes = write_evaluate(&mut self.stream, &entries)?;

        let Some(header) = read_frame_header(self.stream)? else {
            return Err(RunError::ProductsShape {
                problem: "the connection closed before the products response",
            });
        };
        let mut frame = FrameReader::new(self.stream, header.payload_len);
        let products_bytes;
        let products = match header.kind {
            FrameKind::Products => {
                let plan = plan_products(&mut frame)?;
                self.products.reserve(&plan)?;
                let products = decode_products(&mut frame, &plan, &mut self.products)?;
                products_bytes = header.payload_len + HEADER_BYTES;
                products
            }
            FrameKind::Error => {
                let error = read_error_payload(&mut frame)?;
                frame.finish()?;
                return Err(RunError::Server(error));
            }
            _ => {
                return Err(RunError::ProductsShape {
                    problem: "the evaluation response was not a products frame",
                });
            }
        };
        let evaluate_duration = evaluate_start.elapsed();

        let verify_start = Instant::now();
        let verified_queries = verify_products(
            products,
            instances,
            matrix_ids,
            params,
            &mut self.decoded,
            &mut self.expected,
        )?;
        let verify_duration = verify_start.elapsed();

        Ok(RoundOutcome {
            evaluate_bytes,
            products_bytes,
            evaluate_duration,
            verify_duration,
            verified_queries,
        })
    }

    /// Reads the upload acknowledgment, mapping a rejection to its error.
    ///
    /// The accepted identifiers are copied into the acknowledgment
    /// workspace and returned as a slice of it.
    ///
    /// # Errors
    ///
    /// Returns the transport, codec, or server failure.
    fn read_upload_reply(&mut self) -> Result<&[u64], RunError> {
        let Some(header) = read_frame_header(self.stream)? else {
            return Err(RunError::ProductsShape {
                problem: "the connection closed before the upload acknowledgment",
            });
        };
        let mut frame = FrameReader::new(self.stream, header.payload_len);
        match header.kind {
            FrameKind::UploadAccepted => {
                let plan = plan_upload_accepted(&mut frame)?;
                self.ack_ids.reserve(&plan)?;
                Ok(decode_upload_accepted(&mut frame, &plan, &mut self.ack_ids)?)
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
    let mut session = Session::new(stream)?;
    let handshake_duration = handshake_start.elapsed();

    let upload_start = Instant::now();
    let upload_bytes = session.upload(&instances, config.params)?;
    let upload_duration = upload_start.elapsed();

    let outcome = session.evaluate_and_verify(&instances, config.params)?;

    Ok(Report {
        matrix_ids: session.ack_ids.identifiers().to_vec(),
        upload_bytes,
        evaluate_bytes: outcome.evaluate_bytes,
        products_bytes: outcome.products_bytes,
        handshake_duration,
        derive_duration,
        upload_duration,
        evaluate_duration: outcome.evaluate_duration,
        verify_duration: outcome.verify_duration,
        verified_queries: outcome.verified_queries,
    })
}

/// Checks the upload acknowledgment against the upload: one identifier per
/// matrix, each nonzero and distinct.
///
/// The duplicate scan is a direct pairwise membership check over the
/// acknowledged list: identifier counts are small, and the scan allocates
/// nothing where a set would.
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
    for (index, identifier) in identifiers.iter().enumerate() {
        if identifiers[..index].contains(identifier) {
            return Err(RunError::UploadIdentifiers {
                problem: "an identifier appears twice",
            });
        }
    }
    Ok(())
}

/// Builds the evaluation request's borrowed entries: one entry per
/// instance, its queries the instance's owned query list viewed through
/// the reusable reference scratch.
///
/// The scratch is cleared and refilled each round; growing it to a larger
/// total query count is its only allocation moment.
///
/// # Errors
///
/// Returns the allocation failure of the scratch growth.
fn evaluate_entries<'inst, 'q, 'scratch>(
    instances: &'inst [Instance],
    matrix_ids: &[u64],
    query_refs: &'scratch mut Vec<EncryptedQueryRef<'q, PROTOCOL_MODULUS>>,
) -> Result<Vec<EvaluateEntryInput<'scratch>>, RunError>
where
    'inst: 'q,
    'q: 'scratch,
{
    let total_queries = instances
        .iter()
        .try_fold(0_usize, |total, instance| {
            total.checked_add(instance.queries.len())
        })
        .ok_or(RunError::Protocol(ProtocolError::DimensionOverflow))?;
    query_refs.clear();
    query_refs
        .try_reserve_exact(total_queries.saturating_sub(query_refs.len()))
        .map_err(|_reserve| RunError::Codec(CodecError::AllocationFailed))?;
    query_refs.extend(
        instances
            .iter()
            .flat_map(|instance| instance.queries.iter().map(EncryptedQueryRef::from)),
    );

    let mut entries = Vec::with_capacity(instances.len());
    let mut offset = 0_usize;
    for (instance, &matrix_id) in instances.iter().zip(matrix_ids) {
        let end = offset + instance.queries.len();
        entries.push(EvaluateEntryInput {
            matrix_id,
            queries: &query_refs[offset..end],
        });
        offset = end;
    }
    Ok(entries)
}

/// Validates the products frame against the request and verifies every
/// answer against the plaintext product, reusing both verify buffers.
///
/// Returns the number of verified query products.
///
/// # Errors
///
/// Returns the shape or verification failure; on a verification failure
/// every earlier product was still checked.
fn verify_products(
    products: ProductsViews<'_>,
    instances: &[Instance],
    matrix_ids: &[u64],
    params: EmvpParams,
    decoded: &mut Vec<Field>,
    expected: &mut Vec<Field>,
) -> Result<usize, RunError> {
    if products.len() != instances.len() {
        return Err(RunError::ProductsShape {
            problem: "the entry count does not match the request",
        });
    }
    let blocks = params.blocks().map_err(RunError::Params)?;
    let mut verified_queries = 0_usize;
    for ((instance, matrix_id), entry) in instances.iter().zip(matrix_ids).zip(products.iter()) {
        if entry.matrix_id() != *matrix_id {
            return Err(RunError::ProductsShape {
                problem: "an entry answers a different matrix identifier",
            });
        }
        if entry.instance_id() != instance.encrypted.instance_id() {
            return Err(RunError::ProductsShape {
                problem: "an entry answers a different matrix instance",
            });
        }
        if entry.rows() != instance.rows {
            return Err(RunError::ProductsShape {
                problem: "an entry's answer rows do not match its matrix",
            });
        }
        if entry.blocks() != blocks {
            return Err(RunError::ProductsShape {
                problem: "an entry's answer blocks do not match the parameters",
            });
        }
        if entry.len() != instance.decoding_keys.len() {
            return Err(RunError::ProductsShape {
                problem: "an entry's answer count does not match its query count",
            });
        }
        for (index, key) in instance.decoding_keys.iter().enumerate() {
            let answer = entry.answer(index).ok_or(RunError::ProductsShape {
                problem: "an entry's answer descriptor was missing",
            })?;
            verify_answer(
                &answer,
                key,
                &instance.plaintext_queries[index],
                instance,
                *matrix_id,
                decoded,
                expected,
            )?;
            verified_queries += 1;
        }
    }
    Ok(verified_queries)
}

/// Decodes one answer and compares it with the plaintext product,
/// reusing the verification buffers.
///
/// # Errors
///
/// Returns the pairing, protocol, or verification failure.
fn verify_answer(
    answer: &AnswerRef<'_, PROTOCOL_MODULUS>,
    key: &DecodingKey<PROTOCOL_MODULUS>,
    plaintext_query: &[Field],
    instance: &Instance,
    matrix_id: u64,
    decoded: &mut Vec<Field>,
    expected: &mut Vec<Field>,
) -> Result<(), RunError> {
    if answer.query_id() != key.query_id() {
        return Err(RunError::ProductsShape {
            problem: "an answer is paired with another query's decoding key",
        });
    }
    let rows = instance.rows;
    ensure_capacity(decoded, rows)?;
    ensure_capacity(expected, rows)?;
    decode_into(answer, key, &mut decoded[..rows])?;
    plaintext_product_into(&instance.plaintext, plaintext_query, &mut expected[..rows]);
    if decoded[..rows] != expected[..rows] {
        return Err(RunError::Verification {
            matrix_id,
            query_id: key.query_id(),
        });
    }
    Ok(())
}

/// Grows a verification buffer to hold `len` elements if it is smaller,
/// filling any new slots with the field zero, and never shrinks it.
///
/// Growth is the buffer's only allocation moment: once the first round
/// has sized it to the largest row count, steady-state verification
/// reuses the storage unchanged.
///
/// # Errors
///
/// Returns the allocation failure when the reservation is refused.
fn ensure_capacity(buffer: &mut Vec<Field>, len: usize) -> Result<(), RunError> {
    if buffer.len() < len {
        buffer
            .try_reserve_exact(len - buffer.len())
            .map_err(|_reserve| RunError::Codec(CodecError::AllocationFailed))?;
        let zero = PrimeField::<PROTOCOL_MODULUS>::new().element_u32(0);
        buffer.resize(len, zero);
    }
    Ok(())
}

/// Writes the plaintext matrix-vector product of one query into `out`
/// (one slot per row), the reference every answer is compared with.
fn plaintext_product_into(plaintext: &[Field], query: &[Field], out: &mut [Field]) {
    let field = PrimeField::<PROTOCOL_MODULUS>::new();
    let ell = query.len();
    for (row, slot) in out.iter_mut().enumerate() {
        let mut accumulator = field.element_u32(0);
        for (column, &coefficient) in query.iter().enumerate() {
            accumulator += plaintext[row * ell + column] * coefficient;
        }
        *slot = accumulator;
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
#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::os::unix::net::UnixStream;
    use std::sync::{Arc, Mutex};
    use std::thread;

    use emvp::{AnswerRef, EmvpParams, EncryptedMatrix, answer_batch, answer_into};
    use prime_field_layer::PrimeField;
    use emvp_network::{
        EvaluateEntry, FrameKind, FrameReader, MatrixUpload, ProductEntry, PROTOCOL_VERSION,
        read_evaluate_payload, read_frame_header, read_upload_matrices_payload, server_handshake,
        write_client_hello, write_evaluate, write_products, write_upload_accepted,
        write_upload_matrices,
    };
    use trapdoor_matrices::ToeplitzFastProduct;

    use super::{Config, Instance, Session, derive_instance, run, verify_answer};
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

    /// How the fake server sabotages each matrix's first answer.
    #[derive(Clone, Copy)]
    enum Sabotage {
        /// Faithful answers.
        None,
        /// Flip one answer word, corrupting the product values.
        CorruptValues,
        /// Rename the answer with another query's identifier.
        WrongQueryId,
    }

    /// Rewrites one matrix's first answer in place, sabotaged by `kind`.
    fn sabotage_answer(kind: Sabotage, answers: &mut [emvp::AnswerMatrix<PROTOCOL_MODULUS>]) {
        let answer = &answers[0];
        answers[0] = match kind {
            Sabotage::CorruptValues => {
                let field = prime_field_layer::PrimeField::<PROTOCOL_MODULUS>::new();
                let mut values = answer.values().to_vec();
                values[0] -= field.element_u32(1);
                emvp::AnswerMatrix::from_parts(
                    answer.instance_id(),
                    answer.query_id(),
                    values,
                    answer.rows(),
                    answer.blocks(),
                )
            }
            Sabotage::WrongQueryId => emvp::AnswerMatrix::from_parts(
                answer.instance_id(),
                answer.query_id() + 1,
                answer.values().to_vec(),
                answer.rows(),
                answer.blocks(),
            ),
            Sabotage::None => emvp::AnswerMatrix::from_parts(
                answer.instance_id(),
                answer.query_id(),
                answer.values().to_vec(),
                answer.rows(),
                answer.blocks(),
            ),
        };
    }

    /// Serves one session the way the server binary does, except products
    /// are computed on the CPU, `sabotage` rewrites one answer word or
    /// descriptor of each matrix's first query, and `acknowledged`
    /// overrides the identifiers of the upload acknowledgment.
    fn fake_server(stream: &mut UnixStream, sabotage: Sabotage, acknowledged: Option<Vec<u64>>) {
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
                if !matches!(sabotage, Sabotage::None) {
                    sabotage_answer(sabotage, &mut answers);
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

    /// Re-derives the demo's instances from one config, in upload order.
    fn derive_instances(config: &Config) -> Vec<Instance> {
        let builder = toeplitz_builder(params().n().unwrap());
        config
            .rows
            .iter()
            .enumerate()
            .map(|(index, &rows)| derive_instance(config, index, rows, &builder).unwrap())
            .collect()
    }

    #[test]
    fn client_verifies_products_end_to_end() {
        let (mut client_stream, server_stream) = UnixStream::pair().unwrap();
        let server = thread::spawn(move || {
            let mut server_stream = server_stream;
            fake_server(&mut server_stream, Sabotage::None, None);
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
            fake_server(&mut server_stream, Sabotage::CorruptValues, None);
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
    fn wrong_query_id_in_an_answer_descriptor_is_rejected() {
        let (mut client_stream, server_stream) = UnixStream::pair().unwrap();
        let server = thread::spawn(move || {
            let mut server_stream = server_stream;
            fake_server(&mut server_stream, Sabotage::WrongQueryId, None);
        });
        let outcome = run::<ToeplitzFastProduct<PROTOCOL_MODULUS>, _, _>(
            &mut client_stream,
            &config(),
            toeplitz_builder(params().n().unwrap()),
        );

        assert!(matches!(
            outcome,
            Err(RunError::ProductsShape {
                problem: "an answer is paired with another query's decoding key"
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
            fake_server(&mut server_stream, Sabotage::None, Some(vec![1, 1]));
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

    #[test]
    fn two_rounds_reuse_workspaces_and_match() {
        let (mut client_stream, server_stream) = UnixStream::pair().unwrap();
        let server = thread::spawn(move || {
            let mut server_stream = server_stream;
            fake_server(&mut server_stream, Sabotage::None, None);
        });
        let config = config();
        let instances = derive_instances(&config);
        let mut session = Session::new(&mut client_stream).unwrap();
        assert!(session.upload(&instances, config.params).unwrap() > 0);

        let first = session.evaluate_and_verify(&instances, config.params).unwrap();
        let second = session
            .evaluate_and_verify(&instances, config.params)
            .unwrap();

        assert_eq!(first.verified_queries, 4);
        assert_eq!(second.verified_queries, first.verified_queries);
        assert_eq!(second.evaluate_bytes, first.evaluate_bytes);
        assert_eq!(second.products_bytes, first.products_bytes);
        assert_eq!(session.ack_ids.identifiers(), &[1, 2]);

        drop(session);
        drop(client_stream);
        server.join().unwrap();
    }

    /// A stream wrapper that records every byte written through it and
    /// passes reads straight to the inner stream.
    struct RecordingStream {
        inner: UnixStream,
        written: Arc<Mutex<Vec<u8>>>,
    }

    impl Write for RecordingStream {
        fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
            let written = self.inner.write(buf)?;
            self.written
                .lock()
                .unwrap()
                .extend_from_slice(&buf[..written]);
            Ok(written)
        }

        fn flush(&mut self) -> std::io::Result<()> {
            self.inner.flush()
        }
    }

    impl Read for RecordingStream {
        fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
            self.inner.read(buf)
        }
    }

    #[test]
    fn send_path_frames_match_the_owned_wrapper_bytes() {
        let (client_stream, server_stream) = UnixStream::pair().unwrap();
        let server = thread::spawn(move || {
            let mut server_stream = server_stream;
            fake_server(&mut server_stream, Sabotage::None, None);
        });
        let written = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&written);
        let mut client = RecordingStream {
            inner: client_stream,
            written,
        };

        let config = config();
        let instances = derive_instances(&config);

        // The golden byte sequence, built from the owned wrappers the
        // previous implementation sent through: hello, upload, evaluate.
        let mut expected = Vec::new();
        write_client_hello(&mut expected, PROTOCOL_VERSION, PROTOCOL_MODULUS).unwrap();
        let uploads: Vec<MatrixUpload> = instances
            .iter()
            .map(|instance| MatrixUpload {
                params: config.params,
                matrix: instance.encrypted.clone(),
            })
            .collect();
        write_upload_matrices(&mut expected, &uploads).unwrap();
        let entries: Vec<EvaluateEntry> = instances
            .iter()
            .zip(1_u64..)
            .map(|(instance, matrix_id)| EvaluateEntry {
                matrix_id,
                queries: instance.queries.clone(),
            })
            .collect();
        write_evaluate(&mut expected, &entries).unwrap();

        let report = run::<ToeplitzFastProduct<PROTOCOL_MODULUS>, _, _>(
            &mut client,
            &config,
            toeplitz_builder(params().n().unwrap()),
        )
        .unwrap();
        assert_eq!(report.verified_queries, 4);
        assert_eq!(&*recorded.lock().unwrap(), &expected);

        drop(client);
        server.join().unwrap();
    }

    /// A secure-mode structural test: the whole run completes from a
    /// fresh OS master seed. No equality is asserted between draws.
    #[test]
    fn a_fresh_os_master_seed_completes_the_session() {
        let (mut client_stream, server_stream) = UnixStream::pair().unwrap();
        let server = thread::spawn(move || {
            let mut server_stream = server_stream;
            fake_server(&mut server_stream, Sabotage::None, None);
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

    /// Steady-state verification of one answer must not allocate: the
    /// buffers arrive pre-sized from the warm-up call, and every step —
    /// the pairing check, the protocol decode, the plaintext product, and
    /// the comparison — runs inline on the calling thread.
    #[test]
    fn verify_answer_allocates_nothing_in_steady_state() {
        let config = config();
        let instances = derive_instances(&config);
        let instance = &instances[0];
        let params = config.params;
        let encrypted_query = &instance.queries[0];
        let key = &instance.decoding_keys[0];
        let rows = instance.rows;
        let blocks = params.blocks().unwrap();
        let field = PrimeField::<PROTOCOL_MODULUS>::new();
        let mut answer_values = vec![field.element_u32(0); rows * blocks];
        answer_into(&params, &instance.encrypted, encrypted_query, &mut answer_values).unwrap();
        let answer = AnswerRef::new(
            instance.encrypted.instance_id(),
            encrypted_query.query_id(),
            &answer_values,
            rows,
            blocks,
        )
        .unwrap();
        let mut decoded = Vec::new();
        let mut expected = Vec::new();
        verify_answer(
            &answer,
            key,
            &instance.plaintext_queries[0],
            instance,
            1,
            &mut decoded,
            &mut expected,
        )
        .unwrap();

        let allocations = allocation_counter::measure(|| {
            verify_answer(
                &answer,
                key,
                &instance.plaintext_queries[0],
                instance,
                1,
                &mut decoded,
                &mut expected,
            )
            .unwrap();
        });
        assert_eq!(allocations.count_total, 0);
    }
}
