//! The EMVP demo client: it deterministically generates plaintext
//! matrices and queries, encrypts them, uploads the encrypted matrices to
//! the server, evaluates the encrypted queries, decodes the returned
//! products, and verifies them against the plaintext products.

mod demo;
mod error;
mod session;

use std::net::TcpStream;
use std::process::ExitCode;
use std::time::Instant;

use clap::{Parser, ValueEnum};
use emvp::{ProtocolError, TdmMask};
use rand_chacha::ChaCha20Rng;
use trapdoor_matrices::{IrreducibleRingLpn, RaaWeightedProduct, ToeplitzFastProduct};

use crate::demo::{
    CONTEXT_RAA, CONTEXT_RING, CONTEXT_TOEPLITZ, PROTOCOL_MODULUS, RAA_FACTOR, RING_LPN_WEIGHT,
    check_ring_lpn_configuration, master_seed_from_u64, random_master_seed, select_params,
};
use crate::error::RunError;
use crate::session::{Config, Report, run};

/// The EMVP demo client.
#[derive(Parser)]
struct Args {
    /// The server address to connect to.
    #[arg(long, default_value = "127.0.0.1:4000")]
    server: String,

    /// Comma-separated matrix row counts; one matrix per count.
    #[arg(long, value_delimiter = ',', default_value = "128,256")]
    rows: Vec<usize>,

    /// Encrypted query vectors generated per matrix.
    #[arg(long, default_value_t = 4)]
    queries: usize,

    /// Reproducible (and insecure) demo seed. Explicit values make every
    /// key, matrix, and query bit-for-bit reproducible, but the seed
    /// carries at most 64 bits and is echoed to the log. Omit this flag
    /// for a fresh 256-bit master seed drawn from the operating system,
    /// which is never printed.
    #[arg(long)]
    seed: Option<u64>,

    /// The trapdoored mask construction to encrypt with.
    #[arg(long, value_enum, default_value_t = Mask::Toeplitz)]
    mask: Mask,

    /// The record length `ell`; `k` and `b` are searched at lambda 128.
    #[arg(long, default_value_t = 512)]
    width: usize,
}

/// The compiled mask constructions.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mask {
    /// Fast Toeplitz products.
    Toeplitz,
    /// Weighted RAA products with the compiled repetition factor.
    Raa,
    /// Irreducible Ring-LPN with the compiled fixed secret weight; only
    /// ring degrees the parameter policy assesses sound are accepted.
    RingLpn,
}

/// The stable context identifier of one mask choice.
const fn mask_context(mask: Mask) -> emvp::MaskContextId {
    match mask {
        Mask::Toeplitz => CONTEXT_TOEPLITZ,
        Mask::Raa => CONTEXT_RAA,
        Mask::RingLpn => CONTEXT_RING,
    }
}

fn main() -> ExitCode {
    let args = Args::parse();
    match demo_run(&args) {
        Ok(()) => ExitCode::SUCCESS,
        Err(error) => {
            eprintln!("client: {error}");
            ExitCode::FAILURE
        }
    }
}

fn demo_run(args: &Args) -> Result<(), RunError> {
    let total_start = Instant::now();
    let params = select_params(args.width)?;
    if matches!(args.mask, Mask::RingLpn) {
        check_ring_lpn_configuration(&params)?;
    }
    let code_width = params.n().map_err(RunError::Params)?;
    eprintln!(
        "client: parameters k = {} ell = {} b = {} lambda = {} (n = {}, s = {}), mask {:?}",
        params.k,
        params.ell,
        params.b,
        params.lambda,
        code_width,
        params.blocks().map_err(RunError::Params)?,
        args.mask,
    );
    let master_seed = if let Some(seed) = args.seed {
        eprintln!("client: using the reproducible (insecure) demo seed {seed}");
        master_seed_from_u64(seed)
    } else {
        let seed = random_master_seed()?;
        eprintln!("client: drawing a fresh 256-bit master seed from the OS");
        seed
    };
    let config = Config {
        rows: args.rows.clone(),
        queries: args.queries,
        master_seed,
        params,
        mask_context: mask_context(args.mask),
    };

    let report = match args.mask {
        Mask::RingLpn => connect_and_run::<IrreducibleRingLpn<PROTOCOL_MODULUS>, _>(
            &args.server,
            &config,
            ring_builder(code_width),
        ),
        Mask::Toeplitz => connect_and_run::<ToeplitzFastProduct<PROTOCOL_MODULUS>, _>(
            &args.server,
            &config,
            toeplitz_builder(code_width),
        ),
        Mask::Raa => connect_and_run::<RaaWeightedProduct<PROTOCOL_MODULUS>, _>(
            &args.server,
            &config,
            raa_builder(code_width),
        ),
    }?;
    print_report(&report, args, total_start.elapsed());
    Ok(())
}

/// Connects to the server and runs one whole session on the connection.
///
/// The connection handshake happens exactly once, inside
/// [`session::run`]; this helper owns only the TCP connection itself.
///
/// # Errors
///
/// Returns the connect, handshake, transport, server, protocol, or
/// verification failure.
fn connect_and_run<M, F>(server: &str, config: &Config, build_block: F) -> Result<Report, RunError>
where
    M: TdmMask<PROTOCOL_MODULUS>,
    F: Fn(&mut ChaCha20Rng, usize) -> Result<M, ProtocolError> + Sync,
{
    let connect_start = Instant::now();
    let mut stream = TcpStream::connect(server)?;
    let connect_duration = connect_start.elapsed();
    eprintln!(
        "client: connected to {server} (connect {connect_duration:?}); handshaking and running"
    );
    run(&mut stream, config, build_block)
}

fn ring_builder(
    width: usize,
) -> impl Fn(&mut ChaCha20Rng, usize) -> Result<IrreducibleRingLpn<PROTOCOL_MODULUS>, ProtocolError>
{
    move |stream, _index| {
        // Fails closed on an assessed-broken configuration; the compiled
        // weight fixes the secret density.
        let sampled =
            IrreducibleRingLpn::<PROTOCOL_MODULUS>::sample(width, RING_LPN_WEIGHT, stream)
                .map_err(ProtocolError::Mask)?;
        for warning in sampled.warnings() {
            eprintln!("client: ring-lpn parameter warning: {warning}");
        }
        Ok(sampled.into_instance())
    }
}

fn toeplitz_builder(
    width: usize,
) -> impl Fn(&mut ChaCha20Rng, usize) -> Result<ToeplitzFastProduct<PROTOCOL_MODULUS>, ProtocolError>
{
    move |stream, _index| Ok(ToeplitzFastProduct::sample(width, stream)?)
}

fn raa_builder(
    width: usize,
) -> impl Fn(&mut ChaCha20Rng, usize) -> Result<RaaWeightedProduct<PROTOCOL_MODULUS>, ProtocolError>
{
    move |stream, _index| {
        Ok(RaaWeightedProduct::sample_nonzero(
            width, RAA_FACTOR, stream,
        )?)
    }
}

fn print_report(report: &crate::session::Report, args: &Args, total: std::time::Duration) {
    println!(
        "client: verified {} query products across {} matrices (ids {:?})",
        report.verified_queries,
        args.rows.len(),
        report.matrix_ids
    );
    println!(
        "client: uploaded {} KiB in {:?}, evaluated {} KiB and received {} KiB in {:?}",
        report.upload_bytes / 1024,
        report.upload_duration,
        report.evaluate_bytes / 1024,
        report.products_bytes / 1024,
        report.evaluate_duration,
    );
    println!(
        "client: derive+encrypt {:?}, verify {:?}, handshake {:?}, total {:?}",
        report.derive_duration, report.verify_duration, report.handshake_duration, total
    );
}

#[cfg(test)]
mod tests {
    use std::net::TcpListener;
    use std::thread;

    use emvp_network::{
        FrameKind, FrameReader, ProductEntry, read_evaluate_payload, read_frame_header,
        read_upload_matrices_payload, server_handshake, write_products, write_upload_accepted,
    };
    use trapdoor_matrices::ToeplitzFastProduct;

    use crate::demo::{CONTEXT_TOEPLITZ, PROTOCOL_MODULUS, master_seed_from_u64, select_params};
    use crate::session::Config;

    use super::{connect_and_run, toeplitz_builder};

    /// Serves one connection the way the server binary does, computing
    /// products on the CPU. It asserts that the first post-handshake
    /// frame is the matrix upload: a second client handshake would
    /// surface here as an unknown frame kind (the hello's magic byte
    /// `E` is not a frame kind), which fails this thread and the test.
    fn fake_tcp_server(listener: &TcpListener) {
        let (mut stream, _) = listener.accept().unwrap();
        server_handshake(&mut stream).unwrap();
        let header = read_frame_header(&mut stream).unwrap().unwrap();
        assert_eq!(header.kind, FrameKind::UploadMatrices);
        let mut frame = FrameReader::new(&mut stream, header.payload_len);
        let uploads = read_upload_matrices_payload(&mut frame).unwrap();
        frame.finish().unwrap();

        let identifiers: Vec<u64> = (1..=uploads.len() as u64).collect();
        write_upload_accepted(&mut stream, &identifiers).unwrap();

        loop {
            let Some(header) = read_frame_header(&mut stream).unwrap() else {
                return;
            };
            assert_eq!(header.kind, FrameKind::Evaluate);
            let mut frame = FrameReader::new(&mut stream, header.payload_len);
            let entries = read_evaluate_payload(&mut frame).unwrap();
            frame.finish().unwrap();

            let products: Vec<ProductEntry> = entries
                .iter()
                .map(|entry| {
                    let upload = &uploads[(entry.matrix_id - 1) as usize];
                    let answers =
                        emvp::answer_batch(&upload.params, &upload.matrix, &entry.queries).unwrap();
                    ProductEntry {
                        matrix_id: entry.matrix_id,
                        answers,
                    }
                })
                .collect();
            write_products(&mut stream, &products).unwrap();
        }
    }

    #[test]
    fn orchestration_handshakes_once_and_completes() {
        let listener = TcpListener::bind("127.0.0.1:0").unwrap();
        let address = listener.local_addr().unwrap().to_string();
        let server = thread::spawn(move || fake_tcp_server(&listener));

        let params = select_params(8).unwrap();
        let config = Config {
            rows: vec![2],
            queries: 1,
            master_seed: master_seed_from_u64(42),
            params,
            mask_context: CONTEXT_TOEPLITZ,
        };
        let report = connect_and_run::<ToeplitzFastProduct<PROTOCOL_MODULUS>, _>(
            &address,
            &config,
            toeplitz_builder(params.n().unwrap()),
        )
        .unwrap();

        assert_eq!(report.matrix_ids, vec![1]);
        assert_eq!(report.verified_queries, 1);
        server.join().unwrap();
    }
}
