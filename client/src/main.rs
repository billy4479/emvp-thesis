//! The EMVP demo client: it deterministically generates plaintext
//! matrices and queries, encrypts them, uploads the encrypted matrices to
//! the server, evaluates the encrypted queries, decodes the returned
//! products, and verifies them against the plaintext products.

mod demo;
mod error;
mod session;

use std::net::TcpStream;
use std::process::ExitCode;
use std::time::{Duration, Instant};

use clap::{Parser, ValueEnum};
use emvp::ProtocolError;
use rand_chacha::ChaCha20Rng;
use trapdoor_matrices::{IrreducibleRingLpn, RaaWeightedProduct, ToeplitzFastProduct};

use crate::demo::{
    PROTOCOL_MODULUS, RAA_FACTOR, RING_LPN_WEIGHT, check_ring_lpn_width, select_params,
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

    /// Deterministic seed for every key, matrix, and query.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// The trapdoored mask construction to encrypt with.
    #[arg(long, value_enum, default_value_t = Mask::RingLpn)]
    mask: Mask,

    /// The record length `ell`; `k` and `b` are searched at lambda 128.
    #[arg(long, default_value_t = 512)]
    width: usize,
}

/// The compiled mask constructions.
#[derive(Clone, Copy, Debug, ValueEnum)]
enum Mask {
    /// Irreducible Ring-LPN with the compiled fixed secret weight.
    RingLpn,
    /// Fast Toeplitz products.
    Toeplitz,
    /// Weighted RAA products with the compiled repetition factor.
    Raa,
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
        check_ring_lpn_width(&params)?;
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
    let config = Config {
        rows: args.rows.clone(),
        queries: args.queries,
        seed: args.seed,
        params,
    };

    let mut stream = TcpStream::connect(&args.server)?;
    let handshake_start = Instant::now();
    emvp_network::client_handshake(&mut stream)?;
    let handshake_duration = handshake_start.elapsed();
    eprintln!(
        "client: connected to {} ({:?})",
        args.server, handshake_duration
    );

    let report = match args.mask {
        Mask::RingLpn => run::<IrreducibleRingLpn<PROTOCOL_MODULUS>, _, _>(
            &mut stream,
            &config,
            ring_builder(code_width),
        ),
        Mask::Toeplitz => run::<ToeplitzFastProduct<PROTOCOL_MODULUS>, _, _>(
            &mut stream,
            &config,
            toeplitz_builder(code_width),
        ),
        Mask::Raa => run::<RaaWeightedProduct<PROTOCOL_MODULUS>, _, _>(
            &mut stream,
            &config,
            raa_builder(code_width),
        ),
    }?;
    print_report(&report, args, total_start.elapsed(), handshake_duration);
    Ok(())
}

fn ring_builder(
    width: usize,
) -> impl Fn(&mut ChaCha20Rng, usize) -> Result<IrreducibleRingLpn<PROTOCOL_MODULUS>, ProtocolError>
{
    move |stream, _index| {
        Ok(
            trapdoor_matrices::testing::ring_block::<PROTOCOL_MODULUS, _>(
                width,
                RING_LPN_WEIGHT,
                stream,
            )?,
        )
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

fn print_report(report: &Report, args: &Args, total: Duration, handshake: Duration) {
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
        report.derive_duration, report.verify_duration, handshake, total
    );
}
