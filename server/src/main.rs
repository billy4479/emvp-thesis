//! The EMVP answer server: a thread-per-connection TCP server that keeps
//! encrypted matrices resident on the compute device and answers encrypted
//! queries.
//!
//! Every connection negotiates the version and modulus handshake, uploads
//! one matrix set into the answer engine's residency budget, and evaluates
//! encrypted queries against it until the client disconnects. The engine
//! owns the device and the `--max-gpu-bytes` residency budget; a machine
//! without a compute adapter falls back to answering every matrix on the
//! CPU.

mod failure;
mod session;

use std::net::TcpListener;
use std::process::ExitCode;
use std::sync::Arc;

use clap::Parser;
use emvp::AnswerEngine;
use emvp_network::PROTOCOL_MODULUS;

/// The EMVP answer server.
#[derive(Parser)]
struct Args {
    /// The address and port to listen on.
    #[arg(long, default_value = "127.0.0.1:4000")]
    listen: String,

    /// The maximum GPU memory usable by stored encrypted matrices, in
    /// bytes.
    #[arg(long)]
    max_gpu_bytes: u64,
}

fn main() -> ExitCode {
    let args = Args::parse();
    let engine = match AnswerEngine::<PROTOCOL_MODULUS>::new(args.max_gpu_bytes) {
        Ok(engine) => Arc::new(engine),
        Err(error) => {
            eprintln!("server: answer engine initialization failed: {error}");
            return ExitCode::FAILURE;
        }
    };
    let listener = match TcpListener::bind(&args.listen) {
        Ok(listener) => listener,
        Err(error) => {
            eprintln!("server: cannot listen on {}: {error}", args.listen);
            return ExitCode::FAILURE;
        }
    };
    if engine.has_device() {
        eprintln!(
            "server: listening on {} (gpu engine, matrix residency budget {} bytes)",
            args.listen, args.max_gpu_bytes
        );
    } else {
        eprintln!(
            "server: listening on {} (cpu fallback, no device residency budget is active)",
            args.listen
        );
    }
    for stream in listener.incoming() {
        let Ok(mut stream) = stream else {
            eprintln!("server: accept failed");
            continue;
        };
        let engine = Arc::clone(&engine);
        std::thread::spawn(move || {
            let peer = stream.peer_addr().map_or_else(
                |_| "unknown peer".to_string(),
                |address| address.to_string(),
            );
            match session::serve_connection(&mut stream, &engine) {
                Ok(()) => eprintln!("server: {peer} disconnected"),
                Err(error) => eprintln!("server: {peer} session failed: {error}"),
            }
        });
    }
    ExitCode::SUCCESS
}
