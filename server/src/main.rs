//! The EMVP answer server: a thread-per-connection TCP server that keeps
//! encrypted matrices resident on the GPU and answers encrypted queries.
//!
//! Every connection negotiates the version and modulus handshake, uploads
//! one matrix set into the GPU residency budget, and evaluates encrypted
//! queries against it until the client disconnects. GPU operations are
//! serialized process-wide; matrix residency is capped by
//! `--max-gpu-bytes`.

mod coordinator;
mod executor;
mod failure;
mod session;

use std::net::TcpListener;
use std::process::ExitCode;
use std::sync::Arc;

use clap::Parser;

use crate::coordinator::GpuCoordinator;

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
    let gpu = match GpuCoordinator::new(args.max_gpu_bytes) {
        Ok(coordinator) => Arc::new(coordinator),
        Err(error) => {
            eprintln!("server: GPU initialization failed: {error}");
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
    eprintln!(
        "server: listening on {} (GPU residency budget {} bytes)",
        args.listen, args.max_gpu_bytes
    );
    for stream in listener.incoming() {
        let Ok(mut stream) = stream else {
            eprintln!("server: accept failed");
            continue;
        };
        let gpu = Arc::clone(&gpu);
        std::thread::spawn(move || {
            let peer = stream.peer_addr().map_or_else(
                |_| "unknown peer".to_string(),
                |address| address.to_string(),
            );
            match session::serve_connection(&mut stream, &*gpu) {
                Ok(()) => eprintln!("server: {peer} disconnected"),
                Err(error) => eprintln!("server: {peer} session failed: {error}"),
            }
        });
    }
    ExitCode::SUCCESS
}
