# EMVP Thesis

This repo contains the work for my bachelor thesis.

The thesis handout can be found in the releases page as a compiled PDF.

## Building

This project has been tested on Linux only. It may work on other platforms but you're on your own.

You need a recent stable Rust toolchain, then run
```sh
cargo build --release
```

This produces the `client` and `server` binaries in `target/release/`.
The tests can be run with `cargo test --workspace`.

Notes:

- The workspace compiles with `-C target-cpu=native` (see `.cargo/config.toml`),
  so the resulting binaries are optimized for the machine they were built on and are not portable.
  It is highly recommended to keep this flag active, as performance degrades significantly when LLVM
  cannot vectorize operations using AVX2.
- The `server` uses the GPU through `wgpu`, which requires a working Vulkan driver at runtime
  (e.g. `mesa-vulkan-drivers` on AMD/Intel, or the vendor driver on NVIDIA).

To compile the handout you additionally need [typst](https://typst.app) and the appropriate fonts.

```sh
typst compile handout/thesis.typ build/thesis.pdf
```
### Nix users

Nix users can build everything with:

```sh
nix build            # Rust binaries (client and server)
nix build .#handout  # Thesis handout PDF
```

A development shell with all tools is available via `nix develop`.

Note that building the Rust binaries with nix disables `-C target-cpu=native`, which substantially
degrades performance.


## Licensing

The software in this repository is licensed under the [PolyForm Perimeter License 1.0.0](./LICENSE).

The accompanying handout and documentation are licensed separately under the 
[Creative Commons Attribution-NonCommercial-ShareAlike 4.0 International License](./handout/LICENSE).

The license applicable to the handout does not grant any additional rights to the software,
and the software license does not grant additional rights to the handout.
