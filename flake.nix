{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
      in
      {
        packages = rec {
          app = pkgs.rustPlatform.buildRustPackage {
            pname = "emvp-experiments";
            version = "0.1.0";
            src = ./.;
            cargoLock.lockFile = ./Cargo.lock;
          };
          default = app;
        };

        devShells.default = pkgs.mkShell {
          packages = with pkgs; [
            cargo
            rust-analyzer
            rustfmt
            rustc
            clippy

            perf
            cargo-flamegraph
            samply

            gnuplot

            # Runtime loader and diagnostics for the GPU answer path
            # (wgpu/Vulkan); `vulkaninfo` validates the target machine.
            vulkan-loader
            vulkan-tools
          ];

          # Cargo-built binaries dlopen libvulkan.so.1 at runtime, so the
          # loader library must be findable outside the shell's link-time
          # flags.
          shellHook = ''
            export LD_LIBRARY_PATH="${pkgs.vulkan-loader}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
          '';
        };
      }
    );
}
