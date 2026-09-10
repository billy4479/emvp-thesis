{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";

    typix = {
      url = "github:loqusion/typix";
      inputs.nixpkgs.follows = "nixpkgs";
    };
  };

  outputs =
    {
      self,
      nixpkgs,
      flake-utils,
      typix,
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        # allowUnfree: corefonts in the font set below.
        pkgs = import nixpkgs {
          inherit system;
          config.allowUnfree = true;
        };

        typixLib = typix.lib.${system};
        fonts = with pkgs; [
          nerd-fonts.fira-code
          ubuntu-sans
          noto-fonts-color-emoji

          corefonts
        ];

        handoutArgs = {
          src = typixLib.cleanTypstSource ./handout;

          fontPaths = map (pkg: "${pkg}/share/fonts/truetype") fonts;
          emojiFont = "noto";

          unstable_typstPackages = [
            {
              name = "physica";
              version = "0.9.7";
              hash = "sha256-G2M428hUZ8g7IwDjK9gRji9owpTiqn5I+QYMS4XcyfA=";
            }
          ];

          pname = "thesis-handout";
          version = "0.1.0";
          typstSource = "thesis.typ";
          typstOutput = "build/thesis.pdf";
        };

        handoutDrv = typixLib.buildTypstProject handoutArgs;

        handoutBuildScript = typixLib.buildTypstProjectLocal handoutArgs // {
          scriptName = "typst-build-handout";
        };

        handoutWatchScript =
          typixLib.watchTypstProject (
            builtins.removeAttrs handoutArgs [
              "src"
              "unstable_typstPackages"
              "pname"
              "version"
            ]
            // {
              typstSource = "handout/thesis.typ";
            }
          )
          // {
            scriptName = "typst-watch-handout";
          };

        # Self-asserting test suite for the bibliography renderer; a failed
        # assertion fails the build. The raw handout directory is used as
        # source since cleanTypstSource only keeps files reachable from the
        # handout entry point.
        bibliographyTestsDrv = typixLib.buildTypstProject ((builtins.removeAttrs handoutArgs [
          "src"
        ]) // {
          src = ./handout;
          pname = "bibliography-tests";
          typstSource = "lib/bibliography-tests.typ";
          typstOutput = "bibliography-tests.pdf";
        });
      in
      {
        packages = rec {
          app = pkgs.rustPlatform.buildRustPackage {
            pname = "emvp-experiments";
            version = "0.1.0";
            src = ./.;
            cargoLock.lockFile = ./Cargo.lock;
          };
          handout = handoutDrv;
          default = app;
          script = handoutWatchScript;
        };

        apps = {
          build-handout = flake-utils.lib.mkApp {
            drv = handoutBuildScript;
          };
          watch-handout = flake-utils.lib.mkApp {
            drv = handoutWatchScript;
          };
        };

        checks = {
          bibliography-tests = bibliographyTestsDrv;
        };

        devShells = {
          default = pkgs.mkShell {
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

              vulkan-loader
              vulkan-tools

              typstyle
              tinymist
              typst
              handoutWatchScript
            ];

            # Cargo-built binaries dlopen libvulkan.so.1 at runtime, so the
            # loader library must be findable outside the shell's link-time
            # flags.
            shellHook = ''
              export LD_LIBRARY_PATH="${pkgs.vulkan-loader}/lib''${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
            '';
          };
        };
      }
    );
}
