{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        devShells = {
        rs = pkgs.mkShell {
          name = "dev-rust";
          buildInputs = with pkgs; [
            gcc
            pkg-config
            rustc
            cargo
            rustfmt
            clippy
            rust-analyzer
          ];

          shellHook = ''
            export RUST_BACKTRACE=1
              
            echo " : $(rustc --version | cut -d' ' -f2)"
          '';
        };
        py = pkgs.mkShell {
          name = "dev-python";
          buildInputs = with pkgs; [
            gcc
            pkg-config
            python314
            ruff
            uv
            pyright
          ];

           shellHook = ''
            echo " : $(python3 --version)"
          '';
        };
        js = pkgs.mkShell {
          name = "dev-js";
          buildInputs = with pkgs; [
            gcc
            pkg-config
            nodejs
            typescript
          ];

           shellHook = ''
            echo " : $(node --version)"
          '';
        };
        c = pkgs.mkShell {
          name = "dev-c";
          buildInputs = with pkgs; [
            nasm
            gcc
            gcc.libc
            gdb
            glibc.static
            perf
          ];

          shellHook = ''
            echo " : $(gcc --version)"
          '';
        };
      };
     }
  );
}
