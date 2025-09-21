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
          default = pkgs.mkShell {
            name = "dev";
            buildInputs = with pkgs; [
              # c
              gcc
              gdb
              perf
            
              # rust
              rustc
              cargo
              rustfmt
              clippy
              rust-analyzer

              # python
              python314
              ruff
              uv
              pyright

              # js/ts
              nodejs
              typescript
            ];

            shellHook = ''
              export RUST_BACKTRACE=1
              
              echo " : $(rustc --version)"
              echo " : $(python3 --version)"
              echo " : $(node --version)"
              echo " : $(gcc --version)"
            '';
          };
        };
       }
    );
}
