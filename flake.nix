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
            name = "dev-mode";
            buildInputs = with pkgs; [
              # c/asm
              gcc
              gcc.libc
              gdb
              glibc.static
              perf
              pkg-config
              gnumake
              clang-tools
            
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
          clib = pkgs.mkShell {
            name = "clib-prod";
            buildInputs = with pkgs; [
              gcc
              gnumake
            ];
  
            shellHook = ''
              echo " : $(gcc --version)"
            '';
          };
        };
       }
    );
}
