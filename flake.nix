{
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  };

  outputs = { self, nixpkgs }: {
    devShells.x86_64-linux.default =
      let
        pkgs = import nixpkgs {
          system = "x86_64-linux";
        };
      in
      pkgs.mkShell {
        buildInputs = with pkgs; [
          nasm
          gcc
          gcc.libc
          gdb
          glibc.static
          perf
          rustc
          cargo
          rustfmt
          rust-analyzer
          clippy
          cargo-show-asm
          nodejs
          typescript
          python3
        ];
      };
  };
}
