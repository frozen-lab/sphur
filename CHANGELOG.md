### V-0.3.0

- **BREAKING**: New and improved API for `Sphur`
- Improved internal structure for optimal performance
- Re-Impl of SSE SIMD engine w/ ISA and manual loop unrolling
- Re-Impl of NEON SIMD engine w/ ISA and manual loop unrolling

### V-0.2.0

- Update internal state structure `[u128; 16]` -> `[u64; 32]`
- Improved `gen_range` for both inclusive and exclusive range objects.
- **Breaking**: `gen_u128` is no more available.

### V-0.1.3

- Using median values in bench script
- Eliminate CPU boost and hot cache bias for benches.

### V-0.1.2

- Added distribution plot for visulization
- Updated docs
- Updated bench script

### V-0.1.0

- Initial version using SFMT algo
- Support for linux, windows and mac across 64-bit (x86 and aarch64)

### Proof-Of-Concept in C

- Completed minimal working prototype
- Header only C library
- SIMD Intrincis per ISA's.
- Good Perf (~250 Million PRNG's per second w/ AVX2)
