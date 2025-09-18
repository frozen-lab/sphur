.global function_gen_seeds

// simd uitls (`./simd_a64.s`)
.extern function_split_mix_64

.text
// Generate N independent 64-bit "sub-seeds" based on a input seed
//
// Args:
//   x0 - Initial 64-bit seed
//   x1 - Pointer to output buffer (must be 8 * N bytes long)
//   x2 - No. of sub-seeds to generate (u64 & >0)
//
// Returns:
//   x0 - `0` on success, `1` otherwise
//   x1 - (preserved) buffer pointer (same as input)
function_gen_seeds:
        cbz x0, .err                // sanity check (x0 != 0)
        cbz x2, .err                // sanity check (x2 != 0)

        bl function_split_mix_64
        ret
.err:
        // exit(1)
        mov x0, #0x01
        ret
