.text
.global function_split_mix_64
.align 2

// Generate N independent 64-bit "sub-seeds" based on a input seed
//
// Args:
//   x0 - Initial 64-bit seed
//   x1 - Pointer to output buffer (must be 8 * N bytes long)
//   x2 - No. of sub-seeds to generate (u64 & >0)
//
// Retruns:
//   x0 - `0` on success, `1` otherwise 
// 
// Clobbers:
//   NA
function_split_mix_64:
        cbz x2, .err                   // sanity check

        // prolouge
        stp x3, x4, [sp, #-16]!
        stp x5, xzr, [sp, #-16]!

        mov x3, x0                     // z = seed
        mov x4, #0x00                  // loop_counter = 0
.seed_loop:
        // x5 = golden_ratio
        movz x5, #0x7C15
        movk x5, #0x9F4A, lsl #16
        movk x5, #0x79B9, lsl #32
        movk x5, #0x9E37, lsl #48

        // z += golden_ratio
        add x3, x3, x5

        // 1st mix (diffuse high bits into low)
        // i.e. z += z ^ 30
        lsr x5, x3, #0x1E
        eor x3, x3, x5

        // x5 = MULT_ONE
        movz x5, #0xE5B9
        movk x5, #0x1CE4, lsl #16
        movk x5, #0x476D, lsl #32
        movk x5, #0xBF58, lsl #48
        
        // z *= MULT_ONE
        mul x3, x3, x5

        // 2nd mix (z += z ^ 27)
        lsr x5, x3, #0x1B
        eor x3, x3, x5

        // x5 = MULT_TWO
        movz x5, #0x11EB
        movk x5, #0x1331, lsl #16
        movk x5, #0x49BB, lsl #32
        movk x5, #0x94D0, lsl #48
        
        // z *= MULT_TWO
        mul x3, x3, x5

        // 3rd mix (z += z ^ 31)
        lsr x5, x3, #0x1F
        eor x3, x3, x5

        // buf[i] = z (store the generated seed)
        str x3, [x1, x4, lsl #0x03]

        // advance loop
        add x4, x4, #0x01

        cmp x4, x2
        blt .seed_loop
.success:
        // epilouge
        ldp x5, xzr, [sp], #16
        ldp x3, x4, [sp], #16
        
        mov x0, #0x00
        ret
.err:
        mov x0, #0x01
        ret
