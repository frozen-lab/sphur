bits 64
default rel

global function_gen_seeds

; simd uitls (`./x64_simd.asm`)
extern function_split_mix_64

section .text

; Generate N independent 64-bit "sub-seeds" based on a input seed
;
; Args:
;   rdi - Initial 64-bit seed
;   rsi - Pointer to output buffer (must be 8 * N bytes long)
;   rdx - No. of sub-seeds to generate (u64 & >0)
;
; Retruns:
;   rax - `1` on error, `0` otherwise
;   rsi - (preserved) the input pointer is preserved, w/ the buffer
;         containing N 64-bit sub-seeds
function_gen_seeds:
        call function_split_mix_64
        ret
