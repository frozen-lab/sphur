bits 64
default rel

global function_gen_seeds

; simd uitls (`./x64_simd.asm`)
extern function_split_mix_64

section .text

; Generate N independent 64-bit "sub-seeds" based on a input seed
;
; Args:
;   rcx - Initial 64-bit seed
;   rdx - Pointer to output buffer (must be 8 * N bytes long)
;   r8 - No. of sub-seeds to generate (u64 & >0)
;
; Retruns:
;   rax - `1` on error, `0` otherwise
;   rdx - (preserved) the input pointer is preserved, w/ the buffer
;         containing N 64-bit sub-seeds
function_gen_seeds:
        ;; prepare args for split_mix (System V style)
        mov rdi, rcx
        mov rsi, rdx
        mov rdx, r8

        sub rsp, 0x20                ; creating shadow space
        call function_split_mix_64
        add rsp, 0x20                ; destroying shadow space
        
        ret
