bits 64
default rel

global _function_gen_seeds

; simd uitls (`./simd_x64.asm`)
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
;   rax - `0` on success, `1` otherwise
;   rsi - (preserved) the input pointer is preserved, w/ the buffer
;         containing N 64-bit sub-seeds
_function_gen_seeds:
        push r9                       ; used as temp 
        
        test rdi, rdi
        jnz .split_mix

        mov r9, rdx

        ;; read the CPU time stamp counter
        ;; 
        ;; NOTE: it outputs 64-bit split across
        ;; `rax` & `rdx` (in the lower 32 bits each)
        rdtsc
        
        xor rax, rdx         ; combine 32 + 32 to get 64 bit value
        mov rdi, rax         ; seed = rax ^ rdx

        mov rdx, r9
.split_mix:
        call function_split_mix_64
.ret:
        pop r9
        ret
