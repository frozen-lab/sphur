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
;   rax - `0` on success, `1` otherwise
;   rdx - (preserved) the input pointer is preserved, w/ the buffer
;         containing N 64-bit sub-seeds
function_gen_seeds:
        push r9
        
        test rcx, rcx
        jnz .split_mix

        mov r9, rdx

        ;; read the CPU time stamp counter
        rdtsc
        
        xor rax, rdx
        mov rcx, rax         ; seed = rax ^ rdx

        mov rdx, r9
.split_mix:
        ;; prepare args for split_mix (System V style)
        mov rdi, rcx
        mov rsi, rdx
        mov rdx, r8

        sub rsp, 0x28                   ; shadow space
        call function_split_mix_64
        add rsp, 0x28                   ; clear shadow space

        pop r9
        ret
