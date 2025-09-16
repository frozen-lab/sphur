bits 64
default rel

global function_split_mix_64

section .rodata
        GOLDEN_RATIO: dq 0x9E3779B97F4A7C15
        MULT_ONE: dq 0xBF58476D1CE4E5B9
        MULT_TWO: dq 0x94D049BB133111EB

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
;
; Clobbers:
;   rax
function_split_mix_64:
        ;; preamble
        push rcx
        push r8
        push r9
        
        mov rax, rdi         ; z = initial_seed
        mov rcx, 0x00        ; loop counter

        test rdx, rdx
        jz .err
.seed_loop:
        lea r8, [GOLDEN_RATIO]
        add rax, [r8]        ; z += golden_ratio

        ;; 1st mix (diffuse high bits into low)
        mov r9, rax
        shr r9, 0x1E         ; r9 >> 30
        xor rax, r9          ; xor into rax

        ;; inp *= C1 (this spreads bit correlation nonlinearly)
        lea r8, [MULT_ONE]
        imul rax, [r8]

        ;; 2nd mix (diffusion)
        mov r9, rax
        shr r9, 0x1B        ; r9 >> 27
        xor rax, r9

        ;; inp *= C2
        lea r8, [MULT_TWO]
        imul rax, [r8]

        ;; 3rd mix
        mov r9, rax
        shr r9, 0x1F        ; r9 >> 31
        xor rax, r9

        ;; out[i] = rax
        ;; store the ith (rcx) sub-seed
        mov [rsi + rcx * 8], rax     
        inc rcx

        cmp rcx, rdx
        jl .seed_loop
        
        jmp .success
.err:
        ;; exit(1)
        mov rax, 0x01
        jmp .ret
.success:
        ;; exit(0)
        xor rax, rax
.ret:
        pop r9
        pop r8
        pop rcx

        ret
