bits 64
default rel

global function_gen_seeds

; simd uitls (`./x64_simd.asm`)
extern function_split_mix_64

section .bss
        time_val resq 0x02        ; 16-bytes for `tval` struct

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
function_gen_seeds:
        test rdi, rdi
        jnz .split_mix
.gen_seed:
        lea rdi, [time_val]
        call function_clock_get_time

        test rax, rax
        jnz .err

        ;; generate µs since epoch
        mov rdi, qword [time_val]      ; seed = tv_sec
        imul rdi, rdi, 1000000         ; seed = seed * 1_000_000
        add rdi, qword [time_val + 8]  ; seed += tv_usec
.split_mix:
        call function_split_mix_64
        ret
.err:
        mov rax, 0x01
        ret

; obtain current epoch time using `clock_gettime` syscall
; 
; Args:
;   rdi - Pointer to out buf (16 bytes) to store `tval` struct
;
; Returns:
;   rax - `0` on success, `-1` otherwise
;
; Clobbers:
;   rax
function_clock_get_time:
        push rsi
        
        ;; `clock_gettime` syscall
        mov rax, 0xE4
        mov rsi, rdi
        xor rdi, rdi
        syscall

        pop rsi

        ret
        
