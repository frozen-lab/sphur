bits 64
default rel

global _function_gen_seeds

; simd uitls (`./x64_simd.asm`)
extern function_split_mix_64
extern _gettimeofday

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
_function_gen_seeds:
        test rdi, rdi
        jnz .split_mix

        ;; we generate custom seed here (current epoch micro-seconds)
        lea rdi, [time_val]
        call function_get_time

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

; obtain current epoch time using gettimeofday
;
; Args:
;   rdi - Pointer to out buf (16 bytes) to store timeval { tv_sec, tv_usec }
;
; Returns:
;   rax - `0` on success, `-1` on error
;
; Clobbers:
;   rax
function_get_time:
        push rsi

        ; rdi already = pointer to timeval
        xor rsi, rsi         ; timezone arg = NULL
        call _gettimeofday   ; int gettimeofday(struct timeval*, void*)

        pop rsi
        
        ret
