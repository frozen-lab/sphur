bits 64
default rel

global function_gen_seeds

; simd uitls (`./x64_simd.asm`)
extern function_split_mix_64
extern QueryPerformanceCounter

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
        test rcx, rcx
        jnz .split_mix

        sub rsp, 0x28                   ; shadow space
        lea rcx, [rsp + 0x20]           ; pointer to 64-bit buffer

        ;; this never fails ╰(*°▽°*)╯
        ;; let's believe in microsoft, for more info ->
        ;; `https://learn.microsoft.com/en-us/windows/win32/api/profileapi/nf-profileapi-queryperformancecounter`
        call QueryPerformanceCounter
        
        mov rcx, [rsp+0x20]             ; seed = performance counter
        add rsp, 0x28                   ; clear shadow space
.split_mix:
        ;; prepare args for split_mix (System V style)
        mov rdi, rcx
        mov rsi, rdx
        mov rdx, r8

        sub rsp, 0x20                   ; creating shadow space
        call function_split_mix_64
        add rsp, 0x20                   ; clear shadow space
        
        ret
