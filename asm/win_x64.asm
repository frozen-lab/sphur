bits 64
default rel

global asm_add_one

section .text
asm_add_one:
        mov rax, rcx
        add rax, 0x01
        ret
