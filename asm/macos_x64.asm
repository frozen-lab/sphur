global _asm_add_one

section .text
_asm_add_one:
        mov rax, rdi
        add rax, 0x01
        ret
        
