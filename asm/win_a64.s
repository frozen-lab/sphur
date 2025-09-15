.global asm_add_one
.type asm_add_one, %function

.text
asm_add_one:
    add x0, x0, #1
    ret
