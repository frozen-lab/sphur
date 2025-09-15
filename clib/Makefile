CC=gcc
CFLAGS=-Wall -Wextra -O2 -Iinclude
ASM=nasm

UNAME_S := $(shell uname -s)
UNAME_M := $(shell uname -m)

# Linux
ifeq ($(UNAME_S),Linux)
  # x86-64
  ifeq ($(UNAME_M),x86_64)
    ASM_SRC=../asm/linux_x64.asm
    ASMFLAGS=-f elf64
    ASM_OBJ=src/linux_x64.o

    $(ASM_OBJ): $(ASM_SRC)
		$(ASM) $(ASMFLAGS) $< -o $@
  endif

  # AArch64
  ifeq ($(UNAME_M),aarch64)
    ASM_SRC=../asm/linux_a64.s
    ASM_OBJ=src/linux_a64.o

    $(ASM_OBJ): $(ASM_SRC)
		$(CC) $(CFLAGS) -c $< -o $@
  endif
endif

# Windows (x64)
ifeq ($(findstring MINGW,$(UNAME_S)),MINGW)
	ASM_SRC=../asm/win_x64.asm
	ASMFLAGS=-f win64
	ASM_OBJ=src/win_x64.o

    $(ASM_OBJ): $(ASM_SRC)
		$(ASM) $(ASMFLAGS) $< -o $@

endif

# MacOS (x64)
ifeq ($(UNAME_S),Darwin)
	ASM_SRC=../asm/macos_x64.asm
	ASMFLAGS=-f macho64
	ASM_OBJ=src/macos_x64.o

    $(ASM_OBJ): $(ASM_SRC)
		$(ASM) $(ASMFLAGS) $< -o $@  
endif

OBJS=$(ASM_OBJ) src/sphur.o

all: libsphur.a

src/sphur.o: src/sphur.c
	$(CC) $(CFLAGS) -c $< -o $@

libsphur.a: $(OBJS)
	ar rcs $@ $(OBJS)

example: libsphur.a examples/example.c
	$(CC) $(CFLAGS) examples/example.c libsphur.a -o example
	./example
	$(MAKE) clean

test: libsphur.a tests/test_add_one.c
	$(CC) $(CFLAGS) tests/test_add_one.c libsphur.a -o test_runner
	./test_runner
	$(MAKE) clean

clean:
	rm -f src/*.o libsphur.a example test_runner
