CC=gcc
CFLAGS=-Wall -Wextra -O2 -Iinclude
ASM=nasm
ASMFLAGS=-f elf64

OBJS=src/linux_x86.o src/sphur.o

all: libsphur.a

src/linux_x86.o: ./asm/linux_x86.asm
	$(ASM) $(ASMFLAGS) $< -o $@

src/sphur.o: src/sphur.c
	$(CC) $(CFLAGS) -c $< -o $@

libsphur.a: $(OBJS)
	ar rcs $@ $(OBJS)

example: libsphur.a examples/example.c
	$(CC) $(CFLAGS) examples/example.c libsphur.a -o example
	./example

test: libsphur.a tests/test_add_one.c
	$(CC) $(CFLAGS) tests/test_add_one.c libsphur.a -o test_runner
	./test_runner

clean:
	rm -f **/*.o libsphur.a example test_runner

