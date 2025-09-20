# Sphūr

[![Unit Tests](https://github.com/frozen-lab/sphur/actions/workflows/unit_test.yaml/badge.svg)](https://github.com/frozen-lab/sphur/actions/workflows/unit_test.yaml)
[![Last Commits](https://img.shields.io/github/last-commit/frozen-lab/sphur?logo=git&logoColor=white)](https://github.com/frozen-lab/sphur/commits/main)
[![Pull Requests](https://img.shields.io/github/issues-pr/frozen-lab/sphur?logo=github&logoColor=white)](https://github.com/frozen-lab/sphur/pulls)
[![GitHub Issues or Pull Requests](https://img.shields.io/github/issues/frozen-lab/sphur?logo=github&logoColor=white)](https://github.com/frozen-lab/sphur/issues)
[![License](https://img.shields.io/github/license/frozen-lab/sphur?logo=open-source-initiative&logoColor=white)](https://github.com/frozen-lab/sphur/blob/master/LICENSE)

Sphūr (स्फुर्) is SIMD accelerated Pseudo-Random Number Generator.
It's fast, header-only, and designed for 64-bit numbers.

## Platforms

| Platform         |  Support  |
|------------------|:---------:|
| Windows x86_64   |    ✅     |
| Windows ARM64    |    ✅     |
| Linux x86_64     |    ✅     |
| Linux ARM64      |    ✅     |
| macOS x86_64     |    ✅     |
| macOS ARM64      |    ✅     |

## Installation

Simply download the header file,

```sh
curl -O https://raw.githubusercontent.com/frozen-lab/sphur/refs/heads/master/clib/include/sphur.h
```

## Exmaple

```c
#include "sphur.h"
#include <stdint.h>
#include <stdio.h>

int main(void) {
 sphur_t sphur;

 if (sphur_init(&sphur) != 0) {
   printf("Failed to init sphur\n");
   return 1;
 }

 uint64_t dice = sphur_gen_rand_range(&sphur, 1, 6);
 printf("Rolled a dice: %lu\n", dice);

 return 0;
}
```

and now simply include the `sphur.h` header file into your compilation,

```sh
gcc -Wall -Wextra -O2 main.c sphur.h -o main
```

## Usage

### Default Initilization

```c
sphur_t rng;

if (sphur_init(&rng) != 0) {
    printf("Failed to initialize Sphūr\n");
    return 1;
}
```

### Initilization w/ Custom Seed

```c
sphur_t rng;
uint64_t seed = 123456789ULL;

if (sphur_init_seeded(&rng, seed) != 0) {
    printf("Failed to initialize Sphūr\n");
    return 1;
}
```

### Generate 64-bit random numer

```c
uint64_t r = sphur_gen_rand(&rng);
printf("Random number: %lu\n", r);
```
### Generate random numer w/ range

```c
uint64_t dice = sphur_gen_rand_range(&rng, 1, 6);
printf("Rolled a dice: %llu\n", dice);
```

### Generate random boolean

```c
int coin = sphur_gen_bool(&rng);
printf("Coin flip: %s\n", coin ? "Heads" : "Tails");
```

