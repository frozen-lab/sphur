#include "../include/sphur.h"
#include <stdio.h>

int main(void) {
  sphur_t ctx;
  uint64_t seed = 123456789ULL;

  if (sphur_init_seeded(&ctx, seed) != 0) {
    printf("Failed to init sphur with seed\n");
    return 1;
  }

  printf("sphur_init_seeded(%llu): success\n", (unsigned long long)seed);

  // Generate a few randoms
  for (int i = 0; i < 5; i++) {
    uint64_t r = sphur_gen_rand(&ctx);
    printf("rand[%d] = %llu\n", i, (unsigned long long)r);
  }

  return 0;
}
