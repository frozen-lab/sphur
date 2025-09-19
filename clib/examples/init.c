#include "../include/sphur.h"
#include <stdio.h>

int main(void) {
  sphur_t ctx;
  if (sphur_init(&ctx) != 0) {
    printf("Failed to init sphur\n");
    return 1;
  }

  printf("sphur_init: success\n");

  // Generate a few randoms
  for (int i = 0; i < 5; i++) {
    uint64_t r = sphur_gen_rand(&ctx);
    printf("rand[%d] = %llu\n", i, (unsigned long long)r);
  }

  return 0;
}
