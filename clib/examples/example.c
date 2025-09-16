#include "../include/sphur.h"
#include <stdio.h>

int main(void) {
  sphur_state_t state;

  if (sphur_init(&state, 1234) != 0) {
    printf("Error initializing state!\n");
    return 1;
  }

  printf("Generated sub-seeds:\n");

  for (int i = 0; i < 8; i++) {
    printf("seed[%d] = %llu\n", i, (unsigned long long)state.seeds[i]);
  }

  return 0;
}
