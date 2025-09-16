#include "../include/sphur.h"
#include <stdio.h>

#define SEED_COUNT (sizeof(state.seeds) / sizeof(state.seeds[0]))

int main(void) {
  sphur_state_t state;

  // w/ seed
  if (sphur_init_seeded(&state, 1234) == 0) {
    printf("Generated sub-seeds (w/ Seed): %zu\n", SEED_COUNT);
  } else {
    printf("Error initializing state!\n");
    return 1;
  }

  // w/o seed
  if (sphur_init(&state) == 0) {
    printf("Generated sub-seeds (w/o Seed): %zu\n", SEED_COUNT);
  } else {
    printf("Error initializing state!\n");
    return 1;
  }

  return 0;
}
