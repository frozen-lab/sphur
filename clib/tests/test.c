#include "../include/sphur.h"
#include <assert.h>

#define SEED_COUNT (sizeof(state.seeds) / sizeof(state.seeds[0]))

int main(void) {
  sphur_state_t state;

  // w/ seed
  if (sphur_init_seeded(&state, 1234) == 0) {
    assert(SEED_COUNT == 8);
  } else {
    return 1;
  }

  // w/o seed
  if (sphur_init(&state) == 0) {
    assert(SEED_COUNT == 8);
  } else {
    return 1;
  }

  return 0;
}
