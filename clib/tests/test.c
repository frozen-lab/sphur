#include "../include/sphur.h"
#include <assert.h>

int main(void) {
  sphur_state_t state;

  if (sphur_init(&state, 1234) != 0) {
    return 1;
  }

  // validate seeds
  for (int i = 0; i < 4; i++) {
    assert(state.seeds[i] > 0);
  }

  return 0;
}
