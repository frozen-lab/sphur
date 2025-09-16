#include "../include/sphur.h"

// ASM Function: Generates N 64-bit sub-seeds based on input seed
// Returns: 0 = ok, 1 = error
int function_split_mix_64(uint64_t seed, uint64_t *out, uint64_t n);

int sphur_init(sphur_state_t *state, uint64_t seed) {
  // error (invalid state pointer)
  if (!state)
    return 1;

  int ret = function_split_mix_64(seed, state->seeds, 8);
  return ret;
}
