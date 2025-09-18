#include "../include/sphur.h"

// ASM Function: Generates N 64-bit sub-seeds based on input seed
// Returns: 0 = ok, 1 = error
int function_gen_seeds(uint64_t seed, uint64_t *out, uint64_t n);

int sphur_init(sphur_state_t *state) {
  // error (invalid state pointer)
  if (!state)
    return 1;

// NOTE: We generate platform seed by default for arm
#if defined(__aarch64__) || defined(_M_ARM64)

#include <stdint.h>
#include <time.h>

  uint64_t generate_platform_seed(void) {
    struct timespec ts;
    timespec_get(&ts, TIME_UTC);

    return ((uint64_t)ts.tv_sec << 32) ^ (uint64_t)ts.tv_nsec;
  }

  uint64_t seed = generate_platform_seed();
  int ret = function_gen_seeds(seed, state->seeds, 8);
#else
  int ret = function_gen_seeds((uint64_t)0, state->seeds, 8);
#endif

  return ret;
}

int sphur_init_seeded(sphur_state_t *state, uint64_t seed) {
  // error (invalid state pointer)
  if (!state)
    return 1;

  int ret = function_gen_seeds(seed, state->seeds, 8);
  return ret;
}
