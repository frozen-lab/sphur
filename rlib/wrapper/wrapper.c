#include "sphur.h"

int sphur_init_wrapper(sphur_t *state) { return sphur_init(state); }

int sphur_init_seeded_wrapper(sphur_t *state, uint64_t seed) {
  return sphur_init_seeded(state, seed);
}

uint64_t sphur_gen_rand_wrapper(sphur_t *state) {
  return sphur_gen_rand(state);
}

int sphur_gen_bool_wrapper(sphur_t *state) { return sphur_gen_bool(state); }

uint64_t sphur_gen_rand_range_wrapper(sphur_t *state, uint64_t min,
                                      uint64_t max) {
  return sphur_gen_rand_range(state, min, max);
}
