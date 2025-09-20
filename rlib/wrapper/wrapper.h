#pragma once
#include "../../clib/include/sphur.h"

#ifdef __cplusplus
extern "C" {
#endif

int sphur_init_wrapper(sphur_t *state);
int sphur_init_seeded_wrapper(sphur_t *state, uint64_t seed);
uint64_t sphur_gen_rand_wrapper(sphur_t *state);
int sphur_gen_bool_wrapper(sphur_t *state);
uint64_t sphur_gen_rand_range_wrapper(sphur_t *state, uint64_t min,
                                      uint64_t max);

#ifdef __cplusplus
}
#endif
