#ifndef SPHUR_H
#define SPHUR_H

#include <stdint.h>

typedef struct {
  uint64_t seeds[8];
} sphur_state_t;

int sphur_init(sphur_state_t *state, uint64_t seed);

#endif
