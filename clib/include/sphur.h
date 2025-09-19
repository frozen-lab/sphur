#ifndef SPHUR_H
#define SPHUR_H

#include <stddef.h>
#include <stdint.h>
#include <time.h>

// -----------------------------------------------------------------------------
// Architecture guard! We only support 64-bit architectures (x86_64 and
// AArch64).
// -----------------------------------------------------------------------------
#if !defined(__x86_64__) && !defined(_M_X64) && !defined(__aarch64__) &&       \
    !defined(_M_ARM64)

#error "sphur: 64-bit architecture required (x86_64 or AArch64). \
32-bit targets (i386/armv7) are not supported."

#endif

/*
High performance pseudo-random number generator

  uint64_t _seeds[8];      // sub-seeds used for prng mixer
  uint64_t _rands[8];      // internal storage of random numbers
  uint64_t *_rbuf;         // buffer pointer for _rands, default to NULL
  unsigned _rcnt;          // number of rands in _rands (1..8)
  unsigned _rpos;          // positon of number in _rands (1..8)

NOTE: `_rands` is intentionally NOT zeroed on init.
*/
typedef struct {
  uint64_t _seeds[8];
  uint64_t _rands[8];
  uint64_t *_rbuf;
  unsigned _rcnt;
  unsigned _rpos;
} sphur_t;

// Seed generator to generate default seed
static inline uint64_t _sphur_gen_platform_seed(void) {
  // for x64 we use RDTSC
#if defined(__x86_64__) || defined(_M_X64)
  unsigned int hi, lo;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((uint64_t)hi << 32) | lo;
#else
  struct timespec ts;
  timespec_get(&ts, TIME_UTC);

  return ((uint64_t)ts.tv_sec << 32) ^ (uint64_t)ts.tv_nsec;
#endif
}

// Gereates N sub-seeds to be used in PRNG mixer
static inline void _sphur_splitmix64_generate(uint64_t seed, uint64_t *buf,
                                              size_t n) {
  uint64_t z = seed;

  for (size_t i = 0; i < n; i++) {
    // add golden ration
    z = z + 0x9E3779B97F4A7C15ULL;

    uint64_t x = z;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    x = x ^ (x >> 31);

    buf[i] = x;
  }
}

// check if internal rand buffer is empty
static inline int _sphur_rands_empty(const sphur_t *state) {
  return state == NULL || state->_rbuf == NULL || state->_rcnt == 0;
}

// mark filled internal rand buffer w/ `count` (count <= 8)
static inline void _sphur_rands_mark_filled(sphur_t *state, unsigned count) {
  if (!state)
    return;

  if (count == 0 || count > 8) {
    state->_rbuf = NULL;
    state->_rpos = 0;
    state->_rcnt = count;

    return;
  }

  state->_rbuf = state->_rands;
  state->_rpos = 0;
}

// consume one random value from the buffer
static inline void _sphur_rands_consume_one(sphur_t *state) {
  if (!state || state->_rbuf == NULL || state->_rcnt == 0)
    return;

  ++state->_rpos;

  // exhausted
  if (state->_rpos >= state->_rcnt) {
    state->_rbuf = NULL;
    state->_rcnt = 0;
    state->_rpos = 0;
  }
}

// get next available random without modifying state
static inline int _sphur_rands_peek(const sphur_t *state, uint64_t *out) {
  if (!state || state->_rbuf == NULL || state->_rcnt == 0 || !out)
    return 1;

  *out = state->_rbuf[state->_rpos];
  return 0;
}

// Initialize sphur_t state
static inline int sphur_init(sphur_t *state) {
  if (!state)
    return 1;

  uint64_t seed = _sphur_gen_platform_seed();

  _sphur_splitmix64_generate(seed, state->_seeds, 8);
  state->_rbuf = NULL;
  state->_rpos = 0;
  state->_rcnt = 0;

  return 0;
}

// Initialize sphur_t state w/ an initial seed
static inline int sphur_init_seeded(sphur_t *state, uint64_t seed) {
  if (!state)
    return 1;

  _sphur_splitmix64_generate(seed, state->_seeds, 8);
  state->_rbuf = NULL;
  state->_rpos = 0;
  state->_rcnt = 0;

  return 0;
}

// Generate random numbers
static inline uint64_t sphur_gen_rand(sphur_t *state) {
  if (!state)
    return 1;

  // if no rands buffered, refill from seeds
  if (_sphur_rands_empty(state)) {
    for (unsigned i = 0; i < 8; ++i) {
      state->_rands[i] = state->_seeds[i];
    }

    _sphur_rands_mark_filled(state, 8);
  }

  // fetch current number
  uint64_t out = state->_rands[state->_rpos];

  // advance position
  _sphur_rands_consume_one(state);

  return out;
}

#endif
