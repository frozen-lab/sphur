#ifndef SPHUR_H
#define SPHUR_H

#include <stddef.h>
#include <stdint.h>
#include <time.h>

// aarch specific imports
#if defined(__aarch64__) || defined(_M_ARM64)

#include <arm_neon.h>

#else

#include <cpuid.h>
#include <emmintrin.h>
#include <immintrin.h>

#endif

// -----------------------------------------------------------------------------
// Architecture guard! We only support 64-bit architectures (x86_64 and
// AArch64).
// -----------------------------------------------------------------------------
#if !defined(__x86_64__) && !defined(_M_X64) && !defined(__aarch64__) &&       \
    !defined(_M_ARM64)

#error "sphur: 64-bit architecture required (x86_64 or AArch64). \
32-bit targets (i386/armv7) are not supported."

#endif

// Enum representing SIMD extension ID's
typedef enum {
  SPHUR_SIMD_AVX2 = 0,
  SPHUR_SIMD_SSE2 = 1,
} _sphur_simd_ext_t;

// High performance pseudo-random number generator
//
// It holds,
//
// uint64_t _seeds[8] => sub-seeds used for prng mixer
// uint64_t _rands[4] => internal storage of random numbers
// unsigned _rcnt => number of nums in buffer in _rands (0..4)
// unsigned _rpos => current idx in _rands (0..3)
// _sphur_simd_ext_t _simd_ext => SIMD extension id (based on the availability)
//
// NOTE: `_rands` is intentionally NOT zeroed on init.
typedef struct {
  uint64_t _seeds[8];
  uint64_t _rands[4];
  unsigned _rcnt;
  unsigned _rpos;
  _sphur_simd_ext_t _simd_ext;
} sphur_t;

// -----------------------------------------------------------------------------
// Utils
// -----------------------------------------------------------------------------

#if defined(__x86_64__) || defined(_M_X64)

static inline void _sphur_detect_simd_ext(sphur_t *state) {
  unsigned eax, ebx, ecx, edx;

  if (__get_cpuid_max(0, NULL) >= 7) {
    __cpuid_count(7, 0, eax, ebx, ecx, edx);

    if (ebx & (1 << 5)) {
      state->_simd_ext = SPHUR_SIMD_AVX2;

      return;
    }
  }

  // default to baseline SSE2
  state->_simd_ext = SPHUR_SIMD_SSE2;
}

#endif

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

// -----------------------------------------------------------------------------
// SIMD Intrinsics
// -----------------------------------------------------------------------------

#if defined(__aarch64__) || defined(_M_ARM64)

// Generate 4 prng's w/ aarch64 neon using 8 sub-seeds
//
// Returns no. of PRNG's generated
//
// NOTE: The sub-seeds are updated after PRNG generation
static inline int _sphur_simd_neon_xorshiro_128_plus(uint64_t *seeds,
                                                     uint64_t *out) {
  // sanity check
  if (!seeds || !out)
    return -1;

  // load s0 lane (seeds[0..1])
  uint64x2_t s0 = vld1q_u64(seeds);

  // load s1 lane (seeds[2..3])
  uint64x2_t s1 = vld1q_u64(seeds + 2);

  // res = s0 + s1
  uint64x2_t res = vaddq_u64(s0, s1);

  // store 2 outputs
  vst1q_u64(out, res);

  // s1 ^= s0
  s1 = veorq_u64(s1, s0);

  // rol(s0,55)
  uint64x2_t rol55 = vsliq_n_u64(vshrq_n_u64(s0, 9), s0, 55);

  // new_s0 = rol(s0,55) ^ s1 ^ (s1 << 14)
  uint64x2_t s1_sh14 = vshlq_n_u64(s1, 14);
  uint64x2_t new_s0 = veorq_u64(veorq_u64(rol55, s1), s1_sh14);

  // new_s1 = rol(s1,36)
  uint64x2_t rol36 = vsliq_n_u64(vshrq_n_u64(s1, 28), s1, 36);

  // store updated state
  vst1q_u64(seeds, new_s0);
  vst1q_u64(seeds + 2, rol36);

  // repeat once more for seeds[4..7] to produce next 2 outputs
  s0 = vld1q_u64(seeds + 4);
  s1 = vld1q_u64(seeds + 6);

  res = vaddq_u64(s0, s1);
  vst1q_u64(out + 2, res);

  s1 = veorq_u64(s1, s0);
  rol55 = vsliq_n_u64(vshrq_n_u64(s0, 9), s0, 55);
  s1_sh14 = vshlq_n_u64(s1, 14);
  new_s0 = veorq_u64(veorq_u64(rol55, s1), s1_sh14);
  rol36 = vsliq_n_u64(vshrq_n_u64(s1, 28), s1, 36);

  vst1q_u64(seeds + 4, new_s0);
  vst1q_u64(seeds + 6, rol36);

  return 2;
}

#else

// Generate 4 prng's w/ x86_64 AVX2 using 8 sub-seeds
//
// Returns no. of PRNG's generated
//
// NOTE: The sub-seeds are updated after PRNG generation
__attribute__((target("avx2"))) static inline int
_sphur_simd_avx2_xorshiro_128_plus(uint64_t *seeds, uint64_t *out) {
  // sanity check
  if (!seeds || !out)
    return -1;

  // load s0 lane (seeds[0..3])
  __m256i s0 = _mm256_loadu_si256((const __m256i *)(const void *)seeds);

  // load s1 lane (seeds[4..7])
  __m256i s1 = _mm256_loadu_si256((const __m256i *)(const void *)(seeds + 4));

  // res = s0 + s1
  __m256i res = _mm256_add_epi64(s0, s1);

  // write 4 generated prng's to [out] buf
  _mm256_storeu_si256((__m256i *)(void *)out, res);

  //
  // Update old state (sub-seeds)
  //

  // s1 ^= s0
  s1 = _mm256_xor_si256(s1, s0);

  // rol(s0, 55) = (s0 << 55) | (s0 >> 9)
  __m256i left55 = _mm256_slli_epi64(s0, 55);
  __m256i right9 = _mm256_srli_epi64(s0, 9);
  __m256i rol55 = _mm256_or_si256(left55, right9);

  // s1 << 14
  __m256i s1_sh14 = _mm256_slli_epi64(s1, 14);

  // new_s0 = rol(s0,55) ^ s1 ^ (s1 << 14)
  __m256i new_s0 = _mm256_xor_si256(rol55, s1);
  new_s0 = _mm256_xor_si256(new_s0, s1_sh14);

  // new_s1 = rol(s1,36) = (s1 << 36) | (s1 >> 28)
  __m256i left36 = _mm256_slli_epi64(s1, 36);
  __m256i right28 = _mm256_srli_epi64(s1, 28);
  __m256i new_s1 = _mm256_or_si256(left36, right28);

  // storing the new state into [seeds] buf
  _mm256_storeu_si256((__m256i *)(seeds), new_s0);
  _mm256_storeu_si256((__m256i *)(seeds + 4), new_s1);

  // avoid AVX -> SSE transition penalty (just in case)
  _mm256_zeroupper();

  return 4;
}

// Generate 2 prng's w/ x86_64 SSE2 using 4 sub-seeds
//
// Returns no. of PRNG's generated
//
// NOTE: The sub-seeds are updated after PRNG generation
//
// NOTE: Only the initial 4 of 8 sub-seeds are being used here
// cause of register size
__attribute__((target("sse2"))) static inline int
_sphur_simd_sse2_xorshiro_128_plus(uint64_t *seeds, uint64_t *out) {
  // sanity check
  if (!seeds || !out)
    return -1;

  // load s0 lane (seeds[0..1])
  __m128i s0 = _mm_loadu_si128((const __m128i *)(const void *)seeds);

  // load s1 lane (seeds[2..3])
  __m128i s1 = _mm_loadu_si128((const __m128i *)(const void *)(seeds + 2));

  // res = s0 + s1
  __m128i res = _mm_add_epi64(s0, s1);

  // write 2 generated prng's to [out] buf
  _mm_storeu_si128((__m128i *)(void *)out, res);

  //
  // Update old state (sub-seeds)
  //

  // s1 ^= s0
  s1 = _mm_xor_si128(s1, s0);

  // rol(s0, 55) = (s0 << 55) | (s0 >> 9)
  __m128i left55 = _mm_slli_epi64(s0, 55);
  __m128i right9 = _mm_srli_epi64(s0, 9);
  __m128i rol55 = _mm_or_si128(left55, right9);

  // s1 << 14
  __m128i s1_sh14 = _mm_slli_epi64(s1, 14);

  // new s0 = rol(s0, 55) ^ s1 ^ (s1 << 14)
  __m128i new_s0 = _mm_xor_si128(rol55, s1);
  new_s0 = _mm_xor_si128(new_s0, s1_sh14);

  // new s1 = rol(s1, 36) = (s1 << 36) | (s1 >> 28)
  __m128i left36 = _mm_slli_epi64(s1, 36);
  __m128i right28 = _mm_srli_epi64(s1, 28);
  __m128i new_s1 = _mm_or_si128(left36, right28);

  // storing the new state into [seeds] buf
  _mm_storeu_si128((__m128i *)(seeds), new_s0);
  _mm_storeu_si128((__m128i *)(seeds + 2), new_s1);

  return 2;
}

#endif

static inline int _sphur_simd_xorshiro_128_plus(sphur_t *state) {
#if defined(__aarch64__) || defined(_M_ARM64)
  // NOTE: We assume all arm cpus support base neon, as in all x86_64
  // support sse2 impl!
  return _sphur_simd_neon_xorshiro_128_plus(state->_seeds, state->_rands);
#else
  switch (state->_simd_ext) {
  case SPHUR_SIMD_AVX2:
    return _sphur_simd_avx2_xorshiro_128_plus(state->_seeds, state->_rands);
  case SPHUR_SIMD_SSE2:
    return _sphur_simd_sse2_xorshiro_128_plus(state->_seeds, state->_rands);
  }

  // NOTE: This is a very rare error state, cause, all x86_64 bits CPU's
  // support sse2 as a strict requirment
  return -1;
#endif
}

// -----------------------------------------------------------------------------
// Public Interface
// -----------------------------------------------------------------------------

// Initialize sphur_t state
static inline int sphur_init(sphur_t *state) {
  if (!state)
    return -1;

  uint64_t seed = _sphur_gen_platform_seed();

  // detch simd extension ox x64
#if defined(__x86_64__) || defined(_M_X64)
  _sphur_detect_simd_ext(state);
#endif

  _sphur_splitmix64_generate(seed, state->_seeds, 8);
  state->_rpos = 0;
  state->_rcnt = 0;

  return 0;
}

// Initialize sphur_t state w/ an initial seed
static inline int sphur_init_seeded(sphur_t *state, uint64_t seed) {
  if (!state)
    return -1;

  // detch simd extension ox x64
#if defined(__x86_64__) || defined(_M_X64)
  _sphur_detect_simd_ext(state);
#endif

  _sphur_splitmix64_generate(seed, state->_seeds, 8);
  state->_rpos = 0;
  state->_rcnt = 0;

  return 0;
}

// Generate a random number
static inline __attribute__((always_inline)) uint64_t
sphur_gen_rand(sphur_t *state) {
  if (!state)
    return -1;

  // if empty refill buf
  if (__builtin_expect(state->_rcnt == 0, 0)) {

    int gen_count = _sphur_simd_xorshiro_128_plus(state);

    // error occurred while generating random numbers
    if (gen_count == -1) {
      return -1;
    }

    state->_rpos = 0;
    state->_rcnt = gen_count;
  }

  uint64_t out = state->_rands[state->_rpos++];

  // buffer consumed
  if (state->_rpos >= state->_rcnt) {
    state->_rcnt = 0;
    state->_rpos = 0;
  }

  return out;
}

// Generate a random boolean (1 or 0)
static inline int sphur_gen_bool(sphur_t *state) {
  if (!state)
    return -1;

  // mask lowest bit of random num
  return (int)(sphur_gen_rand(state) & 1ULL);
}

// Generate a random number in the range [min, max] (inclusive)
static inline uint64_t sphur_gen_rand_range(sphur_t *state, uint64_t min,
                                            uint64_t max) {
  if (!state || min > max)
    return 0;

  // rejection sampling to avoid modulo bias
  uint64_t span = max - min + 1;
  uint64_t limit = UINT64_MAX - (UINT64_MAX % span);

  uint64_t r = sphur_gen_rand(state);

  if (r <= limit)
    return min + (r % span);

  do {
    r = sphur_gen_rand(state);
  } while (r > limit);

  return min + (r % span);
}

#endif
