#include "../include/sphur.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

static void dump_u64_buf(const char *label, const uint64_t *buf, size_t n) {
  printf("%s:", label);

  for (size_t i = 0; i < n; i++) {
    printf(" %llu", (unsigned long long)buf[i]);
  }

  printf("\n");
}

// -----------------------------------------------------------------------------
// AVX2
// -----------------------------------------------------------------------------

static void test_avx2_null_args(void) {
  uint64_t seeds[8] = {0};
  uint64_t out[4] = {0};

  assert(_sphur_simd_avx2_xorshiro_128_plus(NULL, out) == -1);
  assert(_sphur_simd_avx2_xorshiro_128_plus(seeds, NULL) == -1);
}

static void test_avx2_basic_run(void) {
  uint64_t seeds[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t out[4] = {0};

  int ret = _sphur_simd_avx2_xorshiro_128_plus(seeds, out);
  assert(ret == 0);

  // ensure output was filled (nonzero)
  int allzero = 1;

  for (int i = 0; i < 4; i++) {
    if (out[i] != 0) {
      allzero = 0;

      break;
    }
  }

  assert(!allzero);

  // ensure seeds updated (not identical to original)
  uint64_t new_seeds[8];
  memcpy(new_seeds, seeds, sizeof(seeds));

  int changed =
      memcmp(new_seeds, (uint64_t[]){1, 2, 3, 4, 5, 6, 7, 8}, sizeof(seeds));
  assert(changed != 0);

  dump_u64_buf("AVX2 out", out, 4);
  dump_u64_buf("AVX2 seeds", seeds, 8);
}

static void test_avx2_multiple_runs(void) {
  uint64_t seeds[8] = {42, 43, 44, 45, 46, 47, 48, 49};
  uint64_t out1[4], out2[4];

  _sphur_simd_avx2_xorshiro_128_plus(seeds, out1);
  _sphur_simd_avx2_xorshiro_128_plus(seeds, out2);

  // outputs should differ
  int same = memcmp(out1, out2, sizeof(out1));
  assert(same != 0);

  dump_u64_buf("AVX2 out1", out1, 4);
  dump_u64_buf("AVX2 out2", out2, 4);
}

// -----------------------------------------------------------------------------
// SSE2
// -----------------------------------------------------------------------------

static void test_sse2_null_args(void) {
  uint64_t seeds[8] = {0};
  uint64_t out[2] = {0};

  assert(_sphur_simd_sse2_xorshiro_128_plus(NULL, out) == -1);
  assert(_sphur_simd_sse2_xorshiro_128_plus(seeds, NULL) == -1);
}

static void test_sse2_basic_run(void) {
  uint64_t seeds[8] = {9, 10, 11, 12, 13, 14, 15, 16};
  uint64_t out[2] = {0};

  int ret = _sphur_simd_sse2_xorshiro_128_plus(seeds, out);
  assert(ret == 0);

  // output not all zero
  int allzero = (out[0] == 0 && out[1] == 0);
  assert(!allzero);

  // seeds updated
  uint64_t old[4] = {9, 10, 11, 12};
  assert(memcmp(seeds, old, sizeof(old)) != 0);

  dump_u64_buf("SSE2 out", out, 2);
  dump_u64_buf("SSE2 seeds", seeds, 8);
}

static void test_sse2_multiple_runs(void) {
  uint64_t seeds[8] = {100, 101, 102, 103, 104, 105, 106, 107};
  uint64_t out1[2], out2[2];

  _sphur_simd_sse2_xorshiro_128_plus(seeds, out1);
  _sphur_simd_sse2_xorshiro_128_plus(seeds, out2);

  // outputs should differ
  int same = memcmp(out1, out2, sizeof(out1));
  assert(same != 0);

  dump_u64_buf("SSE2 out1", out1, 2);
  dump_u64_buf("SSE2 out2", out2, 2);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(void) {
  printf("----SIMD TESTS----\n\n");

  test_avx2_null_args();
  test_avx2_basic_run();
  test_avx2_multiple_runs();

  test_sse2_null_args();
  test_sse2_basic_run();
  test_sse2_multiple_runs();

  printf("▶ All tests passed!\n\n");
  return 0;
}
