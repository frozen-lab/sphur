#include "../include/sphur.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define ASSERT_MSG(cond, msg)                                                  \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "Assertion failed: %s\n", msg);                          \
      assert(cond);                                                            \
    }                                                                          \
  } while (0)

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

#if defined(__x86_64__) || defined(_M_X64)

static void test_avx2_null_args(void) {
  uint64_t seeds[8] = {0};
  uint64_t out[4] = {0};

  ASSERT_MSG(_sphur_simd_avx2_xorshiro_128_plus(NULL, out) == -1,
             "NULL seeds must fail");
  ASSERT_MSG(_sphur_simd_avx2_xorshiro_128_plus(seeds, NULL) == -1,
             "NULL out must fail");
}

static void test_avx2_basic_run(void) {
  uint64_t seeds[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t out[4] = {0};
  int allzero = 1;

  int ret = _sphur_simd_avx2_xorshiro_128_plus(seeds, out);

  ASSERT_MSG(ret == 4, "AVX2 must produce 4 outputs");

  for (int i = 0; i < 4; i++) {
    if (out[i] != 0) {
      allzero = 0;

      break;
    }
  }

  uint64_t old[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  ASSERT_MSG(!allzero, "Output must not be all zero");
  ASSERT_MSG(memcmp(seeds, old, sizeof(old)) != 0, "Seeds must update");

  dump_u64_buf("AVX2 out", out, 4);
  dump_u64_buf("AVX2 seeds", seeds, 8);
}

static void test_avx2_multiple_runs(void) {
  uint64_t seeds[8] = {42, 43, 44, 45, 46, 47, 48, 49};
  uint64_t out1[4], out2[4];

  _sphur_simd_avx2_xorshiro_128_plus(seeds, out1);
  _sphur_simd_avx2_xorshiro_128_plus(seeds, out2);

  ASSERT_MSG(memcmp(out1, out2, sizeof(out1)) != 0,
             "Two consecutive runs must differ");

  dump_u64_buf("AVX2 out1", out1, 4);
  dump_u64_buf("AVX2 out2", out2, 4);
}

static void test_avx2_output_count(void) {
  uint64_t seeds[8] = {1, 2, 3, 4, 5, 6, 7, 8};
  uint64_t out[4];

  for (int i = 0; i < 4; i++)
    out[i] = 0xDEADBEEFCAFEBABEULL;

  int ret = _sphur_simd_avx2_xorshiro_128_plus(seeds, out);
  assert(ret == 4);

  // check exactly first 4 entries changed
  for (int i = 0; i < 4; i++)
    assert(out[i] != 0xDEADBEEFCAFEBABEULL);
}

#endif

// -----------------------------------------------------------------------------
// SSE2
// -----------------------------------------------------------------------------

#if defined(__x86_64__) || defined(_M_X64)

static void test_sse2_null_args(void) {
  uint64_t seeds[8] = {0};
  uint64_t out[2] = {0};

  ASSERT_MSG(_sphur_simd_sse2_xorshiro_128_plus(NULL, out) == -1,
             "NULL seeds must fail");
  ASSERT_MSG(_sphur_simd_sse2_xorshiro_128_plus(seeds, NULL) == -1,
             "NULL out must fail");
}

static void test_sse2_basic_run(void) {
  uint64_t seeds[8] = {9, 10, 11, 12, 13, 14, 15, 16};
  uint64_t out[2] = {0};

  int ret = _sphur_simd_sse2_xorshiro_128_plus(seeds, out);

  ASSERT_MSG(ret == 2, "SSE2 must produce 2 outputs");
  ASSERT_MSG(!(out[0] == 0 && out[1] == 0), "Output must not be all zero");

  uint64_t old[4] = {9, 10, 11, 12};
  ASSERT_MSG(memcmp(seeds, old, sizeof(old)) != 0, "Seeds must update");

  dump_u64_buf("SSE2 out", out, 2);
  dump_u64_buf("SSE2 seeds", seeds, 8);
}

static void test_sse2_multiple_runs(void) {
  uint64_t seeds[8] = {100, 101, 102, 103, 104, 105, 106, 107};
  uint64_t out1[2], out2[2];

  _sphur_simd_sse2_xorshiro_128_plus(seeds, out1);
  _sphur_simd_sse2_xorshiro_128_plus(seeds, out2);

  ASSERT_MSG(memcmp(out1, out2, sizeof(out1)) != 0,
             "Two consecutive runs must differ");

  dump_u64_buf("SSE2 out1", out1, 2);
  dump_u64_buf("SSE2 out2", out2, 2);
}

static void test_sse2_output_count(void) {
  uint64_t seeds[8] = {10, 11, 12, 13, 14, 15, 16, 17};
  uint64_t out[4];

  for (int i = 0; i < 4; i++)
    out[i] = 0xDEADBEEFCAFEBABEULL;

  int ret = _sphur_simd_sse2_xorshiro_128_plus(seeds, out);
  assert(ret == 2);

  // check first 2 changed
  for (int i = 0; i < 2; i++)
    assert(out[i] != 0xDEADBEEFCAFEBABEULL);

  // check remaining 2 untouched
  for (int i = 2; i < 4; i++)
    assert(out[i] == 0xDEADBEEFCAFEBABEULL);
}

#endif

// -----------------------------------------------------------------------------
// NEON
// -----------------------------------------------------------------------------

#if defined(__aarch64__) || defined(_M_ARM64)

static void test_neon_null_args(void) {
  uint64_t seeds[8] = {0};
  uint64_t out[4] = {0};

  ASSERT_MSG(_sphur_simd_neon_xorshiro_128_plus(NULL, out) == -1,
             "NULL seeds must fail");
  ASSERT_MSG(_sphur_simd_neon_xorshiro_128_plus(seeds, NULL) == -1,
             "NULL out must fail");
}

static void test_neon_basic_run(void) {
  uint64_t seeds[8] = {21, 22, 23, 24, 25, 26, 27, 28};
  uint64_t out[4] = {0};

  int ret = _sphur_simd_neon_xorshiro_128_plus(seeds, out);
  ASSERT_MSG(ret == 2, "NEON must produce 2 outputs (2x64-bit)");

  int allzero = 1;
  for (int i = 0; i < 4; i++) {
    if (out[i] != 0) {
      allzero = 0;

      break;
    }
  }

  ASSERT_MSG(!allzero, "Output must not be all zero");

  uint64_t old[8] = {21, 22, 23, 24, 25, 26, 27, 28};
  ASSERT_MSG(memcmp(seeds, old, sizeof(old)) != 0, "Seeds must update");

  dump_u64_buf("NEON out", out, 4);
  dump_u64_buf("NEON seeds", seeds, 8);
}

static void test_neon_multiple_runs(void) {
  uint64_t seeds[8] = {200, 201, 202, 203, 204, 205, 206, 207};
  uint64_t out1[4], out2[4];

  _sphur_simd_neon_xorshiro_128_plus(seeds, out1);
  _sphur_simd_neon_xorshiro_128_plus(seeds, out2);

  ASSERT_MSG(memcmp(out1, out2, sizeof(out1)) != 0,
             "Two consecutive runs must differ");

  dump_u64_buf("NEON out1", out1, 4);
  dump_u64_buf("NEON out2", out2, 4);
}

static void test_neon_output_count(void) {
  uint64_t seeds[8] = {20, 21, 22, 23, 24, 25, 26, 27};
  uint64_t out[4];

  for (int i = 0; i < 4; i++)
    out[i] = 0xDEADBEEFCAFEBABEULL;

  int ret = _sphur_simd_neon_xorshiro_128_plus(seeds, out);
  assert(ret == 2);

  // check first 2 changed
  for (int i = 0; i < 2; i++)
    assert(out[i] != 0xDEADBEEFCAFEBABEULL);

  // check remaining 2 untouched
  for (int i = 2; i < 4; i++)
    assert(out[i] == 0xDEADBEEFCAFEBABEULL);
}

#endif

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(void) {
  printf("----SIMD TESTS----\n\n");

#if defined(__aarch64__) || defined(_M_ARM64) || defined(__arm64__)
  test_neon_null_args();
  test_neon_basic_run();
  test_neon_multiple_runs();
  test_neon_output_count();
#else
  test_avx2_null_args();
  test_avx2_basic_run();
  test_avx2_multiple_runs();
  test_avx2_output_count();

  test_sse2_null_args();
  test_sse2_basic_run();
  test_sse2_multiple_runs();
  test_sse2_output_count();
#endif

  printf("▶ All SIMD tests passed!\n\n");
  return 0;
}
