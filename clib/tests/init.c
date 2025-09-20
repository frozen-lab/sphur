#include "../include/sphur.h"
#include <assert.h>
#include <stdio.h>

#define ASSERT_MSG(cond, msg)                                                  \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "Assertion failed: %s\n", msg);                          \
      assert(cond);                                                            \
    }                                                                          \
  } while (0)

void test_init(void) {
  sphur_t ctx;
  int rc = sphur_init(&ctx);

  ASSERT_MSG(rc == 0, "sphur_init should return 0");
  ASSERT_MSG(ctx._rcnt == 0, "_rcnt should start at 0");
  ASSERT_MSG(ctx._rpos == 0, "_rpos should start at 0");

#if defined(__x86_64__) || defined(_M_X64)
  ASSERT_MSG(ctx._simd_ext == SPHUR_SIMD_AVX2 ||
                 ctx._simd_ext == SPHUR_SIMD_SSE2,
             "SIMD extension must be AVX2 or SSE2");
#endif

  int all_zero = 1;

  for (int i = 0; i < 8; i++) {
    if (ctx._seeds[i] != 0) {
      all_zero = 0;

      break;
    }
  }

  ASSERT_MSG(!all_zero, "Seeds must not be all zero");
  printf("test_init_default: PASS\n");
}

void test_init_seeded(void) {
  sphur_t ctx;
  int rc = sphur_init_seeded(&ctx, 123456789ULL);

  ASSERT_MSG(rc == 0, "sphur_init_seeded should return 0");
  ASSERT_MSG(ctx._rcnt == 0, "_rcnt should start at 0");
  ASSERT_MSG(ctx._rpos == 0, "_rpos should start at 0");

#if defined(__x86_64__) || defined(_M_X64)
  ASSERT_MSG(ctx._simd_ext == SPHUR_SIMD_AVX2 ||
                 ctx._simd_ext == SPHUR_SIMD_SSE2,
             "SIMD extension must be AVX2 or SSE2");
#endif

  int all_zero = 1;

  for (int i = 0; i < 8; i++) {
    if (ctx._seeds[i] != 0) {
      all_zero = 0;

      break;
    }
  }

  ASSERT_MSG(!all_zero, "Seeded seeds must not be all zero");
  printf("test_init_seeded: PASS\n");
}

void test_seed_reproducibility(void) {
  sphur_t ctx1, ctx2, ctx3;
  sphur_init_seeded(&ctx1, 42);
  sphur_init_seeded(&ctx2, 42);
  sphur_init_seeded(&ctx3, 1337);

  for (int i = 0; i < 8; i++) {
    ASSERT_MSG(ctx1._seeds[i] == ctx2._seeds[i],
               "Same seed should produce identical sub-seeds");
  }

  int diff = 0;

  for (int i = 0; i < 8; i++) {
    if (ctx1._seeds[i] != ctx3._seeds[i]) {
      diff = 1;

      break;
    }
  }

  ASSERT_MSG(diff, "Different seeds should produce different sub-seeds");
  printf("test_seed_reproducibility: PASS\n");
}

void test_sanity_check(void) {
  ASSERT_MSG(sphur_init(NULL) == -1, "sphur_init(NULL) should return -1");
  ASSERT_MSG(sphur_init_seeded(NULL, 123) == -1,
             "sphur_init_seeded(NULL, ..) should return -1");

  printf("test_null: PASS\n");
}

int main(void) {
  printf("----INIT TESTS----\n\n");

  test_init();
  test_init_seeded();
  test_seed_reproducibility();
  test_sanity_check();

  printf("▶ All tests passed!\n\n");
  return 0;
}
