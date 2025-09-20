#include "../include/sphur.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

#define ASSERT_MSG(cond, msg)                                                  \
  do {                                                                         \
    if (!(cond)) {                                                             \
      fprintf(stderr, "Assertion failed: %s\n", msg);                          \
      assert(cond);                                                            \
    }                                                                          \
  } while (0)

static void test_platform_seed(void) {
  uint64_t s1 = _sphur_gen_platform_seed();
  uint64_t s2 = _sphur_gen_platform_seed();

  ASSERT_MSG(s1 != 0, "platform seed must not be zero");
  ASSERT_MSG(s2 != 0, "platform seed must not be zero");

  // They may be equal by chance, but should differ most of the time
  if (s1 == s2) {
    printf("⚠️ Warning: two consecutive platform seeds are equal (%llu)\n",
           (unsigned long long)s1);
  }

  printf("test_platform_seed: PASS (%llu, %llu)\n", (unsigned long long)s1,
         (unsigned long long)s2);
}

static void test_splitmix64_basic(void) {
  uint64_t buf[4] = {0};
  int allzero = 1;

  _sphur_splitmix64_generate(12345ULL, buf, 4);

  for (int i = 0; i < 4; i++) {
    if (buf[i] != 0) {
      allzero = 0;

      break;
    }
  }

  ASSERT_MSG(!allzero, "splitmix64 output must not be all zeros");
  printf("test_splitmix64_basic: PASS (%llu, %llu, %llu, %llu)\n",
         (unsigned long long)buf[0], (unsigned long long)buf[1],
         (unsigned long long)buf[2], (unsigned long long)buf[3]);
}

static void test_splitmix64_reproducibility(void) {
  uint64_t buf1[4], buf2[4];
  _sphur_splitmix64_generate(98765ULL, buf1, 4);
  _sphur_splitmix64_generate(98765ULL, buf2, 4);

  for (int i = 0; i < 4; i++) {
    ASSERT_MSG(buf1[i] == buf2[i],
               "same seed must produce identical splitmix64 sequence");
  }

  uint64_t buf3[4];
  int diff = 0;

  _sphur_splitmix64_generate(11111ULL, buf3, 4);

  for (int i = 0; i < 4; i++) {
    if (buf1[i] != buf3[i]) {
      diff = 1;

      break;
    }
  }

  ASSERT_MSG(diff, "different seeds must produce different splitmix64 outputs");
  printf("test_splitmix64_reproducibility: PASS\n");
}

int main(void) {
  printf("----UTILS TESTS----\n\n");

  test_platform_seed();
  test_splitmix64_basic();
  test_splitmix64_reproducibility();

  printf("▶ All utils tests passed!\n\n");
  return 0;
}
