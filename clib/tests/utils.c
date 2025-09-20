#include "../include/sphur.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

// -----------------------------------------------------------------------------
// Tests for `_sphur_gen_platform_seed`
// -----------------------------------------------------------------------------

static void test_platform_seed(void) {
  uint64_t s1 = _sphur_gen_platform_seed();
  uint64_t s2 = _sphur_gen_platform_seed();

  // seeds should not always be equal
  assert(s1 != 0);
  assert(s2 != 0);

  printf("platform_seed: %llu, %llu\n", (unsigned long long)s1,
         (unsigned long long)s2);
}

// -----------------------------------------------------------------------------
// Tests for `_sphur_splitmix64_generate`
// -----------------------------------------------------------------------------

static void test_splitmix64_basic(void) {
  uint64_t buf[4] = {0};
  _sphur_splitmix64_generate(12345ULL, buf, 4);

  // buffer must not be all zeros
  int allzero = 1;

  for (int i = 0; i < 4; i++) {
    if (buf[i] != 0) {
      allzero = 0;

      break;
    }
  }

  assert(!allzero);

  printf("splitmix64 out: %llu %llu %llu %llu\n", (unsigned long long)buf[0],
         (unsigned long long)buf[1], (unsigned long long)buf[2],
         (unsigned long long)buf[3]);
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(void) {
  printf("----UTILS TESTS----\n\n");

  test_platform_seed();
  test_splitmix64_basic();

  printf("▶ All tests passed!\n\n");
  return 0;
}
