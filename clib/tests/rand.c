#include "../include/sphur.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>

static void test_gen_rand_basic(void) {
  sphur_t ctx;
  int rc = sphur_init(&ctx);
  uint64_t prev = sphur_gen_rand(&ctx);

  assert(rc == 0);

  for (int i = 0; i < 100; i++) {
    uint64_t next = sphur_gen_rand(&ctx);

    // basic sanity check (should produce non-repeating in small batch)
    assert(next != prev || ctx._rcnt == 0);

    prev = next;
  }

  printf("test_gen_rand_basic: PASS\n");
}

static void test_gen_rand_range(void) {
  sphur_t ctx;
  int rc = sphur_init(&ctx);
  uint64_t min = 10, max = 20;

  assert(rc == 0);

  for (int i = 0; i < 1000; i++) {
    uint64_t r = sphur_gen_rand_range(&ctx, min, max);

    assert(r >= min && r <= max);
  }

  // test edge case: min == max
  uint64_t val = sphur_gen_rand_range(&ctx, 42, 42);
  assert(val == 42);

  // test invalid range: min > max
  val = sphur_gen_rand_range(&ctx, 50, 40);
  assert(val == 0);

  printf("test_gen_rand_range: PASS\n");
}

static void test_gen_bool(void) {
  sphur_t ctx;
  int rc = sphur_init(&ctx);
  int ones = 0, zeros = 0;

  assert(rc == 0);

  for (int i = 0; i < 1000; i++) {
    int b = sphur_gen_bool(&ctx);
    assert(b == 0 || b == 1);

    if (b == 0)
      zeros++;
    else
      ones++;
  }

  // simple check: both 0 and 1 appeared
  assert(zeros > 0 && ones > 0);
  printf("test_gen_bool: PASS\n");
}

int main(void) {
  printf("----GEN RAND TESTS----\n\n");

  test_gen_rand_basic();
  test_gen_rand_range();
  test_gen_bool();

  printf("▶ All gen_rand tests passed!\n\n");
  return 0;
}
