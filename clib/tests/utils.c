#include "../include/sphur.h"
#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

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
// Tests for `rand_buffer`
// -----------------------------------------------------------------------------

static void test_rands_empty_and_mark(void) {
  sphur_t state;
  memset(&state, 0, sizeof(state));

  assert(_sphur_rands_empty(&state) == 1);

  // fill with count=2
  _sphur_rands_mark_filled(&state, 2);

  assert(_sphur_rands_empty(&state) == 0);
  assert(state._rcnt == 2);
  assert(state._rpos == 0);
  assert(state._rbuf == state._rands);

  // invalid count clears buffer
  _sphur_rands_mark_filled(&state, 0);
  assert(_sphur_rands_empty(&state) == 1);

  _sphur_rands_mark_filled(&state, 5); // too big
  assert(_sphur_rands_empty(&state) == 1);

  printf("rands_empty_and_mark passed\n");
}

static void test_rands_consume_and_peek(void) {
  sphur_t state;
  memset(&state, 0, sizeof(state));

  // preload buffer
  state._rands[0] = 111;
  state._rands[1] = 222;
  _sphur_rands_mark_filled(&state, 2);

  uint64_t val;
  assert(_sphur_rands_peek(&state, &val) == 0);
  assert(val == 111);

  // consume first
  _sphur_rands_consume_one(&state);
  assert(state._rpos == 1);
  assert(state._rcnt == 2);

  // peek second
  assert(_sphur_rands_peek(&state, &val) == 0);
  assert(val == 222);

  // consume second
  _sphur_rands_consume_one(&state);
  assert(_sphur_rands_empty(&state));

  // peek should fail when empty
  assert(_sphur_rands_peek(&state, &val) == -1);

  printf("rands_consume_and_peek passed\n");
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------

int main(void) {
  printf("----UTILS TESTS----\n\n");

  test_platform_seed();
  test_splitmix64_basic();
  test_rands_empty_and_mark();
  test_rands_consume_and_peek();

  printf("▶ All tests passed!\n\n");
  return 0;
}
