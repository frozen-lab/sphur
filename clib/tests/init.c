#include "../include/sphur.h"
#include <assert.h>
#include <stdio.h>

void test_init(void) {
  sphur_t ctx;
  int rc = sphur_init(&ctx);

  assert(rc == 0);
  assert(ctx._rbuf == NULL);
  assert(ctx._rcnt == 0);
  assert(ctx._rpos == 0);
  assert(ctx._simd_ext == 0 || ctx._simd_ext == 1);

  printf("test_init_default: PASS\n");
}

void test_init_seeded(void) {
  sphur_t ctx;
  int rc = sphur_init_seeded(&ctx, 123456789ULL);

  assert(rc == 0);
  assert(ctx._rbuf == NULL);
  assert(ctx._rcnt == 0);
  assert(ctx._rpos == 0);
  assert(ctx._simd_ext == 0 || ctx._simd_ext == 1);

  printf("test_init_seeded: PASS\n");
}

void test_sanity_check(void) {
  assert(sphur_init(NULL) == -1);
  assert(sphur_init_seeded(NULL, 123) == -1);

  printf("test_null: PASS\n");
}

int main(void) {
  printf("----INIT TESTS----\n\n");

  test_init();
  test_init_seeded();
  test_sanity_check();

  printf("▶ All tests passed!\n\n");
  return 0;
}
