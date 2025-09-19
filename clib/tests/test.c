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

  printf("test_init_default: PASS\n");
}

void test_init_seeded(void) {
  sphur_t ctx;
  int rc = sphur_init_seeded(&ctx, 123456789ULL);

  assert(rc == 0);
  assert(ctx._rbuf == NULL);
  assert(ctx._rcnt == 0);
  assert(ctx._rpos == 0);

  printf("test_init_seeded: PASS\n");
}

void test_rand(void) {
  sphur_t ctx;
  sphur_init_seeded(&ctx, 42);

  uint64_t first = sphur_gen_rand(&ctx);
  uint64_t second = sphur_gen_rand(&ctx);

  assert(first != 0);
  assert(second != 0);

  // Consume remaining 6
  for (int i = 0; i < 6; i++) {
    (void)sphur_gen_rand(&ctx);
  }

  // 9th call should wrap around
  uint64_t again = sphur_gen_rand(&ctx);
  assert(again == first);

  printf("test_rand: PASS\n");
}

void test_null(void) {
  assert(sphur_init(NULL) == 1);
  assert(sphur_init_seeded(NULL, 123) == 1);
  assert(sphur_gen_rand(NULL) == 1);

  uint64_t out;
  assert(_sphur_rands_peek(NULL, &out) == 1);

  printf("test_null: PASS\n");
}

int main(void) {
  test_init();
  test_init_seeded();
  test_rand();
  test_null();

  return 0;
}
