#include "../include/sphur.h"
#include <assert.h>
#include <stdio.h>

int main(void) {
  assert(add_one(0) == 1);
  assert(add_one(41) == 42);

  printf("All tests passed!\n");
  return 0;
}
