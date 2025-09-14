#include <assert.h>
#include "sphur.h"

int main(void) {
  assert(add_one(41) == 42);
  assert(add_one(-1) == 0);
  assert(add_one(0) == 1);

  return 0;
}
