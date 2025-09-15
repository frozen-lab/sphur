#include "sphur.h"
#include <node_api.h>

napi_value AddOne(napi_env env, napi_callback_info info) {
  size_t argc = 1;
  napi_value args[1];

  napi_get_cb_info(env, info, &argc, args, NULL, NULL);

  int32_t input;
  napi_get_value_int32(env, args[0], &input);

  int32_t result = asm_add_one(input);

  napi_value out;
  napi_create_int32(env, result, &out);

  return out;
}

napi_value Init(napi_env env, napi_value exports) {
  napi_value fn;

  napi_create_function(env, NULL, 0, AddOne, NULL, &fn);
  napi_set_named_property(env, exports, "add_one", fn);

  return exports;
}

NAPI_MODULE(NODE_GYP_MODULE_NAME, Init)
