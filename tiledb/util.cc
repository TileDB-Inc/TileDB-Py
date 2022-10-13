#include "util.h"

#include <tiledb/tiledb>

std::string _get_tiledb_err_str(tiledb_error_t *err_ptr) {
  const char *err_msg_ptr = NULL;
  int ret = tiledb_error_message(err_ptr, &err_msg_ptr);

  if (ret != TILEDB_OK) {
    tiledb_error_free(&err_ptr);
    if (ret == TILEDB_OOM) {
      throw std::bad_alloc();
    }
    return "error retrieving error message";
  }
  return std::string(err_msg_ptr);
}

std::string get_last_ctx_err_str(tiledb_ctx_t *ctx_ptr, int rc) {
  if (rc == TILEDB_OOM)
    throw std::bad_alloc();

  tiledb_error_t *err_ptr = NULL;
  int ret = tiledb_ctx_get_last_error(ctx_ptr, &err_ptr);

  if (ret != TILEDB_OK) {
    tiledb_error_free(&err_ptr);
    if (ret == TILEDB_OOM) {
      throw std::bad_alloc();
    }
    return "error retrieving error object from ctx";
  }
  return _get_tiledb_err_str(err_ptr);
}