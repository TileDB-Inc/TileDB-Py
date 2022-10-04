#include <pybind11/pybind11.h>
#include <tiledb/tiledb.h>
#include <tiledb/tiledb_experimental.h>

#include "util.h"

namespace tiledbpy {

// using namespace tiledb;
namespace py = pybind11;

typedef struct {
  tiledb_ctx_t *ctx_;
  tiledb_array_schema_evolution_t *evol_;
} PyArraySchemaEvolution;

using ArraySchemaEvolution = PyArraySchemaEvolution;

void _throw_tiledb_error(tiledb_error_t *err_ptr) {
  const char *err_msg_ptr = NULL;
  int ret = tiledb_error_message(err_ptr, &err_msg_ptr);
  if (ret != TILEDB_OK) {
    tiledb_error_free(&err_ptr);
    if (ret == TILEDB_OOM) {
      throw std::bad_alloc();
    }
    TPY_ERROR_LOC("error retrieving error message");
  }

  TPY_ERROR_LOC(std::string(err_msg_ptr));
}

void _throw_ctx_err(tiledb_ctx_t *ctx_ptr, int rc) {
  if (rc == TILEDB_OK)
    return;
  if (rc == TILEDB_OOM)
    throw std::bad_alloc();

  tiledb_error_t *err_ptr = NULL;
  int ret = tiledb_ctx_get_last_error(ctx_ptr, &err_ptr);
  if (ret != TILEDB_OK) {
    tiledb_error_free(&err_ptr);
    if (ret == TILEDB_OOM)
      throw std::bad_alloc();
    TPY_ERROR_LOC("error retrieving error object from ctx");
  }
  _throw_tiledb_error(err_ptr);
}

void init_schema_evolution(py::module &m) {
  py::class_<ArraySchemaEvolution>(m, "ArraySchemaEvolution")
      .def(py::init([](py::object ctx_py) {
        tiledb_ctx_t *ctx_c = (py::capsule)ctx_py.attr("__capsule__")();
        if (ctx_c == nullptr)
          TPY_ERROR_LOC("Invalid context pointer");

        tiledb_array_schema_evolution_t *evol_p;
        int rc = tiledb_array_schema_evolution_alloc(ctx_c, &evol_p);
        if (rc != TILEDB_OK) {
          _throw_ctx_err(ctx_c, rc);
        }

        return new PyArraySchemaEvolution({ctx_c, evol_p});
      }))
      .def("add_attribute",
           [](ArraySchemaEvolution &inst, py::object attr_py) {
             tiledb_attribute_t *attr_c =
                 (py::capsule)attr_py.attr("__capsule__")();
             if (attr_c == nullptr)
               TPY_ERROR_LOC("Invalid Attribute!");

             int rc = tiledb_array_schema_evolution_add_attribute(
                 inst.ctx_, inst.evol_, attr_c);
             if (rc != TILEDB_OK) {
               _throw_ctx_err(inst.ctx_, rc);
             }
           })
      .def("drop_attribute",
           [](ArraySchemaEvolution &inst, std::string attr_name) {
             int rc = tiledb_array_schema_evolution_drop_attribute(
                 inst.ctx_, inst.evol_, attr_name.c_str());
             if (rc != TILEDB_OK) {
               _throw_ctx_err(inst.ctx_, rc);
             }
           })
      .def("array_evolve", [](ArraySchemaEvolution &inst, std::string uri) {
        int rc = tiledb_array_evolve(inst.ctx_, uri.c_str(), inst.evol_);
        if (rc != TILEDB_OK) {
          _throw_ctx_err(inst.ctx_, rc);
        }
      });
}

}; // namespace tiledbpy
