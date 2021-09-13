#include <pybind11/pybind11.h>
//#include <tiledb/tiledb_experimental>
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

void init_schema_evolution(py::module &m) {
  py::class_<ArraySchemaEvolution>(m, "ArraySchemaEvolution")
      .def(py::init([](py::object ctx_py) {
        tiledb_ctx_t *ctx_c = (py::capsule)ctx_py.attr("__capsule__")();
        if (ctx_c == nullptr)
          TPY_ERROR_LOC("Invalid context pointer");

        tiledb_array_schema_evolution_t *evol_p;
        int rc = tiledb_array_schema_evolution_alloc(ctx_c, &evol_p);
        if (rc != TILEDB_OK) {
          TPY_ERROR_LOC("Failed to allocate ArraySchemaEvolution");
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
               TPY_ERROR_LOC("Failed to add attribute to ArraySchemaEvolution");
             }
           })
      .def("drop_attribute",
           [](ArraySchemaEvolution &inst, std::string attr_name) {
             int rc = tiledb_array_schema_evolution_drop_attribute(
                 inst.ctx_, inst.evol_, attr_name.c_str());
             if (rc != TILEDB_OK) {
               TPY_ERROR_LOC(
                   "Failed to drop attribute from ArraySchemaEvolution");
             }
           })
      .def("array_evolve", [](ArraySchemaEvolution &inst, std::string uri) {
        int rc = tiledb_array_evolve(inst.ctx_, uri.c_str(), inst.evol_);
        if (rc != TILEDB_OK) {
          TPY_ERROR_LOC("Failed to drop attribute from ArraySchemaEvolution");
        }
      });
}

}; // namespace tiledbpy
