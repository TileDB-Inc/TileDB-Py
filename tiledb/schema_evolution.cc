#include <pybind11/pybind11.h>
#include <tiledb/tiledb.h>
#include <tiledb/tiledb_experimental.h>

#include "util.h"

namespace tiledbpy {

// using namespace tiledb;
namespace nb = nanobind;

typedef struct {
    tiledb_ctx_t* ctx_;
    tiledb_array_schema_evolution_t* evol_;
} PyArraySchemaEvolution;

using ArraySchemaEvolution = PyArraySchemaEvolution;

void init_schema_evolution(nb::module& m) {
    nb::class_<ArraySchemaEvolution>(m, "ArraySchemaEvolution")
        .def(
            "__init__",
            [](PyArraySchemaEvolution* self, nb::object ctx_py) {
                tiledb_ctx_t* ctx_c = (nb::capsule)ctx_py.attr("__capsule__")();
                if (ctx_c == nullptr)
                    TPY_ERROR_LOC("Invalid context pointer");

                tiledb_array_schema_evolution_t* evol_p;
                int rc = tiledb_array_schema_evolution_alloc(ctx_c, &evol_p);
                if (rc != TILEDB_OK) {
                    TPY_ERROR_LOC(get_last_ctx_err_str(ctx_c, rc));
                }

                new (self) PyArraySchemaEvolution({ctx_c, evol_p});
            })
        .def(
            "add_attribute",
            [](ArraySchemaEvolution& inst, nb::object attr_py) {
                tiledb_attribute_t* attr_c = (nb::capsule)attr_py.attr(
                    "__capsule__")();
                if (attr_c == nullptr)
                    TPY_ERROR_LOC("Invalid Attribute!");

                int rc = tiledb_array_schema_evolution_add_attribute(
                    inst.ctx_, inst.evol_, attr_c);
                if (rc != TILEDB_OK) {
                    TPY_ERROR_LOC(get_last_ctx_err_str(inst.ctx_, rc));
                }
            })
        .def(
            "drop_attribute",
            [](ArraySchemaEvolution& inst, std::string attr_name) {
                int rc = tiledb_array_schema_evolution_drop_attribute(
                    inst.ctx_, inst.evol_, attr_name.c_str());
                if (rc != TILEDB_OK) {
                    TPY_ERROR_LOC(get_last_ctx_err_str(inst.ctx_, rc));
                }
            })
        .def(
            "array_evolve",
            [](ArraySchemaEvolution& inst, std::string uri) {
                int rc = tiledb_array_evolve(
                    inst.ctx_, uri.c_str(), inst.evol_);
                if (rc != TILEDB_OK) {
                    TPY_ERROR_LOC(get_last_ctx_err_str(inst.ctx_, rc));
                }
            })
        .def(
            "set_timestamp_range",
            [](ArraySchemaEvolution& inst, uint64_t timestamp) {
                int rc = tiledb_array_schema_evolution_set_timestamp_range(
                    inst.ctx_, inst.evol_, timestamp, timestamp);
                if (rc != TILEDB_OK) {
                    TPY_ERROR_LOC(get_last_ctx_err_str(inst.ctx_, rc));
                }
            })
        .def(
            "add_enumeration",
            [](ArraySchemaEvolution& inst, nb::object enum_py) {
                tiledb_enumeration_t* enum_c = (nb::capsule)enum_py.attr(
                    "__capsule__")();
                if (enum_c == nullptr)
                    TPY_ERROR_LOC("Invalid Enumeration!");
                int rc = tiledb_array_schema_evolution_add_enumeration(
                    inst.ctx_, inst.evol_, enum_c);
                if (rc != TILEDB_OK) {
                    TPY_ERROR_LOC(get_last_ctx_err_str(inst.ctx_, rc));
                }
            })
        .def(
            "drop_enumeration",
            [](ArraySchemaEvolution& inst,
               const std::string& enumeration_name) {
                int rc = tiledb_array_schema_evolution_drop_enumeration(
                    inst.ctx_, inst.evol_, enumeration_name.c_str());
                if (rc != TILEDB_OK) {
                    TPY_ERROR_LOC(get_last_ctx_err_str(inst.ctx_, rc));
                }
            })
        .def(
            "extend_enumeration",
            [](ArraySchemaEvolution& inst, nb::object enum_py) {
                tiledb_enumeration_t* enum_c = (nb::capsule)enum_py.attr(
                    "__capsule__")();
                if (enum_c == nullptr)
                    TPY_ERROR_LOC("Invalid Enumeration!");
                int rc = tiledb_array_schema_evolution_extend_enumeration(
                    inst.ctx_, inst.evol_, enum_c);
                if (rc != TILEDB_OK) {
                    TPY_ERROR_LOC(get_last_ctx_err_str(inst.ctx_, rc));
                }
            })

#if TILEDB_VERSION_MAJOR >= 2 && TILEDB_VERSION_MINOR >= 26
        .def(
            "expand_current_domain",
            [](ArraySchemaEvolution& inst, nb::object current_domain_py) {
                tiledb_current_domain_t*
                    current_domain_c = (nb::capsule)current_domain_py.attr(
                        "__capsule__")();
                if (current_domain_c == nullptr)
                    TPY_ERROR_LOC("Invalid Current Domain!");
                int rc = tiledb_array_schema_evolution_expand_current_domain(
                    inst.ctx_, inst.evol_, current_domain_c);
                if (rc != TILEDB_OK) {
                    TPY_ERROR_LOC(get_last_ctx_err_str(inst.ctx_, rc));
                }
            })
#endif
        ;
}

};  // namespace tiledbpy
