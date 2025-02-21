#include <tiledb/tiledb>
#include <tiledb/tiledb_experimental>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// #include <pybind11/pytypes.h>
// #include <pybind11/stl.h>

#include "common.h"

namespace libtiledbcpp {

using namespace tiledb;
using namespace tiledbnb::common;
namespace nb = nanobind;

void init_dimension_label(nb::module_& m) {
    nb::class_<DimensionLabel>(m, "DimensionLabel")
        .def(nb::init<DimensionLabel>())

        .def(nb::init<const Context&, nb::capsule>())

        .def(
            "__capsule__",
            [](DimensionLabel& dim_label) {
                return nb::capsule(dim_label.ptr().get(), "dim_label");
            })

        .def_prop_rw_readonly(
            "_label_attr_name", &DimensionLabel::label_attr_name)

        .def_prop_rw_readonly("_dim_index", &DimensionLabel::dimension_index)

        .def_prop_rw_readonly(
            "_tiledb_label_order", &DimensionLabel::label_order)

        .def_prop_rw_readonly(
            "_tiledb_label_dtype", &DimensionLabel::label_type)

        .def_prop_rw_readonly(
            "_label_ncell", &DimensionLabel::label_cell_val_num)

        .def_prop_rw_readonly("_name", &DimensionLabel::name)

        .def_prop_rw_readonly("_uri", &DimensionLabel::uri);
}

};  // namespace libtiledbcpp
