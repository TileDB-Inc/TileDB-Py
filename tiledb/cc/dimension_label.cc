#include <tiledb/tiledb>
#include <tiledb/tiledb_experimental>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "common.h"

namespace libtiledbcpp {

using namespace tiledb;
using namespace tiledbpy::common;
namespace py = pybind11;

#if TILEDB_VERSION_MAJOR == 2 && TILEDB_VERSION_MINOR < 15

struct tiledb_dimension_label_t;

class DimensionLabel {
public:
  DimensionLabel(const Context &, tiledb_dimension_label_t *) {
    throw TileDBError(
        "Using dimension labels requires libtiledb version 2.15.0 or greater");
  }

  DimensionLabel() {
    throw TileDBError(
        "Using dimension labels requires libtiledb version 2.15.0 or greater");
  }

  uint32_t dimension_index() const {
    throw TileDBError(
        "Using dimension labels requires libtiledb version 2.15.0 or greater");
  }

  std::string label_attr_name() const {
    throw TileDBError(
        "Using dimension labels requires libtiledb version 2.15.0 or greater");
  }

  uint32_t label_cell_val_num() const {
    throw TileDBError(
        "Using dimension labels requires libtiledb version 2.15.0 or greater");
  }

  tiledb_data_order_t label_order() const {
    throw TileDBError(
        "Using dimension labels requires libtiledb version 2.15.0 or greater");
  }

  tiledb_datatype_t label_type() const {
    throw TileDBError(
        "Using dimension labels requires libtiledb version 2.15.0 or greater");
  }

  std::string name() const {
    throw TileDBError(
        "Using dimension labels requires libtiledb version 2.15.0 or greater");
  }

  std::shared_ptr<tiledb_dimension_label_t> ptr() const {
    throw TileDBError(
        "Using dimension labels requires libtiledb version 2.15.0 or greater");
  }

  std::string uri() const {
    throw TileDBError(
        "Using dimension labels requires libtiledb version 2.15.0 or greater");
  }
};
#endif

void init_dimension_label(py::module &m) {
  py::class_<DimensionLabel>(m, "DimensionLabel")
      .def(py::init<DimensionLabel>())

      .def(py::init<const Context &, py::capsule>())

      .def("__capsule__",
           [](DimensionLabel &dim_label) {
             return py::capsule(dim_label.ptr().get(), "dim_label", nullptr);
           })

      .def_property_readonly("_label_attr_name",
                             &DimensionLabel::label_attr_name)

      .def_property_readonly("_dim_index", &DimensionLabel::dimension_index)

      .def_property_readonly("_tiledb_label_order",
                             &DimensionLabel::label_order)

      .def_property_readonly("_tiledb_label_dtype", &DimensionLabel::label_type)

      .def_property_readonly("_label_ncell",
                             &DimensionLabel::label_cell_val_num)

      .def_property_readonly("_name", &DimensionLabel::name)

      .def_property_readonly("_uri", &DimensionLabel::uri);
}

}; // namespace libtiledbcpp
