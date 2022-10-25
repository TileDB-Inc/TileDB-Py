// #include <tiledb/tiledb>
#include <tiledb/tiledb_dimension_label.h>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "common.h"

namespace libtiledbcpp {

using namespace tiledb;
using namespace tiledbpy::common;
namespace py = pybind11;

class DimensionLabel {
public:
  DimensionLabel(const Context &ctx, tiledb_label_order_t label_order,
                 tiledb_datatype_t label_type, tiledb_datatype_t index_type,
                 py::array domain, py::array tile_extent) {
    tiledb_ctx_t *c_ctx = ctx.ptr().get();

    py::buffer domain_buffer = py::buffer(domain);
    py::buffer_info domain_info = domain_buffer.request();

    py::buffer tile_buffer = py::buffer(tile_extent);
    py::buffer_info tile_extent_info = tile_buffer.request();

    ctx.handle_error(tiledb_dimension_label_schema_alloc(
        c_ctx, label_order, label_type, index_type, domain_info.ptr,
        tile_extent_info.ptr, &dimension_label_));
  }

  ~DimensionLabel() { tiledb_dimension_label_schema_free(&dimension_label_); }

  tiledb_dimension_label_schema_t *ptr() { return dimension_label_; }

private:
  tiledb_dimension_label_schema_t *dimension_label_;
};

void init_dimension_label(py::module &m) {
  py::class_<DimensionLabel>(m, "DimensionLabel")
      .def(py::init<const Context &, tiledb_label_order_t, tiledb_datatype_t,
                    tiledb_datatype_t, py::array, py::array>())
      .def("__capsule__", [](DimensionLabel &dl) {
        return py::capsule(dl.ptr(), "dl", nullptr);
      });
}

}; // namespace libtiledbcpp