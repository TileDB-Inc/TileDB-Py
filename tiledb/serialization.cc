
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <exception>

#include "util.h"
#include <tiledb/tiledb>                 // C++
#include <tiledb/tiledb_serialization.h> // C

#if TILEDB_VERSION_MAJOR == 2 && TILEDB_VERSION_MINOR >= 2

#if !defined(NDEBUG)
// #include "debug.cc"
#endif

namespace tiledbpy {

using namespace std;
using namespace tiledb;
namespace py = pybind11;
using namespace pybind11::literals;

class PySerialization {

public:
  static void *deserialize_query(py::object ctx, py::object array,
                                 py::buffer buffer,
                                 tiledb_serialization_type_t serialize_type,
                                 int32_t client_side) {
    int rc;

    tiledb_ctx_t *ctx_c;
    tiledb_array_t *arr_c;
    tiledb_query_t *qry_c;
    tiledb_buffer_t *buf_c;

    ctx_c = (py::capsule)ctx.attr("__capsule__")();
    if (ctx_c == nullptr)
      TPY_ERROR_LOC("Invalid context pointer.");

    arr_c = (py::capsule)array.attr("__capsule__")();
    if (arr_c == nullptr)
      TPY_ERROR_LOC("Invalid array pointer.");

    rc = tiledb_query_alloc(ctx_c, arr_c, TILEDB_READ, &qry_c);
    if (rc == TILEDB_ERR)
      TPY_ERROR_LOC("Could not allocate query.");

    rc = tiledb_buffer_alloc(ctx_c, &buf_c);
    if (rc == TILEDB_ERR)
      TPY_ERROR_LOC("Could not allocate buffer.");

    py::buffer_info buf_info = buffer.request();
    rc = tiledb_buffer_set_data(ctx_c, buf_c, buf_info.ptr, buf_info.shape[0]);
    if (rc == TILEDB_ERR)
      TPY_ERROR_LOC("Could not set buffer.");

    rc = tiledb_deserialize_query(ctx_c, buf_c, serialize_type, client_side,
                                  qry_c);
    if (rc == TILEDB_ERR)
      TPY_ERROR_LOC("Could not deserialize query.");

    return qry_c;
  }
};

void init_serialization(py::module &m) {
  py::class_<PySerialization>(m, "serialization")
      .def_static("deserialize_query", &PySerialization::deserialize_query);

  py::enum_<tiledb_serialization_type_t>(m, "tiledb_serialization_type_t",
                                         py::arithmetic())
      .value("TILEDB_CAPNP", TILEDB_CAPNP)
      .value("TILEDB_JSON", TILEDB_JSON)
      .export_values();
}

}; // namespace tiledbpy

#endif
