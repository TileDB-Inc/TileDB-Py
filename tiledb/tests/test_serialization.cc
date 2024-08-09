
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <exception>

#define TILEDB_DEPRECATED
#define TILEDB_DEPRECATED_EXPORT

#include "../util.h"
#include <tiledb/tiledb>                 // C++
#include <tiledb/tiledb_serialization.h> // C

#if !defined(NDEBUG)
// #include "debug.cc"
#endif

namespace tiledbpy {

using namespace std;
using namespace tiledb;
namespace py = pybind11;
using namespace pybind11::literals;

class PySerializationTest {

public:
  static py::bytes create_serialized_test_query(py::object pyctx,
                                                py::object pyarray) {
    int rc;

    tiledb_ctx_t *ctx;
    tiledb_array_t *array;

    ctx = (py::capsule)pyctx.attr("__capsule__")();
    if (ctx == nullptr)
      TPY_ERROR_LOC("Invalid context pointer.");

    tiledb_ctx_alloc(NULL, &ctx);
    array = (py::capsule)pyarray.attr("__capsule__")();
    if (array == nullptr)
      TPY_ERROR_LOC("Invalid array pointer.");

    uint32_t subarray_v[] = {3, 7};
    int64_t data[5];
    uint64_t data_size = sizeof(data);

    tiledb_subarray_t *subarray;
    tiledb_subarray_alloc(ctx, array, &subarray);
    tiledb_subarray_set_subarray(ctx, subarray, &subarray_v);

    tiledb_query_t *query;
    tiledb_query_alloc(ctx, array, TILEDB_READ, &query);
    tiledb_query_set_subarray_t(ctx, query, subarray);
    tiledb_query_set_layout(ctx, query, TILEDB_UNORDERED);
    tiledb_query_set_data_buffer(ctx, query, "", data, &data_size);

    tiledb_buffer_list_t *buff_list;
    tiledb_buffer_t *buff;

    rc = tiledb_serialize_query(ctx, query, TILEDB_CAPNP, 1, &buff_list);
    if (rc == TILEDB_ERR)
      TPY_ERROR_LOC("Could not serialize the query.");

    rc = tiledb_buffer_list_flatten(ctx, buff_list, &buff);
    if (rc == TILEDB_ERR)
      TPY_ERROR_LOC("Could not flatten the buffer list.");

    void *buff_data;
    uint64_t buff_num_bytes;

    rc = tiledb_buffer_get_data(ctx, buff, &buff_data, &buff_num_bytes);
    if (rc == TILEDB_ERR)
      TPY_ERROR_LOC("Could not get the data from the buffer.");

    py::bytes output((char *)buff_data, buff_num_bytes);

    tiledb_buffer_free(&buff);
    tiledb_buffer_list_free(&buff_list);
    tiledb_subarray_free(&subarray);
    tiledb_query_free(&query);

    return output;
  }
};

void init_test_serialization(py::module &m) {
  py::class_<PySerializationTest>(m, "test_serialization")
      .def_static("create_serialized_test_query",
                  &PySerializationTest::create_serialized_test_query);
}

}; // namespace tiledbpy
