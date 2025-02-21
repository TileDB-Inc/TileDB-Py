
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// #include <pybind11/pytypes.h>
// #include <pybind11/stl.h>

#include <exception>

#define TILEDB_DEPRECATED
#define TILEDB_DEPRECATED_EXPORT

#include <tiledb/tiledb_serialization.h>  // C
#include <tiledb/tiledb>                  // C++
#include "../util.h"

#if !defined(NDEBUG)
// #include "debug.cc"
#endif

namespace tiledbpy {

using namespace std;
using namespace tiledb;
namespace nb = nanobind;
using namespace nb::literals;

class PySerializationTest {
   public:
    static nb::bytes create_serialized_test_query(
        nb::object pyctx, nb::object pyarray) {
        int rc;

        tiledb_ctx_t* ctx;
        tiledb_array_t* array;

        // ctx = (nb::capsule)pyctx.attr("__capsule__")();
        ctx = nb::cast<tiledb_ctx_t*>(pyctx.attr("__capsule__")());

        if (ctx == nullptr)
            TPY_ERROR_LOC("Invalid context pointer.");

        tiledb_ctx_alloc(NULL, &ctx);
        // array = (nb::capsule)pyarray.attr("__capsule__")();
        array = nb::cast<tiledb_array_t*>(pyarray.attr("__capsule__")());

        if (array == nullptr)
            TPY_ERROR_LOC("Invalid array pointer.");

        uint32_t subarray_v[] = {3, 7};
        int64_t data[5];
        uint64_t data_size = sizeof(data);

        tiledb_subarray_t* subarray;
        tiledb_subarray_alloc(ctx, array, &subarray);
        tiledb_subarray_set_subarray(ctx, subarray, &subarray_v);

        tiledb_query_t* query;
        tiledb_query_alloc(ctx, array, TILEDB_READ, &query);
        tiledb_query_set_subarray_t(ctx, query, subarray);
        tiledb_query_set_layout(ctx, query, TILEDB_UNORDERED);
        tiledb_query_set_data_buffer(ctx, query, "", data, &data_size);

        tiledb_buffer_list_t* buff_list;
        tiledb_buffer_t* buff;

        rc = tiledb_serialize_query(ctx, query, TILEDB_CAPNP, 1, &buff_list);
        if (rc == TILEDB_ERR)
            TPY_ERROR_LOC("Could not serialize the query.");

        rc = tiledb_buffer_list_flatten(ctx, buff_list, &buff);
        if (rc == TILEDB_ERR)
            TPY_ERROR_LOC("Could not flatten the buffer list.");

        void* buff_data;
        uint64_t buff_num_bytes;

        rc = tiledb_buffer_get_data(ctx, buff, &buff_data, &buff_num_bytes);
        if (rc == TILEDB_ERR)
            TPY_ERROR_LOC("Could not get the data from the buffer.");

        nb::bytes output((char*)buff_data, buff_num_bytes);

        tiledb_buffer_free(&buff);
        tiledb_buffer_list_free(&buff_list);
        tiledb_subarray_free(&subarray);
        tiledb_query_free(&query);

        return output;
    }
};

void init_test_serialization(nb::module_& m) {
    nb::class_<PySerializationTest>(m, "test_serialization")
        .def_static(
            "create_serialized_test_query",
            &PySerializationTest::create_serialized_test_query);
}

};  // namespace tiledbpy
