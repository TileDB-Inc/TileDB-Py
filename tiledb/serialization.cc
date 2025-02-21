
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// #include <pybind11/pytypes.h>
// #include <pybind11/stl.h>

#include <exception>

#include <tiledb/tiledb_serialization.h>  // C
#include <tiledb/tiledb>                  // C++
#include "util.h"

#if !defined(NDEBUG)
// #include "debug.cc"
#endif

namespace tiledbpy {

using namespace std;
using namespace tiledb;
namespace nb = nanobind;
using namespace nb::literals;

class PySerialization {
   public:
    static void* deserialize_query(
        nb::object ctx,
        nb::object array,
        nb::buffer buffer,
        tiledb_serialization_type_t serialize_type,
        int32_t client_side) {
        int rc;

        tiledb_ctx_t* ctx_c;
        tiledb_array_t* arr_c;
        tiledb_query_t* qry_c;
        tiledb_buffer_t* buf_c;

        // ctx_c = (nb::capsule)ctx.attr("__capsule__")();
        ctx_c = nb::cast<tiledb_ctx_t*>(ctx.attr("__capsule__")());

        if (ctx_c == nullptr)
            TPY_ERROR_LOC("Invalid context pointer.");

        // arr_c = (nb::capsule)array.attr("__capsule__")();
        arr_c = nb::cast<tiledb_array_t*>(array.attr("__capsule__")());

        if (arr_c == nullptr)
            TPY_ERROR_LOC("Invalid array pointer.");

        rc = tiledb_query_alloc(ctx_c, arr_c, TILEDB_READ, &qry_c);
        if (rc == TILEDB_ERR)
            TPY_ERROR_LOC("Could not allocate query.");

        rc = tiledb_buffer_alloc(ctx_c, &buf_c);
        if (rc == TILEDB_ERR)
            TPY_ERROR_LOC("Could not allocate buffer.");

        nb::buffer_info buf_info = buffer.request();
        rc = tiledb_buffer_set_data(
            ctx_c, buf_c, buf_info.ptr, buf_info.shape[0]);
        if (rc == TILEDB_ERR)
            TPY_ERROR_LOC("Could not set buffer.");

        rc = tiledb_deserialize_query(
            ctx_c, buf_c, serialize_type, client_side, qry_c);
        if (rc == TILEDB_ERR)
            TPY_ERROR_LOC("Could not deserialize query.");

        return qry_c;
    }
};

void init_serialization(nb::module_& m) {
    nb::class_<PySerialization>(m, "serialization")
        .def_static("deserialize_query", &PySerialization::deserialize_query);

    nb::enum_<tiledb_serialization_type_t>(
        m, "tiledb_serialization_type_t", nb::arithmetic())
        .value("TILEDB_CAPNP", TILEDB_CAPNP)
        .value("TILEDB_JSON", TILEDB_JSON)
        .export_values();
}

};  // namespace tiledbpy
