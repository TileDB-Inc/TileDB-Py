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

class Filestore {
   public:
    // TODO this works, but isn't actually in use at the moment.
    // we are still using tiledb.libtiledb.ArraySchema. when we switch to using
    // tiledb.libtiledb.ArraySchema, use this function instead.
    static ArraySchema schema_create(const Context& ctx, const char* uri) {
        tiledb_array_schema_t* schema;
        tiledb_filestore_schema_create(ctx.ptr().get(), uri, &schema);
        return ArraySchema(ctx, nb::capsule(schema));
    }

    static void uri_import(
        const Context& ctx,
        const char* filestore_array_uri,
        const char* file_uri,
        tiledb_mime_type_t mime_type) {
        ctx.handle_error(tiledb_filestore_uri_import(
            ctx.ptr().get(), filestore_array_uri, file_uri, mime_type));
    }

    static void uri_export(
        const Context& ctx,
        const char* filestore_array_uri,
        const char* file_uri) {
        ctx.handle_error(tiledb_filestore_uri_export(
            ctx.ptr().get(), file_uri, filestore_array_uri));
    }

    static void buffer_import(
        const Context& ctx,
        const char* filestore_array_uri,
        nb::buffer buf,
        tiledb_mime_type_t mime_type) {
        nb::buffer_info buffer = buf.request();
        ctx.handle_error(tiledb_filestore_buffer_import(
            ctx.ptr().get(),
            filestore_array_uri,
            buffer.ptr,
            nb::len(buf),
            mime_type));
    }

    static nb::bytes buffer_export(
        const Context& ctx,
        const char* filestore_array_uri,
        size_t offset,
        size_t size) {
        nb::ndarray data = nb::ndarray(nb::dtype<std::byte>(), size);
        nb::buffer_info buffer = data.request();

        ctx.handle_error(tiledb_filestore_buffer_export(
            ctx.ptr().get(), filestore_array_uri, offset, buffer.ptr, size));

        auto np = nb::module_::import_("numpy");
        auto to_bytes = np.attr("ndarray").attr("tobytes");

        return to_bytes(data);
    }

    static size_t size(const Context& ctx, const char* filestore_array_uri) {
        size_t size;
        ctx.handle_error(
            tiledb_filestore_size(ctx.ptr().get(), filestore_array_uri, &size));
        return size;
    }

    static const char* mime_type_to_str(tiledb_mime_type_t mime_type) {
        const char* str;
        tiledb_mime_type_to_str(mime_type, &str);
        return str;
    }

    static tiledb_mime_type_t mime_type_from_str(const char* str) {
        tiledb_mime_type_t mime_type;
        tiledb_mime_type_from_str(str, &mime_type);
        return mime_type;
    }
};

void init_filestore(nb::module_& m) {
    nb::class_<Filestore>(m, "Filestore")
        .def_static(
            "_schema_create", &Filestore::schema_create, nb::keep_alive<1, 2>())
        .def_static(
            "_uri_import", &Filestore::uri_import, nb::keep_alive<1, 2>())
        .def_static(
            "_uri_export", &Filestore::uri_export, nb::keep_alive<1, 2>())
        .def_static(
            "_buffer_import", &Filestore::buffer_import, nb::keep_alive<1, 2>())
        .def_static(
            "_buffer_export", &Filestore::buffer_export, nb::keep_alive<1, 2>())
        .def_static("_size", &Filestore::size, nb::keep_alive<1, 2>())
        .def_static("_mime_type_to_str", &Filestore::mime_type_to_str)
        .def_static("_mime_type_from_str", &Filestore::mime_type_from_str);
    ;
};

};  // namespace libtiledbcpp
