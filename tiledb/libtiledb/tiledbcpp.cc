#include <tiledb/tiledb>  // C++

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// #include <pybind11/pytypes.h>
// #include <pybind11/stl.h>
#include "common.h"

namespace libtiledbcpp {

using namespace tiledb;
namespace nb = nanobind;

template <typename... Args>
using overload_cast_ = nb::detail::overload_cast_impl<Args...>;

void init_array(nb::module_&);
void init_attribute(nb::module_&);
void init_context(nb::module_&);
void init_config(nb::module_&);
void init_consolidation_plan(nb::module_& m);
void init_current_domain(nb::module_& m);
void init_enums(nb::module_&);
void init_enumeration(nb::module_&);
void init_dimension_label(nb::module_& m);
void init_domain(nb::module_& m);
void init_file_handle(nb::module_&);
void init_filestore(nb::module_& m);
void init_filter(nb::module_&);
void init_group(nb::module_&);
void init_object(nb::module_& m);
void init_query(nb::module_& m);
void init_schema(nb::module_&);
void init_subarray(nb::module_&);
void init_vfs(nb::module_& m);

NB_MODULE(libtiledb, m) {
    init_array(m);
    init_attribute(m);
    init_context(m);
    init_config(m);
    init_consolidation_plan(m);
    init_current_domain(m);
    init_dimension_label(m);
    init_domain(m);
    init_enums(m);
    init_enumeration(m);
    init_file_handle(m);
    init_filestore(m);
    init_filter(m);
    init_group(m);
    init_object(m);
    init_query(m);
    init_schema(m);
    init_subarray(m);
    init_vfs(m);

    m.def("version", []() {
        int major = 0;
        int minor = 0;
        int rev = 0;
        tiledb_version(&major, &minor, &rev);
        return std::make_tuple(major, minor, rev);
    });

    nb::exception<TileDBError>(m, "TileDBError");

    /*
     We need to make sure C++ TileDBError is translated to a correctly-typed py
     error. Note that using nb::exception(..., "TileDBError") creates a new
     exception in the *readquery* module, so we must import to reference.
    */
    nb::exception_translator([](std::exception_ptr p) {
        auto tiledb_py_error = (nb::object)nb::module_::import_("tiledb").attr(
            "TileDBError");

        try {
            if (p)
                std::rethrow_exception(p);
        } catch (const TileDBPyError& e) {
            PyErr_SetString(tiledb_py_error.ptr(), e.what());
        } catch (const tiledb::TileDBError& e) {
            PyErr_SetString(tiledb_py_error.ptr(), e.what());
        } catch (nb::builtin_exception& e) {
            throw;
        };
    });
}

};  // namespace libtiledbcpp
