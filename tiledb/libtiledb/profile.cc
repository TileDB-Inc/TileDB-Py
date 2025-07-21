#include <tiledb/tiledb>
#include <tiledb/tiledb_experimental>

#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "common.h"

namespace libtiledbcpp {

using namespace tiledb;
using namespace tiledbpy::common;
namespace py = pybind11;

void init_profile(py::module& m) {
#if TILEDB_VERSION_MAJOR > 2 ||                                 \
    (TILEDB_VERSION_MAJOR == 2 && TILEDB_VERSION_MINOR > 28) || \
    (TILEDB_VERSION_MAJOR == 2 && TILEDB_VERSION_MINOR == 28 && \
     TILEDB_VERSION_PATCH >= 1)
    py::class_<tiledb::Profile>(m, "Profile")

        .def(
            py::init<std::optional<std::string>, std::optional<std::string>>(),
            py::arg("name") = std::nullopt,
            py::arg("dir") = std::nullopt)

        .def(py::init<Profile>())

        .def_property_readonly("_name", &tiledb::Profile::name)

        .def_property_readonly("_dir", &tiledb::Profile::dir)

        .def(
            "_set_param",
            &tiledb::Profile::set_param,
            py::arg("param"),
            py::arg("value"))

        .def("_get_param", &tiledb::Profile::get_param, py::arg("param"))

        .def("_save", &tiledb::Profile::save)

        .def_static(
            "_load",
            py::overload_cast<
                std::optional<std::string>,
                std::optional<std::string>>(&tiledb::Profile::load),
            py::arg("name") = std::nullopt,
            py::arg("dir") = std::nullopt)

        .def_static(
            "_remove",
            py::overload_cast<
                std::optional<std::string>,
                std::optional<std::string>>(&tiledb::Profile::remove),
            py::arg("name") = std::nullopt,
            py::arg("dir") = std::nullopt)

        .def("_dump", &tiledb::Profile::dump);
#endif
}

}  // namespace libtiledbcpp
