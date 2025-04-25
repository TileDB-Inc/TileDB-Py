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
    py::class_<tiledb::Profile>(m, "Profile")

        .def(
            py::init<std::optional<std::string>, std::optional<std::string>>(),
            py::keep_alive<1, 2>())

        .def_property_readonly("_name", &tiledb::Profile::get_name)

        .def_property_readonly("_homedir", &tiledb::Profile::get_homedir)

        .def(
            "_set_param",
            &tiledb::Profile::set_param,
            py::arg("param"),
            py::arg("value"))

        .def("_get_param", &tiledb::Profile::get_param, py::arg("param"))

        .def("_save", &tiledb::Profile::save)

        .def("_remove", &tiledb::Profile::remove)

        .def("_dump", &tiledb::Profile::dump);
}

}  // namespace libtiledbcpp
