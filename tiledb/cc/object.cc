#include <tiledb/tiledb>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "common.h"

namespace libtiledbcpp {

using namespace tiledb;
using namespace tiledbpy::common;
namespace py = pybind11;

void init_object(py::module &m) {
  py::class_<Object>(m, "Object")
      .def(py::init<const Object::Type &, const std::string &,
                    const std::optional<std::string> &>())
      .def(py::init<tiledb_object_t, const std::string &,
                    const std::optional<std::string> &>())

      .def_property_readonly("_type", &Object::type)
      .def_property_readonly("_uri", &Object::uri)
      .def_property_readonly("_name", &Object::name)
      .def("__repr__", &Object::to_str)

      .def_static("_object", &Object::object)
      .def_static("_remove", &Object::remove)
      .def_static("_move", &Object::move);
}

} // namespace libtiledbcpp
