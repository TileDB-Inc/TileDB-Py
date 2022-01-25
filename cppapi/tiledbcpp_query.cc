#include <tiledb/tiledb> // C++

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace libtiledbcpp {

using namespace tiledb;
namespace py = pybind11;


void init_query(py::module& m) {
      py::class_<tiledb::Query>(m, "Query")
          //.def(py::init<py::object, py::object, py::iterable, py::object,
          //              py::object, py::object>())
          .def(py::init<Context, Array, tiledb_query_type_t>());
          //.def("set_data_buffer", &Query::set_data_buffer)
          //.def("add_range", &Query::add_range)
}

} // namespace