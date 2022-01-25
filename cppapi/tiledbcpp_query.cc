#include <tiledb/tiledb> // C++

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace libtiledbcpp {

using namespace tiledb;
namespace py = pybind11;


void init_query(py::module& m) {

    PYBIND11_MODULE(libtiledbcpp, m) {
      py::class_<tiledb::Query>(m, "Query")
          .def(py::init<py::object, py::object, py::iterable, py::object,
                        py::object, py::object>())
          .def("set_ranges", &PyQuery::set_ranges)
          .def("set_subarray", &PyQuery::set_subarray)
    }
}

} // namespace