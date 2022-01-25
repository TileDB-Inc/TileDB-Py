#include <tiledb/tiledb> // C++

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace libtiledbcpp {

using namespace tiledb;
namespace py = pybind11;


void init_array(py::module& m) {
      py::class_<tiledb::Array>(m, "Array")
          //.def(py::init<py::object, py::object, py::iterable, py::object,
          //              py::object, py::object>())
          .def(py::init<const Context&, const std::string&, tiledb_query_type_t>(),
               py::keep_alive<1,2>() /* Array keeps Context alive */)
          // TODO capsule Array(const Context& ctx, tiledb_array_t* carray, tiledb_config_t* config)
          .def("is_open", &Array::is_open)
          .def("uri", &Array::uri)
          .def("schema", &Array::schema)
          //.def("ptr", [](Array& arr){ return py::capsule(arr.ptr()); } )
          // open with encryption key
          .def("open", (void (Array::*)(tiledb_query_type_t, tiledb_encryption_type_t, const std::string&))&Array::open)
          // open with encryption key and timestamp
          .def("open", (void (Array::*)(tiledb_query_type_t, tiledb_encryption_type_t, const std::string&, uint64_t))&Array::open)
          .def("reopen", &Array::reopen);
}

} // namespace