#include <tiledb/tiledb> // C++


#include "common.h"

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
        .def(py::init<Context&, Array&, tiledb_query_type_t>(),
            py::keep_alive<1,2>() )
        .def(py::init<Context&, Array&>(),
            py::keep_alive<1,2>() ) // TODO keepalive for the Array as well?
        // TODO .def("ptr", [&]() -> py::capsule)
        .def("query_type", &Query::query_type)
        .def("set_layout", &Query::set_layout)
        .def("query_layout", &Query::query_layout)
        .def("set_condition", &Query::set_condition)
        // TODO .def("array") -> Array&
        .def("query_status", &Query::query_status)
        .def("has_results", &Query::has_results)
        .def("submit", &Query::submit)
        //TODO .def("submit_async", Fn& callback)
        //.def("submit_async")
        .def("finalize", &Query::finalize)
        // TODO .def("result_buffer_elements", &Query::result_buffer_elements)
        // TODO .def("result_buffer_elements_nullable", &Query::result_buffer_eleents_nullable)
        // TODO .def("add_range")
        // note the pointer cast here. overload_cast *does not work*
        //   https://github.com/pybind/pybind11/issues/1153
        //   https://pybind11.readthedocs.io/en/latest/classes.html#overloaded-methods
        .def("add_range", (Query& (Query::*)(const std::string&, const std::string&, const std::string&))&Query::add_range)
        .def("add_range", (Query& (Query::*)(uint32_t, const std::string&, const std::string&))&Query::add_range)
        // TODO generic add_range implemented here
        .def("set_data_buffer",
             (Query& (Query::*)(const std::string&, void*, uint64_t))&Query::set_data_buffer);
}

} // namespace