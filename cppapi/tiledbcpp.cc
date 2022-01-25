#include <tiledb/tiledb> // C++

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace libtiledbcpp {

using namespace tiledb;
namespace py = pybind11;

PYBIND11_MODULE(libtiledbcpp, m) {
  py::class_<Context>(m, "Context")
    .def(py::init())
    .def(py::init<Config>())
    .def("config", &Context::config);

  py::class_<tiledb::Config>(m, "Config")
    .def(py::init())
    .def("set", &Config::set)
    .def("get", &Config::get)
    .def("save_to_file", &Config::save_to_file)
    .def("__eq__", &Config::operator==)
    .def("__ne__", &Config::operator!=)
    //.def("_ptr", &Config::ptr) // TBD should this be capsule?
    .def("__setitem__", [](Config& cfg, std::string &param, std::string &val) {
       cfg[param] = val;
    })
    .def("__getitem__", [](Config& cfg, std::string &param) {
        return cfg.get(param);
    })
    .def("__iter__", [](Config& cfg) {
        return py::make_iterator(cfg.begin(), cfg.end());
    }, py::keep_alive<0,1>())
    .def("__del__", &Config::unset)
    .def("unset", &Config::unset);

}


/*
PYBIND11_MODULE(libtiledbcpp, m) {
  py::class_<tiledb::Query>(m, "Query")
      .def(py::init<py::object, py::object, py::iterable, py::object,
                    py::object, py::object>())
      .def("set_ranges", &PyQuery::set_ranges)
      .def("set_subarray", &PyQuery::set_subarray)
*/

}; // namespace libtiledbcpp