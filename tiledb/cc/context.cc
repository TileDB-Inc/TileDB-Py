#include <tiledb/tiledb> // C++

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace libtiledbcpp {

using namespace tiledb;
namespace py = pybind11;

void init_context(py::module &m) {
  py::class_<Context>(m, "Context")
      .def(py::init())
      .def(py::init<Config>())
      .def(py::init<py::capsule, bool>())
      .def(py::init([](py::object ctx, bool own) {
             return Context(py::capsule(ctx.attr("__capsule__")()), own);
           }),
           py::keep_alive<1, 2>())

      .def("config", &Context::config)
      .def("set_tag", &Context::set_tag)
      .def("get_stats", &Context::stats)
      .def("is_supported_fs", &Context::is_supported_fs);
}

void init_config(py::module &m) {
  py::class_<tiledb::Config>(m, "Config")
      .def(py::init())
      .def(py::init<std::map<std::string, std::string>>())
      .def(py::init<std::string>())

      .def("set", &Config::set)
      .def("get", &Config::get)
      .def("save_to_file", &Config::save_to_file)
      .def("__eq__", &Config::operator==)
      .def("__ne__", &Config::operator!=)
      //.def("_ptr", &Config::ptr) // TBD should this be capsule?
      .def("__setitem__", [](Config &cfg, std::string &param,
                             std::string &val) { cfg[param] = val; })
      .def("__getitem__",
           [](const Config &cfg, std::string &param) {
             try {
               return cfg.get(param);
             } catch (TileDBError &e) {
               throw py::key_error();
             }
           })
      .def("__delitem__",
           [](Config &cfg, const std::string &param) {
             try {
               cfg.unset(param);
             } catch (TileDBError &e) {
               throw py::key_error();
             }
           })
      .def(
          "__iter__",
          [](Config &cfg) { return py::make_iterator(cfg.begin(), cfg.end()); },
          py::keep_alive<0, 1>())
      .def("__del__", &Config::unset)
      .def("unset", &Config::unset);
}

}; // namespace libtiledbcpp
