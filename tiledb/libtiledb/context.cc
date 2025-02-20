#include <tiledb/tiledb>  // C++

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace libtiledbcpp {

using namespace tiledb;
namespace nb = nanobind;

void init_context(nb::module& m) {
    nb::class_<Context>(m, "Context")
        .def(nb::init<Context>())
        .def(nb::init())
        .def(nb::init<Config>())
        .def(nb::init<nb::capsule, bool>())

        .def(
            "__capsule__",
            [](Context& ctx) { return nb::capsule(ctx.ptr().get(), "ctx"); })

        .def("config", &Context::config)
        .def("set_tag", &Context::set_tag)
        .def("get_stats", &Context::stats)
        .def("is_supported_fs", &Context::is_supported_fs);
}

void init_config(nb::module& m) {
    nb::class_<tiledb::Config>(m, "Config")
        .def(nb::init<Config>())
        .def(nb::init())
        .def(nb::init<std::map<std::string, std::string>>())
        .def(nb::init<std::string>())

        .def(
            "__capsule__",
            [](Config& config) {
                return nb::capsule(config.ptr().get(), "config");
            })

        .def("set", &Config::set)
        .def("get", &Config::get)
        .def(
            "update",
            [](Config& cfg, nb::dict& odict) {
                for (auto item : odict) {
                    cfg.set(
                        item.first.cast<nb::str>(),
                        item.second.cast<nb::str>());
                }
            })

        .def("save_to_file", &Config::save_to_file)
        .def("__eq__", &Config::operator==)
        .def("__ne__", &Config::operator!=)
        //.def("_ptr", &Config::ptr) // TBD should this be capsule?
        .def(
            "__setitem__",
            [](Config& cfg, std::string& param, std::string& val) {
                cfg[param] = val;
            })
        .def(
            "__getitem__",
            [](const Config& cfg, std::string& param) {
                try {
                    return cfg.get(param);
                } catch (TileDBError& e) {
                    throw nb::key_error();
                }
            })
        .def(
            "__delitem__",
            [](Config& cfg, const std::string& param) {
                try {
                    cfg.unset(param);
                } catch (TileDBError& e) {
                    throw nb::key_error();
                }
            })
        .def(
            "_iter",
            [](Config& cfg, std::string prefix) {
                return nb::make_iterator(cfg.begin(prefix), cfg.end());
            },
            nb::keep_alive<0, 1>(),
            nb::arg("prefix") = "")
        .def("unset", &Config::unset);
}
};  // namespace libtiledbcpp
