#include <tiledb/tiledb>
#include <tiledb/tiledb_experimental>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "metadata.h"

namespace libtiledbcpp {

using namespace tiledb;
using namespace tiledbpy::common;
namespace py = pybind11;

void init_group(py::module& m) {
    py::class_<Group>(m, "Group")
        .def(
            py::init<const Context&, const std::string&, tiledb_query_type_t>(),
            py::keep_alive<1, 2>())
        .def(
            py::init<
                const Context&,
                const std::string&,
                tiledb_query_type_t,
                const Config&>(),
            py::keep_alive<1, 2>())

        .def("_open", &Group::open)
        .def("_set_config", &Group::set_config)
        .def("_config", &Group::config)
        .def("_close", [](Group& self) { self.close(true); })
        .def_property_readonly("_isopen", &Group::is_open)
        .def_property_readonly("_uri", &Group::uri)
        .def_property_readonly("_query_type", &Group::query_type)

        .def(
            "_put_metadata",
            [](Group& group, const std::string& key, py::array value) {
                MetadataAdapter<Group> a;
                a.put_metadata_numpy(group, key, value);
            })
        .def(
            "_put_metadata",
            [](Group& group,
               const std::string& key,
               tiledb_datatype_t value_type,
               uint32_t value_num,
               py::buffer value) {
                MetadataAdapter<Group> a;
                a.put_metadata(group, key, value_type, value_num, value);
            })
        .def("_delete_metadata", &Group::delete_metadata)
        .def(
            "_has_metadata",
            [](Group& group, const std::string& key) {
                MetadataAdapter<Group> a;
                return a.has_metadata(group, key);
            })
        .def("_metadata_num", &Group::metadata_num)
        .def(
            "_get_metadata",
            [](Group& group, const std::string& key, bool is_ndarray) {
                MetadataAdapter<Group> a;
                return a.get_metadata(group, key, is_ndarray);
            })
        .def(
            "_get_key_from_index",
            [](Group& group, uint64_t index) {
                MetadataAdapter<Group> a;
                return a.get_key_from_index(group, index);
            })

        .def(
            "_add",
            &Group::add_member,
            py::arg("uri"),
            py::arg("relative") = false,
            py::arg("name") = std::nullopt
#if TILEDB_VERSION_MAJOR >= 3 || \
    (TILEDB_VERSION_MAJOR == 2 && TILEDB_VERSION_MINOR >= 27)
            ,
            py::arg("type") = std::nullopt
#endif
            )
        .def("_remove", &Group::remove_member)
        .def("_delete_group", &Group::delete_group)
        .def("_member_count", &Group::member_count)
        .def(
            "_member",
            static_cast<Object (Group::*)(uint64_t) const>(&Group::member))
        .def(
            "_member",
            static_cast<Object (Group::*)(std::string) const>(&Group::member))
        .def(
            "_has_member",
            [](Group& group, std::string obj) {
                MetadataAdapter<Group> a;
                return a.has_member(group, obj);
            })
        .def("_is_relative", &Group::is_relative)
        .def("_dump", &Group::dump)

        /* static methods */
        .def_static("_create", &Group::create)
        .def_static(
            "_consolidate_metadata",
            &Group::consolidate_metadata,
            py::arg("ctx"),
            py::arg("uri"),
            py::arg("config") = (Config*)nullptr)
        .def_static(
            "_vacuum_metadata",
            &Group::vacuum_metadata,
            py::arg("ctx"),
            py::arg("uri"),
            py::arg("config") = (Config*)nullptr);
}

}  // namespace libtiledbcpp
