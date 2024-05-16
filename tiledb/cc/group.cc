#include <tiledb/tiledb>
#include <tiledb/tiledb_experimental>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "common.h"

namespace libtiledbcpp {

using namespace tiledb;
using namespace tiledbpy::common;
namespace py = pybind11;

void put_metadata_numpy(Group &group, const std::string &key, py::array value) {
  tiledb_datatype_t value_type;
  try {
    value_type = np_to_tdb_dtype(value.dtype());
  } catch (const TileDBPyError &e) {
    throw py::type_error(e.what());
  }

  if (is_tdb_str(value_type) && value.size() > 1)
    throw py::type_error("array/list of strings not supported");

  py::buffer_info value_buffer = value.request();
  if (value_buffer.ndim != 1)
    throw py::type_error("Only 1D Numpy arrays can be stored as metadata");

  py::size_t ncells = get_ncells(value.dtype());
  if (ncells != 1)
    throw py::type_error("Unsupported dtype for metadata");

  auto value_num = is_tdb_str(value_type) ? value.nbytes() : value.size();
  group.put_metadata(key, value_type, value_num,
                     value_num > 0 ? value.data() : nullptr);
}

void put_metadata(Group &group, const std::string &key,
                  tiledb_datatype_t value_type, uint32_t value_num,
                  const char *value) {
  group.put_metadata(key, value_type, value_num, value);
}

bool has_metadata(Group &group, const std::string &key) {
  tiledb_datatype_t _unused_value_type;
  return group.has_metadata(key, &_unused_value_type);
}

std::string get_key_from_index(Group &group, uint64_t index) {
  std::string key;
  tiledb_datatype_t tdb_type;
  uint32_t value_num;
  const void *value;

  group.get_metadata_from_index(index, &key, &tdb_type, &value_num, &value);

  return key;
}

py::tuple get_metadata(Group &group, const std::string &key) {
  tiledb_datatype_t tdb_type;
  uint32_t value_num;
  const void *value;

  group.get_metadata(key, &tdb_type, &value_num, &value);

  py::dtype value_type = tdb_to_np_dtype(tdb_type, 1);

  py::array py_buf;
  if (value == nullptr) {
    py_buf = py::array(value_type, 0);
    return py::make_tuple(py_buf, tdb_type);
  }

  if (tdb_type == TILEDB_STRING_UTF8) {
    value_type = py::dtype("|S1");
  }

  py_buf = py::array(value_type, value_num, value);

  return py::make_tuple(py_buf, tdb_type);
}

bool has_member(Group &group, std::string obj) {
  try {
    group.member(obj);
  } catch (const TileDBError &e) {
    return false;
  }
  return true;
}

void init_group(py::module &m) {
  py::class_<Group>(m, "Group")
      .def(
          py::init<const Context &, const std::string &, tiledb_query_type_t>(),
          py::keep_alive<1, 2>())
      .def(py::init<const Context &, const std::string &, tiledb_query_type_t,
                    const Config &>(),
           py::keep_alive<1, 2>())

      .def("_open", &Group::open)
      .def("_set_config", &Group::set_config)
      .def("_config", &Group::config)
      .def("_close", [](Group &self) { self.close(true); })
      .def_property_readonly("_isopen", &Group::is_open)
      .def_property_readonly("_uri", &Group::uri)
      .def_property_readonly("_query_type", &Group::query_type)

      .def("_put_metadata", put_metadata_numpy)
      .def("_put_metadata", put_metadata)

      .def("_delete_metadata", &Group::delete_metadata)
      .def("_has_metadata", has_metadata)
      .def("_metadata_num", &Group::metadata_num)
      .def("_get_metadata", get_metadata)
      .def("_get_key_from_index", get_key_from_index)

      .def("_add", &Group::add_member, py::arg("uri"),
           py::arg("relative") = false, py::arg("name") = std::nullopt)
      .def("_remove", &Group::remove_member)
      .def("_delete_group", &Group::delete_group)
      .def("_member_count", &Group::member_count)
      .def("_member",
           static_cast<Object (Group::*)(uint64_t) const>(&Group::member))
      .def("_member",
           static_cast<Object (Group::*)(std::string) const>(&Group::member))
      .def("_has_member", has_member)
      .def("_is_relative", &Group::is_relative)
      .def("_dump", &Group::dump)

      /* static methods */
      .def_static("_create", &Group::create)
      .def_static("_consolidate_metadata", &Group::consolidate_metadata,
                  py::arg("ctx"), py::arg("uri"),
                  py::arg("config") = (Config *)nullptr)
      .def_static("_vacuum_metadata", &Group::vacuum_metadata, py::arg("ctx"),
                  py::arg("uri"), py::arg("config") = (Config *)nullptr);
}

} // namespace libtiledbcpp
