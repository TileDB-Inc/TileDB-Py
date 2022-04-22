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

void put_metadata(Group &group, const std::string &key, py::array value) {
  tiledb_datatype_t value_type = np_to_tdb_dtype(value.dtype());

  if (is_tdb_str(value_type) && value.size() > 1)
    TPY_ERROR_LOC("array/list of strings not supported");

  py::buffer_info value_buffer = value.request();
  if (value_buffer.ndim != 1)
    TPY_ERROR_LOC("Only 1D Numpy arrays can be stored as metadata");

  py::size_t ncells = get_ncells(value.dtype());
  if (ncells != 1)
    TPY_ERROR_LOC("Unsupported dtype for metadata");

  auto value_num = is_tdb_str(value_type) ? value.nbytes() : value.size();
  group.put_metadata(key, value_type, value_num, value.data());
}

bool has_metadata(Group &group, const std::string &key) {
  tiledb_datatype_t _unused_value_type;
  return group.has_metadata(key, &_unused_value_type);
}

py::array get_metadata(Group &group, const std::string &key) {
  tiledb_datatype_t tdb_type;
  uint32_t value_num;
  const void *c_buf;

  group.get_metadata(key, &tdb_type, &value_num, &c_buf);

  py::dtype value_type = tdb_to_np_dtype(tdb_type, 1);

  if (tdb_type == TILEDB_STRING_UTF8)
    value_num /= value_type.itemsize();

  py::array py_buf(value_type, value_num);
  memcpy(py_buf.mutable_data(), c_buf, py_buf.nbytes());

  return py_buf;
}

void init_group(py::module &m) {
  py::class_<Group>(m, "Group")
      .def(
          py::init<const Context &, const std::string &, tiledb_query_type_t>(),
          py::keep_alive<1, 2>())

      .def("_open", &Group::open)
      .def("_set_config", &Group::set_config)
      .def("_config", &Group::config)
      .def("_close", &Group::close)
      .def("_create", &Group::create)

      .def_property_readonly("_isopen", &Group::is_open)
      .def_property_readonly("_uri", &Group::uri)
      .def_property_readonly("_query_type", &Group::query_type)

      .def("_put_metadata", put_metadata)
      .def("_delete_metadata", &Group::delete_metadata)
      .def("_has_metadata", has_metadata)
      .def("_metadata_num", &Group::metadata_num)
      .def("_get_metadata", get_metadata)

      // NOTE is this worth implementing?
      //   .def("get_metadata_from_index", get_metadata_from_index)

      .def("_add", &Group::add_member, py::arg("uri"),
           py::arg("relative") = false, py::arg("name") = std::nullopt)
      .def("_remove", &Group::remove_member)
      .def("_member_count", &Group::member_count)
      .def("_member",
           static_cast<Object (Group::*)(uint64_t) const>(&Group::member))
      .def("_member",
           static_cast<Object (Group::*)(std::string) const>(&Group::member))
      .def("_dump", &Group::dump);
}

} // namespace libtiledbcpp
