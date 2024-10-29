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

  if (value.ndim() != 1)
    throw py::type_error("Only 1D Numpy arrays can be stored as metadata");

  py::size_t ncells = get_ncells(value.dtype());
  // we can't store multi-cell arrays as metadata
  // e.g. an array of strings containing strings of more than one character
  if (ncells != 1 && value.size() > 1)
    throw py::type_error("Unsupported dtype '" +
                         std::string(py::str(value.dtype())) +
                         "' for metadata");

  auto value_num = is_tdb_str(value_type) ? value.nbytes() : value.size();
  group.put_metadata(key, value_type, value_num,
                     value_num > 0 ? value.data() : nullptr);
}

void put_metadata(Group &group, const std::string &key,
                  tiledb_datatype_t value_type, uint32_t value_num,
                  py::buffer &value) {

  py::buffer_info info = value.request();
  group.put_metadata(key, value_type, value_num, info.ptr);
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

py::object unpack_metadata_val(tiledb_datatype_t value_type, uint32_t value_num,
                               const char *value_ptr) {
  if (value_num == 0)
    throw TileDBError("internal error: unexpected value_num==0");

  if (value_type == TILEDB_STRING_UTF8) {
    return value_ptr == nullptr ? py::str() : py::str(value_ptr, value_num);
  }

  if (value_type == TILEDB_BLOB || value_type == TILEDB_CHAR ||
      value_type == TILEDB_STRING_ASCII) {
    return value_ptr == nullptr ? py::bytes() : py::bytes(value_ptr, value_num);
  }

  if (value_ptr == nullptr)
    return py::tuple();

  py::tuple unpacked(value_num);
  for (uint32_t i = 0; i < value_num; i++) {
    switch (value_type) {
    case TILEDB_INT64:
      unpacked[i] = *((int64_t *)value_ptr);
      break;
    case TILEDB_FLOAT64:
      unpacked[i] = *((double *)value_ptr);
      break;
    case TILEDB_FLOAT32:
      unpacked[i] = *((float *)value_ptr);
      break;
    case TILEDB_INT32:
      unpacked[i] = *((int32_t *)value_ptr);
      break;
    case TILEDB_UINT32:
      unpacked[i] = *((uint32_t *)value_ptr);
      break;
    case TILEDB_UINT64:
      unpacked[i] = *((uint64_t *)value_ptr);
      break;
    case TILEDB_INT8:
      unpacked[i] = *((int8_t *)value_ptr);
      break;
    case TILEDB_UINT8:
      unpacked[i] = *((uint8_t *)value_ptr);
      break;
    case TILEDB_INT16:
      unpacked[i] = *((int16_t *)value_ptr);
      break;
    case TILEDB_UINT16:
      unpacked[i] = *((uint16_t *)value_ptr);
      break;
    default:
      throw TileDBError("TileDB datatype not supported");
    }
    value_ptr += tiledb_datatype_size(value_type);
  }

  if (value_num > 1)
    return unpacked;

  // for single values, return the value directly
  return unpacked[0];
}

py::array unpack_metadata_ndarray(tiledb_datatype_t value_type,
                                  uint32_t value_num, const char *value_ptr) {
  py::dtype dtype = tdb_to_np_dtype(value_type, 1);

  if (value_ptr == nullptr) {
    auto np = py::module::import("numpy");
    return np.attr("empty")(py::make_tuple(0), dtype);
  }

  // special case for TILEDB_STRING_UTF8: TileDB assumes size=1
  if (value_type != TILEDB_STRING_UTF8) {
    value_num *= tiledb_datatype_size(value_type);
  }

  auto buf = py::memoryview::from_memory(value_ptr, value_num);

  auto np = py::module::import("numpy");
  return np.attr("frombuffer")(buf, dtype);
}

py::tuple get_metadata(Group &group, const py::str &key, bool is_ndarray) {
  tiledb_datatype_t tdb_type;
  uint32_t value_num;
  const char *value_ptr;

  group.get_metadata(key, &tdb_type, &value_num, (const void **)&value_ptr);
  if (is_ndarray) {
    auto arr = unpack_metadata_ndarray(tdb_type, value_num, value_ptr);
    return py::make_tuple(arr, tdb_type);
  } else {
    auto arr = unpack_metadata_val(tdb_type, value_num, value_ptr);
    return py::make_tuple(arr, tdb_type);
  }
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
           py::arg("relative") = false, py::arg("name") = std::nullopt
#if TILEDB_VERSION_MAJOR >= 2 && TILEDB_VERSION_MINOR >= 27
           ,
           py::arg("type") = std::nullopt
#endif
           )
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
