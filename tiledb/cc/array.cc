#include <tiledb/tiledb.h> // for enums
#include <tiledb/tiledb>   // C++

#include "common.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace libtiledbcpp {

using namespace tiledb;
namespace py = pybind11;

void init_array(py::module &m) {
  py::class_<tiledb::Array>(m, "Array")
      //.def(py::init<py::object, py::object, py::iterable, py::object,
      //              py::object, py::object>())
      .def(
          py::init<const Context &, const std::string &, tiledb_query_type_t>(),
          py::keep_alive<1, 2>() /* Array keeps Context alive */)

      // Temporary initializer while Array is converted from Cython to PyBind.
      .def(py::init([](const Context &ctx, py::object array) {
             tiledb_array_t *c_array = (py::capsule)array.attr("__capsule__")();
             return std::make_unique<Array>(ctx, c_array, false);
           }),
           py::keep_alive<1, 2>(), py::keep_alive<1, 3>())

      // TODO capsule Array(const Context& ctx, tiledb_array_t* carray,
      // tiledb_config_t* config)
      .def("is_open", &Array::is_open)
      .def("uri", &Array::uri)
      .def("schema", &Array::schema)
      //.def("ptr", [](Array& arr){ return py::capsule(arr.ptr()); } )
      .def("open", (void(Array::*)(tiledb_query_type_t)) & Array::open)
      // open with encryption key
      .def("open",
           (void(Array::*)(tiledb_query_type_t, tiledb_encryption_type_t,
                           const std::string &)) &
               Array::open)
      // open with encryption key and timestamp
      .def("open",
           (void(Array::*)(tiledb_query_type_t, tiledb_encryption_type_t,
                           const std::string &, uint64_t)) &
               Array::open)
      .def("reopen", &Array::reopen)
      .def("set_open_timestamp_start", &Array::set_open_timestamp_start)
      .def("set_open_timestamp_end", &Array::set_open_timestamp_end)
      .def_property_readonly("open_timestamp_start",
                             &Array::open_timestamp_start)
      .def_property_readonly("open_timestamp_end", &Array::open_timestamp_end)
      .def("set_config", &Array::set_config)
      .def("config", &Array::config)
      .def("close", &Array::close)
      .def("consolidate",
           py::overload_cast<const Context &, const std::string &,
                             Config *const>(&Array::consolidate),
           py::call_guard<py::gil_scoped_release>())
      .def("consolidate",
           py::overload_cast<const Context &, const std::string &,
                             tiledb_encryption_type_t, const std::string &,
                             Config *const>(&Array::consolidate),
           py::call_guard<py::gil_scoped_release>())
      //(void (Array::*)(const Context&, const std::string&,
      //                 tiledb_encryption_type_t, const std::string&,
      //                 Config* const)&Array::consolidate)&Array::consolidate)
      .def("vacuum", &Array::vacuum)
      .def("create",
           py::overload_cast<const std::string &, const ArraySchema &,
                             tiledb_encryption_type_t, const std::string &>(
               &Array::create))
      .def("create",
           py::overload_cast<const std::string &, const ArraySchema &>(
               &Array::create))
      .def("load_schema",
           py::overload_cast<const Context &, const std::string &>(
               &Array::load_schema))
      .def("encryption_type", &Array::encryption_type)

      // TODO non_empty_domain
      // TODO non_empty_domain_var

      .def("query_type", &Array::query_type)
      .def("consolidate_metadata",
           py::overload_cast<const Context &, const std::string &,
                             tiledb_encryption_type_t, const std::string &,
                             Config *const>(&Array::consolidate_metadata))
      .def("put_metadata",
           [](Array &self, std::string &key, tiledb_datatype_t tdb_type,
              const py::buffer &b) {
             py::buffer_info info = b.request();

             // size_t size = std::reduce(info.shape.begin(),
             // info.shape.end());
             size_t size = 1;
             for (auto s : info.shape) {
               size *= s;
             }
             // size_t nbytes = size * info.itemsize;

             self.put_metadata(key, tdb_type, size, info.ptr);
             /*
             std::cout << "ndim: " << info.ndim << std::endl;


             std::cout << "sz: " << size << std::endl;
             std::cout << "imsz: " << info.itemsize << std::endl;

             std::cout << "--|" << std::endl;
             for (auto& s : info.shape) {
                  std::cout << s << std::endl;
             }
             */
           })
      .def("get_metadata",
           [](Array &self, std::string &key) -> py::buffer {
             tiledb_datatype_t tdb_type;
             uint32_t value_num = 0;
             const void *data_ptr = nullptr;

             self.get_metadata(key, &tdb_type, &value_num, &data_ptr);

             if (data_ptr == nullptr && value_num != 1) {
               throw py::key_error();
             }

             assert(data_ptr != nullptr);
             return py::memoryview::from_memory(
                 data_ptr, value_num * tiledb_datatype_size(tdb_type));
           })
      .def("get_metadata_from_index",
           [](Array &self, uint64_t index) -> py::tuple {
             tiledb_datatype_t tdb_type;
             uint32_t value_num = 0;
             const void *data_ptr = nullptr;
             std::string key;

             self.get_metadata_from_index(index, &key, &tdb_type, &value_num,
                                          &data_ptr);

             if (data_ptr == nullptr && value_num != 1) {
               throw py::key_error();
             }
             // TODO handle empty value case

             assert(data_ptr != nullptr);
             auto buf = py::memoryview::from_memory(
                 data_ptr, value_num * tiledb_datatype_size(tdb_type));

             return py::make_tuple(tdb_type, buf);
           })
      .def("delete_metadata", &Array::delete_metadata)
      .def("has_metadata",
           [](Array &self, std::string &key) -> py::tuple {
             tiledb_datatype_t has_type;
             bool has_it = self.has_metadata(key, &has_type);
             return py::make_tuple(has_it, has_type);
           })
      .def("metadata_num", &Array::metadata_num)
#if TILEDB_VERSION_MAJOR >= 2 && TILEDB_VERSION_MINOR >= 14
      .def("delete_array",
           py::overload_cast<const Context &, const std::string &>(
               &Array::delete_array));
#else
      .def("delete_array", &Array::delete_array);
#endif
}

} // namespace libtiledbcpp
