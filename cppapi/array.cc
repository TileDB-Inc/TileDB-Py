#include <tiledb/tiledb> // C++
#include <tiledb/tiledb.h> // for enums

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
          .def("reopen", &Array::reopen)
          .def("set_open_timestamp_start", &Array::set_open_timestamp_start)
          .def("set_open_timestamp_end", &Array::set_open_timestamp_end)
          .def_property_readonly("open_timestamp_start", &Array::open_timestamp_start)
          .def_property_readonly("open_timestamp_end", &Array::open_timestamp_end)
          .def("set_config", &Array::set_config)
          .def("config", &Array::config)
          .def("close", &Array::close)
          .def("consolidate",
               py::overload_cast<const Context&, const std::string&, Config* const>(&Array::consolidate))
          .def("consolidate",
               py::overload_cast<const Context&, const std::string&,
                                 tiledb_encryption_type_t, const std::string&,
                                 Config* const>(&Array::consolidate))
               //(void (Array::*)(const Context&, const std::string&,
               //                 tiledb_encryption_type_t, const std::string&,
               //                 Config* const)&Array::consolidate)&Array::consolidate)
          .def("vacuum", &Array::vacuum)
          .def("create",
               py::overload_cast<const std::string&, const ArraySchema&, tiledb_encryption_type_t, const std::string&>(&Array::create))
          .def("load_schema",
               py::overload_cast<const Context&, const std::string&>(&Array::load_schema))
          .def("create",
               py::overload_cast<const std::string&, const ArraySchema&,
                                 tiledb_encryption_type_t, const std::string&>(&Array::create))
          .def("encryption_type", &Array::encryption_type)

          // TODO non_empty_domain
          // TODO non_empty_domain_var

          .def("query_type", &Array::query_type)
          .def("consolidate_metadata",
               py::overload_cast<const Context&, const std::string&,
                              tiledb_encryption_type_t, const std::string&,
                              Config* const>(&Array::consolidate_metadata))
          .def("put_metadata", &Array::put_metadata)
          //.def("get_metadata", &Array::get_metadata)
          //.def("has_metadata", &Array::has_metadata)
          ;

}

} // namespace