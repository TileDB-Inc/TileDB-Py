#include <tiledb/tiledb>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "common.h"

namespace libtiledbcpp {

using namespace tiledb;
using namespace tiledbpy::common;
namespace py = pybind11;

void set_fill_value(Attribute &attr, py::array value) {
  attr.set_fill_value(value.data(), value.nbytes());
}

py::object get_fill_value(Attribute &attr) {
  const void *value;
  uint64_t size;

  attr.get_fill_value(&value, &size);

  switch (attr.type()) {
  case TILEDB_DATETIME_YEAR:
  case TILEDB_DATETIME_WEEK:
  case TILEDB_DATETIME_DAY:
  case TILEDB_DATETIME_HR:
  case TILEDB_DATETIME_MIN:
  case TILEDB_DATETIME_SEC:
  case TILEDB_DATETIME_MS:
  case TILEDB_DATETIME_US:
  case TILEDB_DATETIME_NS:
  case TILEDB_DATETIME_PS:
  case TILEDB_DATETIME_FS:
  case TILEDB_DATETIME_AS:
  case TILEDB_INT64: {
    return py::cast((const int64_t *)value);
  }
  case TILEDB_UINT64: {
    return py::cast((const uint64_t *)value);
  }
  case TILEDB_INT32: {
    return py::cast((const int32_t *)value);
  }
  case TILEDB_UINT32: {
    return py::cast((const uint32_t *)value);
  }
  case TILEDB_INT16: {
    return py::cast((const int16_t *)value);
  }
  case TILEDB_UINT16: {
    return py::cast((const uint16_t *)value);
  }
  case TILEDB_INT8: {
    return py::cast((const int8_t *)value);
  }
  case TILEDB_UINT8: {
    return py::cast((const uint8_t *)value);
  }
  case TILEDB_FLOAT64: {
    return py::cast((const double *)value);
  }
  case TILEDB_FLOAT32: {
    return py::cast((const float *)value);
  }
  case TILEDB_STRING_ASCII: {
    return py::cast(std::string((const char *)value, size));
  }
  default:
    TPY_ERROR_LOC("Unsupported dtype for Attribute");
  }
}

void init_attribute(py::module &m) {
  py::class_<tiledb::Attribute>(m, "Attribute")
      .def(py::init([](Context &ctx, std::string name, py::dtype datatype) {
             tiledb_datatype_t attr_dtype;
             try {
               attr_dtype = np_to_tdb_dtype(datatype);
             } catch (const TileDBPyError &e) {
               throw py::type_error(e.what());
             }

             return Attribute(ctx, name, attr_dtype);
           }),
           py::keep_alive<1, 2>())
      //   .def(py::init<Context &, std::string, tiledb_datatype_t>(),
      //        py::keep_alive<1, 2>() /* Attribute keeps Context alive */)
      .def(
          py::init<Context &, std::string &, tiledb_datatype_t, FilterList &>(),
          py::keep_alive<1, 2>() /* Attribute keeps Context alive */)

      .def_property_readonly("_name", &Attribute::name)

      .def_property_readonly("_tiledb_dtype", &Attribute::type)

      .def_property_readonly(
          "_numpy_dtype",
          [](Attribute &attr) { return tdb_to_np_dtype(attr.type(), 1); })

      .def_property("_nullable", &Attribute::nullable, &Attribute::set_nullable)

      .def_property("_ncell", &Attribute::cell_val_num,
                    &Attribute::set_cell_val_num)

      .def_property_readonly("_var", &Attribute::variable_sized)

      .def_property("_filters", &Attribute::filter_list,
                    &Attribute::set_filter_list)

      .def_property_readonly("_cell_size", &Attribute::cell_size)

      .def_property("_fill", get_fill_value, set_fill_value)

      // .def("_set_fill_value", set_fill_value)
      // .def("_get_fill_value", get_fill_value)

      .def("_dump", [](Attribute &attr) { attr.dump(); });
  ;
}

} // namespace libtiledbcpp
