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

py::array get_fill_value(Attribute &attr) {
  switch (attr.type()) {
  case TILEDB_UINT64: {
    const uint64_t *value;
    uint64_t size;
    attr.get_fill_value((const void **)&value, &size);
    return py::array(py::dtype("uint64"), 1);
  }
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
    const int64_t *value;
    uint64_t size;
    attr.get_fill_value((const void **)&value, &size);
    return py::array(py::dtype("int64"), 1);
  }
  case TILEDB_UINT32: {
    const uint32_t *value;
    uint64_t size;
    attr.get_fill_value((const void **)&value, &size);
    return py::array(py::dtype("uint32"), 1);
  }
  case TILEDB_INT32: {
    const int32_t *value;
    uint64_t size;
    attr.get_fill_value((const void **)&value, &size);
    return py::array(py::dtype("int32"), 1);
  }
  case TILEDB_UINT16: {
    const uint16_t *value;
    uint64_t size;
    attr.get_fill_value((const void **)&value, &size);
    return py::array(py::dtype("uint16"), 1);
  }
  case TILEDB_INT16: {
    const int16_t *value;
    uint64_t size;
    attr.get_fill_value((const void **)&value, &size);
    return py::array(py::dtype("int16"), 1);
  }
  case TILEDB_UINT8: {
    const uint8_t *value;
    uint64_t size;
    attr.get_fill_value((const void **)&value, &size);
    return py::array(py::dtype("uint8"), 1);
  }
  case TILEDB_INT8: {
    const int8_t *value;
    uint64_t size;
    attr.get_fill_value((const void **)&value, &size);
    return py::array(py::dtype("int8"), 1);
  }
  case TILEDB_FLOAT64: {
    const double *value;
    uint64_t size;
    attr.get_fill_value((const void **)&value, &size);
    return py::array(py::dtype("double"), 1);
  }
  case TILEDB_FLOAT32: {
    const float *value;
    uint64_t size;
    attr.get_fill_value((const void **)&value, &size);
    return py::array(py::dtype("float"), 1);
  }
  default:
    TPY_ERROR_LOC("Unsupported dtype for Dimension's domain");
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

      .def_property_readonly("_fill_value", get_fill_value)

      .def("_dump", [](Attribute &attr) { attr.dump(); });
  ;
}

} // namespace libtiledbcpp
