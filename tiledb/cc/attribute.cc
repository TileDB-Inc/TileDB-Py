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

py::array get_fill_value(Attribute &attr) {
  const void *value;
  uint64_t size;

  attr.get_fill_value(&value, &size);

  auto value_num = attr.cell_val_num();
  auto value_type = tdb_to_np_dtype(attr.type(), value_num);

  if (is_tdb_str(attr.type())) {
    value_type = py::dtype("|S1");
    value_num = size;
  }

  return py::array(value_type, value_num, value);
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
          py::keep_alive<1, 2>())

      .def_property_readonly("_name", &Attribute::name)

      .def_property_readonly("_tiledb_dtype", &Attribute::type)

      .def_property_readonly("_numpy_dtype",
                             [](Attribute &attr) {
                               return tdb_to_np_dtype(attr.type(),
                                                      attr.cell_val_num());
                             })

      .def_property("_nullable", &Attribute::nullable, &Attribute::set_nullable)

      .def_property("_ncell", &Attribute::cell_val_num,
                    &Attribute::set_cell_val_num)

      .def_property_readonly("_var", &Attribute::variable_sized)

      .def_property("_filters", &Attribute::filter_list,
                    &Attribute::set_filter_list)

      .def_property_readonly("_cell_size", &Attribute::cell_size)

      .def_property("_fill", get_fill_value, set_fill_value)

      .def("_dump", [](Attribute &attr) { attr.dump(); });
  ;
}

} // namespace libtiledbcpp
