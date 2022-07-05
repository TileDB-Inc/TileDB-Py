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

      .def_property_readonly("_dtype", &Attribute::type)

      .def_property("_nullable", &Attribute::nullable, &Attribute::set_nullable)

      .def_property("_ncell", &Attribute::cell_val_num,
                    &Attribute::set_cell_val_num)

      //   .def_property("fill", &Attribute::get_fill_value,
      //                 &Attribute::set_fill_value)

      .def_property_readonly("_var", &Attribute::variable_sized)

      .def_property("_filters", &Attribute::filter_list,
                    &Attribute::set_filter_list)

      .def_property_readonly("_cell_size", &Attribute::cell_size);
}

} // namespace libtiledbcpp
