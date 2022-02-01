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
      .def(py::init<Context &, std::string, tiledb_datatype_t>(),
           py::keep_alive<1, 2>() /* Attribute keeps Context alive */)
      .def(
          py::init<Context &, std::string &, tiledb_datatype_t, FilterList &>(),
          py::keep_alive<1, 2>() /* Attribute keeps Context alive */)
      .def("name", &Attribute::name)
      .def("type", &Attribute::type)
      .def("cell_size", &Attribute::cell_size)
      .def("cell_val_num", &Attribute::cell_val_num)
      .def("set_cell_val_num", &Attribute::set_cell_val_num)
      //.def("set_fill_value", &Attribute::set_fill_value)
      //.def("get_fill_value", &Attribute::get_fill_value)
      .def("filter_list", &Attribute::filter_list)
      .def("set_filter_list", &Attribute::set_filter_list)
      .def("nullable", &Attribute::nullable)
      .def("set_nullable", &Attribute::set_nullable);
}

} // namespace libtiledbcpp
