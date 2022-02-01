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

void init_domain(py::module &m) {
  py::class_<tiledb::Dimension>(m, "Dimension")
      .def("create",
           [](const Context &ctx, const std::string &name,
              tiledb_datatype_t datatype, py::buffer range, py::buffer extent) {
             auto range_info = range.request();
             auto extent_info = extent.request();
             if (datatype != TILEDB_STRING_ASCII) {
               if (!expect_buffer_nbytes(range_info, datatype, 2)) {
                 throw py::value_error(
                     "Unexpected type/shape for range buffer!");
               }
               if (!expect_buffer_nbytes(extent_info, datatype, 1)) {
                 throw py::value_error(
                     "Unexpected type/shape for range buffer!");
               }
             }

             const void *range_data =
                 (datatype != TILEDB_STRING_ASCII) ? range_info.ptr : nullptr;
             const void *extent_data =
                 (datatype != TILEDB_STRING_ASCII) ? extent_info.ptr : nullptr;

             return std::make_unique<Dimension>(Dimension::create(
                 ctx, name, datatype, range_data, extent_data));
           })
      .def_property_readonly("name", &Dimension::name)
      // .def_property_readonly("domain", &Dimension::domain)
      // .def_property_readonly("tile", &Dimension::tile_extent)
      .def_property("filters", &Dimension::filter_list,
                    &Dimension::set_filter_list)
      .def_property("ncell", &Dimension::cell_val_num,
                    &Dimension::set_cell_val_num)
      .def("tiledb_datatype", &Dimension::type)
      // TODO needs numpy <> tiledb type and void*+(type,size) -> numpy
      // translators
      .def("domain_to_str", &Dimension::domain_to_str);

  py::class_<tiledb::Domain>(m, "Domain")
      .def(py::init<Context &>(),
           py::keep_alive<1, 2>() /* ArraySchema keeps Context alive */)

      .def_property_readonly("ncell",
                             [](Domain &dom) { return dom.cell_num(); })
      .def_property_readonly("dtype", &Domain::type)
      .def_property_readonly("ndim", &Domain::ndim)
      .def_property_readonly("dims", &Domain::dimensions)

      .def("dim", py::overload_cast<unsigned>(&Domain::dimension, py::const_))
      .def("dim", py::overload_cast<const std::string &>(&Domain::dimension,
                                                         py::const_))
      .def("add_dim", &Domain::add_dimension);
}

} // namespace libtiledbcpp
