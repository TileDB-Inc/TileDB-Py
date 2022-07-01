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
      .def(py::init([](const Context &ctx, const std::string &name,
                       py::dtype datatype, py::buffer domain,
                       py::buffer tile_extent) {
             tiledb_datatype_t dim_type;
             try {
               dim_type = np_to_tdb_dtype(datatype);
             } catch (const TileDBPyError &e) {
               throw py::type_error(e.what());
             }

             auto domain_info = domain.request();
             auto tile_extent_info = tile_extent.request();
             if (dim_type != TILEDB_STRING_ASCII) {
               if (!expect_buffer_nbytes(domain_info, dim_type, 2)) {
                 throw py::value_error(
                     "Unexpected type/shape for domain buffer!");
               }
               if (!expect_buffer_nbytes(tile_extent_info, dim_type, 1)) {
                 throw py::value_error(
                     "Unexpected type/shape for domain buffer!");
               }
             }

             const void *domain_data =
                 (dim_type != TILEDB_STRING_ASCII) ? domain_info.ptr : nullptr;
             const void *tile_extent_data = (dim_type != TILEDB_STRING_ASCII)
                                                ? tile_extent_info.ptr
                                                : nullptr;

             return std::make_unique<Dimension>(Dimension::create(
                 ctx, name, dim_type, domain_data, tile_extent_data));
           }),
           py::keep_alive<1, 2>())
      .def_property_readonly("_name", &Dimension::name)
      .def_property_readonly("domain", &Dimension::domain)
      // .def_property_readonly("tile", &Dimension::tile_extent)
      .def_property("_filters", &Dimension::filter_list,
                    &Dimension::set_filter_list)
      .def_property("_ncell", &Dimension::cell_val_num,
                    &Dimension::set_cell_val_num)
      // .def("_dtype", &Dimension::type)
      .def_property_readonly(
          "_dtype",
          [](Dimension dim) { return tdb_to_np_dtype(dim.type(), 1); })
      .def("_domain_to_str", &Dimension::domain_to_str);

  py::class_<tiledb::Domain>(m, "Domain")
      .def(py::init<Context &>(),
           py::keep_alive<1, 2>() /* ArraySchema keeps Context alive */)

      .def_property_readonly("_ncell",
                             [](Domain &dom) { return dom.cell_num(); })
      // .def_property_readonly("_ncell", &Domain::cell_num)
      .def_property_readonly("_dtype", &Domain::type)
      .def_property_readonly("_ndim", &Domain::ndim)
      .def_property_readonly("_dims", &Domain::dimensions)

      .def("_dim", py::overload_cast<unsigned>(&Domain::dimension, py::const_))
      .def("_dim", py::overload_cast<const std::string &>(&Domain::dimension,
                                                          py::const_))
      .def("_add_dim", &Domain::add_dimension);
}

} // namespace libtiledbcpp
