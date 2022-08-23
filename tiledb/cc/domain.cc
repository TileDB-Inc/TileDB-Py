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
  py::class_<Dimension>(m, "Dimension")
      .def(py::init([](const Context &ctx, const std::string &name,
                       py::dtype datatype, py::buffer domain,
                       py::buffer tile_extent) {
             tiledb_datatype_t dim_type;
             void *dim_dom = nullptr;
             void *dim_tile = nullptr;

             try {
               dim_type = np_to_tdb_dtype(datatype);
             } catch (const TileDBPyError &e) {
               throw py::type_error(e.what());
             }

             if (dim_type != TILEDB_STRING_ASCII) {
               py::buffer_info domain_info = domain.request();
               py::buffer_info tile_extent_info = tile_extent.request();

               dim_dom = domain_info.ptr;
               dim_tile = tile_extent_info.ptr;
             }

             return std::make_unique<Dimension>(
                 Dimension::create(ctx, name, dim_type, dim_dom, dim_tile));
           }),
           py::keep_alive<1, 2>())

      .def(py::init<const Context &, py::capsule>(), py::keep_alive<1, 2>(),
           py::keep_alive<1, 3>())

      .def_property_readonly("_name", &Dimension::name)

      .def_property_readonly(
          "_domain",
          [](Dimension &dim) {
            switch (dim.type()) {
            case TILEDB_UINT64: {
              auto dom = dim.domain<uint64_t>();
              return py::make_tuple(dom.first, dom.second);
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
              auto dom = dim.domain<int64_t>();
              return py::make_tuple(dom.first, dom.second);
            }
            case TILEDB_UINT32: {
              auto dom = dim.domain<uint32_t>();
              return py::make_tuple(dom.first, dom.second);
            }
            case TILEDB_INT32: {
              auto dom = dim.domain<int32_t>();
              return py::make_tuple(dom.first, dom.second);
            }
            case TILEDB_UINT16: {
              auto dom = dim.domain<uint16_t>();
              return py::make_tuple(dom.first, dom.second);
            }
            case TILEDB_INT16: {
              auto dom = dim.domain<int16_t>();
              return py::make_tuple(dom.first, dom.second);
            }
            case TILEDB_UINT8: {
              auto dom = dim.domain<uint8_t>();
              return py::make_tuple(dom.first, dom.second);
            }
            case TILEDB_INT8: {
              auto dom = dim.domain<int8_t>();
              return py::make_tuple(dom.first, dom.second);
            }
            case TILEDB_FLOAT64: {
              auto dom = dim.domain<double>();
              return py::make_tuple(dom.first, dom.second);
            }
            case TILEDB_FLOAT32: {
              auto dom = dim.domain<float>();
              return py::make_tuple(dom.first, dom.second);
            }
            default:
              TPY_ERROR_LOC("Unsupported dtype for Dimension's domain");
            }
          })

      .def_property_readonly(
          "_tile",
          [](Dimension &dim) {
            switch (dim.type()) {
            case TILEDB_UINT64: {
              return py::cast(dim.tile_extent<uint64_t>());
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
              return py::cast(dim.tile_extent<int64_t>());
            }
            case TILEDB_UINT32: {
              return py::cast(dim.tile_extent<uint32_t>());
            }
            case TILEDB_INT32: {
              return py::cast(dim.tile_extent<int32_t>());
            }
            case TILEDB_UINT16: {
              return py::cast(dim.tile_extent<uint16_t>());
            }
            case TILEDB_INT16: {
              return py::cast(dim.tile_extent<int16_t>());
            }
            case TILEDB_UINT8: {
              return py::cast(dim.tile_extent<uint8_t>());
            }
            case TILEDB_INT8: {
              return py::cast(dim.tile_extent<int8_t>());
            }
            case TILEDB_FLOAT64: {
              return py::cast(dim.tile_extent<double>());
            }
            case TILEDB_FLOAT32: {
              return py::cast(dim.tile_extent<float>());
            }
            default:
              TPY_ERROR_LOC("Unsupported dtype  for Dimension's tile extent");
            }
          })

      .def_property("_filters", &Dimension::filter_list,
                    &Dimension::set_filter_list)

      .def_property("_ncell", &Dimension::cell_val_num,
                    &Dimension::set_cell_val_num)

      .def_property_readonly("_tiledb_dtype", &Dimension::type)

      .def_property_readonly(
          "_numpy_dtype",
          [](Dimension &dim) { return tdb_to_np_dtype(dim.type(), 1); })

      .def("_domain_to_str", &Dimension::domain_to_str);

  py::class_<Domain>(m, "Domain")
      .def(py::init<Context &>(), py::keep_alive<1, 2>())

      .def_property_readonly("_ncell",
                             [](Domain &dom) { return dom.cell_num(); })

      .def_property_readonly("_tiledb_dtype", &Domain::type)

      .def_property_readonly(
          "_numpy_dtype",
          [](Domain &dom) { return tdb_to_np_dtype(dom.type(), 1); })

      .def_property_readonly("_ndim", &Domain::ndim)

      .def_property_readonly("_dims", &Domain::dimensions)

      .def("_dim", py::overload_cast<unsigned>(&Domain::dimension, py::const_))
      .def("_dim", py::overload_cast<const std::string &>(&Domain::dimension,
                                                          py::const_))

      .def("_has_dim", &Domain::has_dimension)

      .def("_add_dim", &Domain::add_dimension, py::keep_alive<1, 2>())

      .def("_dump", [](Domain &dom) { dom.dump(); });
}

} // namespace libtiledbcpp
