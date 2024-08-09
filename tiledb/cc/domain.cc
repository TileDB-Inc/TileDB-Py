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
      .def(py::init<Dimension>())

      .def(py::init([](const Context &ctx, const std::string &name,
                       tiledb_datatype_t datatype, py::object domain,
                       py::object tile_extent) {
             void *dim_dom = nullptr;
             void *dim_tile = nullptr;

             if (!domain.is_none()) {
               py::buffer domain_buffer = py::buffer(domain);
               py::buffer_info domain_info = domain_buffer.request();
               dim_dom = domain_info.ptr;
             }

             if (!tile_extent.is_none()) {
               py::buffer tile_buffer = py::buffer(tile_extent);
               py::buffer_info tile_extent_info = tile_buffer.request();
               dim_tile = tile_extent_info.ptr;
             }

             return std::make_unique<Dimension>(
                 Dimension::create(ctx, name, datatype, dim_dom, dim_tile));
           }),
           py::keep_alive<1, 2>())

      .def(py::init<const Context &, py::capsule>(), py::keep_alive<1, 2>())

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
            case TILEDB_DATETIME_MONTH:
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
            case TILEDB_STRING_ASCII: {
              return py::make_tuple("", "");
            }
            default:
              TPY_ERROR_LOC("Unsupported dtype for Dimension's domain");
            }
          })

      .def_property_readonly(
          "_tile",
          [](Dimension &dim) -> py::object {
            switch (dim.type()) {
            case TILEDB_UINT64: {
              return py::cast(dim.tile_extent<uint64_t>());
            }
            case TILEDB_DATETIME_YEAR:
            case TILEDB_DATETIME_MONTH:
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
            case TILEDB_STRING_ASCII: {
              return py::none();
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

      .def("_domain_to_str", &Dimension::domain_to_str);

  py::class_<Domain>(m, "Domain")
      .def(py::init<Domain>())

      .def(py::init<Context &>())

      .def(py::init<const Context &, py::capsule>())

      .def("__capsule__",
           [](Domain &dom) { return py::capsule(dom.ptr().get(), "dom"); })

      .def_property_readonly("_ncell",
                             [](Domain &dom) { return dom.cell_num(); })

      .def_property_readonly("_tiledb_dtype", &Domain::type)

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
