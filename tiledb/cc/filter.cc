#include <tiledb/tiledb>
// #include <tiledb/tiledb_experimental.h> // for filter_dump, not yet available

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "common.h"

namespace libtiledbcpp {

using namespace tiledb;
using namespace tiledbpy::common;
namespace py = pybind11;

void init_filter(py::module &m) {
  py::class_<Filter>(m, "Filter")
      .def(py::init<const Context &, tiledb_filter_type_t>())

      .def_property_readonly("_type", &Filter::filter_type)

      .def("_set_option",
           [](Filter &filter, Context ctx, tiledb_filter_option_t option,
              py::object value) {
             switch (option) {
             case TILEDB_COMPRESSION_LEVEL:
               filter.set_option(option, value.cast<int32_t>());
               break;
             case TILEDB_BIT_WIDTH_MAX_WINDOW:
             case TILEDB_POSITIVE_DELTA_MAX_WINDOW:
               filter.set_option(option, value.cast<uint32_t>());
               break;
             case TILEDB_SCALE_FLOAT_BYTEWIDTH:
               filter.set_option(option, value.cast<uint64_t>());
               break;
             case TILEDB_SCALE_FLOAT_FACTOR:
             case TILEDB_SCALE_FLOAT_OFFSET:
               filter.set_option(option, value.cast<double>());
               break;
             case TILEDB_WEBP_INPUT_FORMAT:
               filter.set_option(option, value.cast<uint8_t>());
               break;
             case TILEDB_WEBP_QUALITY:
               filter.set_option(option, value.cast<float>());
               break;
             case TILEDB_WEBP_LOSSLESS:
               filter.set_option(option, value.cast<uint8_t>());
               break;
             case TILEDB_COMPRESSION_REINTERPRET_DATATYPE:
               filter.set_option(option, value.cast<uint8_t>());
               break;
             default:
               TPY_ERROR_LOC("Unrecognized filter option to _set_option");
             }
           })

      .def("_get_option",
           [](Filter &filter, Context ctx,
              tiledb_filter_option_t option) -> py::object {
             switch (option) {
             case TILEDB_COMPRESSION_LEVEL: {
               int32_t value;
               filter.get_option(option, &value);
               return py::cast(value);
             }
             case TILEDB_BIT_WIDTH_MAX_WINDOW:
             case TILEDB_POSITIVE_DELTA_MAX_WINDOW: {
               uint32_t value;
               filter.get_option(option, &value);
               return py::cast(value);
             }
             case TILEDB_SCALE_FLOAT_BYTEWIDTH: {
               uint64_t value;
               filter.get_option(option, &value);
               return py::cast(value);
             }
             case TILEDB_SCALE_FLOAT_FACTOR:
             case TILEDB_SCALE_FLOAT_OFFSET: {
               double value;
               filter.get_option(option, &value);
               return py::cast(value);
             }
             case TILEDB_WEBP_INPUT_FORMAT: {
               uint8_t value;
               filter.get_option(option, &value);
               return py::cast(value);
             }
             case TILEDB_WEBP_QUALITY: {
               float value;
               filter.get_option(option, &value);
               return py::cast(value);
             }
             case TILEDB_WEBP_LOSSLESS: {
               uint8_t value;
               filter.get_option(option, &value);
               return py::cast(value);
             }
             case TILEDB_COMPRESSION_REINTERPRET_DATATYPE: {
               auto value = filter.get_option<uint8_t>(option);
               return py::cast(static_cast<tiledb_datatype_t>(value));
             }
             default:
               TPY_ERROR_LOC("Unrecognized filter option to _get_option");
             }
           });
  /* needs core patch */
  /*
   .def("_dump", [](Filter &filter, Context ctx) {
          // TODO check_type(a.dtype)
          // Use the C API here because we are doing typecheck

          ctx.handle_error(tiledb_filter_dump(
              ctx.ptr().get(), filter.ptr().get()));
        });
  */

  py::class_<FilterList>(m, "FilterList")
      .def(py::init<FilterList>())
      .def(py::init<const Context &>())
      .def(py::init<const Context &, py::capsule>())

      .def("__capsule__",
           [](FilterList &filterlist) {
             return py::capsule(filterlist.ptr().get(), "fl");
           })

      .def_property("_chunksize", &FilterList::max_chunk_size,
                    &FilterList::set_max_chunk_size)

      .def("_nfilters", &FilterList::nfilters)
      .def("_filter", &FilterList::filter)
      .def("_add_filter", &FilterList::add_filter);
} // namespace libtiledbcpp

} // namespace libtiledbcpp
