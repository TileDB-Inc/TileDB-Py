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

void init_filter(py::module &m) {
  py::class_<Filter>(m, "Filter")
      .def(py::init<const Context &, tiledb_filter_type_t>(),
           py::keep_alive<1, 2>())

      .def_property_readonly("_type", &Filter::filter_type)

      .def("_set_option",
           [](Filter &filter, Context ctx, tiledb_filter_option_t option,
              int32_t level) {
             // TODO check_type(a.dtype)
             // Use the C API here because we are doing typecheck
             ctx.handle_error(tiledb_filter_set_option(
                 ctx.ptr().get(), filter.ptr().get(), option, &level));
           })
      .def("_get_option",
           [](Filter &filter, Context ctx, tiledb_filter_option_t option) {
             int32_t level;
             // TODO check_type(a.dtype)
             // Use the C API here because we are doing typecheck
             ctx.handle_error(tiledb_filter_get_option(
                 ctx.ptr().get(), filter.ptr().get(), option, &level));
             return level;
           });

  py::class_<FilterList>(m, "FilterList")
      .def(py::init<const Context &>(), py::keep_alive<1, 2>())
      .def(py::init<const Context &, py::capsule>(), py::keep_alive<1, 2>())

      .def("__capsule__",
           [](FilterList &filterlist) {
             return py::capsule(filterlist.ptr().get(), "fl", nullptr);
           })

      .def_property("_chunksize", &FilterList::max_chunk_size,
                    &FilterList::set_max_chunk_size)

      .def("_nfilters", &FilterList::nfilters)
      .def("_filter", &FilterList::filter)
      .def("_add_filter", &FilterList::add_filter);
}

} // namespace libtiledbcpp
