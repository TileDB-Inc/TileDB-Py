#include <tiledb/tiledb> // C++

#include "common.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

// #pragma clang diagnostic ignored "-Wdeprecated-declarations"
// #pragma gcc diagnostic ignored "-Wdeprecated-declarations"

namespace libtiledbcpp {

using namespace tiledb;
using namespace std;
namespace py = pybind11;

// TODO factor and share with core.cc
void add_dim_range(Query &query, uint32_t dim_idx, py::tuple r) {
  if (py::len(r) == 0)
    return;
  else if (py::len(r) != 2)
    TPY_ERROR_LOC("Unexpected range len != 2");

  auto r0 = r[0];
  auto r1 = r[1];
  // no type-check here, because we might allow cast-conversion
  // if (r0.get_type() != r1.get_type())
  //    TPY_ERROR_LOC("Mismatched type");

  auto domain = query.array().schema().domain();
  auto dim = domain.dimension(dim_idx);

  auto tiledb_type = dim.type();

  try {
    switch (tiledb_type) {
    case TILEDB_INT32: {
      using T = int32_t;
      query.add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_INT64: {
      using T = int64_t;
      query.add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_INT8: {
      using T = int8_t;
      query.add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_UINT8: {
      using T = uint8_t;
      query.add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_INT16: {
      using T = int16_t;
      query.add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_UINT16: {
      using T = uint16_t;
      query.add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_UINT32: {
      using T = uint32_t;
      query.add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_UINT64: {
      using T = uint64_t;
      query.add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_FLOAT32: {
      using T = float;
      query.add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_FLOAT64: {
      using T = double;
      query.add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_STRING_ASCII:
    case TILEDB_STRING_UTF8:
    case TILEDB_CHAR: {
      if (!py::isinstance<py::none>(r0) != !py::isinstance<py::none>(r1)) {
        TPY_ERROR_LOC(
            "internal error: ranges must both be strings or (None, None)");
      } else if (!py::isinstance<py::none>(r0) &&
                 !py::isinstance<py::none>(r1) &&
                 !py::isinstance<py::str>(r0) && !py::isinstance<py::str>(r1) &&
                 !py::isinstance<py::bytes>(r0) &&
                 !py::isinstance<py::bytes>(r1)) {
        TPY_ERROR_LOC(
            "internal error: expected string type for var-length dim!");
      }

      if (!py::isinstance<py::none>(r0) && !py::isinstance<py::none>(r0))
        query.add_range(dim_idx, r0.cast<string>(), r1.cast<string>());

      break;
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
    case TILEDB_DATETIME_AS: {
#if TILEDB_VERSION_MAJOR >= 2 && TILEDB_VERSION_MINOR >= 3
    case TILEDB_TIME_HR:
    case TILEDB_TIME_MIN:
    case TILEDB_TIME_SEC:
    case TILEDB_TIME_MS:
    case TILEDB_TIME_US:
    case TILEDB_TIME_NS:
    case TILEDB_TIME_PS:
    case TILEDB_TIME_FS:
    case TILEDB_TIME_AS:
#endif
      py::dtype dtype = tdb_to_np_dtype(tiledb_type, 1);
      auto dt0 = py::isinstance<py::int_>(r0) ? r0 : r0.attr("astype")(dtype);
      auto dt1 = py::isinstance<py::int_>(r1) ? r1 : r1.attr("astype")(dtype);

      // TODO, this is suboptimal, should define pybind converter
      if (py::isinstance<py::int_>(dt0) && py::isinstance<py::int_>(dt1)) {
        query.add_range(dim_idx, py::cast<int64_t>(dt0),
                        py::cast<int64_t>(dt1));
      } else {
        auto darray = py::array(py::make_tuple(dt0, dt1));
        query.add_range(dim_idx, *(int64_t *)darray.data(0),
                        *(int64_t *)darray.data(1));
      }

      break;
    }
    default:
      TPY_ERROR_LOC("Unknown dim type conversion!");
    }
  } catch (py::cast_error &e) {
    (void)e;
    std::string msg = "Failed to cast dim range '" + (string)py::repr(r) +
                      "' to dim type " + tiledb::impl::type_to_str(tiledb_type);
    TPY_ERROR_LOC(msg);
  }
}

void init_query(py::module &m) {
  py::class_<tiledb::Query>(m, "Query")
      //.def(py::init<py::object, py::object, py::iterable, py::object,
      //              py::object, py::object>())
      .def(py::init<Context &, Array &, tiledb_query_type_t>(),
           py::keep_alive<1, 2>())
      .def(py::init<Context &, Array &>(),
           py::keep_alive<1, 2>()) // TODO keepalive for the Array as well?
      // TODO .def("ptr", [&]() -> py::capsule)

      .def_property_readonly("query_type", &Query::query_type)
      .def_property("layout", &Query::query_layout, &Query::set_layout)

      .def("set_condition", &Query::set_condition)
      // TODO .def("array") -> Array&
      .def("query_status", &Query::query_status)
      .def("has_results", &Query::has_results)
      .def("submit", &Query::submit, py::call_guard<py::gil_scoped_release>())
      // TODO .def("submit_async", Fn& callback)
      //.def("submit_async")
      .def("finalize", &Query::finalize)
      // TODO .def("result_buffer_elements", &Query::result_buffer_elements)
      // TODO .def("result_buffer_elements_nullable",
      // &Query::result_buffer_eleents_nullable)
      // TODO .def("add_range")
      // note the pointer cast here. overload_cast *does not work*
      //   https://github.com/pybind/pybind11/issues/1153
      //   https://pybind11.readthedocs.io/en/latest/classes.html#overloaded-methods
      .def("add_range",
           (Query & (Query::*)(const std::string &, const std::string &,
                               const std::string &)) &
               Query::add_range)
      //.def("add_range", (Query& (Query::*)(uint32_t, const std::string&, const
      // std::string&))&Query::add_range)
      .def("add_range",
           [](Query &q, uint32_t dim_idx, py::tuple range) {
             add_dim_range(q, dim_idx, range);
           })
      //.def("add_range", [](Query& q, std::string name, py::tuple range) {
      //  auto array = q.array();
      //  auto schema = array.schema();
      //  uint32_t dim_idx = schema.domain.dim(name).
      //  add_dim_range(q, dim_idx, range);
      //})
      .def("set_subarray",
           [](Query &q, py::array a) {
             // TODO check_type(a.dtype)
             // TODO check size == ndim * 2

             // Use the C API here because we are doing typecheck
             auto ctx = q.ctx(); // NB this requires libtiledb >=2.6
             ctx.handle_error(tiledb_query_set_subarray(
                 ctx.ptr().get(), q.ptr().get(), const_cast<void *>(a.data())));
           })
      //.def("add_range", ([](Query& this, uint32_t dim_idx, py::object,
      // py::object) {
      //    auto schema = this.array().schema();
      //    tiledb_datatype_t dim_type =
      //    schema.domain().dimension(dim_idx).type(); size_t dim_size =
      //    tiledb_datatype_size(dim_type); auto nptype =
      //})
      // TODO generic add_range implemented here
      //.def("set_data_buffer",
      //     (Query& (Query::*)(const std::string&, void*,
      //     uint64_t))&Query::set_data_buffer);
      .def("set_data_buffer",
           [](Query &q, std::string name, py::array a) {
             // TODO check_type(a.dtype)
             //  size_t item_size = a.itemsize();
             q.set_data_buffer(name, const_cast<void *>(a.data()), a.size());
           })
      .def("set_offsets_buffer",
           [](Query &q, std::string name, py::array a) {
             // TODO check_type(a.dtype)
             //  size_t item_size = a.itemsize();
             q.set_offsets_buffer(name, (uint64_t *)(a.data()), a.size());
           })
      .def("set_validity_buffer",
           [](Query &q, std::string name, py::array a) {
             // TODO check_type(a.dtype)
             //  size_t item_size = a.itemsize();
             q.set_validity_buffer(name, (uint8_t *)(a.data()), a.size());
           })
      .def("fragment_num", &Query::fragment_num)
      .def("fragment_uri", &Query::fragment_uri)
      /** hackery from another branch... */
      //.def("set_fragment_uri", &Query::set_fragment_uri)
      //.def("unset_buffer", &Query::unset_buffer)
      //.def("set_continuation", [](Query& q) {
      //  q.ctx().handle_error(
      //    tiledb_query_set_continuation(q.ctx().ptr().get(), q.ptr().get())
      //  );
      //})
      ;
}

} // namespace libtiledbcpp
