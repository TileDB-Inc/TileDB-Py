#include <tiledb/tiledb.h>            // for enums
#include <tiledb/tiledb>              // C++
#include <tiledb/tiledb_experimental> // C++

#include "common.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace libtiledbcpp {

using namespace tiledb;
namespace py = pybind11;

void add_dim_range(Subarray &subarray, uint32_t dim_idx, py::tuple r) {
  if (py::len(r) == 0)
    return;
  else if (py::len(r) != 2)
    TPY_ERROR_LOC("Unexpected range len != 2");

  auto r0 = r[0];
  auto r1 = r[1];

  auto tiledb_type =
      subarray.array().schema().domain().dimension(dim_idx).type();

  try {
    switch (tiledb_type) {
    case TILEDB_INT32: {
      using T = int32_t;
      subarray.add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_INT64: {
      using T = int64_t;
      subarray.add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_INT8: {
      using T = int8_t;
      subarray.add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_UINT8: {
      using T = uint8_t;
      subarray.add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_INT16: {
      using T = int16_t;
      subarray.add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_UINT16: {
      using T = uint16_t;
      subarray.add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_UINT32: {
      using T = uint32_t;
      subarray.add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_UINT64: {
      using T = uint64_t;
      subarray.add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_FLOAT32: {
      using T = float;
      subarray.add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_FLOAT64: {
      using T = double;
      subarray.add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
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
        subarray.add_range(dim_idx, r0.cast<std::string>(),
                           r1.cast<std::string>());

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
        subarray.add_range(dim_idx, py::cast<int64_t>(dt0),
                           py::cast<int64_t>(dt1));
      } else {
        auto darray = py::array(py::make_tuple(dt0, dt1));
        subarray.add_range(dim_idx, *(int64_t *)darray.data(0),
                           *(int64_t *)darray.data(1));
      }

      break;
    }
    default:
      TPY_ERROR_LOC("Unknown dim type conversion!");
    }
  } catch (py::cast_error &e) {
    (void)e;
    std::string msg = "Failed to cast dim range '" + (std::string)py::repr(r) +
                      "' to dim type " + tiledb::impl::type_to_str(tiledb_type);
    TPY_ERROR_LOC(msg);
  }
}

void add_label_range(const Context &ctx, Subarray &subarray,
                     const std::string &label_name, py::tuple r) {
  if (py::len(r) == 0)
    return;
  else if (py::len(r) != 2)
    TPY_ERROR_LOC("Unexpected range len != 2");

  auto r0 = r[0];
  auto r1 = r[1];

  auto tiledb_type = ArraySchemaExperimental::dimension_label(
                         ctx, subarray.array().schema(), label_name)
                         .label_type();

  try {
    switch (tiledb_type) {
    case TILEDB_INT32: {
      using T = int32_t;
      SubarrayExperimental::add_label_range(ctx, subarray, label_name,
                                            r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_INT64: {
      using T = int64_t;
      SubarrayExperimental::add_label_range(ctx, subarray, label_name,
                                            r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_INT8: {
      using T = int8_t;
      SubarrayExperimental::add_label_range(ctx, subarray, label_name,
                                            r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_UINT8: {
      using T = uint8_t;
      SubarrayExperimental::add_label_range(ctx, subarray, label_name,
                                            r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_INT16: {
      using T = int16_t;
      SubarrayExperimental::add_label_range(ctx, subarray, label_name,
                                            r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_UINT16: {
      using T = uint16_t;
      SubarrayExperimental::add_label_range(ctx, subarray, label_name,
                                            r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_UINT32: {
      using T = uint32_t;
      SubarrayExperimental::add_label_range(ctx, subarray, label_name,
                                            r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_UINT64: {
      using T = uint64_t;
      SubarrayExperimental::add_label_range(ctx, subarray, label_name,
                                            r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_FLOAT32: {
      using T = float;
      SubarrayExperimental::add_label_range(ctx, subarray, label_name,
                                            r0.cast<T>(), r1.cast<T>());
      break;
    }
    case TILEDB_FLOAT64: {
      using T = double;
      SubarrayExperimental::add_label_range(ctx, subarray, label_name,
                                            r0.cast<T>(), r1.cast<T>());
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
            "internal error: expected string type for var-length label!");
      }

      /*
       * TODO: FIX
      if (!py::isinstance<py::none>(r0) && !py::isinstance<py::none>(r0)) {
        std::string r0_string = r0.cast<string>();
        std::string r1_string = r1.cast<string>();
        SubarrayExperimental::add_label_range(ctx_, subarray, label_name,
                                              r0_string, r1_string);
      }
      break;
      */
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

      if (py::isinstance<py::int_>(dt0) && py::isinstance<py::int_>(dt1)) {
        SubarrayExperimental::add_label_range(ctx, subarray, label_name,
                                              py::cast<int64_t>(dt0),
                                              py::cast<int64_t>(dt1));
      } else {
        auto darray = py::array(py::make_tuple(dt0, dt1));
        SubarrayExperimental::add_label_range(ctx, subarray, label_name,
                                              *(int64_t *)darray.data(0),
                                              *(int64_t *)darray.data(1));
      }

      break;
    }
    default:
      TPY_ERROR_LOC("Unknown dim type conversion!");
    }
  } catch (py::cast_error &e) {
    (void)e;
    std::string msg = "Failed to cast label range '" +
                      (std::string)py::repr(r) + "' to label type " +
                      tiledb::impl::type_to_str(tiledb_type);
    TPY_ERROR_LOC(msg);
  }
}

void init_subarray(py::module &m) {
  py::class_<tiledb::Subarray>(m, "Subarray")
      .def(py::init<Context &, Array &>())

      .def("add_range",
           [](Subarray &subarray, uint32_t dim_idx, py::tuple range) {
             add_dim_range(subarray, dim_idx, range);
           })

      // Note the static cast here. overload_cast *does not work*
      //   https://github.com/pybind/pybind11/issues/1153
      //   https://pybind11.readthedocs.io/en/latest/classes.html#overloaded-methods
      .def("add_range",
           static_cast<Subarray &(
               Subarray::*)(const std::string &, const std::string &,
                            const std::string &)>(&Subarray::add_range))

      .def("add_label_range",
           [](Subarray &subarray, const Context &ctx,
              const std::string &label_name, py::tuple range) {
             add_label_range(ctx, subarray, label_name, range);
           })

      .def("range_num", py::overload_cast<const std::string &>(
                            &Subarray::range_num, py::const_))

      .def("range_num",
           py::overload_cast<unsigned>(&Subarray::range_num, py::const_))

      // End definitions.
      ;
}

} // namespace libtiledbcpp
