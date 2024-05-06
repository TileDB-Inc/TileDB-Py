#include <tiledb/tiledb.h>              // for enums
#include <tiledb/tiledb>                // C++
#include <tiledb/tiledb_experimental.h> // for `tiledb_subarray_has_label_range`
#include <tiledb/tiledb_experimental>   // C++

#include "common.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace libtiledbcpp {

using namespace tiledb;
namespace py = pybind11;

template <typename T> struct SubarrayDimensionManipulator {

  static void copy(Subarray &subarray, Subarray &original, uint32_t dim_idx) {
    for (uint64_t range_idx{0}; range_idx < original.range_num(dim_idx);
         ++range_idx) {
      std::array<T, 3> range = original.range<T>(dim_idx, range_idx);
      subarray.add_range(dim_idx, range[0], range[1], range[2]);
    }
  }

  static py::ssize_t length(Subarray &subarray, uint32_t dim_idx) {
    uint64_t length = 0;
    for (uint64_t range_idx{0}; range_idx < subarray.range_num(dim_idx);
         ++range_idx) {
      std::array<T, 3> range = subarray.range<T>(dim_idx, range_idx);
      if (range[2] != 0 && range[1] != 1) {
        throw TileDBPyError("Support for getting the lenght of ranges with a "
                            "stride is not yet implemented.");
      }

      auto range_length = static_cast<uint64_t>(range[1] - range[0]);
      if (length > std::numeric_limits<uint64_t>::max() - range_length - 1) {
        throw TileDBPyError("Overflow error computing subarray shape");
      }
      length += range_length + 1;
    }
    if (length > PY_SSIZE_T_MAX) {
      throw TileDBPyError("Overflow error computing subarray shape");
    }
    return Py_SAFE_DOWNCAST(length, Py_ssize_t, uint64_t);
  }
};

template <> struct SubarrayDimensionManipulator<std::string> {

  static void copy(Subarray &subarray, Subarray &original, uint32_t dim_idx) {
    for (uint64_t range_idx{0}; range_idx < original.range_num(dim_idx);
         ++range_idx) {
      std::array<std::string, 2> range = original.range(dim_idx, range_idx);
      subarray.add_range(dim_idx, range[0], range[1]);
    }
  }

  static uint64_t length(Subarray &, uint32_t) {
    throw TileDBPyError(
        "Getting length of ranges is not supported on string dimensions.");
  }
};

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
    case TILEDB_TIME_HR:
    case TILEDB_TIME_MIN:
    case TILEDB_TIME_SEC:
    case TILEDB_TIME_MS:
    case TILEDB_TIME_US:
    case TILEDB_TIME_NS:
    case TILEDB_TIME_PS:
    case TILEDB_TIME_FS:
    case TILEDB_TIME_AS:
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

void copy_ranges_on_dim(Subarray &subarray, Subarray original,
                        uint32_t dim_idx) {

  auto tiledb_type =
      subarray.array().schema().domain().dimension(dim_idx).type();

  switch (tiledb_type) {
  case TILEDB_INT32: {
    using T = int32_t;
    SubarrayDimensionManipulator<T>::copy(subarray, original, dim_idx);
    break;
  }
  case TILEDB_INT64: {
    using T = int64_t;
    SubarrayDimensionManipulator<T>::copy(subarray, original, dim_idx);
    break;
  }
  case TILEDB_INT8: {
    using T = int8_t;
    SubarrayDimensionManipulator<T>::copy(subarray, original, dim_idx);
    break;
  }
  case TILEDB_UINT8: {
    using T = uint8_t;
    SubarrayDimensionManipulator<T>::copy(subarray, original, dim_idx);
    break;
  }
  case TILEDB_INT16: {
    using T = int16_t;
    SubarrayDimensionManipulator<T>::copy(subarray, original, dim_idx);
    break;
  }
  case TILEDB_UINT16: {
    using T = uint16_t;
    SubarrayDimensionManipulator<T>::copy(subarray, original, dim_idx);
    break;
  }
  case TILEDB_UINT32: {
    using T = uint32_t;
    SubarrayDimensionManipulator<T>::copy(subarray, original, dim_idx);
    break;
  }
  case TILEDB_UINT64: {
    using T = uint64_t;
    SubarrayDimensionManipulator<T>::copy(subarray, original, dim_idx);
    break;
  }
  case TILEDB_FLOAT32: {
    using T = float;
    SubarrayDimensionManipulator<T>::copy(subarray, original, dim_idx);
    break;
  }
  case TILEDB_FLOAT64: {
    using T = double;
    SubarrayDimensionManipulator<T>::copy(subarray, original, dim_idx);
    break;
  }
  case TILEDB_STRING_ASCII:
  case TILEDB_STRING_UTF8:
  case TILEDB_CHAR: {
    using T = std::string;
    SubarrayDimensionManipulator<T>::copy(subarray, original, dim_idx);
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
  case TILEDB_TIME_HR:
  case TILEDB_TIME_MIN:
  case TILEDB_TIME_SEC:
  case TILEDB_TIME_MS:
  case TILEDB_TIME_US:
  case TILEDB_TIME_NS:
  case TILEDB_TIME_PS:
  case TILEDB_TIME_FS:
  case TILEDB_TIME_AS:
    using T = int64_t;
    SubarrayDimensionManipulator<T>::copy(subarray, original, dim_idx);
    break;
  }
  default:
    TPY_ERROR_LOC("Unknown dim type conversion!");
  }
}

py::ssize_t length_ranges(Subarray &subarray, uint32_t dim_idx) {

  auto tiledb_type =
      subarray.array().schema().domain().dimension(dim_idx).type();

  switch (tiledb_type) {
  case TILEDB_INT32: {
    using T = int32_t;
    return SubarrayDimensionManipulator<T>::length(subarray, dim_idx);
  }
  case TILEDB_INT64: {
    using T = int64_t;
    return SubarrayDimensionManipulator<T>::length(subarray, dim_idx);
  }
  case TILEDB_INT8: {
    using T = int8_t;
    return SubarrayDimensionManipulator<T>::length(subarray, dim_idx);
  }
  case TILEDB_UINT8: {
    using T = uint8_t;
    return SubarrayDimensionManipulator<T>::length(subarray, dim_idx);
  }
  case TILEDB_INT16: {
    using T = int16_t;
    return SubarrayDimensionManipulator<T>::length(subarray, dim_idx);
  }
  case TILEDB_UINT16: {
    using T = uint16_t;
    return SubarrayDimensionManipulator<T>::length(subarray, dim_idx);
  }
  case TILEDB_UINT32: {
    using T = uint32_t;
    return SubarrayDimensionManipulator<T>::length(subarray, dim_idx);
  }
  case TILEDB_UINT64: {
    using T = uint64_t;
    return SubarrayDimensionManipulator<T>::length(subarray, dim_idx);
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
  case TILEDB_TIME_HR:
  case TILEDB_TIME_MIN:
  case TILEDB_TIME_SEC:
  case TILEDB_TIME_MS:
  case TILEDB_TIME_US:
  case TILEDB_TIME_NS:
  case TILEDB_TIME_PS:
  case TILEDB_TIME_FS:
  case TILEDB_TIME_AS:
    using T = int64_t;
    return SubarrayDimensionManipulator<T>::length(subarray, dim_idx);
  }
  default:
    TPY_ERROR_LOC("Dimension length not supported on a dimension with the "
                  "given datatype.");
  }
}

void add_dim_point_ranges(const Context &ctx, Subarray &subarray,
                          uint32_t dim_idx, pybind11::handle dim_range) {

  // Cast range object to appropriately typed py::array.
  auto tiledb_type =
      subarray.array().schema().domain().dimension(dim_idx).type();
  py::dtype dtype = tdb_to_np_dtype(tiledb_type, 1);
  py::array ranges = dim_range.attr("astype")(dtype);

  // Set point ranges using C-API.
  tiledb_ctx_t *c_ctx = ctx.ptr().get();
  tiledb_subarray_t *c_subarray = subarray.ptr().get();
  ctx.handle_error(tiledb_subarray_add_point_ranges(
      c_ctx, c_subarray, dim_idx, (void *)ranges.data(), ranges.size()));
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

      if (!py::isinstance<py::none>(r0) && !py::isinstance<py::none>(r0)) {
        std::string r0_string = r0.cast<std::string>();
        std::string r1_string = r1.cast<std::string>();
        SubarrayExperimental::add_label_range(ctx, subarray, label_name,
                                              r0_string, r1_string);
      }
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
    case TILEDB_TIME_HR:
    case TILEDB_TIME_MIN:
    case TILEDB_TIME_SEC:
    case TILEDB_TIME_MS:
    case TILEDB_TIME_US:
    case TILEDB_TIME_NS:
    case TILEDB_TIME_PS:
    case TILEDB_TIME_FS:
    case TILEDB_TIME_AS:
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
      TPY_ERROR_LOC("Unknown dimension label type conversion!");
    }
  } catch (py::cast_error &e) {
    (void)e;
    std::string msg = "Failed to cast label range '" +
                      (std::string)py::repr(r) + "' to label type " +
                      tiledb::impl::type_to_str(tiledb_type);
    TPY_ERROR_LOC(msg);
  }
}

bool has_label_range(const Context &ctx, Subarray &subarray, uint32_t dim_idx) {
  int32_t has_label;
  auto rc = tiledb_subarray_has_label_ranges(
      ctx.ptr().get(), subarray.ptr().get(), dim_idx, &has_label);
  if (rc == TILEDB_ERR) {
    throw TileDBError("Failed to check dimension for label ranges");
  }
  return has_label == 1;
}

void init_subarray(py::module &m) {
  py::class_<tiledb::Subarray>(m, "Subarray")
      .def(py::init<Subarray>())

      .def(py::init<const Context &, const Array &>(),
           py::keep_alive<1, 2>() /* Keep context alive. */,
           py::keep_alive<1, 3>() /* Keep array alive. */)

      .def("__capsule__",
           [](Subarray &subarray) {
             return py::capsule(subarray.ptr().get(), "subarray");
           })

      .def("_add_dim_range",
           [](Subarray &subarray, uint32_t dim_idx, py::tuple range) {
             add_dim_range(subarray, dim_idx, range);
           })

      .def("_add_label_range",
           [](Subarray &subarray, const Context &ctx,
              const std::string &label_name, py::tuple range) {
             add_label_range(ctx, subarray, label_name, range);
           })

      .def("_add_ranges_bulk",
           [](Subarray &subarray, const Context &ctx, py::iterable ranges) {
             uint32_t dim_idx = 0;
             for (auto dim_range : ranges) {
               if (py::isinstance<py::array>(dim_range)) {
                 add_dim_point_ranges(ctx, subarray, dim_idx, dim_range);
               } else {
                 py::tuple dim_range_iter = dim_range.cast<py::iterable>();
                 for (auto r : dim_range_iter) {
                   py::tuple range_tuple = r.cast<py::tuple>();
                   add_dim_range(subarray, dim_idx, range_tuple);
                 }
               }
               dim_idx++;
             }
           })

      .def("_add_dim_point_ranges",
           [](Subarray &subarray, const Context &ctx, uint32_t dim_idx,
              pybind11::handle dim_range) {
             add_dim_point_ranges(ctx, subarray, dim_idx, dim_range);
           })

      .def("_add_ranges",
           [](Subarray &subarray, const Context &ctx, py::iterable ranges) {
             uint32_t dim_idx = 0;
             for (auto dim_range : ranges) {
               py::tuple dim_range_iter = dim_range.cast<py::iterable>();
               for (auto r : dim_range_iter) {
                 py::tuple r_tuple = r.cast<py::tuple>();
                 add_dim_range(subarray, dim_idx, r_tuple);
               }
               dim_idx++;
             }
           })

      .def("_add_label_ranges",
           [](Subarray &subarray, const Context &ctx, py::iterable ranges) {
             py::dict label_ranges = ranges.cast<py::dict>();
             for (std::pair<py::handle, py::handle> pair : label_ranges) {
               py::str label_name = pair.first.cast<py::str>();
               py::tuple label_range_iter = pair.second.cast<py::iterable>();
               for (auto r : label_range_iter) {
                 py::tuple r_tuple = r.cast<py::tuple>();
                 add_label_range(ctx, subarray, label_name, r_tuple);
               }
             }
           })

      .def("_has_label_range",
           [](Subarray &subarray, const Context &ctx, uint32_t dim_idx) {
             return has_label_range(ctx, subarray, dim_idx);
           })

      .def("copy_ranges",
           [](Subarray &subarray, Subarray &original, py::iterable dims) {
             for (auto dim_idx : dims) {
               copy_ranges_on_dim(subarray, original, dim_idx.cast<uint32_t>());
             }
           })

      .def("_range_num", py::overload_cast<const std::string &>(
                             &Subarray::range_num, py::const_))

      .def("_range_num",
           py::overload_cast<unsigned>(&Subarray::range_num, py::const_))

      .def("_label_range_num",
           [](Subarray &subarray, const Context &ctx,
              const std::string &label_name) {
             return SubarrayExperimental::label_range_num(ctx, subarray,
                                                          label_name);
           })

      .def("_shape",
           [](Subarray &subarray, const Context &ctx) {
             auto ndim = subarray.array().schema().domain().ndim();
             // Create numpy array and get pointer to data.
             py::array_t<py::ssize_t> shape(ndim);
             py::buffer_info shape_result = shape.request();
             py::ssize_t *shape_ptr =
                 static_cast<py::ssize_t *>(shape_result.ptr);
             // Set size for each dimension.
             for (uint32_t dim_idx{0}; dim_idx < ndim; ++dim_idx) {
               shape_ptr[dim_idx] = length_ranges(subarray, dim_idx);
             }
             return shape;
           })

      // End definitions.
      ;
}

} // namespace libtiledbcpp
