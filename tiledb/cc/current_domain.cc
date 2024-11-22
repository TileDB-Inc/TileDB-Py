#include <tiledb/tiledb>
#include <tiledb/tiledb_experimental>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>
#include <variant>

#include "common.h"

namespace libtiledbcpp {

using namespace tiledb;
using namespace tiledbpy::common;
namespace py = pybind11;

void init_current_domain(py::module &m) {
#if TILEDB_VERSION_MAJOR >= 2 && TILEDB_VERSION_MINOR >= 25
  py::class_<NDRectangle>(m, "NDRectangle")
      .def(py::init<NDRectangle>())

      .def(py::init<const Context &, const Domain &>())

      .def("_set_range",
           py::overload_cast<const std::string &, uint64_t, uint64_t>(
               &NDRectangle::set_range<uint64_t>),
           py::arg("dim_name"), py::arg("start"), py::arg("end"))
      .def("_set_range",
           py::overload_cast<const std::string &, int64_t, int64_t>(
               &NDRectangle::set_range<int64_t>),
           py::arg("dim_name"), py::arg("start"), py::arg("end"))
      .def("_set_range",
           py::overload_cast<const std::string &, uint32_t, uint32_t>(
               &NDRectangle::set_range<uint32_t>),
           py::arg("dim_name"), py::arg("start"), py::arg("end"))
      .def("_set_range",
           py::overload_cast<const std::string &, int32_t, int32_t>(
               &NDRectangle::set_range<int32_t>),
           py::arg("dim_name"), py::arg("start"), py::arg("end"))
      .def("_set_range",
           py::overload_cast<const std::string &, uint16_t, uint16_t>(
               &NDRectangle::set_range<uint16_t>),
           py::arg("dim_name"), py::arg("start"), py::arg("end"))
      .def("_set_range",
           py::overload_cast<const std::string &, int16_t, int16_t>(
               &NDRectangle::set_range<int16_t>),
           py::arg("dim_name"), py::arg("start"), py::arg("end"))
      .def("_set_range",
           py::overload_cast<const std::string &, uint8_t, uint8_t>(
               &NDRectangle::set_range<uint8_t>),
           py::arg("dim_name"), py::arg("start"), py::arg("end"))
      .def("_set_range",
           py::overload_cast<const std::string &, int8_t, int8_t>(
               &NDRectangle::set_range<int8_t>),
           py::arg("dim_name"), py::arg("start"), py::arg("end"))
      .def("_set_range",
           py::overload_cast<const std::string &, double, double>(
               &NDRectangle::set_range<double>),
           py::arg("dim_name"), py::arg("start"), py::arg("end"))
      .def("_set_range",
           py::overload_cast<const std::string &, float, float>(
               &NDRectangle::set_range<float>),
           py::arg("dim_name"), py::arg("start"), py::arg("end"))
      .def(
          "_set_range",
          [](NDRectangle &ndrect, const std::string &dim_name,
             const std::string &start, const std::string &end) {
            return ndrect.set_range(dim_name, start, end);
          },
          py::arg("dim_name"), py::arg("start"), py::arg("end"))
      .def("_set_range",
           py::overload_cast<uint32_t, uint64_t, uint64_t>(
               &NDRectangle::set_range<uint64_t>),
           py::arg("dim_idx"), py::arg("start"), py::arg("end"))
      .def("_set_range",
           py::overload_cast<uint32_t, int64_t, int64_t>(
               &NDRectangle::set_range<int64_t>),
           py::arg("dim_idx"), py::arg("start"), py::arg("end"))
      .def("_set_range",
           py::overload_cast<uint32_t, uint32_t, uint32_t>(
               &NDRectangle::set_range<uint32_t>),
           py::arg("dim_idx"), py::arg("start"), py::arg("end"))
      .def("_set_range",
           py::overload_cast<uint32_t, int32_t, int32_t>(
               &NDRectangle::set_range<int32_t>),
           py::arg("dim_idx"), py::arg("start"), py::arg("end"))
      .def("_set_range",
           py::overload_cast<uint32_t, uint16_t, uint16_t>(
               &NDRectangle::set_range<uint16_t>),
           py::arg("dim_idx"), py::arg("start"), py::arg("end"))
      .def("_set_range",
           py::overload_cast<uint32_t, int16_t, int16_t>(
               &NDRectangle::set_range<int16_t>),
           py::arg("dim_idx"), py::arg("start"), py::arg("end"))
      .def("_set_range",
           py::overload_cast<uint32_t, uint8_t, uint8_t>(
               &NDRectangle::set_range<uint8_t>),
           py::arg("dim_idx"), py::arg("start"), py::arg("end"))
      .def("_set_range",
           py::overload_cast<uint32_t, int8_t, int8_t>(
               &NDRectangle::set_range<int8_t>),
           py::arg("dim_idx"), py::arg("start"), py::arg("end"))
      .def("_set_range",
           py::overload_cast<uint32_t, double, double>(
               &NDRectangle::set_range<double>),
           py::arg("dim_idx"), py::arg("start"), py::arg("end"))
      .def("_set_range",
           py::overload_cast<uint32_t, float, float>(
               &NDRectangle::set_range<float>),
           py::arg("dim_idx"), py::arg("start"), py::arg("end"))
      .def(
          "_set_range",
          [](NDRectangle &ndrect, uint32_t dim_idx, const std::string &start,
             const std::string &end) {
            return ndrect.set_range(dim_idx, start, end);
          },
          py::arg("dim_name"), py::arg("start"), py::arg("end"))

      .def("_range",
           [](NDRectangle &ndrect, const std::string &dim_name) -> py::tuple {
             const tiledb_datatype_t n_type = ndrect.range_dtype(dim_name);
             if (n_type == TILEDB_UINT64) {
               auto range = ndrect.range<uint64_t>(dim_name);
               return py::make_tuple(range[0], range[1]);
             } else if (n_type == TILEDB_INT64) {
               auto range = ndrect.range<int64_t>(dim_name);
               return py::make_tuple(range[0], range[1]);
             } else if (n_type == TILEDB_UINT32) {
               auto range = ndrect.range<uint32_t>(dim_name);
               return py::make_tuple(range[0], range[1]);
             } else if (n_type == TILEDB_INT32) {
               auto range = ndrect.range<int32_t>(dim_name);
               return py::make_tuple(range[0], range[1]);
             } else if (n_type == TILEDB_UINT16) {
               auto range = ndrect.range<uint16_t>(dim_name);
               return py::make_tuple(range[0], range[1]);
             } else if (n_type == TILEDB_INT16) {
               auto range = ndrect.range<int16_t>(dim_name);
               return py::make_tuple(range[0], range[1]);
             } else if (n_type == TILEDB_UINT8) {
               auto range = ndrect.range<uint8_t>(dim_name);
               return py::make_tuple(range[0], range[1]);
             } else if (n_type == TILEDB_INT8) {
               auto range = ndrect.range<int8_t>(dim_name);
               return py::make_tuple(range[0], range[1]);
             } else if (n_type == TILEDB_FLOAT64) {
               auto range = ndrect.range<double>(dim_name);
               return py::make_tuple(range[0], range[1]);
             } else if (n_type == TILEDB_FLOAT32) {
               auto range = ndrect.range<float>(dim_name);
               return py::make_tuple(range[0], range[1]);
             } else if (n_type == TILEDB_STRING_ASCII ||
                        n_type == TILEDB_STRING_UTF8) {
               auto range = ndrect.range<std::string>(dim_name);
               return py::make_tuple(range[0], range[1]);
             } else {
               TPY_ERROR_LOC("Unsupported type for NDRectangle's range");
             }
           })
      .def("_range", [](NDRectangle &ndrect, unsigned dim_idx) -> py::tuple {
        const tiledb_datatype_t n_type = ndrect.range_dtype(dim_idx);
        if (n_type == TILEDB_UINT64) {
          auto range = ndrect.range<uint64_t>(dim_idx);
          return py::make_tuple(range[0], range[1]);
        } else if (n_type == TILEDB_INT64) {
          auto range = ndrect.range<int64_t>(dim_idx);
          return py::make_tuple(range[0], range[1]);
        } else if (n_type == TILEDB_UINT32) {
          auto range = ndrect.range<uint32_t>(dim_idx);
          return py::make_tuple(range[0], range[1]);
        } else if (n_type == TILEDB_INT32) {
          auto range = ndrect.range<int32_t>(dim_idx);
          return py::make_tuple(range[0], range[1]);
        } else if (n_type == TILEDB_UINT16) {
          auto range = ndrect.range<uint16_t>(dim_idx);
          return py::make_tuple(range[0], range[1]);
        } else if (n_type == TILEDB_INT16) {
          auto range = ndrect.range<int16_t>(dim_idx);
          return py::make_tuple(range[0], range[1]);
        } else if (n_type == TILEDB_UINT8) {
          auto range = ndrect.range<uint8_t>(dim_idx);
          return py::make_tuple(range[0], range[1]);
        } else if (n_type == TILEDB_INT8) {
          auto range = ndrect.range<int8_t>(dim_idx);
          return py::make_tuple(range[0], range[1]);
        } else if (n_type == TILEDB_FLOAT64) {
          auto range = ndrect.range<double>(dim_idx);
          return py::make_tuple(range[0], range[1]);
        } else if (n_type == TILEDB_FLOAT32) {
          auto range = ndrect.range<float>(dim_idx);
          return py::make_tuple(range[0], range[1]);
        } else if (n_type == TILEDB_STRING_ASCII ||
                   n_type == TILEDB_STRING_UTF8) {
          auto range = ndrect.range<std::string>(dim_idx);
          return py::make_tuple(range[0], range[1]);
        } else {
          TPY_ERROR_LOC("Unsupported type for NDRectangle's range");
        }
      });

  py::class_<CurrentDomain>(m, "CurrentDomain")
      .def(py::init<CurrentDomain>())

      .def(py::init<const Context &>())

      .def("__capsule__",
           [](CurrentDomain &curr_dom) {
             return py::capsule(curr_dom.ptr().get(), "curr_dom");
           })

      .def_property_readonly("_is_empty", &CurrentDomain::is_empty)

      .def_property_readonly("_type", &CurrentDomain::type)

      .def("_set_ndrectangle", &CurrentDomain::set_ndrectangle,
           py::arg("ndrect"))

      .def("_ndrectangle", &CurrentDomain::ndrectangle);
#endif
}

} // namespace libtiledbcpp
