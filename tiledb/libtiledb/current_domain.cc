#include <tiledb/tiledb>
#include <tiledb/tiledb_experimental>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// #include <pybind11/pytypes.h>
// #include <pybind11/stl.h>
#include <variant>

#include "common.h"

namespace libtiledbcpp {

using namespace tiledb;
using namespace tiledbnb::common;
namespace nb = nanobind;

void init_current_domain(nb::module_& m) {
#if TILEDB_VERSION_MAJOR >= 2 && TILEDB_VERSION_MINOR >= 26
    nb::class_<NDRectangle>(m, "NDRectangle")
        .def(nb::init<NDRectangle>())

        .def(nb::init<const Context&, const Domain&>())

        .def(
            "_set_range",
            [](NDRectangle& ndrect,
               const std::string& dim_name,
               nb::object start,
               nb::object end) {
                const tiledb_datatype_t n_type = ndrect.range_dtype(dim_name);

                if (n_type == TILEDB_UINT64) {
                    auto start_val = start.cast<uint64_t>();
                    auto end_val = end.cast<uint64_t>();
                    ndrect.set_range(dim_name, start_val, end_val);
                } else if (n_type == TILEDB_INT64) {
                    auto start_val = start.cast<int64_t>();
                    auto end_val = end.cast<int64_t>();
                    ndrect.set_range(dim_name, start_val, end_val);
                } else if (n_type == TILEDB_UINT32) {
                    auto start_val = start.cast<uint32_t>();
                    auto end_val = end.cast<uint32_t>();
                    ndrect.set_range(dim_name, start_val, end_val);
                } else if (n_type == TILEDB_INT32) {
                    auto start_val = start.cast<int32_t>();
                    auto end_val = end.cast<int32_t>();
                    ndrect.set_range(dim_name, start_val, end_val);
                } else if (n_type == TILEDB_UINT16) {
                    auto start_val = start.cast<uint16_t>();
                    auto end_val = end.cast<uint16_t>();
                    ndrect.set_range(dim_name, start_val, end_val);
                } else if (n_type == TILEDB_INT16) {
                    auto start_val = start.cast<int16_t>();
                    auto end_val = end.cast<int16_t>();
                    ndrect.set_range(dim_name, start_val, end_val);
                } else if (n_type == TILEDB_UINT8) {
                    auto start_val = start.cast<uint8_t>();
                    auto end_val = end.cast<uint8_t>();
                    ndrect.set_range(dim_name, start_val, end_val);
                } else if (n_type == TILEDB_INT8) {
                    auto start_val = start.cast<int8_t>();
                    auto end_val = end.cast<int8_t>();
                    ndrect.set_range(dim_name, start_val, end_val);
                } else if (n_type == TILEDB_FLOAT64) {
                    auto start_val = start.cast<double>();
                    auto end_val = end.cast<double>();
                    ndrect.set_range(dim_name, start_val, end_val);
                } else if (n_type == TILEDB_FLOAT32) {
                    auto start_val = start.cast<float>();
                    auto end_val = end.cast<float>();
                    ndrect.set_range(dim_name, start_val, end_val);
                } else if (
                    n_type == TILEDB_STRING_ASCII ||
                    n_type == TILEDB_STRING_UTF8) {
                    auto start_val = start.cast<std::string>();
                    auto end_val = end.cast<std::string>();
                    ndrect.set_range(dim_name, start_val, end_val);
                } else {
                    TPY_ERROR_LOC(
                        "Unsupported type for NDRectangle's set_range");
                }
            })

        .def(
            "_set_range",
            [](NDRectangle& ndrect,
               uint32_t dim_idx,
               nb::object start,
               nb::object end) {
                const tiledb_datatype_t n_type = ndrect.range_dtype(dim_idx);

                if (n_type == TILEDB_UINT64) {
                    auto start_val = start.cast<uint64_t>();
                    auto end_val = end.cast<uint64_t>();
                    ndrect.set_range(dim_idx, start_val, end_val);
                } else if (n_type == TILEDB_INT64) {
                    auto start_val = start.cast<int64_t>();
                    auto end_val = end.cast<int64_t>();
                    ndrect.set_range(dim_idx, start_val, end_val);
                } else if (n_type == TILEDB_UINT32) {
                    auto start_val = start.cast<uint32_t>();
                    auto end_val = end.cast<uint32_t>();
                    ndrect.set_range(dim_idx, start_val, end_val);
                } else if (n_type == TILEDB_INT32) {
                    auto start_val = start.cast<int32_t>();
                    auto end_val = end.cast<int32_t>();
                    ndrect.set_range(dim_idx, start_val, end_val);
                } else if (n_type == TILEDB_UINT16) {
                    auto start_val = start.cast<uint16_t>();
                    auto end_val = end.cast<uint16_t>();
                    ndrect.set_range(dim_idx, start_val, end_val);
                } else if (n_type == TILEDB_INT16) {
                    auto start_val = start.cast<int16_t>();
                    auto end_val = end.cast<int16_t>();
                    ndrect.set_range(dim_idx, start_val, end_val);
                } else if (n_type == TILEDB_UINT8) {
                    auto start_val = start.cast<uint8_t>();
                    auto end_val = end.cast<uint8_t>();
                    ndrect.set_range(dim_idx, start_val, end_val);
                } else if (n_type == TILEDB_INT8) {
                    auto start_val = start.cast<int8_t>();
                    auto end_val = end.cast<int8_t>();
                    ndrect.set_range(dim_idx, start_val, end_val);
                } else if (n_type == TILEDB_FLOAT64) {
                    auto start_val = start.cast<double>();
                    auto end_val = end.cast<double>();
                    ndrect.set_range(dim_idx, start_val, end_val);
                } else if (n_type == TILEDB_FLOAT32) {
                    auto start_val = start.cast<float>();
                    auto end_val = end.cast<float>();
                    ndrect.set_range(dim_idx, start_val, end_val);
                } else if (
                    n_type == TILEDB_STRING_ASCII ||
                    n_type == TILEDB_STRING_UTF8) {
                    auto start_val = start.cast<std::string>();
                    auto end_val = end.cast<std::string>();
                    ndrect.set_range(dim_idx, start_val, end_val);
                } else {
                    TPY_ERROR_LOC(
                        "Unsupported type for NDRectangle's set_range");
                }
            })

        .def(
            "_range",
            [](NDRectangle& ndrect, const std::string& dim_name) -> nb::tuple {
                const tiledb_datatype_t n_type = ndrect.range_dtype(dim_name);
                if (n_type == TILEDB_UINT64) {
                    auto range = ndrect.range<uint64_t>(dim_name);
                    return nb::make_tuple(range[0], range[1]);
                } else if (n_type == TILEDB_INT64) {
                    auto range = ndrect.range<int64_t>(dim_name);
                    return nb::make_tuple(range[0], range[1]);
                } else if (n_type == TILEDB_UINT32) {
                    auto range = ndrect.range<uint32_t>(dim_name);
                    return nb::make_tuple(range[0], range[1]);
                } else if (n_type == TILEDB_INT32) {
                    auto range = ndrect.range<int32_t>(dim_name);
                    return nb::make_tuple(range[0], range[1]);
                } else if (n_type == TILEDB_UINT16) {
                    auto range = ndrect.range<uint16_t>(dim_name);
                    return nb::make_tuple(range[0], range[1]);
                } else if (n_type == TILEDB_INT16) {
                    auto range = ndrect.range<int16_t>(dim_name);
                    return nb::make_tuple(range[0], range[1]);
                } else if (n_type == TILEDB_UINT8) {
                    auto range = ndrect.range<uint8_t>(dim_name);
                    return nb::make_tuple(range[0], range[1]);
                } else if (n_type == TILEDB_INT8) {
                    auto range = ndrect.range<int8_t>(dim_name);
                    return nb::make_tuple(range[0], range[1]);
                } else if (n_type == TILEDB_FLOAT64) {
                    auto range = ndrect.range<double>(dim_name);
                    return nb::make_tuple(range[0], range[1]);
                } else if (n_type == TILEDB_FLOAT32) {
                    auto range = ndrect.range<float>(dim_name);
                    return nb::make_tuple(range[0], range[1]);
                } else if (
                    n_type == TILEDB_STRING_ASCII ||
                    n_type == TILEDB_STRING_UTF8) {
                    auto range = ndrect.range<std::string>(dim_name);
                    return nb::make_tuple(range[0], range[1]);
                } else {
                    TPY_ERROR_LOC("Unsupported type for NDRectangle's range");
                }
            })
        .def("_range", [](NDRectangle& ndrect, unsigned dim_idx) -> nb::tuple {
            const tiledb_datatype_t n_type = ndrect.range_dtype(dim_idx);
            if (n_type == TILEDB_UINT64) {
                auto range = ndrect.range<uint64_t>(dim_idx);
                return nb::make_tuple(range[0], range[1]);
            } else if (n_type == TILEDB_INT64) {
                auto range = ndrect.range<int64_t>(dim_idx);
                return nb::make_tuple(range[0], range[1]);
            } else if (n_type == TILEDB_UINT32) {
                auto range = ndrect.range<uint32_t>(dim_idx);
                return nb::make_tuple(range[0], range[1]);
            } else if (n_type == TILEDB_INT32) {
                auto range = ndrect.range<int32_t>(dim_idx);
                return nb::make_tuple(range[0], range[1]);
            } else if (n_type == TILEDB_UINT16) {
                auto range = ndrect.range<uint16_t>(dim_idx);
                return nb::make_tuple(range[0], range[1]);
            } else if (n_type == TILEDB_INT16) {
                auto range = ndrect.range<int16_t>(dim_idx);
                return nb::make_tuple(range[0], range[1]);
            } else if (n_type == TILEDB_UINT8) {
                auto range = ndrect.range<uint8_t>(dim_idx);
                return nb::make_tuple(range[0], range[1]);
            } else if (n_type == TILEDB_INT8) {
                auto range = ndrect.range<int8_t>(dim_idx);
                return nb::make_tuple(range[0], range[1]);
            } else if (n_type == TILEDB_FLOAT64) {
                auto range = ndrect.range<double>(dim_idx);
                return nb::make_tuple(range[0], range[1]);
            } else if (n_type == TILEDB_FLOAT32) {
                auto range = ndrect.range<float>(dim_idx);
                return nb::make_tuple(range[0], range[1]);
            } else if (
                n_type == TILEDB_STRING_ASCII || n_type == TILEDB_STRING_UTF8) {
                auto range = ndrect.range<std::string>(dim_idx);
                return nb::make_tuple(range[0], range[1]);
            } else {
                TPY_ERROR_LOC("Unsupported type for NDRectangle's range");
            }
        });

    nb::class_<CurrentDomain>(m, "CurrentDomain")
        .def(nb::init<CurrentDomain>())

        .def(nb::init<const Context&>())

        .def(
            "__capsule__",
            [](CurrentDomain& curr_dom) {
                return nb::capsule(curr_dom.ptr().get(), "curr_dom");
            })

        .def_prop_rw_readonly("_is_empty", &CurrentDomain::is_empty)

        .def_prop_rw_readonly("_type", &CurrentDomain::type)

        .def(
            "_set_ndrectangle",
            &CurrentDomain::set_ndrectangle,
            nb::arg("ndrect"))

        .def("_ndrectangle", &CurrentDomain::ndrectangle);
#endif
}

}  // namespace libtiledbcpp
