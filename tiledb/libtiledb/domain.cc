#include <tiledb/tiledb>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// #include <pybind11/pytypes.h>
// #include <pybind11/stl.h>

#include "common.h"

namespace libtiledbcpp {

using namespace tiledb;
using namespace tiledbnb::common;
namespace nb = nanobind;

void init_domain(nb::module_& m) {
    nb::class_<Dimension>(m, "Dimension")
        .def(nb::init<Dimension>())

        .def(
            "__init__",
            [](Dimension* self,
               const Context& ctx,
               const std::string& name,
               tiledb_datatype_t datatype,
               nb::object domain,
               nb::object tile_extent) {
                void* dim_dom = nullptr;
                void* dim_tile = nullptr;

                if (!domain.is_none()) {
                    nb::buffer domain_buffer(domain);
                    nb::buffer_info domain_info = domain_buffer.request();
                    dim_dom = domain_info.ptr;
                }

                if (!tile_extent.is_none()) {
                    nb::buffer tile_buffer(tile_extent);
                    nb::buffer_info tile_extent_info = tile_buffer.request();
                    dim_tile = tile_extent_info.ptr;
                }

                new (self) Dimension(
                    Dimension::create(ctx, name, datatype, dim_dom, dim_tile));
            },
            nb::keep_alive<1, 2>())

        .def(nb::init<const Context&, nb::capsule>(), nb::keep_alive<1, 2>())

        .def_prop_rw_readonly("_name", &Dimension::name)

        .def_prop_rw_readonly(
            "_domain",
            [](Dimension& dim) {
                switch (dim.type()) {
                    case TILEDB_UINT64: {
                        auto dom = dim.domain<uint64_t>();
                        return nb::make_tuple(dom.first, dom.second);
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
                        return nb::make_tuple(dom.first, dom.second);
                    }
                    case TILEDB_UINT32: {
                        auto dom = dim.domain<uint32_t>();
                        return nb::make_tuple(dom.first, dom.second);
                    }
                    case TILEDB_INT32: {
                        auto dom = dim.domain<int32_t>();
                        return nb::make_tuple(dom.first, dom.second);
                    }
                    case TILEDB_UINT16: {
                        auto dom = dim.domain<uint16_t>();
                        return nb::make_tuple(dom.first, dom.second);
                    }
                    case TILEDB_INT16: {
                        auto dom = dim.domain<int16_t>();
                        return nb::make_tuple(dom.first, dom.second);
                    }
                    case TILEDB_UINT8: {
                        auto dom = dim.domain<uint8_t>();
                        return nb::make_tuple(dom.first, dom.second);
                    }
                    case TILEDB_INT8: {
                        auto dom = dim.domain<int8_t>();
                        return nb::make_tuple(dom.first, dom.second);
                    }
                    case TILEDB_FLOAT64: {
                        auto dom = dim.domain<double>();
                        return nb::make_tuple(dom.first, dom.second);
                    }
                    case TILEDB_FLOAT32: {
                        auto dom = dim.domain<float>();
                        return nb::make_tuple(dom.first, dom.second);
                    }
                    case TILEDB_STRING_ASCII: {
                        return nb::make_tuple("", "");
                    }
                    default:
                        TPY_ERROR_LOC(
                            "Unsupported dtype for Dimension's domain");
                }
            })

        .def_prop_rw_readonly(
            "_tile",
            [](Dimension& dim) -> nb::object {
                switch (dim.type()) {
                    case TILEDB_UINT64: {
                        return nb::cast(dim.tile_extent<uint64_t>());
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
                        return nb::cast(dim.tile_extent<int64_t>());
                    }
                    case TILEDB_UINT32: {
                        return nb::cast(dim.tile_extent<uint32_t>());
                    }
                    case TILEDB_INT32: {
                        return nb::cast(dim.tile_extent<int32_t>());
                    }
                    case TILEDB_UINT16: {
                        return nb::cast(dim.tile_extent<uint16_t>());
                    }
                    case TILEDB_INT16: {
                        return nb::cast(dim.tile_extent<int16_t>());
                    }
                    case TILEDB_UINT8: {
                        return nb::cast(dim.tile_extent<uint8_t>());
                    }
                    case TILEDB_INT8: {
                        return nb::cast(dim.tile_extent<int8_t>());
                    }
                    case TILEDB_FLOAT64: {
                        return nb::cast(dim.tile_extent<double>());
                    }
                    case TILEDB_FLOAT32: {
                        return nb::cast(dim.tile_extent<float>());
                    }
                    case TILEDB_STRING_ASCII: {
                        return nb::none();
                    }
                    default:
                        TPY_ERROR_LOC(
                            "Unsupported dtype  for Dimension's tile extent");
                }
            })

        .def_prop_rw(
            "_filters", &Dimension::filter_list, &Dimension::set_filter_list)

        .def_prop_rw(
            "_ncell", &Dimension::cell_val_num, &Dimension::set_cell_val_num)

        .def_prop_rw_readonly("_tiledb_dtype", &Dimension::type)

        .def("_domain_to_str", &Dimension::domain_to_str);

    nb::class_<Domain>(m, "Domain")
        .def(nb::init<Domain>())

        .def(nb::init<Context&>())

        .def(nb::init<const Context&, nb::capsule>())

        .def(
            "__capsule__",
            [](Domain& dom) { return nb::capsule(dom.ptr().get(), "dom"); })

        .def_prop_rw_readonly(
            "_ncell", [](Domain& dom) { return dom.cell_num(); })

        .def_prop_rw_readonly("_tiledb_dtype", &Domain::type)

        .def_prop_rw_readonly("_ndim", &Domain::ndim)

        .def_prop_rw_readonly("_dims", &Domain::dimensions)

        .def(
            "_dim", nb::overload_cast<unsigned>(&Domain::dimension, nb::const_))
        .def(
            "_dim",
            nb::overload_cast<const std::string&>(
                &Domain::dimension, nb::const_))

        .def("_has_dim", &Domain::has_dimension)

        .def("_add_dim", &Domain::add_dimension, nb::keep_alive<1, 2>())

        .def("_dump", [](Domain& dom) {
#if TILEDB_VERSION_MAJOR >= 2 && TILEDB_VERSION_MINOR >= 26
            std::stringstream ss;
            ss << dom;
            return ss.str();
#else
        dom.dump();
#endif
        });
}

}  // namespace libtiledbcpp
