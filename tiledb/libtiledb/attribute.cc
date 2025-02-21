#include <tiledb/tiledb>
#include <tiledb/tiledb_experimental>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// #include <pybind11/pytypes.h>
// #include <pybind11/stl.h>

#include "common.h"

namespace libtiledbcpp {

using namespace tiledb;
using namespace tiledbnb::common;
namespace nb = nanobind;

void set_fill_value(Attribute& attr, nb::ndarray value) {
    attr.set_fill_value(value.data(), value.nbytes());
}

nb::ndarray get_fill_value(Attribute& attr) {
    // Get the fill value from the C++ API as a void* value.
    const void* value;
    uint64_t size;
    attr.get_fill_value(&value, &size);

    // If this is a string type, we want to return each value as a single cell.
    if (is_tdb_str(attr.type())) {
        auto value_type = nb::dtype<"|S1">;
        return nb::ndarray(value_type, size, value);
    }

    // If this is a record type (void), return a single cell.
    // If this is a blob-like type, we want to return each value as a single
    // byte cell.
    auto tdb_type = attr.type();
    if (tdb_type == TILEDB_BLOB
#if TILEDB_VERSION_MAJOR >= 2 && TILEDB_VERSION_MINOR >= 21
        || tdb_type == TILEDB_GEOM_WKB || tdb_type == TILEDB_GEOM_WKT
#endif
    ) {
        auto value_type = nb::dtype<"S">;
        return nb::ndarray(value_type, size, value);
    }

    // Get the number of values in a cell and the Python datatype.
    auto value_num = attr.cell_val_num();
    auto value_type = tdb_to_np_dtype(attr.type(), value_num);

    if (nb::getattr(value_type, "kind").is(nb::str("V"))) {
        return nb::ndarray(value_type, 1, value);
    }

    // If this is a complex type both cell values fit in a single complex
    // element.
    if (value_type.is(nb::dtype("complex64")) ||
        value_type.is(nb::dtype("complex128"))) {
        return nb::ndarray(value_type, 1, value);
    }

    return nb::ndarray(value_type, value_num, value);
}

void set_enumeration_name(
    Attribute& attr, const Context& ctx, const std::string& enumeration_name) {
    AttributeExperimental::set_enumeration_name(ctx, attr, enumeration_name);
}

std::optional<std::string> get_enumeration_name(
    Attribute& attr, const Context& ctx) {
    return AttributeExperimental::get_enumeration_name(ctx, attr);
}

void init_attribute(nb::module_& m) {
    nb::class_<tiledb::Attribute>(m, "Attribute")
        .def(nb::init<Attribute>())

        .def(nb::init<Context&, std::string&, tiledb_datatype_t>())

        .def(nb::init<Context&, std::string&, tiledb_datatype_t, FilterList&>())

        .def(nb::init<const Context&, nb::capsule>())

        .def(
            "__capsule__",
            [](Attribute& attr) {
                return nb::capsule(attr.ptr().get(), "attr");
            })

        .def_prop_rw_readonly("_name", &Attribute::name)

        .def_prop_rw_readonly("_tiledb_dtype", &Attribute::type)

        .def_prop_rw(
            "_nullable", &Attribute::nullable, &Attribute::set_nullable)

        .def_prop_rw(
            "_ncell", &Attribute::cell_val_num, &Attribute::set_cell_val_num)

        .def_prop_rw_readonly("_var", &Attribute::variable_sized)

        .def_prop_rw(
            "_filters", &Attribute::filter_list, &Attribute::set_filter_list)

        .def_prop_rw_readonly("_cell_size", &Attribute::cell_size)

        .def_prop_rw("_fill", get_fill_value, set_fill_value)

        .def("_get_enumeration_name", get_enumeration_name)

        .def("_set_enumeration_name", set_enumeration_name)

        .def("_dump", [](Attribute& attr) {
#if TILEDB_VERSION_MAJOR >= 2 && TILEDB_VERSION_MINOR >= 26
            std::stringstream ss;
            ss << attr;
            return ss.str();
#else
        attr.dump();
#endif
        });
}

}  // namespace libtiledbcpp
