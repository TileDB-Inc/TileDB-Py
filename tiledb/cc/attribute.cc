#include <tiledb/tiledb>
#include <tiledb/tiledb_experimental>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "common.h"

namespace libtiledbcpp {

using namespace tiledb;
using namespace tiledbpy::common;
namespace py = pybind11;

void set_fill_value(Attribute &attr, py::array value) {
  attr.set_fill_value(value.data(), value.nbytes());
}

py::array get_fill_value(Attribute &attr) {
  const void *value;
  uint64_t size;

  attr.get_fill_value(&value, &size);

  auto value_num = attr.cell_val_num();
  auto value_type = tdb_to_np_dtype(attr.type(), value_num);

  if (is_tdb_str(attr.type())) {
    value_type = py::dtype("|S1");
    value_num = size;
  }

  // record type
  if (py::str(value_type.attr("kind")) == py::str("V")) {
    value_num = 1;
  }

  // complex type - both cell values fit in a single complex element
  if (value_type == py::dtype("complex64") ||
      value_type == py::dtype("complex128")) {
    value_num = 1;
  }

  return py::array(value_type, value_num, value);
}

void set_enumeration_name(Attribute &attr, const Context &ctx,
                          const std::string &enumeration_name) {
  AttributeExperimental::set_enumeration_name(ctx, attr, enumeration_name);
}

std::optional<std::string> get_enumeration_name(Attribute &attr,
                                                const Context &ctx) {
  return AttributeExperimental::get_enumeration_name(ctx, attr);
}

void init_attribute(py::module &m) {
  py::class_<tiledb::Attribute>(m, "Attribute")
      .def(py::init<Attribute>())

      .def(py::init<Context &, std::string &, tiledb_datatype_t>())

      .def(
          py::init<Context &, std::string &, tiledb_datatype_t, FilterList &>())

      .def(py::init<const Context &, py::capsule>())

      .def("__capsule__",
           [](Attribute &attr) {
             return py::capsule(attr.ptr().get(), "attr", nullptr);
           })

      .def_property_readonly("_name", &Attribute::name)

      .def_property_readonly("_tiledb_dtype", &Attribute::type)

      .def_property("_nullable", &Attribute::nullable, &Attribute::set_nullable)

      .def_property("_ncell", &Attribute::cell_val_num,
                    &Attribute::set_cell_val_num)

      .def_property_readonly("_var", &Attribute::variable_sized)

      .def_property("_filters", &Attribute::filter_list,
                    &Attribute::set_filter_list)

      .def_property_readonly("_cell_size", &Attribute::cell_size)

      .def_property("_fill", get_fill_value, set_fill_value)

      .def("_get_enumeration_name", get_enumeration_name)

      .def("_set_enumeration_name", set_enumeration_name)

      .def("_dump", [](Attribute &attr) { attr.dump(); });
}

} // namespace libtiledbcpp
