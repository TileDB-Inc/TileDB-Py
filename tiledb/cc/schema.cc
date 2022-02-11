#include <tiledb/tiledb> // C++

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace libtiledbcpp {

using namespace tiledb;
namespace py = pybind11;

void init_schema(py::module &m) {
  py::class_<tiledb::ArraySchema>(m, "ArraySchema")
      .def(py::init<Context &, tiledb_array_type_t>(),
           py::keep_alive<1, 2>() /* ArraySchema keeps Context alive */)
      .def(py::init<Context &, std::string &>(),
           py::keep_alive<1, 2>() /* ArraySchema keeps Context alive */)
      .def(py::init<Context &, std::string &, tiledb_encryption_type_t,
                    std::string &>(),
           py::keep_alive<1, 2>() /* ArraySchema keeps Context alive */)
      // TODO .def(py::init<Context, py::capsule>) // tiledb_array_schema_t*
      // signature

      .def("dump", &ArraySchema::dump) // TODO add FILE* signature support?
      .def("dump",
           [](ArraySchema &schema) {
             schema.dump();
           }) // TODO add FILE* signature support?

      .def_property("domain", &ArraySchema::domain, &ArraySchema::set_domain)
      .def_property_readonly("array_type", &ArraySchema::array_type)
      //  .def_property_readonly("timestamp_range",
      //  &ArraySchema::timestamp_range)
      .def_property("capacity", &ArraySchema::capacity,
                    &ArraySchema::set_capacity)
      .def_property("cell_order", &ArraySchema::cell_order,
                    &ArraySchema::set_cell_order)
      .def_property("tile_order", &ArraySchema::tile_order,
                    &ArraySchema::set_tile_order)
      .def_property("allows_dups", &ArraySchema::allows_dups,
                    &ArraySchema::set_allows_dups)
      .def_property("coords_filters", &ArraySchema::coords_filter_list,
                    &ArraySchema::set_coords_filter_list)
      .def_property("offsets_filters", &ArraySchema::offsets_filter_list,
                    &ArraySchema::set_offsets_filter_list)
      .def_property("validity_filters", &ArraySchema::validity_filter_list,
                    &ArraySchema::set_validity_filter_list)

      // TODO? .def("__eq__", &Domain::operator==)

      .def("attr", py::overload_cast<const std::string &>(
                       &ArraySchema::attribute, py::const_))
      .def("attr",
           py::overload_cast<unsigned int>(&ArraySchema::attribute, py::const_))
      .def("nattr", &ArraySchema::attribute_num)
      //  .def("ndim", []() { return domain.ndim })
      .def("add_attr", &ArraySchema::add_attribute)
      .def("check", &ArraySchema::check)
      .def("has_attribute", &ArraySchema::has_attribute);
}

} // namespace libtiledbcpp
