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
      .def(py::init<Context &, tiledb_array_type_t>(), py::keep_alive<1, 2>())
      .def(py::init<Context &, std::string &>(), py::keep_alive<1, 2>())
      .def(py::init<Context &, std::string &, tiledb_encryption_type_t,
                    std::string &>(),
           py::keep_alive<1, 2>())
      .def(py::init<Context &, py::capsule>(), py::keep_alive<1, 2>())

      .def("__capsule__",
           [](ArraySchema &schema) {
             return py::capsule(schema.ptr().get(), "schema", nullptr);
           })

      .def("_dump", &ArraySchema::dump) // TODO add FILE* signature support?
      .def("_dump",
           [](ArraySchema &schema) {
             schema.dump();
           }) // TODO add FILE* signature support?

      .def("_ctx", &ArraySchema::context)
      .def_property("_domain", &ArraySchema::domain, &ArraySchema::set_domain)
      .def_property_readonly("_array_type", &ArraySchema::array_type)
      //  .def_property_readonly("timestamp_range",
      //  &ArraySchema::timestamp_range)
      .def_property("_capacity", &ArraySchema::capacity,
                    &ArraySchema::set_capacity)
      .def_property("_cell_order", &ArraySchema::cell_order,
                    &ArraySchema::set_cell_order)
      .def_property("_tile_order", &ArraySchema::tile_order,
                    &ArraySchema::set_tile_order)
      .def_property("_allows_dups", &ArraySchema::allows_dups,
                    &ArraySchema::set_allows_dups)
      .def_property("_coords_filters", &ArraySchema::coords_filter_list,
                    &ArraySchema::set_coords_filter_list)
      .def_property("_offsets_filters", &ArraySchema::offsets_filter_list,
                    &ArraySchema::set_offsets_filter_list)
      .def_property("_validity_filters", &ArraySchema::validity_filter_list,
                    &ArraySchema::set_validity_filter_list)

      // TODO? .def("__eq__", &Domain::operator==)

      .def("_attr", py::overload_cast<const std::string &>(
                        &ArraySchema::attribute, py::const_))
      .def("_attr",
           py::overload_cast<unsigned int>(&ArraySchema::attribute, py::const_))
      .def_property_readonly("_nattr", &ArraySchema::attribute_num)
      //  .def("ndim", []() { return domain.ndim })
      .def("_add_attr", &ArraySchema::add_attribute)
      .def("_check", &ArraySchema::check)
      .def("_has_attribute", &ArraySchema::has_attribute);
}

} // namespace libtiledbcpp
