#include <tiledb/tiledb> // C++

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace libtiledbcpp {

using namespace tiledb;
namespace py = pybind11;


void init_schema(py::module& m) {
      py::class_<tiledb::ArraySchema>(m, "ArraySchema")
          .def(py::init<Context&, tiledb_array_type_t>(),
               py::keep_alive<1,2>() /* ArraySchema keeps Context alive */)
          .def(py::init<Context&, std::string&>(),
               py::keep_alive<1,2>() /* ArraySchema keeps Context alive */)
          .def(py::init<Context&, std::string&, tiledb_encryption_type_t, std::string&>(),
               py::keep_alive<1,2>() /* ArraySchema keeps Context alive */)
          // TODO .def(py::init<Context, py::capsule>) // tiledb_array_schema_t* signature
          .def("dump", &ArraySchema::dump) // TODO add FILE* signature support?
          .def("dump", [](ArraySchema& schema) {
               schema.dump();
          }) // TODO add FILE* signature support?
          .def("array_type", &ArraySchema::array_type)
          .def("capacity", &ArraySchema::capacity)
          .def("set_capacity", &ArraySchema::set_capacity)
          .def("set_allows_dups", &ArraySchema::set_allows_dups)
          .def("allows_dups", &ArraySchema::allows_dups)
          .def("set_tile_order", &ArraySchema::set_tile_order)
          .def("tile_order", &ArraySchema::tile_order)
          .def("set_order", &ArraySchema::set_tile_order)
          .def("cell_order", &ArraySchema::cell_order)
          .def("set_cell_order", &ArraySchema::set_cell_order)
          //.set("set_coords_filter_list")
          //.def("coords_filter_list", &ArraySchema::coords_filter_list)
          //.def("offsets_filter_list")
          //.def("set_offsets_filter_list")
          .def("domain", &ArraySchema::domain)
          .def("set_domain", &ArraySchema::set_domain)
          // TODO? .def("__eq__", &Domain::operator==)
          .def("add_attribute", &ArraySchema::add_attribute)
          ;
}

} // namespace