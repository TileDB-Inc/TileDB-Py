#include <tiledb/tiledb>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace libtiledbcpp {

using namespace tiledb;
namespace py = pybind11;

void init_domain(py::module& m) {
   py::class_<tiledb::Dimension>(m, "Dimension")
    // TODO: rewrite. this is an MVP placeholder and needs work including:
    // - don't hardcode the dtype
    // - convert from dtype <> tiledb_datatype_t
    // - accept np.array (monotype) as the ranges
    .def("create",
      [](const Context& ctx, const std::string& name, tiledb_datatype_t tiledb_datatype,
          py::object start, py::object end, py::object extent) {
        auto np = py::module::import("numpy");
        auto range_ = py::array(np.attr("array")(py::make_tuple(start, end)));
        auto extent_ = py::array(np.attr("array")(extent));

        const void *range_data = (tiledb_datatype != TILEDB_STRING_ASCII) ? range_.data() : nullptr;
        const void *extent_data = (tiledb_datatype != TILEDB_STRING_ASCII) ? extent_.data() : nullptr;

        return std::make_unique<Dimension>(
          Dimension::create(ctx, name, tiledb_datatype, range_data, extent_data));
      }
    )
    .def("cell_val_nul", &Dimension::cell_val_num)
    .def("set_cell_val_num", &Dimension::set_cell_val_num)
    .def("filter_list", &Dimension::filter_list)
    .def("set_filter_list", &Dimension::set_filter_list)
    .def("name", &Dimension::name)
    .def("tiledb_datatype", &Dimension::type)
    // TODO needs numpy <> tiledb type and void*+(type,size) -> numpy translators
    //.def("domain", &Dimension::domain)
    .def("domain_to_str", &Dimension::domain_to_str);

  py::class_<tiledb::Domain>(m, "Domain")
    .def(py::init<Context>())
    .def("cell_num", [](Domain& dom) { return dom.cell_num(); })
    .def("tiledb_datatype", &Domain::type)
    .def("ndim", &Domain::ndim)
    .def("dimensions", &Domain::dimensions)
    .def("dimension", py::overload_cast<const std::string&>(&Domain::dimension, py::const_))
    .def("dimension", py::overload_cast<const std::string&>(&Domain::dimension, py::const_))
    .def("add_dimension", &Domain::add_dimension);

}

} // namespace libtiledbcpp