#include <tiledb/tiledb>              // C++
#include <tiledb/tiledb_experimental> // (needed for dimension labels)

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace libtiledbcpp {

using namespace tiledb;
namespace py = pybind11;

void init_schema(py::module &m) {
  py::class_<tiledb::ArraySchema>(m, "ArraySchema")
      .def(py::init<ArraySchema>())

      .def(py::init<Context &, tiledb_array_type_t>(), py::keep_alive<1, 2>())

      .def(py::init<Context &, std::string &>())

      .def(py::init<Context &, std::string &, tiledb_encryption_type_t,
                    std::string &>(),
           py::keep_alive<1, 2>())

      .def(py::init<Context &, py::capsule>())

      .def("__capsule__",
           [](ArraySchema &schema) {
             return py::capsule(schema.ptr().get(), "schema", nullptr);
           })

      .def("_dump", &ArraySchema::dump)
      .def("_dump", [](ArraySchema &schema) { schema.dump(); })

      .def("_ctx", &ArraySchema::context)

      .def_property("_domain", &ArraySchema::domain, &ArraySchema::set_domain)

      .def_property_readonly("_array_type", &ArraySchema::array_type)

      //  .def_property_readonly("timestamp_range",
      //  &ArraySchema::timestamp_range)

      .def_property("_capacity", &ArraySchema::capacity,
                    &ArraySchema::set_capacity)

      .def_property_readonly("_version", &ArraySchema::version)

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

      .def("_attr", py::overload_cast<const std::string &>(
                        &ArraySchema::attribute, py::const_))
      .def("_attr",
           py::overload_cast<unsigned int>(&ArraySchema::attribute, py::const_))

#if TILEDB_VERSION_MAJOR == 2 && TILEDB_VERSION_MINOR >= 15
      .def("_dim_label",
           [](const ArraySchema &schema, const Context &context,
              const std::string &name) {
             return ArraySchemaExperimental::dimension_label(context, schema,
                                                             name);
           })
#else
      .def("_dim_label",
           [](const ArraySchema &, const Context &,
              const std::string &) {
            throw TileDBError("Getting dimension labels require libtiledb version 2.15.0 or greater");
           })
#endif

      .def_property_readonly("_nattr", &ArraySchema::attribute_num)

      .def_property_readonly(
          "_ndim", [](ArraySchema schema) { return schema.domain().ndim(); })

      .def("_add_attr", &ArraySchema::add_attribute)

#if TILEDB_VERSION_MAJOR == 2 && TILEDB_VERSION_MINOR >= 15
      .def("_add_dim_label",
           [](ArraySchema &schema, const Context &ctx, uint32_t dim_idx,
              const std::string &name, tiledb_data_order_t label_order,
              tiledb_datatype_t label_type) {
             ArraySchemaExperimental::add_dimension_label(
                 ctx, schema, dim_idx, name, label_order, label_type);
           })
#else
      .def("_add_dim_label",
           [](ArraySchema &schema, const Context &ctx, uint32_t dim_idx,
              const std::string &name, tiledb_data_order_t label_order,
              tiledb_datatype_t label_type) {
            throw TileDBError("Adding dimension labels require libtiledb version 2.15.0 or greater");
           })
#endif

#if TILEDB_VERSION_MAJOR == 2 && TILEDB_VERSION_MINOR >= 15
      .def("_add_dim_label",
           [](ArraySchema &schema, const Context &ctx, uint32_t dim_idx,
              const std::string &name, tiledb_data_order_t label_order,
              tiledb_datatype_t label_type,
              std::optional<FilterList> label_filters = std::nullopt) {
             ArraySchemaExperimental::add_dimension_label(
                 ctx, schema, dim_idx, name, label_order, label_type,
                 label_filters);
           })
#else
      .def("_add_dim_label",
           [](ArraySchema &, const Context &, uint32_t ,
              const std::string &, tiledb_data_order_t ,
              tiledb_datatype_t ,
              std::optional<FilterList> label_filters = std::nullopt) {
            throw TileDBError("Adding dimension labels require libtiledb version 2.15.0 or greater");
           })
#endif

      .def("_check", &ArraySchema::check)

      .def("_has_attribute", &ArraySchema::has_attribute)

#if TILEDB_VERSION_MAJOR == 2 && TILEDB_VERSION_MINOR >= 15
      .def("_has_dim_label", [](const ArraySchema &schema, const Context &ctx,
                                const std::string &name) {
        return ArraySchemaExperimental::has_dimension_label(ctx, schema, name);
      });
#else
      .def("_has_dim_label", [](const ArraySchema &, const Context &,
                                const std::string &) {
        return false; 
      });
#endif
}

} // namespace libtiledbcpp
