#include <tiledb/tiledb>              // C++
#include <tiledb/tiledb_experimental> // (needed for dimension labels)

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace libtiledbcpp {

using namespace tiledb;
namespace py = pybind11;

class DimensionLabelSchema {
public:
  DimensionLabelSchema(tiledb_datatype_t dim_type, py::object dim_tile_extent,
                       tiledb_data_order_t label_order,
                       tiledb_datatype_t label_type)
      : dim_type_{dim_type}, dim_tile_extent_{dim_tile_extent},
        label_order_{label_order}, label_type_{label_type},
        label_filters_{std::nullopt} {}

  DimensionLabelSchema(tiledb_datatype_t dim_type, py::object dim_tile_extent,
                       tiledb_data_order_t label_order,
                       tiledb_datatype_t label_type,
                       const FilterList &label_filters)
      : dim_type_{dim_type}, dim_tile_extent_{dim_tile_extent},
        label_order_{label_order}, label_type_{label_type},
        label_filters_{label_filters} {}

  tiledb_datatype_t dim_type() const { return dim_type_; }

  const void *dim_tile_extent() const {

    if (dim_tile_extent_.is_none()) {
      return nullptr;
    }
    py::buffer tile_buffer = py::buffer(dim_tile_extent_);
    py::buffer_info tile_extent_info = tile_buffer.request();
    return tile_extent_info.ptr;
  }

  bool has_dim_tile_extent() const { return !dim_tile_extent_.is_none(); }

  bool has_label_filters() const { return label_filters_.has_value(); }

  tiledb_datatype_t label_type() const { return label_type_; }

  tiledb_data_order_t label_order() const { return label_order_; }

  const std::optional<FilterList> &label_filters() const {
    return label_filters_;
  }

private:
  tiledb_datatype_t dim_type_;
  py::object dim_tile_extent_;
  tiledb_data_order_t label_order_;
  tiledb_datatype_t label_type_;
  std::optional<FilterList> label_filters_;
};

void init_schema(py::module &m) {
  py::class_<DimensionLabelSchema>(m, "DimensionLabelSchema")
      .def(py::init<tiledb_datatype_t, py::object, tiledb_data_order_t,
                    tiledb_datatype_t>(),
           py::keep_alive<1, 3>())

      .def(py::init<tiledb_datatype_t, py::object, tiledb_data_order_t,
                    tiledb_datatype_t, const FilterList &>(),
           py::keep_alive<1, 3>())

      .def_property_readonly("_dim_dtype", &DimensionLabelSchema::dim_type)

      .def_property_readonly("_has_label_filters",
                             &DimensionLabelSchema::has_label_filters)

      .def_property_readonly("_label_dtype", &DimensionLabelSchema::label_type)

      .def_property_readonly("_label_filters",
                             [](DimensionLabelSchema &dim_label_schema) {
                               return dim_label_schema.label_filters().value();
                             })

      .def_property_readonly("_label_order", &DimensionLabelSchema::label_order)

      .def_property_readonly(
          "_dim_tile_extent",
          [](DimensionLabelSchema &dim_label_schema) -> py::object {
            const void *tile_extent = dim_label_schema.dim_tile_extent();
            if (tile_extent == nullptr) {
              return py::none();
            }
            auto dim_type = dim_label_schema.dim_type();

            switch (dim_type) {
            case TILEDB_UINT64: {
              using T = uint64_t;
              return py::cast(*static_cast<const T *>(tile_extent));
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
              using T = int64_t;
              return py::cast(*static_cast<const T *>(tile_extent));
            }
            case TILEDB_UINT32: {
              using T = uint32_t;
              return py::cast(*static_cast<const T *>(tile_extent));
            }
            case TILEDB_INT32: {
              using T = int32_t;
              return py::cast(*static_cast<const T *>(tile_extent));
            }
            case TILEDB_UINT16: {
              using T = uint16_t;
              return py::cast(*static_cast<const T *>(tile_extent));
            }
            case TILEDB_INT16: {
              using T = int16_t;
              return py::cast(*static_cast<const T *>(tile_extent));
            }
            case TILEDB_UINT8: {
              using T = uint8_t;
              return py::cast(*static_cast<const T *>(tile_extent));
            }
            case TILEDB_INT8: {
              using T = int8_t;
              return py::cast(*static_cast<const T *>(tile_extent));
            }
            case TILEDB_FLOAT64: {
              using T = double;
              return py::cast(*static_cast<const T *>(tile_extent));
            }
            case TILEDB_FLOAT32: {
              using T = float;
              return py::cast(*static_cast<const T *>(tile_extent));
            }
            case TILEDB_STRING_ASCII: {
              // This should have already been caught in the check for a mullptr
              // above.
              throw TileDBError("Setting a tile extent on a string dimension "
                                "is not supported for dimension labels");
            }
            default:
              throw TileDBError("Unsupported dtype for dimension tile extent");
            }
          });

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
           [](ArraySchema &schema, const Context &ctx, const std::string &name,
              uint32_t dim_index,
              const DimensionLabelSchema &dim_label_schema) {
             // Check dimension datatype.
             auto dim_type = schema.domain().dimension(dim_index).type();
             if (dim_label_schema.dim_type() != dim_type) {
               throw TileDBError("Cannot add dimension label '" + name +
                                 "'; The dimension datatype does not match the "
                                 "datatype of the dimension in the array.");
             }

             // Add dimension label.
             ArraySchemaExperimental::add_dimension_label(
                 ctx, schema, dim_index, name, dim_label_schema.label_order(),
                 dim_label_schema.label_type(),
                 dim_label_schema.label_filters());

             // If dimension tile extent is set, add dimension tile extent.
             if (dim_label_schema.has_dim_tile_extent()) {
               ctx.handle_error(
                   tiledb_array_schema_set_dimension_label_tile_extent(
                       ctx.ptr().get(), schema.ptr().get(), name.c_str(),
                       dim_type, dim_label_schema.dim_tile_extent()));
             }
           })
#else
      .def("_add_dim_label",
           [](ArraySchema &, const Context ,
              const std::string &, uint32_t dim_index,  const DimensionLabelSchema&) {
            throw TileDBError("Adding dimension labels require libtiledb version 2.15.0 or greater");
           })
#endif

      .def("_check", &ArraySchema::check)

      .def("_has_attribute", &ArraySchema::has_attribute)

#if TILEDB_VERSION_MAJOR == 2 && TILEDB_VERSION_MINOR >= 15
      .def("_has_dim_label",
           [](const ArraySchema &schema, const Context &ctx,
              const std::string &name) {
             return ArraySchemaExperimental::has_dimension_label(ctx, schema,
                                                                 name);
           })
#else
      .def("_has_dim_label", [](const ArraySchema &, const Context &,
                                const std::string &) {
        return false; 
      })
#endif

      .def("_add_enumeration", [](const ArraySchema &schema, const Context &ctx,
                                  const Enumeration &enmr) {
        ArraySchemaExperimental::add_enumeration(ctx, schema, enmr);
      });
}

} // namespace libtiledbcpp
