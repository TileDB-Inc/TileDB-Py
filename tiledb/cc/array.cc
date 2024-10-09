#include <tiledb/tiledb.h> // for enums
#include <tiledb/tiledb>   // C++
#include <tiledb/tiledb_experimental>

#include "common.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace libtiledbcpp {

using namespace tiledb;
namespace py = pybind11;

template <typename T>
bool _non_empty_domain_is_empty_aux(tiledb::Array &arr, const unsigned &dim_idx,
                                    const Context &ctx) {
  int32_t is_empty = 0;
  T domain[2];
  ctx.handle_error(tiledb_array_get_non_empty_domain_from_index(
      ctx.ptr().get(), arr.ptr().get(), dim_idx, &domain, &is_empty));
  return is_empty == 1;
}

void init_array(py::module &m) {
  py::class_<tiledb::Array>(m, "Array")
      //.def(py::init<py::object, py::object, py::iterable, py::object,
      //              py::object, py::object>())
      .def(py::init<Array>())
      .def(
          py::init<const Context &, const std::string &, tiledb_query_type_t>(),
          py::keep_alive<1, 2>() /* Array keeps Context alive */)
      .def(py::init([](const Context &ctx, const std::string &uri,
                       const tiledb_query_type_t &qt,
                       const std::tuple<std::optional<uint64_t>,
                                        std::optional<uint64_t>> &timestamp) {
             std::optional<uint64_t> start, end;
             std::tie(start, end) = timestamp;

             if (!start.has_value())
               start = 0;
             if (!end.has_value())
               end = UINT64_MAX;

             return std::make_unique<Array>(
                 ctx, uri, qt,
                 TemporalPolicy(tiledb::TimestampStartEnd, start.value(),
                                end.value()));
           }),
           py::keep_alive<1, 2>())
      // Temporary initializer while Array is converted from Cython to PyBind.
      .def(py::init([](const Context &ctx, py::object array) {
             tiledb_array_t *c_array = (py::capsule)array.attr("__capsule__")();
             return std::make_unique<Array>(ctx, c_array, false);
           }),
           py::keep_alive<1, 2>(), py::keep_alive<1, 3>())

      // TODO capsule Array(const Context& ctx, tiledb_array_t* carray,
      // tiledb_config_t* config)
      .def("_is_open", &Array::is_open)
      .def("_uri", &Array::uri)
      .def("_schema", &Array::schema)
      .def("_query_type", &Array::query_type)
      .def("__capsule__",
           [](Array &self) { return py::capsule(self.ptr().get(), "array"); })
      .def("_open", (void(Array::*)(tiledb_query_type_t)) & Array::open)
      .def("_reopen", &Array::reopen)
      .def("_set_open_timestamp_start", &Array::set_open_timestamp_start)
      .def("_set_open_timestamp_end", &Array::set_open_timestamp_end)
      .def_property_readonly("_open_timestamp_start",
                             &Array::open_timestamp_start)
      .def_property_readonly("_open_timestamp_end", &Array::open_timestamp_end)
      .def("_set_config", &Array::set_config)
      .def("_config", &Array::config)
      .def("_close", &Array::close)
      .def("_get_enumeration",
           [](Array &self, const Context &ctx, const std::string &attr_name) {
             return ArrayExperimental::get_enumeration(ctx, self, attr_name);
           })
      .def_static("_consolidate",
                  [](const std::string &uri, const Context &ctx,
                     Config *config) { Array::consolidate(ctx, uri, config); })
      .def_static("_consolidate",
                  [](const std::string &uri, const Context &ctx,
                     const std::vector<std::string> &fragment_uris,
                     Config *config) {
                    std::vector<const char *> c_strings;
                    c_strings.reserve(fragment_uris.size());
                    for (const auto &str : fragment_uris) {
                      c_strings.push_back(str.c_str());
                    }
                    Array::consolidate(ctx, uri, c_strings.data(),
                                       fragment_uris.size(), config);
                  })
      .def_static("_consolidate",
                  [](const std::string &uri, const Context &ctx,
                     const std::tuple<int, int> &timestamp, Config *config) {
                    int start, end;
                    std::tie(start, end) = timestamp;

                    config->set("sm.consolidation.timestamp_start",
                                std::to_string(start));
                    config->set("sm.consolidation.timestamp_end",
                                std::to_string(end));

                    Array::consolidate(ctx, uri, config);
                  })
      .def("_vacuum", &Array::vacuum)
      .def("_create",
           [](const Context &ctx, const std::string &uri,
              const ArraySchema &schema) {
             ctx.handle_error(tiledb_array_create(ctx.ptr().get(), uri.c_str(),
                                                  schema.ptr().get()));
           })
      .def("_load_schema",
           py::overload_cast<const Context &, const std::string &>(
               &Array::load_schema))
      .def("_encryption_type", &Array::encryption_type)

      .def("_non_empty_domain_is_empty",
           [](Array &self, const unsigned &dim_idx, const py::dtype &n_type,
              const Context &ctx) -> bool {
             int32_t is_empty = 0;

             if (py::getattr(n_type, "kind").is(py::str("S")) ||
                 py::getattr(n_type, "kind").is(py::str("U"))) {
               uint64_t start_size, end_size;
               ctx.handle_error(
                   tiledb_array_get_non_empty_domain_var_size_from_index(
                       ctx.ptr().get(), self.ptr().get(), dim_idx, &start_size,
                       &end_size, &is_empty));
               return is_empty == 1;
             } else {
               void *domain = nullptr;
               if (n_type.is(py::dtype::of<uint64_t>())) {
                 return _non_empty_domain_is_empty_aux<uint64_t>(self, dim_idx,
                                                                 ctx);
                 // int64_t also used for datetime64
               } else if (n_type.is(py::dtype::of<int64_t>()) ||
                          py::getattr(n_type, "kind").is(py::str("M"))) {
                 return _non_empty_domain_is_empty_aux<int64_t>(self, dim_idx,
                                                                ctx);
               } else if (n_type.is(py::dtype::of<uint32_t>())) {
                 return _non_empty_domain_is_empty_aux<uint32_t>(self, dim_idx,
                                                                 ctx);
               } else if (n_type.is(py::dtype::of<int32_t>())) {
                 return _non_empty_domain_is_empty_aux<int32_t>(self, dim_idx,
                                                                ctx);
               } else if (n_type.is(py::dtype::of<uint16_t>())) {
                 return _non_empty_domain_is_empty_aux<uint16_t>(self, dim_idx,
                                                                 ctx);
               } else if (n_type.is(py::dtype::of<int16_t>())) {
                 return _non_empty_domain_is_empty_aux<int16_t>(self, dim_idx,
                                                                ctx);
               } else if (n_type.is(py::dtype::of<uint8_t>())) {
                 return _non_empty_domain_is_empty_aux<uint8_t>(self, dim_idx,
                                                                ctx);
               } else if (n_type.is(py::dtype::of<int8_t>())) {
                 return _non_empty_domain_is_empty_aux<int8_t>(self, dim_idx,
                                                               ctx);
               } else if (n_type.is(py::dtype::of<double>())) {
                 return _non_empty_domain_is_empty_aux<double>(self, dim_idx,
                                                               ctx);
               } else if (n_type.is(py::dtype::of<float>())) {
                 return _non_empty_domain_is_empty_aux<float>(self, dim_idx,
                                                              ctx);
               } else {
                 TPY_ERROR_LOC("Unsupported type");
               }
             }
           })
      .def("_non_empty_domain",
           [](Array &self, const unsigned &dim_idx,
              const py::dtype &n_type) -> py::tuple {
             if (n_type.is(py::dtype::of<uint64_t>())) {
               auto domain = self.non_empty_domain<uint64_t>(dim_idx);
               return py::make_tuple(domain.first, domain.second);
             } else if (n_type.is(py::dtype::of<int64_t>())) {
               auto domain = self.non_empty_domain<int64_t>(dim_idx);
               return py::make_tuple(domain.first, domain.second);
             } else if (n_type.is(py::dtype::of<uint32_t>())) {
               auto domain = self.non_empty_domain<uint32_t>(dim_idx);
               return py::make_tuple(domain.first, domain.second);
             } else if (n_type.is(py::dtype::of<int32_t>())) {
               auto domain = self.non_empty_domain<int32_t>(dim_idx);
               return py::make_tuple(domain.first, domain.second);
             } else if (n_type.is(py::dtype::of<uint16_t>())) {
               auto domain = self.non_empty_domain<uint16_t>(dim_idx);
               return py::make_tuple(domain.first, domain.second);
             } else if (n_type.is(py::dtype::of<int16_t>())) {
               auto domain = self.non_empty_domain<int16_t>(dim_idx);
               return py::make_tuple(domain.first, domain.second);
             } else if (n_type.is(py::dtype::of<uint8_t>())) {
               auto domain = self.non_empty_domain<uint8_t>(dim_idx);
               return py::make_tuple(domain.first, domain.second);
             } else if (n_type.is(py::dtype::of<int8_t>())) {
               auto domain = self.non_empty_domain<int8_t>(dim_idx);
               return py::make_tuple(domain.first, domain.second);
             } else if (n_type.is(py::dtype::of<double>())) {
               auto domain = self.non_empty_domain<double>(dim_idx);
               return py::make_tuple(domain.first, domain.second);
             } else if (n_type.is(py::dtype::of<float>())) {
               auto domain = self.non_empty_domain<float>(dim_idx);
               return py::make_tuple(domain.first, domain.second);
             } else if (py::getattr(n_type, "kind").is(py::str("S")) ||
                        py::getattr(n_type, "kind").is(py::str("U"))) {
               auto domain = self.non_empty_domain_var(dim_idx);
               return py::make_tuple(domain.first, domain.second);
               // np.datetime64
             } else if (py::getattr(n_type, "kind").is(py::str("M"))) {
               auto domain = self.non_empty_domain<int64_t>(dim_idx);
               return py::make_tuple(domain.first, domain.second);
             } else {
               TPY_ERROR_LOC("Unsupported type");
             }
           })

      .def("_query_type", &Array::query_type)
      .def("_delete_fragments", &Array::delete_fragments)
      .def("_consolidate_fragments",
           [](const std::string &uri, const Context &ctx,
              const std::vector<std::string> &fragment_uris, Config *config) {
             std::vector<const char *> c_strings;
             c_strings.reserve(fragment_uris.size());
             for (const auto &str : fragment_uris) {
               c_strings.push_back(str.c_str());
             }
             ctx.handle_error(tiledb_array_consolidate_fragments(
                 ctx.ptr().get(), uri.c_str(), c_strings.data(),
                 fragment_uris.size(), config->ptr().get()));
           })
      .def("_consolidate_metadata",
           py::overload_cast<const Context &, const std::string &,
                             Config *const>(&Array::consolidate_metadata))
      .def("_put_metadata",
           [](Array &self, std::string &key, tiledb_datatype_t tdb_type,
              const py::buffer &b) {
             py::buffer_info info = b.request();

             size_t size = 1;
             for (auto s : info.shape) {
               size *= s;
             }
             self.put_metadata(key, tdb_type, size, info.ptr);
           })
      .def("_get_metadata",
           [](Array &self, std::string &key) -> py::buffer {
             tiledb_datatype_t tdb_type;
             uint32_t value_num = 0;
             const void *data_ptr = nullptr;

             self.get_metadata(key, &tdb_type, &value_num, &data_ptr);

             if (data_ptr == nullptr && value_num != 1) {
               throw py::key_error();
             }

             assert(data_ptr != nullptr);
             return py::memoryview::from_memory(
                 data_ptr, value_num * tiledb_datatype_size(tdb_type));
           })
      .def("_get_metadata_from_index",
           [](Array &self, uint64_t index) -> py::tuple {
             tiledb_datatype_t tdb_type;
             uint32_t value_num = 0;
             const void *data_ptr = nullptr;
             std::string key;

             self.get_metadata_from_index(index, &key, &tdb_type, &value_num,
                                          &data_ptr);

             if (data_ptr == nullptr && value_num != 1) {
               throw py::key_error();
             }
             // TODO handle empty value case

             assert(data_ptr != nullptr);
             auto buf = py::memoryview::from_memory(
                 data_ptr, value_num * tiledb_datatype_size(tdb_type));

             return py::make_tuple(tdb_type, buf);
           })
      .def("_delete_metadata", &Array::delete_metadata)
      .def("_has_metadata",
           [](Array &self, std::string &key) -> py::tuple {
             tiledb_datatype_t has_type;
             bool has_it = self.has_metadata(key, &has_type);
             return py::make_tuple(has_it, has_type);
           })
      .def("metadata_num", &Array::metadata_num)
      .def("_delete_array",
           py::overload_cast<const Context &, const std::string &>(
               &Array::delete_array))
      .def("_upgrade_version", &Array::upgrade_version);
}

} // namespace libtiledbcpp
