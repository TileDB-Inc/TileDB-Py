#include <tiledb/tiledb.h>  // for enums
#include <tiledb/tiledb>    // C++
#include <tiledb/tiledb_experimental>

#include "metadata.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace libtiledbcpp {

using namespace tiledb;
namespace nb = nanobind;

template <typename T>
bool _non_empty_domain_is_empty_aux(
    tiledb::Array& arr, const unsigned& dim_idx, const Context& ctx) {
    int32_t is_empty = 0;
    T domain[2];
    ctx.handle_error(tiledb_array_get_non_empty_domain_from_index(
        ctx.ptr().get(), arr.ptr().get(), dim_idx, &domain, &is_empty));
    return is_empty == 1;
}

void init_array(nb::module& m) {
    nb::class_<tiledb::Array>(m, "Array")
        //.def(nb::init<nb::object, nb::object, nb::iterable, nb::object,
        //              nb::object, nb::object>())
        .def(nb::init<Array>())
        .def(
            nb::init<const Context&, const std::string&, tiledb_query_type_t>(),
            nb::keep_alive<1, 2>() /* Array keeps Context alive */)
        .def(
            nb::init([](const Context& ctx,
                        const std::string& uri,
                        const tiledb_query_type_t& qt,
                        const std::tuple<
                            std::optional<uint64_t>,
                            std::optional<uint64_t>>& timestamp) {
                std::optional<uint64_t> start, end;
                std::tie(start, end) = timestamp;

                if (!start.has_value())
                    start = 0;
                if (!end.has_value())
                    end = UINT64_MAX;

                return std::make_unique<Array>(
                    ctx,
                    uri,
                    qt,
                    TemporalPolicy(
                        tiledb::TimestampStartEnd, start.value(), end.value()));
            }),
            nb::keep_alive<1, 2>())
        .def(
            nb::init([](const Context& ctx, nb::object array) {
                tiledb_array_t* c_array = (nb::capsule)array.attr(
                    "__capsule__")();
                return std::make_unique<Array>(ctx, c_array, false);
            }),
            nb::keep_alive<1, 2>(),
            nb::keep_alive<1, 3>())

        // TODO capsule Array(const Context& ctx, tiledb_array_t* carray,
        // tiledb_config_t* config)
        .def("_is_open", &Array::is_open)
        .def("_uri", &Array::uri)
        .def("_schema", &Array::schema)
        .def("_query_type", &Array::query_type)
        .def(
            "__capsule__",
            [](Array& self) { return nb::capsule(self.ptr().get(), "array"); })
        .def("_open", (void(Array::*)(tiledb_query_type_t)) & Array::open)
        .def("_reopen", &Array::reopen)
        .def("_set_open_timestamp_start", &Array::set_open_timestamp_start)
        .def("_set_open_timestamp_end", &Array::set_open_timestamp_end)
        .def_prop_rw_readonly(
            "_open_timestamp_start", &Array::open_timestamp_start)
        .def_prop_rw_readonly(
            "_open_timestamp_end", &Array::open_timestamp_end)
        .def("_set_config", &Array::set_config)
        .def("_config", &Array::config)
        .def("_close", &Array::close)
        .def(
            "_get_enumeration",
            [](Array& self, const Context& ctx, const std::string& attr_name) {
                return ArrayExperimental::get_enumeration(ctx, self, attr_name);
            })
        .def_static(
            "_consolidate",
            [](const std::string& uri, const Context& ctx, Config* config) {
                Array::consolidate(ctx, uri, config);
            })
        .def_static(
            "_consolidate",
            [](const std::string& uri,
               const Context& ctx,
               const std::vector<std::string>& fragment_uris,
               Config* config) {
                std::vector<const char*> c_strings;
                c_strings.reserve(fragment_uris.size());
                for (const auto& str : fragment_uris) {
                    c_strings.push_back(str.c_str());
                }
                Array::consolidate(
                    ctx, uri, c_strings.data(), fragment_uris.size(), config);
            })
        .def_static(
            "_consolidate",
            [](const std::string& uri,
               const Context& ctx,
               const std::tuple<int, int>& timestamp,
               Config* config) {
                int start, end;
                std::tie(start, end) = timestamp;

                config->set(
                    "sm.consolidation.timestamp_start", std::to_string(start));
                config->set(
                    "sm.consolidation.timestamp_end", std::to_string(end));

                Array::consolidate(ctx, uri, config);
            })
        .def("_vacuum", &Array::vacuum)
        .def(
            "_create",
            [](const Context& ctx,
               const std::string& uri,
               const ArraySchema& schema) {
                ctx.handle_error(tiledb_array_create(
                    ctx.ptr().get(), uri.c_str(), schema.ptr().get()));
            })
        .def(
            "_load_schema",
            nb::overload_cast<const Context&, const std::string&>(
                &Array::load_schema))
        .def("_encryption_type", &Array::encryption_type)

        .def(
            "_non_empty_domain_is_empty",
            [](Array& self,
               const unsigned& dim_idx,
               const nb::dtype& n_type,
               const Context& ctx) -> bool {
                int32_t is_empty = 0;

                if (nb::getattr(n_type, "kind").is(nb::str("S")) ||
                    nb::getattr(n_type, "kind").is(nb::str("U"))) {
                    uint64_t start_size, end_size;
                    ctx.handle_error(
                        tiledb_array_get_non_empty_domain_var_size_from_index(
                            ctx.ptr().get(),
                            self.ptr().get(),
                            dim_idx,
                            &start_size,
                            &end_size,
                            &is_empty));
                    return is_empty == 1;
                } else {
                    void* domain = nullptr;
                    if (n_type.is(nb::dtype<uint64_t>())) {
                        return _non_empty_domain_is_empty_aux<uint64_t>(
                            self, dim_idx, ctx);
                        // int64_t also used for datetime64
                    } else if (
                        n_type.is(nb::dtype<int64_t>()) ||
                        nb::getattr(n_type, "kind").is(nb::str("M"))) {
                        return _non_empty_domain_is_empty_aux<int64_t>(
                            self, dim_idx, ctx);
                    } else if (n_type.is(nb::dtype<uint32_t>())) {
                        return _non_empty_domain_is_empty_aux<uint32_t>(
                            self, dim_idx, ctx);
                    } else if (n_type.is(nb::dtype<int32_t>())) {
                        return _non_empty_domain_is_empty_aux<int32_t>(
                            self, dim_idx, ctx);
                    } else if (n_type.is(nb::dtype<uint16_t>())) {
                        return _non_empty_domain_is_empty_aux<uint16_t>(
                            self, dim_idx, ctx);
                    } else if (n_type.is(nb::dtype<int16_t>())) {
                        return _non_empty_domain_is_empty_aux<int16_t>(
                            self, dim_idx, ctx);
                    } else if (n_type.is(nb::dtype<uint8_t>())) {
                        return _non_empty_domain_is_empty_aux<uint8_t>(
                            self, dim_idx, ctx);
                    } else if (n_type.is(nb::dtype<int8_t>())) {
                        return _non_empty_domain_is_empty_aux<int8_t>(
                            self, dim_idx, ctx);
                    } else if (n_type.is(nb::dtype<double>())) {
                        return _non_empty_domain_is_empty_aux<double>(
                            self, dim_idx, ctx);
                    } else if (n_type.is(nb::dtype<float>())) {
                        return _non_empty_domain_is_empty_aux<float>(
                            self, dim_idx, ctx);
                    } else {
                        TPY_ERROR_LOC("Unsupported type");
                    }
                }
            })
        .def(
            "_non_empty_domain",
            [](Array& self,
               const unsigned& dim_idx,
               const nb::dtype& n_type) -> nb::tuple {
                if (n_type.is(nb::dtype<uint64_t>())) {
                    auto domain = self.non_empty_domain<uint64_t>(dim_idx);
                    return nb::make_tuple(domain.first, domain.second);
                } else if (n_type.is(nb::dtype<int64_t>())) {
                    auto domain = self.non_empty_domain<int64_t>(dim_idx);
                    return nb::make_tuple(domain.first, domain.second);
                } else if (n_type.is(nb::dtype<uint32_t>())) {
                    auto domain = self.non_empty_domain<uint32_t>(dim_idx);
                    return nb::make_tuple(domain.first, domain.second);
                } else if (n_type.is(nb::dtype<int32_t>())) {
                    auto domain = self.non_empty_domain<int32_t>(dim_idx);
                    return nb::make_tuple(domain.first, domain.second);
                } else if (n_type.is(nb::dtype<uint16_t>())) {
                    auto domain = self.non_empty_domain<uint16_t>(dim_idx);
                    return nb::make_tuple(domain.first, domain.second);
                } else if (n_type.is(nb::dtype<int16_t>())) {
                    auto domain = self.non_empty_domain<int16_t>(dim_idx);
                    return nb::make_tuple(domain.first, domain.second);
                } else if (n_type.is(nb::dtype<uint8_t>())) {
                    auto domain = self.non_empty_domain<uint8_t>(dim_idx);
                    return nb::make_tuple(domain.first, domain.second);
                } else if (n_type.is(nb::dtype<int8_t>())) {
                    auto domain = self.non_empty_domain<int8_t>(dim_idx);
                    return nb::make_tuple(domain.first, domain.second);
                } else if (n_type.is(nb::dtype<double>())) {
                    auto domain = self.non_empty_domain<double>(dim_idx);
                    return nb::make_tuple(domain.first, domain.second);
                } else if (n_type.is(nb::dtype<float>())) {
                    auto domain = self.non_empty_domain<float>(dim_idx);
                    return nb::make_tuple(domain.first, domain.second);
                } else if (nb::getattr(n_type, "kind").is(nb::str("S"))) {
                    auto domain = self.non_empty_domain_var(dim_idx);
                    return nb::make_tuple(
                        nb::bytes(domain.first), nb::bytes(domain.second));
                } else if (nb::getattr(n_type, "kind").is(nb::str("U"))) {
                    TPY_ERROR_LOC(
                        "Unicode strings are not supported as dimension types");
                    // np.datetime64
                } else if (nb::getattr(n_type, "kind").is(nb::str("M"))) {
                    auto domain = self.non_empty_domain<int64_t>(dim_idx);
                    return nb::make_tuple(domain.first, domain.second);
                } else {
                    TPY_ERROR_LOC("Unsupported type");
                }
            })

        .def("_query_type", &Array::query_type)
        .def("_delete_fragments", &Array::delete_fragments)
        .def(
            "_consolidate_fragments",
            [](const std::string& uri,
               const Context& ctx,
               const std::vector<std::string>& fragment_uris,
               Config* config) {
                std::vector<const char*> c_strings;
                c_strings.reserve(fragment_uris.size());
                for (const auto& str : fragment_uris) {
                    c_strings.push_back(str.c_str());
                }
                ctx.handle_error(tiledb_array_consolidate_fragments(
                    ctx.ptr().get(),
                    uri.c_str(),
                    c_strings.data(),
                    fragment_uris.size(),
                    config->ptr().get()));
            })
        .def(
            "_consolidate_metadata",
            nb::overload_cast<
                const Context&,
                const std::string&,
                Config* const>(&Array::consolidate_metadata))
        .def(
            "_put_metadata",
            [](Array& array, const std::string& key, nb::array value) {
                MetadataAdapter<Array> a;
                a.put_metadata_numpy(array, key, value);
            })
        .def(
            "_put_metadata",
            [](Array& array,
               const std::string& key,
               tiledb_datatype_t value_type,
               uint32_t value_num,
               nb::buffer value) {
                MetadataAdapter<Array> a;
                a.put_metadata(array, key, value_type, value_num, value);
            })
        .def(
            "_get_metadata",
            [](Array& array, const std::string& key, bool is_ndarray) {
                MetadataAdapter<Array> a;
                return a.get_metadata(array, key, is_ndarray);
            },
            nb::arg("key"),
            nb::arg("is_ndarray") = false)
        .def(
            "_get_metadata_from_index",
            [](Array& self, uint64_t index) -> nb::tuple {
                tiledb_datatype_t tdb_type;
                uint32_t value_num = 0;
                const void* data_ptr = nullptr;
                std::string key;

                self.get_metadata_from_index(
                    index, &key, &tdb_type, &value_num, &data_ptr);

                if (data_ptr == nullptr && value_num != 1) {
                    throw nb::key_error();
                }
                // TODO handle empty value case

                assert(data_ptr != nullptr);
                auto buf = nb::memoryview::from_memory(
                    data_ptr, value_num * tiledb_datatype_size(tdb_type));

                return nb::make_tuple(tdb_type, buf);
            })
        .def(
            "_get_key_from_index",
            [](Array& array, uint64_t index) {
                MetadataAdapter<Array> a;
                return a.get_key_from_index(array, index);
            })
        .def("_delete_metadata", &Array::delete_metadata)
        .def(
            "_has_metadata",
            [](Array& array, const std::string& key) {
                MetadataAdapter<Array> a;
                return a.has_metadata(array, key);
            })
        .def("_metadata_num", &Array::metadata_num)
        .def(
            "_delete_array",
            nb::overload_cast<const Context&, const std::string&>(
                &Array::delete_array))
        .def("_upgrade_version", &Array::upgrade_version);
}

}  // namespace libtiledbcpp
