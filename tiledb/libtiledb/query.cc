#include <tiledb/tiledb>  // C++
#include <tiledb/tiledb_experimental>

#include "common.h"

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

// #pragma clang diagnostic ignored "-Wdeprecated-declarations"
// #pragma gcc diagnostic ignored "-Wdeprecated-declarations"

namespace libtiledbcpp {

using namespace tiledb;
using namespace std;
namespace nb = nanobind;

void init_query(nb::module& m) {
    nb::class_<tiledb::Query>(m, "Query")

        //.def(nb::init<nb::object, nb::object, nb::iterable, nb::object,
        //              nb::object, nb::object>())

        .def(
            nb::init<Context&, Array&, tiledb_query_type_t>(),
            nb::keep_alive<1, 2>() /* Keep context alive. */,
            nb::keep_alive<1, 3>() /* Keep array alive. */)

        .def(
            nb::init<Context&, Array&>(),
            nb::keep_alive<1, 2>() /* Keep context alive. */,
            nb::keep_alive<1, 3>() /* Keep array alive. */)

        // TODO .def("ptr", [&]() -> nb::capsule)

        .def_prop_rw("layout", &Query::query_layout, &Query::set_layout)

        .def_prop_rw_readonly("query_type", &Query::query_type)

        .def_prop_rw_readonly(
            "_subarray",
            [](Query& query) {
                // TODO: Before merge make sure the lifetime of
                // the resulting subarray is not tied to this
                // query.
                Subarray subarray(query.ctx(), query.array());
                query.update_subarray_from_query(&subarray);
                return subarray;
            })

        // TODO .def("array") -> Array&

        .def("has_results", &Query::has_results)

        .def(
            "is_complete",
            [](const Query& query) {
                return query.query_status() == Query::Status::COMPLETE;
            })

        .def("finalize", &Query::finalize)

        .def("fragment_num", &Query::fragment_num)

        .def("fragment_uri", &Query::fragment_uri)

        .def("fragment_timestamp_range", &Query::fragment_timestamp_range)

        .def("query_status", &Query::query_status)

        .def("set_condition", &Query::set_condition)

        //.def("set_data_buffer",
        //     (Query& (Query::*)(const std::string&, void*,
        //     uint64_t))&Query::set_data_buffer);

        .def(
            "set_data_buffer",
            [](Query& q, std::string name, nb::array a, uint64_t nelements) {
                QueryExperimental::set_data_buffer(
                    q, name, const_cast<void*>(a.data()), nelements);
            })

        .def(
            "set_offsets_buffer",
            [](Query& q, std::string name, nb::array a, uint64_t nelements) {
                q.set_offsets_buffer(name, (uint64_t*)(a.data()), nelements);
            })

        .def(
            "set_subarray",
            [](Query& query, const Subarray& subarray) {
                return query.set_subarray(subarray);
            })

        .def(
            "set_validity_buffer",
            [](Query& q, std::string name, nb::array a, uint64_t nelements) {
                q.set_validity_buffer(name, (uint8_t*)(a.data()), nelements);
            })

        .def(
            "_submit", &Query::submit, nb::call_guard<nb::gil_scoped_release>())

        /** hackery from another branch... */
        //.def("set_fragment_uri", &Query::set_fragment_uri)
        //.def("unset_buffer", &Query::unset_buffer)
        //.def("set_continuation", [](Query& q) {
        //  q.ctx().handle_error(
        //    tiledb_query_set_continuation(q.ctx().ptr().get(), q.ptr().get())
        //  );
        //})
        ;
}

}  // namespace libtiledbcpp
