#include <tiledb/tiledb>
#include <tiledb/tiledb_experimental>

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// #include <pybind11/pytypes.h>
// #include <pybind11/stl.h>

#include "common.h"

namespace libtiledbcpp {

using namespace tiledb;
using namespace tiledbnb::common;
namespace nb = nanobind;

void init_consolidation_plan(nb::module_& m) {
    nb::class_<ConsolidationPlan>(m, "ConsolidationPlan")

        .def(
            nb::init<Context&, Array&, uint64_t>(),
            nb::keep_alive<1, 2>(),
            nb::keep_alive<1, 3>())

        .def_prop_rw_readonly("_num_nodes", &ConsolidationPlan::num_nodes)
        .def("_num_fragments", &ConsolidationPlan::num_fragments)
        .def("_fragment_uri", &ConsolidationPlan::fragment_uri)
        .def("_dump", &ConsolidationPlan::dump);
}
}  // namespace libtiledbcpp
