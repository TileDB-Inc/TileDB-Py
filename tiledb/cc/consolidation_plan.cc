#include <tiledb/tiledb>
#include <tiledb/tiledb_experimental>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "common.h"

namespace libtiledbcpp {

using namespace tiledb;
using namespace tiledbpy::common;
namespace py = pybind11;

void init_consolidation_plan(py::module &m) {
  py::class_<ConsolidationPlan>(m, "ConsolidationPlan")

      .def(py::init<Context &, Array &, uint64_t>(), py::keep_alive<1, 2>(),
           py::keep_alive<1, 3>())

      .def_property_readonly("_num_nodes", &ConsolidationPlan::num_nodes)
      .def("_num_fragments", &ConsolidationPlan::num_fragments)
      .def("_fragment_uri", &ConsolidationPlan::fragment_uri)
      .def("_dump", &ConsolidationPlan::dump);
}
} // namespace libtiledbcpp
