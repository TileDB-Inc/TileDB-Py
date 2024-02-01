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


void init_query_channel(py::module &m) {
  py::class_<QueryChannel>(m, "QueryChannel")
    .def("apply_aggregate", &QueryChannel::apply_aggregate);
  py::class_<ChannelOperation>(m, "ChannelOperation");
}

} // namespace libtiledbcpp
