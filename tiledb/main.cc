#include <pybind11/pybind11.h>

namespace tiledbpy {

namespace py = pybind11;

void init_core(py::module &);
// void _debug(py::module &);
void init_fragment(py::module &);
// void init_query_condition(py::module &);
void init_schema_evolution(py::module &);
void init_serialization(py::module &);
void init_test_serialization(py::module &);
void init_test_metadata(py::module &);

PYBIND11_MODULE(main, m) {
  init_core(m);
  //_debug(m);
  init_fragment(m);
  //_query_condition(m);
  init_schema_evolution(m);
  init_serialization(m);
  init_test_serialization(m);
  init_test_metadata(m);
}

} // namespace tiledbpy
