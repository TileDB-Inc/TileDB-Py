#include <pybind11/pybind11.h>

namespace tiledbpy {

namespace py = pybind11;

void init_core(py::module &);
// void _debug(py::module &);
void init_fragment(py::module &);
// void init_query_condition(py::module &);
void init_schema_evolution(py::module &);
#if defined(TILEDB_SERIALIZATION)
void init_serialization(py::module &);
void init_test_serialization(py::module &);
#endif
void init_test_metadata(py::module &);
void init_test_webp_filter(py::module &);

PYBIND11_MODULE(main, m) {
  init_core(m);
  //_debug(m);
  init_fragment(m);
  //_query_condition(m);
  init_schema_evolution(m);
#if defined(TILEDB_SERIALIZATION)
  init_serialization(m);
  init_test_serialization(m);
#endif
  init_test_metadata(m);
  init_test_webp_filter(m);
}

} // namespace tiledbpy
