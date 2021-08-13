#include <pybind11/pybind11.h>

namespace tiledbpy {

namespace py = pybind11;

void _core(py::module_ &);
void _debug(py::module_ &);
void _fragment(py::module_ &);
void _npbuffer(py::module_ &);
void _query_condition(py::module_ &);
void _serialization(py::module_ &);

PYBIND11_MODULE(main, m) {
  _core(m);
  _debug(m);
  _fragment(m);
  _npbuffer(m);
  _query_condition(m);
  _serialization(m);
}

} // namespace tiledbpy