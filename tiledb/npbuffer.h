#include "util.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

namespace tiledbpy {

namespace py = pybind11;
using namespace pybind11::literals;

py::tuple convert_np(py::array input, bool allow_unicode = true,
                     bool use_fallback = false);

} // namespace tiledbpy