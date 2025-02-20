#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include "util.h"

namespace tiledbpy {

namespace nb = nanobind;
using namespace pybind11::literals;

py::tuple convert_np(
    py::array input, bool allow_unicode = true, bool use_fallback = false);

}  // namespace tiledbpy
