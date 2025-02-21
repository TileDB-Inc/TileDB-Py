#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// #include <pybind11/pytypes.h>
#include "util.h"

namespace tiledbpy {

namespace nb = nanobind;
using namespace pybind11::literals;

nb::tuple convert_np(
    nb::array input, bool allow_unicode = true, bool use_fallback = false);

}  // namespace tiledbpy
