
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <exception>

#define TILEDB_DEPRECATED
#define TILEDB_DEPRECATED_EXPORT

#include "../util.h"
#include <tiledb/tiledb> // C++

#if !defined(NDEBUG)
// #include "debug.cc"
#endif

namespace tiledbpy {

using namespace std;
using namespace tiledb;
namespace py = pybind11;
using namespace pybind11::literals;

class PyASCIIMetadataTest {

public:
  static void write_ascii(py::str uri) {
    Context ctx;
    Array array(ctx, uri, TILEDB_WRITE);

    std::string st = "xyz";
    array.put_metadata("abc", TILEDB_STRING_ASCII, st.length(), st.c_str());

    array.close();
  }
};

void init_test_metadata(py::module &m) {
  py::class_<PyASCIIMetadataTest>(m, "metadata_test_aux")
      .def_static("write_ascii", &PyASCIIMetadataTest::write_ascii);
}

}; // namespace tiledbpy
