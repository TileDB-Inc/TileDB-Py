
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// #include <pybind11/pytypes.h>
// #include <pybind11/stl.h>

#include <exception>

#define TILEDB_DEPRECATED
#define TILEDB_DEPRECATED_EXPORT

#include <tiledb/tiledb>  // C++
#include "../util.h"

#if !defined(NDEBUG)
// #include "debug.cc"
#endif

namespace tiledbpy {

using namespace std;
using namespace tiledb;
namespace nb = nanobind;
using namespace nb::literals;

class PyASCIIMetadataTest {
   public:
    static void write_ascii(const std::string uri) {
        Context ctx;
        Array array(ctx, uri, TILEDB_WRITE);

        std::string st = "xyz";
        array.put_metadata("abc", TILEDB_STRING_ASCII, st.length(), st.c_str());

        array.close();
    }
};

void init_test_metadata(nb::module_& m) {
    nb::class_<PyASCIIMetadataTest>(m, "metadata_test_aux")
        .def_static("write_ascii", &PyASCIIMetadataTest::write_ascii);
}

};  // namespace tiledbpy
