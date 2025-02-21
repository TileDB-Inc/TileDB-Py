#include <nanobind/nanobind.h>
// #include <pybind11/pytypes.h>

#include <tiledb/tiledb>

namespace tiledbpy {
using namespace tiledb;
namespace nb = nanobind;

class WebpFilter {
   public:
    static bool webp_filter_exists() {
        Context ctx;
        try {
            auto f = Filter(ctx, TILEDB_FILTER_WEBP);
        } catch (TileDBError&) {
            // Can't create WebP filter; built with TILEDB_WEBP=OFF
            return false;
        }
        return true;
    }
};

void init_test_webp_filter(nb::module_& m) {
    nb::class_<WebpFilter>(m, "test_webp_filter")
        .def_static("webp_filter_exists", &WebpFilter::webp_filter_exists);
}

};  // namespace tiledbpy
