#include <tiledb/tiledb.h>  // for enums
#include <tiledb/tiledb>    // C++

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// #include <pybind11/pytypes.h>
// #include <pybind11/stl.h>

using namespace tiledb;
namespace nb = nanobind;
using namespace pybind11::literals;

#define TPY_ERROR_LOC(m)                                                    \
    throw TileDBPyError(                                                    \
        std::string(m) + " (" + __FILE__ + ":" + std::to_string(__LINE__) + \
        ")");

class TileDBPyError : std::runtime_error {
   public:
    explicit TileDBPyError(const char* m)
        : std::runtime_error(m) {
    }
    explicit TileDBPyError(std::string m)
        : std::runtime_error(m.c_str()) {
    }

   public:
    virtual const char* what() const noexcept override {
        return std::runtime_error::what();
    }
};

namespace tiledbnb::common {

size_t buffer_nbytes(nb::buffer_info& info);

bool expect_buffer_nbytes(
    nb::buffer_info& info, tiledb_datatype_t datatype, size_t nbytes);

}  // namespace tiledbnb::common

nb::dlpack::dtype tdb_to_np_dtype(
    tiledb_datatype_t type, uint32_t cell_val_num);
tiledb_datatype_t np_to_tdb_dtype(nb::dlpack::dtype type);

bool is_tdb_num(tiledb_datatype_t type);
bool is_tdb_str(tiledb_datatype_t type);

nb::size_t get_ncells(nb::dlpack::dtype type);
