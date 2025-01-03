#include <tiledb/tiledb.h> // for enums
#include <tiledb/tiledb>   // C++

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

using namespace tiledb;
namespace py = pybind11;
using namespace pybind11::literals;

#define TPY_ERROR_LOC(m)                                                       \
  throw TileDBPyError(std::string(m) + " (" + __FILE__ + ":" +                 \
                      std::to_string(__LINE__) + ")");

class TileDBPyError : std::runtime_error {
public:
  explicit TileDBPyError(const char *m) : std::runtime_error(m) {}
  explicit TileDBPyError(std::string m) : std::runtime_error(m.c_str()) {}

public:
  virtual const char *what() const noexcept override {
    return std::runtime_error::what();
  }
};

namespace tiledbpy::common {

size_t buffer_nbytes(py::buffer_info &info);

bool expect_buffer_nbytes(py::buffer_info &info, tiledb_datatype_t datatype,
                          size_t nbytes);

} // namespace tiledbpy::common

py::dtype tdb_to_np_dtype(tiledb_datatype_t type, uint32_t cell_val_num);
tiledb_datatype_t np_to_tdb_dtype(py::dtype type);

bool is_tdb_num(tiledb_datatype_t type);
bool is_tdb_str(tiledb_datatype_t type);

py::size_t get_ncells(py::dtype type);