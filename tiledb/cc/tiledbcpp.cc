#include <tiledb/tiledb> // C++

#include "common.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

namespace libtiledbcpp {

using namespace tiledb;
namespace py = pybind11;

template <typename... Args>
using overload_cast_ = pybind11::detail::overload_cast_impl<Args...>;

void init_array(py::module &);
void init_attribute(py::module &);
void init_context(py::module &);
void init_config(py::module &);
void init_enums(py::module &);
void init_domain(py::module &m);
void init_file_handle(py::module &);
void init_filestore(py::module &m);
void init_filter(py::module &);
void init_group(py::module &);
void init_object(py::module &m);
void init_query(py::module &m);
void init_schema(py::module &);
void init_subarray(py::module &);
void init_vfs(py::module &m);

PYBIND11_MODULE(cc, m) {

  init_array(m);
  init_attribute(m);
  init_context(m);
  init_config(m);
  init_domain(m);
  init_enums(m);
  init_file_handle(m);
  init_filestore(m);
  init_filter(m);
  init_group(m);
  init_object(m);
  init_query(m);
  init_schema(m);
  init_subarray(m);
  init_vfs(m);

  py::register_exception<TileDBError>(m, "TileDBError");

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p)
        std::rethrow_exception(p);
    } catch (const TileDBPyError &e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    } catch (const tiledb::TileDBError &e) {
      PyErr_SetString(PyExc_RuntimeError, e.what());
    } catch (py::builtin_exception &e) {
      throw;
    };
  });
}

}; // namespace libtiledbcpp
