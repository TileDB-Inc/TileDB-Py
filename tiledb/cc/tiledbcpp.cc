#include <tiledb/tiledb> // C++

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
void init_schema(py::module &);
void init_context(py::module &);
void init_config(py::module &);
void init_filter(py::module &);
void init_enums(py::module &);
void init_domain(py::module &m);
void init_query(py::module &m);

PYBIND11_MODULE(cc, m) {

  init_array(m);
  init_attribute(m);
  init_schema(m);
  init_context(m);
  init_config(m);
  init_enums(m);
  init_filter(m);
  init_domain(m);
  init_query(m);

  py::register_exception<TileDBError>(m, "TileDBError");
}

}; // namespace libtiledbcpp