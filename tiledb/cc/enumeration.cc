#include <tiledb/tiledb>
#include <tiledb/tiledb_experimental>

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "common.h"

namespace libtiledbcpp {

using namespace tiledb;
using namespace tiledbpy::common;
namespace py = pybind11;

void init_enumeration(py::module &m) {
  py::class_<Enumeration>(m, "Enumeration")
      .def(py::init<Enumeration>())

      .def(py::init([](const Context &ctx, const std::string &name,
                       py::dtype type, bool ordered) {
        tiledb_datatype_t data_type;
        try {
          data_type = np_to_tdb_dtype(type);
        } catch (const TileDBPyError &e) {
          throw py::type_error(e.what());
        }
        py::size_t cell_val_num = get_ncells(type);

        return Enumeration::create_empty(ctx, name, data_type, cell_val_num,
                                         ordered);
      }))

      .def(py::init([](const Context &ctx, const std::string &name,
                       std::vector<std::string> &values, bool ordered,
                       tiledb_datatype_t type) {
        return Enumeration::create(ctx, name, values, ordered, type);
      }))

      .def(py::init([](const Context &ctx, const std::string &name,
                       bool ordered, py::array data, py::array offsets) {
        tiledb_datatype_t data_type;
        try {
          data_type = np_to_tdb_dtype(data.dtype());
        } catch (const TileDBPyError &e) {
          throw py::type_error(e.what());
        }

        py::buffer_info data_buffer = data.request();
        if (data_buffer.ndim != 1)
          throw py::type_error("Only 1D Numpy arrays can be stored as "
                               "enumeration values");

        py::size_t cell_val_num =
            offsets.size() == 0 ? get_ncells(data.dtype()) : TILEDB_VAR_NUM;

        return Enumeration::create(
            ctx, name, data_type, cell_val_num, ordered, data.data(),
            data.nbytes(), offsets.size() == 0 ? nullptr : offsets.data(),
            offsets.nbytes());
      }))

      .def(py::init<const Context &, py::capsule>(), py::keep_alive<1, 2>())

      .def("__capsule__",
           [](Enumeration &enmr) {
             return py::capsule(enmr.ptr().get(), "enmr");
           })

      .def_property_readonly("name", &Enumeration::name)

      .def_property_readonly("type", &Enumeration::type)

      .def_property_readonly("cell_val_num", &Enumeration::cell_val_num)

      .def_property_readonly("ordered", &Enumeration::ordered)

      .def("values",
           [](Enumeration &enmr) {
             auto data = enmr.as_vector<std::byte>();
             auto dtype = tdb_to_np_dtype(enmr.type(), enmr.cell_val_num());
             return py::array(dtype, data.size() / dtype.itemsize(),
                              data.data());
           })
      .def("str_values",
           [](Enumeration &enmr) { return enmr.as_vector<std::string>(); })

      .def(
          "extend",
          static_cast<Enumeration (Enumeration::*)(std::vector<std::string> &)>(
              &Enumeration::extend))
      .def("extend", [](Enumeration &enmr, py::array data) {
        return enmr.extend(data.data(), data.nbytes(), nullptr, 0);
      });
}

} // namespace libtiledbcpp
