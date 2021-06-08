#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <exception>

#define TILEDB_DEPRECATED
#define TILEDB_DEPRECATED_EXPORT

#include "util.h"
#include <tiledb/tiledb> // C++

#if TILEDB_VERSION_MAJOR == 2 && TILEDB_VERSION_MINOR >= 2

#if !defined(NDEBUG)
#include "debug.cc"
#endif

namespace tiledbpy {

using namespace std;
using namespace tiledb;
namespace py = pybind11;
using namespace pybind11::literals;

class PyQueryCondition {

private:
  Context ctx_;
  shared_ptr<QueryCondition> qc_;

private:
  PyQueryCondition(shared_ptr<QueryCondition> qc, tiledb_ctx_t *c_ctx)
      : qc_(qc) {
    ctx_ = Context(c_ctx, false);
  }

  void set_ctx(py::object ctx) {
    tiledb_ctx_t *c_ctx;
    if ((c_ctx = (py::capsule)ctx.attr("__capsule__")()) == nullptr)
      TPY_ERROR_LOC("Invalid context pointer!")
    ctx_ = Context(c_ctx, false);
  }

public:
  PyQueryCondition() = delete;

  PyQueryCondition(py::object ctx) {
    try {
      set_ctx(ctx);
      qc_ = shared_ptr<QueryCondition>(new QueryCondition(ctx_));
    } catch (TileDBError &e) {
      TPY_ERROR_LOC(e.what());
    }
  }

  PyQueryCondition(const string &attribute_name, const string &condition_value,
                   tiledb_query_condition_op_t op, py::object ctx) {
    try {
      set_ctx(ctx);
      qc_ = shared_ptr<QueryCondition>(new QueryCondition(ctx_));
      qc_->init(attribute_name, condition_value, op);
    } catch (TileDBError &e) {
      TPY_ERROR_LOC(e.what());
    }
  }

  template <typename T>
  PyQueryCondition(const string &attribute_name, T condition_value,
                   tiledb_query_condition_op_t op, py::object ctx) {
    try {
      set_ctx(ctx);
      qc_ = shared_ptr<QueryCondition>(new QueryCondition(ctx_));
      qc_->init(attribute_name, &condition_value, sizeof(condition_value), op);
    } catch (TileDBError &e) {
      TPY_ERROR_LOC(e.what());
    }
  }

  shared_ptr<QueryCondition> ptr() { return qc_; }

  PyQueryCondition
  combine(PyQueryCondition rhs,
          tiledb_query_condition_combination_op_t combination_op) const {

    auto pyqc = PyQueryCondition(nullptr, ctx_.ptr().get());

    tiledb_query_condition_t *combined_qc = nullptr;
    ctx_.handle_error(
        tiledb_query_condition_alloc(ctx_.ptr().get(), &combined_qc));

    ctx_.handle_error(tiledb_query_condition_combine(
        ctx_.ptr().get(), qc_->ptr().get(), rhs.qc_->ptr().get(),
        combination_op, &combined_qc));

    pyqc.qc_ = std::shared_ptr<QueryCondition>(
        new QueryCondition(pyqc.ctx_, combined_qc));

    return pyqc;
  }

}; // namespace tiledbpy

PYBIND11_MODULE(_query_condition, m) {
  py::class_<PyQueryCondition>(m, "qc")
      .def(py::init<py::object>(), py::arg("ctx") = py::none())
      .def(py::init<const string &, const string &, tiledb_query_condition_op_t,
                    py::object>(),
           py::arg("attribute_name"), py::arg("condition_value"),
           py::arg("tiledb_query_condition_op_t"), py::arg("ctx") = py::none())
      .def(py::init<const string &, double, tiledb_query_condition_op_t,
                    py::object>(),
           py::arg("attribute_name"), py::arg("condition_value"),
           py::arg("tiledb_query_condition_op_t"), py::arg("ctx") = py::none())
      .def(py::init<const string &, float, tiledb_query_condition_op_t,
                    py::object>(),
           py::arg("attribute_name"), py::arg("condition_value"),
           py::arg("tiledb_query_condition_op_t"), py::arg("ctx") = py::none())
      .def(py::init<const string &, int, tiledb_query_condition_op_t,
                    py::object>(),
           py::arg("attribute_name"), py::arg("condition_value"),
           py::arg("tiledb_query_condition_op_t"), py::arg("ctx") = py::none())

      .def("combine", &PyQueryCondition::combine, py::arg("rhs"),
           py::arg("combination_op"));

  py::enum_<tiledb_query_condition_op_t>(m, "tiledb_query_condition_op_t",
                                         py::arithmetic())
      .value("TILEDB_LT", TILEDB_LT)
      .value("TILEDB_LE", TILEDB_LE)
      .value("TILEDB_GT", TILEDB_GT)
      .value("TILEDB_GE", TILEDB_GE)
      .value("TILEDB_EQ", TILEDB_EQ)
      .value("TILEDB_NE", TILEDB_NE)
      .export_values();

  py::enum_<tiledb_query_condition_combination_op_t>(
      m, "tiledb_query_condition_combination_op_t", py::arithmetic())
      .value("TILEDB_AND", TILEDB_AND)
      .export_values();
}
}; // namespace tiledbpy

#endif
