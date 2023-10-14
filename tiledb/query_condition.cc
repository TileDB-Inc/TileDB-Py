#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <exception>

#include "util.h"
#include <tiledb/tiledb> // C++

#if TILEDB_VERSION_MAJOR == 2 && TILEDB_VERSION_MINOR >= 2

#if !defined(NDEBUG)
// #include "debug.cc"
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

  void init(const string &attribute_name, const string &condition_value,
            tiledb_query_condition_op_t op) {
    try {
      qc_->init(attribute_name, condition_value, op);
    } catch (TileDBError &e) {
      TPY_ERROR_LOC(e.what());
    }
  }

  template <typename T>
  void init(const string &attribute_name, T condition_value,
            tiledb_query_condition_op_t op) {
    try {
      qc_->init(attribute_name, &condition_value, sizeof(condition_value), op);
    } catch (TileDBError &e) {
      TPY_ERROR_LOC(e.what());
    }
  }

  shared_ptr<QueryCondition> ptr() { return qc_; }

  py::capsule __capsule__() { return py::capsule(&qc_, "qc", nullptr); }

  void set_use_enumeration(bool use_enumeration) {
    QueryConditionExperimental::set_use_enumeration(ctx_, *qc_,
                                                    use_enumeration);
  }

  template <typename T>
  static PyQueryCondition
  create(py::object pyctx, const std::string &field_name,
         const std::vector<T> &values, tiledb_query_condition_op_t op) {
    auto pyqc = PyQueryCondition(pyctx);

    const Context ctx = std::as_const(pyqc.ctx_);

    auto set_membership_qc =
        QueryConditionExperimental::create(ctx, field_name, values, op);

    pyqc.qc_ = std::make_shared<QueryCondition>(std::move(set_membership_qc));

    return pyqc;
  }

  PyQueryCondition
  combine(PyQueryCondition qc,
          tiledb_query_condition_combination_op_t combination_op) const {
    auto pyqc = PyQueryCondition(nullptr, ctx_.ptr().get());

    tiledb_query_condition_t *combined_qc = nullptr;
    ctx_.handle_error(
        tiledb_query_condition_alloc(ctx_.ptr().get(), &combined_qc));

    ctx_.handle_error(tiledb_query_condition_combine(
        ctx_.ptr().get(), qc_->ptr().get(), qc.qc_->ptr().get(), combination_op,
        &combined_qc));

    pyqc.qc_ = std::shared_ptr<QueryCondition>(
        new QueryCondition(pyqc.ctx_, combined_qc));

    return pyqc;
  }

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
}; // namespace tiledbpy

void init_query_condition(py::module &m) {
  py::class_<PyQueryCondition>(m, "PyQueryCondition")
      .def(py::init<py::object>(), py::arg("ctx") = py::none())

      /* TODO surely there's a better way to deal with templated PyBind11
       * functions? but maybe not?
       * https://github.com/pybind/pybind11/issues/1667
       */

      .def("init_string",
           static_cast<void (PyQueryCondition::*)(
               const string &, const string &, tiledb_query_condition_op_t)>(
               &PyQueryCondition::init))
      .def("init_uint64",
           static_cast<void (PyQueryCondition::*)(const string &, uint64_t,
                                                  tiledb_query_condition_op_t)>(
               &PyQueryCondition::init))
      .def("init_int64",
           static_cast<void (PyQueryCondition::*)(const string &, int64_t,
                                                  tiledb_query_condition_op_t)>(
               &PyQueryCondition::init))
      .def("init_uint32",
           static_cast<void (PyQueryCondition::*)(const string &, uint32_t,
                                                  tiledb_query_condition_op_t)>(
               &PyQueryCondition::init))
      .def("init_int32",
           static_cast<void (PyQueryCondition::*)(const string &, int32_t,
                                                  tiledb_query_condition_op_t)>(
               &PyQueryCondition::init))
      .def("init_uint16",
           static_cast<void (PyQueryCondition::*)(const string &, uint16_t,
                                                  tiledb_query_condition_op_t)>(
               &PyQueryCondition::init))
      .def("init_int16",
           static_cast<void (PyQueryCondition::*)(const string &, int16_t,
                                                  tiledb_query_condition_op_t)>(
               &PyQueryCondition::init))
      .def("init_uint8",
           static_cast<void (PyQueryCondition::*)(const string &, uint8_t,
                                                  tiledb_query_condition_op_t)>(
               &PyQueryCondition::init))
      .def("init_int8",
           static_cast<void (PyQueryCondition::*)(const string &, int8_t,
                                                  tiledb_query_condition_op_t)>(
               &PyQueryCondition::init))
      .def("init_float32",
           static_cast<void (PyQueryCondition::*)(const string &, float,
                                                  tiledb_query_condition_op_t)>(
               &PyQueryCondition::init))
      .def("init_float64",
           static_cast<void (PyQueryCondition::*)(const string &, double,
                                                  tiledb_query_condition_op_t)>(
               &PyQueryCondition::init))

      .def("__capsule__", &PyQueryCondition::__capsule__)

      .def("combine", &PyQueryCondition::combine)

      .def_static(
          "create_string",
          static_cast<PyQueryCondition (*)(
              py::object, const std::string &, const std::vector<std::string> &,
              tiledb_query_condition_op_t)>(&PyQueryCondition::create))
      .def_static(
          "create_uint64",
          static_cast<PyQueryCondition (*)(
              py::object, const std::string &, const std::vector<uint64_t> &,
              tiledb_query_condition_op_t)>(&PyQueryCondition::create))
      .def_static(
          "create_int64",
          static_cast<PyQueryCondition (*)(
              py::object, const std::string &, const std::vector<int64_t> &,
              tiledb_query_condition_op_t)>(&PyQueryCondition::create))
      .def_static(
          "create_uint32",
          static_cast<PyQueryCondition (*)(
              py::object, const std::string &, const std::vector<uint32_t> &,
              tiledb_query_condition_op_t)>(&PyQueryCondition::create))
      .def_static(
          "create_int32",
          static_cast<PyQueryCondition (*)(
              py::object, const std::string &, const std::vector<int32_t> &,
              tiledb_query_condition_op_t)>(&PyQueryCondition::create))
      .def_static(
          "create_uint16",
          static_cast<PyQueryCondition (*)(
              py::object, const std::string &, const std::vector<uint16_t> &,
              tiledb_query_condition_op_t)>(&PyQueryCondition::create))
      .def_static(
          "create_int8",
          static_cast<PyQueryCondition (*)(
              py::object, const std::string &, const std::vector<int8_t> &,
              tiledb_query_condition_op_t)>(&PyQueryCondition::create))
      .def_static(
          "create_uint16",
          static_cast<PyQueryCondition (*)(
              py::object, const std::string &, const std::vector<uint16_t> &,
              tiledb_query_condition_op_t)>(&PyQueryCondition::create))
      .def_static(
          "create_int8",
          static_cast<PyQueryCondition (*)(
              py::object, const std::string &, const std::vector<int8_t> &,
              tiledb_query_condition_op_t)>(&PyQueryCondition::create))
      .def_static(
          "create_float32",
          static_cast<PyQueryCondition (*)(
              py::object, const std::string &, const std::vector<float> &,
              tiledb_query_condition_op_t)>(&PyQueryCondition::create))
      .def_static(
          "create_float64",
          static_cast<PyQueryCondition (*)(
              py::object, const std::string &, const std::vector<double> &,
              tiledb_query_condition_op_t)>(&PyQueryCondition::create));

  py::enum_<tiledb_query_condition_op_t>(m, "tiledb_query_condition_op_t",
                                         py::arithmetic())
      .value("TILEDB_LT", TILEDB_LT)
      .value("TILEDB_LE", TILEDB_LE)
      .value("TILEDB_GT", TILEDB_GT)
      .value("TILEDB_GE", TILEDB_GE)
      .value("TILEDB_EQ", TILEDB_EQ)
      .value("TILEDB_NE", TILEDB_NE)
      .value("TILEDB_IN", TILEDB_IN)
      .value("TILEDB_NOT_IN", TILEDB_NOT_IN)
      .export_values();

  py::enum_<tiledb_query_condition_combination_op_t>(
      m, "tiledb_query_condition_combination_op_t", py::arithmetic())
      .value("TILEDB_AND", TILEDB_AND)
      .value("TILEDB_OR", TILEDB_OR)
      .export_values();
}
}; // namespace tiledbpy

#endif
