#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#include <exception>

#include "util.h"
#include <tiledb/tiledb> // C++

#if TILEDB_VERSION_MAJOR == 2 && TILEDB_VERSION_MINOR >= 2

#if !defined(NDEBUG)
//#include "debug.cc"
#endif

namespace tiledbpy {

using namespace std;
using namespace tiledb;
namespace py = pybind11;
using namespace pybind11::literals;

class PyQueryCondition : public QueryCondition {

private:
  const Context ctx_;
  //shared_ptr<QueryCondition> qc_;

public:
  PyQueryCondition() = delete;

  // Yes, we are making two non-owning Context because this class needs access too
  PyQueryCondition(py::object ctx) :
    QueryCondition(get_ctx(ctx)), ctx_(get_ctx(ctx)) {}

  PyQueryCondition(const Context& ctx, QueryCondition qc) :
    QueryCondition(std::move(qc)), ctx_(ctx) {};

  //void init(const string &attribute_name, const string &condition_value,
  //          tiledb_query_condition_op_t op) {
  //  try {
  //    this->init(attribute_name, condition_value, op);
  //  } catch (TileDBError &e) {
  //    TPY_ERROR_LOC(e.what());
  //  }
  //}

  template <typename T>
  void init_pyqc(const string &attribute_name, T condition_value,
            tiledb_query_condition_op_t op) {
    size_t val_size = 0;
    std::cout << typeid(T).name() << std::endl;
    if constexpr (std::is_same_v<T, std::string>) {
      val_size = condition_value.size();
    } else {
      val_size = sizeof(T);
    }
    try {
      this->init(attribute_name, &condition_value, val_size, op);
    } catch (TileDBError &e) {
      TPY_ERROR_LOC(e.what());
    }
  }

  //shared_ptr<QueryCondition> ptr() { return ; }

  py::capsule __capsule__() { return py::capsule(this, "qc", nullptr); }

  PyQueryCondition
  combine(PyQueryCondition rhs,
          tiledb_query_condition_combination_op_t combination_op) const {

    //auto pyqc = PyQueryCondition(nullptr, ctx_.ptr().get());

    tiledb_query_condition_t *combined_qc = nullptr;
    ctx_.handle_error(
        tiledb_query_condition_alloc(ctx_.ptr().get(), &combined_qc));

    ctx_.handle_error(tiledb_query_condition_combine(
        ctx_.ptr().get(), this->ptr().get(), rhs.ptr().get(),
        combination_op, &combined_qc));

    //pyqc.qc_ = std::shared_ptr<QueryCondition>(
    //    new QueryCondition(pyqc.ctx_, combined_qc));

    //return std::make_shared<const PyQueryCondition>(ctx_, QueryCondition(ctx_, combined_qc));


    return PyQueryCondition(ctx_, QueryCondition(ctx_, combined_qc));
  }

private:
  //PyQueryCondition(shared_ptr<QueryCondition> qc, tiledb_ctx_t *c_ctx)
  //    : QueryCondition(Context(c_ctx, false))  {
  //  //ctx_ = Context(c_ctx, false);
  //}

  //void set_ctx(py::object ctx) {
  //  ctx_ = get_ctx(ctx);
  //}
  Context get_ctx(py::object ctx) {
    tiledb_ctx_t *c_ctx = (py::capsule)ctx.attr("__capsule__")();
    if (c_ctx == nullptr)
      TPY_ERROR_LOC("Invalid context pointer!")
    return Context(c_ctx, false);
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
               &PyQueryCondition::init_pyqc))

      .def("init_uint64",
           static_cast<void (PyQueryCondition::*)(const string &, uint64_t,
                                                  tiledb_query_condition_op_t)>(
               &PyQueryCondition::init_pyqc))
      .def("init_int64",
           static_cast<void (PyQueryCondition::*)(const string &, int64_t,
                                                  tiledb_query_condition_op_t)>(
               &PyQueryCondition::init_pyqc))
      .def("init_uint32",
           static_cast<void (PyQueryCondition::*)(const string &, uint32_t,
                                                  tiledb_query_condition_op_t)>(
               &PyQueryCondition::init_pyqc))
      .def("init_int32",
           static_cast<void (PyQueryCondition::*)(const string &, int32_t,
                                                  tiledb_query_condition_op_t)>(
               &PyQueryCondition::init_pyqc))
      .def("init_uint16",
           static_cast<void (PyQueryCondition::*)(const string &, uint16_t,
                                                  tiledb_query_condition_op_t)>(
               &PyQueryCondition::init_pyqc))
      .def("init_int16",
           static_cast<void (PyQueryCondition::*)(const string &, int16_t,
                                                  tiledb_query_condition_op_t)>(
               &PyQueryCondition::init_pyqc))
      .def("init_uint8",
           static_cast<void (PyQueryCondition::*)(const string &, uint8_t,
                                                  tiledb_query_condition_op_t)>(
               &PyQueryCondition::init_pyqc))
      .def("init_int8",
           static_cast<void (PyQueryCondition::*)(const string &, int8_t,
                                                  tiledb_query_condition_op_t)>(
               &PyQueryCondition::init_pyqc))

      .def("init_float32",
           static_cast<void (PyQueryCondition::*)(const string &, float,
                                                  tiledb_query_condition_op_t)>(
               &PyQueryCondition::init_pyqc))
      .def("init_float64",
           static_cast<void (PyQueryCondition::*)(const string &, double,
                                                  tiledb_query_condition_op_t)>(
               &PyQueryCondition::init_pyqc))

      .def("combine", &PyQueryCondition::combine)

      .def("__capsule__", &PyQueryCondition::__capsule__);

      //.def("__qc__", &PyQueryCondition::ptr);

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
      .value("TILEDB_OR", TILEDB_OR)
      .export_values();
}
}; // namespace tiledbpy

#endif
