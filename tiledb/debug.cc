#include <pybind11/embed.h>

#ifndef TILEDBPY_DEBUGCC
#define TILEDBPY_DEBUGCC

namespace {
extern "C" {

namespace py = pybind11;
using namespace pybind11::literals;

// __attribute__((used)) to make the linker keep the symbol
__attribute__((used)) static void pyprint(pybind11::object o) {
  pybind11::print(o);
}

__attribute__((used)) static void pyprint(pybind11::handle h) {
  pybind11::print(h);
}

__attribute__((used)) static std::string pyrepr(py::handle h) {
  auto locals = py::dict("_v"_a = h);
  return py::cast<std::string>(py::eval("repr(_v)", py::globals(), locals));
}

__attribute__((used)) static std::string pyrepr(py::object o) {
  auto locals = py::dict("_v"_a = o);
  return py::cast<std::string>(py::eval("repr(_v)", py::globals(), locals));
}

__attribute__((used)) static void pycall1(const char *expr,
                                          pybind11::object o = py::none()) {
  // this doesn't work in lldb
  // py::scoped_interpreter guard{};

  /*
   * NOTE: the catch statements below do not work in lldb, because exceptions
   *       are trapped internally. So, an error in eval currently breaks
   *       use of this function until the process is restarted.
   */

  // usage: given some py::object 'o', exec a string w/ 'local _v'==o, e.g.:
  //        (lldb) p pycall1("_v.shape", o)

  py::object res = py::none();
  try {
    if (!o.is(py::none())) {
      auto locals = py::dict("_v"_a = o);
      res = py::eval(expr, py::globals(), locals);
    } else {
      res = py::eval(expr, py::globals());
    }
    if (!res.is(py::none())) {
      py::print(res);
    }
  } catch (py::error_already_set &e) {
    std::cout << "pycall error_already_set: " << std::endl;
  } catch (std::runtime_error &e) {
    std::cout << "pycall runtime_error: " << e.what() << std::endl;
  } catch (...) {
    std::cout << "pycall unknown exception" << std::endl;
  }
}

__attribute__((used)) static void pycall(const char *expr) {
  pycall1(expr, py::none());
}

__attribute__((used)) static void pyerror() {
  // print the last py error, if any
}
}
}; // namespace

#endif
