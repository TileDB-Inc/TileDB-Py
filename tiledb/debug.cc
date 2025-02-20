#include <pybind11/embed.h>

#ifndef TILEDBPY_DEBUGCC
#define TILEDBPY_DEBUGCC

namespace {
extern "C" {

namespace nb = nanobind;
using namespace pybind11::literals;

// __attribute__((used)) to make the linker keep the symbol
__attribute__((used)) static void pyprint(pybind11::object o) {
    pybind11::print(o);
}

__attribute__((used)) static void pyprint(pybind11::handle h) {
    pybind11::print(h);
}

__attribute__((used)) static std::string pyrepr(nb::handle h) {
    auto locals = nb::dict("_v"_a = h);
    return nb::cast<std::string>(nb::eval("repr(_v)", nb::globals(), locals));
}

__attribute__((used)) static std::string pyrepr(nb::object o) {
    auto locals = nb::dict("_v"_a = o);
    return nb::cast<std::string>(nb::eval("repr(_v)", nb::globals(), locals));
}

__attribute__((used)) static void pycall1(
    const char* expr, pybind11::object o = nb::none()) {
    // this doesn't work in lldb
    // nb::scoped_interpreter guard{};

    /*
     * NOTE: the catch statements below do not work in lldb, because exceptions
     *       are trapped internally. So, an error in eval currently breaks
     *       use of this function until the process is restarted.
     */

    // usage: given some nb::object 'o', exec a string w/ 'local _v'==o, e.g.:
    //        (lldb) p pycall1("_v.shape", o)

    nb::object res = nb::none();
    try {
        if (!o.is(nb::none())) {
            auto locals = nb::dict("_v"_a = o);
            res = nb::eval(expr, nb::globals(), locals);
        } else {
            res = nb::eval(expr, nb::globals());
        }
        if (!res.is(nb::none())) {
            nb::print(res);
        }
    } catch (nb::python_error& e) {
        std::cout << "pycall python_error: " << std::endl;
    } catch (std::runtime_error& e) {
        std::cout << "pycall runtime_error: " << e.what() << std::endl;
    } catch (...) {
        std::cout << "pycall unknown exception" << std::endl;
    }
}

__attribute__((used)) static void pycall(const char* expr) {
    pycall1(expr, nb::none());
}

__attribute__((used)) static void pyerror() {
    // print the last py error, if any
}
}
};  // namespace

#endif
