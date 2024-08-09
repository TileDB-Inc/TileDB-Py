#include <chrono>
#include <cstring>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "util.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>

#if !defined(NDEBUG)
// #include "debug.cc"
#endif

#include <tiledb/tiledb> // C++

// anonymous namespace for helper functions
namespace {

namespace py = pybind11;

bool issubdtype(py::dtype t1, py::dtype t2) {
  // TODO importing every time is Not Great...
  auto np = py::module::import("numpy");
  auto npsubdtype = np.attr("issubdtype");

  return py::cast<bool>(npsubdtype(t1, t2));
}

template <typename T> py::dtype get_dtype(T obj) {
  auto &api = py::detail::npy_api::get();

  if (api.PyArray_Check_(obj.ptr())) {
    return py::cast<py::array>(obj).dtype();
  }

  return py::reinterpret_steal<py::dtype>(
      api.PyArray_DescrFromScalar_(obj.ptr()));
}

// check whether dtypes are equivalent from numpy perspective
// note: d1::dtype.is(d2) checks *object identity* which is
//       not what we want.
bool dtype_equal(py::dtype d1, py::dtype d2) {
  auto &api = py::detail::npy_api::get();

  return api.PyArray_EquivTypes_(d1.ptr(), d2.ptr());
}

}; // namespace

namespace tiledbpy {

using namespace std;
using namespace tiledb;
namespace py = pybind11;
using namespace pybind11::literals;

#if PY_MAJOR_VERSION >= 3
class NumpyConvert {
private:
  bool use_iter_ = false;
  bool allow_unicode_ = true;
  size_t data_nbytes_ = 0;
  size_t input_len_ = 0;

  py::array input_;
  // we are using vector as a buffer here because they are grown in some
  // situations
  std::vector<uint8_t> *data_buf_;
  std::vector<uint64_t> *offset_buf_;

  void convert_unicode() {
    // Convert array of strings to UTF-8 buffer+offsets

    // NOTE: NumPy fixed-length string arrays *do not support* embedded nulls.
    //       There is no string size stored, so string end is demarcated by \0
    //       and the slot is filled to the next boundary with \0.
    // For consistency and to avoid complications in other APIs, we are storing
    // all string arrays as var-length.

    // must have fixed-width element
    assert(input_.itemsize() > 0);

    // we know exact offset count
    offset_buf_->resize(input_len_);

    // we reserve the input length as a minimum size for output
    data_buf_->resize(input_len_);

    // size (bytes) of current object data
    Py_ssize_t sz = 0;
    // object data (string or bytes)
    const char *input_p = nullptr;

    unsigned char *output_p = nullptr;
    output_p = data_buf_->data();

    // avoid one interpreter roundtrip
    auto npstrencode = py::module::import("numpy").attr("str_").attr("encode");

    // return status
    int rc;
    // encoded object: this must live outside the if block or else it may be
    // GC'd
    //                 putting outside for loop to avoid repeat unused
    //                 construction
    py::object u_encoded;

    // loop over array objects and write to output buffer
    size_t idx = 0;
    for (auto u : input_) {
      // don't encode if we already have bytes
      if (PyUnicode_Check(u.ptr())) {
        // TODO see if we can do this with PyUnicode_AsUTF8String
        u_encoded = npstrencode(u);
        rc = PyBytes_AsStringAndSize(u_encoded.ptr(),
                                     const_cast<char **>(&input_p), &sz);
      } else {
        rc = PyBytes_AsStringAndSize(u.ptr(), const_cast<char **>(&input_p),
                                     &sz);
      }

      if (rc == -1) {
        throw std::runtime_error(
            "PyBytes_AsStringAndSize failed to encode string");
      }

      // record the offset (equal to the current bytes written)
      offset_buf_->data()[idx] = data_nbytes_;

      if (data_buf_->size() < data_nbytes_ + sz) {
        data_buf_->resize(data_nbytes_ + sz);
        // update the output pointer and adjust for previous iteration
        output_p = data_buf_->data() + data_nbytes_;
      }

      memcpy(output_p, input_p, sz);

      data_nbytes_ += sz;
      output_p += sz;
      idx++;
    }

    data_buf_->resize(data_nbytes_);
  }

  void convert_bytes() {
    // Convert array of bytes objects or ASCII strings to buffer+offsets

    assert(input_.itemsize() > 0); // must have fixed-length array

    // we know exact offset count
    offset_buf_->resize(input_len_);

    // we reserve the input length as a minimum size for output
    data_buf_->resize(input_len_);

    // size (bytes) of current object data
    Py_ssize_t sz = 0;
    // object data (string or bytes)
    const char *input_p = nullptr;

    unsigned char *output_p = nullptr;
    output_p = data_buf_->data();

    int rc;

    // avoid one interpreter roundtrip
    // auto npstrencode =
    // py::module::import("numpy").attr("str_").attr("encode");

    // TODO: ideally we would encode directly here without the intermediate
    // unicode object
    // TODO add test for different memory orderings

    // loop over array objects and write to output buffer
    size_t idx = 0;
    for (auto obj : input_) {
      auto o = obj.ptr();

      // don't encode if we already have bytes
      /*
if (PyUnicode_Check(u.ptr())) {
  // TODO see if we can do this with PyUnicode_AsUTF8String
  u_encoded = npstrencode(u);
}
*/

      rc = PyBytes_AsStringAndSize(o, const_cast<char **>(&input_p), &sz);
      if (rc == -1) {
        throw std::runtime_error(
            "PyBytes_AsStringAndSize failed to encode string");
      }

      // record the offset (equal to the current bytes written)
      offset_buf_->data()[idx] = data_nbytes_;

      if (data_buf_->size() < data_nbytes_ + sz) {
        data_buf_->resize(data_nbytes_ + sz);
        // update the output pointer and adjust for previous iteration
        output_p = data_buf_->data() + data_nbytes_;
      }

      memcpy(output_p, input_p, sz);

      data_nbytes_ += sz;
      output_p += sz;
      idx++;
    }
  }

  void convert_object() {
    // Convert np.dtype("O") array of objects to buffer+offsets

    auto &api = py::detail::npy_api::get();

    offset_buf_->resize(input_len_);

    auto input_unchecked = input_.unchecked<py::object, 1>();

    // size (bytes) of current object data
    Py_ssize_t sz = 0;
    // current data
    const char *input_p = nullptr;

    auto input_size = input_.size();
    py::dtype first_dtype;

    // first pass: calculate final buffer length and cache UTF-8 representations
    for (int64_t idx = 0; idx < input_size; idx++) {
      offset_buf_->data()[idx] = data_nbytes_;

      PyObject *o = input_unchecked.data(idx)->ptr();
      assert(o != nullptr);

      // NOTE: every branch below *must* initialize first_dtype

      if (PyUnicode_Check(o)) {
        if (!allow_unicode_) {
          // TODO TPY_ERROR_LOC
          auto errmsg = std::string(
              "Unexpected unicode object for TILEDB_STRING_ASCII attribute");
          throw std::runtime_error(errmsg);
        }

        if (idx < 1)
          first_dtype = py::dtype("unicode");

        // this will cache a utf-8 representation owned by the PyObject
        input_p = PyUnicode_AsUTF8AndSize(o, &sz);
        if (!input_p) {
          TPY_ERROR_LOC("Internal error: failed to convert unicode to UTF-8");
        }
      } else if (PyBytes_Check(o)) {
        // ASCII only
        auto res =
            PyBytes_AsStringAndSize(o, const_cast<char **>(&input_p), &sz);

        if (idx < 1)
          first_dtype = py::dtype("bytes");

        if (res == -1) {
          // TODO TPY_ERROR_LOC
          throw std::runtime_error(
              "Internal error: failed to get char* from bytes object");
        }
      } else if (api.PyArray_Check_(o)) {
        auto a = py::cast<py::array>(o);
        // handle (potentially) var-len embedded arrays
        if (idx < 1) {
          first_dtype = get_dtype(a);
        } else if (!dtype_equal(get_dtype(a), first_dtype)) {
          throw py::type_error(
              "Mismatched dtype in object array to buffer conversion!");
        }

        sz = a.nbytes();
      } else if (PyBool_Check(o)) {
        if (idx < 1)
          first_dtype = py::dtype("bool");

        auto a = py::cast<py::bool_>(o);
        sz = sizeof(bool);
        bool bool_value = a;
        input_p = reinterpret_cast<const char *>(&bool_value);
      } else {
        // TODO write the type in the error here
        // auto o_h = py::reinterpret_borrow<py::object>(o);
        // auto o_t = py::type::of(o);
        auto errmsg =
            std::string("Unexpected object type in string conversion");
        TPY_ERROR_LOC(errmsg);
      }

      data_nbytes_ += sz;
    }

    data_buf_->resize(data_nbytes_);

    // second pass: copy the data to output buffer
    unsigned char *output_p = data_buf_->data();

    // copy data to output buffers
    for (int64_t idx = 0; idx < input_size; idx++) {
      PyObject *pyobj_p = input_unchecked.data(idx)->ptr();

      assert(pyobj_p != nullptr);

      if (PyUnicode_Check(pyobj_p)) {
        input_p = PyUnicode_AsUTF8AndSize(pyobj_p, &sz);
        assert(input_p != nullptr);
      } else if (PyBytes_Check(pyobj_p)) {
        // TODO error check?
        PyBytes_AsStringAndSize(pyobj_p, const_cast<char **>(&input_p), &sz);
      } else if (api.PyArray_Check_(pyobj_p)) {
        auto arr = py::cast<py::array>(pyobj_p);
        sz = arr.nbytes();
        input_p = (const char *)arr.data();
      } else if (PyBool_Check(pyobj_p)) {
        py::bool_ bool_obj = py::cast<py::bool_>(pyobj_p);
        sz = sizeof(bool);
        bool bool_value = bool_obj;
        input_p = reinterpret_cast<const char *>(&bool_value);
      } else {
        // TODO add object type
        TPY_ERROR_LOC("Unexpected object type in buffer conversion");
      }

      memcpy(output_p, input_p, sz);
      // increment the output pointer for the next object
      output_p += sz;
    }
  }

  void convert_iter() {
    // Convert array of non-contiguous objects to buffer+offsets
    // using iterator protocol.
    // For non-contiguous arrays (such as views) we must iterate rather
    // than indexing directly.

    auto &npy_api = py::detail::npy_api::get();

    offset_buf_->resize(input_.size());

    auto iter = input_.attr("flat");

    // size (bytes) of current object data
    Py_ssize_t sz = 0;
    // current data
    const char *input_p = nullptr;

    size_t idx = 0;

    py::dtype first_dtype;

    for (auto obj_h : iter) {
      if (idx < 1) {
        // record the first dtype for consistency check
        first_dtype = get_dtype(obj_h);
      }
      offset_buf_->data()[idx] = data_nbytes_;

      PyObject *obj_p = obj_h.ptr();

      // we must check each dtype because object arrays are not guaranteed to
      // be homogenous
      auto cur_dtype = get_dtype(obj_h);
      auto err_str =
          std::string("Mismatched element type in buffer conversion!");
      if ((first_dtype.kind() == cur_dtype.kind()) ||
          (first_dtype.kind() == cur_dtype.kind())) {
        // pass
      } else if (!dtype_equal(cur_dtype, first_dtype)) {
        throw py::type_error(err_str);
      }

      if (PyUnicode_Check(obj_p)) {
        if (!allow_unicode_) {
          // TODO TPY_ERROR_LOC
          auto errmsg = std::string(
              "Unexpected unicode object for TILEDB_STRING_ASCII attribute");
          throw std::runtime_error(errmsg);
        }

        // this will cache a utf-8 representation owned by the PyObject
        input_p = PyUnicode_AsUTF8AndSize(obj_p, &sz);
        if (!input_p) {
          TPY_ERROR_LOC("Internal error: failed to convert unicode to UTF-8");
        }
      } else if (PyBytes_Check(obj_p)) {
        // ASCII only
        auto res =
            PyBytes_AsStringAndSize(obj_p, const_cast<char **>(&input_p), &sz);

        if (res == -1) {
          // TODO TPY_ERROR_LOC
          throw std::runtime_error(
              "Internal error: failed to get char* from bytes object");
        }
      } else if (npy_api.PyArray_Check_(obj_p)) {
        // handle (potentially) var-len embedded arrays
        sz = py::cast<py::array>(obj_p).nbytes();
      } else if (PyBool_Check(obj_p)) {
        if (idx < 1)
          first_dtype = py::dtype("bool");

        py::bool_ bool_obj = py::cast<py::bool_>(obj_p);
        sz = sizeof(bool);
        bool bool_value = bool_obj;
        input_p = reinterpret_cast<const char *>(&bool_value);
      } else {
        auto errmsg =
            std::string("Unexpected object type in string conversion");
        TPY_ERROR_LOC(errmsg);
      }
      data_nbytes_ += sz;
      idx++;
    }

    data_buf_->resize(data_nbytes_);
    // second pass: write the data to output buffer
    unsigned char *output_p = data_buf_->data();

    // reset the iterator
    iter = input_.attr("flat");

    // copy data to output buffers
    for (auto obj_h : iter) {
      auto obj_p = obj_h.ptr();

      if (PyUnicode_Check(obj_p)) {
        input_p = PyUnicode_AsUTF8AndSize(obj_p, &sz);
        assert(input_p != nullptr);
      } else if (PyBytes_Check(obj_p)) {
        // TODO error check?
        PyBytes_AsStringAndSize(obj_p, const_cast<char **>(&input_p), &sz);
      } else if (npy_api.PyArray_Check_(obj_p)) {
        // auto pao = (PyArrayObject*)o;
        // input_p = (const char*)PyArray_DATA(pao);
        // sz = PyArray_NBYTES(pao);
        auto o_a = py::cast<py::array>(obj_h);
        sz = o_a.nbytes();
        input_p = (const char *)o_a.data();
      } else if (PyBool_Check(obj_p)) {
        py::bool_ bool_obj = py::cast<py::bool_>(obj_p);
        sz = sizeof(bool);
        bool bool_value = bool_obj;
        input_p = reinterpret_cast<const char *>(&bool_value);
      } else {
        TPY_ERROR_LOC("Unexpected object type in buffer conversion");
      }

      memcpy(output_p, input_p, sz);
      // increment the output pointer for the next object
      output_p += sz;
    }
  }

public:
  /*
  Initialize the converter
*/
  NumpyConvert(py::array input) {
    // require a flat buffer
    if (input.ndim() != 1) {
      // try to take a 1D view on the input
      auto v = input.attr("view")();
      // this will throw if the shape cannot be modified zero-copy,
      // which is what we want
      try {
        v.attr("shape") = py::int_(input.size());
      } catch (py::error_already_set &e) {
        if (e.matches(PyExc_AttributeError)) {
          use_iter_ = true;
        } else {
          throw;
        }
      } catch (std::exception &e) {
        std::cout << e.what() << std::endl;
      }
      input_ = v;
    } else {
      input_ = input;
    }

    input_len_ = py::len(input_);

    data_buf_ = new std::vector<uint8_t>();
    offset_buf_ = new std::vector<uint64_t>(input_len_);
  }

  ~NumpyConvert() {
    if (data_buf_)
      delete data_buf_;
    if (offset_buf_)
      delete offset_buf_;
  }

  /*
  Set allow_unicode_ flag
*/
  bool allow_unicode() { return allow_unicode_; }
  void allow_unicode(bool allow_unicode) { allow_unicode_ = allow_unicode; }

  /*
  Returns a tuple of py::array containing
    (data:array_t<uint8>, offsets:array_t<uint64_t>)
*/
  py::tuple get() {
    auto input_dtype = input_.dtype();

    if (use_iter_) {
      // slow, safe path
      convert_iter();
    } else if (issubdtype(input_dtype, py::dtype("unicode"))) {
      if (allow_unicode_) {
        convert_unicode();
      } else {
        throw std::runtime_error("Unexpected fixed-length unicode array");
      }
    } else if (issubdtype(input_dtype, py::dtype("bytes"))) {
      convert_bytes();
    } else if (!input_dtype.equal(py::dtype("O"))) {
      // TODO TPY_ERROR_LOC
      throw std::runtime_error("expected object array");
    } else {
      convert_object();
    }

    auto tmp_data_buf_p = data_buf_;
    auto data_ref = py::capsule(data_buf_, [](void *v) {
      delete reinterpret_cast<std::vector<uint8_t> *>(v);
    });
    data_buf_ = nullptr; // disown: capsule owns it

    auto tmp_offset_buf_p = offset_buf_;
    auto offset_ref = py::capsule(offset_buf_, [](void *v) {
      delete reinterpret_cast<std::vector<uint64_t> *>(v);
    });
    offset_buf_ = nullptr; // disown: capsule owns it now

    auto data_np = py::array_t<uint8_t>(tmp_data_buf_p->size(),
                                        tmp_data_buf_p->data(), data_ref);
    auto offset_np = py::array_t<uint64_t>(
        tmp_offset_buf_p->size(), tmp_offset_buf_p->data(), offset_ref);

    return py::make_tuple(data_np, offset_np);
  }
};
#endif

py::tuple convert_np(py::array input, bool allow_unicode,
                     bool use_fallback = false) {
#if PY_MAJOR_VERSION >= 3
  if (use_fallback) {
#endif
    auto tiledb = py::module::import("tiledb");
    auto libtiledb = tiledb.attr("libtiledb");
    auto array_to_buffer = libtiledb.attr("array_to_buffer");
    return array_to_buffer(input);
#if PY_MAJOR_VERSION >= 3
  } else {
    NumpyConvert cvt(input);
    cvt.allow_unicode(allow_unicode);
    return cvt.get();
  }
#endif
}

}; // namespace tiledbpy
