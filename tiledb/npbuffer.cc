#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>
#include <cstring>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#undef NPY_NO_DEPRECATED_API

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/pytypes.h>
#include "util.h"

#define TILEDB_DEPRECATED
#define TILEDB_DEPRECATED_EXPORT

#include <tiledb/tiledb> // C++

namespace {

namespace py = pybind11;

bool issubdtype(py::dtype t1, py::dtype t2) {
  // TODO importing every time is Not Great...
  auto np = py::module::import("numpy");
  auto npsubdtype = np.attr("issubdtype");

  return py::cast<bool>(npsubdtype(t1, t2));
}

template <typename T>
py::dtype get_dtype(T o) {
  static_assert(std::is_base_of<py::handle, T>::value,
    "get_dtype requires a PyObject* convertible argument"
  );

  auto& api = py::detail::npy_api::get();

  if (api.PyArray_Check_(o.ptr())) {
    return py::cast<py::array>(o).dtype();
  }

  return py::reinterpret_steal<py::dtype>(
    api.PyArray_DescrFromScalar_(o.ptr()));
}

// check whether dtypes are equivalent from numpy perspective
// note: d1::dtype.is(d2) checks object identity
bool dtype_equal(py::dtype d1, py::dtype d2) {
  auto& api = py::detail::npy_api::get();

  return api.PyArray_EquivTypes_(
      d1.ptr(), d2.ptr());
}

};


namespace tiledbpy {

using namespace std;
using namespace tiledb;
namespace py = pybind11;
using namespace pybind11::literals;

/*

list of types to test

- 1d O arrays of u and b
- Nd O arrays of u and b
- 1d O arrays of scalars
- Nd O arrays of scalars

*/

class NumpyConvert {
  private:
    bool use_iter_ = false;
    bool allow_unicode_ = true;
    size_t data_nbytes_ = 0;
    size_t input_len_ = 0;

    py::array input_;
    // we are using vector as a buffer here because they are grown in some situations
    std::vector<uint8_t>* data_buf_;
    std::vector<uint64_t>* offset_buf_;

  void convert_u() {
    // NOTE: NumPy fixed-length string arrays *do not support* embedded nulls.
    //       There is no string size stored, so string end is demarcated by \0 and the
    //       slot is filled to the next boundary with \0.
    // For consistency and to avoid complications in other APIs, we are storing
    // all string arrays as var-length.

    size_t input_itemsize = input_.itemsize();
    assert(input_itemsize > 0); // must have fixed-length array

    // we know exact offset count
    offset_buf_->resize(input_len_);

    // we reserve the input length as a minimum size for output
    data_buf_->resize(input_len_);

    // size (bytes) of current object data
    Py_ssize_t sz = 0;
    // object data (string or bytes)
    const char* input_p = nullptr;

    unsigned char* output_p = nullptr;
    output_p = data_buf_->data();

    // avoid one interpreter roundtrip
    auto npstrencode = py::module::import("numpy").attr("str").attr("encode");

    // TODO: ideally we would encode directly here without the intermediate unicode object
    // TODO add test for different memory orderings

    // loop over array objects and write to output buffer
    size_t idx = 0;
    for (auto u : input_) {
      py::handle u_encoded = u;

      // don't encode if we already have bytes
      if (PyUnicode_Check(u.ptr())) {
        // TODO see if we can do this with PyUnicode_AsUTF8String
        u_encoded = npstrencode(u);
      }

      PyBytes_AsStringAndSize(u_encoded.ptr(), const_cast<char**>(&input_p), &sz);

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

  void convert_b() {
    // NOTE: NumPy fixed-length string arrays *do not support* embedded nulls.
    //       There is no string size stored, so string end is demarcated by \0 and the
    //       slot is filled to the next boundary with \0.
    // For consistency and to avoid complications in other APIs, we are storing
    // all string arrays as var-length.

    size_t input_itemsize = input_.itemsize();
    assert(input_itemsize > 0); // must have fixed-length array

    // we know exact offset count
    offset_buf_->resize(input_len_);

    // we reserve the input length as a minimum size for output
    data_buf_->resize(input_len_);

    // size (bytes) of current object data
    Py_ssize_t sz = 0;
    // object data (string or bytes)
    const char* input_p = nullptr;

    unsigned char* output_p = nullptr;
    output_p = data_buf_->data();

    // avoid one interpreter roundtrip
    //auto npstrencode = py::module::import("numpy").attr("str").attr("encode");

    // TODO: ideally we would encode directly here without the intermediate unicode object
    // TODO add test for different memory orderings

    // loop over array objects and write to output buffer
    size_t idx = 0;
    for (auto u : input_) {
      auto o = u.ptr();

      // don't encode if we already have bytes
      /*
      if (PyUnicode_Check(u.ptr())) {
        // TODO see if we can do this with PyUnicode_AsUTF8String
        u_encoded = npstrencode(u);
      }
      */

      PyBytes_AsStringAndSize(o, const_cast<char**>(&input_p), &sz);

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

  void convert_o() {
    auto& api = py::detail::npy_api::get();

    offset_buf_->resize(input_len_);

    // pointer to the underlying PyObject* data
    // impl 1
    //auto data_p = ((PyObject**)PyArray_DATA((PyArrayObject*)input_.ptr()));

    // impl 2
    auto input_unck = input_.unchecked<py::object, 1>();

    // size (bytes) of current object data
    Py_ssize_t sz = 0;
    // current data
    const char* input_p = nullptr;

    auto input_size = input_.size();
    py::dtype first_dtype = py::none();

    // first pass: calculate final buffer length and cache UTF-8 representations
    for (int64_t idx = 0; idx < input_size; idx++) {
      offset_buf_->data()[idx] = data_nbytes_;

      // impl 1:
      //PyObject* o = data_p[idx];

      // impl 2:
      PyObject* o = input_unck.data(idx)->ptr();
      assert(o != nullptr);

      if (PyUnicode_Check(o)) {
        if (!allow_unicode_) {
          // TODO TPY_ERROR_LOC
          auto errmsg = std::string("Unexpected unicode object for TILEDB_STRING_ASCII attribute");
          throw std::runtime_error(errmsg);
        }

        // this will cache a utf-8 representation owned by the PyObject
        input_p = PyUnicode_AsUTF8AndSize(o, &sz);
        if (!input_p) {
          TPY_ERROR_LOC("Internal error: failed to convert unicode to UTF-8");
        }
      } else if (PyBytes_Check(o)) {
        // ASCII only
        auto res = PyBytes_AsStringAndSize(o, const_cast<char**>(&input_p), &sz);

        if (res == -1) {
          // TODO TPY_ERROR_LOC
          throw std::runtime_error("Internal error: failed to get char* from bytes object");
        }
      } else if (api.PyArray_Check_(o)) {
        auto a = py::cast<py::array>(o);
        // handle (potentially) var-len embedded arrays
        if (idx < 1) {
          first_dtype = get_dtype(a);
        } else if (!dtype_equal(get_dtype(a), first_dtype)) {
          throw py::type_error("Mismatched dtype in object array to buffer conversion!");
        }

        sz = a.nbytes();
      } else {
        // TODO write the type in the error here
        //auto o_h = py::reinterpret_borrow<py::object>(o);
        //auto o_t = py::type::of(o);
        auto errmsg = std::string("Unexpected object type in string conversion");
        TPY_ERROR_LOC(errmsg);
      }

      data_nbytes_ += sz;
    }

    data_buf_->resize(data_nbytes_);

    // second pass: write the data to output buffer
    unsigned char* output_p = data_buf_->data();

    // copy data to output buffers
    for (int64_t idx = 0; idx < input_size; idx++) {
      // impl 1:
      //PyObject* o = data_p[idx];

      // impl 2:
      auto _o = input_unck.data(idx);
      PyObject* o = _o->ptr();

      assert(o != nullptr);

      if (PyUnicode_Check(o)) {
        input_p = PyUnicode_AsUTF8AndSize(o, &sz);
        assert(input_p != nullptr);
      } else if (PyBytes_Check(o)) {
        // TODO error check?
        PyBytes_AsStringAndSize(o, const_cast<char**>(&input_p), &sz);
      } else if (api.PyArray_Check_(o)) {
        auto pao = (PyArrayObject*)o;
        //input_p = (const char*)PyArray_DATA(pao);
        //sz = PyArray_NBYTES(pao);
        auto a = py::cast<py::array>(o);
        sz = a.nbytes();
        input_p = (const char*)a.data();
      } else {
        TPY_ERROR_LOC("Unexpected object type in buffer conversion");
      }

      memcpy(output_p, input_p, sz);
      // increment the output pointer for the next object
      output_p += sz;
    }
  }

  void convert_iter() {
    auto& npy_api = py::detail::npy_api::get();

    offset_buf_->resize(input_.size());

    auto iter = input_.attr("flat");

    // size (bytes) of current object data
    Py_ssize_t sz = 0;
    // current data
    const char* input_p = nullptr;

    size_t idx = 0;

    py::dtype first_dtype = py::none();

    for (auto o_h : iter) {
      if (idx < 1) {
        first_dtype = get_dtype(o_h);
      }
      offset_buf_->data()[idx] = data_nbytes_;

      auto o = o_h.ptr();

      // we must check each dtype because object arrays need not be homogenous
      auto cur_dtype = get_dtype(o_h);
      auto err_str = std::string("Mismatched element type in buffer conversion!");
      if ((first_dtype.kind() == cur_dtype.kind()) ||
          (first_dtype.kind() == cur_dtype.kind())) {
        // pass
      } else if (!dtype_equal(cur_dtype, first_dtype)) {
        throw py::type_error(err_str);
      }

      if (PyUnicode_Check(o)) {
        if (!allow_unicode_) {
          // TODO TPY_ERROR_LOC
          auto errmsg = std::string("Unexpected unicode object for TILEDB_STRING_ASCII attribute");
          throw std::runtime_error(errmsg);
        }

        // this will cache a utf-8 representation owned by the PyObject
        input_p = PyUnicode_AsUTF8AndSize(o, &sz);
        if (!input_p) {
          TPY_ERROR_LOC("Internal error: failed to convert unicode to UTF-8");
        }
      } else if (PyBytes_Check(o)) {
        // ASCII only
        auto res = PyBytes_AsStringAndSize(o, const_cast<char**>(&input_p), &sz);

        if (res == -1) {
          // TODO TPY_ERROR_LOC
          throw std::runtime_error("Internal error: failed to get char* from bytes object");
        }
      } else if (npy_api.PyArray_Check_(o)) {
        // handle (potentially) var-len embedded arrays
        //sz = PyArray_NBYTES((PyArrayObject*)o);
        sz = py::cast<py::array>(o_h).nbytes();
      } else {
        auto errmsg = std::string("Unexpected object type in string conversion");
        TPY_ERROR_LOC(errmsg);
      }
      data_nbytes_ += sz;
      idx++;
    }

    data_buf_->resize(data_nbytes_);
    // second pass: write the data to output buffer
    unsigned char* output_p = data_buf_->data();

    // reset the iterator
    iter = input_.attr("flat");

    // copy data to output buffers
    for (auto o_h : iter) {
        auto o = o_h.ptr();

      if (PyUnicode_Check(o)) {
        input_p = PyUnicode_AsUTF8AndSize(o, &sz);
        assert(input_p != nullptr);
      } else if (PyBytes_Check(o)) {
        // TODO error check?
        PyBytes_AsStringAndSize(o, const_cast<char**>(&input_p), &sz);
      } else if (npy_api.PyArray_Check_(o)) {
        //auto pao = (PyArrayObject*)o;
        //input_p = (const char*)PyArray_DATA(pao);
        //sz = PyArray_NBYTES(pao);
        auto o_a = py::cast<py::array>(o_h);
        sz = o_a.nbytes();
        input_p = (const char*)o_a.data();
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
  bool allow_unicode() {
    return allow_unicode_;
  }
  void allow_unicode(bool allow_unicode) {
    allow_unicode_ = allow_unicode;
  }

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
        convert_u();
      } else {
        throw std::runtime_error("Unexpected fixed-length unicode array");
      }
    } else if (issubdtype(input_dtype, py::dtype("bytes"))) {
        convert_b();
    } else if (!input_dtype.is(py::dtype("O"))) {
      // TODO TPY_ERROR_LOC
      throw std::runtime_error("expected object array");
    } else {
      convert_o();
    }

    auto tmp_data_buf_p = data_buf_;
    auto data_ref = py::capsule(data_buf_,
                             [](void *v) { delete reinterpret_cast<std::vector<uint8_t>*>(v); });
    data_buf_ = nullptr; // disown: capsule owns it

    auto tmp_offset_buf_p = offset_buf_;
    auto offset_ref = py::capsule(offset_buf_,
                             [](void *v) { delete reinterpret_cast<std::vector<uint64_t>*>(v); });
    offset_buf_ = nullptr; // disown: capsule owns it now

    auto data_np = py::array_t<uint8_t>(tmp_data_buf_p->size(), tmp_data_buf_p->data(), data_ref);
    auto offset_np = py::array_t<uint64_t>(tmp_offset_buf_p->size(), tmp_offset_buf_p->data(), offset_ref);

    return py::make_tuple(data_np, offset_np);
  }

};

py::tuple convert_np(py::array input, bool allow_unicode) {
  NumpyConvert cvt(input);
  cvt.allow_unicode(allow_unicode);
  auto res = cvt.get();
  //py::print(res);
  return res; //cvt.get();
}

}; // namespace tiledbpy