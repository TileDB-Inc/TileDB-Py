#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "numpy/arrayobject.h"
#undef NPY_NO_DEPRECATED_API

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <tiledb/tiledb> // C++

#if !defined(NDEBUG)
#include "debug.cc"
#endif

namespace tiledbpy {

using namespace std;
using namespace tiledb;
namespace py = pybind11;
using namespace pybind11::literals;

#define TPY_ERROR_LOC(m)                                                       \
  throw TileDBPyError(std::string(m) + " (" + __FILE__ + ":" +                 \
                      std::to_string(__LINE__) + ")");

const uint64_t DEFAULT_INIT_BUFFER_BYTES = 1310720 * 8;
const uint64_t DEFAULT_EXP_ALLOC_MAX_BYTES = 4 * pow(2, 30);

class TileDBPyError : std::runtime_error {
public:
  explicit TileDBPyError(const char *m) : std::runtime_error(m) {}
  explicit TileDBPyError(std::string m) : std::runtime_error(m.c_str()) {}

public:
  virtual const char *what() const noexcept override {
    return std::runtime_error::what();
  }
};

py::dtype tiledb_dtype(tiledb_datatype_t type, uint32_t cell_val_num);

struct BufferInfo {

  BufferInfo(std::string name, size_t data_nbytes, tiledb_datatype_t data_type,
             uint32_t cell_val_num, size_t offsets_num, bool isvar = false)

      : name(name), type(data_type), cell_val_num(cell_val_num), isvar(isvar) {

    dtype = tiledb_dtype(data_type, cell_val_num);
    elem_nbytes = tiledb_datatype_size(type);
    data = py::array(py::dtype("uint8"), data_nbytes);
    offsets = py::array_t<uint64_t>(offsets_num);
  }

  string name;
  tiledb_datatype_t type;
  py::dtype dtype;
  size_t elem_nbytes = 1;
  uint64_t data_vals_read = 0;
  uint32_t cell_val_num;
  uint64_t offsets_read = 0;
  bool isvar;

  py::array data;
  py::array_t<uint64_t> offsets;
};

py::dtype tiledb_dtype(tiledb_datatype_t type, uint32_t cell_val_num) {
  if (cell_val_num == 1) {
    auto np = py::module::import("numpy");
    auto datetime64 = np.attr("datetime64");
    switch (type) {
    case TILEDB_INT32:
      return py::dtype("int32");
    case TILEDB_INT64:
      return py::dtype("int64");
    case TILEDB_FLOAT32:
      return py::dtype("float32");
    case TILEDB_FLOAT64:
      return py::dtype("float64");
    case TILEDB_INT8:
      return py::dtype("int8");
    case TILEDB_UINT8:
      return py::dtype("uint8");
    case TILEDB_INT16:
      return py::dtype("int16");
    case TILEDB_UINT16:
      return py::dtype("uint16");
    case TILEDB_UINT32:
      return py::dtype("uint32");
    case TILEDB_UINT64:
      return py::dtype("uint64");
    case TILEDB_STRING_ASCII:
      return py::dtype("S1");
    case TILEDB_STRING_UTF8:
      return py::dtype("U1");
    case TILEDB_STRING_UTF16:
    case TILEDB_STRING_UTF32:
      TPY_ERROR_LOC("Unimplemented UTF16 or UTF32 string conversion!");
    case TILEDB_STRING_UCS2:
    case TILEDB_STRING_UCS4:
      TPY_ERROR_LOC("Unimplemented UCS2 or UCS4 string conversion!");
    case TILEDB_CHAR:
      return py::dtype("S1");
    case TILEDB_DATETIME_YEAR:
      return datetime64("", "Y");
    case TILEDB_DATETIME_MONTH:
      return datetime64("", "M");
    case TILEDB_DATETIME_WEEK:
      return datetime64("", "W");
    case TILEDB_DATETIME_DAY:
      return datetime64("", "D");
    case TILEDB_DATETIME_HR:
      return datetime64("", "h");
    case TILEDB_DATETIME_MIN:
      return datetime64("", "m");
    case TILEDB_DATETIME_SEC:
      return datetime64("", "s");
    case TILEDB_DATETIME_MS:
      return datetime64("", "ms");
    case TILEDB_DATETIME_US:
      return datetime64("", "us");
    case TILEDB_DATETIME_NS:
      return datetime64("", "ns");
    case TILEDB_DATETIME_PS:
      return datetime64("", "ps");
    case TILEDB_DATETIME_FS:
      return datetime64("", "fs");
    case TILEDB_DATETIME_AS:
      return datetime64("", "as");
    case TILEDB_ANY:
      break;
    }
  } else if (cell_val_num == 2 && type == TILEDB_FLOAT32) {
    return py::dtype("complex64");
  } else if (cell_val_num == 2 && type == TILEDB_FLOAT64) {
    return py::dtype("complex128");
  } else if (type == TILEDB_CHAR || type == TILEDB_STRING_UTF8 ||
             type == TILEDB_STRING_ASCII) {
    std::string base_str;
    switch (type) {
    case TILEDB_CHAR:
    case TILEDB_STRING_ASCII:
      base_str = "|S";
      break;
    case TILEDB_STRING_UTF8:
      base_str = "|U";
      break;
    default:
      TPY_ERROR_LOC("internal error: unhandled string type");
    }
    if (cell_val_num < TILEDB_VAR_NUM) {
      base_str = base_str + std::to_string(cell_val_num);
    }
    return py::dtype(base_str);
  } else if (cell_val_num == TILEDB_VAR_NUM) {
    return tiledb_dtype(type, 1);
  } else if (cell_val_num > 1) {
    py::dtype base_dtype = tiledb_dtype(type, 1);
    py::tuple rec_elem = py::make_tuple("", base_dtype);
    py::list rec_list;
    for (size_t i = 0; i < cell_val_num; i++)
      rec_list.append(rec_elem);
    auto np = py::module::import("numpy");
    // note: we call the 'dtype' constructor b/c py::dtype does not accept list
    auto np_dtype = np.attr("dtype");
    return np_dtype(rec_list);
  }

  TPY_ERROR_LOC("tiledb datatype not understood ('" +
                tiledb::impl::type_to_str(type) +
                "', cell_val_num: " + std::to_string(cell_val_num) + ")");
}

class PyQuery {

private:
  Context ctx_;
  shared_ptr<tiledb::Array> array_;
  shared_ptr<tiledb::Query> query_;
  std::vector<std::string> attrs_;
  map<string, BufferInfo> buffers_;
  bool include_coords_;
  uint64_t init_buffer_bytes_ = DEFAULT_INIT_BUFFER_BYTES;
  uint64_t exp_alloc_max_bytes_ = DEFAULT_EXP_ALLOC_MAX_BYTES;

public:
  tiledb_ctx_t *c_ctx_;
  tiledb_array_t *c_array_;

public:
  PyQuery() = delete;

  PyQuery(py::object ctx, py::object array, py::iterable attrs,
          py::object coords) {

    tiledb_ctx_t *c_ctx_ = (py::capsule)ctx.attr("__capsule__")();
    if (c_ctx_ == nullptr)
      TPY_ERROR_LOC("Invalid context pointer!")
    ctx_ = Context(c_ctx_, false);

    tiledb_array_t *c_array_ = (py::capsule)array.attr("__capsule__")();

    // we never own this pointer, pass own=false
    array_ = std::shared_ptr<tiledb::Array>(new Array(ctx_, c_array_, false),
                                            [](Array *p) {} /* no deleter*/);

    query_ = std::shared_ptr<tiledb::Query>(
        new Query(ctx_, *array_, TILEDB_READ)); //,
    //                     [](Query* p){} /* no deleter*/);

    if (coords.is(py::none()))
      include_coords_ = true;
    else
      include_coords_ = coords.cast<bool>();

    for (auto a : attrs) {
      attrs_.push_back(a.cast<string>());
    }

    // get config parameters
    try {
      init_buffer_bytes_ =
          std::atoi(ctx_.config().get("py.init_buffer_bytes").c_str());
    } catch (TileDBError &e) { /* pass, key not found */
    }
    try {
      exp_alloc_max_bytes_ =
          std::atoi(ctx_.config().get("py.exp_alloc_max_bytes").c_str());
    } catch (TileDBError &e) {
      /* pass, key not found */
    }
  }

  void add_dim_range(uint32_t dim_idx, py::tuple r) {
    if (py::len(r) == 0)
      return;
    else if (py::len(r) != 2)
      TPY_ERROR_LOC("Unexpected range len != 2");

    auto r0 = r[0];
    auto r1 = r[1];
    // no type-check here, because we might allow cast-conversion
    // if (r0.get_type() != r1.get_type())
    //    TPY_ERROR_LOC("Mismatched type");

    auto domain = array_->schema().domain();
    auto dim = domain.dimension(dim_idx);

    auto tiledb_type = dim.type();

    try {
      switch (tiledb_type) {
      case TILEDB_INT32: {
        using T = int32_t;
        query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
        break;
      }
      case TILEDB_INT64: {
        using T = int64_t;
        query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
        break;
      }
      case TILEDB_INT8: {
        using T = uint8_t;
        query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
        break;
      }
      case TILEDB_UINT8: {
        using T = uint8_t;
        query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
        break;
      }
      case TILEDB_INT16: {
        using T = int16_t;
        query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
        break;
      }
      case TILEDB_UINT16: {
        using T = uint16_t;
        query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
        break;
      }
      case TILEDB_UINT32: {
        using T = uint32_t;
        query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
        break;
      }
      case TILEDB_UINT64: {
        using T = uint64_t;
        query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
        break;
      }
      case TILEDB_FLOAT32: {
        using T = float;
        query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
        break;
      }
      case TILEDB_FLOAT64: {
        using T = double;
        query_->add_range(dim_idx, r0.cast<T>(), r1.cast<T>());
        break;
      }
      case TILEDB_STRING_ASCII:
      case TILEDB_STRING_UTF8:
      case TILEDB_CHAR: {
        if (!py::isinstance<py::str>(r0))
          TPY_ERROR_LOC(
              "internal error: expected string type for var-length dim!");
        query_->add_range(dim_idx, r0.cast<string>(), r1.cast<string>());
        break;
      }
      case TILEDB_DATETIME_YEAR:
      case TILEDB_DATETIME_MONTH:
      case TILEDB_DATETIME_WEEK:
      case TILEDB_DATETIME_DAY:
      case TILEDB_DATETIME_HR:
      case TILEDB_DATETIME_MIN:
      case TILEDB_DATETIME_SEC:
      case TILEDB_DATETIME_MS:
      case TILEDB_DATETIME_US:
      case TILEDB_DATETIME_NS:
      case TILEDB_DATETIME_PS:
      case TILEDB_DATETIME_FS:
      case TILEDB_DATETIME_AS: {
        py::dtype dtype = tiledb_dtype(tiledb_type, 1);
        auto dt0 = r0.attr("astype")(dtype);
        auto dt1 = r1.attr("astype")(dtype);
        // TODO, this is suboptimal, should define pybind converter
        auto darray = py::array(py::make_tuple(dt0, dt1));
        query_->add_range(dim_idx, *(int64_t *)darray.data(0),
                          *(int64_t *)darray.data(1));
        break;
      }
      default:
        TPY_ERROR_LOC("Unknown dim type conversion!");
      }
    } catch (py::cast_error &e) {
      std::string msg = "Failed to cast dim range '" + (string)py::repr(r) +
                        "' to dim type " +
                        tiledb::impl::type_to_str(tiledb_type);
      TPY_ERROR_LOC(msg);
    }
  }

  void set_ranges(py::iterable ranges) {
    // ranges are specified as one iterable per dimension

    uint32_t dim_idx = 0;
    for (auto dim_range : ranges) {
      py::tuple dim_range_iter = dim_range.cast<py::iterable>();
      for (auto r : dim_range_iter) {
        py::tuple r_tuple = r.cast<py::tuple>();
        add_dim_range(dim_idx, r_tuple);
      }
      dim_idx++;
    }
  }

  void set_subarray(py::array subarray) {
    auto schema = array_->schema();
    auto ndim = schema.domain().ndim();
    if (subarray.size() != (2 * ndim))
      TPY_ERROR_LOC(
          "internal error: failed to set subarray (mismatched dimension count");

    py::object r0, r1;
    for (unsigned dim_idx = 0; dim_idx < ndim; dim_idx++) {
      auto r = subarray[py::int_(dim_idx)];
      r0 = r[py::int_(0)];
      r1 = r[py::int_(1)];

      add_dim_range(dim_idx, py::make_tuple(r0, r1));
    }
  }

  bool is_dimension(std::string name) {
    return array_->schema().domain().has_dimension(name);
  }

  bool is_attribute(std::string name) {
    return array_->schema().has_attribute(name);
  }

  bool is_var(std::string name) {
    if (is_dimension(name)) {
      auto domain = array_->schema().domain();
      auto dim = domain.dimension(name);
      return dim.cell_val_num() == TILEDB_VAR_NUM;
    } else if (is_attribute(name)) {
      auto attr = array_->schema().attribute(name);
      return attr.cell_val_num() == TILEDB_VAR_NUM;
    } else {
      TPY_ERROR_LOC("Unknown buffer type for is_var check (expected attribute "
                    "or dimension)")
    }
  }

  std::pair<tiledb_datatype_t, uint32_t> buffer_type(std::string name) {
    auto schema = array_->schema();
    tiledb_datatype_t type;
    uint32_t cell_val_num;
    if (is_dimension(name)) {
      type = schema.domain().dimension(name).type();
      cell_val_num = schema.domain().dimension(name).cell_val_num();
    } else if (is_attribute(name)) {
      type = schema.attribute(name).type();
      cell_val_num = schema.attribute(name).cell_val_num();
    } else {
      TPY_ERROR_LOC("Unknown buffer '" + name + "'");
    }
    return {type, cell_val_num};
  }

  uint32_t buffer_ncells(std::string name) {
    auto schema = array_->schema();
    if (is_dimension(name)) {
      return schema.domain().dimension(name).cell_val_num();
    } else if (is_attribute(name)) {
      return schema.attribute(name).cell_val_num();
    }
    TPY_ERROR_LOC("Unknown buffer '" + name + "' for buffer_ncells");
  }

  py::dtype buffer_dtype(std::string name) {
    try {
      auto t = buffer_type(name);
      return tiledb_dtype(t.first, t.second);
    } catch (TileDBError &e) {
      return py::none();
    }
  }

  void set_buffer(py::str name, py::object data) {
    // set input data for an attribute or dimension buffer
    if (is_attribute(name))
      set_buffer(name, data);
    else if (is_dimension(name))
      set_buffer(name, data);
    else
      TPY_ERROR_LOC("Unknown attribute or dim '" + (string)name + "'")
  }

  bool is_sparse() { return array_->schema().array_type() == TILEDB_SPARSE; }

  void alloc_buffer(std::string name) {
    auto schema = array_->schema();

    tiledb_datatype_t type;
    uint32_t cell_val_num;
    std::tie(type, cell_val_num) = buffer_type(name);
    uint64_t cell_nbytes = tiledb_datatype_size(type);
    if (cell_val_num != TILEDB_VAR_NUM)
      cell_nbytes *= cell_val_num;
    auto dtype = tiledb_dtype(type, cell_val_num);

    uint64_t buf_nbytes = 0;
    uint64_t offsets_num = 0;
    bool var = is_var(name);

    if (var) {
      auto size_pair = query_->est_result_size_var(name);
      buf_nbytes = size_pair.second;
      offsets_num = size_pair.first;
    } else {
      buf_nbytes = query_->est_result_size(name);
    }

    if ((var || is_sparse()) && (buf_nbytes < init_buffer_bytes_)) {
      buf_nbytes = init_buffer_bytes_;
      offsets_num = init_buffer_bytes_ / sizeof(uint64_t);
    }

    buffers_.insert({name, BufferInfo(name, buf_nbytes, type, cell_val_num,
                                      offsets_num, var)});
  }

  void set_buffers() {
    for (auto bp : buffers_) {
      auto name = bp.first;
      const BufferInfo b = bp.second;
      void *data_ptr =
          (void *)((char *)b.data.data() + (b.data_vals_read * b.elem_nbytes));
      uint64_t data_nelem =
          (b.data.size() - (b.data_vals_read * b.elem_nbytes)) / b.elem_nbytes;

      if (b.isvar) {
        uint64_t *offsets_ptr = (uint64_t *)b.offsets.data() + b.offsets_read;
        query_->set_buffer(b.name, offsets_ptr,
                           b.offsets.size() - b.offsets_read, data_ptr,
                           data_nelem);
      } else {
        query_->set_buffer(b.name, data_ptr, data_nelem);
      }
    }
  }

  void update_read_elem_num() {
    for (const auto &read_info : query_->result_buffer_elements()) {
      auto name = read_info.first;
      uint64_t offset_elem_num, data_vals_num;
      std::tie(offset_elem_num, data_vals_num) = read_info.second;
      BufferInfo &buf = buffers_.at(name);
      buf.data_vals_read += data_vals_num;
      buf.offsets_read += offset_elem_num;
    }
  }

  void submit_read() {
    auto schema = array_->schema();
    auto issparse = schema.array_type() == TILEDB_SPARSE;
    auto need_dim_buffers = include_coords_ || issparse;

    if (need_dim_buffers) {
      auto domain = schema.domain();
      for (auto dim : domain.dimensions()) {
        alloc_buffer(dim.name());
      }
    }

    for (auto attr_pair : schema.attributes()) {
      alloc_buffer(attr_pair.first);
    }

    set_buffers();

    size_t max_retries = 0, retries = 0;
    try {
      max_retries =
          std::atoi(ctx_.config().get("py.max_incomplete_retries").c_str());
    } catch (tiledb::TileDBError &e) {
      max_retries = 100;
    }

    // TODO ideally only have one call to submit below
    {
      py::gil_scoped_release release;
      query_->submit();
    }

    // TODO: would be nice to have a callback here for custom realloc strategy
    while (query_->query_status() == Query::Status::INCOMPLETE) {
      if (++retries > max_retries)
        TPY_ERROR_LOC(
            "Exceeded maximum retries ('py.max_incomplete_retries': " +
            std::to_string(max_retries) + "')");

      update_read_elem_num();

      for (auto bp : buffers_) {
        auto buf = bp.second;

        if ((int64_t)(buf.data_vals_read * buf.elem_nbytes) < buf.data.nbytes() * 2) {
          buf.data.resize({buf.data.size() * 2}, false);

          if (buf.isvar)
            buf.offsets.resize({buf.offsets.size() * 2}, false);
        }
      }

      set_buffers();
      {
        py::gil_scoped_release release;
        query_->submit();
      }
    }

    update_read_elem_num();

    // resize the output buffers to match the final read total
    for (auto bp : buffers_) {
      auto name = bp.first;
      auto &buf = bp.second;
      buf.data.resize({buf.data_vals_read * buf.elem_nbytes});
      buf.offsets.resize({buf.offsets_read});
    }
  }

  void submit_write() {}

  void submit() {
    if (array_->query_type() == TILEDB_READ)
      submit_read();
    else if (array_->query_type() == TILEDB_WRITE)
      submit_write();
    else
      TPY_ERROR_LOC("Unknown query type!")
  }

  py::dict results() {
    py::dict results;
    for (auto &bp : buffers_) {
      results[py::str(bp.first)] =
          py::make_tuple(bp.second.data, bp.second.offsets);
    }
    return results;
  }

  py::array test_array() {
    py::array_t<uint8_t> a;
    a.resize({10});

    a.resize({20});
    return std::move(a);
  }
};

PYBIND11_MODULE(core, m) {
  py::class_<PyQuery>(m, "PyQuery")
      .def(py::init<py::object, py::object, py::iterable, py::object>())
      .def("set_ranges", &PyQuery::set_ranges)
      .def("set_subarray", &PyQuery::set_subarray)
      .def("submit", &PyQuery::submit)
      .def("results", &PyQuery::results)
      .def("buffer_dtype", &PyQuery::buffer_dtype)
      .def("test_array", &PyQuery::test_array)
      .def("test_err",
           [](py::object self, std::string s) { throw TileDBPyError(s); });

  /*
     We need to make sure C++ TileDBError is translated to a correctly-typed py
     error. Note that using py::exception(..., "TileDBError") creates a new
     exception in the *readquery* module, so we must import to reference.
  */
  static auto tiledb_py_error =
      (py::object)py::module::import("tiledb").attr("TileDBError");

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p)
        std::rethrow_exception(p);
    } catch (const TileDBPyError &e) {
      PyErr_SetString(tiledb_py_error.ptr(), e.what());
    } catch (const tiledb::TileDBError &e) {
      PyErr_SetString(tiledb_py_error.ptr(), e.what());
    } catch (std::exception &e) {
    }
  });
}

}; // namespace tiledbpy
