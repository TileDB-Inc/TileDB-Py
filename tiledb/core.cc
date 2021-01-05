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
#include "npbuffer.h"
#include "util.h"

#define TILEDB_DEPRECATED
#define TILEDB_DEPRECATED_EXPORT

#include <tiledb/tiledb> // C++
#include <tiledb/arrowio>

#include "../external/string_view.hpp"
#include "../external/tsl/robin_map.h"

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
const uint64_t DEFAULT_EXP_ALLOC_MAX_BYTES = uint64_t(4 * pow(2, 30));

class TileDBPyError : std::runtime_error {
public:
  explicit TileDBPyError(const char *m) : std::runtime_error(m) {}
  explicit TileDBPyError(std::string m) : std::runtime_error(m.c_str()) {}

public:
  virtual const char *what() const noexcept override {
    return std::runtime_error::what();
  }
};

using TimerType = std::chrono::duration<double>;
struct StatsInfo {
  std::map<std::string, TimerType> counters;
};

bool config_has_key(tiledb::Config config, std::string key) {
  try {
    config.get(key);
  } catch (TileDBError &e) {
    (void)e;
    return false;
  }
  return true;
}

struct PAPair {
  int64_t get_array() {
    if (!exported_) {
      TPY_ERROR_LOC("Cannot export uninitialized array!");
    }
    return (int64_t)&array_;
  };
  int64_t get_schema() {
    if (!exported_) {
      TPY_ERROR_LOC("Cannot export uninitialized schema!");
    }
    return (int64_t)&schema_;
  }

  ArrowSchema schema_;
  ArrowArray array_;
  bool exported_ = false;
};

// global stats counters
static std::unique_ptr<StatsInfo> g_stats;

// forward declaration
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
  std::vector<std::string> dims_;
  map<string, BufferInfo> buffers_;
  vector<string> buffers_order_;
  bool deduplicate_ = true;
  bool use_arrow_ = false;
  uint64_t init_buffer_bytes_ = DEFAULT_INIT_BUFFER_BYTES;
  uint64_t exp_alloc_max_bytes_ = DEFAULT_EXP_ALLOC_MAX_BYTES;

public:
  tiledb_ctx_t *c_ctx_;
  tiledb_array_t *c_array_;

public:
  PyQuery() = delete;

  PyQuery(py::object ctx,
          py::object array,
          py::iterable attrs,
          py::iterable dims,
          py::object py_layout,
          py::object use_arrow) {

    tiledb_ctx_t *c_ctx_ = (py::capsule)ctx.attr("__capsule__")();
    if (c_ctx_ == nullptr)
      TPY_ERROR_LOC("Invalid context pointer!")
    ctx_ = Context(c_ctx_, false);

    tiledb_array_t *c_array_ = (py::capsule)array.attr("__capsule__")();

    // we never own this pointer, pass own=false
    array_ = std::shared_ptr<tiledb::Array>(new Array(ctx_, c_array_, false),
                                            [](Array *p) {} /* no deleter*/);

    bool issparse = array_->schema().array_type() == TILEDB_SPARSE;

    query_ = std::shared_ptr<tiledb::Query>(
        new Query(ctx_, *array_, TILEDB_READ));
        //        [](Query* p){} /* note: no deleter*/);

    tiledb_layout_t layout = (tiledb_layout_t)py_layout.cast<int32_t>();
    if (!issparse && layout == TILEDB_UNORDERED) {
          TPY_ERROR_LOC("TILEDB_UNORDERED read is not supported for dense arrays")
    }
    query_->set_layout(layout);

    for (auto a : attrs) {
      attrs_.push_back(a.cast<string>());
    }

    for (auto d : dims) {
      dims_.push_back(d.cast<string>());
    }

    // get config parameters
    std::string tmp_str;
    if (config_has_key(ctx_.config(), "py.init_buffer_bytes")) {
      tmp_str = ctx_.config().get("py.init_buffer_bytes");
      try {
        init_buffer_bytes_ =
          std::stoull(tmp_str);
      } catch (const std::invalid_argument &e) {
        (void)e;
        throw std::invalid_argument("Failed to convert 'py.init_buffer_bytes' to uint64_t ('" + tmp_str + "')");
      }
    }

    if (config_has_key(ctx_.config(), "py.exp_alloc_max_bytes")) {
      tmp_str = ctx_.config().get("py.exp_alloc_max_bytes");
      try {
        exp_alloc_max_bytes_ =
          std::stoull(tmp_str);
      } catch (const std::invalid_argument &e) {
        (void)e;
        throw std::invalid_argument("Failed to convert 'py.exp_alloc_max_bytes' to uint64_t ('" + tmp_str + "')");
      }
    }

    if (config_has_key(ctx_.config(), "py.deduplicate")) {
      tmp_str = ctx_.config().get("py.deduplicate");
      if (tmp_str == "true") {
        deduplicate_ = true;
      } else if (tmp_str == "false") {
        deduplicate_ = false;
      } else {
        throw std::invalid_argument("Failed to convert configuration 'py.deduplicate' to bool ('" + tmp_str + "')");
      }
    }

    if (use_arrow.is(py::none())) {
      if (config_has_key(ctx_.config(), "py.use_arrow")) {
        tmp_str = ctx_.config().get("py.use_arrow");
        if (tmp_str == "True") {
          use_arrow_ = true;
        } else if (tmp_str == "False") {
          use_arrow_ = false;
        } else {
          throw std::invalid_argument("Failed to convert configuration 'py.use_arrow' to bool ('" + tmp_str + "')");
        }
      }
    } else {
      use_arrow_ = py::cast<bool>(use_arrow);
    }

    py::object pre_buffers = array.attr("_buffers");
    if (!pre_buffers.is(py::none())) {
      py::dict pre_buffers_dict = pre_buffers.cast<py::dict>();

      // iterate over (key, value) pairs
      for (std::pair<py::handle, py::handle> b: pre_buffers_dict) {
        py::str name = b.first.cast<py::str>();

        // unpack value tuple of (data, offsets)
        auto bfrs = b.second.cast<std::pair<py::handle, py::handle>>();
        auto data_array = bfrs.first.cast<py::array>();
        auto offsets_array = bfrs.second.cast<py::array>();

        import_buffer(name, data_array, offsets_array);
      }
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
        auto dt0 = py::isinstance<py::int_>(r0) ? r0: r0.attr("astype")(dtype);
        auto dt1 = py::isinstance<py::int_>(r1) ? r1: r1.attr("astype")(dtype);

        // TODO, this is suboptimal, should define pybind converter
        if(py::isinstance<py::int_>(dt0) && py::isinstance<py::int_>(dt1))
        {
          query_->add_range(dim_idx, py::cast<int64_t>(dt0), py::cast<int64_t>(dt1));
        }
        else
        {
          auto darray = py::array(py::make_tuple(dt0, dt1));
          query_->add_range(dim_idx, *(int64_t *)darray.data(0),
                            *(int64_t *)darray.data(1));
        }

        break;
      }
      default:
        TPY_ERROR_LOC("Unknown dim type conversion!");
      }
    } catch (py::cast_error &e) {
      (void)e;
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
      (void)e;
      return py::none();
    }
  }

  bool is_sparse() { return array_->schema().array_type() == TILEDB_SPARSE; }

  void import_buffer(std::string name, py::array data, py::array offsets) {
    auto schema = array_->schema();

    tiledb_datatype_t type;
    uint32_t cell_val_num;
    std::tie(type, cell_val_num) = buffer_type(name);
    uint64_t cell_nbytes = tiledb_datatype_size(type);
    if (cell_val_num != TILEDB_VAR_NUM)
      cell_nbytes *= cell_val_num;
    auto dtype = tiledb_dtype(type, cell_val_num);
    bool var = is_var(name);

    buffers_order_.push_back(name);
    // set nbytes and noffsets=0 here to avoid allocation; buffers set below
    auto buffer_info = BufferInfo(name, 0, type, cell_val_num, 0, var);
    buffer_info.data = data;
    buffer_info.offsets = offsets;
    buffers_.insert({name, buffer_info});
  }

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
#if TILEDB_VERSION_MAJOR == 2 && TILEDB_VERSION_MINOR == 2
      buf_nbytes = size_pair[0];
      offsets_num = size_pair[1];
#else
      buf_nbytes = size_pair.first;
      offsets_num = size_pair.second;
#endif
    } else {
      buf_nbytes = query_->est_result_size(name);
    }

    if ((var || is_sparse()) && (buf_nbytes < init_buffer_bytes_)) {
      buf_nbytes = init_buffer_bytes_;
      offsets_num = init_buffer_bytes_ / sizeof(uint64_t);
    }

    buffers_order_.push_back(name);
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
        // if we are using arrow, then we must reserve the last (uint64)
        // by not telling libtiledb about it.
        // note: we always initialize at least 10MB per buffer by default
        size_t arrow_offset_size = use_arrow_ ? 1 : 0;

        size_t offsets_size = b.offsets.size() - b.offsets_read - arrow_offset_size;
        assert(offsets_size > 0);
        uint64_t *offsets_ptr = (uint64_t *)b.offsets.data() + b.offsets_read;
        query_->set_buffer(b.name, offsets_ptr,
                           offsets_size,
                           data_ptr,
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

      // update offsets due to incomplete resets
      auto offset_ptr = buf.offsets.mutable_data();
      if (buf.isvar && (buf.offsets_read > 0)) {
        if (offset_ptr[buf.offsets_read] == 0) {
          auto last_offset = offset_ptr[buf.offsets_read - 1];
          auto last_size = (buf.data_vals_read * buf.elem_nbytes) - last_offset;

          for (uint64_t i = 0; i < offset_elem_num; i++) {
            offset_ptr[buf.offsets_read + i] += last_offset + last_size;
          }
        }
      }

      buf.data_vals_read += data_vals_num;
      buf.offsets_read += offset_elem_num;
    }
  }

  void submit_read() {
    auto start = std::chrono::high_resolution_clock::now();

    if (buffers_.size() != 0) {
      // we have externally imported buffers
      return;
    }

    auto schema = array_->schema();

    // return dims first, if any
    auto domain = schema.domain();
    for (size_t dim_idx = 0; dim_idx < domain.ndim(); dim_idx++) {
      auto dim = domain.dimension(dim_idx);
      if ((std::find(dims_.begin(), dims_.end(), dim.name()) == dims_.end()) &&
          // we need to also check if this is an attr for backward-compatibility
          (std::find(attrs_.begin(), attrs_.end(), dim.name()) == attrs_.end())) {
        continue;
      }
      alloc_buffer(dim.name());
    }

    // schema.attributes() is unordered, but we need to return ordered results
    for (size_t attr_idx = 0; attr_idx < schema.attribute_num(); attr_idx++) {
      auto attr = schema.attribute(attr_idx);
      if (std::find(attrs_.begin(), attrs_.end(), attr.name()) == attrs_.end()) {
        continue;
      }
      alloc_buffer(attr.name());
    }

    set_buffers();

    size_t max_retries = 0, retries = 0;
    std::string tmp_str;
    try {
      tmp_str = ctx_.config().get("py.max_incomplete_retries");
      max_retries =
          std::stoull(tmp_str);
    } catch (const std::invalid_argument &e) {
      (void)e;
      throw TileDBError("Failed to convert 'py.max_incomplete_retries' to uint64_t ('" + tmp_str + "')");
    } catch (tiledb::TileDBError &e) {
      (void)e;
      max_retries = 100;
    }

    auto start_submit = std::chrono::high_resolution_clock::now();
    {
      py::gil_scoped_release release;
      query_->submit();
    }

    if (g_stats) {
      auto now = std::chrono::high_resolution_clock::now();
      g_stats.get()->counters["py.read_query_initial_submit_time"] += now - start_submit;
    }

    // TODO: would be nice to have a callback here for custom realloc strategy
    while (query_->query_status() == Query::Status::INCOMPLETE) {
      if (++retries > max_retries)
        TPY_ERROR_LOC(
            "Exceeded maximum retries ('py.max_incomplete_retries': '" +
            std::to_string(max_retries) + "')");

      // Track the time we spend inside of incomplete updates only
      auto start_incomplete_update = std::chrono::high_resolution_clock::now();

      update_read_elem_num();

      for (auto &bp : buffers_) {
        auto buf = bp.second;

        // Check if values buffer should be resized
        if ((int64_t)(buf.data_vals_read * buf.elem_nbytes) > (buf.data.nbytes() + 1) / 2) {
          size_t new_size = buf.data.size() * 2;
          buf.data.resize({new_size}, false);
        }

        // Check if offset buffer should be resized
        if (buf.isvar && (int64_t)(buf.offsets_read * sizeof(uint64_t)) > (buf.offsets.nbytes() + 1) / 2) {
          size_t new_offsets_size = buf.offsets.size() * 2;
          buf.offsets.resize({new_offsets_size}, false);
        }
      }

      // note: this block confuses lldb. continues from here unless bp set after block.
      set_buffers();

      if (g_stats) {
        auto now = std::chrono::high_resolution_clock::now();
        g_stats.get()->counters["py.read_incomplete_update_time"] += now - start_incomplete_update;
      }

      auto start_incomplete = std::chrono::high_resolution_clock::now();
      {
        py::gil_scoped_release release;
        query_->submit();
      }
      if (g_stats) {
        auto now = std::chrono::high_resolution_clock::now();
        g_stats.get()->counters["py.read_query_incomplete_submit_time"] += now - start_incomplete;
      }
    }

    auto start_finalize = std::chrono::high_resolution_clock::now();

    update_read_elem_num();

    // resize the output buffers to match the final read total
    size_t arrow_offset_size = use_arrow_ ? 1 : 0;
    for (auto& bp : buffers_) {
      auto name = bp.first;
      auto &buf = bp.second;

      buf.data.resize({buf.data_vals_read * buf.elem_nbytes});
      buf.offsets.resize({buf.offsets_read + arrow_offset_size});

      // if we are using arrow, we need to reset the read counts
      // calling set_buffers below.
      if (use_arrow_) {
        buf.data_vals_read = 0;
        buf.offsets_read = 0;
      }
    }

    if (use_arrow_) {
      // this is a very light hack:
      // call set_buffers here to reset the buffers to the *full*
      // buffer in case there were incomplete queries. without this call,
      // the active tiledb::Query only knows about the buffer ptr/size
      // for the *last* submit loop, so we don't get full result set.
      set_buffers();
    }

    if (g_stats) {
      auto now = std::chrono::high_resolution_clock::now();
      g_stats.get()->counters["py.finalize_buffers_time"] += now - start_finalize;
    }

    if (g_stats) {
      auto now = std::chrono::high_resolution_clock::now();
      g_stats.get()->counters["py.read_query_time"] += now - start;
    }
  }

  py::array unpack_buffer(std::string name,
      py::array buf, py::array_t<uint64_t> off) {

    auto start = std::chrono::high_resolution_clock::now();

    if (buf.size() < 1)
      TPY_ERROR_LOC(std::string("Unexpected empty buffer array ('") + name + "')");
    if (off.size() < 1)
      TPY_ERROR_LOC(std::string("Unexpected empty offsets array ('") + name + "')");

    auto dtype = buffer_dtype(name);
    bool is_unicode = dtype.is(py::dtype("U"));
    bool is_str = dtype.is(py::dtype("S"));
    if (is_unicode || is_str) {
      dtype = py::dtype("O");
    }

    // Hashmap for string deduplication
    //fastest so far:
    typedef tsl::robin_map<size_t, uint64_t> MapType;
    MapType map;
    std::vector<py::object> object_v;
    if (is_unicode) {
      map.reserve(size_t(off.size() / 10) + 1);
    }

    auto result_array = py::array(py::dtype("O"), off.size());
    auto result_p = (py::object*) result_array.mutable_data();
    uint64_t last = 0;
    uint64_t cur = 0;
    size_t size = 0;
    uint64_t create = 0;

    auto off_data = off.data();
    last = off_data[0]; // initial should always be 0
    for (auto i = 1; i < off.size() + 1; i++) {
      if (i == off.size())
        cur = buf.nbytes();
      else {
        cur = off_data[i];
      }

      size = cur - last;

      py::object o;
      auto data_ptr = (char*)buf.data() + last;
      if (is_unicode)
        if (data_ptr[0] == '\0' && size == 1) {
          o = py::str("");
        } else {
          if (!deduplicate_) {
            o = py::str(data_ptr, size);
          } else {
            auto v = nonstd::string_view{data_ptr, size};
            auto h = std::hash<nonstd::string_view>()(v);
            auto needle = map.find(h);
            if (needle == map.end()) {
              o = py::str(data_ptr, size);
              map.insert(needle, {h, create});
              object_v.push_back(o);
              create++;
            } else {
              auto idx = needle->second;
              o = object_v[idx];
            }
          }
        }
      else if (is_str)
        if (data_ptr[0] == '\0' && size == 1) {
          o = py::bytes("");
        } else {
          o = py::bytes(data_ptr, size);
        }
      else {
        o = py::array(py::dtype("uint8"), size, data_ptr);
        o.attr("dtype") = dtype;
      }

      result_p[i-1] = o;
      last = cur;
    }

    if (g_stats) {
      auto now = std::chrono::high_resolution_clock::now();
      g_stats.get()->counters["py.buffer_conversion_time"] += now - start;
    }

    return result_array;
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
    for (auto &buffer_name : buffers_order_) {
      auto bp = buffers_.at(buffer_name);
      results[py::str(buffer_name)] =
          py::make_tuple(bp.data, bp.offsets);
    }
    return results;
  }

  std::unique_ptr<PAPair> buffer_to_pa(std::string name) {
    if (query_->query_status() != tiledb::Query::Status::COMPLETE)
      TPY_ERROR_LOC("Cannot convert buffers unless Query is complete");

    tiledb::arrow::ArrowAdapter adapter(query_);
    std::unique_ptr<PAPair> pa_pair(new PAPair());

    adapter.export_buffer(name.c_str(), &(pa_pair->array_), &(pa_pair->schema_));
    pa_pair->exported_ = true;

    return pa_pair;
  }

  py::object buffers_to_pa_table() {
    using namespace pybind11::literals;

    if (query_->query_status() != tiledb::Query::Status::COMPLETE)
      TPY_ERROR_LOC("Query must be complete to convert buffers");

    auto pa = py::module::import("pyarrow");
    auto pa_array_import = pa.attr("Array").attr("_import_from_c");
    tiledb::arrow::ArrowAdapter adapter(query_);

    py::list names;
    py::list results;
    for (auto &buffer_name : buffers_order_) {
      ArrowArray c_pa_array;
      ArrowSchema c_pa_schema;
      adapter.export_buffer(buffer_name.c_str(), static_cast<void*>(&c_pa_array),
                                                 static_cast<void*>(&c_pa_schema));

      py::object pa_array = pa_array_import(py::int_((ptrdiff_t)&c_pa_array),
                                            py::int_((ptrdiff_t)&c_pa_schema));
      results.append(pa_array);
      names.append(buffer_name);
    }

    auto pa_table = pa.attr("Table").attr("from_arrays")(results, "names"_a=names);

    return pa_table;
  }

  py::array _test_array() {
    py::array_t<uint8_t> a;
    a.resize({10});

    a.resize({20});
    return std::move(a);
  }

  uint64_t _test_init_buffer_bytes() {
    return init_buffer_bytes_;
  }
};

void init_stats() {
  g_stats.reset(new StatsInfo());

  auto stats_counters = g_stats.get()->counters;
  stats_counters["py.read_query_time"] = TimerType();
  stats_counters["py.write_query_time"] = TimerType();
  stats_counters["py.buffer_conversion_time"] = TimerType();
  stats_counters["py.read_query_initial_submit_time"] = TimerType();
  stats_counters["py.read_incomplete_update_time"] = TimerType();
}

void disable_stats() {
  g_stats.reset(nullptr);
}

void increment_stat(std::string key, double value) {
  auto &stats_counters = g_stats.get()->counters;

  if (stats_counters.count(key) == 0)
    stats_counters[key] = TimerType();

  auto &timer = stats_counters[key];
  auto incr = std::chrono::duration<double>(value);
  timer += incr;
}

bool use_stats() {
  return (bool)g_stats;
}

py::object get_stats() {
  if (!g_stats) {
    TPY_ERROR_LOC("Stats counters are not uninitialized!")
  }

  auto &stats_counters = g_stats.get()->counters;

  py::dict res;
  for (auto iter = stats_counters.begin(); iter != stats_counters.end(); ++iter) {
    auto val = std::chrono::duration<double>(iter->second);
    res[py::str(iter->first)] = py::float_(val.count());
  }
  return std::move(res);
}

std::string python_internal_stats() {
  if (!g_stats) {
    TPY_ERROR_LOC("Stats counters are not uninitialized!")
  }

  auto counters = g_stats.get()->counters;

  std::ostringstream os;

  // core.cc is only tracking read time right now; don't print if we
  // have no query submission time
  auto rq_time = counters["py.read_query_initial_submit_time"].count();
  if (rq_time == 0)
    return os.str();

  os << std::endl;
  os << "==== Python Stats ====" << std::endl << std::endl;
  os << "- TileDB-Py Indexing Time: " << counters["py.__getitem__time"].count() << std::endl;
  os << "  * TileDB-Py query execution time: " << counters["py.read_query_time"].count() << std::endl;
  os << "    > TileDB C++ Core initial query submit time: " <<
    counters["py.read_query_initial_submit_time"].count() << std::endl;

  std::string key1 = "py.read_query_incomplete_submit_time";
  if (counters.count(key1) == 1) {
      os << "    > TileDB C++ Core incomplete resubmit(s) time: " << counters[key1].count() << std::endl;
      os << "    > TileDB-Py incomplete buffer updates: " <<
        counters["py.read_incomplete_update_time"].count() << std::endl;
  }

  std::string key3 = "py.buffer_conversion_time";
  if (counters.count(key3) == 1) {
      os << "  * TileDB-Py buffer conversion time: " << counters[key3].count() << std::endl;
  }

  return os.str();
}

PYBIND11_MODULE(core, m) {
  py::class_<PyQuery>(m, "PyQuery")
      .def(py::init<py::object, py::object, py::iterable, py::object, py::object, py::object>())
      .def("set_ranges", &PyQuery::set_ranges)
      .def("set_subarray", &PyQuery::set_subarray)
      .def("submit", &PyQuery::submit)
      .def("results", &PyQuery::results)
      .def("buffer_dtype", &PyQuery::buffer_dtype)
      .def("unpack_buffer", &PyQuery::unpack_buffer)
      .def("_buffer_to_pa", &PyQuery::buffer_to_pa)
      .def("_buffers_to_pa_table", &PyQuery::buffers_to_pa_table)
      .def("_test_array", &PyQuery::_test_array)
      .def("_test_err",
           [](py::object self, std::string s) { throw TileDBPyError(s); })
      .def_property_readonly("_test_init_buffer_bytes", &PyQuery::_test_init_buffer_bytes);

  m.def("array_to_buffer", &convert_np);

  m.def("init_stats", &init_stats);
  m.def("disable_stats", &init_stats);
  m.def("python_internal_stats", &python_internal_stats);
  m.def("increment_stat", &increment_stat);
  m.def("get_stats", &get_stats);
  m.def("use_stats", &use_stats);

  py::class_<PAPair>(m, "PAPair")
    .def(py::init())
    .def("get_array", &PAPair::get_array)
    .def("get_schema", &PAPair::get_schema);

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
    } catch (py::builtin_exception &e) {
      // just forward the error
      throw;
    //} catch (std::runtime_error &e) {
    //  std::cout << "unexpected runtime_error: " << e.what() << std::endl;
    }
  });
}

}; // namespace tiledbpy
