#include <chrono>
#include <cmath>
#include <cstring>
#include <future>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "npbuffer.h"
#include "util.h"
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <tiledb/tiledb.h>              // C
#include <tiledb/tiledb>                // C++
#include <tiledb/tiledb_experimental.h> // C
#include <tiledb/tiledb_experimental>   // C++

// clang-format off
// do not re-order these headers
#include "py_arrowio"
#include "py_arrow_io_impl.h"
// clang-format on

#if defined(TILEDB_SERIALIZATION)
#include <tiledb/tiledb_serialization.h> // C
#endif

#include "../external/tsl/robin_map.h"

#if !defined(NDEBUG)
// #include "debug.cc"
#endif

#include "query_condition.cc"

namespace tiledbpy {

using namespace std;
using namespace tiledb;
namespace py = pybind11;
using namespace pybind11::literals;

using TimerType = std::chrono::duration<double>;
struct StatsInfo {
  std::map<std::string, TimerType> counters;
};

bool config_has_key(tiledb::Config config, std::string key) {
#if TILEDB_VERSION_MAJOR >= 2 && TILEDB_VERSION_MINOR >= 9
  return config.contains(key);
#else
  try {
    config.get(key);
  } catch (TileDBError &e) {
    (void)e;
    return false;
  }
  return true;
#endif
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
             uint32_t cell_val_num, size_t offsets_num, size_t validity_num,
             bool isvar = false, bool isnullable = false)

      : name(name), type(data_type), cell_val_num(cell_val_num), isvar(isvar),
        isnullable(isnullable) {

    try {
      dtype = tiledb_dtype(data_type, cell_val_num);
      elem_nbytes = tiledb_datatype_size(type);
      data = py::array(py::dtype("uint8"), data_nbytes);
      offsets = py::array_t<uint64_t>(offsets_num);
      validity = py::array_t<uint8_t>(validity_num);
    } catch (py::error_already_set &e) {
      TPY_ERROR_LOC(e.what())
    }
    // TODO use memset here for zero'd buffers in debug mode
  }

  string name;
  tiledb_datatype_t type;
  py::dtype dtype;
  size_t elem_nbytes = 1;
  uint64_t data_vals_read = 0;
  uint32_t cell_val_num;
  uint64_t offsets_read = 0;
  uint64_t validity_vals_read = 0;
  bool isvar;
  bool isnullable;

  py::array data;
  py::array_t<uint64_t> offsets;
  py::array_t<uint8_t> validity;
};

struct BufferHolder {
  py::object data;
  py::object offsets;
  py::object validity;

  BufferHolder(py::object d, py::object o, py::object v)
      : data(d), offsets(o), validity(v) {}

  static void free_buffer_holder(BufferHolder *self) { delete self; }
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
      return py::dtype("M8[Y]");
    case TILEDB_DATETIME_MONTH:
      return py::dtype("M8[M]");
    case TILEDB_DATETIME_WEEK:
      return py::dtype("M8[W]");
    case TILEDB_DATETIME_DAY:
      return py::dtype("M8[D]");
    case TILEDB_DATETIME_HR:
      return py::dtype("M8[h]");
    case TILEDB_DATETIME_MIN:
      return py::dtype("M8[m]");
    case TILEDB_DATETIME_SEC:
      return py::dtype("M8[s]");
    case TILEDB_DATETIME_MS:
      return py::dtype("M8[ms]");
    case TILEDB_DATETIME_US:
      return py::dtype("M8[us]");
    case TILEDB_DATETIME_NS:
      return py::dtype("M8[ns]");
    case TILEDB_DATETIME_PS:
      return py::dtype("M8[ps]");
    case TILEDB_DATETIME_FS:
      return py::dtype("M8[fs]");
    case TILEDB_DATETIME_AS:
      return py::dtype("M8[as]");

#if TILEDB_VERSION_MAJOR >= 2 && TILEDB_VERSION_MINOR >= 3
    /* duration types map to timedelta */
    case TILEDB_TIME_HR:
      return py::dtype("m8[h]");
    case TILEDB_TIME_MIN:
      return py::dtype("m8[m]");
    case TILEDB_TIME_SEC:
      return py::dtype("m8[s]");
    case TILEDB_TIME_MS:
      return py::dtype("m8[ms]");
    case TILEDB_TIME_US:
      return py::dtype("m8[us]");
    case TILEDB_TIME_NS:
      return py::dtype("m8[ns]");
    case TILEDB_TIME_PS:
      return py::dtype("m8[ps]");
    case TILEDB_TIME_FS:
      return py::dtype("m8[fs]");
    case TILEDB_TIME_AS:
      return py::dtype("m8[as]");
#endif
#if TILEDB_VERSION_MAJOR >= 2 && TILEDB_VERSION_MINOR >= 9
    case TILEDB_BLOB:
      return py::dtype("byte");
#endif
#if TILEDB_VERSION_MAJOR >= 2 && TILEDB_VERSION_MINOR >= 10
    case TILEDB_BOOL:
      return py::dtype("bool");
#endif

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

py::array_t<uint8_t>
uint8_bool_to_uint8_bitmap(py::array_t<uint8_t> validity_array) {
  // TODO profile, probably replace; avoid inplace reassignment
  auto np = py::module::import("numpy");
  auto packbits = np.attr("packbits");
  auto tmp = packbits(validity_array, "bitorder"_a = "little");
  return tmp;
}

uint64_t count_zeros(py::array_t<uint8_t> a) {
  uint64_t count = 0;
  for (Py_ssize_t idx = 0; idx < a.size(); idx++)
    count += (a.data()[idx] == 0) ? 1 : 0;
  return count;
}

class PyQuery {

private:
  Context ctx_;
  shared_ptr<tiledb::Domain> domain_;
  shared_ptr<tiledb::ArraySchema> array_schema_;
  shared_ptr<tiledb::Array> array_;
  shared_ptr<tiledb::Query> query_;
  std::vector<std::string> attrs_;
  std::vector<std::string> dims_;
  map<string, BufferInfo> buffers_;
  vector<string> buffers_order_;

  bool deduplicate_ = true;
  bool use_arrow_ = false;
  // initialize the query buffers with exactly `init_buffer_bytes`
  // rather than the estimated result size. for incomplete testing.
  bool exact_init_bytes_ = false;
  uint64_t init_buffer_bytes_ = DEFAULT_INIT_BUFFER_BYTES;
  uint64_t alloc_max_bytes_ = DEFAULT_ALLOC_MAX_BYTES;
  tiledb_layout_t layout_ = TILEDB_ROW_MAJOR;

  // label buffer list
  std::unordered_map<string, uint64_t> label_input_buffer_data_;

  std::string uri_;

public:
  tiledb_ctx_t *c_ctx_;
  tiledb_array_t *c_array_;
  bool preload_metadata_ = false;
  bool return_incomplete_ = false;
  size_t retries_ = 0;

public:
  PyQuery() = delete;

  PyQuery(const Context &ctx, py::object array, py::iterable attrs,
          py::iterable dims, py::object py_layout, py::object use_arrow)
      : ctx_(ctx) {
    init_config();
    // initialize arrow argument from user, if provided
    // call after init_config
    if (!use_arrow.is(py::none())) {
      use_arrow_ = py::cast<bool>(use_arrow);
    }

    tiledb_array_t *c_array_ = (py::capsule)array.attr("__capsule__")();

    // we never own this pointer, pass own=false
    array_ = std::shared_ptr<tiledb::Array>(new Array(ctx_, c_array_, false));

    array_schema_ =
        std::shared_ptr<tiledb::ArraySchema>(new ArraySchema(array_->schema()));

    domain_ =
        std::shared_ptr<tiledb::Domain>(new Domain(array_schema_->domain()));

    uri_ = array.attr("uri").cast<std::string>();

    bool issparse = array_->schema().array_type() == TILEDB_SPARSE;

    std::string mode = py::str(array.attr("mode"));
    auto query_mode = TILEDB_READ;
    if (mode == "r") {
      query_mode = TILEDB_READ;
    } else if (mode == "d") {
      query_mode = TILEDB_DELETE;
    } else {
      throw std::invalid_argument("Invalid query mode: " + mode);
    }

    // initialize the dims that we are asked to read
    for (auto d : dims) {
      dims_.push_back(d.cast<string>());
    }

    // initialize the attrs that we are asked to read
    for (auto a : attrs) {
      attrs_.push_back(a.cast<string>());
    }

    if (query_mode == TILEDB_READ) {
      py::object pre_buffers = array.attr("_buffers");
      if (!pre_buffers.is(py::none())) {
        py::dict pre_buffers_dict = pre_buffers.cast<py::dict>();

        // iterate over (key, value) pairs
        for (std::pair<py::handle, py::handle> b : pre_buffers_dict) {
          py::str name = b.first.cast<py::str>();

          // unpack value tuple of (data, offsets)
          auto bfrs = b.second.cast<std::pair<py::handle, py::handle>>();
          auto data_array = bfrs.first.cast<py::array>();
          auto offsets_array = bfrs.second.cast<py::array>();

          import_buffer(name, data_array, offsets_array);
        }
      }
    }

    query_ =
        std::shared_ptr<tiledb::Query>(new Query(ctx_, *array_, query_mode));
    //        [](Query* p){} /* note: no deleter*/);

    if (query_mode == TILEDB_READ) {
      layout_ = (tiledb_layout_t)py_layout.cast<int32_t>();
      if (!issparse && layout_ == TILEDB_UNORDERED) {
        TPY_ERROR_LOC("TILEDB_UNORDERED read is not supported for dense arrays")
      }
      query_->set_layout(layout_);
    }

#if TILEDB_VERSION_MAJOR >= 2 && TILEDB_VERSION_MINOR >= 2
    if (use_arrow_) {
      // enable arrow mode in the Query
      auto tmp_config = ctx_.config();
      tmp_config.set("sm.var_offsets.bitsize", "64");
      tmp_config.set("sm.var_offsets.mode", "elements");
      tmp_config.set("sm.var_offsets.extra_element", "true");
      ctx_.handle_error(tiledb_query_set_config(
          ctx_.ptr().get(), query_->ptr().get(), tmp_config.ptr().get()));
    }
#endif
  }

  void set_subarray(py::object py_subarray) {
    tiledb::Subarray *subarray = py_subarray.cast<tiledb::Subarray *>();
    query_->set_subarray(*subarray);
  }

#if defined(TILEDB_SERIALIZATION)
  void set_serialized_query(py::buffer serialized_query) {
    int rc;
    tiledb_query_t *c_query;
    tiledb_buffer_t *c_buffer;
    tiledb_ctx_t *c_ctx = ctx_.ptr().get();

    rc = tiledb_buffer_alloc(c_ctx, &c_buffer);
    if (rc == TILEDB_ERR)
      TPY_ERROR_LOC("Could not allocate c_buffer.");

    py::buffer_info buffer_info = serialized_query.request();
    rc = tiledb_buffer_set_data(c_ctx, c_buffer, buffer_info.ptr,
                                buffer_info.shape[0]);
    if (rc == TILEDB_ERR)
      TPY_ERROR_LOC("Could not set c_buffer.");

    c_query = query_.get()->ptr().get();
    rc = tiledb_deserialize_query(c_ctx, c_buffer, TILEDB_CAPNP, 0, c_query);
    if (rc == TILEDB_ERR)
      TPY_ERROR_LOC("Could not deserialize query.");
  }
#endif

  void set_cond(py::object cond) {
    py::object init_pyqc = cond.attr("init_query_condition");

    try {
      init_pyqc(uri_, attrs_, ctx_);
    } catch (tiledb::TileDBError &e) {
      TPY_ERROR_LOC(e.what());
    } catch (py::error_already_set &e) {
      TPY_ERROR_LOC(e.what());
    }
    auto pyqc = (cond.attr("c_obj")).cast<PyQueryCondition>();
    auto qc = pyqc.ptr().get();
    query_->set_condition(*qc);
  }

  bool is_dimension(std::string name) { return domain_->has_dimension(name); }

  bool is_attribute(std::string name) {
    return array_schema_->has_attribute(name);
  }

  bool is_dimension_label(std::string name) {
#if TILEDB_VERSION_MAJOR == 2 && TILEDB_VERSION_MINOR >= 15
    return ArraySchemaExperimental::has_dimension_label(ctx_, *array_schema_,
                                                        name);
#else
    return false;
#endif
  }

  bool is_var(std::string name) {
    if (is_dimension(name)) {
      auto dim = domain_->dimension(name);
      return dim.cell_val_num() == TILEDB_VAR_NUM;
    } else if (is_attribute(name)) {
      auto attr = array_schema_->attribute(name);
      return attr.cell_val_num() == TILEDB_VAR_NUM;
#if TILEDB_VERSION_MAJOR == 2 && TILEDB_VERSION_MINOR >= 15
    } else if (is_dimension_label(name)) {
      auto dim_label =
          ArraySchemaExperimental::dimension_label(ctx_, *array_schema_, name);
      return dim_label.label_cell_val_num() == TILEDB_VAR_NUM;
#endif
    } else {
      TPY_ERROR_LOC("Unknown buffer type for is_var check (expected attribute "
                    "or dimension)")
    }
  }

  bool is_nullable(std::string name) {
    if (is_dimension(name) || is_dimension_label(name)) {
      return false;
    }

    auto attr = array_schema_->attribute(name);
    return attr.nullable();
  }

  std::pair<tiledb_datatype_t, uint32_t> buffer_type(std::string name) {
    tiledb_datatype_t type;
    uint32_t cell_val_num;
    if (is_dimension(name)) {
      type = domain_->dimension(name).type();
      cell_val_num = domain_->dimension(name).cell_val_num();
    } else if (is_attribute(name)) {
      type = array_schema_->attribute(name).type();
      cell_val_num = array_schema_->attribute(name).cell_val_num();
#if TILEDB_VERSION_MAJOR == 2 && TILEDB_VERSION_MINOR >= 15
    } else if (is_dimension_label(name)) {
      auto dim_label =
          ArraySchemaExperimental::dimension_label(ctx_, *array_schema_, name);
      type = dim_label.label_type();
      cell_val_num = dim_label.label_cell_val_num();
#endif
    } else {
      TPY_ERROR_LOC("Unknown buffer '" + name + "'");
    }
    return {type, cell_val_num};
  }

  uint32_t buffer_ncells(std::string name) {
    if (is_dimension(name)) {
      return domain_->dimension(name).cell_val_num();
    } else if (is_attribute(name)) {
      return array_schema_->attribute(name).cell_val_num();
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

    tiledb_datatype_t type;
    uint32_t cell_val_num;
    std::tie(type, cell_val_num) = buffer_type(name);
    auto dtype = tiledb_dtype(type, cell_val_num);

    buffers_order_.push_back(name);
    // set nbytes and noffsets=0 here to avoid allocation; buffers set below
    auto buffer_info = BufferInfo(name, 0, type, cell_val_num, 0,
                                  0, // TODO
                                  is_var(name), is_nullable(name));
    buffer_info.data = data;
    buffer_info.offsets = offsets;
    buffers_.insert({name, buffer_info});
  }

  void alloc_buffer(std::string name) {
    tiledb_datatype_t type;
    uint32_t cell_val_num;
    uint64_t cell_nbytes;
    bool var;
    bool nullable;
    uint64_t buf_nbytes = 0;
    uint64_t offsets_num = 0;
    uint64_t validity_num = 0;
    bool dense = array_schema_->array_type() == TILEDB_DENSE;
    if (is_dimension_label(name)) {
#if TILEDB_VERSION_MAJOR == 2 && TILEDB_VERSION_MINOR >= 15
      auto dim_label =
          ArraySchemaExperimental::dimension_label(ctx_, *array_schema_, name);
      type = dim_label.label_type();
      cell_val_num = dim_label.label_cell_val_num();
      var = cell_val_num == TILEDB_VAR_NUM;
      nullable = false;

      cell_nbytes = tiledb_datatype_size(type);
      uint64_t ncells = label_input_buffer_data_[name];

      if (!var) {
        cell_nbytes *= cell_val_num;
      } else {
        offsets_num = ncells;
      }
      buf_nbytes = ncells * cell_nbytes;
#endif
    } else {
      std::tie(type, cell_val_num) = buffer_type(name);
      cell_nbytes = tiledb_datatype_size(type);
      if (cell_val_num != TILEDB_VAR_NUM) {
        cell_nbytes *= cell_val_num;
      }
      var = is_var(name);
      nullable = is_nullable(name);

      if (retries_ < 1 && dense) {
        // we must not call after submitting
        if (nullable && var) {
          auto sizes = query_->est_result_size_var_nullable(name);
          offsets_num = sizes[0];
          buf_nbytes = sizes[1];
          validity_num = sizes[2] / sizeof(uint8_t);
        } else if (nullable && !var) {
          auto sizes = query_->est_result_size_nullable(name);
          buf_nbytes = sizes[0];
          validity_num = sizes[1] / sizeof(uint8_t);
        } else if (!nullable && var) {
          auto size_pair = query_->est_result_size_var(name);
#if TILEDB_VERSION_MAJOR == 2 && TILEDB_VERSION_MINOR < 2
          buf_nbytes = size_pair.first;
          offsets_num = size_pair.second;
#else
          buf_nbytes = size_pair[0];
          offsets_num = size_pair[1];
#endif
        } else { // !nullable && !var
          buf_nbytes = query_->est_result_size(name);
        }

        // Add extra offset to estimate in order to avoid incomplete resubmit
        // libtiledb 2.7.* does not include extra element in estimate.
        // Remove this section after resolution of SC-16301.
        offsets_num += (var && use_arrow_) ? 1 : 0;
      }
    }

    // - for sparse arrays: don't try to allocate more than alloc_max_bytes_
    // - for dense arrays: the estimate should be exact, so don't cap
    if (is_sparse() && buf_nbytes > alloc_max_bytes_) {
      buf_nbytes = alloc_max_bytes_;
    }
    // use max to avoid overflowing to zero in the multiplication, in case the
    //   estimate is too large
    if (max(validity_num, validity_num * sizeof(uint8_t)) > alloc_max_bytes_) {
      validity_num = alloc_max_bytes_ / sizeof(uint8_t);
    }
    if (max(offsets_num, offsets_num * sizeof(uint64_t)) > alloc_max_bytes_) {
      offsets_num = alloc_max_bytes_ / sizeof(uint64_t);
    }

    // use init_buffer_bytes configuration option if the
    // estimate is smaller
    if ((var || is_sparse()) &&
        (buf_nbytes < init_buffer_bytes_ || exact_init_bytes_)) {
      buf_nbytes = init_buffer_bytes_;
      offsets_num = init_buffer_bytes_ / sizeof(uint64_t);
      validity_num = init_buffer_bytes_ / cell_nbytes;
    }

    buffers_order_.push_back(name);
    buffers_.insert(
        {name, BufferInfo(name, buf_nbytes, type, cell_val_num, offsets_num,
                          validity_num, var, nullable)});
  }

  void add_label_buffer(std::string &label_name, uint64_t ncells) {
    label_input_buffer_data_[label_name] = ncells;
  }

  py::object get_buffers() {
    py::dict rmap;
    for (auto &bp : buffers_) {
      py::list result;
      const auto name = bp.first;
      const BufferInfo b = bp.second;
      result.append(b.data);
      result.append(b.offsets);
      rmap[py::str{name}] = result;
    }
    return std::move(rmap);
  }

  void set_buffers() {
    for (auto bp : buffers_) {
      auto name = bp.first;
      const BufferInfo b = bp.second;

      size_t offsets_read = b.offsets_read;
      size_t data_vals_read = b.data_vals_read;
      size_t validity_vals_read = b.validity_vals_read;

      void *data_ptr =
          (void *)((char *)b.data.data() + (data_vals_read * b.elem_nbytes));
      uint64_t data_nelem =
          (b.data.size() - (data_vals_read * b.elem_nbytes)) / b.elem_nbytes;

#if TILEDB_VERSION_MAJOR >= 2 && TILEDB_VERSION_MINOR >= 15
      // Experimental version of API call is needed to support type-checking
      // on dimension label buffers.
      QueryExperimental::set_data_buffer(*query_, b.name, data_ptr, data_nelem);
#else
      query_->set_data_buffer(b.name, data_ptr, data_nelem);
#endif

      if (b.isvar) {
        size_t offsets_size = b.offsets.size() - offsets_read;
        uint64_t *offsets_ptr = (uint64_t *)b.offsets.data() + offsets_read;

        query_->set_offsets_buffer(b.name, (uint64_t *)(offsets_ptr),
                                   offsets_size);
      }
      if (b.isnullable) {
        uint64_t validity_size = b.validity.size() - validity_vals_read;
        uint8_t *validity_ptr =
            (uint8_t *)b.validity.data() + validity_vals_read;

        query_->set_validity_buffer(b.name, validity_ptr, validity_size);
      }
    }
  }

  void update_read_elem_num() {
#if TILEDB_VERSION_MAJOR >= 2 && TILEDB_VERSION_MINOR >= 16
    auto result_elements =
        QueryExperimental::result_buffer_elements_nullable_labels(*query_);
#elif TILEDB_VERSION_MAJOR >= 2 && TILEDB_VERSION_MINOR >= 3
    // needs https://github.com/TileDB-Inc/TileDB/pull/2238
    auto result_elements = query_->result_buffer_elements_nullable();
#else
    auto result_elements = query_->result_buffer_elements_nullable();
    auto result_offsets_tmp = query_->result_buffer_elements();
#endif

    for (const auto &read_info : result_elements) {
      auto name = read_info.first;
      uint64_t offset_elem_num = 0, data_vals_num = 0, validity_elem_num = 0;
      std::tie(offset_elem_num, data_vals_num, validity_elem_num) =
          read_info.second;

#if TILEDB_VERSION_MAJOR >= 2 && TILEDB_VERSION_MINOR < 3
      // we need to fix-up the offset count b/c incorrect before 2.3
      // (https://github.com/TileDB-Inc/TileDB/pull/2238)
      offset_elem_num = result_offsets_tmp[name].first;
#endif

      BufferInfo &buf = buffers_.at(name);

      // TODO if we ever support per-attribute read offset bitsize
      // then need to handle here. Currently this is hard-coded to
      // 64-bit to match query config.
      auto offset_ptr = buf.offsets.mutable_data();

      if (buf.isvar) {

        if (offset_elem_num > 0) {
          // account for 'sm.var_offsets.extra_element'
          offset_elem_num -= (use_arrow_) ? 1 : 0;
        }

        if (buf.offsets_read > 0) {
          if (offset_ptr[buf.offsets_read] == 0) {
            auto last_size = (buf.data_vals_read * buf.elem_nbytes);

            for (uint64_t i = 0; i < offset_elem_num; i++) {
              offset_ptr[buf.offsets_read + i] += last_size;
            }
          }
        }
      }

      buf.data_vals_read += data_vals_num;
      buf.offsets_read += offset_elem_num;
      buf.validity_vals_read += validity_elem_num;

      if ((Py_ssize_t)(buf.data_vals_read * buf.elem_nbytes) >
          (Py_ssize_t)buf.data.size()) {
        throw TileDBError(
            "After read query, data buffer out of bounds: " + name + " (" +
            std::to_string(buf.data_vals_read * buf.elem_nbytes) + " > " +
            std::to_string(buf.data.size()) + ")");
      }
      if ((Py_ssize_t)buf.offsets_read > buf.offsets.size()) {
        throw TileDBError("After read query, offsets buffer out of bounds: " +
                          name + " (" + std::to_string(buf.offsets_read) +
                          " > " + std::to_string(buf.offsets.size()) + ")");
      }
      if ((Py_ssize_t)buf.validity_vals_read > buf.validity.size()) {
        throw TileDBError("After read query, validity buffer out of bounds: " +
                          name + " (" + std::to_string(buf.validity_vals_read) +
                          " > " + std::to_string(buf.validity.size()) + ")");
      }
    }
  }

  void reset_read_elem_num() {
    for (auto &bp : buffers_) {
      auto &buf = bp.second;

      buf.offsets_read = 0;
      buf.data_vals_read = 0;
      buf.validity_vals_read = 0;
    }
  }

  uint64_t get_max_retries() {
    // should make this a templated getter for any key w/ default
    std::string tmp_str;
    size_t max_retries;
    try {
      tmp_str = ctx_.config().get("py.max_incomplete_retries");
      max_retries = std::stoull(tmp_str);
    } catch (const std::invalid_argument &e) {
      (void)e;
      throw TileDBError(
          "Failed to convert 'py.max_incomplete_retries' to uint64_t ('" +
          tmp_str + "')");
    } catch (tiledb::TileDBError &e) {
      (void)e;
      max_retries = 100;
    }
    return max_retries;
  }

  void resubmit_read() {
#if TILEDB_VERSION_MAJOR == 2 && TILEDB_VERSION_MINOR >= 6
    tiledb_query_status_details_t status_details;
    tiledb_query_get_status_details(ctx_.ptr().get(), query_.get()->ptr().get(),
                                    &status_details);

    if (status_details.incomplete_reason == TILEDB_REASON_USER_BUFFER_SIZE) {
#else
    if (true) {
#endif
      auto start_incomplete_buffer_update =
          std::chrono::high_resolution_clock::now();
      for (auto &bp : buffers_) {
        auto &buf = bp.second;

        // Check if values buffer should be resized
        if ((buf.data_vals_read == 0) ||
            (int64_t)(buf.data_vals_read * buf.elem_nbytes) >
                (buf.data.nbytes() + 1) / 2) {
          size_t new_size = buf.data.size() * 2;
          buf.data.resize({new_size}, false);
        }

        // Check if offset buffer should be resized
        if ((buf.isvar && buf.offsets_read == 0) ||
            ((int64_t)(buf.offsets_read * sizeof(uint64_t)) >
             (buf.offsets.nbytes() + 1) / 2)) {
          size_t new_offsets_size = buf.offsets.size() * 2;
          buf.offsets.resize({new_offsets_size}, false);
        }

        // Check if validity buffer should be resized
        if ((buf.isnullable && buf.validity_vals_read == 0) ||
            ((int64_t)(buf.validity_vals_read * sizeof(uint8_t)) >
             (buf.validity.nbytes() + 1) / 2)) {
          size_t new_validity_size = buf.validity.size() * 2;
          buf.validity.resize({new_validity_size}, false);
        }
      }

      // note: this block confuses lldb. continues from here unless bp set after
      // block.
      set_buffers();

      if (g_stats) {
        auto now = std::chrono::high_resolution_clock::now();
        g_stats.get()
            ->counters["py.read_query_incomplete_buffer_resize_time"] +=
            now - start_incomplete_buffer_update;
      }
    }

    {
      py::gil_scoped_release release;
      query_->submit();
    }

    if (query_->query_status() == Query::Status::UNINITIALIZED) {
      TPY_ERROR_LOC("Unexpected state: Query::Submit returned uninitialized");
    }

    update_read_elem_num();

    return;
  }

  void resize_output_buffers() {
    // resize the output buffers to match the final read total
    // the higher level code uses the size of the buffers to
    // determine how much to unpack, but we may have over-allocated

    // account for the extra element at the end of offsets in arrow mode
    size_t arrow_offset_size = use_arrow_ ? 1 : 0;

    for (auto &bp : buffers_) {
      auto name = bp.first;
      auto &buf = bp.second;

      Py_ssize_t final_data_nbytes = buf.data_vals_read * buf.elem_nbytes;
      Py_ssize_t final_offsets_count = buf.offsets_read + arrow_offset_size;
      Py_ssize_t final_validity_count = buf.validity_vals_read;

      assert(final_data_nbytes <= buf.data.size());
      assert(final_offsets_count <=
             (Py_ssize_t)(buf.offsets.size() + arrow_offset_size));

      buf.data.resize({final_data_nbytes});
      buf.offsets.resize({final_offsets_count});
      buf.validity.resize({final_validity_count});

      if (use_arrow_) {
        if (retries_ > 0) {
          // we need to write the final size to the final offset slot
          // because core doesn't track size between incomplete submits
          buf.offsets.mutable_data()[buf.offsets_read] = final_data_nbytes;
        }

        // reset bytes-read so that set_buffers uses the full buffer size
        buf.data_vals_read = 0;
        buf.offsets_read = 0;
        buf.validity_vals_read = 0;
      }
    }
    if (use_arrow_) {
      // this is a very light hack:
      // call set_buffers here to reset the buffers to the *full*
      // buffer in case there were incomplete queries. without this call,
      // the active tiledb::Query only knows about the buffer ptr/size
      // for the *last* submit loop, so we don't get full result set.
      // ArrowAdapter gets the buffer sizes from tiledb::Query.
      set_buffers();
    }
  }

  void allocate_buffers() {

    // allocate buffers for dims
    //   - we want to return dims first, if any requested
    for (size_t dim_idx = 0; dim_idx < domain_->ndim(); dim_idx++) {
      auto dim = domain_->dimension(dim_idx);
      if ((std::find(dims_.begin(), dims_.end(), dim.name()) == dims_.end()) &&
          // we need to also check if this is an attr for backward-compatibility
          (std::find(attrs_.begin(), attrs_.end(), dim.name()) ==
           attrs_.end())) {
        continue;
      }
      alloc_buffer(dim.name());
    }

    // allocate buffers for label dimensions
    for (const auto &label_data : label_input_buffer_data_) {
      alloc_buffer(label_data.first);
    }

    // allocate buffers for attributes
    //   - schema.attributes() is unordered, but we need to return ordered
    //   results
    for (size_t attr_idx = 0; attr_idx < array_schema_->attribute_num();
         attr_idx++) {
      auto attr = array_schema_->attribute(attr_idx);
      if (std::find(attrs_.begin(), attrs_.end(), attr.name()) ==
          attrs_.end()) {
        continue;
      }
      alloc_buffer(attr.name());
    }
  }

  void submit_read() {
    if (retries_ > 0 &&
        query_->query_status() == tiledb::Query::Status::INCOMPLETE) {
      buffers_.clear();
      assert(buffers_.size() == 0);

      buffers_order_.clear();
      // reset_read_elem_num();
    } else if (buffers_.size() != 0) {
      // we have externally imported buffers
      return;
    }

    // start time
    auto start = std::chrono::high_resolution_clock::now();

    // Initiate a metadata API request to make libtiledb fetch the
    // metadata ahead of time. In some queries we know we will always
    // access metadata, so initiating this call saves time when loading
    // from remote arrays because metadata i/o is lazy in core.
    /*
    // This section is disabled pending final disposition of SC-11720
    // This call is not currently safe as of TileDB 2.5, and has caused
    // reproducible deadlocks with the subesequent calls to
    // Array::est_result_sizes from the main thread.
    //
    std::future<uint64_t> metadata_num_preload;
    if (preload_metadata_) {
      metadata_num_preload = std::async(
          std::launch::async, [this]() { return array_->metadata_num(); });
    }
    */

    allocate_buffers();

    // set the buffers on the Query
    set_buffers();

    size_t max_retries = get_max_retries();

    auto start_submit = std::chrono::high_resolution_clock::now();
    {
      py::gil_scoped_release release;
      query_->submit();
    }

    if (query_->query_status() == Query::Status::UNINITIALIZED) {
      TPY_ERROR_LOC("Unexpected state: Query::Submit returned uninitialized");
    }

    if (g_stats) {
      auto now = std::chrono::high_resolution_clock::now();
      g_stats.get()->counters["py.core_read_query_initial_submit_time"] +=
          now - start_submit;
    }

    // update the BufferInfo read-counts to match the query results read
    update_read_elem_num();

    // fetch the result of the metadata get task
    // this will block if not yet completed
    /*
    // disabled, see comment above
    if (preload_metadata_) {
      metadata_num_preload.get();
    }
    */

    auto incomplete_start = std::chrono::high_resolution_clock::now();

    // TODO: would be nice to have a callback here for custom realloc strategy
    while (!return_incomplete_ &&
           query_->query_status() == Query::Status::INCOMPLETE) {
      if (++retries_ > max_retries)
        TPY_ERROR_LOC(
            "Exceeded maximum retries ('py.max_incomplete_retries': '" +
            std::to_string(max_retries) + "')");

      resubmit_read();
    }

    if (g_stats && retries_ > 0) {
      auto now = std::chrono::high_resolution_clock::now();
      g_stats.get()->counters["py.core_read_query_incomplete_retry_time"] +=
          now - incomplete_start;
    }

    // update TileDB-Py stat counter
    if (g_stats) {
      auto now = std::chrono::high_resolution_clock::now();
      g_stats.get()->counters["py.core_read_query_total_time"] += now - start;
    }

    if (g_stats) {
      g_stats.get()->counters["py.query_retries_count"] += TimerType(retries_);
    }

    resize_output_buffers();

    if (return_incomplete_) {
      // increment in case we submit again
      retries_++;
    }
  }

  py::array unpack_buffer(std::string name, py::array buf,
                          py::array_t<uint64_t> off) {

    auto start = std::chrono::high_resolution_clock::now();

    if (off.size() < 1)
      TPY_ERROR_LOC(std::string("Unexpected empty offsets array ('") + name +
                    "')");

    auto dtype = buffer_dtype(name);
    bool is_unicode = dtype.is(py::dtype("U"));
    bool is_str = dtype.is(py::dtype("S"));
    if (is_unicode || is_str) {
      dtype = py::dtype("O");
    }

    // Hashmap for string deduplication
    // fastest so far:
    typedef tsl::robin_map<size_t, uint64_t> MapType;
    MapType map;
    std::vector<py::object> object_v;
    if (is_unicode) {
      map.reserve(size_t(off.size() / 10) + 1);
    }

    auto result_array = py::array(py::dtype("O"), off.size());
    auto result_p = (py::object *)result_array.mutable_data();
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
      auto data_ptr = (char *)buf.data() + last;
      if (is_unicode)
        if (size == 0 || (data_ptr[0] == '\0' && size == 1)) {
          o = py::str("");
        } else {
          if (!deduplicate_) {
            o = py::str(data_ptr, size);
          } else {
            auto v = std::string_view{data_ptr, size};
            auto h = std::hash<std::string_view>()(v);
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
        if (size == 0 || (data_ptr[0] == '\0' && size == 1)) {
          o = py::bytes("");
        } else {
          o = py::bytes(data_ptr, size);
        }
      else {
        o = py::array(py::dtype("uint8"), size, data_ptr);
        o.attr("dtype") = dtype;
      }

      result_p[i - 1] = o;
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
    if (array_->query_type() == TILEDB_READ) {
      submit_read();
    } else if (array_->query_type() == TILEDB_WRITE) {
      submit_write();
    } else if (array_->query_type() == TILEDB_DELETE) {
      query_->submit();
      if (query_->query_status() == Query::Status::UNINITIALIZED) {
        TPY_ERROR_LOC("Unexpected state: Query::Submit returned uninitialized");
      }
    } else {
      TPY_ERROR_LOC("Unknown query type!")
    }
  }

  py::dict results() {
    py::dict results;
    for (auto &buffer_name : buffers_order_) {
      auto bp = buffers_.at(buffer_name);
      results[py::str(buffer_name)] =
          py::make_tuple(bp.data, bp.offsets, bp.validity);
    }
    return results;
  }

  std::unique_ptr<PAPair> buffer_to_pa(std::string name) {
    if (query_->query_status() != tiledb::Query::Status::COMPLETE)
      TPY_ERROR_LOC("Cannot convert buffers unless Query is complete");

#if TILEDB_VERSION_MAJOR == 2 && TILEDB_VERSION_MINOR < 2
    tiledb::arrow::ArrowAdapter adapter(query_);
#else
    tiledb::arrow::ArrowAdapter adapter(&ctx_, query_.get());
#endif

    std::unique_ptr<PAPair> pa_pair(new PAPair());

    adapter.export_buffer(name.c_str(), &(pa_pair->array_), &(pa_pair->schema_),
                          nullptr, nullptr);
    pa_pair->exported_ = true;

    return pa_pair;
  }

  py::object buffers_to_pa_table() {
    using namespace pybind11::literals;

    auto pa = py::module::import("pyarrow");
    auto pa_array_import = pa.attr("Array").attr("_import_from_c");

#if TILEDB_VERSION_MAJOR == 2 && TILEDB_VERSION_MINOR < 2
    tiledb::arrow::ArrowAdapter adapter(query_);
#else
    tiledb::arrow::ArrowAdapter adapter(&ctx_, query_.get());
#endif

    py::list names;
    py::list results;
    for (auto &buffer_name : buffers_order_) {
      BufferInfo &buffer_info = buffers_.at(buffer_name);

      auto buffer_holder = new BufferHolder(
          buffer_info.data, buffer_info.validity, buffer_info.offsets);

      ArrowArray c_pa_array;
      ArrowSchema c_pa_schema;
      adapter.export_buffer(buffer_name.c_str(),
                            static_cast<void *>(&c_pa_array),
                            static_cast<void *>(&c_pa_schema),
                            (tiledb::arrow::ArrowAdapter::release_cb)
                                BufferHolder::free_buffer_holder,
                            buffer_holder);

      if (is_nullable(buffer_name)) {
        // count zeros before converting to bitmap
        c_pa_array.null_count = count_zeros(buffer_info.validity);
        // convert to bitmap
        buffer_info.validity = uint8_bool_to_uint8_bitmap(buffer_info.validity);
        c_pa_array.buffers[0] = buffer_info.validity.data();
        c_pa_array.n_buffers = is_var(buffer_name) ? 3 : 2;
        c_pa_schema.flags |= ARROW_FLAG_NULLABLE;
      } else if (!is_var(buffer_name)) {
        // reset the number of buffers for non-nullable data
        c_pa_array.n_buffers = 2;
      }

      // work around for SC-11522: metadata field must be set to nullptr
      c_pa_schema.metadata = nullptr;
      py::object pa_array = pa_array_import(py::int_((ptrdiff_t)&c_pa_array),
                                            py::int_((ptrdiff_t)&c_pa_schema));

      results.append(pa_array);
      names.append(buffer_name);
    }

    auto pa_table =
        pa.attr("Table").attr("from_arrays")(results, "names"_a = names);
    return pa_table;
  }

  py::object is_incomplete() {
    if (!query_) {
      throw TileDBPyError("Internal error: PyQuery not initialized!");
    }
    return py::cast<bool>(query_->query_status() ==
                          tiledb::Query::Status::INCOMPLETE);
  }

  py::object estimated_result_sizes() {
    // vector of names to estimate
    std::vector<std::string> estim_names;

    for (size_t dim_idx = 0; dim_idx < domain_->ndim(); dim_idx++) {
      auto dim = domain_->dimension(dim_idx);
      if ((std::find(dims_.begin(), dims_.end(), dim.name()) == dims_.end()) &&
          // we need to also check if this is an attr for backward-compatibility
          (std::find(attrs_.begin(), attrs_.end(), dim.name()) ==
           attrs_.end())) {
        continue;
      }
      estim_names.push_back(dim.name());
    }

    // iterate by idx: schema.attributes() is unordered, but we need to
    //                 return ordered results
    for (size_t attr_idx = 0; attr_idx < array_schema_->attribute_num();
         attr_idx++) {
      auto attr = array_schema_->attribute(attr_idx);
      if (std::find(attrs_.begin(), attrs_.end(), attr.name()) ==
          attrs_.end()) {
        continue;
      }
      estim_names.push_back(attr.name());
    }

    py::dict results;

    for (auto const &name : estim_names) {
      size_t est_offsets = 0, est_data_bytes = 0;

      if (is_var(name)) {
        query_->est_result_size_var(name);
        auto est_sizes = query_->est_result_size_var(name);
        est_offsets = std::get<0>(est_sizes);
        est_data_bytes = std::get<1>(est_sizes);
      } else {
        est_data_bytes = query_->est_result_size(name);
      }
      results[py::str(name)] = py::make_tuple(est_offsets, est_data_bytes);
    }

    return std::move(results);
  }

  py::array _test_array() {
    py::array_t<uint8_t> a;
    a.resize({10});

    a.resize({20});
    return std::move(a);
  }

  uint64_t _test_init_buffer_bytes() {
    // test helper to get the configured init_buffer_bytes
    return init_buffer_bytes_;
  }
  uint64_t _test_alloc_max_bytes() {
    // test helper to get the configured init_buffer_bytes
    return alloc_max_bytes_;
  }

  std::string get_stats() { return query_->stats(); }

private:
  void init_config() {
    // get config parameters
    std::string tmp_str;
    if (config_has_key(ctx_.config(), "py.init_buffer_bytes")) {
      tmp_str = ctx_.config().get("py.init_buffer_bytes");
      try {
        init_buffer_bytes_ = std::stoull(tmp_str);
      } catch (const std::invalid_argument &e) {
        (void)e;
        throw std::invalid_argument(
            "Failed to convert 'py.init_buffer_bytes' to uint64_t ('" +
            tmp_str + "')");
      }
    }

    if (config_has_key(ctx_.config(), "py.alloc_max_bytes")) {
      tmp_str = ctx_.config().get("py.alloc_max_bytes");
      try {
        alloc_max_bytes_ = std::stoull(tmp_str);
      } catch (const std::invalid_argument &e) {
        (void)e;
        throw std::invalid_argument(
            "Failed to convert 'py.alloc_max_bytes' to uint64_t ('" + tmp_str +
            "')");
      }
      if (alloc_max_bytes_ < pow(1024, 2)) {
        throw std::invalid_argument("Invalid parameter: 'py.alloc_max_bytes' "
                                    "must be >= 1 MB (1024 ** 2 bytes)");
      };
    }

    if (config_has_key(ctx_.config(), "py.deduplicate")) {
      tmp_str = ctx_.config().get("py.deduplicate");
      if (tmp_str == "true") {
        deduplicate_ = true;
      } else if (tmp_str == "false") {
        deduplicate_ = false;
      } else {
        throw std::invalid_argument(
            "Failed to convert configuration 'py.deduplicate' to bool ('" +
            tmp_str + "')");
      }
    }

    if (config_has_key(ctx_.config(), "py.exact_init_buffer_bytes")) {
      tmp_str = ctx_.config().get("py.exact_init_buffer_bytes");
      if (tmp_str == "true") {
        exact_init_bytes_ = true;
      } else if (tmp_str == "false") {
        exact_init_bytes_ = false;
      } else {
        throw std::invalid_argument("Failed to convert configuration "
                                    "'py.exact_init_buffer_bytes' to bool ('" +
                                    tmp_str + "')");
      }
    }

    if (config_has_key(ctx_.config(), "py.use_arrow")) {
      tmp_str = ctx_.config().get("py.use_arrow");
      if (tmp_str == "True") {
        use_arrow_ = true;
      } else if (tmp_str == "False") {
        use_arrow_ = false;
      } else {
        throw std::invalid_argument(
            "Failed to convert configuration 'py.use_arrow' to bool ('" +
            tmp_str + "')");
      }
    }
  }

}; // namespace tiledbpy

void init_stats() {
  g_stats.reset(new StatsInfo());

  auto stats_counters = g_stats.get()->counters;
  stats_counters["py.core_read_query_initial_submit_time"] = TimerType();
  stats_counters["py.core_read_query_total_time"] = TimerType();
  stats_counters["py.core_read_query_incomplete_retry_time"] = TimerType();
  stats_counters["py.buffer_conversion_time"] = TimerType();
  stats_counters["py.read_query_incomplete_buffer_resize_time"] = TimerType();
  stats_counters["py.query_retries_count"] = TimerType();
}

void disable_stats() { g_stats.reset(nullptr); }

void increment_stat(std::string key, double value) {
  auto &stats_counters = g_stats.get()->counters;

  if (stats_counters.count(key) == 0)
    stats_counters[key] = TimerType();

  auto &timer = stats_counters[key];
  auto incr = std::chrono::duration<double>(value);
  timer += incr;
}

bool use_stats() { return (bool)g_stats; }

py::object get_stats() {
  if (!g_stats) {
    TPY_ERROR_LOC("Stats counters are not uninitialized!")
  }

  auto &stats_counters = g_stats.get()->counters;

  py::dict res;
  for (auto iter = stats_counters.begin(); iter != stats_counters.end();
       ++iter) {
    auto val = std::chrono::duration<double>(iter->second);
    res[py::str(iter->first)] = py::float_(val.count());
  }
  return std::move(res);
}

py::object python_internal_stats(bool dict = false) {
  if (!g_stats) {
    TPY_ERROR_LOC("Stats counters are not uninitialized!")
  }

  auto counters = g_stats.get()->counters;
  auto rq_time = counters["py.core_read_query_initial_submit_time"].count();

  if (dict) {
    py::dict stats;

    // core.cc is only tracking read time right now; don't output if we
    // have no query submission time
    if (rq_time == 0) {
      return std::move(stats);
    }

    for (auto &stat : counters) {
      stats[py::str(stat.first)] = stat.second.count();
    }

    return std::move(stats);
  } else {
    std::ostringstream os;

    // core.cc is only tracking read time right now; don't output if we
    // have no query submission time
    if (rq_time == 0) {
      return py::str(os.str());
    }

    os << std::endl;
    os << "==== Python Stats ====" << std::endl << std::endl;

    for (auto &stat : counters) {
      os << "  " << stat.first << " : " << stat.second.count() << std::endl;
    }

    return py::str(os.str());
  }
}

void init_core(py::module &m) {
  init_query_condition(m);

  auto pq =
      py::class_<PyQuery>(m, "PyQuery")
          .def(py::init<const Context &, py::object, py::iterable, py::iterable,
                        py::object, py::object>())
          .def("add_label_buffer", &PyQuery::add_label_buffer)
          .def("buffer_dtype", &PyQuery::buffer_dtype)
          .def("results", &PyQuery::results)
          .def("set_subarray", &PyQuery::set_subarray)
          .def("set_cond", &PyQuery::set_cond)
#if defined(TILEDB_SERIALIZATION)
          .def("set_serialized_query", &PyQuery::set_serialized_query)
#endif
          .def("submit", &PyQuery::submit)
          .def("unpack_buffer", &PyQuery::unpack_buffer)
          .def("estimated_result_sizes", &PyQuery::estimated_result_sizes)
          .def("get_stats", &PyQuery::get_stats)
          .def("_allocate_buffers", &PyQuery::allocate_buffers)
          .def("_get_buffers", &PyQuery::get_buffers)
          .def("_buffer_to_pa", &PyQuery::buffer_to_pa)
          .def("_buffers_to_pa_table", &PyQuery::buffers_to_pa_table)
          .def("_test_array", &PyQuery::_test_array)
          .def("_test_err",
               [](py::object self, std::string s) { throw TileDBPyError(s); })
          .def_readwrite("_preload_metadata", &PyQuery::preload_metadata_)
          .def_readwrite("_return_incomplete", &PyQuery::return_incomplete_)
          // properties
          .def_property_readonly("is_incomplete", &PyQuery::is_incomplete)
          .def_property_readonly("_test_init_buffer_bytes",
                                 &PyQuery::_test_init_buffer_bytes)
          .def_property_readonly("_test_alloc_max_bytes",
                                 &PyQuery::_test_alloc_max_bytes)
          .def_readonly("retries", &PyQuery::retries_);

  m.def("array_to_buffer", &convert_np);

  m.def("init_stats", &init_stats);
  m.def("disable_stats", &init_stats);
  m.def("python_internal_stats", &python_internal_stats,
        py::arg("dict") = false);
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
};
}; // namespace tiledbpy
