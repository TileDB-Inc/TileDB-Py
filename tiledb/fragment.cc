
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

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

class PyFragmentInfo {

private:
  Context ctx_;
  unique_ptr<FragmentInfo> fi_;

  py::object schema_;

  uint32_t num_fragments_;
  py::tuple uri_;
  py::tuple version_;
  py::tuple nonempty_domain_;
  py::tuple cell_num_;
  py::tuple timestamp_range_;
  py::tuple sparse_;
  uint32_t unconsolidated_metadata_num_;
  py::tuple has_consolidated_metadata_;
  py::tuple to_vacuum_;
  py::tuple mbrs_;
  py::tuple array_schema_name_;

public:
  tiledb_ctx_t *c_ctx_;

public:
  PyFragmentInfo() = delete;

  PyFragmentInfo(const string &uri, py::object schema, py::bool_ include_mbrs,
                 py::object ctx) {
    schema_ = schema;

    tiledb_ctx_t *c_ctx_ = (py::capsule)ctx.attr("__capsule__")();

    if (c_ctx_ == nullptr)
      TPY_ERROR_LOC("Invalid context pointer!");

    ctx_ = Context(c_ctx_, false);

    fi_ = unique_ptr<tiledb::FragmentInfo>(new FragmentInfo(ctx_, uri));

    load();

    num_fragments_ = fragment_num();
    uri_ = fill_uri();
    version_ = fill_version();
    nonempty_domain_ = fill_non_empty_domain();
    cell_num_ = fill_cell_num();
    timestamp_range_ = fill_timestamp_range();
    sparse_ = fill_sparse();
    unconsolidated_metadata_num_ = unconsolidated_metadata_num();
    has_consolidated_metadata_ = fill_has_consolidated_metadata();
    to_vacuum_ = fill_to_vacuum_uri();
    array_schema_name_ = fill_array_schema_name();

#if TILEDB_VERSION_MAJOR == 2 && TILEDB_VERSION_MINOR >= 5
    if (include_mbrs)
      mbrs_ = fill_mbr();
#endif

    close();
  }

  uint32_t get_num_fragments() { return num_fragments_; };
  py::tuple get_uri() { return uri_; };
  py::tuple get_version() { return version_; };
  py::tuple get_nonempty_domain() { return nonempty_domain_; };
  py::tuple get_cell_num() { return cell_num_; };
  py::tuple get_timestamp_range() { return timestamp_range_; };
  py::tuple get_sparse() { return sparse_; };
  uint32_t get_unconsolidated_metadata_num() {
    return unconsolidated_metadata_num_;
  };
  py::tuple get_has_consolidated_metadata() {
    return has_consolidated_metadata_;
  };
  py::tuple get_to_vacuum() { return to_vacuum_; };
  py::tuple get_mbrs() { return mbrs_; };
  py::tuple get_array_schema_name() { return array_schema_name_; };

  void dump() const { return fi_->dump(stdout); }

private:
  template <typename T>
  py::object for_all_fid(T (FragmentInfo::*fn)(uint32_t) const) const {
    py::list l;
    uint32_t nfrag = fragment_num();

    for (uint32_t i = 0; i < nfrag; ++i)
      l.append((fi_.get()->*fn)(i));
    return py::tuple(l);
  }

  void load() const {
    try {
      fi_->load();
    } catch (TileDBError &e) {
      TPY_ERROR_LOC(e.what());
    }
  }

  void close() { fi_.reset(); }

  py::tuple fill_uri() const {
    return for_all_fid(&FragmentInfo::fragment_uri);
  }

  py::tuple fill_non_empty_domain() const {
    py::list all_frags;
    uint32_t nfrag = fragment_num();

    for (uint32_t fid = 0; fid < nfrag; ++fid)
      all_frags.append(fill_non_empty_domain(fid));

    return std::move(all_frags);
  }

  py::tuple fill_non_empty_domain(uint32_t fid) const {
    py::list all_dims;
    int ndim = (schema_.attr("domain").attr("ndim")).cast<int>();

    for (int did = 0; did < ndim; ++did)
      all_dims.append(fill_non_empty_domain(fid, did));

    return std::move(all_dims);
  }

  template <typename T>
  py::tuple fill_non_empty_domain(uint32_t fid, T did) const {
    py::bool_ isvar = get_dim_isvar(schema_.attr("domain"), did);

    if (isvar) {
      pair<string, string> lims = fi_->non_empty_domain_var(fid, did);
      return py::make_tuple(lims.first, lims.second);
    }

    py::dtype type = get_dim_type(schema_.attr("domain"), did);
    py::dtype array_type =
        type.kind() == 'M' ? pybind11::dtype::of<uint64_t>() : type;

    py::array limits = py::array(array_type, 2);
    py::buffer_info buffer = limits.request();
    fi_->get_non_empty_domain(fid, did, buffer.ptr);

    if (type.kind() == 'M') {
      auto np = py::module::import("numpy");
      auto datetime64 = np.attr("datetime64");
      auto datetime_data = np.attr("datetime_data");

      uint64_t *dates = static_cast<uint64_t *>(buffer.ptr);
      limits = py::make_tuple(datetime64(dates[0], datetime_data(type)),
                              datetime64(dates[1], datetime_data(type)));
    }

    return std::move(limits);
  }

  py::bool_ get_dim_isvar(py::object dom, uint32_t did) const {
    // passing templated type "did" to Python function dom.attr("dim")
    // does not work
    return (dom.attr("dim")(did).attr("isvar")).cast<py::bool_>();
  }

  py::bool_ get_dim_isvar(py::object dom, string did) const {
    // passing templated type "did" to Python function dom.attr("dim")
    // does not work
    return (dom.attr("dim")(did).attr("isvar")).cast<py::bool_>();
  }

  py::dtype get_dim_type(py::object dom, uint32_t did) const {
    // passing templated type "did" to Python function dom.attr("dim")
    // does not work
    return (dom.attr("dim")(did).attr("dtype")).cast<py::dtype>();
  }

  py::dtype get_dim_type(py::object dom, string did) const {
    // passing templated type "did" to Python function dom.attr("dim")
    // does not work
    return (dom.attr("dim")(did).attr("dtype")).cast<py::dtype>();
  }

  py::tuple fill_timestamp_range() const {
    return for_all_fid(&FragmentInfo::timestamp_range);
  }

  uint32_t fragment_num() const { return fi_->fragment_num(); }

  py::tuple fill_sparse() const { return for_all_fid(&FragmentInfo::sparse); }

  py::tuple fill_cell_num() const {
    return for_all_fid(&FragmentInfo::cell_num);
  }

  py::tuple fill_version() const { return for_all_fid(&FragmentInfo::version); }

  py::tuple fill_has_consolidated_metadata() const {
    return for_all_fid(&FragmentInfo::has_consolidated_metadata);
  }

  uint32_t unconsolidated_metadata_num() const {
    return fi_->unconsolidated_metadata_num();
  }

  uint32_t to_vacuum_num() const { return fi_->to_vacuum_num(); }

  py::tuple fill_to_vacuum_uri() const {
    py::list l;
    uint32_t nfrag = to_vacuum_num();

    for (uint32_t i = 0; i < nfrag; ++i)
      l.append((fi_->to_vacuum_uri(i)));
    return py::tuple(l);
  }

#if TILEDB_VERSION_MAJOR == 2 && TILEDB_VERSION_MINOR >= 5
  py::tuple fill_mbr() const {
    py::list all_frags;
    uint32_t nfrag = fragment_num();

    for (uint32_t fid = 0; fid < nfrag; ++fid)
      all_frags.append(fill_mbr(fid));

    return std::move(all_frags);
  }

  py::tuple fill_mbr(uint32_t fid) const {
    py::list all_mbrs;
    uint64_t nmbr = fi_->mbr_num(fid);

    for (uint32_t mid = 0; mid < nmbr; ++mid)
      all_mbrs.append(fill_mbr(fid, mid));

    return std::move(all_mbrs);
  }

  py::tuple fill_mbr(uint32_t fid, uint32_t mid) const {
    py::list all_dims;
    int ndim = (schema_.attr("domain").attr("ndim")).cast<int>();

    for (int did = 0; did < ndim; ++did)
      all_dims.append(fill_mbr(fid, mid, did));

    return std::move(all_dims);
  }

  py::tuple fill_mbr(uint32_t fid, uint32_t mid, uint32_t did) const {
    py::bool_ isvar = get_dim_isvar(schema_.attr("domain"), did);

    if (isvar) {
      auto limits = fi_->mbr_var(fid, mid, did);
      return py::make_tuple(limits.first, limits.second);
    }

    py::dtype type = get_dim_type(schema_.attr("domain"), did);
    py::dtype array_type =
        type.kind() == 'M' ? pybind11::dtype::of<uint64_t>() : type;
    py::array limits = py::array(array_type, 2);
    py::buffer_info buffer = limits.request();
    fi_->get_mbr(fid, mid, did, buffer.ptr);
    return std::move(limits);
  }

  py::tuple fill_array_schema_name() const {
    return for_all_fid(&FragmentInfo::array_schema_name);
  }

#endif
};

void init_fragment(py::module &m) {
  py::class_<PyFragmentInfo>(m, "PyFragmentInfo")
      .def(py::init<const string &, py::object, py::bool_, py::object>())
      .def("get_num_fragments", &PyFragmentInfo::get_num_fragments)
      .def("get_uri", &PyFragmentInfo::get_uri)
      .def("get_version", &PyFragmentInfo::get_version)
      .def("get_nonempty_domain", &PyFragmentInfo::get_nonempty_domain)
      .def("get_cell_num", &PyFragmentInfo::get_cell_num)
      .def("get_timestamp_range", &PyFragmentInfo::get_timestamp_range)
      .def("get_sparse", &PyFragmentInfo::get_sparse)
      .def("get_unconsolidated_metadata_num",
           &PyFragmentInfo::get_unconsolidated_metadata_num)
      .def("get_has_consolidated_metadata",
           &PyFragmentInfo::get_has_consolidated_metadata)
      .def("get_to_vacuum", &PyFragmentInfo::get_to_vacuum)
#if TILEDB_VERSION_MAJOR == 2 && TILEDB_VERSION_MINOR >= 5
      .def("get_mbrs", &PyFragmentInfo::get_mbrs)
      .def("get_array_schema_name", &PyFragmentInfo::get_array_schema_name)
#endif
      .def("dump", &PyFragmentInfo::dump);
}

}; // namespace tiledbpy

#endif
