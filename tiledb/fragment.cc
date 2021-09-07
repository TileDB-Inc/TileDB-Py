
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include <exception>

#define TILEDB_DEPRECATED
#define TILEDB_DEPRECATED_EXPORT

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

class PyFragmentInfo {

private:
  Context ctx_;
  shared_ptr<FragmentInfo> fi_;

public:
  tiledb_ctx_t *c_ctx_;

public:
  PyFragmentInfo() = delete;

  PyFragmentInfo(const string &uri, py::object ctx) {
    if (ctx.is(py::none())) {
      auto tiledblib = py::module::import("tiledb");
      auto default_ctx = tiledblib.attr("default_ctx");
      ctx = default_ctx();
    }

    tiledb_ctx_t *c_ctx_ = (py::capsule)ctx.attr("__capsule__")();

    if (c_ctx_ == nullptr)
      TPY_ERROR_LOC("Invalid context pointer!");

    ctx_ = Context(c_ctx_, false);

    fi_ = shared_ptr<tiledb::FragmentInfo>(new FragmentInfo(ctx_, uri));
  }

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

  void load(tiledb_encryption_type_t encryption_type,
            const string &encryption_key) const {
    try {
      fi_->load(encryption_type, encryption_key);
    } catch (TileDBError &e) {
      TPY_ERROR_LOC(e.what());
    }
  }

  py::object fragment_uri(py::object fid) const {
    return fid.is(py::none())
               ? for_all_fid(&FragmentInfo::fragment_uri)
               : py::str(fi_->fragment_uri(py::cast<uint32_t>(fid)));
  }

  py::tuple get_non_empty_domain(py::object schema) const {
    py::list all_frags;
    uint32_t nfrag = fragment_num();

    for (uint32_t fid = 0; fid < nfrag; ++fid)
      all_frags.append(get_non_empty_domain(schema, fid));

    return std::move(all_frags);
  }

  py::tuple get_non_empty_domain(py::object schema, uint32_t fid) const {
    py::list all_dims;
    int ndim = (schema.attr("domain").attr("ndim")).cast<int>();

    for (int did = 0; did < ndim; ++did)
      all_dims.append(get_non_empty_domain(schema, fid, did));

    return std::move(all_dims);
  }

  template <typename T>
  py::tuple get_non_empty_domain(py::object schema, uint32_t fid, T did) const {
    py::bool_ isvar = get_dim_isvar(schema.attr("domain"), did);

    if (isvar) {
      pair<string, string> lims = fi_->non_empty_domain_var(fid, did);
      return py::make_tuple(lims.first, lims.second);
    }

    py::dtype type = get_dim_type(schema.attr("domain"), did);
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

  py::object timestamp_range(py::object fid) const {
    if (fid.is(py::none()))
      return for_all_fid(&FragmentInfo::timestamp_range);

    auto p = fi_->timestamp_range(py::cast<uint32_t>(fid));
    return py::make_tuple(p.first, p.second);
  }

  uint32_t fragment_num() const { return fi_->fragment_num(); }

  py::object dense(py::object fid) const {
    return fid.is(py::none()) ? for_all_fid(&FragmentInfo::dense)
                              : py::bool_(fi_->dense(py::cast<uint32_t>(fid)));
  }

  py::object sparse(py::object fid) const {
    return fid.is(py::none()) ? for_all_fid(&FragmentInfo::sparse)
                              : py::bool_(fi_->sparse(py::cast<uint32_t>(fid)));
  }

  py::object cell_num(py::object fid) const {
    return fid.is(py::none())
               ? for_all_fid(&FragmentInfo::cell_num)
               : py::int_(fi_->cell_num(py::cast<uint32_t>(fid)));
  }

  py::object version(py::object fid) const {
    return fid.is(py::none()) ? for_all_fid(&FragmentInfo::version)
                              : py::int_(fi_->version(py::cast<uint32_t>(fid)));
  }

  py::object has_consolidated_metadata(py::object fid) const {
    return fid.is(py::none())
               ? for_all_fid(&FragmentInfo::has_consolidated_metadata)
               : py::bool_(
                     fi_->has_consolidated_metadata(py::cast<uint32_t>(fid)));
  }

  uint32_t unconsolidated_metadata_num() const {
    return fi_->unconsolidated_metadata_num();
  }

  uint32_t to_vacuum_num() const { return fi_->to_vacuum_num(); }

  py::object to_vacuum_uri(py::object fid) const {
    if (fid.is(py::none())) {
      py::list l;
      uint32_t nfrag = to_vacuum_num();

      for (uint32_t i = 0; i < nfrag; ++i)
        l.append((fi_->to_vacuum_uri(i)));
      return py::tuple(l);
    }

    return py::str(fi_->to_vacuum_uri(py::cast<uint32_t>(fid)));
  }

  void dump() const { return fi_->dump(stdout); }
};

void init_fragment(py::module &m) {
  py::class_<PyFragmentInfo>(m, "PyFragmentInfo")
      .def(py::init<const string &, py::object>(), py::arg("uri"),
           py::arg("ctx") = py::none())

      .def("load",
           static_cast<void (PyFragmentInfo::*)() const>(&PyFragmentInfo::load))
      .def("load", static_cast<void (PyFragmentInfo::*)(
                       tiledb_encryption_type_t, const string &) const>(
                       &PyFragmentInfo::load))

      .def("get_non_empty_domain",
           static_cast<py::tuple (PyFragmentInfo::*)(py::object) const>(
               &PyFragmentInfo::get_non_empty_domain))
      .def("get_non_empty_domain",
           static_cast<py::tuple (PyFragmentInfo::*)(py::object, uint32_t)
                           const>(&PyFragmentInfo::get_non_empty_domain))
      .def("get_non_empty_domain", static_cast<py::tuple (PyFragmentInfo::*)(
                                       py::object, uint32_t, uint32_t) const>(
                                       &PyFragmentInfo::get_non_empty_domain))
      .def("get_non_empty_domain",
           static_cast<py::tuple (PyFragmentInfo::*)(py::object, uint32_t,
                                                     const string &) const>(
               &PyFragmentInfo::get_non_empty_domain))

      .def("fragment_uri", &PyFragmentInfo::fragment_uri,
           py::arg("fid") = py::none())
      .def("timestamp_range", &PyFragmentInfo::timestamp_range,
           py::arg("fid") = py::none())
      .def("fragment_num", &PyFragmentInfo::fragment_num)
      .def("dense", &PyFragmentInfo::dense, py::arg("fid") = py::none())
      .def("sparse", &PyFragmentInfo::sparse, py::arg("fid") = py::none())
      .def("cell_num", &PyFragmentInfo::cell_num, py::arg("fid") = py::none())
      .def("version", &PyFragmentInfo::version, py::arg("fid") = py::none())
      .def("has_consolidated_metadata",
           &PyFragmentInfo::has_consolidated_metadata,
           py::arg("fid") = py::none())
      .def("unconsolidated_metadata_num",
           &PyFragmentInfo::unconsolidated_metadata_num)
      .def("to_vacuum_num", &PyFragmentInfo::to_vacuum_num)
      .def("to_vacuum_uri", &PyFragmentInfo::to_vacuum_uri,
           py::arg("fid") = py::none())
      .def("dump", &PyFragmentInfo::dump);
}

}; // namespace tiledbpy

#endif
