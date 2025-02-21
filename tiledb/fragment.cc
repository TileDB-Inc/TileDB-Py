
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
// #include <pybind11/pytypes.h>
// #include <pybind11/stl.h>

#include <exception>

#include <tiledb/tiledb>  // C++
#include "util.h"

#if !defined(NDEBUG)
// #include "debug.cc"
#endif

namespace tiledbpy {

using namespace std;
using namespace tiledb;
namespace nb = nanobind;
using namespace nb::literals;

class PyFragmentInfo {
   private:
    Context ctx_;
    unique_ptr<FragmentInfo> fi_;

    nb::object schema_;

    uint32_t num_fragments_;
    nb::tuple uri_;
    nb::tuple version_;
    nb::tuple nonempty_domain_;
    nb::tuple cell_num_;
    nb::tuple timestamp_range_;
    nb::tuple sparse_;
    uint32_t unconsolidated_metadata_num_;
    nb::tuple has_consolidated_metadata_;
    nb::tuple to_vacuum_;
    nb::tuple mbrs_;
    nb::tuple array_schema_name_;

   public:
    tiledb_ctx_t* c_ctx_;

   public:
    PyFragmentInfo() = delete;

    PyFragmentInfo(
        const string& uri,
        nb::object schema,
        nb::bool_ include_mbrs,
        nb::object ctx) {
        schema_ = schema;

        // tiledb_ctx_t* c_ctx_ = (nb::capsule)ctx.attr("__capsule__")();
        tiledb_ctx_t* c_ctx_ = nb::cast<tiledb_ctx_t*>(
            ctx.attr("__capsule__")());

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

        if (include_mbrs)
            mbrs_ = fill_mbr();

        close();
    }

    uint32_t get_num_fragments() {
        return num_fragments_;
    };
    nb::tuple get_uri() {
        return uri_;
    };
    nb::tuple get_version() {
        return version_;
    };
    nb::tuple get_nonempty_domain() {
        return nonempty_domain_;
    };
    nb::tuple get_cell_num() {
        return cell_num_;
    };
    nb::tuple get_timestamp_range() {
        return timestamp_range_;
    };
    nb::tuple get_sparse() {
        return sparse_;
    };
    uint32_t get_unconsolidated_metadata_num() {
        return unconsolidated_metadata_num_;
    };
    nb::tuple get_has_consolidated_metadata() {
        return has_consolidated_metadata_;
    };
    nb::tuple get_to_vacuum() {
        return to_vacuum_;
    };
    nb::tuple get_mbrs() {
        return mbrs_;
    };
    nb::tuple get_array_schema_name() {
        return array_schema_name_;
    };

    void dump() const {
        return fi_->dump(stdout);
    }

   private:
    template <typename T>
    nb::tuple for_all_fid(T (FragmentInfo::*fn)(uint32_t) const) const {
        nb::list l;
        uint32_t nfrag = fragment_num();

        for (uint32_t i = 0; i < nfrag; ++i)
            l.append((fi_.get()->*fn)(i));
        return nb::tuple(l);
    }

    void load() const {
        try {
            fi_->load();
        } catch (TileDBError& e) {
            TPY_ERROR_LOC(e.what());
        }
    }

    void close() {
        fi_.reset();
    }

    nb::tuple fill_uri() const {
        return for_all_fid(&FragmentInfo::fragment_uri);
    }

    nb::tuple fill_non_empty_domain() const {
        nb::list all_frags;
        uint32_t nfrag = fragment_num();

        for (uint32_t fid = 0; fid < nfrag; ++fid)
            all_frags.append(fill_non_empty_domain(fid));

        return nb::tuple(all_frags);  // Explicit conversion to tuple
    }

    nb::tuple fill_non_empty_domain(uint32_t fid) const {
        nb::list all_dims;
        // int ndim = (schema_.attr("domain").attr("ndim")).cast<int>();
        int ndim = nb::cast<int>(schema_.attr("domain").attr("ndim"));

        for (int did = 0; did < ndim; ++did)
            all_dims.append(fill_non_empty_domain(fid, did));

        return nb::tuple(all_dims);
    }

    template <typename T>
    nb::tuple fill_non_empty_domain(uint32_t fid, T did) const {
        nb::bool_ isvar = get_dim_isvar(schema_.attr("domain"), did);

        if (isvar) {
            pair<string, string> lims = fi_->non_empty_domain_var(fid, did);
            return nb::make_tuple(lims.first, lims.second);
        }

        nb::dlpack::dtype type = get_dim_type(schema_.attr("domain"), did);
        nb::dlpack::dtype array_type = type.kind() == 'M' ?
                                           nb::dtype<uint64_t>() :
                                           type;

        nb::ndarray limits = nb::ndarray(array_type, 2);
        nb::buffer_info buffer = limits.request();
        fi_->get_non_empty_domain(fid, did, buffer.ptr);

        if (type.kind() == 'M') {
            auto np = nb::module_::import_("numpy");
            auto datetime64 = np.attr("datetime64");
            auto datetime_data = np.attr("datetime_data");

            uint64_t* dates = static_cast<uint64_t*>(buffer.ptr);
            limits = nb::make_tuple(
                datetime64(dates[0], datetime_data(type)),
                datetime64(dates[1], datetime_data(type)));
        }

        return std::move(limits);
    }

    nb::bool_ get_dim_isvar(nb::object dom, uint32_t did) const {
        // passing templated type "did" to Python function dom.attr("dim")
        // does not work
        return nb::cast<nb::bool_>(dom.attr("dim")(did).attr("isvar"));
    }

    nb::bool_ get_dim_isvar(nb::object dom, string did) const {
        // passing templated type "did" to Python function dom.attr("dim")
        // does not work
        return nb::cast<nb::bool_>(dom.attr("dim")(did).attr("isvar"));
    }

    nb::dlpack::dtype get_dim_type(nb::object dom, uint32_t did) const {
        // passing templated type "did" to Python function dom.attr("dim")
        // does not work
        return nb::cast<nb::dtype>(dom.attr("dim")(did).attr("dtype"));
    }

    nb::dlpack::dtype get_dim_type(nb::object dom, string did) const {
        // passing templated type "did" to Python function dom.attr("dim")
        // does not work
        return nb::cast<nb::dtype>(dom.attr("dim")(did).attr("dtype"));
    }

    nb::tuple fill_timestamp_range() const {
        return for_all_fid(&FragmentInfo::timestamp_range);
    }

    uint32_t fragment_num() const {
        return fi_->fragment_num();
    }

    nb::tuple fill_sparse() const {
        return for_all_fid(&FragmentInfo::sparse);
    }

    nb::tuple fill_cell_num() const {
        return for_all_fid(&FragmentInfo::cell_num);
    }

    nb::tuple fill_version() const {
        return for_all_fid(&FragmentInfo::version);
    }

    nb::tuple fill_has_consolidated_metadata() const {
        return for_all_fid(&FragmentInfo::has_consolidated_metadata);
    }

    uint32_t unconsolidated_metadata_num() const {
        return fi_->unconsolidated_metadata_num();
    }

    uint32_t to_vacuum_num() const {
        return fi_->to_vacuum_num();
    }

    nb::tuple fill_to_vacuum_uri() const {
        nb::list l;
        uint32_t nfrag = to_vacuum_num();

        for (uint32_t i = 0; i < nfrag; ++i)
            l.append((fi_->to_vacuum_uri(i)));
        return nb::tuple(l);
    }

    nb::tuple fill_mbr() const {
        nb::list all_frags;
        uint32_t nfrag = fragment_num();

        for (uint32_t fid = 0; fid < nfrag; ++fid)
            all_frags.append(fill_mbr(fid));

        return nb::tuple(all_frags);
    }

    nb::tuple fill_mbr(uint32_t fid) const {
        nb::list all_mbrs;
        uint64_t nmbr = fi_->mbr_num(fid);

        for (uint32_t mid = 0; mid < nmbr; ++mid)
            all_mbrs.append(fill_mbr(fid, mid));

        return nb::tuple(all_mbrs);
    }

    nb::tuple fill_mbr(uint32_t fid, uint32_t mid) const {
        nb::list all_dims;
        // int ndim = (schema_.attr("domain").attr("ndim")).cast<int>();
        int ndim = nb::cast<int>(schema_.attr("domain").attr("ndim"));

        for (int did = 0; did < ndim; ++did)
            all_dims.append(fill_mbr(fid, mid, did));

        return nb::tuple(all_dims);
    }

    nb::tuple fill_mbr(uint32_t fid, uint32_t mid, uint32_t did) const {
        nb::bool_ isvar = get_dim_isvar(schema_.attr("domain"), did);

        if (isvar) {
            auto limits = fi_->mbr_var(fid, mid, did);
            return nb::make_tuple(limits.first, limits.second);
        }

        nb::dlpack::dtype type = get_dim_type(schema_.attr("domain"), did);
        nb::dlpack::dtype array_type = type.kind() == 'M' ?
                                           nb::dtype<uint64_t>() :
                                           type;
        nb::ndarray limits = nb::ndarray(array_type, 2);
        nb::buffer_info buffer = limits.request();
        fi_->get_mbr(fid, mid, did, buffer.ptr);
        return std::move(limits);
    }

    nb::tuple fill_array_schema_name() const {
        return for_all_fid(&FragmentInfo::array_schema_name);
    }
};

void init_fragment(nb::module_& m) {
    nb::class_<PyFragmentInfo>(m, "PyFragmentInfo")
        .def(nb::init<const string&, nb::object, nb::bool_, nb::object>())
        .def("get_num_fragments", &PyFragmentInfo::get_num_fragments)
        .def("get_uri", &PyFragmentInfo::get_uri)
        .def("get_version", &PyFragmentInfo::get_version)
        .def("get_nonempty_domain", &PyFragmentInfo::get_nonempty_domain)
        .def("get_cell_num", &PyFragmentInfo::get_cell_num)
        .def("get_timestamp_range", &PyFragmentInfo::get_timestamp_range)
        .def("get_sparse", &PyFragmentInfo::get_sparse)
        .def(
            "get_unconsolidated_metadata_num",
            &PyFragmentInfo::get_unconsolidated_metadata_num)
        .def(
            "get_has_consolidated_metadata",
            &PyFragmentInfo::get_has_consolidated_metadata)
        .def("get_to_vacuum", &PyFragmentInfo::get_to_vacuum)
        .def("get_mbrs", &PyFragmentInfo::get_mbrs)
        .def("get_array_schema_name", &PyFragmentInfo::get_array_schema_name)
        .def("dump", &PyFragmentInfo::dump);
}

};  // namespace tiledbpy
