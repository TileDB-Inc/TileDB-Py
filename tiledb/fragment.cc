#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

#include <exception>

#define TILEDB_DEPRECATED
#define TILEDB_DEPRECATED_EXPORT

#include <tiledb/tiledb> // C++
#include "util.h"

#if !defined(NDEBUG)
#include "debug.cc"
#endif

namespace tiledbpy {

using namespace std;
using namespace tiledb;
namespace py = pybind11;
using namespace pybind11::literals;

class PyFragmentInfo {

private:
    Context ctx_;
    shared_ptr<tiledb::FragmentInfo> fi_;

public:
    tiledb_ctx_t *c_ctx_;

public:
    PyFragmentInfo() = delete;

    PyFragmentInfo(py::object ctx, const string& uri) {
        tiledb_ctx_t *c_ctx_ = (py::capsule)ctx.attr("__capsule__")();
        if (c_ctx_ == nullptr)
            TPY_ERROR_LOC("Invalid context pointer!");
        ctx_ = Context(c_ctx_, false);

        fi_ = shared_ptr<tiledb::FragmentInfo>(new FragmentInfo(ctx_, uri));
    }

    void load() const {
        try{
            fi_->load();
        } catch(TileDBError& e) {
            TPY_ERROR_LOC(e.what());
        }
    }

    void load(tiledb_encryption_type_t encryption_type,
              const string& encryption_key) const {
        try{
            fi_->load(encryption_type, encryption_key);
        } catch(TileDBError& e) {
            TPY_ERROR_LOC(e.what());
        }
    }

    string fragment_uri(uint32_t fid) const {
        return fi_->fragment_uri(fid);
    }

    template <typename T>
    py::tuple get_non_empty_domain(uint32_t fid, T did, py::dtype type) const {
        py::dtype array_type = type.kind() == 'M' ? pybind11::dtype::of<uint64_t>() : type;
        py::array domain = py::array(array_type, 2);
        py::buffer_info buffer = domain.request();
        fi_->get_non_empty_domain(fid, did, buffer.ptr);

        if(type.kind() == 'M'){
            auto np = py::module::import("numpy");
            auto datetime64 = np.attr("datetime64");
            auto datetime_data = np.attr("datetime_data");

            uint64_t* dates = static_cast<uint64_t*>(buffer.ptr);
            domain = py::make_tuple(datetime64(dates[0], datetime_data(type)), 
                                    datetime64(dates[1], datetime_data(type)));
        }

        return domain;
    }

    template <typename T>
    pair<string, string> non_empty_domain_var(uint32_t fid, T did) const {
        return fi_->non_empty_domain_var(fid, did);
    }

    pair<uint64_t, uint64_t> timestamp_range(uint32_t fid) const {
        return fi_->timestamp_range(fid);
    }

    uint32_t fragment_num() const {
        return fi_->fragment_num();
    }

    bool dense(uint32_t fid) const {
        return fi_->dense(fid);
    }

    bool sparse(uint32_t fid) const {
        return fi_->sparse(fid);
    }

    uint64_t cell_num(uint32_t fid) const {
        return fi_->cell_num(fid);
    }

    uint32_t version(uint32_t fid) const {
        return fi_->version(fid);
    }

    bool has_consolidated_metadata(uint32_t fid) const {
        return fi_->has_consolidated_metadata(fid);
    }

    uint32_t unconsolidated_metadata_num() const {
        return fi_->unconsolidated_metadata_num();
    }
    
    uint32_t to_vacuum_num() const {
        return fi_->to_vacuum_num();
    }

    string to_vacuum_uri(uint32_t fid) const {
        return fi_->to_vacuum_uri(fid);
    }

    void dump(FILE* out = nullptr) const {
        return fi_->dump(out);
    }
};

PYBIND11_MODULE(fragment, m) {
    py::class_<PyFragmentInfo>(m, "info")
        .def(py::init<py::object, const string&>())
        .def("load", static_cast<void (PyFragmentInfo::*)() const>(&PyFragmentInfo::load))
        .def("load", 
            static_cast<void (PyFragmentInfo::*)(
                tiledb_encryption_type_t, const string&) const>
            (&PyFragmentInfo::load))
        .def("get_non_empty_domain", 
            static_cast<py::tuple (PyFragmentInfo::*)(uint32_t, uint32_t, py::dtype) const>
                (&PyFragmentInfo::get_non_empty_domain))
        .def("get_non_empty_domain", 
            static_cast<py::tuple (PyFragmentInfo::*)(uint32_t, const string&, py::dtype) const>
                (&PyFragmentInfo::get_non_empty_domain))
        .def("non_empty_domain_var", 
            static_cast<pair<string, string> (PyFragmentInfo::*)(
                    uint32_t, uint32_t) const>
                (&PyFragmentInfo::non_empty_domain_var))
        .def("non_empty_domain_var", 
            static_cast<pair<string, string> (PyFragmentInfo::*)(
                uint32_t, const string&) const>
            (&PyFragmentInfo::non_empty_domain_var))
        .def("fragment_uri", &PyFragmentInfo::fragment_uri)
        .def("timestamp_range", &PyFragmentInfo::timestamp_range)
        .def("fragment_num", &PyFragmentInfo::fragment_num)
        .def("dense", &PyFragmentInfo::dense)
        .def("sparse", &PyFragmentInfo::sparse)
        .def("cell_num", &PyFragmentInfo::cell_num)
        .def("version", &PyFragmentInfo::version)
        .def("has_consolidated_metadata", &PyFragmentInfo::has_consolidated_metadata)
        .def("unconsolidated_metadata_num", &PyFragmentInfo::unconsolidated_metadata_num)
        .def("to_vacuum_num", &PyFragmentInfo::to_vacuum_num)
        .def("to_vacuum_uri", &PyFragmentInfo::to_vacuum_uri)
        .def("dump", &PyFragmentInfo::dump);

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
        } catch (std::runtime_error &e) {
            std::cout << "unexpected runtime_error: " << e.what() << std::endl;
        }
    });
}

}; // namespace tiledbpy
