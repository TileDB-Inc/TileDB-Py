#include <tiledb/tiledb>
#include <tiledb/tiledb_experimental>

#include <pybind11/functional.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/pytypes.h>
#include <pybind11/stl.h>

#include "common.h"

namespace libtiledbcpp {

using namespace tiledb;
using namespace tiledbpy::common;
namespace py = pybind11;

void init_vfs(py::module& m) {
    py::class_<VFS>(m, "VFS")
        .def(py::init<const Context&>(), py::keep_alive<1, 2>())
        .def(py::init<const Context&, const Config&>(), py::keep_alive<1, 2>())

        .def_property_readonly("_ctx", &VFS::context)
        .def_property_readonly("_config", &VFS::config)

        .def(
            "_create_bucket",
            &VFS::create_bucket,
            py::call_guard<py::gil_scoped_release>())
        .def(
            "_remove_bucket",
            &VFS::remove_bucket,
            py::call_guard<py::gil_scoped_release>())
        .def(
            "_is_bucket",
            &VFS::is_bucket,
            py::call_guard<py::gil_scoped_release>())
        .def(
            "_empty_bucket",
            &VFS::empty_bucket,
            py::call_guard<py::gil_scoped_release>())
        .def(
            "_is_empty_bucket",
            &VFS::is_empty_bucket,
            py::call_guard<py::gil_scoped_release>())

        .def(
            "_create_dir",
            &VFS::create_dir,
            py::call_guard<py::gil_scoped_release>())
        .def("_is_dir", &VFS::is_dir, py::call_guard<py::gil_scoped_release>())
        .def(
            "_remove_dir",
            &VFS::remove_dir,
            py::call_guard<py::gil_scoped_release>())
        .def(
            "_dir_size",
            &VFS::dir_size,
            py::call_guard<py::gil_scoped_release>())
        .def(
            "_move_dir",
            &VFS::move_dir,
            py::call_guard<py::gil_scoped_release>())
        .def(
            "_copy_dir",
            &VFS::copy_dir,
            py::call_guard<py::gil_scoped_release>())

        .def(
            "_is_file", &VFS::is_file, py::call_guard<py::gil_scoped_release>())
        .def(
            "_remove_file",
            &VFS::remove_file,
            py::call_guard<py::gil_scoped_release>())
        .def(
            "_file_size",
            &VFS::file_size,
            py::call_guard<py::gil_scoped_release>())
        .def(
            "_move_file",
            &VFS::move_file,
            py::call_guard<py::gil_scoped_release>())
        .def(
            "_copy_file",
            &VFS::copy_file,
            py::call_guard<py::gil_scoped_release>())

        .def("_ls", &VFS::ls, py::call_guard<py::gil_scoped_release>())
        .def(
            "_ls_recursive",
            // 1. the user provides a callback, the user callback is passed to
            // the C++ function. Return an empty vector.
            // 2. no callback is passed and a default callback is used. The
            // default callback just appends each path gathered. Return the
            // paths.
            [](VFS& vfs, std::string uri, py::object callback) {
                tiledb::Context ctx;
                if (callback.is_none()) {
                    std::vector<std::string> paths;
                    tiledb::VFSExperimental::ls_recursive(
                        ctx,
                        vfs,
                        uri,
                        [&](const std::string_view path,
                            uint64_t object_size) -> bool {
                            paths.push_back(std::string(path));
                            return true;
                        });
                    return paths;
                } else {
                    tiledb::VFSExperimental::ls_recursive(
                        ctx,
                        vfs,
                        uri,
                        [&](const std::string_view path,
                            uint64_t object_size) -> bool {
                            return callback(path, object_size).cast<bool>();
                        });
                    return std::vector<std::string>();
                }
            })
        .def("_touch", &VFS::touch, py::call_guard<py::gil_scoped_release>());
}

class FileHandle {
   private:
    Context _ctx;
    tiledb_vfs_fh_t* _fh;

   public:
    FileHandle(
        const Context& ctx,
        const VFS& vfs,
        std::string uri,
        tiledb_vfs_mode_t mode)
        : _ctx(ctx) {
        _ctx.handle_error(tiledb_vfs_open(
            _ctx.ptr().get(), vfs.ptr().get(), uri.c_str(), mode, &this->_fh));
    }

    void close() {
        py::gil_scoped_release release;
        _ctx.handle_error(tiledb_vfs_close(_ctx.ptr().get(), this->_fh));
    }

    py::bytes read(uint64_t offset, uint64_t nbytes) {
        py::array data = py::array(py::dtype::of<std::byte>(), nbytes);
        py::buffer_info buffer = data.request();

        {
            py::gil_scoped_release release;
            _ctx.handle_error(tiledb_vfs_read(
                _ctx.ptr().get(), this->_fh, offset, buffer.ptr, nbytes));
        }

        auto np = py::module::import("numpy");
        auto to_bytes = np.attr("ndarray").attr("tobytes");

        return to_bytes(data);
    }

    void write(py::buffer data) {
        py::buffer_info buffer = data.request();
        py::gil_scoped_release release;
        _ctx.handle_error(tiledb_vfs_write(
            _ctx.ptr().get(), this->_fh, buffer.ptr, buffer.shape[0]));
    }

    void flush() {
        py::gil_scoped_release release;
        _ctx.handle_error(tiledb_vfs_sync(_ctx.ptr().get(), this->_fh));
    }

    bool closed() {
        int32_t is_closed;
        _ctx.handle_error(
            tiledb_vfs_fh_is_closed(_ctx.ptr().get(), this->_fh, &is_closed));
        return is_closed;
    }
};

void init_file_handle(py::module& m) {
    py::class_<FileHandle>(m, "FileHandle")
        .def(
            py::init<
                const Context&,
                const VFS&,
                std::string,
                tiledb_vfs_mode_t>(),
            py::keep_alive<1, 2>())

        .def_property_readonly("_closed", &FileHandle::closed)

        .def("_close", &FileHandle::close)
        .def("_read", &FileHandle::read)
        .def("_write", &FileHandle::write)
        .def("_flush", &FileHandle::flush);
}

}  // namespace libtiledbcpp
