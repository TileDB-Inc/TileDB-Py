import os

import tiledb.cc as lt


def test_dir(tmp_path):
    ctx = lt.Context()
    vfs = lt.VFS(ctx)

    path = os.path.join(tmp_path, "test_dir")

    vfs._create_dir(path)
    assert vfs._is_dir(path) is True
    assert vfs._dir_size(path) == 0
    vfs._remove_dir(path)
    assert vfs._is_dir(path) is False


def test_file_handle(tmp_path):
    ctx = lt.Context()
    vfs = lt.VFS(ctx)

    path = os.path.join(tmp_path, "test_file_handle")

    fh = lt.FileHandle(ctx, vfs, path, lt.VFSMode.WRITE)
    fh._write(b"Hello")

    fh = lt.FileHandle(ctx, vfs, path, lt.VFSMode.READ)
    assert fh._read(0, 5) == b"Hello"

    fh = lt.FileHandle(ctx, vfs, path, lt.VFSMode.APPEND)
    fh._write(b", world!")

    fh = lt.FileHandle(ctx, vfs, path, lt.VFSMode.READ)
    assert fh._read(0, 13) == b"Hello, world!"

    assert fh._closed is False
