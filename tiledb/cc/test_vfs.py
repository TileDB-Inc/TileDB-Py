import numpy as np
import tiledb
import hypothesis
import time
import tempfile
import os

from tiledb import cc as lt
from tiledb.tests.common import paths_equal

import pytest


def test_dir(tmp_path):
    ctx = lt.Context()
    vfs = lt.VFS(ctx)

    path = os.path.join(tmp_path, "test_dir")

    vfs.create_dir(path)
    assert vfs.is_dir(path) == True
    assert vfs.dir_size(path) == 0
    vfs.remove_dir(path)
    assert vfs.is_dir(path) == False


def test_file_handle(tmp_path):
    ctx = lt.Context()
    vfs = lt.VFS(ctx)

    path = os.path.join(tmp_path, "test_file_handle")

    fh = lt.FileHandle(ctx, vfs, path, lt.VFSMode.WRITE)
    fh.write(b"Hello")

    fh = lt.FileHandle(ctx, vfs, path, lt.VFSMode.READ)
    assert fh.read(0, 5) == b"Hello"

    fh = lt.FileHandle(ctx, vfs, path, lt.VFSMode.APPEND)
    fh.write(b", world!")

    fh = lt.FileHandle(ctx, vfs, path, lt.VFSMode.READ)
    assert fh.read(0, 13) == b"Hello, world!"

    assert fh.closed == False
