import io
import numpy as np
import os
import pytest
import random
import sys

import tiledb
from tiledb.tests.common import DiskTestCase, rand_utf8


class TestVFS(DiskTestCase):
    def test_supports(self):
        vfs = tiledb.VFS()

        self.assertTrue(vfs.supports("file"))
        self.assertIsInstance(vfs.supports("s3"), bool)
        self.assertIsInstance(vfs.supports("hdfs"), bool)
        self.assertIsInstance(vfs.supports("gcs"), bool)
        self.assertIsInstance(vfs.supports("azure"), bool)

        with self.assertRaises(ValueError):
            vfs.supports("invalid")

    def test_vfs_config(self):
        opt = {"region": "us-west-x1234"}
        params = [opt, tiledb.Config(opt)]
        for param in params:
            vfs = tiledb.VFS(param)
            assert vfs.config()["region"] == opt["region"]

    def test_dir(self):
        vfs = tiledb.VFS()

        dir = self.path("foo")
        self.assertFalse(vfs.is_dir(dir))

        # create
        vfs.create_dir(dir)
        if pytest.tiledb_vfs != "s3":
            self.assertTrue(vfs.is_dir(dir))

        # remove
        vfs.remove_dir(dir)
        self.assertFalse(vfs.is_dir(dir))

        # create nested path
        dir = self.path("foo/bar")
        if pytest.tiledb_vfs != "s3":
            # this fails locally because "foo" base path does not exist
            # this will not fail on s3 because there is no concept of directory
            with self.assertRaises(tiledb.TileDBError):
                vfs.create_dir(dir)

        vfs.create_dir(self.path("foo"))
        vfs.create_dir(self.path("foo/bar"))
        if pytest.tiledb_vfs != "s3":
            self.assertTrue(vfs.is_dir(dir))

    def test_file(self):
        vfs = tiledb.VFS()

        file = self.path("foo")
        self.assertFalse(vfs.is_file(file))

        # create
        vfs.touch(file)
        self.assertTrue(vfs.is_file(file))

        # remove
        vfs.remove_file(file)
        self.assertFalse(vfs.is_file(file))

        # check nested path
        file = self.path("foo/bar")
        if pytest.tiledb_vfs != "s3":
            # this fails locally because "foo" base path does not exist
            # this will not fail on s3 because there is no concept of directory
            with self.assertRaises(tiledb.TileDBError):
                vfs.touch(file)

    def test_move(self):
        vfs = tiledb.VFS()

        vfs.create_dir(self.path("foo"))
        vfs.create_dir(self.path("bar"))
        vfs.touch(self.path("bar/baz"))

        self.assertTrue(vfs.is_file(self.path("bar/baz")))

        vfs.move_file(self.path("bar/baz"), self.path("foo/baz"))

        self.assertFalse(vfs.is_file(self.path("bar/baz")))
        self.assertTrue(vfs.is_file(self.path("foo/baz")))

        # moving to invalid dir should raise an error
        if pytest.tiledb_vfs != "s3":
            # this fails locally because "foo" base path does not exist
            # this will not fail on s3 because there is no concept of directory
            with self.assertRaises(tiledb.TileDBError):
                vfs.move_dir(self.path("foo/baz"), self.path("do_not_exist/baz"))

    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="VFS copy commands from core are not supported on Windows",
    )
    def test_copy(self):
        vfs = tiledb.VFS()

        vfs.create_dir(self.path("foo"))
        vfs.create_dir(self.path("bar"))
        vfs.touch(self.path("foo/baz"))

        self.assertTrue(vfs.is_file(self.path("foo/baz")))

        vfs.copy_file(self.path("foo/baz"), self.path("bar/baz"))

        self.assertTrue(vfs.is_file(self.path("foo/baz")))
        self.assertTrue(vfs.is_file(self.path("bar/baz")))

        vfs.copy_dir(self.path("foo"), self.path("baz"))

        self.assertTrue(vfs.is_file(self.path("baz/baz")))

        # copying to invalid dir should raise an error
        if pytest.tiledb_vfs != "s3":
            # this fails locally because "foo" base path does not exist
            # this will not fail on s3 because there is no concept of directory
            with self.assertRaises(tiledb.TileDBError):
                vfs.copy_dir(self.path("foo/baz"), self.path("do_not_exist/baz"))

    def test_write_read(self):
        vfs = tiledb.VFS()

        buffer = b"bar"
        fio = vfs.open(self.path("foo"), "wb")
        fio.write(buffer)
        self.assertEqual(vfs.file_size(self.path("foo")), 3)

        fio = vfs.open(self.path("foo"), "rb")
        self.assertEqual(fio.read(3), buffer)
        fio.close()

        buffer = b"abc"
        fio = vfs.open(self.path("abc"), "wb")
        with pytest.warns(DeprecationWarning, match="Use `FileIO.write`"):
            vfs.write(fio, buffer)
        with pytest.warns(DeprecationWarning, match="Use `FileIO.close`"):
            vfs.close(fio)
        self.assertEqual(vfs.file_size(self.path("abc")), 3)

        fio = vfs.open(self.path("abc"), "rb")
        with pytest.warns(
            DeprecationWarning, match="Use `FileIO.seek` and `FileIO.read`"
        ):
            self.assertEqual(vfs.read(fio, 0, 3), buffer)
        fio.close()

        # write / read empty input
        fio = vfs.open(self.path("baz"), "wb")
        fio.write(b"")
        fio.close()
        self.assertEqual(vfs.file_size(self.path("baz")), 0)

        fio = vfs.open(self.path("baz"), "rb")
        self.assertEqual(fio.read(0), b"")
        fio.close()

        # read from file that does not exist
        with self.assertRaises(tiledb.TileDBError):
            vfs.open(self.path("do_not_exist"), "rb")

    def test_io(self):
        vfs = tiledb.VFS()

        buffer = b"0123456789"
        with tiledb.FileIO(vfs, self.path("foo"), mode="wb") as fio:
            fio.write(buffer)
            fio.flush()
            self.assertEqual(fio.tell(), len(buffer))

        with tiledb.FileIO(vfs, self.path("foo"), mode="rb") as fio:
            with self.assertRaises(IOError):
                fio.write(b"foo")

        self.assertEqual(vfs.file_size(self.path("foo")), len(buffer))

        fio = tiledb.FileIO(vfs, self.path("foo"), mode="rb")
        self.assertEqual(fio.read(3), b"012")
        self.assertEqual(fio.tell(), 3)
        self.assertEqual(fio.read(3), b"345")
        self.assertEqual(fio.tell(), 6)
        self.assertEqual(fio.read(10), b"6789")
        self.assertEqual(fio.tell(), 10)

        # seek from beginning
        fio.seek(0)
        self.assertEqual(fio.tell(), 0)
        self.assertEqual(fio.read(), buffer)

        # seek must be positive when SEEK_SET
        with self.assertRaises(ValueError):
            fio.seek(-1, 0)

        # seek from current positfion
        fio.seek(5)
        self.assertEqual(fio.tell(), 5)
        fio.seek(3, 1)
        self.assertEqual(fio.tell(), 8)
        fio.seek(-3, 1)
        self.assertEqual(fio.tell(), 5)

        # seek from end
        fio.seek(-4, 2)
        self.assertEqual(fio.tell(), 6)

        # Test readall
        fio.seek(0)
        self.assertEqual(fio.readall(), buffer)
        self.assertEqual(fio.tell(), 10)

        fio.seek(5)
        self.assertEqual(fio.readall(), buffer[5:])
        self.assertEqual(fio.readall(), b"")

        # Test readinto
        fio.seek(0)
        test_bytes = bytearray(10)
        self.assertEqual(fio.readinto(test_bytes), 10)
        self.assertEqual(test_bytes, buffer)

        # Reading from the end should return empty
        fio.seek(0)
        fio.read()
        self.assertEqual(fio.read(), b"")

        # Test writing and reading lines with TextIOWrapper
        lines = [rand_utf8(random.randint(0, 50)) + "\n" for _ in range(10)]
        rand_uri = self.path("test_fio.rand")
        with tiledb.FileIO(vfs, rand_uri, "wb") as f:
            txtio = io.TextIOWrapper(f, encoding="utf-8")
            txtio.writelines(lines)
            txtio.flush()

        with tiledb.FileIO(vfs, rand_uri, "rb") as f2:
            txtio = io.TextIOWrapper(f2, encoding="utf-8")
            self.assertEqual(txtio.readlines(), lines)

    def test_ls(self):
        basepath = self.path("test_vfs_ls")
        self.vfs.create_dir(basepath)
        for id in (1, 2, 3):
            dir = os.path.join(basepath, "dir" + str(id))
            self.vfs.create_dir(dir)
            fname = os.path.join(basepath, "file_" + str(id))
            with tiledb.FileIO(self.vfs, fname, "wb") as fio:
                fio.write(b"")

        expected = ("file_1", "file_2", "file_3")
        # empty directories do not "exist" on s3
        if pytest.tiledb_vfs != "s3":
            expected = expected + ("dir1", "dir2", "dir3")

        self.assertSetEqual(
            set(expected),
            set(
                map(
                    lambda x: os.path.basename(x.split("test_vfs_ls")[1]),
                    self.vfs.ls(basepath),
                )
            ),
        )

    def test_dir_size(self):
        vfs = tiledb.VFS()

        path = self.path("test_vfs_dir_size")
        vfs.create_dir(path)
        rand_sizes = np.random.choice(100, size=4, replace=False)
        for size in rand_sizes:
            file_path = os.path.join(path, "f_" + str(size))
            with tiledb.FileIO(vfs, file_path, "wb") as f:
                data = os.urandom(size)
                f.write(data)

        self.assertEqual(vfs.dir_size(path), sum(rand_sizes))

    def test_open_with(self):
        uri = self.path("test_open_with")
        vfs = tiledb.VFS()
        buffer = b"0123456789"

        with vfs.open(uri, mode="wb") as fio:
            fio.write(buffer)
            fio.flush()
            self.assertEqual(fio.tell(), len(buffer))

        with vfs.open(uri, mode="rb") as fio:
            with self.assertRaises(IOError):
                fio.write(b"foo")
            self.assertEqual(fio.read(len(buffer)), buffer)
