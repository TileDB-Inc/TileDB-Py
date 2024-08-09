import numpy as np
import pytest

import tiledb

from .common import DiskTestCase, assert_captured


class FilestoreTest(DiskTestCase):
    @pytest.fixture
    def text_fname(self):
        path = self.path("text_fname")
        vfs = tiledb.VFS()
        vfs.touch(path)
        with vfs.open(path, "wb") as fio:
            fio.write(b"Simple text file.\n")
            fio.write(b"With two lines.")
        return path

    def test_buffer(self, capfd):
        path = self.path("test_buffer")
        data = b"buffer"

        fs = tiledb.Filestore(path)

        with self.assertRaises(tiledb.TileDBError):
            fs.write(data)

        schema = tiledb.ArraySchema.from_file()
        tiledb.Array.create(path, schema)

        assert schema.attr(0).name == "contents"
        assert schema.attr(0).dtype == np.bytes_

        schema.attr(0).dump()
        assert_captured(capfd, "Type: BLOB")

        fs = tiledb.Filestore(path)
        fs.write(data)
        assert bytes(data) == fs.read()

    def test_small_buffer(self, capfd):
        path = self.path("test_small_buffer")
        # create a 4 byte array
        data = b"abcd"

        fs = tiledb.Filestore(path)

        with self.assertRaises(tiledb.TileDBError):
            fs.write(data)

        schema = tiledb.ArraySchema.from_file()
        tiledb.Array.create(path, schema)

        assert schema.attr(0).name == "contents"
        assert schema.attr(0).dtype == np.bytes_

        schema.attr(0).dump()
        assert_captured(capfd, "Type: BLOB")

        fs = tiledb.Filestore(path)
        fs.write(data)
        assert data[3:4] == fs.read(offset=3, size=1)

    def test_uri(self, text_fname):
        path = self.path("test_uri")
        schema = tiledb.ArraySchema.from_file(text_fname)
        tiledb.Array.create(path, schema)

        fs = tiledb.Filestore(path)
        tiledb.Filestore.copy_from(path, text_fname)
        with open(text_fname, "rb") as text:
            data = text.read()
            assert data == fs.read(0, len(data))
            assert len(fs) == len(data)

    def test_deprecated_uri(self, text_fname):
        path = self.path("test_uri")
        schema = tiledb.ArraySchema.from_file(text_fname)
        tiledb.Array.create(path, schema)

        fs = tiledb.Filestore(path)
        with pytest.warns(
            DeprecationWarning, match="Filestore.uri_import is deprecated"
        ):
            fs.uri_import(text_fname)

        with open(text_fname, "rb") as text:
            data = text.read()
            assert data == fs.read(0, len(data))
            assert len(fs) == len(data)

    def test_multiple_writes(self):
        path = self.path("test_buffer")
        schema = tiledb.ArraySchema.from_file()
        tiledb.Array.create(path, schema)

        fs = tiledb.Filestore(path)
        for i in range(1, 4):
            fs.write(("x" * i).encode())

        assert fs.read() == ("x" * i).encode()

        timestamps = [t[0] for t in tiledb.array_fragments(path).timestamp_range]
        for i, ts in enumerate(timestamps, start=1):
            with tiledb.open(path, timestamp=ts) as A:
                assert A.meta["file_size"] == i
