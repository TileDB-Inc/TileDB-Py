import sys

import numpy as np
import pytest

import tiledb

from .common import DiskTestCase


class VersionTest(DiskTestCase):
    def test_libtiledb_version(self):
        v = tiledb.libtiledb.version()
        self.assertIsInstance(v, tuple)
        self.assertTrue(len(v) == 3)
        self.assertTrue(v[0] >= 1, "TileDB major version must be >= 1")

    def test_tiledbpy_version(self):
        v = tiledb.version.version
        self.assertIsInstance(v, str)

        v = tiledb.version()
        self.assertIsInstance(v, tuple)
        self.assertTrue(3 <= len(v) <= 5)


class GetStatsTest(DiskTestCase):
    def test_ctx(self):
        tiledb.stats_enable()
        ctx = tiledb.default_ctx()
        uri = self.path("test_ctx")
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 2), dtype=np.int64))
        att = tiledb.Attr(dtype=np.int64)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,))
        tiledb.Array.create(uri, schema)

        with tiledb.open(uri, mode="w", ctx=ctx) as T:
            T[:] = np.random.randint(10, size=3)

        stats = ctx.get_stats(print_out=False)
        assert "Context.StorageManager.write_store" in stats

    def test_query(self):
        tiledb.stats_enable()
        uri = self.path("test_ctx")
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 2), dtype=np.int64))
        att = tiledb.Attr(dtype=np.int64)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,))
        tiledb.Array.create(uri, schema)

        with tiledb.open(uri, mode="w") as T:
            T[:] = np.random.randint(10, size=3)

        with tiledb.open(uri, mode="r") as T:
            q = T.query()
            assert "" == q.get_stats()

            q[:]

            stats = q.get_stats(print_out=False)
            assert "Context.StorageManager.Query" in stats


class TestPath(DiskTestCase):
    def test_path(self, pytestconfig):
        path = self.path("foo")
        if pytestconfig.getoption("vfs") == "s3":
            assert path.startswith("s3://")

    @pytest.mark.skipif(
        sys.platform == "win32", reason="no_output fixture disabled on Windows"
    )
    @pytest.mark.xfail(
        True, reason="This test prints, and should fail because of no_output fixture!"
    )
    def test_no_output(self):
        print("this test should fail")


class TestAsBuilt(DiskTestCase):
    def test_as_built(self):
        dump = tiledb.as_built(return_json_string=True)
        assert isinstance(dump, str)
        # ensure we get a non-empty string
        assert len(dump) > 0
        dump_dict = tiledb.as_built()
        assert isinstance(dump_dict, dict)
        # ensure we get a non-empty dict
        assert len(dump_dict) > 0

        # validate top-level key
        assert "as_built" in dump_dict
        assert isinstance(dump_dict["as_built"], dict)
        assert len(dump_dict["as_built"]) > 0

        # validate parameters key
        assert "parameters" in dump_dict["as_built"]
        assert isinstance(dump_dict["as_built"]["parameters"], dict)
        assert len(dump_dict["as_built"]["parameters"]) > 0

        # validate storage_backends key
        assert "storage_backends" in dump_dict["as_built"]["parameters"]
        assert isinstance(dump_dict["as_built"]["parameters"]["storage_backends"], dict)
        assert len(dump_dict["as_built"]["parameters"]["storage_backends"]) > 0

        x = dump_dict["as_built"]["parameters"]["storage_backends"]

        # validate storage_backends attributes
        vfs = tiledb.VFS()
        if vfs.supports("azure"):
            assert x["azure"]["enabled"] == True
        else:
            assert x["azure"]["enabled"] == False

        if vfs.supports("gcs"):
            assert x["gcs"]["enabled"] == True
        else:
            assert x["gcs"]["enabled"] == False

        if vfs.supports("hdfs"):
            assert x["hdfs"]["enabled"] == True
        else:
            assert x["hdfs"]["enabled"] == False

        if vfs.supports("s3"):
            assert x["s3"]["enabled"] == True
        else:
            assert x["s3"]["enabled"] == False

        # validate support key
        assert "support" in dump_dict["as_built"]["parameters"]
        assert isinstance(dump_dict["as_built"]["parameters"]["support"], dict)
        assert len(dump_dict["as_built"]["parameters"]["support"]) > 0

        # validate support attributes - check only if boolean
        assert dump_dict["as_built"]["parameters"]["support"]["serialization"][
            "enabled"
        ] in [True, False]
