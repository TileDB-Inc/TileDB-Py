import gc
import io
import itertools
import os
import pickle
import random
import re
import urllib
import subprocess
import sys
import textwrap
import time
import unittest
import warnings
import xml.etree.ElementTree
from collections import OrderedDict
from contextlib import redirect_stdout

import numpy as np
import psutil
import pytest
from numpy.testing import assert_array_equal


import tiledb
from tiledb.tests.common import (
    assert_captured,
    assert_subarrays_equal,
    assert_unordered_equal,
    DiskTestCase,
    has_pandas,
    rand_ascii,
    rand_ascii_bytes,
    rand_utf8,
)
from tiledb.tests.fixtures import (
    sparse_cell_order,
    test_incomplete_return_array,
    INTEGER_DTYPES,
)  # pyright: reportUnusedVariable=warning
from tiledb.util import schema_from_dict


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


class StatsTest(DiskTestCase):
    def test_stats(self, capfd):
        tiledb.libtiledb.stats_enable()
        tiledb.libtiledb.stats_reset()
        tiledb.libtiledb.stats_disable()

        tiledb.libtiledb.stats_enable()
        with tiledb.from_numpy(self.path("test_stats"), np.arange(10)) as T:
            pass

        # basic output check for read stats
        tiledb.libtiledb.stats_reset()
        with tiledb.open(self.path("test_stats")) as T:
            tiledb.libtiledb.stats_enable()
            assert_array_equal(T, np.arange(10))

            # test stdout version
            tiledb.stats_dump()
            assert_captured(capfd, "TileDB Embedded Version:")

            # test string version
            stats_v = tiledb.stats_dump(print_out=False)
            if tiledb.libtiledb.version() < (2, 3):
                self.assertTrue("==== READ ====" in stats_v)
            else:
                self.assertTrue('"timers": {' in stats_v)
            self.assertTrue("==== Python Stats ====" in stats_v)

            if tiledb.libtiledb.version() < (2, 3):
                stats_quiet = tiledb.stats_dump(print_out=False, verbose=False)
                self.assertTrue("Time to load array schema" not in stats_quiet)

                # TODO seems to be a regression, no JSON
                stats_json = tiledb.stats_dump(json=True)
                self.assertTrue(isinstance(stats_json, dict))
                self.assertTrue("CONSOLIDATE_COPY_ARRAY" in stats_json)


@pytest.mark.skipif(
    "pytest.tiledb_vfs == 's3'", reason="Test not yet supported with S3"
)
class TestConfig(DiskTestCase):
    def test_config(self):
        config = tiledb.Config()
        config["sm.tile_cache_size"] = 100
        assert repr(config) is not None
        tiledb.Ctx(config)

    def test_ctx_config(self):
        ctx = tiledb.Ctx({"sm.tile_cache_size": 100})
        config = ctx.config()
        self.assertEqual(config["sm.tile_cache_size"], "100")

    def test_vfs_config(self):
        config = tiledb.Config()
        config["vfs.min_parallel_size"] = 1
        ctx = tiledb.Ctx()
        self.assertEqual(ctx.config()["vfs.min_parallel_size"], "10485760")
        vfs = tiledb.VFS(config, ctx=ctx)
        self.assertEqual(vfs.config()["vfs.min_parallel_size"], "1")

    def test_config_iter(self):
        config = tiledb.Config()
        k, v = [], []
        for p in config.items():
            k.append(p[0])
            v.append(p[1])
        self.assertTrue(len(k) > 0)

        k, v = [], []
        for p in config.items("vfs.s3."):
            k.append(p[0])
            v.append(p[1])
        self.assertTrue(len(k) > 0)

    def test_config_bad_param(self):
        config = tiledb.Config()
        config["sm.foo"] = "bar"
        ctx = tiledb.Ctx(config)
        self.assertEqual(ctx.config()["sm.foo"], "bar")

    def test_config_unset(self):
        config = tiledb.Config()
        config["sm.tile_cach_size"] = 100
        del config["sm.tile_cache_size"]
        # check that config parameter is default
        self.assertEqual(
            config["sm.tile_cache_size"], tiledb.Config()["sm.tile_cache_size"]
        )

    def test_config_from_file(self):
        # skip: beacuse Config.load doesn't support VFS-supported URIs?
        if pytest.tiledb_vfs == "s3":
            pytest.skip(
                "TODO need more plumbing to make pandas use TileDB VFS to read CSV files"
            )

        config_path = self.path("config")
        with tiledb.FileIO(self.vfs, config_path, "wb") as fh:
            fh.write("sm.tile_cache_size 100")
        config = tiledb.Config.load(config_path)
        self.assertEqual(config["sm.tile_cache_size"], "100")

    def test_ctx_config_from_file(self):
        config_path = self.path("config")
        vfs = tiledb.VFS()
        with tiledb.FileIO(vfs, config_path, "wb") as fh:
            fh.write("sm.tile_cache_size 100")
        ctx = tiledb.Ctx(config=tiledb.Config.load(config_path))
        config = ctx.config()
        self.assertEqual(config["sm.tile_cache_size"], "100")

    def test_ctx_config_dict(self):
        ctx = tiledb.Ctx(config={"sm.tile_cache_size": "100"})
        config = ctx.config()
        assert issubclass(type(config), tiledb.libtiledb.Config)
        self.assertEqual(config["sm.tile_cache_size"], "100")

    def test_config_repr_html(self):
        config = tiledb.Config()
        try:
            assert xml.etree.ElementTree.fromstring(config._repr_html_()) is not None
        except:
            pytest.fail(
                f"Could not parse config._repr_html_(). Saw {config._repr_html_()}"
            )


class DimensionTest(unittest.TestCase):
    def test_minimal_dimension(self):
        dim = tiledb.Dim(domain=(0, 4), tile=5)
        self.assertEqual(dim.name, "__dim_0", "automatic dimension name is incorrect")
        self.assertEqual(dim.shape, (5,))
        self.assertEqual(dim.tile, 5)

    def test_dimension(self):
        dim = tiledb.Dim(name="d1", domain=(0, 3), tile=2)
        self.assertEqual(dim.name, "d1")
        self.assertEqual(dim.shape, (4,))
        self.assertEqual(dim.tile, 2)
        try:
            assert xml.etree.ElementTree.fromstring(dim._repr_html_()) is not None
        except:
            pytest.fail(f"Could not parse dim._repr_html_(). Saw {dim._repr_html_()}")

    def test_dimension_filter(self):
        filters = [tiledb.GzipFilter(2)]
        dim = tiledb.Dim(name="df", domain=(0, 2), tile=1, filters=filters)
        self.assertEqual(dim.filters, filters)

        filter_list = tiledb.FilterList(filters)
        dim = tiledb.Dim(name="df", domain=(0, 2), tile=1, filters=filter_list)
        self.assertEqual(dim.filters, filter_list)

        with self.assertRaises(TypeError):
            tiledb.Dim(name="df", domain=(0, 2), tile=1, filters=1)

    def test_datetime_dimension(self):
        # Regular usage
        dim = tiledb.Dim(
            name="d1",
            domain=(np.datetime64("2010-01-01"), np.datetime64("2020-01-01")),
            tile=np.timedelta64(20, "D"),
            dtype=np.datetime64("", "D"),
        )
        self.assertEqual(dim.dtype, np.dtype(np.datetime64("", "D")))
        self.assertEqual(dim.tile, np.timedelta64(20, "D"))
        self.assertNotEqual(dim.tile, np.timedelta64(21, "D"))
        self.assertNotEqual(dim.tile, np.timedelta64(20, "W"))  # Sanity check unit
        self.assertTupleEqual(
            dim.domain, (np.datetime64("2010-01-01"), np.datetime64("2020-01-01"))
        )
        self.assertEqual(dim.shape, (3653,))

        # No tile extent specified: this is not an error in 2.2
        if tiledb.libtiledb.version() < (2, 2):
            with self.assertRaises(tiledb.TileDBError):
                tiledb.Dim(
                    name="d1",
                    domain=(np.datetime64("2010-01-01"), np.datetime64("2020-01-01")),
                    dtype=np.datetime64("", "D"),
                )

        # Integer tile extent is ok
        dim = tiledb.Dim(
            name="d1",
            domain=(np.datetime64("2010-01-01"), np.datetime64("2020-01-01")),
            tile=20,
            dtype=np.datetime64("", "D"),
        )
        self.assertEqual(dim.dtype, np.dtype(np.datetime64("", "D")))
        self.assertEqual(dim.tile, np.timedelta64(20, "D"))

        # Year resolution
        dim = tiledb.Dim(
            name="d1",
            domain=(np.datetime64("2010"), np.datetime64("2020")),
            tile=5,
            dtype=np.datetime64("", "Y"),
        )
        self.assertEqual(dim.dtype, np.dtype(np.datetime64("", "Y")))
        self.assertEqual(dim.tile, np.timedelta64(5, "Y"))
        self.assertTupleEqual(
            dim.domain, (np.datetime64("2010", "Y"), np.datetime64("2020", "Y"))
        )

        # End domain promoted to day resolution
        dim = tiledb.Dim(
            name="d1",
            domain=(np.datetime64("2010-01-01"), np.datetime64("2020")),
            tile=2,
            dtype=np.datetime64("", "D"),
        )
        self.assertEqual(dim.tile, np.timedelta64(2, "D"))
        self.assertTupleEqual(
            dim.domain,
            (np.datetime64("2010-01-01", "D"), np.datetime64("2020-01-01", "D")),
        )

        # Domain values can't be integral
        with self.assertRaises(TypeError):
            dim = tiledb.Dim(
                name="d1", domain=(-10, 10), tile=2, dtype=np.datetime64("", "D")
            )

    def test_shape(self):
        dim = tiledb.Dim(name="", dtype="|S0", var=True)
        with self.assertRaisesRegex(
            TypeError,
            "shape only valid for integer and datetime dimension domains",
        ):
            dim.shape


class DomainTest(DiskTestCase):
    def test_domain(self, capfd):
        dims = [
            tiledb.Dim("d1", (1, 4), 2, dtype="u8"),
            tiledb.Dim("d2", (1, 4), 2, dtype="u8"),
        ]
        dom = tiledb.Domain(*dims)

        # check that dumping works
        dom.dump()
        assert_captured(capfd, "Name: d1")

        self.assertEqual(dom.ndim, 2)
        self.assertEqual(dom.dtype, np.dtype("uint64"))
        self.assertEqual(dom.shape, (4, 4))

        # check that we can iterate over the dimensions
        dim_names = [dim.name for dim in dom]
        self.assertEqual(["d1", "d2"], dim_names)

        # check that we can access dim by name
        dim_d1 = dom.dim("d1")
        self.assertEqual(dim_d1, dom.dim(0))

        # check that we can construct directly from a List[Dim]
        dom2 = tiledb.Domain(dims)
        self.assertEqual(dom, dom2)

        try:
            assert xml.etree.ElementTree.fromstring(dom._repr_html_()) is not None
        except:
            pytest.fail(f"Could not parse dom._repr_html_(). Saw {dom._repr_html_()}")

    def test_datetime_domain(self):
        dim = tiledb.Dim(
            name="d1",
            domain=(np.datetime64("2010-01-01"), np.datetime64("2020-01-01")),
            tile=np.timedelta64(20, "D"),
            dtype=np.datetime64("", "D"),
        )
        dom = tiledb.Domain(dim)
        self.assertEqual(dom.dtype, np.datetime64("", "D"))

    def test_domain_mixed_names_error(self):
        with self.assertRaises(tiledb.TileDBError):
            tiledb.Domain(
                tiledb.Dim("d1", (1, 4), 2, dtype="u8"),
                tiledb.Dim("__dim_0", (1, 4), 2, dtype="u8"),
            )

    def test_ascii_domain(self, capfd):
        path = self.path("test_ascii_domain")

        dim = tiledb.Dim(name="d", dtype="ascii")
        assert dim.dtype == np.bytes_

        dom = tiledb.Domain(dim)
        dom.dump()
        assert_captured(capfd, "Type: STRING_ASCII")

        att = tiledb.Attr(name="a", dtype=np.int64)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True)
        tiledb.SparseArray.create(path, schema)

        ascii_coords = ["a", "b", "c", "ABC"]
        unicode_coords = ["±", "×", "÷", "√"]
        data = [1, 2, 3, 4]

        with tiledb.open(path, "w") as A:
            with self.assertRaises(tiledb.TileDBError):
                A[unicode_coords] = data
            A[ascii_coords] = data


class AttributeTest(DiskTestCase):
    def test_minimal_attribute(self):
        attr = tiledb.Attr()
        self.assertTrue(attr.isanon)
        self.assertEqual(attr.name, "")
        self.assertEqual(attr.dtype, np.float_)
        # self.assertEqual(attr.compressor, (None, -1))
        self.assertFalse(attr.isvar)
        self.assertFalse(attr.isnullable)

        try:
            assert xml.etree.ElementTree.fromstring(attr._repr_html_()) is not None
        except:
            pytest.fail(f"Could not parse attr._repr_html_(). Saw {attr._repr_html_()}")

    def test_attribute(self, capfd):
        attr = tiledb.Attr("foo")

        attr.dump()
        assert_captured(capfd, "Name: foo")

        assert attr.name == "foo"
        assert attr.dtype == np.float64, "default attribute type is float64"
        # compressor, level = attr.compressor
        # self.assertEqual(compressor, None, "default to no compression")
        # self.assertEqual(level, -1, "default compression level when none is specified")

    @pytest.mark.parametrize(
        "dtype, fill",
        [
            (np.dtype(bytes), b"abc"),
            # (str, "defg"),
            (np.float32, np.float32(0.4023573667780681)),
            (np.float64, np.float64(0.0560602549760851)),
            (np.dtype("M8[ns]"), np.timedelta64(11, "ns")),
            (np.dtype([("f0", "<i4"), ("f1", "<i4"), ("f2", "<i4")]), (1, 2, 3)),
        ],
    )
    def test_attribute_fill(self, dtype, fill):
        attr = tiledb.Attr("", dtype=dtype, fill=fill)
        assert np.array(attr.fill, dtype=dtype) == np.array(fill, dtype=dtype)

        path = self.path()
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 0), tile=1, dtype=np.int64))
        schema = tiledb.ArraySchema(domain=dom, attrs=(attr,))
        tiledb.DenseArray.create(path, schema)

        with tiledb.open(path) as R:
            assert R.multi_index[0][""] == np.array(fill, dtype=dtype)
            assert R[0] == np.array(fill, dtype=dtype)
            if has_pandas() and not hasattr(dtype, "fields"):
                # record type unsupported for .df
                assert R.df[0][""].values == np.array(fill, dtype=dtype)

    def test_full_attribute(self, capfd):
        filter_list = tiledb.FilterList([tiledb.ZstdFilter(10)])
        filter_list = tiledb.FilterList([tiledb.ZstdFilter(10)])
        attr = tiledb.Attr("foo", dtype=np.int64, filters=filter_list)

        attr.dump()
        assert_captured(capfd, "Name: foo")

        self.assertEqual(attr.name, "foo")
        self.assertEqual(attr.dtype, np.int64)

        # <todo>
        # compressor, level = attr.compressor
        # self.assertEqual(compressor, "zstd")
        # self.assertEqual(level, 10)

    def test_ncell_attribute(self):
        dtype = np.dtype([("", np.int32), ("", np.int32), ("", np.int32)])
        attr = tiledb.Attr("foo", dtype=dtype)

        self.assertEqual(attr.dtype, dtype)
        self.assertEqual(attr.ncells, 3)

        # dtype subarrays not supported
        with self.assertRaises(TypeError):
            tiledb.Attr("foo", dtype=np.dtype((np.int32, 2)))

        # mixed type record arrays not supported
        with self.assertRaises(TypeError):
            tiledb.Attr("foo", dtype=np.dtype([("", np.float32), ("", np.int32)]))

    def test_ncell_bytes_attribute(self):
        dtype = np.dtype((np.bytes_, 10))
        attr = tiledb.Attr("foo", dtype=dtype)

        self.assertEqual(attr.dtype, dtype)
        self.assertEqual(attr.ncells, 10)

    def test_bytes_var_attribute(self):
        with pytest.warns(DeprecationWarning, match="Attr given `var=True` but"):
            attr = tiledb.Attr("foo", var=True, dtype="S1")
            self.assertEqual(attr.dtype, np.dtype("S"))
            self.assertTrue(attr.isvar)

        with pytest.warns(DeprecationWarning, match="Attr given `var=False` but"):
            attr = tiledb.Attr("foo", var=False, dtype="S")
            self.assertEqual(attr.dtype, np.dtype("S"))
            self.assertTrue(attr.isvar)

        attr = tiledb.Attr("foo", var=True, dtype="S")
        self.assertEqual(attr.dtype, np.dtype("S"))
        self.assertTrue(attr.isvar)

        attr = tiledb.Attr("foo", var=False, dtype="S1")
        self.assertEqual(attr.dtype, np.dtype("S1"))
        self.assertFalse(attr.isvar)

        attr = tiledb.Attr("foo", dtype="S1")
        self.assertEqual(attr.dtype, np.dtype("S1"))
        self.assertFalse(attr.isvar)

        attr = tiledb.Attr("foo", dtype="S")
        self.assertEqual(attr.dtype, np.dtype("S"))
        self.assertTrue(attr.isvar)

    def test_nullable_attribute(self):
        attr = tiledb.Attr("nullable", nullable=True, dtype=np.int32)
        self.assertEqual(attr.dtype, np.dtype(np.int32))
        self.assertTrue(attr.isnullable)

    def test_datetime_attribute(self):
        attr = tiledb.Attr("foo", dtype=np.datetime64("", "D"))
        assert attr.dtype == np.dtype(np.datetime64("", "D"))
        assert attr.dtype != np.dtype(np.datetime64("", "Y"))
        assert attr.dtype != np.dtype(np.datetime64)

    @pytest.mark.parametrize("sparse", [True, False])
    def test_ascii_attribute(self, sparse, capfd):
        path = self.path("test_ascii")
        dom = tiledb.Domain(
            tiledb.Dim(name="d", domain=(1, 4), tile=1, dtype=np.uint32)
        )
        attrs = [tiledb.Attr(name="A", dtype="ascii", var=True)]
        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=sparse)
        tiledb.Array.create(path, schema)

        ascii_data = ["a", "b", "c", "ABC"]
        unicode_data = ["±", "×", "÷", "√"]

        with tiledb.open(path, "w") as A:
            if sparse:
                with self.assertRaises(tiledb.TileDBError):
                    A[np.arange(1, 5)] = unicode_data
                A[np.arange(1, 5)] = ascii_data
            else:
                with self.assertRaises(tiledb.TileDBError):
                    A[:] = unicode_data
                A[:] = ascii_data

        with tiledb.open(path, "r") as A:
            assert A.schema.nattr == 1
            A.schema.dump()
            assert_captured(capfd, "Type: STRING_ASCII")
            assert A.schema.attr("A").dtype == np.bytes_
            assert A.schema.attr("A").isascii
            assert_array_equal(A[:]["A"], np.asarray(ascii_data, dtype=np.bytes_))


class ArraySchemaTest(DiskTestCase):
    def test_schema_basic(self):
        dom = tiledb.Domain(
            tiledb.Dim("d1", (1, 4), 2, dtype="u8"),
            tiledb.Dim("d2", (1, 4), 2, dtype="u8"),
        )

        attr1 = tiledb.Attr("foo", dtype=float)
        attr2 = tiledb.Attr("foo", dtype=int)

        # test unique attributes
        with self.assertRaises(tiledb.TileDBError):
            tiledb.ArraySchema(domain=dom, attrs=(attr1, attr2))

        # test schema.check
        schema = tiledb.ArraySchema(domain=dom, attrs=(attr1,))
        # valid schema does not raise
        schema.check()

        try:
            assert xml.etree.ElementTree.fromstring(schema._repr_html_()) is not None
        except:
            pytest.fail(
                f"Could not parse schema._repr_html_(). Saw {schema._repr_html_()}"
            )

        with self.assertRaises(tiledb.TileDBError):
            schema._make_invalid()
            schema.check()

    def test_dense_array_schema(self):
        domain = tiledb.Domain(
            tiledb.Dim(domain=(1, 8), tile=2), tiledb.Dim(domain=(1, 8), tile=2)
        )
        a1 = tiledb.Attr("val", dtype="f8")
        schema = tiledb.ArraySchema(domain=domain, attrs=(a1,))
        assert schema.sparse == False
        assert schema.cell_order == "row-major"
        assert schema.tile_order == "row-major"
        assert schema.domain == domain
        assert schema.ndim == 2
        assert schema.shape == (8, 8)
        assert schema.nattr == 1
        assert schema.domain.homogeneous == True
        assert hasattr(schema, "version")  # don't pin to a specific version
        assert schema.attr(0) == a1
        assert schema.has_attr("val") == True
        assert schema.has_attr("nononoattr") == False
        assert schema == tiledb.ArraySchema(domain=domain, attrs=(a1,))
        assert schema != tiledb.ArraySchema(domain=domain, attrs=(a1,), sparse=True)

        with self.assertRaises(tiledb.TileDBError):
            schema.allows_duplicates
        # test iteration over attributes
        assert list(schema) == [a1]

        with self.assertRaisesRegex(
            tiledb.TileDBError,
            "Cannot set cell order; Hilbert order is only applicable to sparse arrays",
        ):
            tiledb.ArraySchema(
                domain=domain, attrs=(a1,), sparse=False, cell_order="hilbert"
            )

    def test_dense_array_schema_fp_domain_error(self):
        dom = tiledb.Domain(tiledb.Dim(domain=(1, 8), tile=2, dtype=np.float64))
        att = tiledb.Attr("val", dtype=np.float64)

        with self.assertRaises(tiledb.TileDBError):
            tiledb.ArraySchema(domain=dom, attrs=(att,))

    def test_sparse_schema(self, capfd):
        # create dimensions
        d1 = tiledb.Dim("d1", domain=(1, 1000), tile=10, dtype="uint64")
        d2 = tiledb.Dim("d2", domain=(101, 10000), tile=100, dtype="uint64")

        # create domain
        domain = tiledb.Domain(d1, d2)

        # create attributes
        a1 = tiledb.Attr("a1", dtype="int32,int32,int32")
        a2 = tiledb.Attr(
            "a2", filters=tiledb.FilterList([tiledb.GzipFilter(-1)]), dtype="float32"
        )

        # create sparse array with schema
        coords_filters = tiledb.FilterList([tiledb.ZstdFilter(4)])
        offsets_filters = tiledb.FilterList([tiledb.LZ4Filter(5)])
        validity_filters = tiledb.FilterList([tiledb.GzipFilter(9)])

        with pytest.warns(
            DeprecationWarning,
            match="coords_filters is deprecated; set the FilterList for each dimension",
        ):
            schema = tiledb.ArraySchema(
                domain=domain,
                attrs=(a1, a2),
                capacity=10,
                cell_order="col-major",
                tile_order="row-major",
                allows_duplicates=True,
                sparse=True,
                coords_filters=coords_filters,
                offsets_filters=offsets_filters,
                validity_filters=validity_filters,
            )

        schema.dump()
        assert_captured(capfd, "Array type: sparse")

        assert schema.sparse == True
        assert schema.capacity == 10
        assert schema.cell_order, "co == major"
        assert schema.tile_order, "ro == major"

        # <todo>
        # assert schema.coords_compressor, ('zstd' == 4)
        # assert schema.offsets_compressor, ('lz4' == 5)
        assert len(schema.coords_filters) == 0
        assert len(schema.offsets_filters) == 1
        assert len(schema.validity_filters) == 1

        assert schema.domain == domain
        assert schema.ndim == 2
        assert schema.shape, 1000 == 9900
        assert schema.nattr == 2
        assert schema.attr(0) == a1
        assert schema.attr("a2") == a2
        assert schema.allows_duplicates == True

        assert schema.domain.dim("d1").filters == coords_filters
        assert schema.domain.dim("d2").filters == coords_filters

        with pytest.warns(
            DeprecationWarning,
            match="coords_filters is deprecated; set the FilterList for each dimension",
        ):
            schema2 = tiledb.ArraySchema(
                domain=domain,
                attrs=(a1, a2),
                capacity=10,
                cell_order="col-major",
                tile_order="row-major",
                allows_duplicates=True,
                sparse=True,
                coords_filters=coords_filters,
                offsets_filters=offsets_filters,
                validity_filters=validity_filters,
            )
        assert schema == schema2

        # test iteration over attributes
        assert list(schema) == [a1, a2]

        with self.assertRaisesRegex(
            tiledb.TileDBError,
            "Cannot set tile order; Hilbert order is not applicable to tiles",
        ):
            tiledb.ArraySchema(
                domain=domain, attrs=(a1,), sparse=True, tile_order="hilbert"
            )

    def test_sparse_schema_filter_list(self, capfd):
        # create dimensions
        d1 = tiledb.Dim("d1", domain=(1, 1000), tile=10, dtype="uint64")
        d2 = tiledb.Dim("d2", domain=(101, 10000), tile=100, dtype="uint64")

        # create domain
        domain = tiledb.Domain(d1, d2)

        # create attributes
        a1 = tiledb.Attr("a1", dtype="int32,int32,int32")
        filter_list = tiledb.FilterList([tiledb.GzipFilter()])
        a2 = tiledb.Attr("a2", filters=filter_list, dtype="float32")

        off_filters_pylist = [tiledb.ZstdFilter(level=10)]
        off_filters = tiledb.FilterList(filters=off_filters_pylist, chunksize=2048)

        coords_filters_pylist = [tiledb.Bzip2Filter(level=5)]
        coords_filters = tiledb.FilterList(
            filters=coords_filters_pylist, chunksize=4096
        )

        validity_filters_pylist = [tiledb.GzipFilter(level=9)]
        validity_filters = tiledb.FilterList(
            filters=validity_filters_pylist, chunksize=1024
        )

        # create sparse array with schema
        with pytest.warns(
            DeprecationWarning,
            match="coords_filters is deprecated; set the FilterList for each dimension",
        ):
            schema = tiledb.ArraySchema(
                domain=domain,
                attrs=(a1, a2),
                capacity=10,
                cell_order="col-major",
                tile_order="row-major",
                coords_filters=coords_filters,
                offsets_filters=off_filters,
                validity_filters=validity_filters,
                sparse=True,
            )
        self.assertTrue(schema.sparse)

        assert len(schema.coords_filters) == 0

        assert len(schema.domain.dim("d1").filters) == 1
        assert schema.domain.dim("d1").filters[0] == tiledb.Bzip2Filter(level=5)
        assert schema.domain.dim("d2").filters[0] == tiledb.Bzip2Filter(level=5)

        assert len(schema.offsets_filters) == 1
        assert schema.offsets_filters[0] == tiledb.ZstdFilter(level=10)

        assert len(schema.validity_filters) == 1
        assert schema.validity_filters[0] == tiledb.GzipFilter(level=9)

        schema.dump()
        assert_captured(capfd, "Array type: sparse")

        # make sure we can construct ArraySchema with python lists of filters
        with pytest.warns(
            DeprecationWarning,
            match="coords_filters is deprecated; set the FilterList for each dimension",
        ):
            schema2 = tiledb.ArraySchema(
                domain=domain,
                attrs=(a1, a2),
                capacity=10,
                cell_order="col-major",
                tile_order="row-major",
                coords_filters=coords_filters_pylist,
                offsets_filters=off_filters,
                validity_filters=validity_filters,
                sparse=True,
            )
        assert len(schema2.coords_filters) == 0

        assert schema.domain.dim("d1").filters == coords_filters_pylist
        assert schema.domain.dim("d2").filters == coords_filters_pylist

        assert len(schema2.domain.dim("d1").filters) == 1
        assert schema2.domain.dim("d1").filters[0] == tiledb.Bzip2Filter(level=5)
        assert schema2.domain.dim("d2").filters[0] == tiledb.Bzip2Filter(level=5)

        assert len(schema2.offsets_filters) == 1
        assert schema2.offsets_filters[0] == tiledb.ZstdFilter(level=10)

        assert len(schema2.validity_filters) == 1
        assert schema2.validity_filters[0] == tiledb.GzipFilter(level=9)

    def test_none_filter_list(self):
        with self.assertRaises(ValueError):
            tiledb.FilterList([None])

        with self.assertRaises(ValueError):
            fl = tiledb.FilterList()
            fl.append(None)

    def test_mixed_string_schema(self):
        path = self.path("test_mixed_string_schema")

        dims = [
            tiledb.Dim(name="dpos", domain=(-100.0, 100.0), tile=10, dtype=np.float64),
            tiledb.Dim(name="str_index", tile=None, dtype=np.bytes_),
        ]
        dom = tiledb.Domain(*dims)
        attrs = [tiledb.Attr(name="val", dtype=np.float64)]

        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=True)

        self.assertTrue(schema.domain.has_dim("str_index"))
        self.assertFalse(schema.domain.has_dim("nonono_str_index"))
        self.assertTrue(schema.domain.dim("str_index").isvar)
        self.assertFalse(schema.domain.dim("dpos").isvar)
        self.assertEqual(schema.domain.dim("dpos").dtype, np.double)
        self.assertEqual(schema.domain.dim("str_index").dtype, np.bytes_)
        self.assertFalse(schema.domain.homogeneous)

        tiledb.Array.create(path, schema)
        with tiledb.open(path, "r") as arr:
            assert_array_equal(arr[:]["str_index"], np.array([], dtype="|S1"))


class ArrayTest(DiskTestCase):
    def create_array_schema(self):
        domain = tiledb.Domain(
            tiledb.Dim(domain=(1, 8), tile=2), tiledb.Dim(domain=(1, 8), tile=2)
        )
        a1 = tiledb.Attr("val", dtype="f8")
        return tiledb.ArraySchema(domain=domain, attrs=(a1,))

    def test_array_create(self):
        config = tiledb.Config()
        config["sm.consolidation.step_min_frag"] = 0
        config["sm.consolidation.steps"] = 1
        schema = self.create_array_schema()

        # persist array schema
        tiledb.libtiledb.Array.create(self.path("foo"), schema)

        # these should be no-ops
        #   full signature
        tiledb.consolidate(self.path("foo"), config=config)
        #   kw signature
        tiledb.consolidate(uri=self.path("foo"))

        # load array in readonly mode
        array = tiledb.libtiledb.Array(self.path("foo"), mode="r")
        self.assertTrue(array.isopen)
        self.assertEqual(array.schema, schema)
        self.assertEqual(array.mode, "r")
        self.assertEqual(array.uri, self.path("foo"))

        # test that we cannot consolidate an array in readonly mode
        with self.assertRaises(tiledb.TileDBError):
            array.consolidate()

        # we have not written anything, so the array is empty
        self.assertIsNone(array.nonempty_domain())

        array.reopen()
        self.assertTrue(array.isopen)

        array.close()
        self.assertEqual(array.isopen, False)

        with self.assertRaises(tiledb.TileDBError):
            # cannot get schema from closed array
            array.schema

        with self.assertRaises(tiledb.TileDBError):
            # cannot re-open a closed array
            array.reopen()

    def test_array_create_with_ctx(self):
        schema = self.create_array_schema()

        with self.assertRaises(TypeError):
            tiledb.libtiledb.Array.create(self.path("foo"), schema, ctx="foo")

        # persist array schema
        tiledb.libtiledb.Array.create(self.path("foo"), schema, ctx=tiledb.Ctx())

    @pytest.mark.skipif(
        not (sys.platform == "win32" and tiledb.libtiledb.version() >= (2, 3, 0)),
        reason="Shared network drive only on Win32",
    )
    def test_array_create_on_shared_drive(self):
        schema = self.create_array_schema()
        uri = self.path(basename="foo", shared=True)

        tiledb.libtiledb.Array.create(uri, schema)

        # load array in readonly mode
        array = tiledb.libtiledb.Array(uri, mode="r")
        self.assertTrue(array.isopen)
        self.assertEqual(array.schema, schema)
        self.assertEqual(array.mode, "r")
        self.assertEqual(array.uri, uri)

        # we have not written anything, so the array is empty
        self.assertIsNone(array.nonempty_domain())

        array.reopen()
        self.assertTrue(array.isopen)

        array.close()
        self.assertEqual(array.isopen, False)

        with self.assertRaises(tiledb.TileDBError):
            # cannot get schema from closed array
            array.schema

        with self.assertRaises(tiledb.TileDBError):
            # cannot re-open a closed array
            array.reopen()

    def test_array_create_encrypted(self):
        config = tiledb.Config()
        config["sm.consolidation.step_min_frags"] = 0
        config["sm.consolidation.steps"] = 1
        schema = self.create_array_schema()
        # persist array schema
        tiledb.libtiledb.Array.create(
            self.path("foo"), schema, key=b"0123456789abcdeF0123456789abcdeF"
        )

        # check that we can open the array sucessfully
        for key in (
            b"0123456789abcdeF0123456789abcdeF",
            "0123456789abcdeF0123456789abcdeF",
        ):
            with tiledb.libtiledb.Array(self.path("foo"), mode="r", key=key) as array:
                self.assertTrue(array.isopen)
                self.assertEqual(array.schema, schema)
                self.assertEqual(array.mode, "r")
            with tiledb.open(self.path("foo"), mode="r", key=key) as array:
                self.assertTrue(array.isopen)
                self.assertEqual(array.schema, schema)
                self.assertEqual(array.mode, "r")

            tiledb.consolidate(uri=self.path("foo"), config=config, key=key)

        # check that opening the array with the wrong key fails:
        with self.assertRaises(tiledb.TileDBError):
            tiledb.libtiledb.Array(
                self.path("foo"), mode="r", key=b"0123456789abcdeF0123456789abcdeX"
            )

        # check that opening the array with the wrong key length fails:
        with self.assertRaises(tiledb.TileDBError):
            tiledb.libtiledb.Array(
                self.path("foo"), mode="r", key=b"0123456789abcdeF0123456789abcde"
            )

        # check that consolidating the array with the wrong key fails:
        with self.assertRaises(tiledb.TileDBError):
            tiledb.consolidate(
                self.path("foo"), config=config, key=b"0123456789abcdeF0123456789abcde"
            )

    # needs core fix in 2.2.4
    @pytest.mark.skipif(
        (sys.platform == "win32" and tiledb.libtiledb.version() == (2, 2, 3)),
        reason="Skip array_doesnt_exist test on Win32 / libtiledb 2.2.3",
    )
    def test_array_doesnt_exist(self):
        with self.assertRaises(tiledb.TileDBError):
            tiledb.libtiledb.Array(self.path("foo"), mode="r")

    def test_create_schema_matches(self):
        dims = (tiledb.Dim(domain=(0, 6), tile=2),)
        dom = tiledb.Domain(*dims)
        att = tiledb.Attr(dtype=np.byte)

        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True)
        uri = self.path("s1")
        with self.assertRaises(ValueError):
            tiledb.DenseArray.create(uri, schema)

        dense_schema = tiledb.ArraySchema(domain=dom, attrs=(att,))
        uri = self.path("d1")
        with self.assertRaises(ValueError):
            tiledb.SparseArray.create(uri, dense_schema)

        class MySparseArray(tiledb.SparseArray):
            pass

        with self.assertRaises(ValueError):
            MySparseArray.create(uri, dense_schema)

    def test_nonempty_domain_scalar(self):
        uri = self.path("test_nonempty_domain_scalar")
        dims = tiledb.Dim(domain=(-10, 10), dtype=np.int64, tile=1)
        schema = tiledb.ArraySchema(
            tiledb.Domain(dims), attrs=[tiledb.Attr(dtype=np.int32)], sparse=True
        )

        tiledb.Array.create(uri, schema)

        with tiledb.open(uri, "w") as A:
            A[-1] = 10
            A[1] = 11

        with tiledb.open(uri, "r") as A:
            ned = A.nonempty_domain()
            assert_array_equal(ned, ((-1, 1),))
            assert isinstance(ned[0][0], int)
            assert isinstance(ned[0][1], int)

    def test_create_array_overwrite(self):
        uri = self.path("test_create_array_overwrite")
        dims = tiledb.Dim(domain=(0, 10), dtype=np.int64)
        schema = tiledb.ArraySchema(
            tiledb.Domain(dims), attrs=[tiledb.Attr(dtype=np.int32)], sparse=True
        )

        with pytest.warns(UserWarning, match="Overwrite set, but array does not exist"):
            tiledb.Array.create(uri, schema, overwrite=True)

        with tiledb.open(uri, "w") as A:
            A[0] = 1

        with tiledb.open(uri, "r") as A:
            assert A.nonempty_domain() == ((0, 0),)

        # cannot overwrite the array by default
        with self.assertRaises(tiledb.TileDBError):
            tiledb.Array.create(uri, schema)

        tiledb.Array.create(uri, schema, overwrite=True)

        # make the old array has been deleted and replaced
        with tiledb.open(uri, "r") as A:
            assert A.nonempty_domain() is None


class DenseArrayTest(DiskTestCase):
    def test_array_1d(self):
        A = np.arange(1050)

        dom = tiledb.Domain(tiledb.Dim(domain=(0, 1049), tile=100, dtype=np.int64))
        att = tiledb.Attr(dtype=A.dtype)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,))
        tiledb.DenseArray.create(self.path("foo"), schema)

        with tiledb.DenseArray(self.path("foo"), mode="r") as T:
            self.assertEqual(len(A), len(T))
            self.assertEqual(A.ndim, T.ndim)
            self.assertEqual(A.shape, T.shape)

            self.assertEqual(1, T.nattr)
            self.assertEqual(A.dtype, T.attr(0).dtype)
            self.assertEqual(T.dim(T.schema.domain.dim(0).name), T.dim(0))
            with self.assertRaises(ValueError):
                T.dim(1.0)

            self.assertIsInstance(T.timestamp_range, tuple)
            self.assertTrue(T.timestamp_range[1] > 0)

            # check empty array
            B = T[:]

            self.assertEqual(A.shape, B.shape)
            self.assertEqual(A.dtype, B.dtype)
            self.assertIsNone(T.nonempty_domain())

        with tiledb.DenseArray(self.path("foo"), mode="w") as T:
            # check set array
            T[:] = A

        read1_timestamp = -1
        with tiledb.DenseArray(self.path("foo"), mode="r") as T:
            self.assertEqual(((0, 1049),), T.nonempty_domain())

            # check timestamp
            read1_timestamp = T.timestamp_range
            self.assertTrue(read1_timestamp[1] > 0)

            # check slicing
            assert_array_equal(A, np.array(T))
            assert_array_equal(A, T[:])
            assert_array_equal(A, T[...])
            assert_array_equal(A, T[slice(None)])
            assert_array_equal(A[:10], T[:10])
            assert_array_equal(A[10:20], T[10:20])
            assert_array_equal(A[-10:], T[-10:])

            # ellipsis
            assert_array_equal(A[:10, ...], T[:10, ...])
            assert_array_equal(A[10:50, ...], T[10:50, ...])
            assert_array_equal(A[-50:, ...], T[-50:, ...])
            assert_array_equal(A[..., :10], T[..., :10])
            assert_array_equal(A[..., 10:20], T[..., 10:20])
            assert_array_equal(A[..., -50:], T[..., -50:])

            # across tiles
            assert_array_equal(A[:150], T[:150])
            assert_array_equal(A[-250:], T[-250:])

            # point index
            self.assertEqual(A[0], T[0])
            self.assertEqual(A[-1], T[-1])

            # point index with all index types
            self.assertEqual(A[123], T[np.int8(123)])
            self.assertEqual(A[123], T[np.uint8(123)])
            self.assertEqual(A[123], T[np.int16(123)])
            self.assertEqual(A[123], T[np.uint16(123)])
            self.assertEqual(A[123], T[np.int64(123)])
            self.assertEqual(A[123], T[np.uint64(123)])
            self.assertEqual(A[123], T[np.int32(123)])
            self.assertEqual(A[123], T[np.uint32(123)])

            # mixed-type slicing
            # https://github.com/TileDB-Inc/TileDB-Py/issues/140
            self.assertEqual(A[0:1], T[0 : np.uint16(1)])
            self.assertEqual(A[0:1], T[np.int64(0) : 1])
            with self.assertRaises(IndexError):
                # this is a consequence of NumPy promotion rules
                self.assertEqual(A[0:1], T[np.uint64(0) : 1])

            # basic step
            assert_array_equal(A[:50:2], T[:50:2])
            assert_array_equal(A[:2:50], T[:2:50])
            assert_array_equal(A[10:-1:50], T[10:-1:50])

            # indexing errors
            with self.assertRaises(IndexError):
                T[:, :]
            with self.assertRaises(IndexError):
                T[:, 50]
            with self.assertRaises(IndexError):
                T[50, :]
            with self.assertRaises(IndexError):
                T[0, 0]

            # check single ellipsis
            with self.assertRaises(IndexError):
                T[..., 1:5, ...]

        with tiledb.DenseArray(self.path("foo"), mode="w") as T:
            # check partial assignment
            B = np.arange(1e5, 2e5).astype(A.dtype)
            T[190:310] = B[190:310]

        read2_timestamp = -1
        with tiledb.DenseArray(self.path("foo"), mode="r") as T:
            assert_array_equal(A[:190], T[:190])
            assert_array_equal(B[190:310], T[190:310])
            assert_array_equal(A[310:], T[310:])

            # test timestamps are updated
            read2_timestamp = T.timestamp_range
            self.assertTrue(read2_timestamp > read1_timestamp)

    def test_array_1d_set_scalar(self):
        A = np.zeros(50)

        dom = tiledb.Domain(tiledb.Dim(domain=(0, 49), tile=50))
        att = tiledb.Attr(dtype=A.dtype)
        schema = tiledb.ArraySchema(dom, (att,))

        tiledb.DenseArray.create(self.path("foo"), schema)

        with tiledb.DenseArray(self.path("foo"), mode="w") as T:
            T[:] = A

        with tiledb.DenseArray(self.path("foo"), mode="r") as T:
            assert_array_equal(A, T[:])

        with tiledb.DenseArray(self.path("foo"), mode="w") as T:
            value = -1, 3, 10
            A[0], A[1], A[3] = value
            T[0], T[1], T[3] = value
        with tiledb.DenseArray(self.path("foo"), mode="r") as T:
            assert_array_equal(A, T[:])

        for value in (-1, 3, 10):
            with tiledb.DenseArray(self.path("foo"), mode="w") as T:
                A[5:25] = value
                T[5:25] = value
            with tiledb.DenseArray(self.path("foo"), mode="r") as T:
                assert_array_equal(A, T[:])
            with tiledb.DenseArray(self.path("foo"), mode="w") as T:
                A[:] = value
                T[:] = value
            with tiledb.DenseArray(self.path("foo"), mode="r") as T:
                assert_array_equal(A, T[:])

    def test_array_id_point_queries(self):
        # TODO: handle queries like T[[2, 5, 10]] = ?
        pass

    @pytest.mark.parametrize("dtype", INTEGER_DTYPES)
    def test_dense_index_dtypes(self, dtype):
        path = self.path()
        data = np.arange(0, 3).astype(dtype)
        with tiledb.from_numpy(path, data) as A:
            pass
        with tiledb.open(path) as B:
            assert_array_equal(B[:], data)

    def test_array_2d(self):
        A = np.arange(10000).reshape((1000, 10))

        dom = tiledb.Domain(
            tiledb.Dim(domain=(0, 999), tile=100), tiledb.Dim(domain=(0, 9), tile=2)
        )
        att = tiledb.Attr(dtype=A.dtype)
        schema = tiledb.ArraySchema(dom, (att,))

        tiledb.DenseArray.create(self.path("foo"), schema)

        with tiledb.DenseArray(self.path("foo"), mode="r") as T:
            self.assertEqual(len(A), len(T))
            self.assertEqual(A.ndim, T.ndim)
            self.assertEqual(A.shape, T.shape)

            self.assertEqual(1, T.nattr)
            self.assertEqual(A.dtype, T.attr(0).dtype)

            # check that the non-empty domain is None
            self.assertIsNone(T.nonempty_domain())

        with tiledb.DenseArray(self.path("foo"), mode="w") as T:
            # Set data
            T[:] = A

        with tiledb.DenseArray(self.path("foo"), mode="r") as T:
            assert_array_equal(A, T[:])

            # check the non-empty domain spans the whole domain
            self.assertEqual(((0, 999), (0, 9)), T.nonempty_domain())

            # check array-like
            assert_array_equal(A, np.array(T))

            # slicing
            assert_array_equal(A, T[:])
            assert_array_equal(A, T[...])
            assert_array_equal(A, T[slice(None)])

            # slice first dimension
            assert_array_equal(A[:10], T[:10])
            assert_array_equal(A[:10], T[:10])
            assert_array_equal(A[10:20], T[10:20])
            assert_array_equal(A[-10:], T[-10:])
            assert_array_equal(A[:10, :], T[:10, :])
            assert_array_equal(A[10:20, :], T[10:20, :])
            assert_array_equal(A[-10:, :], T[-10:, :])
            assert_array_equal(A[:10, ...], T[:10, ...])
            assert_array_equal(A[10:20, ...], T[10:20, ...])
            assert_array_equal(A[-10:, ...], T[-10:, ...])
            assert_array_equal(A[:10, :, ...], T[:10, :, ...])
            assert_array_equal(A[10:20, :, ...], T[10:20, :, ...])
            assert_array_equal(A[-10:, :, ...], T[-10:, :, ...])

            # slice second dimension
            assert_array_equal(A[:, :2], T[:, :2])
            assert_array_equal(A[:, 2:4], T[:, 2:4])
            assert_array_equal(A[:, -2:], T[:, -2:])
            assert_array_equal(A[..., :2], T[..., :2])
            assert_array_equal(A[..., 2:4], T[..., 2:4])
            assert_array_equal(A[..., -2:], T[..., -2:])
            assert_array_equal(A[:, ..., :2], T[:, ..., :2])
            assert_array_equal(A[:, ..., 2:4], T[:, ..., 2:4])
            assert_array_equal(A[:, ..., -2:], T[:, ..., -2:])

            # slice both dimensions
            assert_array_equal(A[:10, :2], T[:10, :2])
            assert_array_equal(A[10:20, 2:4], T[10:20, 2:4])
            assert_array_equal(A[-10:, -2:], T[-10:, -2:])

            # slice across tile boundries
            assert_array_equal(A[:110], T[:110])
            assert_array_equal(A[190:310], T[190:310])
            assert_array_equal(A[-110:], T[-110:])
            assert_array_equal(A[:110, :], T[:110, :])
            assert_array_equal(A[190:310, :], T[190:310, :])
            assert_array_equal(A[-110:, :], T[-110:, :])
            assert_array_equal(A[:, :3], T[:, :3])
            assert_array_equal(A[:, 3:7], T[:, 3:7])
            assert_array_equal(A[:, -3:], T[:, -3:])
            assert_array_equal(A[:110, :3], T[:110, :3])
            assert_array_equal(A[190:310, 3:7], T[190:310, 3:7])
            assert_array_equal(A[-110:, -3:], T[-110:, -3:])

            # single row/col/item
            assert_array_equal(A[0], T[0])
            assert_array_equal(A[-1], T[-1])
            assert_array_equal(A[:, 0], T[:, 0])
            assert_array_equal(A[:, -1], T[:, -1])
            self.assertEqual(A[0, 0], T[0, 0])
            self.assertEqual(A[-1, -1], T[-1, -1])

            # too many indices
            with self.assertRaises(IndexError):
                T[:, :, :]
            with self.assertRaises(IndexError):
                T[0, :, :]
            with self.assertRaises(IndexError):
                T[:, 0, :]
            with self.assertRaises(IndexError):
                T[:, :, 0]
            with self.assertRaises(IndexError):
                T[0, 0, 0]

            # only single ellipsis allowed
            with self.assertRaises(IndexError):
                T[..., ...]

        with tiledb.DenseArray(self.path("foo"), mode="w") as T:
            # check partial assignment
            B = np.arange(10000, 20000).reshape((1000, 10))
            T[190:310, 3:7] = B[190:310, 3:7]

        with tiledb.DenseArray(self.path("foo"), mode="r") as T:
            assert_array_equal(A[:190], T[:190])
            assert_array_equal(A[:, :3], T[:, :3])
            assert_array_equal(B[190:310, 3:7], T[190:310, 3:7])
            assert_array_equal(A[310:], T[310:])
            assert_array_equal(A[:, 7:], T[:, 7:])

    @pytest.mark.skipif(
        not (sys.platform == "win32" and tiledb.libtiledb.version() >= (2, 3, 0)),
        reason="Shared network drive only on Win32",
    )
    def test_array_1d_shared_drive(self):
        A = np.zeros(50)

        dom = tiledb.Domain(tiledb.Dim(domain=(0, 49), tile=50))
        att = tiledb.Attr(dtype=A.dtype)
        schema = tiledb.ArraySchema(dom, (att,))
        uri = self.path("foo", shared=True)

        tiledb.DenseArray.create(uri, schema)

        with tiledb.DenseArray(uri, mode="w") as T:
            T[:] = A

        with tiledb.DenseArray(uri, mode="r") as T:
            assert_array_equal(A, T[:])

        with tiledb.DenseArray(uri, mode="w") as T:
            value = -1, 3, 10
            A[0], A[1], A[3] = value
            T[0], T[1], T[3] = value
        with tiledb.DenseArray(uri, mode="r") as T:
            assert_array_equal(A, T[:])

        for value in (-1, 3, 10):
            with tiledb.DenseArray(uri, mode="w") as T:
                A[5:25] = value
                T[5:25] = value
            with tiledb.DenseArray(uri, mode="r") as T:
                assert_array_equal(A, T[:])
            with tiledb.DenseArray(uri, mode="w") as T:
                A[:] = value
                T[:] = value
            with tiledb.DenseArray(uri, mode="r") as T:
                assert_array_equal(A, T[:])

    def test_fixed_string(self):
        a = np.array(["ab", "cd", "ef", "gh", "ij", "kl", "", "op"], dtype="|S2")
        with tiledb.from_numpy(self.path("fixed_string"), a) as T:
            with tiledb.open(self.path("fixed_string")) as R:
                self.assertEqual(T.dtype, R.dtype)
                self.assertEqual(R.attr(0).ncells, 2)
                assert_array_equal(T, R)

    def test_ncell_int(self):
        a = np.array([(1, 2), (3, 4), (5, 6)], dtype=[("", np.int16), ("", np.int16)])
        with tiledb.from_numpy(self.path("ncell_int16"), a) as T:
            with tiledb.open(self.path("ncell_int16")) as R:
                self.assertEqual(T.dtype, R.dtype)
                self.assertEqual(R.attr(0).ncells, 2)
                assert_array_equal(T, R)
                assert_array_equal(T, R.multi_index[0:2][""])

    def test_open_with_timestamp(self):
        A = np.zeros(3)

        dom = tiledb.Domain(tiledb.Dim(domain=(0, 2), tile=3, dtype=np.int64))
        att = tiledb.Attr(dtype=A.dtype)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,))
        tiledb.DenseArray.create(self.path("foo"), schema)

        # write
        with tiledb.DenseArray(self.path("foo"), mode="w") as T:
            T[:] = A

        read1_timestamp = -1
        with tiledb.DenseArray(self.path("foo"), mode="r") as T:
            read1_timestamp = T.timestamp_range
            self.assertEqual(T[0], 0)
            self.assertEqual(T[1], 0)
            self.assertEqual(T[2], 0)

        # sleep 200ms and write
        time.sleep(0.2)
        with tiledb.DenseArray(self.path("foo"), mode="w") as T:
            T[0:1] = 1

        read2_timestamp = -1
        with tiledb.DenseArray(self.path("foo"), mode="r") as T:
            read2_timestamp = T.timestamp_range
            self.assertTrue(read2_timestamp > read1_timestamp)

        # sleep 200ms and write
        time.sleep(0.2)
        with tiledb.DenseArray(self.path("foo"), mode="w") as T:
            T[1:2] = 2

        read3_timestamp = -1
        with tiledb.DenseArray(self.path("foo"), mode="r") as T:
            read3_timestamp = T.timestamp_range
            self.assertTrue(read3_timestamp > read2_timestamp > read1_timestamp)

        # read at first timestamp
        with tiledb.DenseArray(
            self.path("foo"), timestamp=read1_timestamp, mode="r"
        ) as T:
            self.assertEqual(T[0], 0)
            self.assertEqual(T[1], 0)
            self.assertEqual(T[2], 0)

        # read at second timestamp
        with tiledb.DenseArray(
            self.path("foo"), timestamp=read2_timestamp, mode="r"
        ) as T:
            self.assertEqual(T[0], 1)
            self.assertEqual(T[1], 0)
            self.assertEqual(T[2], 0)

        # read at third timestamp
        with tiledb.DenseArray(
            self.path("foo"), timestamp=read3_timestamp, mode="r"
        ) as T:
            self.assertEqual(T[0], 1)
            self.assertEqual(T[1], 2)
            self.assertEqual(T[2], 0)

    def test_open_timestamp_range(self):
        A = np.zeros(3)
        path = self.path("open_timestamp_range")

        dom = tiledb.Domain(tiledb.Dim(domain=(0, 2), tile=3, dtype=np.int64))
        att = tiledb.Attr(dtype=A.dtype)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,))
        tiledb.DenseArray.create(path, schema)

        # write
        with tiledb.DenseArray(path, timestamp=1, mode="w") as T:
            T[:] = A * 1
        with tiledb.DenseArray(path, timestamp=2, mode="w") as T:
            T[:] = A * 2
        with tiledb.DenseArray(path, timestamp=3, mode="w") as T:
            T[:] = A * 3
        with tiledb.DenseArray(path, timestamp=4, mode="w") as T:
            T[:] = A * 4

        def assert_ts(timestamp, result):
            with tiledb.DenseArray(path, mode="r", timestamp=timestamp) as T:
                assert_array_equal(T, result)

        assert_ts(0, A * np.nan)
        assert_ts(1, A * 1)
        assert_ts(2, A * 2)
        assert_ts(3, A * 3)
        assert_ts((1, 2), A * 2)
        assert_ts((0, 3), A * 3)
        assert_ts((1, 3), A * 3)
        assert_ts((2, 3), A * 3)
        assert_ts((2, 4), A * 3)
        assert_ts((None, 2), A * 2)
        assert_ts((None, 3), A * 3)
        assert_ts((2, None), A * 3)
        assert_ts((3, None), A * 3)
        assert_ts((3, None), A * 3)

    def test_open_attr(self):
        uri = self.path("test_open_attr")
        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(name="dim0", dtype=np.uint32, domain=(1, 4))
            ),
            attrs=(
                tiledb.Attr(name="x", dtype=np.int32),
                tiledb.Attr(name="y", dtype=np.int32),
            ),
        )
        tiledb.Array.create(uri, schema)

        with tiledb.open(uri, mode="w") as A:
            A[:] = {"x": np.array((1, 2, 3, 4)), "y": np.array((5, 6, 7, 8))}

        with self.assertRaises(KeyError):
            tiledb.open(uri, attr="z")

        with self.assertRaises(KeyError):
            tiledb.open(uri, attr="dim0")

        with tiledb.open(uri, attr="x") as A:
            assert_array_equal(A[:], np.array((1, 2, 3, 4)))
            assert list(A.multi_index[:].keys()) == ["x"]

    def test_ncell_attributes(self):
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=10, dtype=int))
        attr = tiledb.Attr(dtype=[("", np.int32), ("", np.int32), ("", np.int32)])
        schema = tiledb.ArraySchema(domain=dom, attrs=(attr,))
        tiledb.DenseArray.create(self.path("foo"), schema)

        A = np.ones((10,), dtype=[("", np.int32), ("", np.int32), ("", np.int32)])
        self.assertEqual(A.dtype, attr.dtype)

        with tiledb.DenseArray(self.path("foo"), mode="w") as T:
            T[:] = A
        with tiledb.DenseArray(self.path("foo"), mode="r") as T:
            assert_array_equal(A, T[:])
            assert_array_equal(A[:5], T[:5])

    def test_complex_attributes(self):
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=10, dtype=int))
        attr = tiledb.Attr(dtype=np.complex64)
        schema = tiledb.ArraySchema(domain=dom, attrs=(attr,))
        tiledb.DenseArray.create(self.path("foo"), schema)

        A = np.random.rand(20).astype(np.float32).view(dtype=np.complex64)

        self.assertEqual(schema, tiledb.schema_like(A, dim_dtype=int))
        self.assertEqual(A.dtype, attr.dtype)

        with tiledb.DenseArray(self.path("foo"), mode="w") as T:
            T[:] = A

        with tiledb.DenseArray(self.path("foo"), mode="r") as T:
            assert_array_equal(A, T[:])
            assert_array_equal(A[:5], T[:5])

    def test_multiple_attributes(self):
        dom = tiledb.Domain(
            tiledb.Dim(domain=(0, 1), tile=1, dtype=np.int64),
            tiledb.Dim(domain=(0, 3), tile=4, dtype=np.int64),
        )
        attr_int = tiledb.Attr("ints", dtype=int)
        attr_float = tiledb.Attr("floats", dtype=float)
        schema = tiledb.ArraySchema(domain=dom, attrs=(attr_int, attr_float))
        tiledb.DenseArray.create(self.path("foo"), schema)

        V_ints = np.array([[0, 1, 2, 3], [4, 6, 7, 5]])
        V_floats = np.array([[0.0, 1.0, 2.0, 3.0], [4.0, 6.0, 7.0, 5.0]])

        V = {"ints": V_ints, "floats": V_floats}
        with tiledb.DenseArray(self.path("foo"), mode="w") as T:
            T[:] = V

        # check setting attribute in different order from Attr definition
        #   https://github.com/TileDB-Inc/TileDB-Py/issues/299
        V2 = {"floats": V_floats, "ints": V_ints}
        with tiledb.DenseArray(self.path("foo"), mode="w") as T:
            T[:] = V

        with tiledb.DenseArray(self.path("foo"), mode="r") as T:
            R = T[:]
            assert_array_equal(V["ints"], R["ints"])
            assert_array_equal(V["floats"], R["floats"])

            R = T.query(attrs=("ints",))[1:3]
            assert_array_equal(V["ints"][1:3], R["ints"])

            R = T.query(attrs=("floats",), order="F")[:]
            self.assertTrue(R["floats"].flags.f_contiguous)

            R = T.query(attrs=("ints",), coords=True)[0, 0:3]
            self.assertTrue("__dim_0" in R)
            self.assertTrue("__dim_1" in R)
            assert_array_equal(R["__dim_0"], np.array([0, 0, 0]))
            assert_array_equal(R["__dim_1"], np.array([0, 1, 2]))

            # Global order returns results as a linear buffer
            R = T.query(attrs=("ints",), order="G")[:]
            self.assertEqual(R["ints"].shape, (8,))

            with self.assertRaises(tiledb.TileDBError):
                T.query(attrs=("unknown",))[:]

        with tiledb.DenseArray(self.path("foo"), mode="w") as T:
            # check error ncells length
            V["ints"] = V["ints"][1:2].copy()
            with self.assertRaises(tiledb.TileDBError):
                T[:] = V

            # check error attribute does not exist
            V["foo"] = V["ints"].astype(np.int8)
            with self.assertRaises(tiledb.TileDBError):
                T[:] = V

    def test_array_2d_s1(self):
        # This array is currently read back with dtype object
        A = np.array([["A", "B"], ["C", ""]], dtype="S")

        uri = self.path()
        dom = tiledb.Domain(
            tiledb.Dim(name="rows", domain=(0, 1), tile=2, dtype=np.int64),
            tiledb.Dim(name="cols", domain=(0, 1), tile=2, dtype=np.int64),
        )

        schema = tiledb.ArraySchema(
            domain=dom, sparse=False, attrs=[tiledb.Attr(name="a", dtype="S")]
        )

        tiledb.DenseArray.create(uri, schema)

        with tiledb.DenseArray(uri, mode="w") as T:
            T[...] = A

        with tiledb.DenseArray(uri) as T:
            assert_array_equal(A, T)

            res = T.multi_index[(0, 1), (0, 1)]["a"]
            assert_array_equal(A, res)

    def test_nd_roundtrip(self):
        dim_set = np.int64([3 + x % 2 for x in range(2, 12)])
        for i, last in enumerate(range(2, len(dim_set))):
            dims = dim_set[:last]
            data = np.random.rand(*dims).astype("int32")
            with tiledb.from_numpy(self.path(f"nd_roundtrip{i}"), data) as A:
                assert_array_equal(data, A[:])

    def test_array_2d_s3_mixed(self):
        # This array is currently read back with dtype object
        A = np.array([["AAA", "B"], ["AB", "C"]], dtype="S3")

        uri = self.path()
        dom = tiledb.Domain(
            tiledb.Dim(name="rows", domain=(0, 1), tile=2, dtype=np.int64),
            tiledb.Dim(name="cols", domain=(0, 1), tile=2, dtype=np.int64),
        )

        schema = tiledb.ArraySchema(
            domain=dom, sparse=False, attrs=[tiledb.Attr(name="a", dtype="S3")]
        )

        tiledb.DenseArray.create(uri, schema)

        with tiledb.DenseArray(uri, mode="w") as T:
            T[...] = A

        with tiledb.DenseArray(uri) as T:
            assert_array_equal(A, T)

            res = T.multi_index[(0, 1), (0, 1)]["a"]
            assert_array_equal(A, res)

    def test_incomplete_dense(self):
        path = self.path("incomplete_dense")
        # create 10 MB array
        data = np.arange(1310720, dtype=np.int64)
        # if `tile` is not set, it defaults to the full array and we
        # only read 8 bytes at a time.
        use_tile = 131072
        # use_tile = None
        with tiledb.from_numpy(path, data, tile=use_tile) as A:
            pass

        # create context with 1 MB memory budget (2 MB total, 1 MB usable)
        config = tiledb.Config(
            {"sm.memory_budget": 2 * 1024**2, "py.init_buffer_bytes": 1024**2}
        )
        self.assertEqual(config["py.init_buffer_bytes"], str(1024**2))
        # TODO would be good to check repeat count here. Not currently exposed by retry loop.
        with tiledb.DenseArray(path, ctx=tiledb.Ctx(config)) as A:
            res_mr = A.multi_index[slice(0, len(data) - 1)]
            assert_array_equal(res_mr[""], data)
            res_idx = A[:]
            assert_array_equal(res_idx, data)

            if has_pandas():
                df = A.df[:]
                assert_array_equal(df[""], data)

    def test_written_fragment_info(self):
        uri = self.path("test_written_fragment_info")

        dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=10, dtype=np.int64))
        att = tiledb.Attr(dtype=np.int64)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,))
        tiledb.DenseArray.create(uri, schema)

        with tiledb.DenseArray(uri, mode="w") as T:
            T[:] = np.arange(0, 10, dtype=np.int64)

            self.assertTrue(T.last_write_info is not None)
            self.assertTrue(len(T.last_write_info.keys()) == 1)
            t_w1, t_w2 = list(T.last_write_info.values())[0]
            self.assertTrue(t_w1 > 0)
            self.assertTrue(t_w2 > 0)

    def test_missing_schema_error(self):
        uri = self.path("test_missing_schema_error")

        dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=10, dtype=np.int64))
        att = tiledb.Attr(dtype=np.int64)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,))
        tiledb.DenseArray.create(uri, schema)

        with tiledb.DenseArray(uri, mode="w") as T:
            T[:] = np.arange(0, 10, dtype=np.int64)

        if tiledb.libtiledb.version() < (2, 4):
            tiledb.VFS().remove_file(os.path.join(uri, "__array_schema.tdb"))
        else:
            tiledb.VFS().remove_dir(os.path.join(uri, "__schema"))

        # new ctx is required running against S3 because otherwise the schema
        # will simply be read from the cache.
        with tiledb.scope_ctx():
            with self.assertRaises(tiledb.TileDBError):
                tiledb.DenseArray(uri)

    @pytest.mark.xfail(
        tiledb.libtiledb.version() >= (2, 5),
        reason="Skip sparse_write_to_dense with libtiledb 2.5+",
    )
    def test_sparse_write_to_dense(self):
        class AssignAndCheck:
            def __init__(self, outer, *shape):
                self.outer = outer
                self.shape = shape

            def __setitem__(self, s, v):
                A = np.random.rand(*self.shape)

                uri = self.outer.path(
                    f"sparse_write_to_dense{random.randint(0,np.uint64(-1))}"
                )

                tiledb.from_numpy(uri, A).close()
                with tiledb.open(uri, "w") as B:
                    B[s] = v

                A[s] = v
                with tiledb.open(uri) as B:
                    assert_array_equal(A, B[:])

        D = AssignAndCheck(self, 5, 5)
        with pytest.warns(
            DeprecationWarning, match="Sparse writes to dense arrays is deprecated"
        ):
            D[np.array([1, 2]), np.array([0, 0])] = np.array([0, 2])

    def test_reopen_dense_array(self):
        uri = self.path("test_reopen_dense_array")

        dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=10, dtype=np.int64))
        att = tiledb.Attr(dtype=np.int64)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,))
        tiledb.DenseArray.create(uri, schema)

        data = np.arange(0, 10, dtype=np.int64)

        with tiledb.DenseArray(uri, mode="w", timestamp=1) as T:
            T[:] = data

        with tiledb.DenseArray(uri, mode="w", timestamp=2) as T:
            T[:] = data * 2

        T = tiledb.DenseArray(uri, mode="r", timestamp=1)
        assert_array_equal(T[:], data)

        T.reopen()
        assert_array_equal(T[:], data * 2)

        T.close()


class TestVarlen(DiskTestCase):
    def test_varlen_write_bytes(self):
        A = np.array(
            [
                "aa",
                "bbb",
                "ccccc",
                "ddddddddddddddddddddd",
                "ee",
                "ffffff",
                "g",
                "hhhhhhhhhh",
            ],
            dtype=bytes,
        )

        # basic write
        dom = tiledb.Domain(tiledb.Dim(domain=(1, len(A)), tile=len(A)))
        att = tiledb.Attr(dtype=np.bytes_)

        schema = tiledb.ArraySchema(dom, (att,))

        tiledb.DenseArray.create(self.path("foo"), schema)
        with tiledb.DenseArray(self.path("foo"), mode="w") as T:
            T[:] = A

        with tiledb.DenseArray(self.path("foo"), mode="r") as T:
            assert_array_equal(A[:], T[:])

            assert_array_equal(A, T.multi_index[1 : len(A)][""])

    def test_varlen_sparse_all_empty_strings(self):
        # this test addresses a fix for specific need for reads on a
        # large existing array, see
        #   https://github.com/TileDB-Inc/TileDB-Py/pull/475
        # we currently have to write a placeholder at the end to
        # avoid zero-length cell error
        # TODO: follow-up with improved testing for empty var-length/strings
        A = np.array(["", "", "", "", "", "\x00"], dtype=object)
        dim_len = len(A)
        uri = self.path("varlen_all_empty_strings")

        dom = tiledb.Domain(tiledb.Dim(domain=(1, dim_len), tile=dim_len))
        att = tiledb.Attr(name="a1", dtype=np.str_, var=True)

        schema = tiledb.ArraySchema(dom, (att,), sparse=True)

        tiledb.Array.create(uri, schema)

        with tiledb.open(uri, mode="w") as T:
            T[np.arange(1, dim_len + 1)] = {"a1": A}

        with tiledb.open(uri, mode="r") as T:
            # check interior range
            assert_array_equal(A[1:-1], T[2:-1]["a1"])
            assert_array_equal(A[1:-1], T.multi_index[2 : dim_len - 1]["a1"])

    def test_varlen_write_unicode(self):
        A = np.array(
            [
                "aa",
                "bbb",
                "ccccc",
                "ddddddddddddddddddddd",
                "ee",
                "ffffff",
                "g",
                "",
                "hhhhhhhhhh",
            ],
            dtype=np.unicode_,
        )

        # basic write
        dom = tiledb.Domain(tiledb.Dim(domain=(1, len(A)), tile=len(A)))
        att = tiledb.Attr(dtype=np.unicode_, var=True)

        schema = tiledb.ArraySchema(dom, (att,))

        tiledb.DenseArray.create(self.path("foo"), schema)
        with tiledb.DenseArray(self.path("foo"), mode="w") as T:
            T[:] = A

        with tiledb.DenseArray(self.path("foo"), mode="r") as T:
            assert_array_equal(A[:], T[:])

    def test_varlen_write_floats(self):
        # Generates 8 variable-length float64 subarrays (subarray len and content are randomized)
        A = np.array(
            [np.random.rand(x) for x in np.random.randint(1, 12, 8)], dtype=object
        )

        # basic write
        dom = tiledb.Domain(tiledb.Dim(domain=(1, len(A)), tile=len(A)))
        att = tiledb.Attr(dtype=np.float64, var=True)

        schema = tiledb.ArraySchema(dom, (att,))
        tiledb.DenseArray.create(self.path("foo"), schema)

        with tiledb.DenseArray(self.path("foo"), mode="w") as T:
            T[:] = A

        with tiledb.DenseArray(self.path("foo"), mode="r") as T:
            T_ = T[:]
            # TODO/note: the return is a 0-element array.
            assert_array_equal(A[0], T[1][()])
            assert_array_equal(A[-1], T[-1][()])
            self.assertEqual(len(A), len(T_))
            # can't use assert_array_equal w/ object array
            self.assertTrue(all(np.array_equal(x, A[i]) for i, x in enumerate(T_)))

    def test_varlen_write_floats_2d(self):
        A = np.array(
            [np.random.rand(x) for x in np.arange(1, 10)], dtype=object
        ).reshape(3, 3)

        # basic write
        dom = tiledb.Domain(
            tiledb.Dim(domain=(1, 3), tile=len(A)),
            tiledb.Dim(domain=(1, 3), tile=len(A)),
        )
        att = tiledb.Attr(dtype=np.float64, var=True)

        schema = tiledb.ArraySchema(dom, (att,))
        tiledb.DenseArray.create(self.path("foo"), schema)

        with tiledb.DenseArray(self.path("foo"), mode="w") as T:
            T[:] = A

        with tiledb.DenseArray(self.path("foo"), mode="r") as T:
            T_ = T[:]
            self.assertEqual(len(A), len(T_))
            # can't use assert_array_equal w/ object array
            self.assertTrue(
                np.all(
                    [np.array_equal(A.flat[i], T[:].flat[i]) for i in np.arange(0, 9)]
                )
            )

    def test_varlen_write_int_subarray(self):
        A = np.array(
            list(
                map(
                    lambda x: np.array(x, dtype=np.uint64),
                    [np.arange(i, 2 * i + 1) for i in np.arange(0, 16)],
                )
            ),
            dtype="O",
        ).reshape(4, 4)

        uri = self.path("test_varlen_write_int_subarray")

        dom = tiledb.Domain(
            tiledb.Dim(domain=(0, 3), tile=len(A)),
            tiledb.Dim(domain=(0, 3), tile=len(A)),
        )
        att = tiledb.Attr(dtype=np.uint64, var=True)
        schema = tiledb.ArraySchema(dom, (att,))

        tiledb.DenseArray.create(uri, schema)

        # NumPy forces single-element object arrays into a contiguous layout
        #       so we alternate the size to get a consistent baseline array.
        A_onestwos = np.array(
            list(
                map(
                    lambda x: np.array(x, dtype=np.uint64),
                    list([(1,) if x % 2 == 0 else (1, 2) for x in range(16)]),
                )
            ),
            dtype=np.dtype("O"),
        ).reshape(4, 4)

        with tiledb.open(uri, "w") as T:
            T[:] = A_onestwos

        with tiledb.open(uri, "w") as T:
            T[1:3, 1:3] = A[1:3, 1:3]

        A_assigned = A_onestwos.copy()
        A_assigned[1:3, 1:3] = A[1:3, 1:3]

        with tiledb.open(uri) as T:
            assert_subarrays_equal(A_assigned, T[:])

    def test_varlen_write_fixedbytes(self):
        # The actual dtype of this array is 'S21'
        A = np.array(
            [
                "aa",
                "bbb",
                "ccccc",
                "ddddddddddddddddddddd",
                "ee",
                "ffffff",
                "g",
                "hhhhhhhhhh",
            ],
            dtype=np.dtype("S"),
        )

        # basic write
        dom = tiledb.Domain(tiledb.Dim(domain=(1, len(A)), tile=len(A)))
        att = tiledb.Attr(dtype=np.bytes_)

        schema = tiledb.ArraySchema(dom, (att,))

        tiledb.DenseArray.create(self.path("foo"), schema)
        with tiledb.DenseArray(self.path("foo"), mode="w") as T:
            T[:] = A

        with tiledb.DenseArray(self.path("foo"), mode="r") as T:
            assert_array_equal(A[:], T[:])

    def test_varlen_write_fixedunicode(self):
        A = np.array(
            [
                "aa",
                "bbb",
                "ccccc",
                "ddddddddddddddddddddd",
                "ee",
                "ffffff",
                "",
                "g",
                "hhhhhhhhhh",
            ],
            dtype=np.dtype("U"),
        )

        # basic write
        dom = tiledb.Domain(tiledb.Dim(domain=(1, len(A)), tile=len(A)))
        att = tiledb.Attr(dtype=np.unicode_)

        schema = tiledb.ArraySchema(dom, (att,))

        tiledb.DenseArray.create(self.path("foo"), schema)
        with tiledb.DenseArray(self.path("foo"), mode="w") as T:
            T[:] = A

        with tiledb.DenseArray(self.path("foo"), mode="r") as T:
            assert_array_equal(A[:], T[:])

    def test_varlen_write_ints(self):
        A = np.array(
            [
                np.uint64(np.random.randint(0, pow(10, 6), x))
                for x in np.random.randint(1, 12, 8)
            ],
            dtype=object,
        )

        # basic write
        dom = tiledb.Domain(tiledb.Dim(domain=(1, len(A)), tile=len(A)))
        att = tiledb.Attr(dtype=np.int64, var=True)

        schema = tiledb.ArraySchema(dom, (att,))

        tiledb.DenseArray.create(self.path("foo"), schema)
        with tiledb.DenseArray(self.path("foo"), mode="w") as T:
            T[:] = A

        with tiledb.DenseArray(self.path("foo"), mode="r") as T:
            T_ = T[:]
            self.assertEqual(len(A), len(T))
            # can't use assert_array_equal w/ object array
            self.assertTrue(all(np.array_equal(x, A[i]) for i, x in enumerate(T_)))

    def test_varlen_wrong_domain(self):
        A = np.array(
            [
                "aa",
                "bbb",
                "ccccc",
                "ddddddddddddddddddddd",
                "ee",
                "ffffff",
                "g",
                "hhhhhhhhhh",
            ]
        )
        dom = tiledb.Domain(tiledb.Dim(domain=(1, 3), tile=3))
        att = tiledb.Attr(dtype=np.bytes_)

        schema = tiledb.ArraySchema(dom, (att,))

        tiledb.DenseArray.create(self.path("foo"), schema)
        with tiledb.DenseArray(self.path("foo"), mode="w") as T:
            with self.assertRaises(tiledb.TileDBError):
                T[:] = A

    def test_array_varlen_mismatched(self):
        # Test that we raise a TypeError when passing a heterogeneous object array.
        A = np.array([b"aa", b"bbb", b"cccc", np.uint64([1, 3, 4])], dtype=object)

        dom = tiledb.Domain(tiledb.Dim(domain=(0, 3), tile=4))
        att = tiledb.Attr(dtype=np.bytes_, var=True)

        schema = tiledb.ArraySchema(dom, (att,))

        tiledb.DenseArray.create(self.path("foo"), schema)
        with tiledb.DenseArray(self.path("foo"), mode="w") as T:
            with self.assertRaises(TypeError):
                T[:] = A

    def test_array_varlen_2d_s_fixed(self):
        A = np.array(
            [["AAAAAAAAAa", "BBB"], ["ACCC", "BBBCBCBCBCCCBBCBCBCCBC"]], dtype="S"
        )

        uri = self.path("varlen_2d_s_fixed")
        dom = tiledb.Domain(
            tiledb.Dim(name="rows", domain=(0, 1), tile=2, dtype=np.int64),
            tiledb.Dim(name="cols", domain=(0, 1), tile=2, dtype=np.int64),
        )

        schema = tiledb.ArraySchema(
            domain=dom, sparse=False, attrs=[tiledb.Attr(name="a", dtype="S", var=True)]
        )

        tiledb.DenseArray.create(uri, schema)

        with tiledb.DenseArray(uri, mode="w") as T:
            T[...] = A

        with tiledb.DenseArray(uri) as T:
            assert_array_equal(A, T)


class TestSparseArray(DiskTestCase):
    @pytest.mark.xfail
    def test_simple_1d_sparse_vector(self):
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 3), tile=4, dtype=int))
        att = tiledb.Attr(dtype=int)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True)
        tiledb.SparseArray.create(self.path("foo"), schema)

        values = np.array([3, 4])
        with tiledb.SparseArray(self.path("foo"), mode="w") as T:
            T[[1, 2]] = values

        with tiledb.SparseArray(self.path("foo"), mode="r") as T:
            assert_array_equal(T[[1, 2]], values)

    @pytest.mark.xfail
    def test_simple_2d_sparse_vector(self):
        attr = tiledb.Attr(ctx, dtype=float)
        dom = tiledb.Domain(
            tiledb.Dim(ctx, domain=(0, 3), tile=4, dtype=int),
            tiledb.Dim(ctx, domain=(0, 3), tile=4, dtype=int),
        )
        schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(attr,), sparse=True)
        tiledb.SparseArray.create(self.path("foo"), schema)

        values = np.array([3, 4], dtype=float)
        with tiledb.SparseArray(ctx, self.path("foo"), mode="w") as T:
            T[[1, 2], [1, 2]] = values

        with tiledb.SparseArray(ctx, self.path("foo"), mode="r") as T:
            assert_array_equal(T[[1, 2], [1, 2]], values)

    @pytest.mark.xfail
    def test_simple3d_sparse_vector(self):
        dom = tiledb.Domain(
            ctx,
            tiledb.Dim(ctx, "x", domain=(0, 3), tile=4, dtype=int),
            tiledb.Dim(ctx, "y", domain=(0, 3), tile=4, dtype=int),
            tiledb.Dim(ctx, "z", domain=(0, 3), tile=4, dtype=int),
        )
        attr = tiledb.Attr(ctx, dtype=float)
        schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(attr,), sparse=True)
        tiledb.SparseArray.create(self.path("foo"), schema)

        values = np.array([3, 4], dtype=float)
        with tiledb.SparseArray(ctx, self.path("foo"), mode="w") as T:
            T[[1, 2], [1, 2], [1, 2]] = values

        with tiledb.SparseArray(ctx, self.path("foo"), mode="r") as T:
            assert_array_equal(T[[1, 2], [1, 2], [1, 2]], values)

    @pytest.mark.xfail
    def test_sparse_ordered_fp_domain(self):
        dom = tiledb.Domain(tiledb.Dim("x", domain=(0.0, 10.0), tile=2.0, dtype=float))
        attr = tiledb.Attr(dtype=float)
        attr = tiledb.Attr(dtype=float)
        schema = tiledb.ArraySchema(domain=dom, attrs=(attr,), sparse=True)
        tiledb.SparseArray.create(self.path("foo"), schema)

        values = np.array([3.3, 2.7])
        with tiledb.SparseArray(self.path("foo"), mode="w") as T:
            T[[2.5, 4.2]] = values
        with tiledb.SparseArray(self.path("foo"), mode="r") as T:
            assert_array_equal(T[[2.5, 4.2]], values)

    @pytest.mark.xfail
    def test_sparse_unordered_fp_domain(self):
        dom = tiledb.Domain(tiledb.Dim("x", domain=(0.0, 10.0), tile=2.0, dtype=float))
        attr = tiledb.Attr(dtype=float)
        schema = tiledb.ArraySchema(domain=dom, attrs=(attr,), sparse=True)
        tiledb.SparseArray.create(self.path("foo"), schema)
        values = np.array([3.3, 2.7])
        with tiledb.SparseArray(self.path("foo"), mode="w") as T:
            T[[4.2, 2.5]] = values

        with tiledb.SparseArray(self.path("foo"), mode="r") as T:
            assert_array_equal(T[[2.5, 4.2]], values[::-1])

    @pytest.mark.xfail
    def test_multiple_attributes(self):
        uri = self.path()

        dom = tiledb.Domain(
            tiledb.Dim(domain=(1, 10), tile=10, dtype=int),
            tiledb.Dim(domain=(1, 10), tile=10, dtype=int),
        )
        attr_int = tiledb.Attr("ints", dtype=int)
        attr_float = tiledb.Attr("floats", dtype="float")
        schema = tiledb.ArraySchema(
            domain=dom, attrs=(attr_int, attr_float), sparse=True
        )
        tiledb.SparseArray.create(self.path("foo"), schema)

        I = np.array([1, 1, 1, 2, 3, 3, 3, 4])
        J = np.array([1, 2, 4, 3, 1, 6, 7, 5])

        V_ints = np.array([0, 1, 2, 3, 4, 6, 7, 5])
        V_floats = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 5.0])

        V = {"ints": V_ints, "floats": V_floats}
        with tiledb.SparseArray(uri, mode="w") as T:
            T[I, J] = V
        with tiledb.SparseArray(uri, mode="r") as T:
            R = T[I, J]
        assert_array_equal(V["ints"], R["ints"])
        assert_array_equal(V["floats"], R["floats"])

        # check error attribute does not exist
        # TODO: should this be an attribute error?
        with tiledb.SparseArray(uri, mode="w") as T:
            V["foo"] = V["ints"].astype(np.int8)
            with self.assertRaises(tiledb.TileDBError):
                T[I, J] = V

            # check error ncells length
            V["ints"] = V["ints"][1:2].copy()
            with self.assertRaises(AttributeError):
                T[I, J] = V

    def test_query_real_multi_index(self, sparse_cell_order):
        uri = self.path("query_real_multi_index")

        dom = tiledb.Domain(
            tiledb.Dim("x", domain=(-10.0, 10.0), tile=2.0, dtype=float)
        )
        attr = tiledb.Attr("a", dtype=np.float32)
        schema = tiledb.ArraySchema(
            domain=dom, attrs=(attr,), sparse=True, cell_order=sparse_cell_order
        )
        tiledb.SparseArray.create(uri, schema)

        values = np.array([3.3, 2.7])
        with tiledb.SparseArray(uri, mode="w") as T:
            T[[2.5, 4.2]] = values
        with tiledb.SparseArray(uri, mode="r") as T:
            assert_array_equal(
                T.query(coords=True).multi_index[-10.0 : np.nextafter(4.2, 0)]["a"],
                np.float32(3.3),
            )
            assert_array_equal(
                T.query(coords=True).multi_index[-10.0 : np.nextafter(4.2, 0)]["x"],
                np.float32([2.5]),
            )
            assert_array_equal(
                T.query(coords=False).multi_index[-10.0:5.0]["a"],
                np.float32([3.3, 2.7]),
            )
            self.assertTrue(
                "coords" not in T.query(coords=False).multi_index[-10.0:5.0]
            )

    @pytest.mark.skipif(not has_pandas(), reason="pandas not installed")
    @pytest.mark.parametrize("dtype", INTEGER_DTYPES)
    def test_sparse_index_dtypes(self, dtype):
        path = self.path()
        data = np.arange(0, 3).astype(dtype)

        schema = schema_from_dict(attrs={"attr": data}, dims={"d0": data})
        tiledb.SparseArray.create(path, schema)

        with tiledb.open(path, "w") as A:
            A[data] = data

        with tiledb.open(path) as B:
            assert_array_equal(B[:]["attr"], data)
            assert B[data[0]]["attr"] == data[0]
            assert B[data[1]]["attr"] == data[1]
            assert B.multi_index[data[0]]["attr"] == data[0]

    def test_query_real_exact(self, sparse_cell_order):
        """
        Test and demo of querying at floating point representable boundaries

        Concise representation of expected behavior:

        c0,c1,c2 = [3.0100000000000002, 3.0100000000000007, 3.010000000000001]
        values = [1,2,3]

        [c0:c0] -> [1]
        [c1:c1] -> [2]
        [c2:c2] -> [3]

        [c0:c1] -> [1,2]
        [c0:c2] -> [1,2,3]

        [c0 - nextafter(c0,0) : c0] -> [1]
        [c0 - nextafter(c0,0) : c0 - nextafter(c0,0)] -> []

        [c2:c2+nextafter(c2)] -> [3]
        [c2+nextafter(c2) : c2+nextafter(c2)] -> []

        """
        uri = self.path()

        dom = tiledb.Domain(
            tiledb.Dim("x", domain=(-10.0, 10.0), tile=2.0, dtype=float)
        )
        attr = tiledb.Attr("", dtype=np.float32)
        schema = tiledb.ArraySchema(
            domain=dom, attrs=(attr,), sparse=True, cell_order=sparse_cell_order
        )
        tiledb.SparseArray.create(uri, schema)

        c0 = np.nextafter(3.01, 4)  # smaller
        c1 = np.nextafter(c0, 4)
        c2 = np.nextafter(c1, 4)  # larger

        # for debugging use:
        # np.set_printoptions(precision=16, floatmode='maxprec')
        # print(c0,c1,c2)

        values = np.array([1, 2, 3])
        with tiledb.SparseArray(uri, mode="w") as T:
            T[[c0, c1, c2]] = values

        with tiledb.SparseArray(uri, mode="r") as T:
            for i, c in enumerate([c0, c1, c2]):
                assert_array_equal(T.query(coords=True).multi_index[c:c][""], values[i])
            # test (coord, coord + nextafter)
            c0_prev = np.nextafter(c0, 0)
            c2_next = np.nextafter(c2, 4)
            assert_array_equal(T.query(coords=True).multi_index[c0:c1][""], [1, 2])
            assert_array_equal(T.query(coords=True).multi_index[c0:c2][""], [1, 2, 3])
            assert_array_equal(T.query(coords=True).multi_index[c2:c2_next][""], 3)
            assert_array_equal(T.query(coords=True).multi_index[c0_prev:c0][""], 1)
            assert_array_equal(
                T.query(coords=True).multi_index[c0_prev:c0_prev][""], []
            )
            # test (coord + nextafter, coord + nextafter)
            assert_array_equal(
                T.query(coords=True).multi_index[c2_next:c2_next][""], np.array([])
            )
            # test (coord - nextafter, coord)
            assert_array_equal(
                T.query(coords=True).multi_index[c0:c1][""], values[[0, 1]]
            )
            # test (coord - nextafter, coord + nextafter)
            assert_array_equal(
                T.query(coords=True).multi_index[c0:c2][""], values[[0, 1, 2]]
            )

    def test_sparse_query_specified_dim_coords(self, sparse_cell_order):
        uri = self.path("sparse_query_specified_dim_coords")

        dom = tiledb.Domain(
            tiledb.Dim("i", domain=(1, 10), tile=1, dtype=int),
            tiledb.Dim("j", domain=(11, 20), tile=1, dtype=int),
        )
        att = tiledb.Attr("", dtype=int)
        schema = tiledb.ArraySchema(
            domain=dom, attrs=(att,), sparse=True, cell_order=sparse_cell_order
        )
        tiledb.SparseArray.create(uri, schema)

        i = np.array([1, 1, 1, 2, 3, 3, 3, 4])
        j = np.array([11, 12, 14, 13, 11, 16, 17, 15])

        with tiledb.SparseArray(uri, mode="w") as A:
            A[i, j] = np.array([0, 1, 2, 3, 4, 6, 7, 5])

        # data is returned in Hilbert order, so we need to check sorted
        with tiledb.SparseArray(uri, mode="r") as A:
            Ai = A.query(dims=["i"])[:]
            self.assertTrue("i" in Ai)
            self.assertFalse("j" in Ai)
            assert_unordered_equal(Ai["i"], i, sparse_cell_order == "hilbert")

            Aj = A.query(dims=["j"])[:]
            self.assertFalse("i" in Aj)
            self.assertTrue("j" in Aj)
            assert_unordered_equal(Aj["j"], j, sparse_cell_order == "hilbert")

            Aij = A.query(dims=["i", "j"])[:]
            self.assertTrue("i" in Aij)
            self.assertTrue("j" in Aij)
            assert_unordered_equal(Aij["i"], i, sparse_cell_order == "hilbert")
            assert_unordered_equal(Aij["j"], j, sparse_cell_order == "hilbert")

    def test_dense_query_specified_dim_coords(self):
        uri = self.path("dense_query_specified_dim_coords")

        dom = tiledb.Domain(
            tiledb.Dim("i", domain=(1, 3), tile=1, dtype=int),
            tiledb.Dim("j", domain=(4, 6), tile=1, dtype=int),
        )
        att = tiledb.Attr("", dtype=int)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=False)
        tiledb.DenseArray.create(uri, schema)

        with tiledb.DenseArray(uri, mode="w") as A:
            A[:, :] = np.arange(9)

        with tiledb.DenseArray(uri, mode="r") as A:
            i = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
            j = np.array([[4, 5, 6], [4, 5, 6], [4, 5, 6]])

            Ai = A.query(dims=["i"])[:]
            self.assertTrue("i" in Ai)
            self.assertFalse("j" in Ai)
            assert_array_equal(Ai["i"], i)

            Aj = A.query(dims=["j"])[:]
            self.assertFalse("i" in Aj)
            self.assertTrue("j" in Aj)
            assert_array_equal(Aj["j"], j)

            Aij = A.query(dims=["i", "j"])[:]
            self.assertTrue("i" in Aij)
            self.assertTrue("j" in Aij)
            assert_array_equal(Aij["i"], i)
            assert_array_equal(Aij["j"], j)

    def test_subarray(self, sparse_cell_order):
        dom = tiledb.Domain(tiledb.Dim("x", domain=(1, 10000), tile=100, dtype=int))
        att = tiledb.Attr("", dtype=float)
        schema = tiledb.ArraySchema(
            domain=dom, attrs=(att,), sparse=True, cell_order=sparse_cell_order
        )
        tiledb.SparseArray.create(self.path("foo"), schema)

        with tiledb.SparseArray(self.path("foo"), mode="r") as T:
            self.assertIsNone(T.nonempty_domain())

        with tiledb.SparseArray(self.path("foo"), mode="w") as T:
            T[[50, 60, 100]] = [1.0, 2.0, 3.0]

        with tiledb.SparseArray(self.path("foo"), mode="r") as T:
            self.assertEqual(((50, 100),), T.nonempty_domain())

            # stepped ranges are not supported
            with self.assertRaises(IndexError) as idxerr:
                T[40:61:5]
            assert str(idxerr.value) == "steps are not supported for sparse arrays"

            # retrieve just valid coordinates in subarray T[40:60]
            assert_array_equal(T[40:61]["x"], [50, 60])

            # TODO: dropping coords with one anon value returns just an array
            res = T.query(coords=False)[40:61]
            assert_array_equal(res[""], [1.0, 2.0])
            self.assertEqual(("coords" in res), False)

    def test_sparse_bytes(self, sparse_cell_order):
        dom = tiledb.Domain(tiledb.Dim("x", domain=(1, 10000), tile=100, dtype=int))
        att = tiledb.Attr("", var=True, dtype=np.bytes_)
        schema = tiledb.ArraySchema(
            domain=dom, attrs=(att,), sparse=True, cell_order=sparse_cell_order
        )
        tiledb.SparseArray.create(self.path("foo"), schema)

        with tiledb.SparseArray(self.path("foo"), mode="r") as T:
            self.assertIsNone(T.nonempty_domain())
        A = np.array(
            [b"aaa", b"bbbbbbbbbbbbbbbbbbbb", b"ccccccccccccccccccccccccc"],
            dtype=np.bytes_,
        )

        with tiledb.SparseArray(self.path("foo"), mode="w") as T:
            T[[50, 60, 100]] = A

        with tiledb.SparseArray(self.path("foo"), mode="r") as T:
            self.assertEqual(((50, 100),), T.nonempty_domain())

            # retrieve just valid coordinates in subarray T[40:60]
            assert_array_equal(T[40:61]["x"], [50, 60])

            # TODO: dropping coords with one anon value returns just an array
            res = T.query(coords=False)[40:61]
            assert_array_equal(res[""], A[0:2])
            self.assertEqual(("coords" in res), False)

            # empty sparse varlen result
            res = T[1000]
            assert_array_equal(res[""], np.array("", dtype="S1"))
            assert_array_equal(res["x"], np.array([], dtype=np.int64))

    def test_sparse_unicode(self, sparse_cell_order):
        dom = tiledb.Domain(tiledb.Dim("x", domain=(1, 10000), tile=100, dtype=int))
        att = tiledb.Attr("", var=True, dtype=np.unicode_)
        schema = tiledb.ArraySchema(
            domain=dom, attrs=(att,), sparse=True, cell_order=sparse_cell_order
        )
        tiledb.SparseArray.create(self.path("foo"), schema)

        with tiledb.SparseArray(self.path("foo"), mode="r") as T:
            self.assertIsNone(T.nonempty_domain())

        A = np_array = np.array(
            [
                "1234545lkjalsdfj",
                "mnopqrs",
                "ijkl",
                "gh",
                "abcdef",
                "aαbββcγγγdδδδδ",
                "aαbββc",
                "",
                "γγγdδδδδ",
            ],
            dtype=object,
        )

        with tiledb.SparseArray(self.path("foo"), mode="w") as T:
            T[[3, 4, 5, 6, 7, 50, 60, 70, 100]] = A

        with tiledb.SparseArray(self.path("foo"), mode="r") as T:
            self.assertEqual(((3, 100),), T.nonempty_domain())

            # retrieve just valid coordinates in subarray T[40:60]
            assert_array_equal(T[40:61]["x"], [50, 60])

            # TODO: dropping coords with one anon value returns just an array
            res = T.query(coords=False)[40:61]
            assert_array_equal(res[""], A[5:7])
            self.assertEqual(("coords" in res), False)

            # empty sparse varlen result
            res = T[1000]
            assert_array_equal(res[""], np.array("", dtype="U1"))
            assert_array_equal(res["x"], np.array([], dtype=np.int64))

    def test_sparse_query(self, sparse_cell_order):
        uri = self.path("test_sparse_query")
        dom = tiledb.Domain(
            tiledb.Dim("x", domain=(1, 10000), tile=100, dtype=np.float64)
        )

        att = tiledb.Attr("", dtype=float)
        schema = tiledb.ArraySchema(
            domain=dom, attrs=(att,), sparse=True, cell_order=sparse_cell_order
        )
        tiledb.SparseArray.create(uri, schema)

        coords = np.random.uniform(low=1, high=10000, size=100)
        data = np.random.rand(100)

        with tiledb.SparseArray(uri, mode="w") as T:
            T[coords] = data

        # Test that TILEDB_UNORDERED works correctly
        with tiledb.SparseArray(uri, mode="r") as A:
            res = A[1:10001][""]  # index past the end here to ensure inclusive result
            res = A.multi_index[1:10000][""]
            assert_array_equal(np.sort(res), np.sort(data))
            res = A.query(order="U").multi_index[1:10000][""]
            assert_array_equal(np.sort(res), np.sort(data))

    def test_sparse_fixes(self, sparse_cell_order):
        uri = self.path("test_sparse_fixes")
        # indexing a 1 element item in a sparse array
        # (issue directly reported)
        # the test here is that the indexing does not raise
        dims = (
            tiledb.Dim("foo", domain=(0, 6), tile=2),
            tiledb.Dim("bar", domain=(0, 6), tile=1),
            tiledb.Dim("baz", domain=(0, 100), tile=1),
        )
        dom = tiledb.Domain(*dims)
        att = tiledb.Attr(name="strattr", dtype="S1")
        schema = tiledb.ArraySchema(
            domain=dom, attrs=(att,), sparse=True, cell_order=sparse_cell_order
        )
        tiledb.SparseArray.create(uri, schema)
        with tiledb.SparseArray(uri) as T:
            T[:]

        # - test that assigning incompatible value to fixed-len str raises error
        # - test that value-conversion error raises exception w/ attr name context
        c = np.vstack(
            list((x, y, z) for x in range(7) for y in range(7) for z in range(101))
        )
        with tiledb.SparseArray(uri, "w") as T:
            with self.assertRaises(ValueError):
                T[c[:, 0], c[:, 1], c[:, 2]] = {"strattr": np.random.rand(7, 7, 101)}
            save_exc = list()
            try:
                T[c[:, 0], c[:, 1], c[:, 2]] = {"strattr": np.random.rand(7, 7, 101)}
            except ValueError as e:
                save_exc.append(e)
            exc = save_exc.pop()
            self.assertEqual(
                str(exc.__context__),
                "Cannot write a string value to non-string typed attribute 'strattr'!",
            )

    @tiledb.scope_ctx({"sm.check_coord_dups": False})
    def test_sparse_fixes_ch1560(self, sparse_cell_order):
        uri = self.path("sparse_fixes_ch1560")
        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(
                *[tiledb.Dim(name="id", domain=(1, 5000), tile=25, dtype="int32")]
            ),
            attrs=[
                tiledb.Attr(name="a1", dtype="datetime64[s]"),
                tiledb.Attr(name="a2", dtype="|S0"),
                tiledb.Attr(name="a3", dtype="|S0"),
                tiledb.Attr(name="a4", dtype="int32"),
                tiledb.Attr(name="a5", dtype="int8"),
                tiledb.Attr(name="a6", dtype="int32"),
            ],
            cell_order=sparse_cell_order,
            tile_order="row-major",
            sparse=True,
        )

        tiledb.SparseArray.create(uri, schema)

        data = OrderedDict(
            [
                (
                    "a1",
                    np.array(
                        [
                            "2017-04-01T04:00:00",
                            "2019-10-01T00:00:00",
                            "2019-10-01T00:00:00",
                            "2019-10-01T00:00:00",
                        ],
                        dtype="datetime64[s]",
                    ),
                ),
                ("a2", [b"Bus", b"The RIDE", b"The RIDE", b"The RIDE"]),
                ("a3", [b"Bus", b"The RIDE", b"The RIDE", b"The RIDE"]),
                ("a4", np.array([6911721, 138048, 138048, 138048], dtype="int32")),
                ("a5", np.array([20, 23, 23, 23], dtype="int8")),
                ("a6", np.array([345586, 6002, 6002, 6002], dtype="int32")),
            ]
        )

        with tiledb.open(uri, "w") as A:
            A[[1, 462, 462, 462]] = data

        with tiledb.open(uri) as A:
            res = A[:]
            res.pop("id")
            for k, v in res.items():
                if isinstance(data[k], (np.ndarray, list)):
                    assert_array_equal(res[k], data[k])
                else:
                    self.assertEqual(res[k], data[k])

    def test_sparse_2d_varlen_int(self, sparse_cell_order):
        path = self.path("test_sparse_2d_varlen_int")
        dtype = np.int32
        dom = tiledb.Domain(
            tiledb.Dim(domain=(1, 4), tile=2), tiledb.Dim(domain=(1, 4), tile=2)
        )
        att = tiledb.Attr(dtype=dtype, var=True)
        schema = tiledb.ArraySchema(
            dom, (att,), sparse=True, cell_order=sparse_cell_order
        )

        tiledb.SparseArray.create(path, schema)

        if tiledb.libtiledb.version() >= (2, 3) and sparse_cell_order == "hilbert":
            c1 = np.array([2, 1, 3, 4])
            c2 = np.array([1, 2, 3, 4])
        else:
            c1 = np.array([1, 2, 3, 4])
            c2 = np.array([2, 1, 3, 4])

        data = np.array(
            [
                np.array([1, 1], dtype=np.int32),
                np.array([2], dtype=np.int32),
                np.array([3, 3, 3], dtype=np.int32),
                np.array([4], dtype=np.int32),
            ],
            dtype="O",
        )

        with tiledb.SparseArray(path, "w") as A:
            A[c1, c2] = data

        with tiledb.SparseArray(path) as A:
            res = A[:]
            assert_subarrays_equal(res[""], data)
            assert_unordered_equal(res["__dim_0"], c1)
            assert_unordered_equal(res["__dim_1"], c2)

    def test_sparse_mixed_domain_uint_float64(self, sparse_cell_order):
        path = self.path("mixed_domain_uint_float64")
        dims = [
            tiledb.Dim(name="index", domain=(0, 51), tile=11, dtype=np.uint64),
            tiledb.Dim(name="dpos", domain=(-100.0, 100.0), tile=10, dtype=np.float64),
        ]
        dom = tiledb.Domain(*dims)
        attrs = [tiledb.Attr(name="val", dtype=np.float64)]

        schema = tiledb.ArraySchema(
            domain=dom, attrs=attrs, sparse=True, cell_order=sparse_cell_order
        )
        tiledb.SparseArray.create(path, schema)

        data = np.random.rand(50, 63)
        coords1 = np.repeat(np.arange(0, 50), 63)
        coords2 = np.linspace(-100.0, 100.0, num=3150)

        with tiledb.open(path, "w") as A:
            A[coords1, coords2] = data

        # tiledb returns coordinates in sorted order, so we need to check the output
        # sorted by the first dim coordinates
        sidx = np.argsort(coords1, kind="stable")
        coords2_idx = np.tile(np.arange(0, 63), 50)[sidx]

        with tiledb.open(path) as A:
            res = A[:]
            assert_subarrays_equal(
                data[coords1[sidx], coords2_idx[sidx]],
                res["val"],
                sparse_cell_order != "hilbert",
            )
            a_nonempty = A.nonempty_domain()
            self.assertEqual(a_nonempty[0], (0, 49))
            self.assertEqual(a_nonempty[1], (-100.0, 100.0))

    def test_sparse_string_domain(self, sparse_cell_order):
        path = self.path("sparse_string_domain")
        dom = tiledb.Domain(tiledb.Dim(name="d", domain=(None, None), dtype=np.bytes_))
        att = tiledb.Attr(name="a", dtype=np.int64)
        schema = tiledb.ArraySchema(
            domain=dom,
            attrs=(att,),
            sparse=True,
            cell_order=sparse_cell_order,
            capacity=10000,
        )
        tiledb.SparseArray.create(path, schema)

        data = [1, 2, 3, 4]
        coords = [b"aa", b"bbb", b"c", b"dddd"]

        with tiledb.open(path, "w") as A:
            A[coords] = data

        with tiledb.open(path) as A:
            ned = A.nonempty_domain()[0]
            res = A[ned[0] : ned[1]]
            assert_array_equal(res["a"], data)
            self.assertEqual(set(res["d"]), set(coords))
            self.assertEqual(A.nonempty_domain(), ((b"aa", b"dddd"),))

    def test_sparse_string_domain2(self, sparse_cell_order):
        path = self.path("sparse_string_domain2")
        with self.assertRaises(ValueError):
            dims = [
                tiledb.Dim(
                    name="str", domain=(None, None, None), tile=None, dtype=np.bytes_
                )
            ]
        dims = [tiledb.Dim(name="str", domain=(None, None), tile=None, dtype=np.bytes_)]
        dom = tiledb.Domain(*dims)
        attrs = [tiledb.Attr(name="val", dtype=np.float64)]

        schema = tiledb.ArraySchema(
            domain=dom, attrs=attrs, sparse=True, cell_order=sparse_cell_order
        )
        tiledb.SparseArray.create(path, schema)

        data = np.random.rand(10)
        coords = [rand_ascii_bytes(random.randint(5, 50)) for _ in range(10)]

        with tiledb.open(path, "w") as A:
            A[coords] = data

        with tiledb.open(path) as A:
            ned = A.nonempty_domain()[0]
            res = A[ned[0] : ned[1]]
            self.assertTrue(set(res["str"]) == set(coords))
            # must check data ordered by coords
            assert_array_equal(res["val"], data[np.argsort(coords, kind="stable")])

    def test_sparse_mixed_domain(self, sparse_cell_order):
        uri = self.path("sparse_mixed_domain")
        dims = [
            tiledb.Dim(name="p", domain=(-100.0, 100.0), tile=10, dtype=np.float64),
            tiledb.Dim(name="str", domain=(None, None), tile=None, dtype=np.bytes_),
        ]
        dom = tiledb.Domain(*dims)
        attrs = [tiledb.Attr(name="val", dtype=np.float64)]

        schema = tiledb.ArraySchema(
            domain=dom, attrs=attrs, sparse=True, cell_order=sparse_cell_order
        )
        tiledb.SparseArray.create(uri, schema)

        nrows = 5
        idx_f64 = np.random.rand(nrows)
        idx_str = [rand_ascii(5).encode("utf-8") for _ in range(nrows)]
        data = np.random.rand(nrows)

        with tiledb.SparseArray(uri, "w") as A:
            A[idx_f64, idx_str] = {"val": data}

        # test heterogeneous dim nonempty_domain
        ned_f64 = (np.array(np.min(idx_f64)), np.array(np.max(idx_f64)))
        idx_str.sort()
        ned_str = idx_str[0], idx_str[-1]

        with tiledb.SparseArray(uri, "r") as A:
            self.assertEqual(A.nonempty_domain(), (ned_f64, ned_str))

    def test_sparse_get_unique_dim_values(self, sparse_cell_order):
        uri = self.path("get_non_empty_coords")
        dim1 = tiledb.Dim(name="dim1", domain=(None, None), tile=None, dtype=np.bytes_)
        dim2 = tiledb.Dim(name="dim2", domain=(0, 1), tile=1, dtype=np.float64)
        attr = tiledb.Attr(name="attr", dtype=np.float32)
        dom = tiledb.Domain(dim1, dim2)
        schema = tiledb.ArraySchema(
            domain=dom, sparse=True, cell_order=sparse_cell_order, attrs=[attr]
        )
        tiledb.Array.create(uri, schema)

        with tiledb.open(uri, "w") as A:
            A["a1", 0] = 1
            A["a1", 0.25] = 2
            A["a2", 0.5] = 3
            A["a3", 0.25] = 4

        with tiledb.open(uri, "r") as A:
            self.assertEqual(
                A.unique_dim_values(),
                OrderedDict(
                    [("dim1", (b"a1", b"a2", b"a3")), ("dim2", (0.0, 0.25, 0.5))]
                ),
            )

            self.assertEqual(A.unique_dim_values("dim1"), (b"a1", b"a2", b"a3"))
            self.assertEqual(A.unique_dim_values("dim2"), (0, 0.25, 0.5))

            with self.assertRaises(ValueError):
                A.unique_dim_values(0)

            with self.assertRaises(ValueError):
                A.unique_dim_values("dim3")

    def test_sparse_write_for_zero_attrs(self):
        uri = self.path("test_sparse_write_to_zero_attrs")
        dim = tiledb.Dim(name="dim", domain=(0, 9), dtype=np.float64)
        schema = tiledb.ArraySchema(domain=tiledb.Domain(dim), sparse=True)
        tiledb.Array.create(uri, schema)

        coords = [1, 2.0, 3.5]

        with tiledb.open(uri, "w") as A:
            A[coords] = None

        with tiledb.open(uri, "r") as A:
            output = A.query()[:]
            assert list(output.keys()) == ["dim"]
            assert_array_equal(output["dim"][:], coords)

    def test_sparse_write_nullable_default(self):
        uri = self.path("test_sparse_write_nullable_default")

        dim1 = tiledb.Dim(name="d1", dtype="|S0", var=True)
        att = tiledb.Attr(name="a1", dtype="<U0", var=True, nullable=True)

        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(dim1),
            attrs=(att,),
            sparse=True,
            allows_duplicates=False,
        )
        tiledb.Array.create(uri, schema)

        with tiledb.open(uri, "w") as A:
            A[["a", "b", "c"]] = np.array(["aaa", "bb", "c"])

        if has_pandas():
            import pandas as pd

            with tiledb.open(uri) as A:
                pd._testing.assert_frame_equal(
                    A.query(dims=False).df[:],
                    pd.DataFrame({"a1": pd.Series(["aaa", "bb", "c"])}),
                )


class TestDenseIndexing(DiskTestCase):
    def _test_index(self, A, T, idx):
        expected = A[idx]
        actual = T[idx]
        assert_array_equal(expected, actual)

    good_index_1d = [
        # single value
        42,
        -1,
        # slices
        slice(0, 1050),
        slice(50, 150),
        slice(0, 2000),
        slice(-150, -50),
        # TODO: indexing failures
        # slice(-2000, 2000),
        # slice(0, 0),  # empty result
        # slice(-1, 0),  # empty result
        # total selections
        slice(None),
        Ellipsis,
        (),
        (Ellipsis, slice(None)),
        # slice with step
        slice(None),
        slice(None, None),
        slice(None, None, 1),
        slice(None, None, 10),
        slice(None, None, 100),
        slice(None, None, 1000),
        slice(None, None, 10000),
        slice(0, 1050),
        slice(0, 1050, 1),
        slice(0, 1050, 10),
        slice(0, 1050, 100),
        slice(0, 1050, 1000),
        slice(0, 1050, 10000),
        slice(1, 31, 3),
        slice(1, 31, 30),
        slice(1, 31, 300),
        slice(81, 121, 3),
        slice(81, 121, 30),
        slice(81, 121, 300),
        slice(50, 150),
        slice(50, 150, 1),
        slice(50, 150, 10),
        # TODO: negative steps
        slice(None, None, -1),
        slice(None, None, -10),
        slice(None, None, -100),
        slice(None, None, -1000),
        slice(None, None, -10000),
        # slice(1050, -1, -1),
        # slice(1050, -1, -10),
        # slice(1050, -1, -100),
        # slice(1050, -1, -1000),
        # slice(1050, -1, -10000),
        # slice(1050, 0, -1),
        # slice(1050, 0, -10),
        # slice(1050, 0, -100),
        # slice(1050, 0, -1000),
        # slice(1050, 0, -10000),
        # slice(150, 50, -1),
        # slice(150, 50, -10),
        # slice(31, 1, -3),
        # slice(121, 81, -3),
        # slice(-1, 0, -1),
    ]

    bad_index_1d = [2.3, "foo", b"xxx", None, (0, 0), (slice(None), slice(None))]

    def test_index_1d(self):
        A = np.arange(1050, dtype=int)

        dom = tiledb.Domain(tiledb.Dim(domain=(0, 1049), tile=100))
        att = tiledb.Attr(dtype=int)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,))
        tiledb.DenseArray.create(self.path("foo"), schema)

        with tiledb.DenseArray(self.path("foo"), mode="w") as T:
            T[:] = A

        with tiledb.DenseArray(self.path("foo"), mode="r") as T:
            for idx in self.good_index_1d:
                self._test_index(A, T, idx)

            for idx in self.bad_index_1d:
                with self.assertRaises(IndexError):
                    T[idx]

    good_index_2d = [
        # single row
        42,
        -1,
        (42, slice(None)),
        (-1, slice(None)),
        # single col
        (slice(None), 4),
        (slice(None), -1),
        # row slices
        slice(None),
        slice(0, 1000),
        slice(250, 350),
        slice(0, 2000),
        slice(-350, -250),
        slice(0, 0),  # empty result
        slice(-1, 0),  # empty result
        slice(-2000, 0),
        slice(-2000, 2000),
        # 2D slices
        (slice(None), slice(1, 5)),
        (slice(250, 350), slice(None)),
        (slice(250, 350), slice(1, 5)),
        (slice(250, 350), slice(-5, -1)),
        (slice(250, 350), slice(-50, 50)),
        (slice(250, 350, 10), slice(1, 5)),
        (slice(250, 350), slice(1, 5, 2)),
        (slice(250, 350, 33), slice(1, 5, 3)),
        # total selections
        (slice(None), slice(None)),
        Ellipsis,
        (),
        (Ellipsis, slice(None)),
        (Ellipsis, slice(None), slice(None)),
        # TODO: negative steps
        # slice(None, None, -1),
        # (slice(None, None, -1), slice(None)),
    ]

    bad_index_2d = [
        2.3,
        "foo",
        b"xxx",
        None,
        (2.3, slice(None)),
        (0, 0, 0),
        (slice(None), slice(None), slice(None)),
    ]

    def test_index_2d(self):
        A = np.arange(10000).reshape((1000, 10))

        dom = tiledb.Domain(
            tiledb.Dim(domain=(0, 999), tile=100), tiledb.Dim(domain=(0, 9), tile=2)
        )
        att = tiledb.Attr(dtype=A.dtype)
        schema = tiledb.ArraySchema(dom, (att,))
        tiledb.DenseArray.create(self.path("foo"), schema)

        with tiledb.DenseArray(self.path("foo"), mode="w") as T:
            T[:] = A

        with tiledb.DenseArray(self.path("foo"), mode="r") as T:
            for idx in self.good_index_1d:
                self._test_index(A, T, idx)

            for idx in self.bad_index_2d:
                with self.assertRaises(IndexError):
                    T[idx]


class TestDatetimeSlicing(DiskTestCase):
    def test_dense_datetime_vector(self):
        uri = self.path("foo_datetime_vector")

        # Domain is 10 years, day resolution, one tile per 365 days
        dim = tiledb.Dim(
            name="d1",
            domain=(np.datetime64("2010-01-01"), np.datetime64("2020-01-01")),
            tile=np.timedelta64(365, "D"),
            dtype=np.datetime64("", "D").dtype,
        )
        dom = tiledb.Domain(dim)
        schema = tiledb.ArraySchema(
            domain=dom, attrs=(tiledb.Attr("a1", dtype=np.float64),)
        )
        tiledb.Array.create(uri, schema)

        # Write a few years of data at the beginning using a timedelta object
        ndays = 365 * 2
        a1_vals = np.random.rand(ndays)
        start = np.datetime64("2010-01-01")
        # Datetime indexing is inclusive, so a delta of one less
        end = start + np.timedelta64(ndays - 1, "D")
        with tiledb.DenseArray(uri, "w") as T:
            T[start:end] = {"a1": a1_vals}

        # Read back data
        with tiledb.DenseArray(uri, "r", attr="a1") as T:
            assert_array_equal(T[start:end], a1_vals)

        # Check nonempty domain
        with tiledb.DenseArray(uri, "r") as T:
            nonempty = T.nonempty_domain()
            d1_nonempty = nonempty[0]
            self.assertEqual(d1_nonempty[0].dtype, np.datetime64("", "D"))
            self.assertEqual(d1_nonempty[1].dtype, np.datetime64("", "D"))
            self.assertTupleEqual(d1_nonempty, (start, end))

        # Slice a few days from the middle using two datetimes
        with tiledb.DenseArray(uri, "r", attr="a1") as T:
            # Slice using datetimes
            actual = T[np.datetime64("2010-11-01") : np.datetime64("2011-01-31")]

            # Convert datetime interval to integer offset/length into original array
            # must be cast to int because float slices are not allowed in NumPy 1.12+
            read_offset = int(
                (np.datetime64("2010-11-01") - start) / np.timedelta64(1, "D")
            )
            read_ndays = int(
                (np.datetime64("2011-01-31") - np.datetime64("2010-11-01") + 1)
                / np.timedelta64(1, "D")
            )
            expected = a1_vals[read_offset : read_offset + read_ndays]
            assert_array_equal(actual, expected)

        # Slice the first year
        with tiledb.DenseArray(uri, "r", attr="a1") as T:
            actual = T[np.datetime64("2010") : np.datetime64("2011")]

            # Convert datetime interval to integer offset/length into original array
            read_offset = int(
                (np.datetime64("2010-01-01") - start) / np.timedelta64(1, "D")
            )
            read_ndays = int(
                (np.datetime64("2011-01-01") - np.datetime64("2010-01-01") + 1)
                / np.timedelta64(1, "D")
            )
            expected = a1_vals[read_offset : read_offset + read_ndays]
            assert_array_equal(actual, expected)

        # Slice open spans
        with tiledb.DenseArray(uri, "r", attr="a1") as T:

            # Convert datetime interval to integer offset/length into original array
            read_offset = int(
                (np.datetime64("2010-01-01") - start) / np.timedelta64(1, "D")
            )
            read_ndays = int(
                (np.datetime64("2011-01-31") - np.datetime64("2010-01-01") + 1)
                / np.timedelta64(1, "D")
            )
            expected = a1_vals[read_offset : read_offset + read_ndays]

            # note we only wrote first two years
            actual = T.multi_index[np.datetime64("2010-01-01") :]["a1"][:read_ndays]
            assert_array_equal(actual, expected)

            actual2 = T[np.datetime64("2010-01-01") :][:read_ndays]
            assert_array_equal(actual2, expected)

    def test_sparse_datetime_vector(self, sparse_cell_order):
        uri = self.path("foo_datetime_sparse_vector")

        # ns resolution, one tile per second, max domain possible
        dim = tiledb.Dim(
            name="d1",
            domain=(
                np.datetime64(0, "ns"),
                np.datetime64(int(np.iinfo(np.int64).max) - 1000000000, "ns"),
            ),
            tile=np.timedelta64(1, "s"),
            dtype=np.datetime64("", "ns").dtype,
        )
        self.assertEqual(dim.tile, np.timedelta64("1000000000", "ns"))
        dom = tiledb.Domain(dim)
        schema = tiledb.ArraySchema(
            domain=dom,
            sparse=True,
            cell_order=sparse_cell_order,
            attrs=(tiledb.Attr("a1", dtype=np.float64),),
        )
        tiledb.Array.create(uri, schema)

        # Write 10k cells every 1000 ns starting at time 0
        coords = np.datetime64(0, "ns") + np.arange(0, 10000 * 1000, 1000)
        a1_vals = np.random.rand(len(coords))
        with tiledb.SparseArray(uri, "w") as T:
            T[coords] = {"a1": a1_vals}

        # Read all
        with tiledb.SparseArray(uri, "r") as T:
            assert_array_equal(T[:]["a1"], a1_vals)

        # Read back first 10 cells
        with tiledb.SparseArray(uri, "r") as T:
            start = np.datetime64(0, "ns")
            vals = T[start : start + np.timedelta64(10000, "ns")]["a1"]
            assert_array_equal(vals, a1_vals[0:11])

            # Test open ended ranges multi_index
            vals2 = T.multi_index[start:]["a1"]
            assert_array_equal(vals2, a1_vals)

            stop = np.datetime64(int(np.iinfo(np.int64).max) - 1000000000, "ns")
            vals3 = T.multi_index[:stop]["a1"]
            assert_array_equal(vals3, a1_vals)

    def test_datetime_types(self, sparse_cell_order):
        units = ["h", "m", "s", "ms", "us", "ns", "ps", "fs"]

        for res in units:
            uri = self.path("test_datetime_type_" + res)

            tmax = 1000
            tile = np.timedelta64(1, res)

            dim = tiledb.Dim(
                name="d1",
                domain=(None, None),
                tile=tile,
                dtype=np.datetime64("", res).dtype,
            )
            dom = tiledb.Domain(dim)
            schema = tiledb.ArraySchema(
                domain=dom,
                sparse=True,
                cell_order=sparse_cell_order,
                attrs=(tiledb.Attr("a1", dtype=np.float64),),
            )

            tiledb.Array.create(uri, schema)

            # Write tmax cells every 10 units starting at time 0
            coords = np.datetime64(0, res) + np.arange(
                0, tmax, 10
            )  # np.arange(0, 10000 * 1000, 1000)
            a1_vals = np.random.rand(len(coords))
            with tiledb.SparseArray(uri, "w") as T:
                T[coords] = {"a1": a1_vals}

            # Read all
            with tiledb.SparseArray(uri, "r") as T:
                assert_array_equal(T[:]["a1"], a1_vals)

            # Read back first 10 cells
            with tiledb.SparseArray(uri, "r") as T:
                start = np.datetime64(0, res)
                vals = T[start : start + np.timedelta64(int(tmax / 10), res)]["a1"]
                assert_array_equal(vals, a1_vals[0:11])


class PickleTest(DiskTestCase):
    # test that DenseArray and View can be pickled for multiprocess use
    # note that the current pickling is by URI and attributes (it is
    #     not, and likely should not be, a way to serialize array data)

    @pytest.mark.parametrize("sparse", [True, False])
    def test_pickle_roundtrip(self, sparse):
        uri = self.path("test_pickle_roundtrip")
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 2), tile=3))
        schema = tiledb.ArraySchema(domain=dom, attrs=(tiledb.Attr(""),), sparse=sparse)
        tiledb.libtiledb.Array.create(uri, schema)

        with tiledb.open(uri, "w") as T:
            if sparse:
                T[[0, 1, 2]] = np.random.randint(10, size=3)
            else:
                T[:] = np.random.randint(10, size=3)

        with tiledb.open(uri, "r") as T:
            with io.BytesIO() as buf:
                pickle.dump(T, buf)
                buf.seek(0)
                with pickle.load(buf) as T2:
                    if sparse:
                        assert_array_equal(T[:][""], T2[:][""])
                    else:
                        assert_array_equal(T[:], T2[:])

            with io.BytesIO() as buf, tiledb.open(uri) as V:
                pickle.dump(V, buf)
                buf.seek(0)
                with pickle.load(buf) as V2:
                    # make sure anonymous view pickles and round-trips
                    if sparse:
                        assert_array_equal(V[:][""], V2[:][""])
                    else:
                        assert_array_equal(V[:], V2[:])

    @tiledb.scope_ctx({"vfs.s3.region": "kuyper-belt-1", "vfs.max_parallel_ops": "1"})
    def test_pickle_with_config(self):
        uri = self.path("pickle_config")
        T = tiledb.DenseArray.from_numpy(uri, np.random.rand(3, 3))

        with io.BytesIO() as buf:
            pickle.dump(T, buf)
            buf.seek(0)
            T2 = pickle.load(buf)
            assert_array_equal(T, T2)
            self.maxDiff = None
            d1 = tiledb.default_ctx().config().dict()
            d2 = T2._ctx_().config().dict()
            self.assertEqual(d1["vfs.s3.region"], d2["vfs.s3.region"])
            self.assertEqual(d1["vfs.max_parallel_ops"], d2["vfs.max_parallel_ops"])
        T.close()
        T2.close()

    @pytest.mark.parametrize("sparse", [True, False])
    def test_pickle_with_tuple_timestamps(self, sparse):
        A = np.random.randint(10, size=3)
        path = self.path("test_pickle_with_tuple_timestamps")

        dom = tiledb.Domain(tiledb.Dim(domain=(0, 2), tile=3, dtype=np.int64))
        att = tiledb.Attr(dtype=A.dtype)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=sparse)
        tiledb.libtiledb.Array.create(path, schema)

        for ts in range(1, 5):
            with tiledb.open(path, timestamp=ts, mode="w") as T:
                if sparse:
                    T[[0, 1, 2]] = A * ts
                else:
                    T[:] = A * ts

        with tiledb.open(path, timestamp=(2, 3), mode="r") as T:
            with io.BytesIO() as buf:
                pickle.dump(T, buf)
                buf.seek(0)
                with pickle.load(buf) as T2:
                    if sparse:
                        assert_array_equal(T[:][""], T2[:][""])
                    else:
                        assert_array_equal(T[:], T2[:])
                    assert T2.timestamp_range == (2, 3)

            with io.BytesIO() as buf, tiledb.open(path, timestamp=(2, 3)) as V:
                pickle.dump(V, buf)
                buf.seek(0)
                with pickle.load(buf) as V2:
                    # make sure anonymous view pickles and round-trips
                    if sparse:
                        assert_array_equal(V[:][""], V2[:][""])
                    else:
                        assert_array_equal(V[:], V2[:])
                    assert V2.timestamp_range == (2, 3)


class ArrayViewTest(DiskTestCase):
    def test_view_multiattr(self):
        uri = self.path("foo_multiattr")
        dom = tiledb.Domain(
            tiledb.Dim(domain=(0, 2), tile=3), tiledb.Dim(domain=(0, 2), tile=3)
        )
        schema = tiledb.ArraySchema(
            domain=dom, attrs=(tiledb.Attr(""), tiledb.Attr("named"))
        )
        tiledb.libtiledb.Array.create(uri, schema)

        anon_ar = np.random.rand(3, 3)
        named_ar = np.random.rand(3, 3)

        with tiledb.DenseArray(uri, "w") as T:
            T[:] = {"": anon_ar, "named": named_ar}

        with self.assertRaises(KeyError):
            T = tiledb.DenseArray(uri, "r", attr="foo111")

        with tiledb.DenseArray(uri, "r", attr="named") as T:
            assert_array_equal(T, named_ar)
            # make sure each attr view can pickle and round-trip
            with io.BytesIO() as buf:
                pickle.dump(T, buf)
                buf.seek(0)
                with pickle.load(buf) as T_rt:
                    assert_array_equal(T, T_rt)

        with tiledb.DenseArray(uri, "r", attr="") as T:
            assert_array_equal(T, anon_ar)

            with io.BytesIO() as buf:
                pickle.dump(T, buf)
                buf.seek(0)
                with pickle.load(buf) as tmp:
                    assert_array_equal(tmp, anon_ar)

        # set subarray on multi-attribute
        range_ar = np.arange(0, 9).reshape(3, 3)
        with tiledb.DenseArray(uri, "w", attr="named") as V_named:
            V_named[1:3, 1:3] = range_ar[1:3, 1:3]

        with tiledb.DenseArray(uri, "r", attr="named") as V_named:
            assert_array_equal(V_named[1:3, 1:3], range_ar[1:3, 1:3])


class RWTest(DiskTestCase):
    def test_read_write(self, capfd):
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 2), tile=3))
        att = tiledb.Attr(dtype="int64")
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,))
        tiledb.libtiledb.Array.create(self.path("foo"), schema)

        np_array = np.array([1, 2, 3], dtype="int64")

        with tiledb.DenseArray(self.path("foo"), mode="w") as arr:
            arr.write_direct(np_array)

        with tiledb.DenseArray(self.path("foo"), mode="r") as arr:
            arr.dump()

            assert_captured(capfd, "Array type: dense")
            self.assertEqual(arr.nonempty_domain(), ((0, 2),))
            self.assertEqual(arr.ndim, np_array.ndim)
            assert_array_equal(arr.read_direct(), np_array)


class TestNumpyToArray(DiskTestCase):
    def test_to_array0d(self):
        # Cannot create 0-dim arrays in TileDB
        np_array = np.array(1)
        with self.assertRaises(tiledb.TileDBError):
            with tiledb.DenseArray.from_numpy(self.path("foo"), np_array) as A:
                pass

    def test_to_array1d(self):
        np_array = np.array([1.0, 2.0, 3.0])
        with tiledb.DenseArray.from_numpy(self.path("foo"), np_array) as arr:
            assert_array_equal(arr[:], np_array)

    def test_to_array2d(self):
        np_array = np.ones((100, 100), dtype="i8")
        with tiledb.DenseArray.from_numpy(self.path("foo"), np_array) as arr:
            assert_array_equal(arr[:], np_array)

    def test_to_array3d(self):
        np_array = np.ones((1, 1, 1), dtype="i1")
        with tiledb.DenseArray.from_numpy(self.path("foo"), np_array) as arr:
            assert_array_equal(arr[:], np_array)

    def test_bytes_to_array1d(self):
        np_array = np.array(
            [b"abcdef", b"gh", b"ijkl", b"mnopqrs", b"", b"1234545lkjalsdfj"],
            dtype=object,
        )
        with tiledb.DenseArray.from_numpy(self.path("foo"), np_array) as arr:
            assert_array_equal(arr[:], np_array)

        with tiledb.DenseArray(self.path("foo")) as arr_reload:
            assert_array_equal(arr_reload[:], np_array)

    def test_unicode_to_array1d(self):
        np_array = np.array(
            [
                "1234545lkjalsdfj",
                "mnopqrs",
                "ijkl",
                "gh",
                "abcdef",
                "aαbββcγγγdδδδδ",
                "",
                '"aαbββc',
                "",
                "γγγdδδδδ",
            ],
            dtype=object,
        )
        with tiledb.DenseArray.from_numpy(self.path("foo"), np_array) as arr:
            assert_array_equal(arr[:], np_array)

        with tiledb.DenseArray(self.path("foo")) as arr_reload:
            assert_array_equal(arr_reload[:], np_array)

    def test_array_interface(self):
        # Tests that __array__ interface works
        np_array1 = np.arange(1, 10)
        with tiledb.DenseArray.from_numpy(self.path("arr1"), np_array1) as arr1:
            assert_array_equal(np.array(arr1), np_array1)

        # Test that __array__ interface throws an error when number of attributes > 1
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 2), tile=3))
        foo = tiledb.Attr("foo", dtype="i8")
        bar = tiledb.Attr("bar", dtype="i8")
        schema = tiledb.ArraySchema(domain=dom, attrs=(foo, bar))
        tiledb.DenseArray.create(self.path("arr2"), schema)
        with self.assertRaises(ValueError):
            with tiledb.DenseArray(self.path("arr2"), mode="r") as arr2:
                np.array(arr2)

    def test_array_getindex(self):
        # Tests that __getindex__ interface works
        np_array = np.arange(1, 10)
        with tiledb.DenseArray.from_numpy(self.path("foo"), np_array) as arr:
            assert_array_equal(arr[5:10], np_array[5:10])

    def test_to_array1d_attr_name(self):
        np_array = np.array([1.0, 2.0, 3.0])
        with tiledb.DenseArray.from_numpy(
            self.path("foo"), np_array, attr_name="a"
        ) as arr:
            assert_array_equal(arr[:]["a"], np_array)

    def test_from_numpy_timestamp(self):
        path = self.path()
        with tiledb.from_numpy(path, np.array([1, 2, 3]), timestamp=10) as A:
            pass
        with tiledb.open(path, timestamp=(0, 9)) as A:
            assert A.nonempty_domain() == None
        with tiledb.open(path, timestamp=(10, 10)) as A:
            assert A.nonempty_domain() == ((0, 2),)


class ConsolidationTest(DiskTestCase):
    def test_array_vacuum(self):
        dshape = (0, 19)
        num_writes = 10

        def create_array(target_path):
            dom = tiledb.Domain(tiledb.Dim(domain=dshape, tile=3))
            att = tiledb.Attr(dtype="int64")
            schema = tiledb.ArraySchema(domain=dom, attrs=(att,))
            tiledb.libtiledb.Array.create(target_path, schema)

        def write_fragments(target_path):
            for i in range(num_writes):
                with tiledb.open(target_path, "w") as A:
                    A[i : dshape[1]] = np.random.rand(dshape[1] - i)

        # array #1
        path = self.path("test_array_vacuum")
        create_array(path)
        write_fragments(path)

        fi = tiledb.FragmentInfoList(path)
        self.assertEqual(len(fi), num_writes)

        tiledb.consolidate(path)
        tiledb.vacuum(path)

        fi = tiledb.FragmentInfoList(path)
        self.assertEqual(len(fi), 1)

        # array #2
        path2 = self.path("test_array_vacuum_fragment_meta")
        create_array(path2)
        write_fragments(path2)

        fi = tiledb.FragmentInfoList(path2)
        self.assertEqual(fi.unconsolidated_metadata_num, num_writes)

        tiledb.consolidate(
            path2, config=tiledb.Config({"sm.consolidation.mode": "fragment_meta"})
        )
        tiledb.vacuum(path2, config=tiledb.Config({"sm.vacuum.mode": "fragment_meta"}))

        fi = tiledb.FragmentInfoList(path2)
        self.assertEqual(fi.unconsolidated_metadata_num, 0)

        # array #3
        path3 = self.path("test_array_vacuum2")
        create_array(path3)
        write_fragments(path3)

        fi = tiledb.FragmentInfoList(path3)
        self.assertEqual(fi.unconsolidated_metadata_num, num_writes)

        conf = tiledb.Config({"sm.consolidation.mode": "fragment_meta"})
        with tiledb.open(path3, "w") as A:
            A.consolidate(config=conf)

        fi = tiledb.FragmentInfoList(path3)
        self.assertEqual(fi.unconsolidated_metadata_num, 0)

    def test_array_consolidate_with_timestamp(self):
        dshape = (1, 3)
        num_writes = 10

        def create_array(target_path, dshape):
            dom = tiledb.Domain(tiledb.Dim(domain=dshape, tile=len(dshape)))
            att = tiledb.Attr(dtype="int64")
            schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True)
            tiledb.libtiledb.Array.create(target_path, schema)

        def write_fragments(target_path, dshape, num_writes):
            for i in range(1, num_writes + 1):
                with tiledb.open(target_path, "w", timestamp=i) as A:
                    A[[1, 2, 3]] = np.random.rand(dshape[1])

        path = self.path("test_array_consolidate_with_timestamp")

        create_array(path, dshape)
        write_fragments(path, dshape, num_writes)
        frags = tiledb.FragmentInfoList(path)
        assert len(frags) == 10

        tiledb.consolidate(path, timestamp=(1, 4))
        frags = tiledb.FragmentInfoList(path)
        assert len(frags) == 7
        assert len(frags.to_vacuum) == 4

        with pytest.warns(
            DeprecationWarning,
            match=(
                "Partial vacuuming via timestamp will be deprecrated in "
                "a future release and replaced by passing in fragment URIs."
            ),
        ):
            tiledb.vacuum(path, timestamp=(1, 2))

        tiledb.vacuum(path)
        frags = tiledb.FragmentInfoList(path)
        assert len(frags.to_vacuum) == 0

        conf = tiledb.Config(
            {"sm.consolidation.timestamp_start": 5, "sm.consolidation.timestamp_end": 9}
        )
        tiledb.consolidate(path, config=conf)
        tiledb.vacuum(path)
        frags = tiledb.FragmentInfoList(path)
        assert len(frags.timestamp_range) == 3


@pytest.mark.skipif(sys.platform == "win32", reason="Only run MemoryTest on linux")
class MemoryTest(DiskTestCase):
    # sanity check that memory usage doesn't increase more than 2x when reading 40MB 100x
    # https://github.com/TileDB-Inc/TileDB-Py/issues/150
    @staticmethod
    def use_many_buffers(path):
        # https://stackoverflow.com/questions/938733/total-memory-used-by-python-process
        process = psutil.Process(os.getpid())

        x = np.ones(10000000, dtype=np.float32)
        d1 = tiledb.Dim(
            "test_domain", domain=(0, x.shape[0] - 1), tile=10000, dtype="uint32"
        )
        domain = tiledb.Domain(d1)
        v = tiledb.Attr("test_value", dtype="float32")

        schema = tiledb.ArraySchema(
            domain=domain, attrs=(v,), cell_order="row-major", tile_order="row-major"
        )

        A = tiledb.DenseArray.create(path, schema)

        with tiledb.DenseArray(path, mode="w") as A:
            A[:] = {"test_value": x}

        with tiledb.DenseArray(path, mode="r") as data:
            data[:]
            initial = process.memory_info().rss
            print("  initial RSS: {}".format(round(initial / (10**6)), 2))
            for i in range(100):
                # read but don't store: this memory should be freed
                data[:]

                if i % 10 == 0:
                    print(
                        "    read iter {}, RSS (MB): {}".format(
                            i, round(process.memory_info().rss / (10**6), 2)
                        )
                    )

        return initial

    def test_memory_cleanup(self, capfd):
        # run function which reads 100x from a 40MB test array
        # TODO: RSS is too loose to do this end-to-end, so should use instrumentation.
        print("Starting TileDB-Py memory test:")
        initial = self.use_many_buffers(self.path("test_memory_cleanup"))

        process = psutil.Process(os.getpid())
        final = process.memory_info().rss
        print("  final RSS: {}".format(round(final / (10**6)), 2))

        gc.collect()

        final_gc = process.memory_info().rss
        print("  final RSS after forced GC: {}".format(round(final_gc / (10**6)), 2))

        assert_captured(capfd, "final RSS")
        self.assertTrue(final < (2 * initial))


class TestHighlevel(DiskTestCase):
    def test_open(self):
        uri = self.path("test_open")
        array = np.random.rand(10)
        schema = tiledb.schema_like(array)
        tiledb.Array.create(uri, schema)
        with tiledb.open(uri, "w") as A:
            A[:] = array * 10
            A[:] = array
            last_fragment_ts = list(A.last_write_info.items())[0][1][0]

        ctx = tiledb.Ctx()
        with tiledb.DenseArray(uri, ctx=ctx) as A:
            self.assertEqual(A._ctx_(), ctx)

        # test `open` with timestamp
        with tiledb.open(uri, timestamp=last_fragment_ts) as A:
            assert_array_equal(A[:], array)

        with tiledb.open(uri, ctx=ctx) as A:
            self.assertEqual(A._ctx_(), ctx)

        config = tiledb.Config()
        with tiledb.open(uri, config=config) as A:
            self.assertEqual(A._ctx_().config(), config)

        with self.assertRaises(KeyError):
            # This path must test `tiledb.open` specifically
            # https://github.com/TileDB-Inc/TileDB-Py/issues/277
            tiledb.open(uri, "r", attr="the-missing-attr")

    def test_ctx_thread_cleanup(self):
        # This test checks that contexts are destroyed correctly.
        # It creates new contexts repeatedly, in-process, and
        # checks that the total number of threads stays stable.
        config = {"sm.num_reader_threads": 128}
        ll = list()
        uri = self.path("test_ctx_thread_cleanup")
        with tiledb.from_numpy(uri, np.random.rand(100)) as A:
            pass

        thisproc = psutil.Process(os.getpid())

        for n in range(0, 10):
            if n > 0:
                retry = 0
                while retry < 3:
                    try:
                        # checking exact thread count is unreliable, so
                        # make sure we are holding < 2x per run.
                        self.assertTrue(len(thisproc.threads()) < 2 * start_threads)
                        break
                    except AssertionError as exc:
                        raise exc
                    except RuntimeError as rterr:
                        retry += 1
                        if retry > 2:
                            raise rterr
                        warnings.warn(
                            "Thread cleanup test RuntimeError: {} \n    on iteration: {}".format(
                                str(rterr), n
                            )
                        )

            with tiledb.DenseArray(uri, ctx=tiledb.Ctx(config)) as A:
                res = A[:]

            if n == 0:
                start_threads = len(thisproc.threads())


# Wrapper to execute specific code in subprocess so that we can ensure the thread count
# init is correct. Necessary because multiprocess.get_context is only available in Python 3.4+,
# and the multiprocessing method may be set to fork by other tests (e.g. dask).
def init_test_wrapper(cfg=None):
    python_exe = sys.executable
    cmd = "from test_libtiledb import *; init_test_helper({})".format(cfg)
    test_path = os.path.dirname(os.path.abspath(__file__))

    sp_output = subprocess.check_output([python_exe, "-c", cmd], cwd=test_path)
    return int(sp_output.decode("UTF-8").strip())


def init_test_helper(cfg=None):
    tiledb.libtiledb.default_ctx(cfg)
    concurrency_level = tiledb.default_ctx().config()["sm.io_concurrency_level"]
    print(int(concurrency_level))


class ContextTest(unittest.TestCase):
    def test_default_ctx(self):
        ctx = tiledb.default_ctx()
        self.assertIsInstance(ctx, tiledb.Ctx)
        assert isinstance(ctx.config(), tiledb.libtiledb.Config)

    def test_default_ctx_errors(self):
        config = tiledb.Config()
        ctx = tiledb.Ctx(config=config)

        with pytest.raises(ValueError) as excinfo:
            tiledb.default_ctx(ctx)
        assert (
            "default_ctx takes in `tiledb.Config` object or dictionary with "
            "config parameters."
        ) == str(excinfo.value)

    def test_scope_ctx(self):
        key = "sm.tile_cache_size"
        ctx0 = tiledb.default_ctx()
        new_config_dict = {key: 42}
        new_config = tiledb.Config({key: 78})
        new_ctx = tiledb.Ctx({key: 61})

        assert tiledb.default_ctx() is ctx0
        assert tiledb.default_ctx().config()[key] == "10000000"

        with tiledb.scope_ctx(new_config_dict) as ctx1:
            assert tiledb.default_ctx() is ctx1
            assert tiledb.default_ctx().config()[key] == "42"
            with tiledb.scope_ctx(new_config) as ctx2:
                assert tiledb.default_ctx() is ctx2
                assert tiledb.default_ctx().config()[key] == "78"
                with tiledb.scope_ctx(new_ctx) as ctx3:
                    assert tiledb.default_ctx() is ctx3 is new_ctx
                    assert tiledb.default_ctx().config()[key] == "61"
                assert tiledb.default_ctx() is ctx2
                assert tiledb.default_ctx().config()[key] == "78"
            assert tiledb.default_ctx() is ctx1
            assert tiledb.default_ctx().config()[key] == "42"

        assert tiledb.default_ctx() is ctx0
        assert tiledb.default_ctx().config()[key] == "10000000"

    def test_scope_ctx_error(self):
        with pytest.raises(ValueError) as excinfo:
            with tiledb.scope_ctx([]):
                pass
        assert (
            "scope_ctx takes in `tiledb.Ctx` object, `tiledb.Config` object, "
            "or dictionary with config parameters."
        ) == str(excinfo.value)

    @pytest.mark.skipif(
        "pytest.tiledb_vfs == 's3'", reason="Test not yet supported with S3"
    )
    def test_init_config(self):
        self.assertEqual(
            int(tiledb.default_ctx().config()["sm.io_concurrency_level"]),
            init_test_wrapper(),
        )

        self.assertEqual(3, init_test_wrapper({"sm.io_concurrency_level": 3}))


class GetStatsTest(DiskTestCase):
    def test_ctx(self):
        tiledb.libtiledb.stats_enable()
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
        tiledb.libtiledb.stats_enable()
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


class ReprTest(DiskTestCase):
    def test_attr_repr(self):
        attr = tiledb.Attr(name="itsanattr", dtype=np.float64)
        self.assertTrue(
            re.match(
                r"Attr\(name=[u]?'itsanattr', dtype='float64', var=False, nullable=False\)",
                repr(attr),
            )
        )

        g = dict()
        exec("from tiledb import Attr; from numpy import float64", g)
        self.assertEqual(eval(repr(attr), g), attr)

    def test_dim_repr(self):
        dtype_set = [bytes, np.bytes_]
        opts = {
            None: None,
            "var": True,
            "domain": (None, None),
            "filters": [tiledb.GzipFilter()],
        }

        dim_test_imports = textwrap.dedent(
            """
            from tiledb import Dim, FilterList, GzipFilter
            import numpy
            from numpy import float64
            """
        )

        for dtype in dtype_set:
            opt_choices = [
                itertools.combinations(opts.keys(), r=n)
                for n in range(1, len(opts) + 1)
            ]
            for opt_set in itertools.chain(*opt_choices):
                opt_kwarg = {k: opts[k] for k in opt_set if k}
                g = dict()
                exec(dim_test_imports, g)

                dim = tiledb.Dim(name="d1", dtype=dtype, **opt_kwarg)
                self.assertEqual(eval(repr(dim), g), dim)

        # test datetime
        g = dict()
        exec(dim_test_imports, g)
        dim = tiledb.Dim(
            name="d1",
            domain=(np.datetime64("2010-01-01"), np.datetime64("2020")),
            tile=2,
            dtype=np.datetime64("", "D"),
        )
        self.assertEqual(eval(repr(dim), g), dim)

    def test_arrayschema_repr(self, sparse_cell_order):
        filters = tiledb.FilterList([tiledb.ZstdFilter(-1)])
        for sparse in [False, True]:
            cell_order = sparse_cell_order if sparse else None
            domain = tiledb.Domain(
                tiledb.Dim(domain=(1, 8), tile=2), tiledb.Dim(domain=(1, 8), tile=2)
            )
            a1 = tiledb.Attr("val", dtype="f8", filters=filters)
            orig_schema = tiledb.ArraySchema(
                domain=domain, attrs=(a1,), sparse=sparse, cell_order=cell_order
            )

            schema_repr = repr(orig_schema)
            g = dict()
            setup = "from tiledb import *\n" "import numpy as np\n"

            exec(setup, g)
            new_schema = None
            try:
                new_schema = eval(schema_repr, g)
            except Exception as exc:
                warn_str = (
                    """Exception during ReprTest schema eval"""
                    + """, schema string was:\n"""
                    + """'''"""
                    + """\n{}\n'''""".format(schema_repr)
                )
                warnings.warn(warn_str)
                raise

            self.assertEqual(new_schema, orig_schema)

    def test_arrayschema_repr_hilbert(self):
        domain = tiledb.Domain(tiledb.Dim(domain=(1, 8), tile=2))
        a = tiledb.Attr("a", dtype="f8")
        schema = tiledb.ArraySchema(
            domain=domain, attrs=(a,), cell_order="hilbert", sparse=True
        )

        assert schema.cell_order == "hilbert"
        assert schema.tile_order == None


class NullableIOTest(DiskTestCase):
    def test_nullable_write(self):
        uri = self.path("nullable_write_test")

        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(
                *[tiledb.Dim(name="__dim_0", domain=(0, 3), tile=4, dtype="uint64")]
            ),
            attrs=[tiledb.Attr(name="", dtype="int64", var=False, nullable=True)],
        )
        tiledb.Array.create(uri, schema)

        with tiledb.open(uri, "w") as A:
            A._setitem_impl(
                slice(0, 4), np.ones(4), {"": np.array([0, 1, 0, 1], dtype=np.uint8)}
            )


class IncompleteTest(DiskTestCase):
    @pytest.mark.parametrize("non_overlapping_ranges", [True, False])
    def test_incomplete_dense_varlen(self, non_overlapping_ranges):
        ncells = 10
        path = self.path("incomplete_dense_varlen")
        str_data = [rand_utf8(random.randint(0, n)) for n in range(ncells)]
        data = np.array(str_data, dtype=np.unicode_)

        # basic write
        dom = tiledb.Domain(tiledb.Dim(domain=(1, len(data)), tile=len(data)))
        att = tiledb.Attr(dtype=np.unicode_, var=True)

        schema = tiledb.ArraySchema(dom, (att,))

        tiledb.DenseArray.create(path, schema)
        with tiledb.DenseArray(path, mode="w") as T:
            T[:] = data

        with tiledb.DenseArray(path, mode="r") as T:
            assert_array_equal(data, T[:])

        # set the memory to the max length of a cell
        # these settings force ~100 retries
        # TODO would be good to check repeat count here; not yet exposed
        #      Also would be useful to have max cell config in libtiledb.
        init_buffer_bytes = 1024**2
        config = tiledb.Config(
            {
                "sm.memory_budget": ncells,
                "sm.memory_budget_var": ncells,
                "py.init_buffer_bytes": init_buffer_bytes,
                "sm.query.sparse_unordered_with_dups.non_overlapping_ranges": non_overlapping_ranges,
            }
        )
        self.assertEqual(config["py.init_buffer_bytes"], str(init_buffer_bytes))

        with tiledb.DenseArray(path, mode="r", ctx=tiledb.Ctx(config)) as T2:
            result = T2.query(attrs=[""])[:]
            assert_array_equal(result, data)

    @pytest.mark.parametrize("allows_duplicates", [True, False])
    @pytest.mark.parametrize("non_overlapping_ranges", [True, False])
    def test_incomplete_sparse_varlen(self, allows_duplicates, non_overlapping_ranges):
        ncells = 100

        path = self.path("incomplete_sparse_varlen")
        str_data = [rand_utf8(random.randint(0, n)) for n in range(ncells)]
        data = np.array(str_data, dtype=np.unicode_)
        coords = np.arange(ncells)

        # basic write
        dom = tiledb.Domain(tiledb.Dim(domain=(0, len(data) + 100), tile=len(data)))
        att = tiledb.Attr(dtype=np.unicode_, var=True)

        schema = tiledb.ArraySchema(
            dom, (att,), sparse=True, allows_duplicates=allows_duplicates
        )

        tiledb.SparseArray.create(path, schema)
        with tiledb.SparseArray(path, mode="w") as T:
            T[coords] = data

        with tiledb.SparseArray(path, mode="r") as T:
            assert_array_equal(data, T[:][""])

        # set the memory to the max length of a cell
        # these settings force ~100 retries
        # TODO would be good to check repeat count here; not yet exposed
        #      Also would be useful to have max cell config in libtiledb.
        init_buffer_bytes = 1024**2
        config = tiledb.Config(
            {
                "sm.memory_budget": ncells,
                "sm.memory_budget_var": ncells,
                "py.init_buffer_bytes": init_buffer_bytes,
            }
        )
        self.assertEqual(config["py.init_buffer_bytes"], str(init_buffer_bytes))

        with tiledb.SparseArray(path, mode="r", ctx=tiledb.Ctx(config)) as T2:
            assert_array_equal(data, T2[:][""])

            assert_array_equal(data, T2.multi_index[0:ncells][""])

            # ensure that empty results are handled correctly
            assert_array_equal(
                T2.multi_index[101:105][""], np.array([], dtype=np.dtype("<U"))
            )

    @pytest.mark.skipif(not has_pandas(), reason="pandas not installed")
    @pytest.mark.parametrize(
        "return_arrow, indexer", [(True, "df"), (False, "df"), (False, "multi_index")]
    )
    @pytest.mark.parametrize(
        "test_incomplete_return_array", [True, False], indirect=True
    )
    @pytest.mark.parametrize("non_overlapping_ranges", [True, False])
    def test_incomplete_return(
        self,
        test_incomplete_return_array,
        return_arrow,
        indexer,
        non_overlapping_ranges,
    ):
        import pyarrow as pa
        from tiledb.multirange_indexing import EstimatedResultSize
        import pandas as pd

        path = test_incomplete_return_array

        init_buffer_bytes = 200
        cfg = tiledb.Config(
            {
                "py.init_buffer_bytes": init_buffer_bytes,
                "py.exact_init_buffer_bytes": "true",
                "sm.query.sparse_unordered_with_dups.non_overlapping_ranges": non_overlapping_ranges,
            }
        )

        with tiledb.open(path) as A:
            full_data = A[:][""]

        # count number of elements retrieved so that we can slice the comparison array
        idx = 0
        with tiledb.open(path, ctx=tiledb.Ctx(cfg)) as A:
            query = A.query(return_incomplete=True, return_arrow=return_arrow)
            iterable = getattr(query, indexer)[:]

            est_results = iterable.estimated_result_sizes()
            assert isinstance(est_results[""], EstimatedResultSize)
            assert isinstance(est_results["__dim_0"], EstimatedResultSize)
            assert est_results["__dim_0"].offsets_bytes == 0
            assert est_results["__dim_0"].data_bytes > 0
            assert est_results[""].offsets_bytes > 0
            assert est_results[""].data_bytes > 0

            for result in iterable:
                if return_arrow:
                    assert isinstance(result, pa.Table)
                    df = result.to_pandas()
                else:
                    if indexer == "df":
                        assert isinstance(result, pd.DataFrame)
                        df = result
                    else:
                        assert isinstance(result, OrderedDict)
                        df = pd.DataFrame(result)

                to_slice = slice(idx, idx + len(df))
                chunk = full_data[to_slice]

                assert np.all(chunk == df[""].values)
                assert np.all(df["__dim_0"] == np.arange(idx, idx + len(df)))
                # update the current read count
                idx += len(df)

        assert idx == len(full_data)

    @pytest.mark.parametrize("cell_order", ["col-major", "row-major", "hilbert"])
    @pytest.mark.parametrize("tile_order", ["col-major", "row-major"])
    @pytest.mark.parametrize("non_overlapping_ranges", [True, False])
    def test_incomplete_global_order(
        self, cell_order, tile_order, non_overlapping_ranges
    ):
        uri = self.path("test_incomplete_global_order")
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 30), tile=10, dtype=np.int64))
        att = tiledb.Attr(dtype=np.int64)
        schema = tiledb.ArraySchema(
            domain=dom,
            attrs=(att,),
            sparse=True,
            allows_duplicates=True,
            cell_order=cell_order,
            tile_order=tile_order,
        )
        tiledb.Array.create(uri, schema)

        expected_data = np.random.randint(0, 10, 30)

        with tiledb.open(uri, mode="w") as T:
            T[np.arange(30)] = expected_data

        init_buffer_bytes = 200
        cfg = tiledb.Config(
            {
                "py.init_buffer_bytes": init_buffer_bytes,
                "py.exact_init_buffer_bytes": "true",
                "sm.query.sparse_unordered_with_dups.non_overlapping_ranges": non_overlapping_ranges,
            }
        )

        with tiledb.open(uri, mode="r", ctx=tiledb.Ctx(cfg)) as T:
            actual_data = T.query(order="G")[:][""]
            assert_array_equal(actual_data, expected_data)

    @pytest.mark.parametrize("exact_init_buffer_bytes", ["true", "false"])
    @pytest.mark.parametrize("non_overlapping_ranges", [True, False])
    def test_offset_can_fit_data_var_size_cannot(
        self, exact_init_buffer_bytes, non_overlapping_ranges
    ):
        """
        One condition that would be nice to get more coverage on is when the offset buffer can fit X cells, but the var size data of those cells cannot fit the buffer. In this case, the reader does adjust the results back.
        @Luc Rancourt so would we test this by having really large var-size content in each cell?
        Isaiah  4 days ago
        eg something like: we set buffers that can hold 100kb, but each var-len cell has 20kb, so we can read at most 5 cells into the data buffer, but theoretically the offsets buffer could hold many more?
        """
        tiledb.stats_enable()
        uri = self.path("test_incomplete_global_order")
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 4), tile=1, dtype=np.int64))
        att = tiledb.Attr(dtype=np.int64, var=True)
        schema = tiledb.ArraySchema(
            domain=dom,
            attrs=(att,),
            sparse=True,
            allows_duplicates=True,
        )
        tiledb.Array.create(uri, schema)

        with tiledb.open(uri, mode="w") as T:
            T[np.arange(5)] = np.array(
                [
                    np.random.randint(0, 10, 10000, dtype=np.int64),
                    np.random.randint(0, 10, 10000, dtype=np.int64),
                    np.random.randint(0, 10, 10000, dtype=np.int64),
                    np.random.randint(0, 10, 10000, dtype=np.int64),
                    np.random.randint(0, 10, 101, dtype=np.int64),
                ],
                dtype="O",
            )

        init_buffer_bytes = 160000
        cfg = tiledb.Config(
            {
                "py.init_buffer_bytes": init_buffer_bytes,
                "py.exact_init_buffer_bytes": exact_init_buffer_bytes,
                "sm.query.sparse_unordered_with_dups.non_overlapping_ranges": non_overlapping_ranges,
            }
        )

        with tiledb.open(uri, mode="r", ctx=tiledb.Ctx(cfg)) as T:
            qry = T.query()
            actual_data = qry[:][""]
            # assert_array_equal(actual_data, expected_data)
            # print(tiledb.main.python_internal_stats())

        tiledb.stats_disable()


class TestTest(DiskTestCase):
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


# if __name__ == '__main__':
#    # run a single example for in-process debugging
#    # better to use `pytest --gdb` if available
#    t = DenseArrayTest()
#    t.setUp()
#    t.test_array_1d()
