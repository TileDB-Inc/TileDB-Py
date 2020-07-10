# -*- coding: utf-8 -*-

from __future__ import absolute_import

import sys, os, io, re, platform, unittest, random, warnings

import numpy as np
from numpy.testing import assert_array_equal

import tiledb
from tiledb.tests.common import DiskTestCase, assert_subarrays_equal, rand_utf8, rand_ascii, rand_ascii_bytes

def safe_dump(obj):
    # TODO this doesn't actually redirect the C level stdout used by libtiledb dump
    #      functions...
    try:
        import io
        from contextlib import redirect_stdout
        with io.StringIO() as buf, redirect_stdout(buf):
            obj.dump()
    except ImportError:
        obj.dump()
    except Exception as exc:
        print("Exception occurred calling 'obj.dump()' with redirect.", exc,
              "\nTrying 'obj.dump()' alone.")
        obj.dump()

class VersionTest(unittest.TestCase):

    def test_version(self):
        v = tiledb.libtiledb.version()
        self.assertIsInstance(v, tuple)
        self.assertTrue(len(v) == 3)
        self.assertTrue(v[0] >= 1, "TileDB major version must be >= 1")


class StatsTest(DiskTestCase):

    def test_stats(self):
        tiledb.libtiledb.stats_enable()
        tiledb.libtiledb.stats_reset()
        tiledb.libtiledb.stats_disable()

        with tiledb.from_numpy(self.path("test_stats"), np.arange(10)) as T:
            assert_array_equal(T,np.arange(10))
            tiledb.stats_dump()

class Config(DiskTestCase):

    def test_config(self):
        config = tiledb.Config()
        config["sm.tile_cache_size"] = 100
        assert(repr(config) is not None)
        ctx = tiledb.Ctx(config)

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
        self.assertEqual(config["sm.tile_cache_size"], tiledb.Config()["sm.tile_cache_size"])

    def test_config_from_file(self):
        config_path = self.path("config")
        with open(config_path, "w") as fh:
            fh.write("sm.tile_cache_size 100")
        config = tiledb.Config.load(config_path)
        self.assertEqual(config["sm.tile_cache_size"], "100")

    def test_ctx_config_from_file(self):
        config_path = self.path("config")
        with open(config_path, "w") as fh:
            fh.write("sm.tile_cache_size 100")
        ctx = tiledb.Ctx(config=tiledb.Config.load(config_path))
        config = ctx.config()
        self.assertEqual(config["sm.tile_cache_size"], "100")

    def test_ctx_config_dict(self):
        ctx = tiledb.Ctx(config={"sm.tile_cache_size": '100'})
        config = ctx.config()
        self.assertIsInstance(config, tiledb.Config)
        self.assertEqual(config["sm.tile_cache_size"], '100')


class GroupTestCase(DiskTestCase):

    def setUp(self):
        super(GroupTestCase, self).setUp()

        ctx = tiledb.Ctx()
        self.group1 = self.path("group1")
        self.group2 = self.path("group1/group2")
        self.group3 = self.path("group1/group3")
        self.group4 = self.path("group1/group3/group4")

        tiledb.group_create(self.group1, ctx=ctx)
        tiledb.group_create(self.group2, ctx=ctx)
        tiledb.group_create(self.group3, ctx=ctx)
        tiledb.group_create(self.group4, ctx=ctx)

    def is_group(self, ctx, uri):
        return tiledb.object_type(uri, ctx=ctx) == "group"


class GroupTest(GroupTestCase):

    def test_is_group(self):
        ctx = tiledb.Ctx()
        self.assertTrue((ctx, self.group1))
        self.assertTrue(self.is_group(ctx, self.group2))
        self.assertTrue(self.is_group(ctx, self.group3))
        self.assertTrue(self.is_group(ctx, self.group4))

    def test_walk_group(self):
        ctx = tiledb.Ctx()

        groups = []
        def append_to_groups(path, obj):
            groups.append((os.path.normpath(path), obj))

        tiledb.walk(self.path(""), append_to_groups, order="preorder", ctx=ctx)

        groups.sort()

        self.assertTrue(groups[0][0].endswith(self.group1) and groups[0][1] == "group")
        self.assertTrue(groups[1][0].endswith(self.group2) and groups[1][1] == "group")
        self.assertTrue(groups[2][0].endswith(self.group3) and groups[2][1] == "group")
        self.assertTrue(groups[3][0].endswith(self.group4) and groups[3][1] == "group")

        groups = []

        tiledb.walk(self.path(""), append_to_groups, order="postorder", ctx=ctx)

        self.assertTrue(groups[0][0].endswith(self.group2) and groups[0][1] == "group")
        self.assertTrue(groups[1][0].endswith(self.group4) and groups[1][1] == "group")
        self.assertTrue(groups[2][0].endswith(self.group3) and groups[2][1] == "group")
        self.assertTrue(groups[3][0].endswith(self.group1) and groups[3][1] == "group")

    def test_remove_group(self):
        ctx = tiledb.Ctx()

        tiledb.remove(self.group3, ctx=ctx)

        self.assertFalse(self.is_group(ctx, self.group3))
        self.assertFalse(self.is_group(ctx, self.group4))

    def test_move_group(self):
        ctx = tiledb.Ctx()

        self.assertTrue(self.is_group(ctx, self.group2))
        tiledb.move(self.group2, self.group2 + "_moved", ctx=ctx)
        self.assertFalse(self.is_group(ctx, self.group2))
        self.assertTrue(self.is_group(ctx, self.group2 + "_moved"))


class DimensionTest(unittest.TestCase):

    def test_minimal_dimension(self):
        ctx = tiledb.Ctx()
        dim = tiledb.Dim(domain=(0, 4), tile=5, ctx=ctx)
        self.assertEqual(dim.name, "__dim_0", "automatic dimension name is incorrect")
        self.assertEqual(dim.shape, (5,))
        self.assertEqual(dim.tile, 5)

    def test_dimension(self):
        ctx = tiledb.Ctx()
        dim = tiledb.Dim(name="d1", ctx=ctx, domain=(0, 3), tile=2)
        self.assertEqual(dim.name, "d1")
        self.assertEqual(dim.shape, (4,))
        self.assertEqual(dim.tile, 2)

    def test_datetime_dimension(self):
        ctx = tiledb.Ctx()

        # Regular usage
        dim = tiledb.Dim(name="d1", ctx=ctx, domain=(np.datetime64('2010-01-01'), np.datetime64('2020-01-01')),
                         tile=np.timedelta64(20, 'D'), dtype=np.datetime64('', 'D'))
        self.assertEqual(dim.dtype, np.dtype(np.datetime64('', 'D')))
        self.assertEqual(dim.tile, np.timedelta64(20, 'D'))
        self.assertNotEqual(dim.tile, np.timedelta64(21, 'D'))
        self.assertNotEqual(dim.tile, np.timedelta64(20, 'W')) # Sanity check unit
        self.assertTupleEqual(dim.domain, (np.datetime64('2010-01-01'), np.datetime64('2020-01-01')))

        # No tile extent specified
        with self.assertRaises(tiledb.TileDBError):
            tiledb.Dim(name="d1", ctx=ctx, domain=(np.datetime64('2010-01-01'), np.datetime64('2020-01-01')),
                       dtype=np.datetime64('', 'D'))

        # Integer tile extent is ok
        dim = tiledb.Dim(name="d1", ctx=ctx, domain=(np.datetime64('2010-01-01'), np.datetime64('2020-01-01')),
                         tile=20, dtype=np.datetime64('', 'D'))
        self.assertEqual(dim.dtype, np.dtype(np.datetime64('', 'D')))
        self.assertEqual(dim.tile, np.timedelta64(20, 'D'))

        # Year resolution
        dim = tiledb.Dim(name="d1", ctx=ctx, domain=(np.datetime64('2010'), np.datetime64('2020')),
                         tile=5, dtype=np.datetime64('', 'Y'))
        self.assertEqual(dim.dtype, np.dtype(np.datetime64('', 'Y')))
        self.assertEqual(dim.tile, np.timedelta64(5, 'Y'))
        self.assertTupleEqual(dim.domain, (np.datetime64('2010', 'Y'), np.datetime64('2020', 'Y')))

        # End domain promoted to day resolution
        dim = tiledb.Dim(name="d1", ctx=ctx, domain=(np.datetime64('2010-01-01'), np.datetime64('2020')),
                         tile=2, dtype=np.datetime64('', 'D'))
        self.assertEqual(dim.tile, np.timedelta64(2, 'D'))
        self.assertTupleEqual(dim.domain, (np.datetime64('2010-01-01', 'D'), np.datetime64('2020-01-01', 'D')))

        # Domain values can't be integral
        with self.assertRaises(TypeError):
            dim = tiledb.Dim(name="d1", ctx=ctx, domain=(-10, 10), tile=2, dtype=np.datetime64('', 'D'))


class DomainTest(unittest.TestCase):

    def test_domain(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(
            tiledb.Dim("d1", (1, 4), 2, dtype='u8'),
            tiledb.Dim("d2", (1, 4), 2, dtype='u8'))
        safe_dump(dom)
        self.assertEqual(dom.ndim, 2)
        self.assertEqual(dom.dtype, np.dtype("uint64"))
        self.assertEqual(dom.shape, (4, 4))

        # check that we can iterate over the dimensions
        dim_names = [dim.name for dim in dom]
        self.assertEqual(["d1", "d2"], dim_names)

        # check that we can access dim by name
        dim_d1 = dom.dim("d1")
        self.assertEqual(dim_d1, dom.dim(0))

    def test_datetime_domain(self):
        ctx = tiledb.Ctx()
        dim = tiledb.Dim(name="d1", ctx=ctx, domain=(np.datetime64('2010-01-01'), np.datetime64('2020-01-01')),
                         tile=np.timedelta64(20, 'D'), dtype=np.datetime64('', 'D'))
        dom = tiledb.Domain(dim)
        self.assertEqual(dom.dtype, np.datetime64('', 'D'))

    def test_domain_mixed_names_error(self):
        ctx = tiledb.Ctx()
        with self.assertRaises(tiledb.TileDBError):
            tiledb.Domain(
                tiledb.Dim("d1", (1, 4), 2, dtype='u8'),
                tiledb.Dim("__dim_0", (1, 4), 2, dtype='u8'))

class AttributeTest(unittest.TestCase):

    def test_minimal_attribute(self):
        ctx = tiledb.Ctx()
        attr = tiledb.Attr(ctx=ctx)
        self.assertTrue(attr.isanon)
        self.assertEqual(attr.name, u"")
        self.assertEqual(attr.dtype, np.float_)
        #self.assertEqual(attr.compressor, (None, -1))

    def test_attribute(self):
        ctx = tiledb.Ctx()
        attr = tiledb.Attr("foo", ctx=ctx)
        safe_dump(attr)
        self.assertEqual(attr.name, "foo")
        self.assertEqual(attr.dtype, np.float64,
                         "default attribute type is float64")
        #compressor, level = attr.compressor
        #self.assertEqual(compressor, None, "default to no compression")
        #self.assertEqual(level, -1, "default compression level when none is specified")

    def test_full_attribute(self):
        ctx = tiledb.Ctx()
        filter_list = tiledb.FilterList([tiledb.ZstdFilter(10, ctx=ctx)], ctx=ctx)
        attr = tiledb.Attr("foo", dtype=np.int64, filters=filter_list, ctx=ctx)
        #attr = tiledb.Attr(ctx, "foo", dtype=np.int64, compressor=("zstd", 10))
        filter_list = tiledb.FilterList([tiledb.ZstdFilter(10, ctx=ctx)], ctx=ctx)
        attr = tiledb.Attr("foo", ctx=ctx, dtype=np.int64, filters=filter_list)
        safe_dump(attr)
        self.assertEqual(attr.name, "foo")
        self.assertEqual(attr.dtype, np.int64)

        # <todo>
        #compressor, level = attr.compressor
        #self.assertEqual(compressor, "zstd")
        #self.assertEqual(level, 10)

    def test_ncell_attribute(self):
        ctx = tiledb.Ctx()
        dtype = np.dtype([("", np.int32), ("", np.int32), ("", np.int32)])
        attr = tiledb.Attr("foo", ctx=ctx, dtype=dtype)

        self.assertEqual(attr.dtype, dtype)
        self.assertEqual(attr.ncells, 3)

        # dtype subarrays not supported
        with self.assertRaises(TypeError):
            tiledb.Attr("foo", ctx=ctx, dtype=np.dtype((np.int32, 2)))

        # mixed type record arrays not supported
        with self.assertRaises(TypeError):
            tiledb.Attr("foo", ctx=ctx, dtype=np.dtype([("", np.float32), ("", np.int32)]))

    def test_ncell_bytes_attribute(self):
        ctx = tiledb.Ctx()
        dtype = np.dtype((np.bytes_, 10))
        attr = tiledb.Attr("foo", ctx=ctx, dtype=dtype)

        self.assertEqual(attr.dtype, dtype)
        self.assertEqual(attr.ncells, 10)

    def test_vararg_attribute(self):
        ctx = tiledb.Ctx()
        attr = tiledb.Attr("foo", ctx=ctx, dtype=np.bytes_)
        self.assertEqual(attr.dtype, np.dtype(np.bytes_))
        self.assertTrue(attr.isvar)

    def test_datetime_attribute(self):
        ctx = tiledb.Ctx()
        attr = tiledb.Attr("foo", ctx=ctx, dtype=np.datetime64('', 'D'))
        self.assertEqual(attr.dtype, np.dtype(np.datetime64('', 'D')))
        self.assertNotEqual(attr.dtype, np.dtype(np.datetime64))
        self.assertNotEqual(attr.dtype, np.dtype(np.datetime64('', 'Y')))

    def test_filter(self):
        ctx = tiledb.Ctx()
        gzip_filter = tiledb.libtiledb.GzipFilter(ctx=ctx, level=10)
        self.assertIsInstance(gzip_filter, tiledb.libtiledb.Filter)
        self.assertEqual(gzip_filter.level, 10)

        bw_filter = tiledb.libtiledb.BitWidthReductionFilter(ctx=ctx, window=10)
        self.assertIsInstance(bw_filter, tiledb.libtiledb.Filter)
        self.assertEqual(bw_filter.window, 10)

        filter_list = tiledb.libtiledb.FilterList([gzip_filter, bw_filter], chunksize=1024, ctx=ctx)
        self.assertEqual(filter_list.chunksize, 1024)
        self.assertEqual(len(filter_list), 2)
        self.assertEqual(filter_list[0].level, gzip_filter.level)
        self.assertEqual(filter_list[1].window, bw_filter.window)

        # test filter list iteration
        self.assertEqual(len(list(filter_list)), 2)

        # test `filters` kwarg accepts python list of filters
        tiledb.Attr("foo", ctx=ctx, dtype=np.int64, filters=[gzip_filter])
        tiledb.Attr("foo", ctx=ctx, dtype=np.int64, filters=(gzip_filter,))

        attr = tiledb.Attr("foo",
                           ctx=ctx,
                           dtype=np.int64,
                           filters=filter_list)

        self.assertEqual(len(attr.filters), 2)
        self.assertEqual(attr.filters.chunksize, filter_list.chunksize)

    def test_filter_list(self):
        ctx = tiledb.Ctx()
        # should be constructible without a `filters` keyword arg set
        filter_list1 = tiledb.FilterList(ctx=ctx)
        filter_list1.append(tiledb.GzipFilter(ctx=ctx))
        self.assertEqual(len(filter_list1), 1)

        filter_list2 = [x for x in filter_list1]
        attr = tiledb.Attr(filters=filter_list2)
        self.assertEqual(len(attr.filters), 1)


class ArraySchemaTest(unittest.TestCase):

    def test_unique_attributes(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(
            tiledb.Dim("d1", (1, 4), 2, dtype='u8', ctx=ctx),
            tiledb.Dim("d2", (1, 4), 2, dtype='u8', ctx=ctx),
            ctx=ctx)

        attr1 = tiledb.Attr("foo", ctx=ctx, dtype=float)
        attr2 = tiledb.Attr("foo", ctx=ctx, dtype=int)

        with self.assertRaises(tiledb.TileDBError):
            tiledb.ArraySchema(ctx=ctx, domain=dom, attrs=(attr1, attr2))

    def test_dense_array_schema(self):
        ctx = tiledb.Ctx()
        domain = tiledb.Domain(
            tiledb.Dim(domain=(1, 8), tile=2, ctx=ctx),
            tiledb.Dim(domain=(1, 8), tile=2, ctx=ctx))
        a1 = tiledb.Attr("val", ctx=ctx, dtype='f8')
        schema = tiledb.ArraySchema(ctx=ctx, domain=domain, attrs=(a1,))
        self.assertFalse(schema.sparse)
        self.assertEqual(schema.cell_order, "row-major")
        self.assertEqual(schema.tile_order, "row-major")
        self.assertEqual(schema.domain, domain)
        self.assertEqual(schema.ndim, 2)
        self.assertEqual(schema.shape, (8, 8))
        self.assertEqual(schema.nattr, 1)
        self.assertEqual(schema.domain.homogeneous, True)
        self.assertEqual(schema.attr(0), a1)
        self.assertTrue(schema.has_attr("val"))
        self.assertFalse(schema.has_attr("nononoattr"))
        self.assertEqual(schema,
            tiledb.ArraySchema(ctx=ctx, domain=domain, attrs=(a1,)))
        self.assertNotEqual(schema,
            tiledb.ArraySchema(domain=domain, attrs=(a1,), sparse=True, ctx=ctx))
        with self.assertRaises(tiledb.TileDBError):
            schema.allows_duplicates
        # test iteration over attributes
        self.assertEqual(list(schema), [a1])

    def test_dense_array_schema_fp_domain_error(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(
            tiledb.Dim(ctx=ctx, domain=(1, 8), tile=2, dtype=np.float64), ctx=ctx)
        att = tiledb.Attr("val", ctx=ctx, dtype=np.float64)

        with self.assertRaises(tiledb.TileDBError):
            tiledb.ArraySchema(ctx=ctx, domain=dom, attrs=(att,))

    def test_sparse_schema(self):
        ctx = tiledb.Ctx()

        # create dimensions
        d1 = tiledb.Dim("d1", domain=(1, 1000), tile=10, dtype="uint64", ctx=ctx)
        d2 = tiledb.Dim("d2", domain=(101, 10000), tile=100, dtype="uint64", ctx=ctx)

        # create domain
        domain = tiledb.Domain(d1, d2, ctx=ctx)

        # create attributes
        a1 = tiledb.Attr("a1", dtype="int32,int32,int32", ctx=ctx)
        a2 = tiledb.Attr("a2",
                         filters=tiledb.FilterList([tiledb.GzipFilter(-1, ctx=ctx)], ctx=ctx),
                         dtype="float32", ctx=ctx)

        # create sparse array with schema
        coords_filters = tiledb.FilterList([tiledb.ZstdFilter(4, ctx=ctx)], ctx=ctx)
        offsets_filters = tiledb.FilterList([tiledb.LZ4Filter(5, ctx=ctx)], ctx=ctx)

        schema = tiledb.ArraySchema(domain=domain,
                                    attrs=(a1, a2),
                                    capacity=10,
                                    cell_order='col-major',
                                    tile_order='row-major',
                                    allows_duplicates=True,
                                    sparse=True,
                                    coords_filters=coords_filters,
                                    offsets_filters=offsets_filters,
                                    ctx=ctx)

        safe_dump(schema)
        self.assertTrue(schema.sparse)
        self.assertEqual(schema.capacity, 10)
        self.assertEqual(schema.cell_order, "col-major")
        self.assertEqual(schema.tile_order, "row-major")

        # <todo>
        #self.assertEqual(schema.coords_compressor, ('zstd', 4))
        #self.assertEqual(schema.offsets_compressor, ('lz4', 5))

        self.assertEqual(schema.domain, domain)
        self.assertEqual(schema.ndim, 2)
        self.assertEqual(schema.shape, (1000, 9900))
        self.assertEqual(schema.nattr, 2)
        self.assertEqual(schema.attr(0), a1)
        self.assertEqual(schema.attr("a2"), a2)
        self.assertEqual(schema.allows_duplicates, True)
        self.assertEqual(schema,
                         tiledb.ArraySchema(
                                    domain=domain,
                                    attrs=(a1, a2),
                                    capacity=10,
                                    cell_order='col-major',
                                    tile_order='row-major',
                                    allows_duplicates=True,
                                    sparse=True,
                                    coords_filters=coords_filters,
                                    offsets_filters=offsets_filters,
                                    ctx=ctx))

        # test iteration over attributes
        self.assertEqual(list(schema), [a1, a2])

    def test_sparse_schema_filter_list(self):
        ctx = tiledb.Ctx()

        # create dimensions
        d1 = tiledb.Dim("d1", domain=(1, 1000), tile=10, dtype="uint64", ctx=ctx)
        d2 = tiledb.Dim("d2", domain=(101, 10000), tile=100, dtype="uint64", ctx=ctx)

        # create domain
        domain = tiledb.Domain(d1, d2, ctx=ctx)

        # create attributes
        a1 = tiledb.Attr("a1", dtype="int32,int32,int32", ctx=ctx)
        #a2 = tiledb.Attr(ctx, "a2", compressor=("gzip", -1), dtype="float32")
        filter_list = tiledb.FilterList([tiledb.GzipFilter(ctx=ctx)], ctx=ctx)
        a2 = tiledb.Attr("a2", filters=filter_list, dtype="float32", ctx=ctx)

        off_filters_pylist = [tiledb.libtiledb.ZstdFilter(level=10,ctx=ctx)]
        off_filters = tiledb.libtiledb.FilterList(
                        filters=off_filters_pylist,
                        chunksize=2048,
                        ctx=ctx)

        coords_filters_pylist = [tiledb.libtiledb.Bzip2Filter(level=5, ctx=ctx)]
        coords_filters = tiledb.libtiledb.FilterList(
                        filters=coords_filters_pylist,
                        chunksize=4096, ctx=ctx)

        # create sparse array with schema
        schema = tiledb.ArraySchema(domain=domain,
                                    attrs=(a1, a2),
                                    capacity=10,
                                    cell_order='col-major',
                                    tile_order='row-major',
                                    coords_filters=coords_filters,
                                    offsets_filters=off_filters,
                                    sparse=True,
                                    ctx=ctx)
        safe_dump(schema)
        self.assertTrue(schema.sparse)

        # make sure we can construct ArraySchema with python lists of filters
        schema2 = tiledb.ArraySchema(domain=domain,
                                    attrs=(a1, a2),
                                    capacity=10,
                                    cell_order='col-major',
                                    tile_order='row-major',
                                    coords_filters=coords_filters_pylist,
                                    offsets_filters=off_filters,
                                    sparse=True,
                                    ctx=ctx)
        self.assertEqual(len(schema2.coords_filters), 1)
        self.assertEqual(len(schema2.offsets_filters), 1)

    def test_mixed_string_schema(self):
        ctx = tiledb.Ctx()
        dims = [
            tiledb.Dim(name="dpos", ctx=ctx, domain=(-100.0, 100.0), tile=10, dtype=np.float64),
            tiledb.Dim(name="str_index", domain=(None,None), tile=None, dtype=np.bytes_)
        ]
        dom = tiledb.Domain(*dims)
        attrs = [
            tiledb.Attr(name="val", dtype=np.float64, ctx=ctx)
        ]

        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=True, ctx=ctx)

        self.assertTrue(schema.domain.has_dim("str_index"))
        self.assertFalse(schema.domain.has_dim("nonono_str_index"))
        self.assertTrue(schema.domain.dim("str_index").isvar)
        self.assertFalse(schema.domain.dim("dpos").isvar)
        self.assertEqual(schema.domain.dim("dpos").dtype, np.double)
        self.assertEqual(schema.domain.dim("str_index").dtype, np.bytes_)
        self.assertFalse(schema.domain.homogeneous)

class ArrayTest(DiskTestCase):

    def create_array_schema(self, ctx):
        domain = tiledb.Domain(
            tiledb.Dim(domain=(1, 8), tile=2, ctx=ctx),
            tiledb.Dim(domain=(1, 8), tile=2, ctx=ctx),
            ctx=ctx)
        a1 = tiledb.Attr("val", dtype='f8', ctx=ctx)
        return tiledb.ArraySchema(domain=domain, attrs=(a1,), ctx=ctx)

    def test_array_create(self):
        config = tiledb.Config()
        config["sm.consolidation.step_min_frag"] = 0
        config["sm.consolidation.steps"] = 1
        ctx = tiledb.Ctx(config=config)
        schema = self.create_array_schema(ctx)

        # persist array schema
        tiledb.libtiledb.Array.create(self.path("foo"), schema)

        # these should be no-ops
        #   full signature
        tiledb.consolidate(self.path("foo"), config=config, ctx=ctx)
        #   kw signature
        tiledb.consolidate(uri=self.path("foo"), ctx=ctx)

        # load array in readonly mode
        array = tiledb.libtiledb.Array(self.path("foo"), mode='r', ctx=ctx)
        self.assertTrue(array.isopen)
        self.assertEqual(array.schema, schema)
        self.assertEqual(array.mode, 'r')
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
        config = tiledb.Config()
        config["sm.consolidation.step_min_frag"] = 0
        config["sm.consolidation.steps"] = 1
        ctx = tiledb.Ctx(config=config)
        schema = self.create_array_schema(ctx)

        with self.assertRaises(TypeError):
            tiledb.libtiledb.Array.create(self.path("foo"), schema, ctx="foo")

        # persist array schema
        tiledb.libtiledb.Array.create(self.path("foo"), schema, ctx=tiledb.Ctx())

    def test_array_create_encrypted(self):
        config = tiledb.Config()
        config["sm.consolidation.step_min_frags"] = 0
        config["sm.consolidation.steps"] = 1
        ctx = tiledb.Ctx(config=config)
        schema = self.create_array_schema(ctx)
        # persist array schema
        tiledb.libtiledb.Array.create(self.path("foo"), schema,
                                      key=b"0123456789abcdeF0123456789abcdeF")

        # check that we can open the array sucessfully
        for key in (b"0123456789abcdeF0123456789abcdeF", "0123456789abcdeF0123456789abcdeF"):
            with tiledb.libtiledb.Array(self.path("foo"), ctx=ctx, mode='r', key=key) as array:
                self.assertTrue(array.isopen)
                self.assertEqual(array.schema, schema)
                self.assertEqual(array.mode, 'r')
            tiledb.consolidate(uri=self.path("foo"), config=config, key=key, ctx=ctx)

        # check that opening the array with the wrong key fails:
        with self.assertRaises(tiledb.TileDBError):
            tiledb.libtiledb.Array(self.path("foo"), ctx=ctx, mode='r',
                                   key=b"0123456789abcdeF0123456789abcdeX")

        # check that opening the array with the wrong key length fails:
        with self.assertRaises(tiledb.TileDBError):
            tiledb.libtiledb.Array(self.path("foo"), ctx=ctx, mode='r',
                                   key=b"0123456789abcdeF0123456789abcde")

        # check that consolidating the array with the wrong key fails:
        with self.assertRaises(tiledb.TileDBError):
            tiledb.consolidate(self.path("foo"), config=config,
                               key=b"0123456789abcdeF0123456789abcde", ctx=ctx)

    def test_array_doesnt_exist(self):
        ctx = tiledb.Ctx()
        with self.assertRaises(tiledb.TileDBError):
            tiledb.libtiledb.Array(self.path("foo"), mode='r', ctx=ctx)

    def test_create_schema_matches(self):
        ctx = tiledb.Ctx()
        dims = (tiledb.Dim(ctx=ctx, domain=(0, 6), tile=2),)
        dom = tiledb.Domain(*dims, ctx=ctx)
        att = tiledb.Attr(ctx=ctx, dtype=np.byte)

        schema = tiledb.ArraySchema(ctx=ctx, domain=dom, attrs=(att,), sparse=True)
        uri = self.path('s1')
        with self.assertRaises(ValueError):
            tiledb.DenseArray.create(uri, schema)

        dense_schema = tiledb.ArraySchema(ctx=ctx, domain=dom, attrs=(att,))
        uri = self.path('d1')
        with self.assertRaises(ValueError):
            tiledb.SparseArray.create(uri, dense_schema)

        class MySparseArray(tiledb.SparseArray):
            pass

        with self.assertRaises(ValueError):
            MySparseArray.create(uri, dense_schema)

class DenseArrayTest(DiskTestCase):

    def test_array_1d(self):
        A = np.arange(1050)

        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 1049), tile=100, dtype=np.int64, ctx=ctx), ctx=ctx)
        att = tiledb.Attr(ctx=ctx, dtype=A.dtype)
        schema = tiledb.ArraySchema(ctx=ctx, domain=dom, attrs=(att,))
        tiledb.DenseArray.create(self.path("foo"), schema)

        with tiledb.DenseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            self.assertEqual(len(A), len(T))
            self.assertEqual(A.ndim, T.ndim)
            self.assertEqual(A.shape, T.shape)

            self.assertEqual(1, T.nattr)
            self.assertEqual(A.dtype, T.attr(0).dtype)
            self.assertEqual(T.dim(T.schema.domain.dim(0).name), T.dim(0))
            with self.assertRaises(ValueError): T.dim(1.0)

            self.assertIsInstance(T.timestamp, int)
            self.assertTrue(T.timestamp > 0)

            # check empty array
            B = T[:]

            self.assertEqual(A.shape, B.shape)
            self.assertEqual(A.dtype, B.dtype)
            self.assertIsNone(T.nonempty_domain())

        with tiledb.DenseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            # check set array
            T[:] = A

        read1_timestamp = -1
        with tiledb.DenseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            self.assertEqual(((0, 1049),), T.nonempty_domain())

            # check timestamp
            read1_timestamp = T.timestamp
            self.assertTrue(read1_timestamp > 0)

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
            self.assertEqual(A[0:1], T[0:np.uint16(1)])
            self.assertEqual(A[0:1], T[np.int64(0):1])
            with self.assertRaises(IndexError):
                # this is a consequence of NumPy promotion rules
                self.assertEqual(A[0:1], T[np.uint64(0):1])

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

        with tiledb.DenseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            # check partial assignment
            B = np.arange(1e5, 2e5).astype(A.dtype)
            T[190:310] = B[190:310]

        read2_timestamp = -1
        with tiledb.DenseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            assert_array_equal(A[:190], T[:190])
            assert_array_equal(B[190:310], T[190:310])
            assert_array_equal(A[310:], T[310:])

            # test timestamps are updated
            read2_timestamp = T.timestamp
            self.assertTrue(read2_timestamp > read1_timestamp)

    def test_array_1d_set_scalar(self):
        A = np.zeros(50)

        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim(ctx=ctx, domain=(0, 49), tile=50), ctx=ctx)
        att = tiledb.Attr(dtype=A.dtype, ctx=ctx)
        schema = tiledb.ArraySchema(dom, (att,), ctx=ctx)

        tiledb.DenseArray.create(self.path("foo"), schema)

        with tiledb.DenseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            T[:] = A

        with tiledb.DenseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            assert_array_equal(A, T[:])

        with tiledb.DenseArray(self.path("foo"), mode='w') as T:
            value = -1,3,10
            A[0], A[1], A[3] = value
            T[0], T[1], T[3] = value
        with tiledb.DenseArray(self.path("foo"), mode='r') as T:
            assert_array_equal(A, T[:])

        for value in (-1, 3, 10):
            with tiledb.DenseArray(self.path("foo"), mode='w') as T:
                A[5:25] = value
                T[5:25] = value
            with tiledb.DenseArray(self.path("foo"), mode='r') as T:
                assert_array_equal(A, T[:])
            with tiledb.DenseArray(self.path("foo"), mode='w', ctx=ctx) as T:
                A[:] = value
                T[:] = value
            with tiledb.DenseArray(self.path("foo"), mode='r', ctx=ctx) as T:
                assert_array_equal(A, T[:])

    def test_array_id_point_queries(self):
        #TODO: handle queries like T[[2, 5, 10]] = ?
        pass

    def test_array_2d(self):
        A = np.arange(10000).reshape((1000, 10))

        ctx = tiledb.Ctx()
        dom = tiledb.Domain(
                       tiledb.Dim(domain=(0, 999), tile=100, ctx=ctx),
                       tiledb.Dim(domain=(0, 9), tile=2, ctx=ctx),
                       ctx=ctx)
        att = tiledb.Attr(dtype=A.dtype, ctx=ctx)
        schema = tiledb.ArraySchema(dom, (att,), ctx=ctx)

        tiledb.DenseArray.create(self.path("foo"), schema)

        with tiledb.DenseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            self.assertEqual(len(A), len(T))
            self.assertEqual(A.ndim, T.ndim)
            self.assertEqual(A.shape, T.shape)

            self.assertEqual(1, T.nattr)
            self.assertEqual(A.dtype, T.attr(0).dtype)

            # check that the non-empty domain is None
            self.assertIsNone(T.nonempty_domain())

        with tiledb.DenseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            # Set data
            T[:] = A

        with tiledb.DenseArray(self.path("foo"), mode='r', ctx=ctx) as T:
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

        with tiledb.DenseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            # check partial assignment
            B = np.arange(10000, 20000).reshape((1000, 10))
            T[190:310, 3:7] = B[190:310, 3:7]

        with tiledb.DenseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            assert_array_equal(A[:190], T[:190])
            assert_array_equal(A[:, :3], T[:, :3])
            assert_array_equal(B[190:310, 3:7], T[190:310, 3:7])
            assert_array_equal(A[310:], T[310:])
            assert_array_equal(A[:, 7:], T[:, 7:])

    def test_fixed_string(self):
        ctx = tiledb.Ctx()
        a = np.array(['ab', 'cd', 'ef', 'gh', 'ij', 'kl'], dtype='|S2')
        with tiledb.from_numpy(self.path('fixed_string'), a) as T:
            with tiledb.open(self.path('fixed_string')) as R:
                self.assertEqual(T.dtype, R.dtype)
                self.assertEqual(R.attr(0).ncells, 2)
                assert_array_equal(T,R)

    def test_ncell_int(self):
        a = np.array([(1, 2), (3, 4), (5, 6)], dtype=[("", np.int16), ("", np.int16)])
        with tiledb.from_numpy(self.path('ncell_int16'), a) as T:
            with tiledb.open(self.path('ncell_int16')) as R:
                self.assertEqual(T.dtype, R.dtype)
                self.assertEqual(R.attr(0).ncells, 2)
                assert_array_equal(T,R)
                assert_array_equal(T, R.multi_index[0:2][''])


    def test_open_with_timestamp(self):
        import time
        A = np.zeros(3)

        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim(ctx=ctx, domain=(0, 2), tile=3, dtype=np.int64), ctx=ctx)
        att = tiledb.Attr(ctx=ctx, dtype=A.dtype)
        schema = tiledb.ArraySchema(ctx=ctx, domain=dom, attrs=(att,))
        tiledb.DenseArray.create(self.path("foo"), schema)

        # write
        with tiledb.DenseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            T[:] = A

        read1_timestamp = -1
        with tiledb.DenseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            read1_timestamp = T.timestamp
            self.assertEqual(T[0], 0)
            self.assertEqual(T[1], 0)
            self.assertEqual(T[2], 0)

        # sleep 200ms and write
        time.sleep(0.2)
        with tiledb.DenseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            T[0:1] = 1

        read2_timestamp = -1
        with tiledb.DenseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            read2_timestamp = T.timestamp
            self.assertTrue(read2_timestamp > read1_timestamp)

        # sleep 200ms and write
        time.sleep(0.2)
        with tiledb.DenseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            T[1:2] = 2

        read3_timestamp = -1
        with tiledb.DenseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            read3_timestamp = T.timestamp
            self.assertTrue(read3_timestamp > read2_timestamp > read1_timestamp)

        # read at first timestamp
        with tiledb.DenseArray(self.path("foo"), timestamp=read1_timestamp, mode='r', ctx=ctx) as T:
            self.assertEqual(T[0], 0)
            self.assertEqual(T[1], 0)
            self.assertEqual(T[2], 0)

        # read at second timestamp
        with tiledb.DenseArray(self.path("foo"), timestamp=read2_timestamp, mode='r', ctx=ctx) as T:
            self.assertEqual(T[0], 1)
            self.assertEqual(T[1], 0)
            self.assertEqual(T[2], 0)

        # read at third timestamp
        with tiledb.DenseArray(self.path("foo"), timestamp=read3_timestamp, mode='r', ctx=ctx) as T:
            self.assertEqual(T[0], 1)
            self.assertEqual(T[1], 2)
            self.assertEqual(T[2], 0)

    def test_ncell_attributes(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim(ctx=ctx, domain=(0, 9), tile=10, dtype=int), ctx=ctx)
        attr = tiledb.Attr(ctx=ctx, dtype=[("", np.int32), ("", np.int32), ("", np.int32)])
        schema = tiledb.ArraySchema(ctx=ctx, domain=dom, attrs=(attr,))
        tiledb.DenseArray.create(self.path("foo"), schema)

        A = np.ones((10,), dtype=[("", np.int32), ("", np.int32), ("", np.int32)])
        self.assertEqual(A.dtype, attr.dtype)

        with tiledb.DenseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            T[:] = A
        with tiledb.DenseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            assert_array_equal(A, T[:])
            assert_array_equal(A[:5], T[:5])

    def test_complex_attributes(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim(ctx=ctx, domain=(0, 9), tile=10,
                                       dtype=int), ctx=ctx)
        attr = tiledb.Attr(ctx=ctx, dtype=np.complex64)
        schema = tiledb.ArraySchema(ctx=ctx, domain=dom, attrs=(attr,))
        tiledb.DenseArray.create(self.path("foo"), schema)

        A = np.random.rand(20).astype(np.float32).view(dtype=np.complex64)

        self.assertEqual(schema, tiledb.schema_like(A, dim_dtype=int))
        self.assertEqual(A.dtype, attr.dtype)

        with tiledb.DenseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            T[:] = A

        with tiledb.DenseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            assert_array_equal(A, T[:])
            assert_array_equal(A[:5], T[:5])

    def test_multiple_attributes(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(
                       tiledb.Dim(domain=(0, 1), tile=1, dtype=np.int64, ctx=ctx),
                       tiledb.Dim(domain=(0, 3), tile=4, dtype=np.int64, ctx=ctx),
                       ctx=ctx)
        attr_int = tiledb.Attr("ints", dtype=int, ctx=ctx)
        attr_float = tiledb.Attr("floats", dtype=float, ctx=ctx)
        schema = tiledb.ArraySchema(ctx=ctx,
                                    domain=dom,
                                    attrs=(attr_int, attr_float))
        tiledb.DenseArray.create(self.path("foo"), schema)

        V_ints = np.array([[0, 1, 2, 3,],
                           [4, 6, 7, 5]])
        V_floats = np.array([[0.0, 1.0, 2.0, 3.0,],
                             [4.0, 6.0, 7.0, 5.0]])

        V = {"ints": V_ints, "floats": V_floats}
        with tiledb.DenseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            T[:] = V

        # check setting attribute in different order from Attr definition
        #   https://github.com/TileDB-Inc/TileDB-Py/issues/299
        V2 = {"floats": V_floats, "ints": V_ints}
        with tiledb.DenseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            T[:] = V

        import tiledb.core as core
        with tiledb.DenseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            R = T[:]
            assert_array_equal(V["ints"], R["ints"])
            assert_array_equal(V["floats"], R["floats"])

            R = T.query(attrs=("ints",))[1:3]
            assert_array_equal(V["ints"][1:3], R["ints"])

            R = T.query(attrs=("floats",), order='F')[:]
            self.assertTrue(R["floats"].flags.f_contiguous)

            R = T.query(attrs=("ints",), coords=True)[0, 0:3]
            self.assertTrue("__dim_0" in R)
            self.assertTrue("__dim_1" in R)
            assert_array_equal(R["__dim_0"], np.array([0,0,0]))
            assert_array_equal(R["__dim_1"], np.array([0,1,2]))

            # Global order returns results as a linear buffer
            R = T.query(attrs=("ints",), order='G')[:]
            self.assertEqual(R["ints"].shape, (8,))

            with self.assertRaises(tiledb.TileDBError):
                T.query(attrs=("unknown"))[:]

            # Ensure that query only returns specified attributes
            q = core.PyQuery(ctx, T, ("ints",), False, 0)
            q.set_ranges([[(0,1)]])
            q.submit()
            r = q.results()
            self.assertTrue("ints" in r)
            self.assertTrue("floats" not in r)
            del q


        with tiledb.DenseArray(self.path("foo"), mode='w', ctx=ctx) as T:
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
        A = np.array([['A', 'B'], ['C', '']], dtype='S')

        uri = self.path("varlen_2d_s1")
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim(name="rows", domain=(0, 1), tile=2, dtype=np.int64),
                            tiledb.Dim(name="cols", domain=(0, 1), tile=2, dtype=np.int64), ctx=ctx)

        schema = tiledb.ArraySchema(domain=dom, sparse=False,
                                    attrs=[tiledb.Attr(name="a", dtype='S', ctx=ctx)],
                                    ctx=ctx)

        tiledb.DenseArray.create(uri, schema)

        with tiledb.DenseArray(uri, mode='w', ctx=ctx) as T:
            T[...] = A

        with tiledb.DenseArray(uri) as T:
            assert_array_equal(A, T)

            res = T.multi_index[(0,1), (0,1)]['a']
            assert_array_equal(A, res)

    def test_array_2d_s3_mixed(self):
        # This array is currently read back with dtype object
        A = np.array([['AAA', 'B'], ['AB', 'C']], dtype='S3')

        uri = self.path("varlen_2d_s1")
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim(name="rows", domain=(0, 1), tile=2, dtype=np.int64),
                            tiledb.Dim(name="cols", domain=(0, 1), tile=2, dtype=np.int64), ctx=ctx)

        schema = tiledb.ArraySchema(domain=dom, sparse=False,
                                    attrs=[tiledb.Attr(name="a", dtype='S3', ctx=ctx)],
                                    ctx=ctx)

        tiledb.DenseArray.create(uri, schema)

        with tiledb.DenseArray(uri, mode='w', ctx=ctx) as T:
            T[...] = A

        with tiledb.DenseArray(uri) as T:
            assert_array_equal(A, T)

            res = T.multi_index[(0,1), (0,1)]['a']
            assert_array_equal(A, res)

    def test_incomplete_dense(self):
        path = self.path("incomplete_dense")
        # create 10 MB array
        data = np.arange(1310720, dtype=np.int64)
        # if `tile` is not set, it defaults to the full array and we
        # only read 8 bytes at a time.
        use_tile=131072
        #use_tile = None
        with tiledb.from_numpy(path, data, tile=use_tile) as A:
            pass

        # create context with 1 MB memory budget (2 MB total, 1 MB usable)
        config = tiledb.Config({'sm.memory_budget': 2 * 1024**2,
                                'py.init_buffer_bytes': 1024**2 })
        ctx = tiledb.Ctx(config=config)
        self.assertEqual(
            config['py.init_buffer_bytes'],
            str(1024**2)
        )
        # TODO would be good to check repeat count here. Not currently exposed by retry loop.
        with tiledb.DenseArray(path, ctx=ctx) as A:
            res_mr = A.multi_index[ slice(0, len(data) - 1) ]
            assert_array_equal(res_mr[""], data)
            res_idx = A[:]
            assert_array_equal(res_idx, data)

    def test_incomplete_dense_varlen(self):
        ncells = 100

        path = self.path("incomplete_dense_varlen")
        str_data = [rand_utf8(random.randint(0, n)) for n in range(ncells)]
        data = np.array(str_data, dtype=np.unicode_)

        # basic write
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim(domain=(1, len(data)), tile=len(data), ctx=ctx), ctx=ctx)
        att = tiledb.Attr(dtype=np.unicode_, var=True, ctx=ctx)

        schema = tiledb.ArraySchema(dom, (att,), ctx=ctx)

        tiledb.DenseArray.create(path, schema)
        with tiledb.DenseArray(path, mode='w', ctx=ctx) as T:
            T[:] = data

        with tiledb.DenseArray(path, mode='r', ctx=ctx) as T:
            assert_array_equal(data, T[:])

        # set the memory to the max length of a cell
        # these settings force ~100 retries
        # TODO would be good to check repeat count here; not yet exposed
        #      Also would be useful to have max cell config in libtiledb.
        init_buffer_bytes = 1024**2
        config = tiledb.Config({'sm.memory_budget': ncells,
                                'sm.memory_budget_var': ncells,
                                'py.init_buffer_bytes': init_buffer_bytes })

        ctx2 = tiledb.Ctx(config=config)
        self.assertEqual(
            config['py.init_buffer_bytes'],
            str(init_buffer_bytes)
        )

        with tiledb.DenseArray(path, mode='r', ctx=ctx2) as T2:
            assert_array_equal(data, T2[:])

    def test_incomplete_sparse_varlen(self):
        ncells = 100

        path = self.path("incomplete_dense_varlen")
        str_data = [rand_utf8(random.randint(0, n)) for n in range(ncells)]
        data = np.array(str_data, dtype=np.unicode_)
        coords = np.arange(ncells)

        # basic write
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim(domain=(0, len(data)+100), tile=len(data), ctx=ctx), ctx=ctx)
        att = tiledb.Attr(dtype=np.unicode_, var=True, ctx=ctx)

        schema = tiledb.ArraySchema(dom, (att,), sparse=True, ctx=ctx)

        tiledb.SparseArray.create(path, schema)
        with tiledb.SparseArray(path, mode='w', ctx=ctx) as T:
            T[coords] = data

        with tiledb.SparseArray(path, mode='r', ctx=ctx) as T:
            assert_array_equal(data, T[:][''])

        # set the memory to the max length of a cell
        # these settings force ~100 retries
        # TODO would be good to check repeat count here; not yet exposed
        #      Also would be useful to have max cell config in libtiledb.
        init_buffer_bytes = 1024**2
        config = tiledb.Config({'sm.memory_budget': ncells,
                                'sm.memory_budget_var': ncells,
                                'py.init_buffer_bytes': init_buffer_bytes })

        ctx2 = tiledb.Ctx(config=config)
        self.assertEqual(
            config['py.init_buffer_bytes'],
            str(init_buffer_bytes)
        )

        with tiledb.SparseArray(path, mode='r', ctx=ctx2) as T2:
            assert_array_equal(
                data,
                T2[:]['']
            )

            assert_array_equal(
                data,
                T2.multi_index[0:ncells]['']
            )

            # ensure that empty results are handled correctly
            assert_array_equal(
                T2.multi_index[101:105][''],
                np.array([], dtype=np.dtype('<U'))
            )

    def test_written_fragment_info(self):
        uri = self.path("test_written_fragment_info")

        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=10, dtype=np.int64, ctx=ctx), ctx=ctx)
        att = tiledb.Attr(ctx=ctx, dtype=np.int64)
        schema = tiledb.ArraySchema(ctx=ctx, domain=dom, attrs=(att,))
        tiledb.DenseArray.create(uri, schema)

        with tiledb.DenseArray(uri, mode='w', ctx=ctx) as T:
            T[:] = np.arange(0, 10, dtype=np.int64)

            self.assertTrue(T.last_write_info is not None)
            self.assertTrue(len(T.last_write_info.keys()) == 1)
            print(T.last_write_info.values())
            t_w1, t_w2 = list(T.last_write_info.values())[0]
            self.assertTrue(t_w1 > 0)
            self.assertTrue(t_w2 > 0)

    def test_missing_schema_error(self):
        uri = self.path("test_missing_schema_error")

        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=10, dtype=np.int64, ctx=ctx), ctx=ctx)
        att = tiledb.Attr(ctx=ctx, dtype=np.int64)
        schema = tiledb.ArraySchema(ctx=ctx, domain=dom, attrs=(att,))
        tiledb.DenseArray.create(uri, schema)

        with tiledb.DenseArray(uri, mode='w', ctx=ctx) as T:
            T[:] = np.arange(0, 10, dtype=np.int64)

        tiledb.VFS().remove_file(os.path.join(uri, "__array_schema.tdb"))

        with self.assertRaises(tiledb.TileDBError):
            tiledb.DenseArray(uri)


class DenseVarlen(DiskTestCase):
    def test_varlen_write_bytes(self):
        A = np.array(['aa','bbb','ccccc','ddddddddddddddddddddd','ee','ffffff','g','hhhhhhhhhh'], dtype=bytes)

        # basic write
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim(ctx=ctx, domain=(1, len(A)), tile=len(A)), ctx=ctx)
        att = tiledb.Attr(dtype=np.bytes_, ctx=ctx)

        schema = tiledb.ArraySchema(dom, (att,), ctx=ctx)

        tiledb.DenseArray.create(self.path("foo"), schema)
        with tiledb.DenseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            T[:] = A

        with tiledb.DenseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            assert_array_equal(A[:], T[:])

            assert_array_equal(A, T.multi_index[1:len(A)][''])


    def test_varlen_write_unicode(self):
        A = np.array(['aa','bbb',
                      'ccccc','ddddddddddddddddddddd',
                      'ee','ffffff','g','','hhhhhhhhhh'], dtype=np.unicode_)

        # basic write
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim(domain=(1, len(A)), tile=len(A), ctx=ctx), ctx=ctx)
        att = tiledb.Attr(dtype=np.unicode_, var=True, ctx=ctx)

        schema = tiledb.ArraySchema(dom, (att,), ctx=ctx)

        tiledb.DenseArray.create(self.path("foo"), schema)
        with tiledb.DenseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            T[:] = A

        with tiledb.DenseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            assert_array_equal(A[:], T[:])

    def test_varlen_write_floats(self):
        # Generates 8 variable-length float64 subarrays (subarray len and content are randomized)
        A = np.array([np.random.rand(x) for x in np.random.randint(1,12,8)], dtype=np.object)

        # basic write
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim(domain=(1, len(A)), tile=len(A)), ctx=ctx)
        att = tiledb.Attr(dtype=np.float64, var=True, ctx=ctx)

        schema = tiledb.ArraySchema(dom, (att,), ctx=ctx)
        tiledb.DenseArray.create(self.path("foo"), schema)

        with tiledb.DenseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            T[:] = A

        with tiledb.DenseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            T_ = T[:]
            # TODO/note: the return is a 0-element array.
            assert_array_equal(A[0], T[1][()])
            assert_array_equal(A[-1], T[-1][()])
            self.assertEqual(len(A), len(T_))
            # can't use assert_array_equal w/ np.object array
            self.assertTrue(all(np.array_equal(x,A[i]) for i,x in enumerate(T_)))

    def test_varlen_write_floats_2d(self):
        A = np.array([np.random.rand(x) for x in np.arange(1,10)], dtype=np.object).reshape(3,3)

        # basic write
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim(domain=(1, 3), tile=len(A)),
                            tiledb.Dim(domain=(1, 3), tile=len(A)),
                            ctx=ctx)
        att = tiledb.Attr(dtype=np.float64, var=True, ctx=ctx)

        schema = tiledb.ArraySchema(dom, (att,), ctx=ctx)
        tiledb.DenseArray.create(self.path("foo"), schema)

        with tiledb.DenseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            T[:] = A

        with tiledb.DenseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            T_ = T[:]
            self.assertEqual(len(A), len(T_))
            # can't use assert_array_equal w/ np.object array
            self.assertTrue(np.all([np.array_equal(A.flat[i], T[:].flat[i]) for i in np.arange(0, 9)]))

    def test_varlen_write_int_subarray(self):
        A = np.array(list(map(lambda x: np.array(x, dtype=np.uint64),
                        [np.arange(i, 2 * i + 1) for i in np.arange(0, 16)])
                        ),
                     dtype='O').reshape(4,4)

        uri = self.path("test_varlen_write_int_subarray")

        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 3), tile=len(A)),
                            tiledb.Dim(domain=(0, 3), tile=len(A)),
                            ctx=ctx)
        att = tiledb.Attr(dtype=np.uint64, var=True, ctx=ctx)
        schema = tiledb.ArraySchema(dom, (att,), ctx=ctx)

        tiledb.DenseArray.create(uri, schema)

        # NumPy forces single-element object arrays into a contiguous layout
        #       so we alternate the size to get a consistent baseline array.
        A_onestwos = np.array(
            list(map(lambda x: np.array(x, dtype=np.uint64),
                     list([(1,) if x % 2 == 0 else (1, 2) for x in range(16)]))),
            dtype=np.dtype('O')).reshape(4,4)

        with tiledb.open(uri, 'w') as T:
            T[:] = A_onestwos

        with tiledb.open(uri, 'w') as T:
            T[1:3,1:3] = A[1:3,1:3]

        A_assigned = A_onestwos.copy()
        A_assigned[1:3,1:3] = A[1:3,1:3]

        with tiledb.open(uri) as T:
            assert_subarrays_equal(A_assigned, T[:])

    def test_varlen_write_fixedbytes(self):
        # The actual dtype of this array is 'S21'
        A = np.array(['aa','bbb','ccccc','ddddddddddddddddddddd','ee',
                      'ffffff','g','hhhhhhhhhh'], dtype=np.dtype('S'))

        # basic write
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim(ctx=ctx, domain=(1, len(A)), tile=len(A)), ctx=ctx)
        att = tiledb.Attr(dtype=np.bytes_, ctx=ctx)

        schema = tiledb.ArraySchema(dom, (att,), ctx=ctx)

        tiledb.DenseArray.create(self.path("foo"), schema)
        with tiledb.DenseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            T[:] = A

        with tiledb.DenseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            assert_array_equal(A[:], T[:])


    def test_varlen_write_fixedunicode(self):
        A = np.array([u'aa',u'bbb',u'ccccc',u'ddddddddddddddddddddd',u'ee',
                      u'ffffff',u'g',u'hhhhhhhhhh'], dtype=np.dtype('U'))

        # basic write
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim(ctx=ctx, domain=(1, len(A)), tile=len(A)), ctx=ctx)
        att = tiledb.Attr(dtype=np.unicode_, ctx=ctx)

        schema = tiledb.ArraySchema(dom, (att,), ctx=ctx)

        tiledb.DenseArray.create(self.path("foo"), schema)
        with tiledb.DenseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            T[:] = A

        with tiledb.DenseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            assert_array_equal(A[:], T[:])


    def test_varlen_write_ints(self):
        A = np.array([np.uint64(np.random.randint(0,pow(10,6),x)) for x in np.random.randint(1,12,8)], dtype=np.object)

        print("random sub-length test array: {}".format(A))

        # basic write
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim(domain=(1, len(A)), tile=len(A), ctx=ctx), ctx=ctx)
        att = tiledb.Attr(dtype=np.int64, var=True, ctx=ctx)

        schema = tiledb.ArraySchema(dom, (att,), ctx=ctx)

        tiledb.DenseArray.create(self.path("foo"), schema)
        with tiledb.DenseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            T[:] = A

        with tiledb.DenseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            T_ = T[:]
            self.assertEqual(len(A), len(T))
            # can't use assert_array_equal w/ np.object array
            self.assertTrue(all(np.array_equal(x,A[i]) for i,x in enumerate(T_)))

    def test_varlen_wrong_domain(self):
        A = np.array(['aa','bbb','ccccc','ddddddddddddddddddddd','ee','ffffff','g','hhhhhhhhhh'])
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim(domain=(1, 3), tile=3), ctx=ctx)
        att = tiledb.Attr(dtype=np.bytes_, ctx=ctx)

        schema = tiledb.ArraySchema(dom, (att,), ctx=ctx)

        tiledb.DenseArray.create(self.path("foo"), schema)
        with tiledb.DenseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            with self.assertRaises(tiledb.TileDBError):
                T[:] = A

    def test_array_varlen_mismatched(self):
        # Test that we raise a TypeError when passing a heterogeneous object array.
        A = np.array(
                [  b'aa', b'bbb', b'cccc',
                   np.uint64([1,3,4]), ],
                dtype = np.object)

        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 3), tile=4, ctx=ctx), ctx=ctx)
        att = tiledb.Attr(dtype=np.bytes_, var=True, ctx=ctx)

        schema = tiledb.ArraySchema(dom, (att,), ctx=ctx)

        tiledb.DenseArray.create(self.path("foo"), schema)
        with tiledb.DenseArray(self.path("foo"), mode='w') as T:
            with self.assertRaises(TypeError):
                T[:] = A

    def test_array_varlen_2d_s_fixed(self):
        A = np.array([['AAAAAAAAAa', 'BBB'], ['ACCC', 'BBBCBCBCBCCCBBCBCBCCBC']], dtype='S')

        uri = self.path("varlen_2d_s_fixed")
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim(name="rows", domain=(0, 1), tile=2, dtype=np.int64),
                            tiledb.Dim(name="cols", domain=(0, 1), tile=2, dtype=np.int64), ctx=ctx)

        schema = tiledb.ArraySchema(domain=dom, sparse=False,
                                    attrs=[tiledb.Attr(name="a", dtype='S', var=True, ctx=ctx)],
                                    ctx=ctx)

        tiledb.DenseArray.create(uri, schema)

        with tiledb.DenseArray(uri, mode='w', ctx=ctx) as T:
            T[...] = A

        with tiledb.DenseArray(uri) as T:
            assert_array_equal(A, T)


class SparseArray(DiskTestCase):

    @unittest.expectedFailure
    def test_simple_1d_sparse_vector(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 3), tile=4, dtype=int, ctx=ctx), ctx=ctx)
        att = tiledb.Attr(dtype=int, ctx=ctx)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True, ctx=ctx)
        tiledb.SparseArray.create(self.path("foo"), schema)

        values = np.array([3, 4])
        with tiledb.SparseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            T[[1, 2]] = values

        with tiledb.SparseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            assert_array_equal(T[[1, 2]], values)

    @unittest.expectedFailure
    def test_simple_2d_sparse_vector(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(ctx,
                            tiledb.Dim(ctx, domain=(0, 3), tile=4, dtype=int),
                            tiledb.Dim(ctx, domain=(0, 3), tile=4, dtype=int))
        attr = tiledb.Attr(ctx, dtype=float)
        schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(attr,), sparse=True)
        tiledb.SparseArray.create(self.path("foo"), schema)

        values = np.array([3, 4], dtype=float)
        with tiledb.SparseArray(ctx, self.path("foo"), mode='w') as T:
            T[[1, 2], [1, 2]] = values

        with tiledb.SparseArray(ctx, self.path("foo"), mode='r') as T:
            assert_array_equal(T[[1, 2], [1, 2]], values)

    @unittest.expectedFailure
    def test_simple3d_sparse_vector(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(ctx,
                            tiledb.Dim(ctx, "x", domain=(0, 3), tile=4, dtype=int),
                            tiledb.Dim(ctx, "y", domain=(0, 3), tile=4, dtype=int),
                            tiledb.Dim(ctx, "z", domain=(0, 3), tile=4, dtype=int))
        attr = tiledb.Attr(ctx, dtype=float)
        schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(attr,), sparse=True)
        tiledb.SparseArray.create(self.path("foo"), schema)

        values = np.array([3, 4], dtype=float)
        with tiledb.SparseArray(ctx, self.path("foo"), mode='w') as T:
            T[[1, 2], [1, 2], [1, 2]] = values

        with tiledb.SparseArray(ctx, self.path("foo"), mode='r') as T:
            assert_array_equal(T[[1, 2], [1, 2], [1, 2]], values)

    @unittest.expectedFailure
    def test_sparse_ordered_fp_domain(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(
                tiledb.Dim("x", domain=(0.0, 10.0), tile=2.0, dtype=float, ctx=ctx),
                ctx=ctx)
        attr = tiledb.Attr(dtype=float, ctx=ctx)
        attr = tiledb.Attr(dtype=float, ctx=ctx)
        schema = tiledb.ArraySchema(domain=dom, attrs=(attr,), sparse=True, ctx=ctx)
        tiledb.SparseArray.create(self.path("foo"), schema)

        values = np.array([3.3, 2.7])
        with tiledb.SparseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            T[[2.5, 4.2]] = values
        with tiledb.SparseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            assert_array_equal(T[[2.5, 4.2]], values)

    @unittest.expectedFailure
    def test_sparse_unordered_fp_domain(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim("x", domain=(0.0, 10.0), tile=2.0, dtype=float), ctx=ctx)
        attr = tiledb.Attr(dtype=float, ctx=ctx)
        schema = tiledb.ArraySchema(domain=dom, attrs=(attr,), sparse=True, ctx=ctx)
        tiledb.SparseArray.create(self.path("foo"), schema)
        values = np.array([3.3, 2.7])
        with tiledb.SparseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            T[[4.2, 2.5]] = values

        with tiledb.SparseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            assert_array_equal(T[[2.5, 4.2]], values[::-1])

    @unittest.expectedFailure
    def test_multiple_attributes(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(
                tiledb.Dim(domain=(1, 10), tile=10, dtype=int, ctx=ctx),
                tiledb.Dim(domain=(1, 10), tile=10, dtype=int, ctx=ctx),
                ctx=ctx)
        attr_int = tiledb.Attr("ints", dtype=int, ctx=ctx)
        attr_float = tiledb.Attr("floats", dtype="float", ctx=ctx)
        schema = tiledb.ArraySchema(
                                domain=dom,
                                attrs=(attr_int, attr_float,),
                                sparse=True,
                                ctx=ctx)
        tiledb.SparseArray.create(self.path("foo"), schema)

        I = np.array([1, 1, 1, 2, 3, 3, 3, 4])
        J = np.array([1, 2, 4, 3, 1, 6, 7, 5])

        V_ints = np.array([0, 1, 2, 3, 4, 6, 7, 5])
        V_floats = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 5.0])

        V = {"ints": V_ints, "floats": V_floats}
        with tiledb.SparseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            T[I, J] = V
        with tiledb.SparseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            R = T[I, J]
        assert_array_equal(V["ints"], R["ints"])
        assert_array_equal(V["floats"], R["floats"])

        # check error attribute does not exist
        # TODO: should this be an attribute error?
        with tiledb.SparseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            V["foo"] = V["ints"].astype(np.int8)
            with self.assertRaises(tiledb.TileDBError):
                T[I, J] = V

            # check error ncells length
            V["ints"] = V["ints"][1:2].copy()
            with self.assertRaises(AttributeError):
                T[I, J] = V

    def test_query_fp_domain_index(self):
        uri = self.path("query_fp_domain_index")

        ctx = tiledb.Ctx()
        dom = tiledb.Domain(
            tiledb.Dim("x", domain=(-10.0, 10.0), tile=2.0, dtype=float, ctx=ctx),
            ctx=ctx)
        attr = tiledb.Attr("a", dtype=np.float32, ctx=ctx)
        schema = tiledb.ArraySchema(domain=dom, attrs=(attr,), sparse=True, ctx=ctx)
        tiledb.SparseArray.create(uri, schema)

        values = np.array([3.3, 2.7])
        with tiledb.SparseArray(uri, mode='w', ctx=ctx) as T:
            T[[2.5, 4.2]] = values
        with tiledb.SparseArray(uri, mode='r', ctx=ctx) as T:
            assert_array_equal(
                T.query(coords=True).domain_index[-10.0: np.nextafter(4.2, 0)]["a"],
                np.float32(3.3)
            )
            assert_array_equal(
                T.query(coords=True).domain_index[-10.0: np.nextafter(4.2, 0)]["x"],
                np.float32([2.5])
            )
            assert_array_equal(
                T.query(coords=False).domain_index[-10.0: 5.0]["a"],
                np.float32([3.3, 2.7])
            )
            self.assertTrue(
                'coords' not in T.query(coords=False).domain_index[-10.0: 5.0]
            )

    def test_subarray(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(
                tiledb.Dim("x", domain=(1, 10000), tile=100, dtype=int, ctx=ctx), ctx=ctx)
        att = tiledb.Attr("", dtype=float, ctx=ctx)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True, ctx=ctx)
        tiledb.SparseArray.create(self.path("foo"), schema)

        with tiledb.SparseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            self.assertIsNone(T.nonempty_domain())

        with tiledb.SparseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            T[[50, 60, 100]] = [1.0, 2.0, 3.0]

        with tiledb.SparseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            self.assertEqual(((50, 100),), T.nonempty_domain())

            # retrieve just valid coordinates in subarray T[40:60]
            assert_array_equal(T[40:61]["x"], [50, 60])

            #TODO: dropping coords with one anon value returns just an array
            res = T.query(coords=False)[40:61]
            assert_array_equal(res[""], [1.0, 2.0])
            self.assertEqual(("coords" in res), False)

    def test_sparse_bytes(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(
            tiledb.Dim("x", domain=(1, 10000), tile=100, dtype=int, ctx=ctx), ctx=ctx)
        att = tiledb.Attr("", var=True, dtype=np.bytes_, ctx=ctx)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True, ctx=ctx)
        tiledb.SparseArray.create(self.path("foo"), schema)

        with tiledb.SparseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            self.assertIsNone(T.nonempty_domain())
        A = np.array([b'aaa', b'bbbbbbbbbbbbbbbbbbbb', b'ccccccccccccccccccccccccc'],
                     dtype=np.bytes_)

        with tiledb.SparseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            T[[50, 60, 100]] = A

        with tiledb.SparseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            self.assertEqual(((50, 100),), T.nonempty_domain())

            # retrieve just valid coordinates in subarray T[40:60]
            assert_array_equal(T[40:61]["x"], [50, 60])

            #TODO: dropping coords with one anon value returns just an array
            res = T.query(coords=False)[40:61]
            assert_array_equal(res[""], A[0:2])
            self.assertEqual(("coords" in res), False)

    def test_sparse_unicode(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(
            tiledb.Dim("x", domain=(1, 10000), tile=100, dtype=int, ctx=ctx), ctx=ctx)
        att = tiledb.Attr("", var=True, dtype=np.unicode_, ctx=ctx)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True, ctx=ctx)
        tiledb.SparseArray.create(self.path("foo"), schema)

        with tiledb.SparseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            self.assertIsNone(T.nonempty_domain())

        A = np_array = np.array([u'1234545lkjalsdfj', u'mnopqrs', u'ijkl', u'gh', u'abcdef',
                                 u'aαbββcγγγdδδδδ', u'aαbββc', u'', u'γγγdδδδδ'], dtype=object)

        with tiledb.SparseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            T[[3, 4, 5, 6, 7, 50, 60, 70, 100]] = A

        with tiledb.SparseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            self.assertEqual(((3, 100),), T.nonempty_domain())

            # retrieve just valid coordinates in subarray T[40:60]
            assert_array_equal(T[40:61]["x"], [50, 60])

            #TODO: dropping coords with one anon value returns just an array
            res = T.query(coords=False)[40:61]
            assert_array_equal(res[""], A[5:7])
            self.assertEqual(("coords" in res), False)

    def test_sparse_fixes(self):
        uri = self.path("test_sparse_fixes")
        # indexing a 1 element item in a sparse array
        # (issue directly reported)
        # the test here is that the indexing does not raise
        ctx = tiledb.Ctx()
        dims = (tiledb.Dim('foo', ctx=ctx, domain=(0, 6), tile=2),
                tiledb.Dim('bar', ctx=ctx, domain=(0, 6), tile=1),
                tiledb.Dim('baz', ctx=ctx, domain=(0, 100), tile=1))
        dom = tiledb.Domain(*dims, ctx=ctx)
        att = tiledb.Attr(name="strattr", ctx=ctx, dtype='S1')
        schema = tiledb.ArraySchema(ctx=ctx, domain=dom, attrs=(att,),
                                    sparse=True)
        tiledb.SparseArray.create(uri, schema)
        with tiledb.SparseArray(uri) as T:
            T[:]

        # - test that assigning incompatible value to fixed-len str raises error
        # - test that value-conversion error raises exception w/ attr name context
        c = np.vstack(list((x,y,z) for x in range(7) for y in range(7) for z in range(101)))
        with tiledb.SparseArray(uri, 'w') as T:
            with self.assertRaises(ValueError):
                T[c[:,0],c[:,1],c[:,2]] = {'strattr': np.random.rand(7,7,101)}
            save_exc = list()
            try:
                T[c[:,0],c[:,1],c[:,2]] = {'strattr': np.random.rand(7,7,101)}
            except ValueError as e:
                save_exc.append(e)
            exc = save_exc.pop()
            if (sys.version_info > (3,3)):
                self.assertEqual(str(exc.__context__),
                                 "Cannot write a string value to non-string typed attribute 'strattr'!")

    def test_sparse_fixes_ch1560(self):
        from tiledb import Domain, Attr, Dim
        from collections import OrderedDict
        from numpy import array

        uri = self.path("sparse_fixes_ch1560")
        ctx = tiledb.Ctx({'sm.check_coord_dups': False})
        schema = tiledb.ArraySchema(
            domain=Domain(*[
                Dim(name='id', domain=(1, 5000), tile=25, dtype='int32', ctx=ctx),
            ]),
            attrs=[
                Attr(name='a1', dtype='datetime64[s]', ctx=ctx),
                Attr(name='a2', dtype='|S0', ctx=ctx),
                Attr(name='a3', dtype='|S0', ctx=ctx),
                Attr(name='a4', dtype='int32', ctx=ctx),
                Attr(name='a5', dtype='int8', ctx=ctx),
                Attr(name='a6', dtype='int32', ctx=ctx),
            ],
            cell_order='row-major',
            tile_order='row-major',
            sparse=True)

        tiledb.SparseArray.create(uri, schema)

        data = OrderedDict(
            [
                ('a1', array(['2017-04-01T04:00:00', '2019-10-01T00:00:00',
                              '2019-10-01T00:00:00', '2019-10-01T00:00:00'],
                             dtype='datetime64[s]')),
                ('a2', [b'Bus', b'The RIDE', b'The RIDE', b'The RIDE']),
                ('a3', [b'Bus', b'The RIDE', b'The RIDE', b'The RIDE']),
                ('a4', array([6911721,  138048,  138048,  138048], dtype='int32')),
                ('a5', array([20, 23, 23, 23], dtype='int8')),
                ('a6', array([345586,   6002,   6002,   6002], dtype='int32'))
            ])

        with tiledb.open(uri, 'w', ctx=ctx) as A:
            A[[1,462, 462, 462]] = data

        with tiledb.open(uri, ctx=ctx) as A:
            res = A[:]
            res.pop('id')
            for k,v in res.items():
                if isinstance(data[k], (np.ndarray,list)):
                    assert_array_equal(res[k], data[k])
                else:
                    self.assertEqual(res[k], data[k])

    def test_sparse_2d_varlen_int(self):
        path = self.path('test_sparse_2d_varlen_int')
        dtype = np.int32
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(
            tiledb.Dim(domain=(1, 4), tile=2, ctx=ctx),
            tiledb.Dim(domain=(1, 4), tile=2, ctx=ctx),
            ctx=ctx)
        att = tiledb.Attr(dtype=dtype, var=True, ctx=ctx)
        schema = tiledb.ArraySchema(dom, (att,), sparse=True, ctx=ctx)

        tiledb.SparseArray.create(path, schema)

        c1 = np.array([1,2,3,4])
        c2 = np.array([2,1,3,4])

        data = np.array([
            np.array([1,1], dtype=np.int32),
            np.array([2], dtype=np.int32),
            np.array([3,3,3], dtype=np.int32),
            np.array([4], dtype=np.int32)
            ], dtype='O')

        with tiledb.SparseArray(path, 'w') as A:
            A[c1, c2] = data

        with tiledb.SparseArray(path) as A:
            res = A[:]
            assert_subarrays_equal(
                res[''],
                data
            )
            assert_array_equal(
                res['__dim_0'],
                c1
            )
            assert_array_equal(
                res['__dim_1'],
                c2
            )

    def test_sparse_mixed_domain_uint_float64(self):
        path = self.path("mixed_domain_uint_float64")
        ctx = tiledb.Ctx()
        dims = [
            tiledb.Dim(name="index", domain=(0, 51), tile=11, dtype=np.uint64),
            tiledb.Dim(name="dpos", ctx=ctx, domain=(-100.0, 100.0), tile=10, dtype=np.float64)
        ]
        dom = tiledb.Domain(*dims)
        attrs = [
            tiledb.Attr(name="val", dtype=np.float64, ctx=ctx)
        ]

        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=True, ctx=ctx)
        tiledb.SparseArray.create(path, schema, ctx=ctx)

        data = np.random.rand(50, 63)
        coords1 = np.repeat(np.arange(0,50), 63)
        coords2 = np.linspace(-100.0,100.0, num=3150)

        with tiledb.open(path, 'w') as A:
            A[coords1, coords2] = data

        # tiledb returns coordinates in sorted order, so we need to check the output
        # sorted by the first dim coordinates
        sidx = np.argsort(coords1, kind='stable')
        coords2_idx = np.tile(np.arange(0,63), 50)[sidx]

        with tiledb.open(path) as A:
            res = A[:]
            assert_subarrays_equal(data[coords1[sidx],coords2_idx[sidx]], res['val'])
            a_nonempty = A.nonempty_domain()
            self.assertEqual(a_nonempty[0], (0,49))
            self.assertEqual(a_nonempty[1], (-100.0, 100.0))

    def test_sparse_string_domain(self):
        path = self.path("sparse_string_domain")
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim(name="d", domain=(None,None), dtype=np.bytes_, ctx=ctx), ctx=ctx)
        att = tiledb.Attr(name="a", ctx=ctx, dtype=np.int64)
        schema = tiledb.ArraySchema(ctx=ctx, domain=dom, attrs=(att,), sparse=True, capacity=10000)
        tiledb.SparseArray.create(path, schema)

        data = [1,2,3,4]
        coords = [b"aa",b"bbb", b"c", b"dddd"]

        with tiledb.open(path, 'w') as A:
            A[coords] = data

        with tiledb.open(path) as A:
            ned = A.nonempty_domain()[0]
            res = A[ned[0] : ned[1]]
            assert_array_equal(res['a'], data)
            self.assertEqual(set(res['d']), set(coords))
            self.assertEqual(A.nonempty_domain(), ((b"aa", b"dddd"),))


    def test_sparse_string_domain2(self):
        path = self.path("sparse_string_domain2")
        ctx = tiledb.Ctx()
        dims = [
            tiledb.Dim(name="str", domain=(None,None), tile=None, dtype=np.bytes_),
        ]
        dom = tiledb.Domain(*dims)
        attrs = [
            tiledb.Attr(name="val", dtype=np.float64, ctx=ctx)
        ]

        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=True, ctx=ctx)
        tiledb.SparseArray.create(path, schema, ctx=ctx)

        data = np.random.rand(10)
        coords = [rand_ascii_bytes(random.randint(5, 50)) for _ in range(10)]

        with tiledb.open(path, 'w') as A:
            A[coords] = data

        with tiledb.open(path) as A:
            ned = A.nonempty_domain()[0]
            res = A[ned[0] : ned[1]]
            self.assertTrue(set(res['str']) == set(coords))
            # must check data ordered by coords
            assert_array_equal(res['val'], data[np.argsort(coords, kind='stable')])

class DenseIndexing(DiskTestCase):

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
        #slice(-2000, 2000),
        #slice(0, 0),  # empty result
        #slice(-1, 0),  # empty result

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
        #slice(1050, -1, -1),
        #slice(1050, -1, -10),
        #slice(1050, -1, -100),
        #slice(1050, -1, -1000),
        #slice(1050, -1, -10000),
        #slice(1050, 0, -1),
        #slice(1050, 0, -10),
        #slice(1050, 0, -100),
        #slice(1050, 0, -1000),
        #slice(1050, 0, -10000),
        #slice(150, 50, -1),
        #slice(150, 50, -10),
        #slice(31, 1, -3),
        #slice(121, 81, -3),
        #slice(-1, 0, -1),
    ]

    bad_index_1d = [
        2.3,
        'foo',
        b'xxx',
        None,
        (0, 0),
        (slice(None), slice(None)),
    ]

    def test_index_1d(self):
        A = np.arange(1050, dtype=int)

        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 1049), tile=100, ctx=ctx), ctx=ctx)
        att = tiledb.Attr(ctx=ctx, dtype=int)
        schema = tiledb.ArraySchema(ctx=ctx, domain=dom, attrs=(att,))
        tiledb.DenseArray.create(self.path("foo"), schema)

        with tiledb.DenseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            T[:] = A

        with tiledb.DenseArray(self.path("foo"), mode='r', ctx=ctx) as T:
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

        #TODO: negative steps
        #slice(None, None, -1),
        #(slice(None, None, -1), slice(None)),
    ]

    bad_index_2d = [
        2.3,
        'foo',
        b'xxx',
        None,
        (2.3, slice(None)),
        (0, 0, 0),
        (slice(None), slice(None), slice(None)),
    ]

    def test_index_2d(self):
        A = np.arange(10000).reshape((1000, 10))

        ctx = tiledb.Ctx()
        dom = tiledb.Domain(
                       tiledb.Dim(ctx=ctx, domain=(0, 999), tile=100),
                       tiledb.Dim(ctx=ctx, domain=(0, 9), tile=2),
                       ctx=ctx)
        att = tiledb.Attr(dtype=A.dtype, ctx=ctx)
        schema = tiledb.ArraySchema(dom, (att,), ctx=ctx)
        tiledb.DenseArray.create(self.path("foo"), schema)


        with tiledb.DenseArray(self.path("foo"), mode='w', ctx=ctx) as T:
            T[:] = A

        with tiledb.DenseArray(self.path("foo"), mode='r', ctx=ctx) as T:
            for idx in self.good_index_1d:
                self._test_index(A, T, idx)

            for idx in self.bad_index_2d:
                with self.assertRaises(IndexError):
                    T[idx]


class DatetimeSlicing(DiskTestCase):
    def test_dense_datetime_vector(self):
        ctx = tiledb.Ctx()
        uri = self.path("foo_datetime_vector")

        # Domain is 10 years, day resolution, one tile per 365 days
        dim = tiledb.Dim(name="d1", ctx=ctx,
                         domain=(np.datetime64('2010-01-01'), np.datetime64('2020-01-01')),
                         tile=np.timedelta64(365, 'D'), dtype=np.datetime64('', 'D').dtype)
        dom = tiledb.Domain(dim, ctx=ctx)
        schema = tiledb.ArraySchema(ctx=ctx, domain=dom,
                                    attrs=(tiledb.Attr('a1', dtype=np.float64),))
        tiledb.Array.create(uri, schema)

        # Write a few years of data at the beginning using a timedelta object
        ndays = 365 * 2
        a1_vals = np.random.rand(ndays)
        start = np.datetime64('2010-01-01')
        # Datetime indexing is inclusive, so a delta of one less
        end = start + np.timedelta64(ndays - 1, 'D')
        with tiledb.DenseArray(uri, 'w', ctx=ctx) as T:
            T[start: end] = {'a1': a1_vals}

        # Read back data
        with tiledb.DenseArray(uri, 'r', attr='a1', ctx=ctx) as T:
            assert_array_equal(T[start: end], a1_vals)

        # Check nonempty domain
        with tiledb.DenseArray(uri, 'r', ctx=ctx) as T:
            nonempty = T.nonempty_domain()
            d1_nonempty = nonempty[0]
            self.assertEqual(d1_nonempty[0].dtype, np.datetime64('', 'D'))
            self.assertEqual(d1_nonempty[1].dtype, np.datetime64('', 'D'))
            self.assertTupleEqual(d1_nonempty, (start, end))

        # Slice a few days from the middle using two datetimes
        with tiledb.DenseArray(uri, 'r', attr='a1', ctx=ctx) as T:
            # Slice using datetimes
            actual = T[np.datetime64('2010-11-01'): np.datetime64('2011-01-31')]

            # Convert datetime interval to integer offset/length into original array
            # must be cast to int because float slices are not allowed in NumPy 1.12+
            read_offset = int( (np.datetime64('2010-11-01') - start) / np.timedelta64(1, 'D') )
            read_ndays = int( (np.datetime64('2011-01-31') - np.datetime64('2010-11-01') + 1) / np.timedelta64(1, 'D') )
            expected = a1_vals[read_offset : read_offset + read_ndays]
            assert_array_equal(actual, expected)

        # Slice the first year
        with tiledb.DenseArray(uri, 'r', attr='a1', ctx=ctx) as T:
            actual = T[np.datetime64('2010'): np.datetime64('2011')]

            # Convert datetime interval to integer offset/length into original array
            read_offset = int( (np.datetime64('2010-01-01') - start) / np.timedelta64(1, 'D') )
            read_ndays = int( (np.datetime64('2011-01-01') - np.datetime64('2010-01-01') + 1) / np.timedelta64(1, 'D') )
            expected = a1_vals[read_offset: read_offset + read_ndays]
            assert_array_equal(actual, expected)

    def test_sparse_datetime_vector(self):
        ctx = tiledb.Ctx()
        uri = self.path("foo_datetime_sparse_vector")

        # ns resolution, one tile per second, max domain possible
        dim = tiledb.Dim(name="d1", ctx=ctx,
                         domain=(np.datetime64(0, 'ns'), np.datetime64(int(np.iinfo(np.int64).max) - 1000000000, 'ns')),
                         tile=np.timedelta64(1, 's'), dtype=np.datetime64('', 'ns').dtype)
        self.assertEqual(dim.tile, np.timedelta64('1000000000', 'ns'))
        dom = tiledb.Domain(dim, ctx=ctx)
        schema = tiledb.ArraySchema(ctx=ctx, domain=dom, sparse=True,
                                    attrs=(tiledb.Attr('a1', dtype=np.float64),))
        tiledb.Array.create(uri, schema)

        # Write 10k cells every 1000 ns starting at time 0
        coords = np.datetime64(0, 'ns') + np.arange(0, 10000 * 1000, 1000)
        a1_vals = np.random.rand(len(coords))
        with tiledb.SparseArray(uri, 'w', ctx=ctx) as T:
            T[coords] = {'a1': a1_vals}

        # Read all
        with tiledb.SparseArray(uri, 'r', ctx=ctx) as T:
            assert_array_equal(T[:]['a1'], a1_vals)

        # Read back first 10 cells
        with tiledb.SparseArray(uri, 'r', ctx=ctx) as T:
            start = np.datetime64(0, 'ns')
            vals = T[start: start + np.timedelta64(10000, 'ns')]['a1']
            assert_array_equal(vals, a1_vals[0: 11])

    def test_datetime_types(self):
        ctx = tiledb.Ctx()

        units = ['h', 'm', 's', 'ms', 'us', 'ns', 'ps', 'fs']

        for res in units:
            uri = self.path("test_datetime_type_" + res)

            tmax = 1000
            tile = np.timedelta64(1, res)

            dim = tiledb.Dim(name="d1", ctx=ctx,
                             domain=(None,None),
                             tile=tile, dtype=np.datetime64('', res).dtype)
            dom = tiledb.Domain(dim, ctx=ctx)
            schema = tiledb.ArraySchema(ctx=ctx, domain=dom, sparse=True,
                                        attrs=(tiledb.Attr('a1', dtype=np.float64),))

            tiledb.Array.create(uri, schema)

            # Write tmax cells every 10 units starting at time 0
            coords = np.datetime64(0, res) + np.arange(0, tmax, 10)  # np.arange(0, 10000 * 1000, 1000)
            a1_vals = np.random.rand(len(coords))
            with tiledb.SparseArray(uri, 'w', ctx=ctx) as T:
                T[coords] = {'a1': a1_vals}

            # Read all
            with tiledb.SparseArray(uri, 'r', ctx=ctx) as T:
                assert_array_equal(T[:]['a1'], a1_vals)

            # Read back first 10 cells
            with tiledb.SparseArray(uri, 'r', ctx=ctx) as T:
                start = np.datetime64(0, res)
                vals = T[start: start + np.timedelta64(int(tmax/10), res)]['a1']
                assert_array_equal(vals, a1_vals[0: 11])


class PickleTest(DiskTestCase):
    # test that DenseArray and View can be pickled for multiprocess use
    # note that the current pickling is by URI and attributes (it is
    #     not, and likely should not be, a way to serialize array data)
    def test_pickle_roundtrip(self):
        import io, pickle

        ctx = tiledb.Ctx()
        uri = self.path("foo")
        with tiledb.DenseArray.from_numpy(uri, np.random.rand(5), ctx=ctx) as T:
            with io.BytesIO() as buf:
                pickle.dump(T, buf)
                buf.seek(0)
                with pickle.load(buf) as T2:
                    assert_array_equal(T, T2)

            with io.BytesIO() as buf, tiledb.DenseArray(uri) as V:
                pickle.dump(V, buf)
                buf.seek(0)
                with pickle.load(buf) as V2:
                    # make sure anonymous view pickles and round-trips
                    assert_array_equal(V, V2)

    def test_pickle_with_config(self):
        import io, pickle
        opts = dict()
        opts['vfs.s3.region'] = 'kuyper-belt-1'
        opts['vfs.max_parallel_ops'] = 1

        config = tiledb.Config(params=opts)
        ctx = tiledb.Ctx(config)

        uri = self.path("pickle_config")
        T = tiledb.DenseArray.from_numpy(uri, np.random.rand(3,3), ctx=ctx)

        with io.BytesIO() as buf:
            pickle.dump(T, buf)
            buf.seek(0)
            T2 = pickle.load(buf)
            assert_array_equal(T, T2)
            self.maxDiff = None
            d1 = ctx.config().dict()
            d2 = T2._ctx_().config().dict()
            self.assertEqual(d1['vfs.s3.region'], d2['vfs.s3.region'])
            self.assertEqual(d1['vfs.max_parallel_ops'], d2['vfs.max_parallel_ops'])
        T.close()
        T2.close()


class ArrayViewTest(DiskTestCase):
    def test_view_multiattr(self):
        import io, pickle
        ctx = tiledb.Ctx()
        uri = self.path("foo_multiattr")
        dom = tiledb.Domain(tiledb.Dim(ctx=ctx, domain=(0, 2), tile=3),
                            tiledb.Dim(ctx=ctx, domain=(0, 2), tile=3),
                            ctx=ctx)
        schema = tiledb.ArraySchema(ctx=ctx,
                                    domain=dom,
                                    attrs=(tiledb.Attr(""), tiledb.Attr("named")))
        tiledb.libtiledb.Array.create(uri, schema)

        anon_ar = np.random.rand(3, 3)
        named_ar = np.random.rand(3, 3)

        with tiledb.DenseArray(uri, 'w', ctx=ctx) as T:
            T[:] = {'': anon_ar, 'named': named_ar}

        with self.assertRaises(KeyError):
            T = tiledb.DenseArray(uri, 'r', attr="foo111", ctx=ctx)

        with tiledb.DenseArray(uri, 'r', attr="named", ctx=ctx) as T:
            assert_array_equal(T, named_ar)
            # make sure each attr view can pickle and round-trip
            with io.BytesIO() as buf:
                pickle.dump(T, buf)
                buf.seek(0)
                with pickle.load(buf) as T_rt:
                    assert_array_equal(T, T_rt)

        with tiledb.DenseArray(uri, 'r', attr="", ctx=ctx) as T:
            assert_array_equal(T, anon_ar)

            with io.BytesIO() as buf:
                pickle.dump(T, buf)
                buf.seek(0)
                with pickle.load(buf) as tmp:
                    assert_array_equal(tmp, anon_ar)

        # set subarray on multi-attribute
        range_ar = np.arange(0,9).reshape(3,3)
        with tiledb.DenseArray(uri, 'w', attr='named', ctx=ctx) as V_named:
            V_named[1:3,1:3] = range_ar[1:3,1:3]

        with tiledb.DenseArray(uri, 'r', attr='named', ctx=ctx) as V_named:
            assert_array_equal(V_named[1:3,1:3], range_ar[1:3,1:3])


class RWTest(DiskTestCase):
    def test_read_write(self):
        ctx = tiledb.Ctx()

        dom = tiledb.Domain(tiledb.Dim(ctx=ctx, domain=(0, 2), tile=3), ctx=ctx)
        att = tiledb.Attr(ctx=ctx, dtype='int64')
        schema = tiledb.ArraySchema(ctx=ctx, domain=dom, attrs=(att,))
        tiledb.libtiledb.Array.create(self.path("foo"), schema)

        np_array = np.array([1, 2, 3], dtype='int64')

        with tiledb.DenseArray(self.path("foo"), mode="w", ctx=ctx) as arr:
            arr.write_direct(np_array)

        with tiledb.DenseArray(self.path("foo"), mode="r", ctx=ctx) as arr:
            safe_dump(arr)
            self.assertEqual(arr.nonempty_domain(), ((0, 2),))
            self.assertEqual(arr.ndim, np_array.ndim)
            assert_array_equal(arr.read_direct(), np_array)


class NumpyToArray(DiskTestCase):

    def test_to_array0d(self):
        # Cannot create 0-dim arrays in TileDB
        ctx = tiledb.Ctx()
        np_array = np.array(1)
        with self.assertRaises(tiledb.TileDBError):
            with tiledb.DenseArray.from_numpy(self.path("foo"), np_array, ctx=ctx) as A:
                pass

    def test_to_array1d(self):
        ctx = tiledb.Ctx()
        np_array = np.array([1.0, 2.0, 3.0])
        with tiledb.DenseArray.from_numpy(self.path("foo"), np_array, ctx=ctx) as arr:
            assert_array_equal(arr[:], np_array)

    def test_to_array2d(self):
        ctx = tiledb.Ctx()
        np_array = np.ones((100, 100), dtype='i8')
        with tiledb.DenseArray.from_numpy(self.path("foo"), np_array, ctx=ctx) as arr:
            assert_array_equal(arr[:], np_array)

    def test_to_array3d(self):
        ctx = tiledb.Ctx()
        np_array = np.ones((1, 1, 1), dtype='i1')
        with tiledb.DenseArray.from_numpy(self.path("foo"), np_array, ctx=ctx) as arr:
            assert_array_equal(arr[:], np_array)

    def test_bytes_to_array1d(self):
        np_array = np.array([b'abcdef', b'gh', b'ijkl', b'mnopqrs', b'', b'1234545lkjalsdfj'], dtype=object)
        with tiledb.DenseArray.from_numpy(self.path("foo"), np_array) as arr:
            assert_array_equal(arr[:], np_array)

        with tiledb.DenseArray(self.path("foo")) as arr_reload:
            assert_array_equal(arr_reload[:], np_array)

    def test_unicode_to_array1d(self):
        np_array = np.array(['1234545lkjalsdfj', 'mnopqrs', 'ijkl', 'gh', 'abcdef',
                             'aαbββcγγγdδδδδ', '"aαbββc', '', 'γγγdδδδδ'], dtype=object)
        with tiledb.DenseArray.from_numpy(self.path("foo"), np_array) as arr:
            assert_array_equal(arr[:], np_array)

        with tiledb.DenseArray(self.path("foo")) as arr_reload:
            assert_array_equal(arr_reload[:], np_array)

    def test_array_interface(self):
        # Tests that __array__ interface works
        ctx = tiledb.Ctx()
        np_array1 = np.arange(1, 10)
        with tiledb.DenseArray.from_numpy(self.path("arr1"), np_array1, ctx=ctx) as arr1:
            assert_array_equal(np.array(arr1), np_array1)

        # Test that __array__ interface throws an error when number of attributes > 1
        dom = tiledb.Domain(tiledb.Dim(ctx=ctx, domain=(0, 2), tile=3), ctx=ctx)
        foo = tiledb.Attr("foo", dtype='i8', ctx=ctx)
        bar = tiledb.Attr("bar", dtype='i8', ctx=ctx)
        schema = tiledb.ArraySchema(ctx=ctx, domain=dom, attrs=(foo, bar))
        tiledb.DenseArray.create(self.path("arr2"), schema)
        with self.assertRaises(ValueError):
            with tiledb.DenseArray(self.path("arr2"), mode='r', ctx=ctx) as arr2:
                np.array(arr2)

    def test_array_getindex(self):
        # Tests that __getindex__ interface works
        ctx = tiledb.Ctx()
        np_array = np.arange(1, 10)
        with tiledb.DenseArray.from_numpy(self.path("foo"), np_array, ctx=ctx) as arr:
            assert_array_equal(arr[5:10], np_array[5:10])

    def test_to_array1d_attr_name(self):
        ctx = tiledb.Ctx()
        np_array = np.array([1.0, 2.0, 3.0])
        with tiledb.DenseArray.from_numpy(self.path("foo"), np_array, attr_name='a', ctx=ctx) as arr:
            assert_array_equal(arr[:]['a'], np_array)

class VFS(DiskTestCase):

    def test_supports(self):
        ctx = tiledb.Ctx()
        vfs = tiledb.VFS(ctx=ctx)

        self.assertTrue(vfs.supports("file"))
        self.assertIsInstance(vfs.supports("s3"), bool)
        self.assertIsInstance(vfs.supports("hdfs"), bool)
        self.assertIsInstance(vfs.supports("gcs"), bool)
        self.assertIsInstance(vfs.supports("azure"), bool)

        with self.assertRaises(ValueError):
            vfs.supports("invalid")

    def test_dir(self):
        ctx = tiledb.Ctx()
        vfs = tiledb.VFS(ctx=ctx)

        dir = self.path("foo")
        self.assertFalse(vfs.is_dir(dir))

        # create
        vfs.create_dir(dir)
        self.assertTrue(vfs.is_dir(dir))

        # remove
        vfs.remove_dir(dir)
        self.assertFalse(vfs.is_dir(dir))

        # create nested path
        dir = self.path("foo/bar")
        with self.assertRaises(tiledb.TileDBError):
            vfs.create_dir(dir)

        vfs.create_dir(self.path("foo"))
        vfs.create_dir(self.path("foo/bar"))
        self.assertTrue(vfs.is_dir(dir))

    def test_file(self):
        ctx = tiledb.Ctx()
        vfs = tiledb.VFS(ctx=ctx)

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
        with self.assertRaises(tiledb.TileDBError):
            vfs.touch(file)

    def test_move(self):
        ctx = tiledb.Ctx()
        vfs = tiledb.VFS(ctx=ctx)

        vfs.create_dir(self.path("foo"))
        vfs.create_dir(self.path("bar"))
        vfs.touch(self.path("bar/baz"))

        self.assertTrue(vfs.is_file(self.path("bar/baz")))

        vfs.move_file(self.path("bar/baz"), self.path("foo/baz"))

        self.assertFalse(vfs.is_file(self.path("bar/baz")))
        self.assertTrue(vfs.is_file(self.path("foo/baz")))

        # moving to invalid dir should raise an error
        with self.assertRaises(tiledb.TileDBError):
            vfs.move_dir(self.path("foo/baz"), self.path("do_not_exist/baz"))

    def test_write_read(self):
        ctx = tiledb.Ctx()
        vfs = tiledb.VFS(ctx=ctx)

        buffer = b"bar"
        fh = vfs.open(self.path("foo"), "wb")
        vfs.write(fh, buffer)
        vfs.close(fh)
        self.assertEqual(vfs.file_size(self.path("foo")), 3)

        fh = vfs.open(self.path("foo"), "rb")
        self.assertEqual(vfs.read(fh, 0, 3), buffer)
        vfs.close(fh)

        # write / read empty input
        fh = vfs.open(self.path("baz"), "wb")
        vfs.write(fh, b"")
        vfs.close(fh)
        self.assertEqual(vfs.file_size(self.path("baz")), 0)

        fh = vfs.open(self.path("baz"), "rb")
        self.assertEqual(vfs.read(fh, 0, 0), b"")
        vfs.close(fh)

        # read from file that does not exist
        with self.assertRaises(tiledb.TileDBError):
            vfs.open(self.path("do_not_exist"), "rb")

    def test_io(self):
        ctx = tiledb.Ctx()
        vfs = tiledb.VFS(ctx=ctx)

        buffer = b"0123456789"
        fio = tiledb.FileIO(vfs, self.path("foo"), mode="wb")
        fio.write(buffer)
        fio.flush()
        self.assertEqual(fio.tell(), len(buffer))

        fio = tiledb.FileIO(vfs, self.path("foo"), mode="rb")
        with self.assertRaises(IOError):
            fio.write(b"foo")

        self.assertEqual(vfs.file_size(self.path("foo")), len(buffer))

        fio = tiledb.FileIO(vfs, self.path("foo"), mode='rb')
        self.assertEqual(fio.read(3), b'012')
        self.assertEqual(fio.tell(), 3)
        self.assertEqual(fio.read(3), b'345')
        self.assertEqual(fio.tell(), 6)
        self.assertEqual(fio.read(10), b'6789')
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

        # Reading from the end should return empty
        fio.seek(0)
        fio.read()
        self.assertEqual(fio.read(), b"")

        # Test writing and reading lines with TextIOWrapper
        lines = [rand_utf8(random.randint(0, 50))+'\n' for _ in range(10)]
        rand_uri = self.path("test_fio.rand")
        with tiledb.FileIO(vfs, rand_uri, 'wb') as f:
            txtio = io.TextIOWrapper(f, encoding='utf-8')
            txtio.writelines(lines)
            txtio.flush()

        with tiledb.FileIO(vfs, rand_uri, 'rb') as f2:
            txtio = io.TextIOWrapper(f2, encoding='utf-8')
            self.assertEqual(txtio.readlines(), lines)

    def test_ls(self):
        import os
        basepath = self.path("test_vfs_ls")
        os.mkdir(basepath)
        for id in (1,2,3):
            dir = os.path.join(basepath, "dir"+str(id))
            os.mkdir(dir)
            fname =os.path.join(basepath, "file_"+str(id))
            os.close(os.open(fname, os.O_CREAT | os.O_EXCL))

        expected = ('dir1','dir2','dir3', 'file_1', 'file_2', 'file_3')
        vfs = tiledb.VFS(ctx=tiledb.Ctx())
        self.assertSetEqual(
            set(os.path.normpath(
                'file://' + os.path.join(basepath, x)) for x in expected),
            set(os.path.normpath(x) for x in vfs.ls(basepath))
        )

    def test_dir_size(self):
        import os
        vfs = tiledb.VFS(ctx=tiledb.Ctx())

        path = self.path("test_vfs_dir_size")
        os.mkdir(path)
        rand_sizes = np.random.choice(100, size=4, replace=False)
        for size in rand_sizes:
            file_path = os.path.join(path, "f_" + str(size))
            with tiledb.FileIO(vfs, file_path, 'wb') as f:
                data = os.urandom(size)
                f.write(data)

        self.assertEqual(vfs.dir_size(path), sum(rand_sizes))

class ConsolidationTest(DiskTestCase):

    def test_array_vacuum(self):
        ctx = tiledb.Ctx()
        vfs = tiledb.VFS(ctx=ctx)
        path = self.path("test_array_vacuum")

        dshape = (0, 19)
        num_writes = 10

        def create_array(target_path):
            dom = tiledb.Domain(tiledb.Dim(ctx=ctx, domain=dshape, tile=3), ctx=ctx)
            att = tiledb.Attr(ctx=ctx, dtype='int64')
            schema = tiledb.ArraySchema(ctx=ctx, domain=dom, attrs=(att,))
            tiledb.libtiledb.Array.create(target_path, schema)

        def write_fragments(target_path):
            for i in range(num_writes):
                with tiledb.open(target_path, 'w') as A:
                    A[i:dshape[1]] = np.random.rand(dshape[1] - i)

        create_array(path)
        write_fragments(path)
        paths = vfs.ls(path)
        self.assertEqual(len(paths), 3 + 2 * num_writes)

        tiledb.consolidate(path, ctx=ctx)
        tiledb.vacuum(path, ctx=ctx)

        paths = vfs.ls(path)
        self.assertEqual(len(paths), 5)

        del path

        path2 = self.path("test_array_vacuum_fragment_meta")
        create_array(path2)
        write_fragments(path2)
        tiledb.consolidate(path2,
                           config=tiledb.Config({'sm.consolidation.mode': 'fragment_meta'}))
        tiledb.vacuum(path2,
                      config=tiledb.Config({'sm.vacuum.mode': 'fragment_meta'}))
        paths = vfs.ls(path2)

        self.assertEqual(len(paths), 3 + 2 * num_writes + 1)

        path3 = self.path("test_array_vacuum2")
        create_array(path3)
        write_fragments(path3)
        conf = tiledb.Config({'sm.consolidation.mode': 'fragment_meta'})
        with tiledb.open(path3, 'w') as A:
            A.consolidate(config=conf)

        paths = vfs.ls(path2)

class RegTests(DiskTestCase):
    def test_tiledb_py_0_6_anon_attr(self):
        # Test that anonymous attributes internally stored as "__attr" are presented as ""
        # Normally, we can't actually write an attribute named "__attr" anymore, so we
        # restore a schema written by a patched libtiledb, and rename the attr file.

        schema_data = b'\x05\x00\x00\x00]\x00\x00\x00\x00\x00\x00\x00q\x00\x00\x00\x00\x00\x00\x00\x04\x01\x00\x00\x00\x00\x00\x00\x00\x00\x12\x00\x00\x00\x00\x00\x01\x00\x01\x00\x00\x00\x01\x05\x00\x00\x00\x01\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00q\x00\x00\x009\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00q\x00\x00\x009\x00\x00\x00x\x01ce\x80\x00\x01u(\x83\x81\x11\x08\x19\x18\x98XA\xc4\x7f `\xc0\x10\x01\xc9\x83p\n\x1b\x88\x84\xb0\x81\x8a\xc1l\x88\x00H\x9c\r\x88\xe3\xe3\x13KJ\x8aP\x94\x01\x00\xa2c\x0bD'

        path = self.path("tiledb_py_0_6_anon_attr")
        ctx = tiledb.default_ctx()
        dom = tiledb.Domain(tiledb.Dim(name="d", domain=(0, 0), tile=1, dtype=np.uint8))
        attrs = (tiledb.Attr(name="_attr_", dtype=np.uint8, ctx=ctx),)

        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False, ctx=ctx)
        tiledb.DenseArray.create(path, schema, ctx=ctx)

        with tiledb.open(path, 'w') as A:
            A[0] = 1

        fragment_name = os.path.split(list(A.last_write_info.keys())[0])[-1]
        fragment_path = os.path.join(path, fragment_name)

        # fix up the array the override schema
        with open(os.path.join(path, "__array_schema.tdb"), 'wb') as f:
            f.write(schema_data)
        import shutil
        shutil.move(
            os.path.join(fragment_path, "_attr_.tdb"),
            os.path.join(fragment_path, "__attr.tdb")
        )
        with tiledb.open(path) as A:
            self.assertEqual(A.schema.attr(0).name, "")
            self.assertEqual(A.schema.attr(0)._internal_name, "__attr")
            self.assertEqual(A[0], 1)
            mres = A.multi_index[0]
            self.assertEqual(mres[''], 1)

            qres = A.query(coords=True).multi_index[0]
            self.assertEqual(qres['d'], 0)

class MemoryTest(DiskTestCase):
    # sanity check that memory usage doesn't increase more than 2x when reading 40MB 100x
    # https://github.com/TileDB-Inc/TileDB-Py/issues/150

    def setUp(self):
        super(MemoryTest, self).setUp()
        import sys
        if not sys.platform.startswith("linux"):
            self.skipTest("Only run MemoryTest on linux")

    @staticmethod
    def use_many_buffers(path):
        import psutil, os
        # https://stackoverflow.com/questions/938733/total-memory-used-by-python-process
        process = psutil.Process(os.getpid())

        x = np.ones(10000000, dtype=np.float32)
        ctx = tiledb.Ctx()
        d1 = tiledb.Dim(
            'test_domain', domain=(0, x.shape[0] - 1), tile=10000, dtype="uint32")
        domain = tiledb.Domain(d1)
        v = tiledb.Attr(
            'test_value',
            dtype="float32")

        schema = tiledb.ArraySchema(
            domain=domain, attrs=(v,), cell_order="row-major", tile_order="row-major")

        A = tiledb.DenseArray.create(path, schema)

        with tiledb.DenseArray(path, mode="w", ctx=ctx) as A:
            A[:] = {'test_value': x}

        with tiledb.DenseArray(path, mode='r') as data:
            data[:]
            initial = process.memory_info().rss
            print("  initial RSS: {}".format(round(initial / (10 ** 6)), 2))
            for i in range(100):
                # read but don't store: this memory should be freed
                data[:]

                if i % 10 == 0:
                    print('    read iter {}, RSS (MB): {}'.format(
                        i, round(process.memory_info().rss / (10 ** 6), 2)))

        return initial

    def test_memory_cleanup(self):
        import tiledb, numpy as np
        import psutil, os

        # run function which reads 100x from a 40MB test array
        # TODO: RSS is too loose to do this end-to-end, so should use instrumentation.
        print("Starting TileDB-Py memory test:")
        initial = self.use_many_buffers(self.path('test_memory_cleanup'))

        process = psutil.Process(os.getpid())
        final = process.memory_info().rss
        print("  final RSS: {}".format(round(final / (10 ** 6)), 2))

        import gc
        gc.collect()

        final_gc = process.memory_info().rss
        print("  final RSS after forced GC: {}".format(round(final_gc / (10 ** 6)), 2))

        self.assertTrue(final < (2 * initial))

has_psutil = False
try:
    import psutil
    has_psutil = True
except ImportError:
    pass

class HighlevelTests(DiskTestCase):
    def test_open(self):
        uri = self.path("test_open")
        with tiledb.from_numpy(uri, np.random.rand(10)) as A:
            pass

        ctx = tiledb.Ctx()
        with tiledb.DenseArray(uri, ctx=ctx) as A:
            self.assertEqual(A._ctx_(), ctx)

        with tiledb.open(uri, ctx=ctx) as A:
            self.assertEqual(A._ctx_(), ctx)

        config = tiledb.Config()
        with tiledb.open(uri, config=config) as A:
            self.assertEqual(A._ctx_().config(), config)

        with self.assertRaises(KeyError):
            # This path must test `tiledb.open` specifically
            # https://github.com/TileDB-Inc/TileDB-Py/issues/277
            tiledb.open(uri, 'r', attr='the-missing-attr')

    @unittest.skipIf(not has_psutil or \
                     sys.version_info < (3,2), "")
    def test_ctx_thread_cleanup(self):
        import warnings
        # This test checks that contexts are destroyed correctly.
        # It creates new contexts repeatedly, in-process, and
        # checks that the total number of threads stays stable.
        config = {
            'sm.num_reader_threads': 128,
        }
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
                        warnings.warn("Thread cleanup test RuntimeError: {} \n    on iteration: {}".format(str(rterr), n))

            with tiledb.DenseArray(uri, ctx=tiledb.Ctx(config)) as A:
                res = A[:]

            if n == 0:
                start_threads = len(thisproc.threads())

# Wrapper to execute specific code in subprocess so that we can ensure the thread count
# init is correct. Necessary because multiprocess.get_context is only available in Python 3.4+,
# and the multiprocessing method may be set to fork by other tests (e.g. dask).
def init_test_wrapper(cfg=None):
    import subprocess, os
    python_exe = sys.executable
    cmd = 'from test_libtiledb import *; init_test_helper({})'.format(cfg)
    test_path = os.path.dirname(os.path.abspath(__file__))

    sp_output = subprocess.check_output([python_exe, '-c', cmd], cwd=test_path)
    return int(sp_output.decode('UTF-8').strip())

def init_test_helper(cfg=None):
    import tiledb
    tiledb.libtiledb.initialize_ctx(cfg)
    num_tbb_threads = tiledb.default_ctx().config()['sm.num_tbb_threads']
    print(int(num_tbb_threads))

class ContextTest(unittest.TestCase):
    def test_default_context(self):
        ctx = tiledb.default_ctx()
        self.assertIsInstance(ctx, tiledb.Ctx)
        self.assertIsInstance(ctx.config(), tiledb.Config)

    def test_init_config(self):
        self.assertEqual(-1, init_test_wrapper())

        self.assertEqual(
            1,
            init_test_wrapper({'sm.num_tbb_threads': 1})
        )


class ReprTest(unittest.TestCase):
    def test_attr_repr(self):
        attr = tiledb.Attr(name="itsanattr", dtype=np.float64)
        self.assertTrue(
            re.match(r"Attr\(name=[u]?'itsanattr', dtype='float64'\)",
                     repr(attr))
        )

        g = dict()
        exec("from tiledb import Attr; from numpy import float64", g)
        self.assertEqual(
            eval(repr(attr), g),
            attr
        )

    def test_arrayschema_repr(self):
        ctx = tiledb.default_ctx()
        filters = tiledb.FilterList([tiledb.ZstdFilter(-1)])
        for sparse in [False, True]:
            domain = tiledb.Domain(
                tiledb.Dim(domain=(1, 8), tile=2, ctx=ctx),
                tiledb.Dim(domain=(1, 8), tile=2, ctx=ctx),
                ctx=ctx)
            a1 = tiledb.Attr("val", dtype='f8', filters=filters, ctx=ctx)
            orig_schema = tiledb.ArraySchema(domain=domain, attrs=(a1,), sparse=sparse, ctx=ctx)

            schema_repr = repr(orig_schema)
            g = dict()
            setup = ("from tiledb import *\n"
                     "import numpy as np\n")

            exec(setup, g)
            new_schema = None
            try:
                new_schema = eval(schema_repr, g)
            except Exception as exc:
                warn_str = """Exception during ReprTest schema eval""" + \
                           """, schema string was:\n""" + \
                           """'''""" + \
                           """\n{}\n'''""".format(schema_repr)
                warnings.warn(warn_str)
                raise

            self.assertEqual(new_schema, orig_schema)

#if __name__ == '__main__':
#    # run a single example for in-process debugging
#    # better to use `pytest --gdb` if available
#    t = DenseArrayTest()
#    t.setUp()
#    t.test_array_1d()
