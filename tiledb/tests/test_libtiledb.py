from __future__ import absolute_import

import unittest, os

import numpy as np
from numpy.testing import assert_array_equal

import tiledb
from tiledb.tests.common import DiskTestCase


class VersionTest(unittest.TestCase):

    def test_version(self):
        v = tiledb.libtiledb.version()
        self.assertIsInstance(v, tuple)
        self.assertTrue(len(v) == 3)
        self.assertTrue(v[0] >= 1, "TileDB major version must be >= 1")


class StatsTest(unittest.TestCase):

    def test_stats(self):
        tiledb.libtiledb.stats_enable()
        tiledb.libtiledb.stats_reset()
        tiledb.libtiledb.stats_disable()


class Config(DiskTestCase):

    def test_config(self):
        config = tiledb.Config()
        config["sm.tile_cache_size"] = 100
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
        vfs = tiledb.VFS(ctx, config)
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

        tiledb.group_create(ctx, self.group1)
        tiledb.group_create(ctx, self.group2)
        tiledb.group_create(ctx, self.group3)
        tiledb.group_create(ctx, self.group4)

    def is_group(self, ctx, uri):
        return tiledb.object_type(ctx, uri) == "group"


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

        tiledb.walk(ctx, self.path(""), append_to_groups, order="preorder")

        groups.sort()

        self.assertTrue(groups[0][0].endswith(self.group1) and groups[0][1] == "group")
        self.assertTrue(groups[1][0].endswith(self.group2) and groups[1][1] == "group")
        self.assertTrue(groups[2][0].endswith(self.group3) and groups[2][1] == "group")
        self.assertTrue(groups[3][0].endswith(self.group4) and groups[3][1] == "group")

        groups = []

        tiledb.walk(ctx, self.path(""), append_to_groups, order="postorder")

        self.assertTrue(groups[0][0].endswith(self.group2) and groups[0][1] == "group")
        self.assertTrue(groups[1][0].endswith(self.group4) and groups[1][1] == "group")
        self.assertTrue(groups[2][0].endswith(self.group3) and groups[2][1] == "group")
        self.assertTrue(groups[3][0].endswith(self.group1) and groups[3][1] == "group")

    def test_remove_group(self):
        ctx = tiledb.Ctx()

        tiledb.remove(ctx, self.group3)

        self.assertFalse(self.is_group(ctx, self.group3))
        self.assertFalse(self.is_group(ctx, self.group4))

    def test_move_group(self):
        ctx = tiledb.Ctx()

        self.assertTrue(self.is_group(ctx, self.group2))
        tiledb.move(ctx, self.group2, self.group2 + "_moved")
        self.assertFalse(self.is_group(ctx, self.group2))
        self.assertTrue(self.is_group(ctx, self.group2 + "_moved"))


class DimensionTest(unittest.TestCase):

    def test_minimal_dimension(self):
        ctx = tiledb.Ctx()
        dim = tiledb.Dim(ctx, domain=(0, 4))
        self.assertEqual(dim.name, "", "dimension name is empty")
        self.assertEqual(dim.shape, (5,))
        self.assertEqual(dim.tile, None, "tiled extent is None (void)")

    def test_dimension(self):
        ctx = tiledb.Ctx()
        dim = tiledb.Dim(ctx, "d1", domain=(0, 3), tile=2)
        self.assertEqual(dim.name, "d1")
        self.assertEqual(dim.shape, (4,))
        self.assertEqual(dim.tile, 2)


class DomainTest(unittest.TestCase):

    def test_domain(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(
            ctx,
            tiledb.Dim(ctx, "d1", (1, 4), 2, dtype='u8'),
            tiledb.Dim(ctx, "d2", (1, 4), 2, dtype='u8'))
        dom.dump()
        self.assertEqual(dom.ndim, 2)
        self.assertEqual(dom.dtype, np.dtype("uint64"))
        self.assertEqual(dom.shape, (4, 4))

        # check that we can iterate over the dimensions
        dim_names = [dim.name for dim in dom]
        self.assertEqual(["d1", "d2"], dim_names)

    def test_domain_dims_not_same_type(self):
        ctx = tiledb.Ctx()
        with self.assertRaises(TypeError):
            tiledb.Domain(
                    ctx,
                    tiledb.Dim(ctx, "d1", (1, 4), 2, dtype=int),
                    tiledb.Dim(ctx, "d2", (1, 4), 2, dtype=float))


class AttributeTest(unittest.TestCase):

    def test_minimal_attribute(self):
        ctx = tiledb.Ctx()
        attr = tiledb.Attr(ctx)
        self.assertTrue(attr.isanon)
        self.assertEqual(attr.name, u"")
        self.assertEqual(attr.dtype, np.float_)
        self.assertEqual(attr.compressor, (None, -1))

    def test_attribute(self):
        ctx = tiledb.Ctx()
        attr = tiledb.Attr(ctx, "foo")
        attr.dump()
        self.assertEqual(attr.name, "foo")
        self.assertEqual(attr.dtype, np.float64,
                         "default attribute type is float64")
        compressor, level = attr.compressor
        self.assertEqual(compressor, None, "default to no compression")
        self.assertEqual(level, -1, "default compression level when none is specified")

    def test_full_attribute(self):
        ctx = tiledb.Ctx()
        attr = tiledb.Attr(ctx, "foo", dtype=np.int64, compressor=("zstd", 10))
        attr.dump()
        self.assertEqual(attr.name, "foo")
        self.assertEqual(attr.dtype, np.int64)
        compressor, level = attr.compressor
        self.assertEqual(compressor, "zstd")
        self.assertEqual(level, 10)

    def test_ncell_attribute(self):
        ctx = tiledb.Ctx()
        dtype = np.dtype([("", np.int32), ("", np.int32)])
        attr = tiledb.Attr(ctx, "foo", dtype=dtype)

        self.assertEqual(attr.dtype, dtype)
        self.assertEqual(attr.ncells, 2)

        # dtype subarrays not supported
        with self.assertRaises(TypeError):
            tiledb.Attr(ctx, "foo", dtype=np.dtype((np.int32, 2)))

        # mixed type record arrays not supported
        with self.assertRaises(TypeError):
            tiledb.Attr(ctx, "foo", dtype=np.dtype([("", np.float32), ("", np.int32)]))

    def test_ncell_bytes_attribute(self):
        ctx = tiledb.Ctx()
        dtype = np.dtype((np.bytes_, 10))
        attr = tiledb.Attr(ctx, "foo", dtype=dtype)

        self.assertEqual(attr.dtype, dtype)
        self.assertEqual(attr.ncells, 10)

    def test_vararg_attribute(self):
        ctx = tiledb.Ctx()
        attr = tiledb.Attr(ctx, "foo", dtype=np.bytes_)
        self.assertEqual(attr.dtype, np.dtype(np.bytes_))
        self.assertTrue(attr.isvar)

    def test_filter(self):
        ctx = tiledb.Ctx()
        gzip_filter = tiledb.libtiledb.GzipFilter(ctx, level=10)
        self.assertIsInstance(gzip_filter, tiledb.libtiledb.Filter)
        self.assertEqual(gzip_filter.level, 10)

        bw_filter = tiledb.libtiledb.BitWidthReductionFilter(ctx, window=10)
        self.assertIsInstance(bw_filter, tiledb.libtiledb.Filter)
        self.assertEqual(bw_filter.window, 10)

        filter_list = tiledb.libtiledb.FilterList(ctx, [gzip_filter, bw_filter], chunksize=1024)
        self.assertEqual(filter_list.chunksize, 1024)
        self.assertEqual(len(filter_list), 2)
        self.assertEqual(filter_list[0].level, gzip_filter.level)
        self.assertEqual(filter_list[1].window, bw_filter.window)

        # test filter list iteration
        self.assertEqual(len(list(filter_list)), 2)

        with self.assertRaises(TypeError):
            tiledb.Attr(ctx, "foo", dtype=np.int64, filters=[gzip_filter])

        attr = tiledb.Attr(ctx, "foo",
                           dtype=np.int64,
                           filters=filter_list)

        self.assertEqual(len(attr.filters), 2)
        self.assertEqual(attr.filters.chunksize, filter_list.chunksize)


class ArraySchemaTest(unittest.TestCase):

    def test_unique_attributes(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(
            ctx,
            tiledb.Dim(ctx, "d1", (1, 4), 2, dtype='u8'),
            tiledb.Dim(ctx, "d2", (1, 4), 2, dtype='u8'))

        attr1 = tiledb.Attr(ctx, "foo", dtype=float)
        attr2 = tiledb.Attr(ctx, "foo", dtype=int)

        with self.assertRaises(tiledb.TileDBError):
            tiledb.ArraySchema(ctx, domain=dom, attrs=(attr1, attr2))

    def test_dense_array_schema(self):
        ctx = tiledb.Ctx()
        domain = tiledb.Domain(
            ctx,
            tiledb.Dim(ctx, domain=(1, 8), tile=2),
            tiledb.Dim(ctx, domain=(1, 8), tile=2))
        a1 = tiledb.Attr(ctx, "val", dtype='f8')
        schema = tiledb.ArraySchema(ctx, domain=domain, attrs=(a1,))
        self.assertFalse(schema.sparse)
        self.assertEqual(schema.cell_order, "row-major")
        self.assertEqual(schema.tile_order, "row-major")
        self.assertEqual(schema.domain, domain)
        self.assertEqual(schema.ndim, 2)
        self.assertEqual(schema.shape, (8, 8))
        self.assertEqual(schema.nattr, 1)
        self.assertEqual(schema.attr(0), a1)
        self.assertEqual(schema,
            tiledb.ArraySchema(ctx, domain=domain, attrs=(a1,)))
        self.assertNotEqual(schema,
            tiledb.ArraySchema(ctx, domain=domain, attrs=(a1,), sparse=True))
        # test iteration over attributes
        self.assertEqual(list(schema), [a1])

    def test_dense_array_schema_fp_domain_error(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(ctx,
            tiledb.Dim(ctx, domain=(1, 8), tile=2, dtype=np.float64))
        att = tiledb.Attr(ctx, "val", dtype=np.float64)

        with self.assertRaises(tiledb.TileDBError):
            tiledb.ArraySchema(ctx, domain=dom, attrs=(att,))

    def test_sparse_schema(self):
        ctx = tiledb.Ctx()

        # create dimensions
        d1 = tiledb.Dim(ctx, "", domain=(1, 1000), tile=10, dtype="uint64")
        d2 = tiledb.Dim(ctx, "d2", domain=(101, 10000), tile=100, dtype="uint64")

        # create domain
        domain = tiledb.Domain(ctx, d1, d2)

        # create attributes
        a1 = tiledb.Attr(ctx, "", dtype="int32,int32,int32")
        a2 = tiledb.Attr(ctx, "a2", compressor=("gzip", -1), dtype="float32")

        # create sparse array with schema
        schema = tiledb.ArraySchema(ctx,
                                    domain=domain,
                                    attrs=(a1, a2),
                                    capacity=10,
                                    cell_order='col-major',
                                    tile_order='row-major',
                                    coords_compressor=('zstd', 4),
                                    offsets_compressor=('lz4', 5),
                                    sparse=True)
        schema.dump()
        self.assertTrue(schema.sparse)
        self.assertEqual(schema.capacity, 10)
        self.assertEqual(schema.cell_order, "col-major")
        self.assertEqual(schema.tile_order, "row-major")
        self.assertEqual(schema.coords_compressor, ('zstd', 4))
        self.assertEqual(schema.offsets_compressor, ('lz4', 5))
        self.assertEqual(schema.domain, domain)
        self.assertEqual(schema.ndim, 2)
        self.assertEqual(schema.shape, (1000, 9900))
        self.assertEqual(schema.nattr, 2)
        self.assertEqual(schema.attr(0), a1)
        self.assertEqual(schema.attr("a2"), a2)
        self.assertEqual(schema,
                         tiledb.ArraySchema(ctx,
                                    domain=domain,
                                    attrs=(a1, a2),
                                    capacity=10,
                                    cell_order='col-major',
                                    tile_order='row-major',
                                    coords_compressor=('zstd', 4),
                                    offsets_compressor=('lz4', 5),
                                    sparse=True))
        # test iteration over attributes
        self.assertEqual(list(schema), [a1, a2])

    def test_sparse_schema_filter_list(self):
        ctx = tiledb.Ctx()

        # create dimensions
        d1 = tiledb.Dim(ctx, "", domain=(1, 1000), tile=10, dtype="uint64")
        d2 = tiledb.Dim(ctx, "d2", domain=(101, 10000), tile=100, dtype="uint64")

        # create domain
        domain = tiledb.Domain(ctx, d1, d2)

        # create attributes
        a1 = tiledb.Attr(ctx, "", dtype="int32,int32,int32")
        a2 = tiledb.Attr(ctx, "a2", compressor=("gzip", -1), dtype="float32")

        off_filters = tiledb.libtiledb.FilterList(ctx,
                        filters=[tiledb.libtiledb.ZstdFilter(ctx, level=10)],
                        chunksize=2048)

        coords_filters = tiledb.libtiledb.FilterList(ctx,
                        filters=[tiledb.libtiledb.Bzip2Filter(ctx, level=5)],
                        chunksize=4096)

        # create sparse array with schema
        schema = tiledb.ArraySchema(ctx,
                                    domain=domain,
                                    attrs=(a1, a2),
                                    capacity=10,
                                    cell_order='col-major',
                                    tile_order='row-major',
                                    coords_filters=coords_filters,
                                    offsets_filters=off_filters,
                                    sparse=True)
        schema.dump()
        self.assertTrue(schema.sparse)


class ArrayTest(DiskTestCase):

    def create_array_schema(self, ctx):
        domain = tiledb.Domain(
            ctx,
            tiledb.Dim(ctx, domain=(1, 8), tile=2),
            tiledb.Dim(ctx, domain=(1, 8), tile=2))
        a1 = tiledb.Attr(ctx, "val", dtype='f8')
        return tiledb.ArraySchema(ctx, domain=domain, attrs=(a1,))

    def test_array_create(self):
        ctx = tiledb.Ctx()
        schema = self.create_array_schema(ctx)

        # persist array schema
        tiledb.libtiledb.Array.create(self.path("foo"), schema)

        # load array in readonly mode
        array = tiledb.libtiledb.Array(ctx, self.path("foo"), mode='r')
        self.assertTrue(array.isopen)
        self.assertEqual(array.schema, schema)
        self.assertEqual(array.mode, 'r')

        # we have not written anything, so the array is empty
        self.assertIsNone(array.nonempty_domain())

        # this should be a no-op
        array.consolidate()

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
        ctx = tiledb.Ctx()
        schema = self.create_array_schema(ctx)
        # persist array schema
        tiledb.libtiledb.Array.create(self.path("foo"), schema,
                                      key=b"0123456789abcdeF0123456789abcdeF")

        # check that we can open the array sucessfully
        for key in (b"0123456789abcdeF0123456789abcdeF", "0123456789abcdeF0123456789abcdeF"):
            with tiledb.libtiledb.Array(ctx, self.path("foo"), mode='r', key=key) as array:
                self.assertTrue(array.isopen)
                self.assertEqual(array.schema, schema)
                self.assertEqual(array.mode, 'r')
            tiledb.consolidate(ctx, uri=self.path("foo"), key=key)

        # check that opening the array with the wrong key fails:
        with self.assertRaises(tiledb.TileDBError):
            tiledb.libtiledb.Array(ctx, self.path("foo"), mode='r',
                                   key=b"0123456789abcdeF0123456789abcdeX")

        # check that opening the array with the wrong key length fails:
        with self.assertRaises(tiledb.TileDBError):
            tiledb.libtiledb.Array(ctx, self.path("foo"), mode='r',
                                   key=b"0123456789abcdeF0123456789abcde")

        # check that consolidating the array with the wrong key fails:
        with self.assertRaises(tiledb.TileDBError):
            tiledb.consolidate(ctx, uri=self.path("foo"),
                               key=b"0123456789abcdeF0123456789abcde")

    def test_array_doesnt_exist(self):
        ctx = tiledb.Ctx()
        with self.assertRaises(tiledb.TileDBError):
            tiledb.libtiledb.Array(ctx, self.path("foo"), mode='r')


class DenseArrayTest(DiskTestCase):

    def test_array_1d(self):
        A = np.arange(1050)

        ctx = tiledb.Ctx()
        dom = tiledb.Domain(ctx, tiledb.Dim(ctx, domain=(0, 1049), tile=100, dtype=np.int64))
        att = tiledb.Attr(ctx, dtype=A.dtype)
        schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(att,))
        tiledb.DenseArray.create(self.path("foo"), schema)

        with tiledb.DenseArray(ctx, self.path("foo"), mode='r') as T:
            self.assertEqual(len(A), len(T))
            self.assertEqual(A.ndim, T.ndim)
            self.assertEqual(A.shape, T.shape)

            self.assertEqual(1, T.nattr)
            self.assertEqual(A.dtype, T.attr(0).dtype)

            self.assertIsInstance(T.timestamp, int)
            self.assertTrue(T.timestamp > 0)

            # check empty array
            B = T[:]

            self.assertEqual(A.shape, B.shape)
            self.assertEqual(A.dtype, B.dtype)
            self.assertIsNone(T.nonempty_domain())

        with tiledb.DenseArray(ctx, self.path("foo"), mode='w') as T:
            # check set array
            T[:] = A

        read1_timestamp = -1
        with tiledb.DenseArray(ctx, self.path("foo"), mode='r') as T:
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

        with tiledb.DenseArray(ctx, self.path("foo"), mode='w') as T:
            # check partial assignment
            B = np.arange(1e5, 2e5).astype(A.dtype)
            T[190:310] = B[190:310]

        read2_timestamp = -1
        with tiledb.DenseArray(ctx, self.path("foo"), mode='r') as T:
            assert_array_equal(A[:190], T[:190])
            assert_array_equal(B[190:310], T[190:310])
            assert_array_equal(A[310:], T[310:])

            # test timestamps are updated
            read2_timestamp = T.timestamp
            self.assertTrue(read2_timestamp > read1_timestamp)

    def test_array_1d_set_scalar(self):
        A = np.zeros(50)

        ctx = tiledb.Ctx()
        dom = tiledb.Domain(ctx, tiledb.Dim(ctx, domain=(0, 49), tile=50))
        att = tiledb.Attr(ctx, dtype=A.dtype)
        schema = tiledb.ArraySchema(ctx, dom, (att,))

        tiledb.DenseArray.create(self.path("foo"), schema)

        with tiledb.DenseArray(ctx, self.path("foo"), mode='w') as T:
            T[:] = A

        with tiledb.DenseArray(ctx, self.path("foo"), mode='r') as T:
            assert_array_equal(A, T[:])

        for value in (-1, 3, 10):
            with tiledb.DenseArray(ctx, self.path("foo"), mode='w') as T:
                A[5:25] = value
                T[5:25] = value
            with tiledb.DenseArray(ctx, self.path("foo"), mode='r') as T:
                assert_array_equal(A, T[:])
            with tiledb.DenseArray(ctx, self.path("foo"), mode='w') as T:
                A[:] = value
                T[:] = value
            with tiledb.DenseArray(ctx, self.path("foo"), mode='r') as T:
                assert_array_equal(A, T[:])

    def test_array_id_point_queries(self):
        #TODO: handle queries like T[[2, 5, 10]] = ?
        pass

    def test_array_2d(self):
        A = np.arange(10000).reshape((1000, 10))

        ctx = tiledb.Ctx()
        dom = tiledb.Domain(ctx,
                       tiledb.Dim(ctx, domain=(0, 999), tile=100),
                       tiledb.Dim(ctx, domain=(0, 9), tile=2))
        att = tiledb.Attr(ctx, dtype=A.dtype)
        schema = tiledb.ArraySchema(ctx, dom, (att,))

        tiledb.DenseArray.create(self.path("foo"), schema)

        with tiledb.DenseArray(ctx, self.path("foo"), mode='r') as T:
            self.assertEqual(len(A), len(T))
            self.assertEqual(A.ndim, T.ndim)
            self.assertEqual(A.shape, T.shape)

            self.assertEqual(1, T.nattr)
            self.assertEqual(A.dtype, T.attr(0).dtype)

            # check that the non-empty domain is None
            self.assertIsNone(T.nonempty_domain())

        with tiledb.DenseArray(ctx, self.path("foo"), mode='w') as T:
            # Set data
            T[:] = A

        with tiledb.DenseArray(ctx, self.path("foo"), mode='r') as T:
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

        with tiledb.DenseArray(ctx, self.path("foo"), mode='w') as T:
            # check partial assignment
            B = np.arange(10000, 20000).reshape((1000, 10))
            T[190:310, 3:7] = B[190:310, 3:7]

        with tiledb.DenseArray(ctx, self.path("foo"), mode='r') as T:
            assert_array_equal(A[:190], T[:190])
            assert_array_equal(A[:, :3], T[:, :3])
            assert_array_equal(B[190:310, 3:7], T[190:310, 3:7])
            assert_array_equal(A[310:], T[310:])
            assert_array_equal(A[:, 7:], T[:, 7:])

    def test_open_with_timestamp(self):
        import time
        A = np.zeros(3)

        ctx = tiledb.Ctx()
        dom = tiledb.Domain(ctx, tiledb.Dim(ctx, domain=(0, 2), tile=3, dtype=np.int64))
        att = tiledb.Attr(ctx, dtype=A.dtype)
        schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(att,))
        tiledb.DenseArray.create(self.path("foo"), schema)

        # write
        with tiledb.DenseArray(ctx, self.path("foo"), mode='w') as T:
            T[:] = A

        read1_timestamp = -1
        with tiledb.DenseArray(ctx, self.path("foo"), mode='r') as T:
            read1_timestamp = T.timestamp
            self.assertEqual(T[0], 0)
            self.assertEqual(T[1], 0)
            self.assertEqual(T[2], 0)

        # sleep 200ms and write
        time.sleep(0.2)
        with tiledb.DenseArray(ctx, self.path("foo"), mode='w') as T:
            T[0:1] = 1

        read2_timestamp = -1
        with tiledb.DenseArray(ctx, self.path("foo"), mode='r') as T:
            read2_timestamp = T.timestamp
            self.assertTrue(read2_timestamp > read1_timestamp)

        # sleep 200ms and write
        time.sleep(0.2)
        with tiledb.DenseArray(ctx, self.path("foo"), mode='w') as T:
            T[1:2] = 2

        read3_timestamp = -1
        with tiledb.DenseArray(ctx, self.path("foo"), mode='r') as T:
            read3_timestamp = T.timestamp
            self.assertTrue(read3_timestamp > read2_timestamp > read1_timestamp)

        # read at first timestamp
        with tiledb.DenseArray(ctx, self.path("foo"), timestamp=read1_timestamp, mode='r') as T:
            self.assertEqual(T[0], 0)
            self.assertEqual(T[1], 0)
            self.assertEqual(T[2], 0)

        # read at second timestamp
        with tiledb.DenseArray(ctx, self.path("foo"), timestamp=read2_timestamp, mode='r') as T:
            self.assertEqual(T[0], 1)
            self.assertEqual(T[1], 0)
            self.assertEqual(T[2], 0)

        # read at third timestamp
        with tiledb.DenseArray(ctx, self.path("foo"), timestamp=read3_timestamp, mode='r') as T:
            self.assertEqual(T[0], 1)
            self.assertEqual(T[1], 2)
            self.assertEqual(T[2], 0)

    def test_ncell_attributes(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(ctx, tiledb.Dim(ctx, domain=(0, 9), tile=10, dtype=int))
        attr = tiledb.Attr(ctx, dtype=[("", np.int32), ("", np.int32)])
        schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(attr,))
        tiledb.DenseArray.create(self.path("foo"), schema)

        A = np.ones((10,), dtype=[("", np.int32), ("", np.int32)])
        self.assertEqual(A.dtype, attr.dtype)

        with tiledb.DenseArray(ctx, self.path("foo"), mode='w') as T:
            T[:] = A
        with tiledb.DenseArray(ctx, self.path("foo"), mode='r') as T:
            assert_array_equal(A, T[:])
            assert_array_equal(A[:5], T[:5])

    def test_multiple_attributes(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(ctx,
                       tiledb.Dim(ctx, domain=(0, 1), tile=1, dtype=np.int64),
                       tiledb.Dim(ctx, domain=(0, 3), tile=4, dtype=np.int64))
        attr_int = tiledb.Attr(ctx, "ints", dtype=int)
        attr_float = tiledb.Attr(ctx, "floats", dtype=float)
        schema = tiledb.ArraySchema(ctx,
                                    domain=dom,
                                    attrs=(attr_int, attr_float))
        tiledb.DenseArray.create(self.path("foo"), schema)

        V_ints = np.array([[0, 1, 2, 3,],
                           [4, 6, 7, 5]])
        V_floats = np.array([[0.0, 1.0, 2.0, 3.0,],
                             [4.0, 6.0, 7.0, 5.0]])

        V = {"ints": V_ints, "floats": V_floats}
        with tiledb.DenseArray(ctx, self.path("foo"), mode='w') as T:
            T[:] = V

        with tiledb.DenseArray(ctx, self.path("foo"), mode='r') as T:
            R = T[:]
            assert_array_equal(V["ints"], R["ints"])
            assert_array_equal(V["floats"], R["floats"])

            R = T.query(attrs=("ints",))[1:3]
            assert_array_equal(V["ints"][1:3], R["ints"])

            R = T.query(attrs=("floats",), order='F')[:]
            self.assertTrue(R["floats"].flags.f_contiguous)

            R = T.query(attrs=("ints",), coords=True)[0, 0:3]
            self.assertTrue("coords" in R)
            assert_array_equal(R["coords"],
                               np.array([(0, 0), (0, 1), (0, 2)],
                                        dtype=[('__dim_0', '<i8'),
                                               ('__dim_1', '<i8')]))

            # Global order returns results as a linear buffer
            R = T.query(attrs=("ints",), order='G')[:]
            self.assertEqual(R["ints"].shape, (8,))

            with self.assertRaises(tiledb.TileDBError):
                T.query(attrs=("unknown"))[:]

        with tiledb.DenseArray(ctx, self.path("foo"), mode='w') as T:
            # check error ncells length
            V["ints"] = V["ints"][1:2].copy()
            with self.assertRaises(tiledb.TileDBError):
                T[:] = V

            # check error attribute does not exist
            V["foo"] = V["ints"].astype(np.int8)
            with self.assertRaises(tiledb.TileDBError):
                T[:] = V


class SparseArray(DiskTestCase):

    @unittest.expectedFailure
    def test_simple_1d_sparse_vector(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(ctx, tiledb.Dim(ctx, domain=(0, 3), tile=4, dtype=int))
        att = tiledb.Attr(ctx, dtype=int)
        schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(att,), sparse=True)
        tiledb.SparseArray.create(self.path("foo"), schema)

        values = np.array([3, 4])
        with tiledb.SparseArray(ctx, self.path("foo"), mode='w') as T:
            T[[1, 2]] = values

        with tiledb.SparseArray(ctx, self.path("foo"), mode='r') as T:
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
        dom = tiledb.Domain(ctx, tiledb.Dim(ctx, "x", domain=(0.0, 10.0), tile=2.0, dtype=float))
        attr = tiledb.Attr(ctx, dtype=float)
        schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(attr,), sparse=True)
        tiledb.SparseArray.create(self.path("foo"), schema)

        values = np.array([3.3, 2.7])
        with tiledb.SparseArray(ctx, self.path("foo"), mode='w') as T:
            T[[2.5, 4.2]] = values
        with tiledb.SparseArray(ctx, self.path("foo"), mode='r') as T:
            assert_array_equal(T[[2.5, 4.2]], values)

    @unittest.expectedFailure
    def test_sparse_unordered_fp_domain(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(ctx, tiledb.Dim(ctx, "x", domain=(0.0, 10.0), tile=2.0, dtype=float))
        attr = tiledb.Attr(ctx, dtype=float)
        schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(attr,), sparse=True)
        tiledb.SparseArray.create(self.path("foo"), schema)
        values = np.array([3.3, 2.7])
        with tiledb.SparseArray(ctx, self.path("foo"), mode='w') as T:
            T[[4.2, 2.5]] = values

        with tiledb.SparseArray(ctx, self.path("foo"), mode='r') as T:
            assert_array_equal(T[[2.5, 4.2]], values[::-1])

    @unittest.expectedFailure
    def test_multiple_attributes(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(ctx,
            tiledb.Dim(ctx, domain=(1, 10), tile=10, dtype=int),
            tiledb.Dim(ctx, domain=(1, 10), tile=10, dtype=int))
        attr_int = tiledb.Attr(ctx, "ints", dtype=int)
        attr_float = tiledb.Attr(ctx, "floats", dtype="float")
        schema = tiledb.ArraySchema(ctx,
                                domain=dom,
                                attrs=(attr_int, attr_float,),
                                sparse=True)
        tiledb.SparseArray.create(self.path("foo"), schema)

        I = np.array([1, 1, 1, 2, 3, 3, 3, 4])
        J = np.array([1, 2, 4, 3, 1, 6, 7, 5])

        V_ints = np.array([0, 1, 2, 3, 4, 6, 7, 5])
        V_floats = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 5.0])

        V = {"ints": V_ints, "floats": V_floats}
        with tiledb.SparseArray(ctx, self.path("foo"), mode='w') as T:
            T[I, J] = V
        with tiledb.SparseArray(ctx, self.path("foo"), mode='r') as T:
            R = T[I, J]
        assert_array_equal(V["ints"], R["ints"])
        assert_array_equal(V["floats"], R["floats"])

        # check error attribute does not exist
        # TODO: should this be an attribute error?
        with tiledb.SparseArray(ctx, self.path("foo"), mode='w') as T:
            V["foo"] = V["ints"].astype(np.int8)
            with self.assertRaises(tiledb.TileDBError):
                T[I, J] = V

            # check error ncells length
            V["ints"] = V["ints"][1:2].copy()
            with self.assertRaises(AttributeError):
                T[I, J] = V

    def test_subarray(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(ctx,
                tiledb.Dim(ctx, "x", domain=(1, 10000), tile=100, dtype=int))
        att = tiledb.Attr(ctx, "", dtype=float)
        schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(att,), sparse=True)
        tiledb.SparseArray.create(self.path("foo"), schema)

        with tiledb.SparseArray(ctx, self.path("foo"), mode='r') as T:
            self.assertIsNone(T.nonempty_domain())

        with tiledb.SparseArray(ctx, self.path("foo"), mode='w') as T:
            T[[50, 60, 100]] = [1.0, 2.0, 3.0]

        with tiledb.SparseArray(ctx, self.path("foo"), mode='r') as T:
            self.assertEqual(((50, 100),), T.nonempty_domain())

            # retrieve just valid coordinates in subarray T[40:60]
            assert_array_equal(T[40:61]["coords"]["x"], [50, 60])

            #TODO: dropping coords with one anon value returns just an array
            res = T.query(coords=False)[40:61]
            assert_array_equal(res[""], [1.0, 2.0])
            self.assertEqual(("coords" in res), False)


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
        dom = tiledb.Domain(ctx, tiledb.Dim(ctx, domain=(0, 1049), tile=100))
        att = tiledb.Attr(ctx, dtype=int)
        schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(att,))
        tiledb.DenseArray.create(self.path("foo"), schema)

        with tiledb.DenseArray(ctx, self.path("foo"), mode='w') as T:
            T[:] = A

        with tiledb.DenseArray(ctx, self.path("foo"), mode='r') as T:
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
        dom = tiledb.Domain(ctx,
                       tiledb.Dim(ctx, domain=(0, 999), tile=100),
                       tiledb.Dim(ctx, domain=(0, 9), tile=2))
        att = tiledb.Attr(ctx, dtype=A.dtype)
        schema = tiledb.ArraySchema(ctx, dom, (att,))
        tiledb.DenseArray.create(self.path("foo"), schema)


        with tiledb.DenseArray(ctx, self.path("foo"), mode='w') as T:
            T[:] = A

        with tiledb.DenseArray(ctx, self.path("foo"), mode='r') as T:
            for idx in self.good_index_1d:
                self._test_index(A, T, idx)

            for idx in self.bad_index_2d:
                with self.assertRaises(IndexError):
                    T[idx]


class RWTest(DiskTestCase):

    def test_read_write(self):
        ctx = tiledb.Ctx()

        dom = tiledb.Domain(ctx, tiledb.Dim(ctx, domain=(0, 2), tile=3))
        att = tiledb.Attr(ctx, dtype='int64')
        schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(att,))
        tiledb.libtiledb.Array.create(self.path("foo"), schema)

        np_array = np.array([1, 2, 3], dtype='int64')

        with tiledb.DenseArray(ctx, self.path("foo"), mode="w") as arr:
            arr.write_direct(np_array)

        with tiledb.DenseArray(ctx, self.path("foo"), mode="r") as arr:
            arr.dump()
            self.assertEqual(arr.nonempty_domain(), ((0, 2),))
            self.assertEqual(arr.ndim, np_array.ndim)
            assert_array_equal(arr.read_direct(), np_array)


class NumpyToArray(DiskTestCase):

    def test_to_array0d(self):
        # Cannot create 0-dim arrays in TileDB
        ctx = tiledb.Ctx()
        np_array = np.array(1)
        with self.assertRaises(tiledb.TileDBError):
            tiledb.DenseArray.from_numpy(ctx, self.path("foo"), np_array)

    def test_to_array1d(self):
        ctx = tiledb.Ctx()
        np_array = np.array([1.0, 2.0, 3.0])
        arr = tiledb.DenseArray.from_numpy(ctx, self.path("foo"), np_array)
        assert_array_equal(arr[:], np_array)

    def test_to_array2d(self):
        ctx = tiledb.Ctx()
        np_array = np.ones((100, 100), dtype='i8')
        arr = tiledb.DenseArray.from_numpy(ctx, self.path("foo"), np_array)
        assert_array_equal(arr[:], np_array)

    def test_to_array3d(self):
        ctx = tiledb.Ctx()
        np_array = np.ones((1, 1, 1), dtype='i1')
        arr = tiledb.DenseArray.from_numpy(ctx, self.path("foo"), np_array)
        assert_array_equal(arr[:], np_array)

    def test_array_interface(self):
        # Tests that __array__ interface works
        ctx = tiledb.Ctx()
        np_array1 = np.arange(1, 10)
        arr1 = tiledb.DenseArray.from_numpy(ctx, self.path("arr1"), np_array1)
        assert_array_equal(np.array(arr1), np_array1)

        # Test that __array__ interface throws an error when number of attributes > 1
        dom = tiledb.Domain(ctx, tiledb.Dim(ctx, domain=(0, 2), tile=3))
        foo = tiledb.Attr(ctx, "foo", dtype='i8')
        bar = tiledb.Attr(ctx, "bar", dtype='i8')
        schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(foo, bar))
        tiledb.DenseArray.create(self.path("arr2"), schema)
        with self.assertRaises(ValueError):
            with tiledb.DenseArray(ctx, self.path("arr2"), mode='r') as arr2:
                np.array(arr2)

    def test_array_getindex(self):
        # Tests that __getindex__ interface works
        ctx = tiledb.Ctx()
        np_array = np.arange(1, 10)
        arr = tiledb.DenseArray.from_numpy(ctx, self.path("foo"), np_array)
        assert_array_equal(arr[5:10], np_array[5:10])


class KVSchema(DiskTestCase):

    def test_basic_kv_schema(self):
        ctx = tiledb.Ctx()
        a1 = tiledb.Attr(ctx, "a1", dtype=bytes)
        schema = tiledb.KVSchema(ctx, attrs=(a1,), capacity=10000)
        self.assertIsInstance(schema, tiledb.KVSchema)
        self.assertEqual(schema.attr("a1"), a1)
        self.assertEqual(schema.attr(0), a1)
        self.assertEqual(schema.capacity, 10000)


class KVArray(DiskTestCase):

    def test_kv_write_schema_load(self):
        # create a kv array
        ctx = tiledb.Ctx()
        a1 = tiledb.Attr(ctx, "value", dtype=bytes)
        schema = tiledb.KVSchema(ctx, attrs=(a1,))
        # persist kv schema
        tiledb.KV.create(ctx, self.path("foo"), schema)
        self.assertEqual(tiledb.KVSchema.load(ctx, self.path("foo")), schema)

    def test_kv_contains(self):
        # create a kv array
        ctx = tiledb.Ctx()
        a1 = tiledb.Attr(ctx, "value", dtype=bytes)
        schema = tiledb.KVSchema(ctx, attrs=(a1,))
        tiledb.KV.create(ctx, self.path("foo"), schema)

        with tiledb.KV(ctx, self.path("foo"), mode='r') as kv:
            self.assertIsInstance(kv.timestamp, int)
            self.assertTrue(kv.timestamp > 0)
            self.assertFalse("foo" in kv)

        with tiledb.KV(ctx, self.path("foo"), mode='w') as kv:
            kv['foo'] = 'bar'

        with tiledb.KV(ctx, self.path("foo"), mode='r') as kv:
            self.assertTrue("foo" in kv)

    def test_kv_write_load_read(self):
        # create a kv array
        ctx = tiledb.Ctx()
        a1 = tiledb.Attr(ctx, "value", dtype=bytes)
        schema = tiledb.KVSchema(ctx, attrs=(a1,))
        # persist kv schema
        tiledb.KV.create(ctx, self.path("foo"), schema)

        # load kv array, write, close / delete
        kv = tiledb.KV(ctx, self.path("foo"), mode='w')
        kv['foo'] = 'bar'
        kv.close()
        del kv

        # try to load it
        with tiledb.KV(ctx, self.path("foo"), mode='r') as kv:
          self.assertEqual(kv["foo"], 'bar')
          self.assertTrue('foo' in kv)
          self.assertFalse('bar' in kv)

    def test_kv_write_load_read_encrypted(self):
         # create a kv array
        ctx = tiledb.Ctx()
        a1 = tiledb.Attr(ctx, "value", dtype=bytes)
        schema = tiledb.KVSchema(ctx, attrs=(a1,))
        # persist kv schema
        tiledb.KV.create(ctx, self.path("foo"), schema, key=b"0123456789abcdeF0123456789abcdeF")

        with tiledb.KV(ctx, self.path("foo"), mode='w', key=b"0123456789abcdeF0123456789abcdeF") as kv:
            kv['foo'] = 'bar'

        # try to load it
        with tiledb.KV(ctx, self.path("foo"), mode='r', key=b"0123456789abcdeF0123456789abcdeF") as kv:
          self.assertEqual(kv["foo"], 'bar')
          self.assertTrue('foo' in kv)
          self.assertFalse('bar' in kv)

        # loading with no key fails
        with self.assertRaises(tiledb.TileDBError):
            with tiledb.KV(ctx, self.path("foo")) as kv:
                self.assertTrue('foo' in kv)

        # loading with wrong key fails
        with self.assertRaises(tiledb.TileDBError):
            with tiledb.KV(ctx, self.path("foo"), key=b"0123456789abcdeF0123456789abcdeZ") as kv:
                self.assertTrue('foo' in kv)

    def test_kv_update_reload(self):
        # create a kv array
        ctx = tiledb.Ctx()
        a1 = tiledb.Attr(ctx, "val", dtype=bytes)
        # persist kv schema
        schema = tiledb.KVSchema(ctx, attrs=(a1,))
        tiledb.KV.create(ctx, self.path("foo"), schema)

        # load kv array
        with tiledb.KV(ctx, self.path("foo"), mode='w') as kv1:
            kv1['foo'] = 'bar'
            kv1.flush()

            with tiledb.KV(ctx, self.path("foo"), mode='r') as kv2:
                self.assertTrue('foo' in kv2)
                kv1['bar'] = 'baz'
                kv1.flush()
                self.assertFalse('bar' in kv2)
                kv2.reopen()
                self.assertTrue('bar' in kv2)

    def test_key_not_found(self):
        # create a kv array
        ctx = tiledb.Ctx()
        a1 = tiledb.Attr(ctx, "value", dtype=bytes)
        schema = tiledb.KVSchema(ctx, attrs=(a1,))
        tiledb.KV.create(ctx, self.path("foo"), schema)
        with tiledb.KV(ctx, self.path("foo"), mode='r') as kv:
            self.assertRaises(KeyError, kv.__getitem__, "not here")

    def test_kv_dict(self):
        # create a kv array
        ctx = tiledb.Ctx()
        a1 = tiledb.Attr(ctx, "value", dtype=bytes)
        schema = tiledb.KVSchema(ctx, attrs=(a1,))
        tiledb.KV.create(ctx, self.path("foo"), schema)

        with tiledb.KV(ctx, self.path("foo"), mode='w') as kv:
            kv['foo'] = 'bar'
            kv['baz'] = 'foo'

        with tiledb.KV(ctx, self.path("foo"), mode='r') as kv:
            self.assertEqual(kv.dict(), {'foo': 'bar', 'baz': 'foo'})
            self.assertEqual(dict(kv), {'foo': 'bar', 'baz': 'foo'})

    def test_kv_timestamp(self):
        import time
        # create a new kv array
        ctx = tiledb.Ctx()
        a1 = tiledb.Attr(ctx, "value", dtype=bytes)
        schema = tiledb.KVSchema(ctx, attrs=(a1,))
        tiledb.KV.create(ctx, self.path("foo"), schema)

        with tiledb.KV(ctx, self.path("foo"), mode='w') as kv:
            kv['foo'] = 'bar'

        read1_timestamp = -1
        with tiledb.KV(ctx, self.path("foo"), mode='r') as kv:
            self.assertEqual(kv['foo'], 'bar')
            read1_timestamp = kv.timestamp
        self.assertTrue(read1_timestamp > 0)

        # sleep for 200 ms, check that timestamp is updated
        time.sleep(0.2)
        with tiledb.KV(ctx, self.path("foo"), mode='r') as kv:
            self.assertTrue(kv.timestamp > read1_timestamp)
            self.assertEqual(kv['foo'], 'bar')

        # write some more data at a later time
        with tiledb.KV(ctx, self.path("foo"), mode='w') as kv:
            kv['foo'] = 'baz'
            kv['aaa'] = 'bbb'

        # check that we can open up at different timepoints deterministically
        read2_timestamp = -1
        with tiledb.KV(ctx, self.path("foo"), mode='r') as kv:
            read2_timestamp = kv.timestamp
            self.assertTrue(read2_timestamp > read1_timestamp)
            self.assertEqual(kv['foo'], 'baz')
            self.assertEqual(kv['aaa'], 'bbb')

        with tiledb.KV(ctx, self.path("foo"), timestamp=read1_timestamp, mode='r') as kv:
            self.assertEqual(kv.timestamp, read1_timestamp)
            self.assertEqual(kv['foo'], 'bar')

        with tiledb.KV(ctx, self.path("foo"), timestamp=read2_timestamp, mode='r') as kv:
            self.assertEqual(kv.timestamp, read2_timestamp)
            self.assertEqual(kv['foo'], 'baz')
            self.assertEqual(kv['aaa'], 'bbb')

    def test_multiattribute(self):
        ctx = tiledb.Ctx()
        a1 = tiledb.Attr(ctx, "ints", dtype=int)
        a2 = tiledb.Attr(ctx, "floats", dtype=float)
        schema = tiledb.KVSchema(ctx, attrs=(a1, a2))
        tiledb.KV.create(ctx, self.path("foo"), schema)

        kv = tiledb.KV(ctx, self.path("foo"), mode='r')
        self.assertIsInstance(kv, tiledb.KV)

        # known failure
        #kv['foo'] = {"ints": 1, "floats": 2.0}
        #self.assertEqual(kv["foo"]["ints"], 1)
        #self.assertEqual(kv["foo"]["floats"], 2.0)


class VFS(DiskTestCase):

    def test_supports(self):
        ctx = tiledb.Ctx()
        vfs = tiledb.VFS(ctx)

        self.assertTrue(vfs.supports("file"))
        self.assertIsInstance(vfs.supports("s3"), bool)
        self.assertIsInstance(vfs.supports("hdfs"), bool)

        with self.assertRaises(ValueError):
            vfs.supports("invalid")

    def test_dir(self):
        ctx = tiledb.Ctx()
        vfs = tiledb.VFS(ctx)

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
        vfs = tiledb.VFS(ctx)

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
        vfs = tiledb.VFS(ctx)

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
        vfs = tiledb.VFS(ctx)

        buffer = b"bar"
        fh = vfs.open(self.path("foo"), "w")
        vfs.write(fh, buffer)
        vfs.close(fh)
        self.assertEqual(vfs.file_size(self.path("foo")), 3)

        fh = vfs.open(self.path("foo"), "r")
        self.assertEqual(vfs.read(fh, 0, 3), buffer)
        vfs.close(fh)

        # write / read empty input
        fh = vfs.open(self.path("baz"), "w")
        vfs.write(fh, b"")
        vfs.close(fh)
        self.assertEqual(vfs.file_size(self.path("baz")), 0)

        fh = vfs.open(self.path("baz"), "r")
        self.assertEqual(vfs.read(fh, 0, 0), b"")
        vfs.close(fh)

        # read from file that does not exist
        with self.assertRaises(tiledb.TileDBError):
            vfs.open(self.path("do_not_exist"), "r")

    def test_io(self):
        ctx = tiledb.Ctx()
        vfs = tiledb.VFS(ctx)

        buffer = b"0123456789"
        io = tiledb.FileIO(vfs, self.path("foo"), mode="w")
        io.write(buffer)
        io.flush()
        self.assertEqual(io.tell(), len(buffer))

        io = tiledb.FileIO(vfs, self.path("foo"), mode="r")
        with self.assertRaises(IOError):
            io.write(b"foo")

        self.assertEqual(vfs.file_size(self.path("foo")), len(buffer))

        io = tiledb.FileIO(vfs, self.path("foo"), mode='r')
        self.assertEqual(io.read(3), b'012')
        self.assertEqual(io.tell(), 3)
        self.assertEqual(io.read(3), b'345')
        self.assertEqual(io.tell(), 6)
        self.assertEqual(io.read(10), b'6789')
        self.assertEqual(io.tell(), 10)

        # seek from beginning
        io.seek(0)
        self.assertEqual(io.tell(), 0)
        self.assertEqual(io.read(), buffer)

        # seek must be positive when SEEK_SET
        with self.assertRaises(ValueError):
            io.seek(-1, 0)

        # seek from current position
        io.seek(5)
        self.assertEqual(io.tell(), 5)
        io.seek(3, 1)
        self.assertEqual(io.tell(), 8)
        io.seek(-3, 1)
        self.assertEqual(io.tell(), 5)

        # seek from end
        io.seek(-4, 2)
        self.assertEqual(io.tell(), 6)

        # Test readall
        io.seek(0)
        self.assertEqual(io.readall(), buffer)
        self.assertEqual(io.tell(), 10)

        io.seek(5)
        self.assertEqual(io.readall(), buffer[5:])
        self.assertEqual(io.readall(), b"")
