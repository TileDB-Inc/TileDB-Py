from __future__ import absolute_import

import unittest

import numpy as np
from numpy.testing import assert_array_equal

import tiledb
from tiledb import libtiledb as t
from tiledb.tests.common import DiskTestCase


class VersionTest(unittest.TestCase):

    def test_version(self):
        v = tiledb.libtiledb.version()
        self.assertIsInstance(v, tuple)
        self.assertTrue(len(v) == 3)
        self.assertTrue(v[0] >= 1, "TileDB major version must be >= 1")


class Config(DiskTestCase):

    def test_config(self):
        config = t.Config()
        config["sm.tile_cache_size"] = 100
        ctx = t.Ctx(config)

    def test_ctx_config(self):
        ctx = t.Ctx({"sm.tile_cache_size": 100})
        config = ctx.config()
        self.assertEqual(config["sm.tile_cache_size"], "100")

    def test_config_bad_param(self):
        config = t.Config()
        config["sm.foo"] = "bar"
        ctx = t.Ctx(config)
        self.assertEqual(ctx.config()["sm.foo"], "bar")

    def test_config_unset(self):
        config = t.Config()
        config["sm.tile_cach_size"] = 100
        del config["sm.tile_cache_size"]
        # check that config parameter is default
        self.assertEqual(config["sm.tile_cache_size"], t.Config()["sm.tile_cache_size"])

    def test_config_from_file(self):
        config_path = self.path("config")
        with open(config_path, "w") as fh:
            fh.write("sm.tile_cache_size 100")
        config = t.Config.load(config_path)
        self.assertEqual(config["sm.tile_cache_size"], "100")

    def test_ctx_config_from_file(self):
        config_path = self.path("config")
        with open(config_path, "w") as fh:
            fh.write("sm.tile_cache_size 100")
        ctx = t.Ctx(config=t.Config.load(config_path))
        config = ctx.config()
        self.assertEqual(config["sm.tile_cache_size"], "100")

    def test_ctx_config_dict(self):
        ctx = t.Ctx(config={"sm.tile_cache_size": '100'})
        config = ctx.config()
        self.assertIsInstance(config, t.Config)
        self.assertEqual(config["sm.tile_cache_size"], '100')


class GroupTestCase(DiskTestCase):

    def setUp(self):
        super().setUp()

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
        return t.object_type(ctx, uri) == "group"


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
            groups.append((path, obj))

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

        tiledb.move(ctx, self.group4, self.path("group1/group4"))

        self.assertTrue(self.is_group(ctx, self.path("group1/group4")))
        self.assertFalse(self.is_group(ctx, self.group4))

        with self.assertRaises(tiledb.TileDBError):
            tiledb.move(ctx, self.path("group1/group4"), self.path("group1/group3"))

        tiledb.move(ctx, self.path("group1/group4"),
                    self.path("group1/group3"),
                    force=True)

        self.assertTrue(self.is_group(ctx, self.path("group1/group3")))
        self.assertFalse(self.is_group(ctx, self.path("group1/group4")))


class DimensionTest(unittest.TestCase):

    def test_minimal_dimension(self):
        ctx = t.Ctx()
        dim = t.Dim(ctx, domain=(0, 4))
        self.assertEqual(dim.name, "", "dimension name is empty")
        self.assertEqual(dim.shape, (5,))
        self.assertEqual(dim.tile, None, "tiled extent is None (void)")

    def test_dimension(self):
        ctx = t.Ctx()
        dim = t.Dim(ctx, "d1", domain=(0, 3), tile=2)
        self.assertEqual(dim.name, "d1")
        self.assertEqual(dim.shape, (4,))
        self.assertEqual(dim.tile, 2)


class DomainTest(unittest.TestCase):

    def test_domain(self):
        ctx = t.Ctx()
        dom = t.Domain(
            ctx,
            t.Dim(ctx, "d1", (1, 4), 2, dtype='u8'),
            t.Dim(ctx, "d2", (1, 4), 2, dtype='u8'))
        dom.dump()
        self.assertEqual(dom.ndim, 2)
        self.assertEqual(dom.rank, dom.ndim)
        self.assertEqual(dom.dtype, np.dtype("uint64"))
        self.assertEqual(dom.shape, (4, 4))

        # check that we can iterate over the dimensions
        dim_names = [dim.name for dim in dom]
        self.assertEqual(["d1", "d2"], dim_names)

    def test_domain_dims_not_same_type(self):
        ctx = t.Ctx()
        with self.assertRaises(TypeError):
            t.Domain(
                    ctx,
                    t.Dim(ctx, "d1", (1, 4), 2, dtype=int),
                    t.Dim(ctx, "d2", (1, 4), 2, dtype=float))


class AttributeTest(unittest.TestCase):

    def test_minimal_attribute(self):
        ctx = t.Ctx()
        attr = t.Attr(ctx)
        self.assertTrue(attr.isanon)
        self.assertEqual(attr.name, u"")
        self.assertEqual(attr.dtype, np.float_)
        self.assertEqual(attr.compressor, (None, -1))

    def test_attribute(self):
        ctx = t.Ctx()
        attr = t.Attr(ctx, "foo")
        attr.dump()
        self.assertEqual(attr.name, "foo")
        self.assertEqual(attr.dtype, np.float64,
                         "default attribute type is float64")
        compressor, level = attr.compressor
        self.assertEqual(compressor, None, "default to no compression")
        self.assertEqual(level, -1, "default compression level when none is specified")

    def test_full_attribute(self):
        ctx = t.Ctx()
        attr = t.Attr(ctx, "foo", dtype=np.int64, compressor=("zstd", 10))
        attr.dump()
        self.assertEqual(attr.name, "foo")
        self.assertEqual(attr.dtype, np.int64)
        compressor, level = attr.compressor
        self.assertEqual(compressor, "zstd")
        self.assertEqual(level, 10)

    def test_ncell_attribute(self):
        ctx = t.Ctx()
        dtype = np.dtype([("", np.int32), ("", np.int32)])
        attr = t.Attr(ctx, "foo", dtype=dtype)

        self.assertEqual(attr.dtype, dtype)
        self.assertEqual(attr.ncells, 2)

        # dtype subarrays not supported
        with self.assertRaises(TypeError):
            t.Attr(ctx, "foo", dtype=np.dtype((np.int32, 2)))

        # mixed type record arrays not supported
        with self.assertRaises(TypeError):
            t.Attr(ctx, "foo", dtype=np.dtype([("", np.float32), ("", np.int32)]))

    def test_ncell_bytes_attribute(self):
        ctx = t.Ctx()
        dtype = np.dtype((np.bytes_, 10))
        attr = t.Attr(ctx, "foo", dtype=dtype)

        self.assertEqual(attr.dtype, dtype)
        self.assertEqual(attr.ncells, 10)

    def test_vararg_attribute(self):
        ctx = t.Ctx()
        attr = t.Attr(ctx, "foo", dtype=np.bytes_)
        self.assertEqual(attr.dtype, np.dtype(np.bytes_))
        self.assertTrue(attr.isvar)

    def test_unique_attributes(self):
        ctx = t.Ctx()
        dom = t.Domain(
            ctx,
            t.Dim(ctx, "d1", (1, 4), 2, dtype='u8'),
            t.Dim(ctx, "d2", (1, 4), 2, dtype='u8'))

        attr1 = t.Attr(ctx, "foo", dtype=float)
        attr2 = t.Attr(ctx, "foo", dtype=int)

        with self.assertRaises(t.TileDBError):
            t.ArraySchema(ctx, "foobar", domain=dom, attrs=(attr1, attr2))


class DenseArrayTest(DiskTestCase):

    def test_dense_array_not_sparse(self):
        ctx = t.Ctx()
        dom = t.Domain(
            ctx,
            t.Dim(ctx, domain=(1, 8), tile=2),
            t.Dim(ctx, domain=(1, 8), tile=2))
        att = t.Attr(ctx, "val", dtype='f8')
        arr = t.DenseArray(ctx, self.path("foo"), domain=dom, attrs=(att,))
        arr.dump()
        self.assertTrue(arr.name == self.path("foo"))
        self.assertFalse(arr.sparse)

    def test_dense_array_fp_domain_error(self):
        ctx = t.Ctx()
        dom = t.Domain(ctx,
            t.Dim(ctx, domain=(1, 8), tile=2, dtype=np.float64))
        att = t.Attr(ctx, "val", dtype=np.float64)

        with self.assertRaises(t.TileDBError):
            t.DenseArray(ctx, self.path("foo"), domain=dom, attrs=(att,))


    def test_array_1d(self):
        A = np.arange(1050)

        ctx = t.Ctx()
        dom = t.Domain(ctx, t.Dim(ctx, domain=(0, 1049), tile=100, dtype=np.int64))
        att = t.Attr(ctx, dtype=A.dtype)
        T = t.DenseArray(ctx, self.path("foo"), domain=dom, attrs=(att,))

        self.assertEqual(len(A), len(T))
        self.assertEqual(A.ndim, T.ndim)
        self.assertEqual(A.shape, T.shape)

        self.assertEqual(1, T.nattr)
        self.assertEqual(A.dtype, T.attr(0).dtype)

        # check empty array
        B = T[:]

        self.assertEqual(A.shape, B.shape)
        self.assertEqual(A.dtype, B.dtype)
        self.assertIsNone(T.nonempty_domain())

        # check set array
        T[:] = A

        self.assertEqual(((0, 1049),), T.nonempty_domain())

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
        
        # check partial assignment
        B = np.arange(1e5, 2e5).astype(A.dtype)
        T[190:310] = B[190:310]

        assert_array_equal(A[:190], T[:190])
        assert_array_equal(B[190:310], T[190:310])
        assert_array_equal(A[310:], T[310:])

    def test_array_1d_set_scalar(self):
        A = np.zeros(50)

        ctx = t.Ctx()
        dom = t.Domain(ctx, t.Dim(ctx, domain=(0, 49), tile=10))
        att = t.Attr(ctx, dtype=A.dtype)
        T = t.DenseArray(ctx, self.path("foo"), dom, (att,))

        T[:] = A
        for value in (-1, 0, 1, 10):
            A[5:25] = value
            T[5:25] = value
            assert_array_equal(A, T[:])
            A[:] = value
            T[:] = value
            assert_array_equal(A, T[:])

    def test_array_id_point_queries(self):
        #TODO: handle queries like T[[2, 5, 10]] = ?
        pass

    def test_array_2d(self):
        A = np.arange(10000).reshape((1000, 10))

        ctx = t.Ctx()
        dom = t.Domain(ctx,
                       t.Dim(ctx, domain=(0, 999), tile=100),
                       t.Dim(ctx, domain=(0, 9), tile=2))
        att = t.Attr(ctx, dtype=A.dtype)
        T = t.DenseArray(ctx, self.path("foo"), dom, (att,))

        self.assertEqual(len(A), len(T))
        self.assertEqual(A.ndim, T.ndim)
        self.assertEqual(A.shape, T.shape)

        self.assertEqual(1, T.nattr)
        self.assertEqual(A.dtype, T.attr(0).dtype)

        # check that the non-empty domain is None
        self.assertIsNone(T.nonempty_domain())
        # Set data
        T[:] = A
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

        # check partial assignment
        B = np.arange(10000, 20000).reshape((1000, 10))
        T[190:310, 3:7] = B[190:310, 3:7]
        assert_array_equal(A[:190], T[:190])
        assert_array_equal(A[:, :3], T[:, :3])
        assert_array_equal(B[190:310, 3:7], T[190:310, 3:7])
        assert_array_equal(A[310:], T[310:])
        assert_array_equal(A[:, 7:], T[:, 7:])

    def test_ncell_attributes(self):
        ctx = t.Ctx()
        dom = t.Domain(ctx, t.Dim(ctx, domain=(0, 9), tile=10, dtype=int))
        attr = t.Attr(ctx, dtype=[("", np.int32), ("", np.int32)])
        T = t.DenseArray(ctx, self.path("foo"), domain=dom, attrs=(attr,))

        A = np.ones((10,), dtype=[("", np.int32), ("", np.int32)])
        self.assertEqual(A.dtype, attr.dtype)

        T[:] = A
        assert_array_equal(A, T[:])
        assert_array_equal(A[:5], T[:5])

    def test_multiple_attributes(self):
        ctx = t.Ctx()
        dom = t.Domain(ctx,
                       t.Dim(ctx, domain=(0, 7), tile=8, dtype=int))
        attr_int = t.Attr(ctx, "ints", dtype=int)
        attr_float = t.Attr(ctx, "floats", dtype=float)
        T = t.DenseArray(ctx,
                         self.path("foo"),
                         domain=dom,
                         attrs=(attr_int, attr_float))

        V_ints = np.array([0, 1, 2, 3, 4, 6, 7, 5])
        V_floats = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 5.0])

        V = {"ints": V_ints, "floats": V_floats}
        T[:] = V

        R = T[:]
        assert_array_equal(V["ints"], R["ints"])
        assert_array_equal(V["floats"], R["floats"])

        # check error ncells length
        V["ints"] = V["ints"][1:2].copy()
        with self.assertRaises(t.TileDBError):
            T[:] = V

        # check error attribute does not exist
        V["foo"] = V["ints"].astype(np.int8)
        with self.assertRaises(t.TileDBError):
            T[:] = V


class SparseArray(DiskTestCase):

    def test_sparse_array_not_dense(self):
        ctx = t.Ctx()
        dom = t.Domain(
            ctx,
            t.Dim(ctx, domain=(1, 8), tile=2),
            t.Dim(ctx, domain=(1, 8), tile=2))
        att = t.Attr(ctx, "val", dtype='f8')
        T = t.SparseArray(ctx, self.path("foo"), domain=dom, attrs=(att,))
        T.dump()
        self.assertTrue(T.name == self.path("foo"))
        self.assertTrue(T.sparse)

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
        schema = tiledb.SparseArray(ctx, self.path("sparse_array_schema"),
                                    domain=domain, attrs=(a1, a2),
                                    capacity=10,
                                    cell_order='col-major',
                                    tile_order='row-major',
                                    coords_compressor=('zstd', 4),
                                    offsets_compressor=('blosc-lz', 5))
        self.assertEqual(schema.capacity, 10)
        self.assertEqual(schema.cell_order, "col-major")
        self.assertEqual(schema.tile_order, "row-major")
        self.assertEqual(schema.coords_compressor, ('zstd', 4))
        self.assertEqual(schema.offsets_compressor, ('blosc-lz', 5))

    @unittest.expectedFailure
    def test_simple_1d_sparse_vector(self):
        ctx = t.Ctx()
        dom = t.Domain(ctx, t.Dim(ctx, domain=(0, 3), tile=4, dtype=int))
        att = t.Attr(ctx, dtype=int)
        T = t.SparseArray(ctx, self.path("foo"), domain=dom, attrs=(att,))

        values = np.array([3, 4])
        T[[1, 2]] = values

        assert_array_equal(T[[1, 2]], values)

    @unittest.expectedFailure
    def test_simple_2d_sparse_vector(self):
        ctx = t.Ctx()
        dom = t.Domain(ctx, t.Dim(ctx, domain=(0, 3), tile=4, dtype=int),
                            t.Dim(ctx, domain=(0, 3), tile=4, dtype=int))
        attr = t.Attr(ctx, dtype=float)
        T = t.SparseArray(ctx, self.path("foo"), domain=dom, attrs=(attr,))

        values = np.array([3, 4], dtype=float)
        T[[1, 2], [1, 2]] = values

        assert_array_equal(T[[1, 2], [1, 2]], values)

    @unittest.expectedFailure
    def test_simple3d_sparse_vector(self):
        ctx = t.Ctx()
        dom = t.Domain(ctx, t.Dim(ctx, "x", domain=(0, 3), tile=4, dtype=int),
                            t.Dim(ctx, "y", domain=(0, 3), tile=4, dtype=int),
                            t.Dim(ctx, "z", domain=(0, 3), tile=4, dtype=int))
        attr = t.Attr(ctx, dtype=float)
        T = t.SparseArray(ctx, self.path("foo"), domain=dom, attrs=(attr,))

        values = np.array([3, 4], dtype=float)
        T[[1, 2], [1, 2], [1, 2]] = values

        assert_array_equal(T[[1, 2], [1, 2], [1, 2]], values)

    @unittest.expectedFailure
    def test_sparse_ordered_fp_domain(self):
        ctx = t.Ctx()
        dom = t.Domain(ctx, t.Dim(ctx, "x", domain=(0.0, 10.0), tile=2.0, dtype=float))
        attr = t.Attr(ctx, dtype=float)
        T = t.SparseArray(ctx, self.path("foo"), domain=dom, attrs=(attr,))

        values = np.array([3.3, 2.7])
        T[[2.5, 4.2]] = values

        assert_array_equal(T[[2.5, 4.2]], values)

    @unittest.expectedFailure
    def test_sparse_unordered_fp_domain(self):
        ctx = t.Ctx()
        dom = t.Domain(ctx, t.Dim(ctx, "x", domain=(0.0, 10.0), tile=2.0, dtype=float))
        attr = t.Attr(ctx, dtype=float)
        T = t.SparseArray(ctx, self.path("foo"), domain=dom, attrs=(attr,))

        values = np.array([3.3, 2.7])
        T[[4.2, 2.5]] = values

        assert_array_equal(T[[2.5, 4.2]], values[::-1])

    @unittest.expectedFailure
    def test_multiple_attributes(self):
        ctx = t.Ctx()
        dom = t.Domain(ctx,
                t.Dim(ctx, domain=(1, 10), tile=10, dtype=int),
                t.Dim(ctx, domain=(1, 10), tile=10, dtype=int))
        attr_int = t.Attr(ctx, "ints", dtype=int)
        attr_float = t.Attr(ctx, "floats", dtype="float")
        T = t.SparseArray(ctx, self.path("foo"),
                          domain=dom,
                          attrs=(attr_int, attr_float,))

        I = np.array([1, 1, 1, 2, 3, 3, 3, 4])
        J = np.array([1, 2, 4, 3, 1, 6, 7, 5])

        V_ints = np.array([0, 1, 2, 3, 4, 6, 7, 5])
        V_floats = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0, 7.0, 5.0])

        V = {"ints": V_ints, "floats": V_floats}
        T[I, J] = V

        R = T[I, J]
        assert_array_equal(V["ints"], R["ints"])
        assert_array_equal(V["floats"], R["floats"])

        # check error attribute does not exist
        # TODO: should this be an attribute error?
        V["foo"] = V["ints"].astype(np.int8)
        with self.assertRaises(t.TileDBError):
            T[I, J] = V

        # check error ncells length
        V["ints"] = V["ints"][1:2].copy()
        with self.assertRaises(AttributeError):
            T[I, J] = V

    def test_subarray(self):
        ctx = t.Ctx()
        dom = t.Domain(ctx, t.Dim(ctx, "x", domain=(1, 10000), tile=100, dtype=int))
        att = t.Attr(ctx, "", dtype=float)

        T = t.SparseArray(ctx, self.path("foo"), domain=dom, attrs=(att,))

        self.assertIsNone(T.nonempty_domain())

        T[[50, 60, 100]] = [1.0, 2.0, 3.0]
        self.assertEqual(((50, 100),), T.nonempty_domain())

        # retrieve just valid coordinates in subarray T[40:60]
        assert_array_equal(T[40:61]["coords"]["x"], [50, 60])


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

        ctx = t.Ctx()
        dom = t.Domain(ctx, t.Dim(ctx, domain=(0, 1049), tile=100))
        att = t.Attr(ctx, dtype=int)

        T = t.DenseArray(ctx, self.path("foo"), domain=dom, attrs=(att,))
        T[:] = A

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

        ctx = t.Ctx()
        dom = t.Domain(ctx,
                       t.Dim(ctx, domain=(0, 999), tile=100),
                       t.Dim(ctx, domain=(0, 9), tile=2))
        att = t.Attr(ctx, dtype=A.dtype)
        T = t.DenseArray(ctx, self.path("foo"), dom, (att,))
        T[:] = A

        for idx in self.good_index_1d:
            self._test_index(A, T, idx)

        for idx in self.bad_index_2d:
            with self.assertRaises(IndexError):
                T[idx]


class RWTest(DiskTestCase):

    def test_read_write(self):
        ctx = t.Ctx()

        dom = t.Domain(ctx, t.Dim(ctx, domain=(0, 2), tile=3))
        att = t.Attr(ctx, dtype='i8')
        arr = t.DenseArray(ctx, self.path("foo"), domain=dom, attrs=[att])

        A = np.array([1, 2, 3])
        arr.write_direct(A)
        arr.dump()
        assert_array_equal(arr.read_direct(), A)
        self.assertEqual(arr.ndim, A.ndim)


class NumpyToArray(DiskTestCase):

    def test_to_array0d(self):
        # Cannot create 0-dim arrays in TileDB
        ctx = t.Ctx()
        A = np.array(1)
        with self.assertRaises(t.TileDBError):
            t.DenseArray.from_numpy(ctx, self.path("foo"), A)

    def test_to_array1d(self):
        ctx = t.Ctx()
        A = np.array([1.0, 2.0, 3.0])
        arr = t.DenseArray.from_numpy(ctx, self.path("foo"), A)
        assert_array_equal(A, arr[:])

    def test_to_array2d(self):
        ctx = t.Ctx()
        A = np.ones((100, 100), dtype='i8')
        arr = t.DenseArray.from_numpy(ctx, self.path("foo"), A)
        assert_array_equal(A, arr[:])

    def test_to_array3d(self):
        ctx = t.Ctx()
        A = np.ones((1, 1, 1), dtype='i1')
        arr = t.DenseArray.from_numpy(ctx, self.path("foo"), A)
        assert_array_equal(A, arr[:])

    def test_array_interface(self):
        # Tests that __array__ interface works
        ctx = t.Ctx()
        A1 = np.arange(1, 10)
        arr1 = t.DenseArray.from_numpy(ctx, self.path("arr1"), A1)
        A2 = np.array(arr1)
        assert_array_equal(A1, A2)

        # Test that __array__ interface throws an error when number of attributes > 1
        dom = t.Domain(ctx, t.Dim(ctx, domain=(0, 2), tile=3))
        foo = t.Attr(ctx, "foo", dtype='i8')
        bar = t.Attr(ctx, "bar", dtype='i8')
        arr2 = t.DenseArray(ctx, self.path("arr2"), domain=dom, attrs=(foo, bar))
        with self.assertRaises(ValueError):
            np.array(arr2)

    def test_array_getindex(self):
        # Tests that __getindex__ interface works
        ctx = t.Ctx()
        A = np.arange(1, 10)
        arr = t.DenseArray.from_numpy(ctx, self.path("foo"), A)
        assert_array_equal(A[5:10], arr[5:10])


class KVArray(DiskTestCase):

    def test_attr(self):
        ctx = t.Ctx()
        a1 = t.Attr(ctx, "", dtype=bytes)
        self.assertTrue(a1.isanon)

    def test_kv_write_read(self):
        # create a kv database
        ctx = t.Ctx()
        a1 = t.Attr(ctx, "value", dtype=bytes)
        kv = t.KV(ctx, self.path("foo"), attrs=(a1,))
        a1.dump()
        kv['foo'] = 'bar'
        kv.dump()
        self.assertEqual(kv["foo"], 'bar')

    def test_kv_write_load_read(self):
        # create a kv database
        ctx = t.Ctx()
        a1 = t.Attr(ctx, "value", dtype=bytes)
        kv = t.KV(ctx, self.path("foo"), attrs=(a1,))

        kv['foo'] = 'bar'
        del kv

        # try to load it
        kv = t.KV.load(ctx, self.path("foo"))
        self.assertEqual(kv["foo"], 'bar')

    def test_kv_update(self):
        # create a kv database
        ctx = t.Ctx()
        a1 = t.Attr(ctx, "val", dtype=bytes)
        kv = t.KV(ctx, self.path("foo"), attrs=(a1,))

        kv['foo'] = 'bar'
        del kv

        kv = t.KV.load(ctx, self.path("foo"))
        kv['foo'] = 'baz'
        del kv

        kv = t.KV.load(ctx, self.path("foo"))
        self.assertEqual(kv['foo'], 'baz')

    def test_key_not_found(self):
        # create a kv database
        ctx = t.Ctx()
        a1 = t.Attr(ctx, "value", dtype=bytes)
        kv = t.KV(ctx, self.path("foo"), attrs=(a1,))

        self.assertRaises(KeyError, kv.__getitem__, "not here")

    def test_kv_contains(self):
        # create a kv database
        ctx = t.Ctx()
        a1 = t.Attr(ctx, "value", dtype=bytes)
        kv = t.KV(ctx, self.path("foo"), attrs=(a1,))
        self.assertFalse("foo" in kv)
        kv['foo'] = 'bar'
        self.assertTrue("foo" in kv)

    def test_kv_dict(self):
        # create a kv database
        ctx = t.Ctx()
        a1 = t.Attr(ctx, "value", dtype=bytes)
        kv = t.KV(ctx, self.path("foo"), attrs=(a1,))
        kv['foo'] = 'bar'
        kv['baz'] = 'foo'
        self.assertEqual(kv.dict(), {'foo': 'bar', 'baz': 'foo'})
        self.assertEqual(dict(kv), {'foo': 'bar', 'baz': 'foo'})

    def test_multiattribute(self):
        ctx = t.Ctx()
        a1 = t.Attr(ctx, "ints", dtype=int)
        a2 = t.Attr(ctx, "floats", dtype=float)
        kv = t.KV(ctx, self.path("foo"), attrs=(a1, a2))

        # known failure
        #kv['foo'] = {"ints": 1, "floats": 2.0}
        #self.assertEqual(kv["foo"]["ints"], 1)
        #self.assertEqual(kv["foo"]["floats"], 2.0)


class VFS(DiskTestCase):

    def test_supports(self):
        ctx = t.Ctx()
        vfs = t.VFS(ctx)

        self.assertTrue(vfs.supports("file"))
        self.assertIsInstance(vfs.supports("s3"), bool)
        self.assertIsInstance(vfs.supports("hdfs"), bool)

        with self.assertRaises(ValueError):
            vfs.supports("invalid")

    def test_dir(self):
        ctx = t.Ctx()
        vfs = t.VFS(ctx)

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
        with self.assertRaises(t.TileDBError):
            vfs.create_dir(dir)

        vfs.create_dir(self.path("foo"))
        vfs.create_dir(self.path("foo/bar"))
        self.assertTrue(vfs.is_dir(dir))

    def test_file(self):
        ctx = t.Ctx()
        vfs = t.VFS(ctx)

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
        with self.assertRaises(t.TileDBError):
            vfs.touch(file)

    def test_move(self):
        ctx = t.Ctx()
        vfs = t.VFS(ctx)

        vfs.create_dir(self.path("foo"))
        vfs.create_dir(self.path("bar"))
        vfs.touch(self.path("bar/baz"))

        self.assertTrue(vfs.is_file(self.path("bar/baz")))

        vfs.move(self.path("bar/baz"), self.path("foo/baz"))

        self.assertFalse(vfs.is_file(self.path("bar/baz")))
        self.assertTrue(vfs.is_file(self.path("foo/baz")))

        # moving to invalid dir should raise an error
        with self.assertRaises(t.TileDBError):
            vfs.move(self.path("foo/baz"), self.path("do_not_exist/baz"))

    def test_write_read(self):
        ctx = t.Ctx()
        vfs = t.VFS(ctx)

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
        with self.assertRaises(t.TileDBError):
            vfs.open(self.path("do_not_exist"), "r")

    def test_io(self):
        ctx = t.Ctx()
        vfs = t.VFS(ctx)

        buffer = b"0123456789"
        io = t.FileIO(vfs, self.path("foo"), mode="w")
        io.write(buffer)
        io.flush()
        self.assertEqual(io.tell(), len(buffer))

        io = t.FileIO(vfs, self.path("foo"), mode="r")
        with self.assertRaises(IOError):
            io.write(b"foo")

        self.assertEqual(vfs.file_size(self.path("foo")), len(buffer))

        io = t.FileIO(vfs, self.path("foo"), mode='r')
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

