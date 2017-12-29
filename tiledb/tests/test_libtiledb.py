from __future__ import absolute_import
from tiledb.tests.common import DiskTestCase
from unittest import TestCase

import tiledb
from tiledb import libtiledb as t

import numpy as np
from numpy.testing import assert_array_equal


def is_group(ctx, path):
   obj = tiledb.libtiledb.object_type(ctx, path)
   return obj == 2


class VersionTest(TestCase):

    def test_version(self):
        v = tiledb.libtiledb_version()
        self.assertIsInstance(v, tuple)
        self.assertTrue(len(v) == 3)
        self.assertTrue(v[0] >= 1, "TileDB major version must be >= 1")


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


class GroupTest(GroupTestCase):

    def test_is_group(self):
        ctx = tiledb.Ctx()
        self.assertTrue(is_group(ctx, self.group1))
        self.assertTrue(is_group(ctx, self.group2))
        self.assertTrue(is_group(ctx, self.group3))
        self.assertTrue(is_group(ctx, self.group4))

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

    def test_delete_group(self):
        ctx = tiledb.Ctx()

        tiledb.delete(ctx, self.group3)

        self.assertFalse(is_group(ctx, self.group3))
        self.assertFalse(is_group(ctx, self.group4))

    def test_move_group(self):
        ctx = tiledb.Ctx()

        tiledb.move(ctx, self.group4, self.path("group1/group4"))

        self.assertTrue(is_group(ctx, self.path("group1/group4")))
        self.assertFalse(is_group(ctx, self.group4))

        with self.assertRaises(tiledb.TileDBError):
            tiledb.move(ctx, self.path("group1/group4"), self.path("group1/group3"))

        tiledb.move(ctx, self.path("group1/group4"),
                    self.path("group1/group3"),
                    force=True)

        self.assertTrue(is_group(ctx, self.path("group1/group3")))
        self.assertFalse(is_group(ctx, self.path("group1/group4")))


class DimensionTest(TestCase):

    def test_minimal_dimension(self):
        ctx = t.Ctx()
        dim = t.Dim(ctx, domain=(0, 4))
        self.assertEqual(dim.name, "", "dimension name is empty")
        self.assertEqual(dim.shape, (5,))
        self.assertEqual(dim.tile, 5, "tile extent should span the whole domain")

    def test_dimension(self):
        ctx = t.Ctx()
        dim = t.Dim(ctx, "d1", domain=(0, 3), tile=2)
        self.assertEqual(dim.name, "d1")
        self.assertEqual(dim.shape, (4,))
        self.assertEqual(dim.tile, 2)


class DomainTest(TestCase):

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

    def test_domain_dims_not_same_type(self):
        ctx = t.Ctx()
        with self.assertRaises(AttributeError):
            t.Domain(
                    ctx,
                    t.Dim(ctx, "d1", (1, 4), 2, dtype=int),
                    t.Dim(ctx, "d2", (1, 4), 2, dtype=float))


class AttributeTest(TestCase):

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
        attr = t.Attr(ctx, "foo", dtype=np.int64, compressor="zstd", level=10)
        attr.dump()
        self.assertEqual(attr.name, "foo")
        self.assertEqual(attr.dtype, np.int64)
        compressor, level = attr.compressor
        self.assertEqual(compressor, "zstd")
        self.assertEqual(level, 10)


class ArrayTest(DiskTestCase):

    def test_array_not_sparse(self):
        ctx = t.Ctx()
        dom = t.Domain(
            ctx,
            t.Dim(ctx, domain=(1, 8), tile=2),
            t.Dim(ctx, domain=(1, 8), tile=2))
        att = t.Attr(ctx, "val", dtype='f8')
        arr = t.Array.create(ctx, self.path("foo"), domain=dom, attrs=[att])
        arr.dump()
        self.assertTrue(arr.name == self.path("foo"))
        self.assertFalse(arr.sparse)

    def test_dense_array_fp_domain_error(self):
        ctx = t.Ctx()
        dom = t.Domain(ctx,
            t.Dim(ctx, domain=(1, 8), tile=2, dtype=np.float64))
        att = t.Attr(ctx, "val", dtype=np.float64)

        with self.assertRaises(t.TileDBError):
            t.Array.create(ctx, self.path("foo"), domain=dom, attrs=(att,))

    def test_array_1d(self):
        A = np.arange(1050)

        ctx = t.Ctx()
        dom = t.Domain(ctx, t.Dim(ctx, domain=(0, 1049), tile=100, dtype=np.int64))
        att = t.Attr(ctx, dtype=A.dtype)
        T = t.Array.create(ctx, self.path("foo"), domain=dom, attrs=(att,))

        self.assertEqual(len(A), len(T))
        self.assertEqual(A.ndim, T.ndim)
        self.assertEqual(A.shape, T.shape)

        self.assertEqual(1, T.nattr)
        self.assertEqual(A.dtype, T.attr(0).dtype)

        # check empty array
        B = T[:]

        self.assertEqual(A.shape, B.shape)
        self.assertEqual(A.dtype, B.dtype)

        # check set array
        T[:] = A

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
        T = t.Array.create(ctx, self.path("foo"), dom, (att,))

        T[:] = A
        for value in (-1, 0, 1, 10):
            A[5:25] = value
            T[5:25] = value
            assert_array_equal(A, T[:])
            A[:] = value
            T[:] = value
            assert_array_equal(A, T[:])


class RWTest(DiskTestCase):

    def test_read_write(self):
        ctx = t.Ctx()

        dom = t.Domain(ctx, t.Dim(ctx, domain=(0, 2), tile=3))
        att = t.Attr(ctx, dtype='i8')
        arr = t.Array.create(ctx, self.path("foo"), domain=dom, attrs=[att])

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
            t.Array.from_numpy(ctx, self.path("foo"), A)

    def test_to_array1d(self):
        ctx = t.Ctx()
        A = np.array([1.0, 2.0, 3.0])
        arr = t.Array.from_numpy(ctx, self.path("foo"), A)
        assert_array_equal(arr.read_direct(), A)

    def test_to_array2d(self):
        ctx = t.Ctx()
        A = np.ones((100, 100), dtype='i8')
        arr = t.Array.from_numpy(ctx, self.path("foo"), A)
        assert_array_equal(arr.read_direct(), A)

    def test_to_array3d(self):
        ctx = t.Ctx()
        A = np.ones((1, 1, 1), dtype='i1')
        arr = t.Array.from_numpy(ctx, self.path("foo"), A)
        assert_array_equal(arr.read_direct(), A)

    def test_array_interface(self):
        # Tests that __array__ interface works
        ctx = t.Ctx()
        A1 = np.arange(1, 10)
        arr = t.Array.from_numpy(ctx, self.path("foo"), A1)
        A2 = np.array(arr)
        assert_array_equal(A1, A2)

    def test_array_getindex(self):
        # Tests that __getindex__ interface works
        ctx = t.Ctx()
        A1 = np.arange(1, 10)
        arr = t.Array.from_numpy(ctx, self.path("foo"), A1)
        A2 = arr[5:10]
        assert_array_equal(A1[5:10], A2)


class AssocArray(DiskTestCase):

    def test_attr(self):
        ctx = t.Ctx()
        a1 = t.Attr(ctx, "", dtype=bytes)
        self.assertTrue(a1.isanon)

    def test_kv_write_read(self):
        # create a kv database
        ctx = t.Ctx()
        a1 = t.Attr(ctx, "value", dtype=bytes)
        kv = t.Assoc(ctx, self.path("foo"), a1)
        a1.dump()
        kv["foo"] = b'bar'
        kv.dump()
        self.assertEqual(kv["foo"], b'bar')

    def test_kv_write_load_read(self):
        # create a kv database
        ctx = t.Ctx()
        a1 = t.Attr(ctx, "value", dtype=bytes)
        kv = t.Assoc(ctx, self.path("foo"), a1)

        kv["foo"] = b'bar'
        del kv

        # try to load it
        kv = t.Assoc.load(ctx, self.path("foo"))
        self.assertEqual(kv["foo"], b'bar')

    def test_kv_update(self):
        # create a kv database
        ctx = t.Ctx()
        a1 = t.Attr(ctx, "val", dtype=bytes)
        kv = t.Assoc(ctx, self.path("foo"), a1)

        kv["foo"] = b'bar'
        del kv

        kv = t.Assoc.load(ctx, self.path("foo"))
        kv["foo"] = b'baz'
        del kv

        kv = t.Assoc.load(ctx, self.path("foo"))
        self.assertEqual(kv["foo"], b'baz')

    def test_key_not_found(self):
        # create a kv database
        ctx = t.Ctx()
        a1 = t.Attr(ctx, "value", dtype=bytes)
        kv = t.Assoc(ctx, self.path("foo"), a1)

        self.assertRaises(KeyError, kv.__getitem__, "not here")

    def test_kv_contains(self):
        # create a kv database
        ctx = t.Ctx()
        a1 = t.Attr(ctx, "value", dtype=bytes)
        kv = t.Assoc(ctx, self.path("foo"), a1)
        self.assertFalse("foo" in kv)
        kv["foo"] = b'bar'
        self.assertTrue("foo" in kv)

    def test_ky_update(self):
        pass

    """
    def test_kv_performance(self):
        import random
        import time

        # create a kv database
        ctx = t.Ctx()
        a1 = t.Attr(ctx, "value", dtype=bytes)
        kv = t.Assoc(ctx, self.path("foo"), a1)
        print("tiledb starting")
        NRECORDS = 10000
        int_values = [random.randint(0, 10000000) for _ in range(NRECORDS)]
        keys = list(map(str, int_values))
        values = [str(k).encode('ascii') for k in keys]
        start = time.time()
        for i in range(NRECORDS):
            kv[keys[i]] = values[i]
        end = time.time()
        print("inserting {} keys took {} seconds".format(NRECORDS,  end - start))

        print("consolidating")
        start = time.time()

        t.array_consolidate(ctx, self.path("foo"))
        end = time.time()
        print("consolidating took {} seconds".format(end - start))

        print("tiledb read starting")
        start = time.time()
        for i in range(NRECORDS):
            key = keys[i]
            val = values[i]
            if kv[key] != val:
                print("key: {}; value: {}, kv[key]: {}".format(key, val, kv[key]))
        end = time.time()
        print("reading {} keys took {} seconds".format(NRECORDS,  end - start))
    """
