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
            t.Dim(ctx, "d1", (1, 4), 2),
            t.Dim(ctx, "d2", (1, 4), 2),
            dtype='u8')
        dom.dump()
        self.assertEqual(dom.ndim, 2)
        self.assertEqual(dom.rank, dom.ndim)
        self.assertEqual(dom.dtype, np.dtype("uint64"))
        self.assertEqual(dom.shape, (4, 4))


class AttributeTest(TestCase):

    def test_minimal_attribute(self):
        ctx = t.Ctx()
        attr = t.Attr(ctx, "foo")
        attr.dump()
        self.assertEqual(attr.name, "foo")
        self.assertEqual(attr.dtype, np.float64,
                         "default attribute type is float64")
        compressor, level = attr.compressor
        self.assertEqual(compressor, None, "default to no compression")
        self.assertEqual(level, -1, "default compression level when none is specified")

    def test_attribute(self):
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
            t.Dim(ctx, domain=(1, 8), tile=2),
            dtype='u8')
        att = t.Attr(ctx, "val", dtype='f8')
        arr = t.Array.create(ctx, self.path("foo"), domain=dom, attrs=[att])
        arr.dump()
        self.assertTrue(arr.name == self.path("foo"))
        self.assertFalse(arr.sparse)


class RWTest(DiskTestCase):

    def test_read_write(self):
        ctx = t.Ctx()

        dom = t.Domain(ctx, t.Dim(ctx, None, (0, 2), 3))
        att = t.Attr(ctx, "val", dtype='i8')
        arr = t.Array.create(ctx, self.path("foo"), domain=dom, attrs=[att])

        A = np.array([1,2,3])
        arr.write_direct("val", A)
        arr.dump()
        assert_array_equal(arr.read_direct("val"), A)


class NumpyToArray(DiskTestCase):

    def test_to_array0d(self):
        #TODO
        pass

    def test_to_array1d(self):
        ctx = t.Ctx()
        A = np.array([1.0, 2.0, 3.0])
        arr = t.Array.from_numpy(ctx, self.path("foo"), A)
        assert_array_equal(arr.read_direct(""), A)

    def test_to_array2d(self):
        ctx = t.Ctx()
        A = np.ones((100, 100), dtype='i8')
        arr = t.Array.from_numpy(ctx, self.path("foo"), A)
        assert_array_equal(arr.read_direct(""), A)

    def test_to_array3d(self):
        ctx = t.Ctx()
        A = np.ones((1, 1, 1), dtype='i1')
        arr = t.Array.from_numpy(ctx, self.path("foo"), A)
        assert_array_equal(arr.read_direct(""), A)

    def test_array_interface(self):
        # This tests that __array__ interface works
        ctx = t.Ctx()
        A1 = np.arange(1, 10)
        arr = t.Array.from_numpy(ctx, self.path("foo"), A1)
        A2 = np.array(arr)
        assert_array_equal(A1, A2)

    def test_array_getindex(self):
        # This tests that __getindex__ interface works
        ctx = t.Ctx()
        A1 = np.arange(1, 10)
        arr = t.Array.from_numpy(ctx, self.path("foo"), A1)
        A2 = arr[5:10]
        assert_array_equal(A1[5:10], A2)


class AssocArray(DiskTestCase):

    def test_kv_write_read(self):
        # create a kv database
        ctx = t.Ctx()
        a1 = t.Attr(ctx, "value", dtype=bytes)
        kv = t.Assoc(ctx, self.path("foo"), a1)
        a1.dump()
        kv["foo"] = b'bar'
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
        kv["foo"] = b'barbar'
        self.assertTrue("foo" in kv)
