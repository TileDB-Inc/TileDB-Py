from __future__ import absolute_import
from tiledb.tests.common import DiskTestCase
from unittest import TestCase

import tiledb
from tiledb import libtiledb as t

def is_group(ctx, path):
   obj = tiledb.libtiledb.object_type(ctx, path)
   return obj == 1


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

    def test_domain(self):
        ctx = t.Ctx()
        dom = t.Domain(
            ctx,
            t.Dim("d1", (1, 4), 2),
            t.Dim("d2", (1, 4), 2),
            dtype='u8')
        dom.dump()


class AttributeTest(TestCase):

    def test_attribute(self):
       ctx = t.Ctx()
       attr = t.Attr(ctx, "foo")
       attr.dump()


class ArrayTest(DiskTestCase):

    def test_array(self):
        ctx = t.Ctx()
        dom = t.Domain(
            ctx,
            t.Dim("d1", (1, 8), 2),
            t.Dim("d2", (1, 8), 2),
            dtype='u8')
        att = t.Attr(ctx, "val", dtype='f8')
        arr = t.Array(ctx, self.path("foo"), domain=dom, attrs=[att])
        arr.dump()





