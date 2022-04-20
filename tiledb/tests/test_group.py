import os
import numpy as np
import pytest

import tiledb
from tiledb.tests.common import DiskTestCase


class GroupTestCase(DiskTestCase):
    def setup_method(self):
        super().setup_method()

        self.group1 = self.path("group1")
        self.group2 = self.path("group1/group2")
        self.group3 = self.path("group1/group3")
        self.group4 = self.path("group1/group3/group4")

        tiledb.group_create(self.group1)
        tiledb.group_create(self.group2)
        tiledb.group_create(self.group3)
        tiledb.group_create(self.group4)

    def is_group(self, uri):
        return tiledb.object_type(uri) == "group"


class GroupTest(GroupTestCase):
    def test_is_group(self):
        self.assertTrue(self.is_group(self.group1))
        self.assertTrue(self.is_group(self.group2))
        self.assertTrue(self.is_group(self.group3))
        self.assertTrue(self.is_group(self.group4))

    def test_walk_group(self):
        if pytest.tiledb_vfs == "s3":
            pytest.skip("S3 does not have empty directories")

        groups = []

        def append_to_groups(path, obj):
            groups.append((os.path.normpath(path), obj))

        tiledb.walk(self.path(""), append_to_groups, order="preorder")

        groups.sort()

        self.assertTrue(groups[0][0].endswith(self.group1) and groups[0][1] == "group")
        self.assertTrue(groups[1][0].endswith(self.group2) and groups[1][1] == "group")
        self.assertTrue(groups[2][0].endswith(self.group3) and groups[2][1] == "group")
        self.assertTrue(groups[3][0].endswith(self.group4) and groups[3][1] == "group")

        groups = []

        tiledb.walk(self.path(""), append_to_groups, order="postorder")

        self.assertTrue(groups[0][0].endswith(self.group2) and groups[0][1] == "group")
        self.assertTrue(groups[1][0].endswith(self.group4) and groups[1][1] == "group")
        self.assertTrue(groups[2][0].endswith(self.group3) and groups[2][1] == "group")
        self.assertTrue(groups[3][0].endswith(self.group1) and groups[3][1] == "group")

    def test_remove_group(self):
        tiledb.remove(self.group3)

        self.assertFalse(self.is_group(self.group3))
        self.assertFalse(self.is_group(self.group4))

    def test_move_group(self):
        self.assertTrue(self.is_group(self.group2))
        tiledb.move(self.group2, self.group2 + "_moved")
        self.assertFalse(self.is_group(self.group2))
        self.assertTrue(self.is_group(self.group2 + "_moved"))

    @pytest.mark.parametrize(
        "int_data,flt_data,str_data",
        (
            (-1, -1.5, "asdf"),
            ([1, 2, 3], [1.5, 2.5, 3.5], b"asdf"),
            (np.array([1, 2, 3]), np.array([1.5, 2.5, 3.5]), np.array(["x"])),
        ),
    )
    def test_group_metadata(self, int_data, flt_data, str_data):
        def values_equal(lhs, rhs):
            if isinstance(lhs, np.ndarray):
                if not isinstance(rhs, np.ndarray):
                    return False
                return np.array_equal(lhs, rhs)
            elif isinstance(lhs, (list, tuple)):
                if not isinstance(rhs, (list, tuple)):
                    return False
                return tuple(lhs) == tuple(rhs)
            else:
                return lhs == rhs

        grp_path = self.path("test_group_metadata")
        tiledb.Group.create(grp_path)

        grp = tiledb.Group(grp_path, "w")
        grp.meta["int"] = int_data
        grp.meta["flt"] = flt_data
        grp.meta["str"] = str_data
        grp.close()

        grp.open("r")
        assert len(grp.meta) == 3
        assert "int" in grp.meta
        assert values_equal(grp.meta["int"], int_data)
        assert "flt" in grp.meta
        assert values_equal(grp.meta["flt"], flt_data)
        assert "str" in grp.meta
        assert values_equal(grp.meta["str"], str_data)
        grp.close()

        # NOTE uncomment when deleting is "synced" in core
        # grp.open("w")
        # del grp.meta["int"]
        # grp.close()

        # grp = tiledb.Group(grp_path, "r")
        # assert len(grp.meta) == 2
        # assert "int" not in grp.meta
        # grp.close()

    def test_group_members(self, capfd):
        grp_path = self.path("test_group_members")
        tiledb.Group.create(grp_path)

        grp = tiledb.Group(grp_path, "w")
        assert os.path.basename(grp.uri) == os.path.basename(grp_path)
        array_path = self.path("test_group_members")
        domain = tiledb.Domain(tiledb.Dim(domain=(1, 8), tile=2))
        a1 = tiledb.Attr("val", dtype="f8")
        schema = tiledb.ArraySchema(domain=domain, attrs=(a1,))
        tiledb.Array.create(array_path, schema)

        grp0_path = self.path("test_group_0")
        tiledb.Group.create(grp0_path)
        grp.add(grp0_path)
        grp.add(array_path)
        grp.close()
        assert not grp.isopen

        grp.open("r")
        assert grp.mode == "r"
        assert grp.isopen
        assert len(grp) == 2

        type_to_basename = {
            tiledb.Array: os.path.basename(array_path),
            tiledb.Group: os.path.basename(grp0_path),
        }

        assert grp[0].type in type_to_basename
        assert type_to_basename[grp[0].type] == os.path.basename(grp[0].uri)
        assert grp[1].type in type_to_basename
        assert type_to_basename[grp[1].type] == os.path.basename(grp[1].uri)

        assert "test_group_members GROUP" in repr(grp)
        assert "|-- test_group_members ARRAY" in repr(grp)
        assert "|-- test_group_0 GROUP" in repr(grp)

        grp.close()

        grp.open("w")
        assert grp.mode == "w"
        grp.remove(grp0_path)
        grp.close()

        grp = tiledb.Group(grp_path, "r")
        assert len(grp) == 1
        for mbr in grp:
            assert os.path.basename(mbr.uri) == os.path.basename(array_path)
            assert mbr.type == tiledb.Array
        grp.close()

    def test_group_named_members(self):
        grp_path = self.path("test_group_named_members")
        tiledb.Group.create(grp_path)

        subgrp_path = self.path("subgroup")
        tiledb.Group.create(subgrp_path)

        array_path = self.path("subarray")
        domain = tiledb.Domain(tiledb.Dim(domain=(1, 8), tile=2))
        a1 = tiledb.Attr("val", dtype="f8")
        schema = tiledb.ArraySchema(domain=domain, attrs=(a1,))
        tiledb.Array.create(array_path, schema)

        grp = tiledb.Group(grp_path, "w")
        grp.add(subgrp_path, "subgroup")
        grp.add(array_path, "subarray")
        grp.close()

        grp.open("r")
        assert os.path.basename(grp["subarray"].uri) == os.path.basename(array_path)
        assert os.path.basename(grp["subgroup"].uri) == os.path.basename(subgrp_path)
        assert grp["subarray"].type == tiledb.Array
        assert grp["subgroup"].type == tiledb.Group
        grp.close()

        grp.open("w")
        del grp["subarray"]
        grp.remove("subgroup")
        grp.close()

        grp.open("r")
        assert len(grp) == 0
        grp.close()
