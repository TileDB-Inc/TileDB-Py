import base64
import io
import os
import pathlib
import tarfile

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as st_np

import tiledb

from .common import DiskTestCase, assert_captured

MIN_INT = np.iinfo(np.int64).min
MAX_INT = np.iinfo(np.int64).max
st_int = st.integers(min_value=MIN_INT, max_value=MAX_INT)
st_float = st.floats(allow_nan=False)
st_metadata = st.fixed_dictionaries(
    {
        "int": st_int,
        "double": st_float,
        "bytes": st.binary(),
        "str": st.text(),
        "list_int": st.lists(st_int),
        "tuple_int": st.lists(st_int).map(tuple),
        "list_float": st.lists(st_float),
        "tuple_float": st.lists(st_float).map(tuple),
    }
)
st_ndarray = st_np.arrays(
    dtype=st.one_of(
        st_np.integer_dtypes(endianness="<"),
        st_np.unsigned_integer_dtypes(endianness="<"),
        st_np.floating_dtypes(endianness="<", sizes=(32, 64)),
        st_np.byte_string_dtypes(max_len=1),
        st_np.unicode_string_dtypes(endianness="<", max_len=1),
        st_np.datetime64_dtypes(endianness="<"),
    ),
    shape=st_np.array_shapes(min_dims=0, max_dims=3, min_side=0, max_side=10),
)


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
        "int_data,flt_data,str_data,str_type",
        (
            (-1, -1.5, "asdf", "STRING_UTF8"),
            ([1, 2, 3], [1.5, 2.5, 3.5], b"asdf", "BLOB"),
            (
                np.array([1, 2, 3]),
                np.array([1.5, 2.5, 3.5]),
                np.array(["x"]),
                "STRING_UTF8",
            ),
        ),
    )
    def test_group_metadata(self, int_data, flt_data, str_data, str_type, capfd):
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

        cfg = tiledb.Config()
        with tiledb.Group(grp_path, "w", cfg) as grp:
            grp.meta["int"] = int_data
            grp.meta["flt"] = flt_data
            grp.meta["str"] = str_data

        cfg = tiledb.Config()
        with tiledb.Group(grp_path, "r", cfg) as grp:
            assert len(grp.meta) == 3
            assert "int" in grp.meta
            assert values_equal(grp.meta["int"], int_data)
            assert "flt" in grp.meta
            assert values_equal(grp.meta["flt"], flt_data)
            assert "str" in grp.meta
            assert values_equal(grp.meta["str"], str_data)

            grp.meta.dump()
            metadata_dump = capfd.readouterr().out

            assert "Type: DataType.FLOAT" in metadata_dump
            assert "Type: DataType.INT" in metadata_dump
            assert f"Type: DataType.{str_type}" in metadata_dump

        cfg = tiledb.Config()
        with tiledb.Group(grp_path, "w", cfg) as grp:
            del grp.meta["int"]

        cfg = tiledb.Config()
        with tiledb.Group(grp_path, "r", cfg) as grp:
            assert len(grp.meta) == 2
            assert "int" not in grp.meta

    def test_group_members(self):
        grp_path = self.path("test_group_members")
        tiledb.Group.create(grp_path)

        grp = tiledb.Group(grp_path, "w")
        assert os.path.basename(grp.uri) == os.path.basename(grp_path)
        array_path = self.path("test_group_members_array")
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
        assert grp[0].name is None

        assert grp[1].type in type_to_basename
        assert type_to_basename[grp[1].type] == os.path.basename(grp[1].uri)
        assert grp[1].name is None

        assert "test_group_members GROUP" in repr(grp)
        assert "|-- test_group_members_array ARRAY" in repr(grp)
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

        assert "dne" not in grp

        assert "subarray" in grp
        assert grp["subarray"].type == tiledb.Array

        assert "subgroup" in grp
        assert grp["subgroup"].type == tiledb.Group

        for mbr in grp:
            if "subarray" in mbr.uri:
                assert mbr.name == "subarray"
            elif "subgroup" in mbr.uri:
                assert mbr.name == "subgroup"

        grp.close()

        with tiledb.Group(grp_path, "w") as grp:  # test __enter__ and __exit__
            del grp["subarray"]
            grp.remove("subgroup")

        grp.open("r")
        assert len(grp) == 0
        grp.close()

    def test_pass_context(self):
        foo = self.path("foo")
        bar = self.path("foo/bar")

        tiledb.group_create(foo)
        tiledb.group_create(bar)

        ctx = tiledb.Ctx()
        with tiledb.Group(foo, mode="w", ctx=ctx) as G:
            G.add(bar, name="bar")

        with tiledb.Group(foo, mode="r", ctx=ctx) as G:
            assert "bar" in G

    def test_relative(self):
        group1 = self.path("group1")
        group2_1 = self.path("group1/group2_1")
        group2_2 = self.path("group1/group2_2")

        tiledb.group_create(group2_1)
        tiledb.group_create(group2_2)

        with tiledb.Group(group1, mode="w") as G:
            G.add(group2_1, name="group2_1", relative=False)
            G.add("group2_2", name="group2_2", relative=True)

        with tiledb.Group(group1, mode="r") as G:
            assert G.is_relative("group2_1") is False
            assert G.is_relative("group2_2") is True

    def test_set_config(self):
        group_uri = self.path("foo")
        array_uri_1 = self.path("foo/a")
        array_uri_2 = self.path("foo/b")

        tiledb.group_create(group_uri)

        dom = tiledb.Domain(tiledb.Dim("id", dtype="ascii"))
        attr = tiledb.Attr("value", dtype=np.int64)
        sch = tiledb.ArraySchema(domain=dom, attrs=(attr,), sparse=True)

        tiledb.Array.create(array_uri_1, sch)
        tiledb.Array.create(array_uri_2, sch)

        cfg = tiledb.Config({"sm.group.timestamp_end": 2000})
        with tiledb.Group(group_uri, "w", cfg) as G:
            G.add(name="a", uri="a", relative=True)

        cfg = tiledb.Config({"sm.group.timestamp_end": 3000})
        with tiledb.Group(group_uri, "w", cfg) as G:
            G.add(name="b", uri="b", relative=True)

        ms = np.arange(1000, 4000, 1000, dtype=np.int64)

        for sz, m in enumerate(ms):
            cfg = tiledb.Config({"sm.group.timestamp_end": m})

            G = tiledb.Group(group_uri)

            # Cannot set config on open group
            with self.assertRaises(ValueError):
                G.set_config(cfg)

            G.close()
            G.set_config(cfg)

            G.open()
            assert len(G) == sz
            G.close()

        for sz, m in enumerate(ms):
            cfg = tiledb.Config({"sm.group.timestamp_end": m})

            with tiledb.Group(group_uri, config=cfg) as G:
                assert len(G) == sz

    def test_invalid_object_type(self):
        path = self.path()
        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(tiledb.Dim("id", dtype="ascii")),
            attrs=(tiledb.Attr("value", dtype=np.int64),),
            sparse=True,
        )
        tiledb.Array.create(path, schema)
        with self.assertRaises(tiledb.TileDBError):
            tiledb.Group(uri=path, mode="w")

    def test_group_does_not_exist(self):
        with self.assertRaises(tiledb.TileDBError):
            tiledb.Group("does-not-exist")


class GroupMetadataTest(GroupTestCase):
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

        cfg = tiledb.Config()
        with tiledb.Group(grp_path, "w", cfg) as grp:
            grp.meta["int"] = int_data
            grp.meta["flt"] = flt_data
            grp.meta["str"] = str_data

        cfg = tiledb.Config()
        with tiledb.Group(grp_path, "r", cfg) as grp:
            assert grp.meta.keys() == {"int", "flt", "str"}
            assert len(grp.meta) == 3
            assert "int" in grp.meta
            assert values_equal(grp.meta["int"], int_data)
            assert "flt" in grp.meta
            assert values_equal(grp.meta["flt"], flt_data)
            assert "str" in grp.meta
            assert values_equal(grp.meta["str"], str_data)

        cfg = tiledb.Config()
        with tiledb.Group(grp_path, "w", cfg) as grp:
            del grp.meta["int"]

        cfg = tiledb.Config()
        with tiledb.Group(grp_path, "r", cfg) as grp:
            assert len(grp.meta) == 2
            assert "int" not in grp.meta

    def assert_equal_md_values(self, written_value, read_value):
        if isinstance(written_value, np.ndarray):
            self.assertIsInstance(read_value, np.ndarray)
            self.assertEqual(read_value.dtype, written_value.dtype)
            np.testing.assert_array_equal(read_value, written_value)
        elif not isinstance(written_value, (list, tuple)):
            assert read_value == written_value
        # we don't round-trip perfectly sequences
        elif len(written_value) == 1:
            # sequences of length 1 are read as a single scalar element
            self.assertEqual(read_value, written_value[0])
        else:
            # sequences of length != 1 are read as tuples
            self.assertEqual(read_value, tuple(written_value))

    def assert_metadata_roundtrip(self, tdb_meta, dict_meta):
        for k, v in dict_meta.items():
            # test __contains__
            self.assertTrue(k in tdb_meta)
            # test __getitem__
            self.assert_equal_md_values(v, tdb_meta[k])
            # test get
            self.assert_equal_md_values(v, tdb_meta.get(k))

        # test __contains__, __getitem__, get for non-key
        non_key = str(object())
        self.assertFalse(non_key in tdb_meta)
        with self.assertRaises(KeyError):
            tdb_meta[non_key]
        self.assertIsNone(tdb_meta.get(non_key))
        self.assertEqual(tdb_meta.get(non_key, 42), 42)

        # test __len__
        self.assertEqual(len(tdb_meta), len(dict_meta))

        # test __iter__()
        self.assertEqual(set(tdb_meta), set(tdb_meta.keys()))

        # test keys()
        self.assertSetEqual(set(tdb_meta.keys()), set(dict_meta.keys()))

        # test values() and items()
        read_values = list(tdb_meta.values())
        read_items = list(tdb_meta.items())
        self.assertEqual(len(read_values), len(read_items))
        for (item_key, item_value), value in zip(read_items, read_values):
            self.assertTrue(item_key in dict_meta)
            self.assert_equal_md_values(dict_meta[item_key], item_value)
            self.assert_equal_md_values(dict_meta[item_key], value)

    def assert_not_implemented_methods(self, tdb_meta):
        with self.assertRaises(NotImplementedError):
            tdb_meta.setdefault("nokey", "hello!")
        with self.assertRaises(NotImplementedError):
            tdb_meta.pop("nokey", "hello!")
        with self.assertRaises(NotImplementedError):
            tdb_meta.popitem()
        with self.assertRaises(NotImplementedError):
            tdb_meta.clear()

    def test_errors(self):
        path = self.path("test_errors")

        tiledb.Group.create(path)

        grp = tiledb.Group(path, "w")
        grp.close()

        # can't read from a closed array
        grp.open("r")
        grp.close()
        with self.assertRaises(tiledb.TileDBError):
            grp.meta["x"]

        grp.open("r")
        # can't write to a mode='r' array
        with self.assertRaises(tiledb.TileDBError):
            grp.meta["invalid_write"] = 1

        # missing key raises KeyError
        with self.assertRaises(KeyError):
            grp.meta["xyz123nokey"]

        self.assert_not_implemented_methods(grp.meta)
        grp.close()

        # test invalid input
        grp.open("w")
        # keys must be strings
        with self.assertRaises(TypeError):
            grp.meta[123] = 1

        # # can't write an int > typemax(Int64)
        with self.assertRaises(OverflowError):
            grp.meta["bigint"] = MAX_INT + 1

        # can't write str list
        with self.assertRaises(TypeError):
            grp.meta["str_list"] = ["1", "2.1"]

        # can't write str tuple
        with self.assertRaises(TypeError):
            grp.meta["mixed_list"] = ("1", "2.1")

        # can't write objects
        with self.assertRaises(TypeError):
            grp.meta["object"] = object()

        self.assert_not_implemented_methods(grp.meta)
        grp.close()

    def test_null(self):
        path = self.path()
        tiledb.Group.create(path)

        grp = tiledb.Group(path, "w")
        grp.meta["empty_byte"] = b""
        grp.meta["null_byte"] = b"\x00"
        grp.meta["zero"] = "xxx"
        grp.close()

        grp = tiledb.Group(path, "r")
        assert grp.meta["empty_byte"] == b""
        assert grp.meta["null_byte"] == b"\x00"
        assert grp.meta["zero"] == "xxx"
        grp.close()

    @given(st_metadata)
    @settings(deadline=None)
    def test_basic(self, test_vals):
        path = self.path()
        tiledb.Group.create(path)

        grp = tiledb.Group(path, "w")
        grp.meta.update(test_vals)
        grp.close()

        grp = tiledb.Group(path, "r")
        self.assert_metadata_roundtrip(grp.meta, test_vals)
        grp.close()

        # test a 1 MB blob
        blob = np.random.rand(int((1024**2) / 8)).tobytes()
        grp = tiledb.Group(path, "w")
        test_vals["bigblob"] = blob
        grp.meta["bigblob"] = blob
        grp.close()

        grp = tiledb.Group(path, "r")
        self.assert_metadata_roundtrip(grp.meta, test_vals)
        grp.close()

        # test del key
        grp = tiledb.Group(path, "w")
        del test_vals["bigblob"]
        del grp.meta["bigblob"]
        grp.close()

        grp = tiledb.Group(path, "r")
        self.assert_metadata_roundtrip(grp.meta, test_vals)
        grp.close()

        # test update
        grp = tiledb.Group(path, "w")
        test_vals.update(foo="bar", double=3.14)
        grp.meta.update(foo="bar", double=3.14)
        grp.close()

        grp = tiledb.Group(path, "r")
        self.assert_metadata_roundtrip(grp.meta, test_vals)
        grp.close()

    @given(st_metadata, st_ndarray)
    @settings(deadline=None)
    def test_numpy(self, test_vals, ndarray):
        test_vals["ndarray"] = ndarray

        path = self.path()
        tiledb.Group.create(path)

        grp = tiledb.Group(path, "w")
        grp.meta.update(test_vals)
        grp.close()

        grp = tiledb.Group(path, "r")
        self.assert_metadata_roundtrip(grp.meta, test_vals)
        grp.close()

        grp = tiledb.Group(path, "w")
        grp.meta["ndarray"] = 42
        test_vals["ndarray"] = 42
        grp.close()

        grp = tiledb.Group(path, "r")
        self.assert_metadata_roundtrip(grp.meta, test_vals)
        grp.close()

        # test resetting a key with a non-ndarray value to a ndarray value
        grp = tiledb.Group(path, "w")
        grp.meta["bytes"] = ndarray
        test_vals["bytes"] = ndarray
        grp.close()

        grp = tiledb.Group(path, "r")
        self.assert_metadata_roundtrip(grp.meta, test_vals)
        grp.close()

        grp = tiledb.Group(path, "w")
        del grp.meta["ndarray"]
        del test_vals["ndarray"]
        grp.close()

        grp = tiledb.Group(path, "r")
        self.assert_metadata_roundtrip(grp.meta, test_vals)
        grp.close()

        grp = tiledb.Group(path, "w")
        test_vals.update(ndarray=np.stack([ndarray, ndarray]), transp=ndarray.T)
        grp.meta.update(ndarray=np.stack([ndarray, ndarray]), transp=ndarray.T)
        grp.close()

        grp = tiledb.Group(path, "r")
        self.assert_metadata_roundtrip(grp.meta, test_vals)
        grp.close()

    def test_consolidation_and_vac(self):
        vfs = tiledb.VFS()
        path = self.path("test_consolidation_and_vac")
        tiledb.Group.create(path)

        cfg = tiledb.Config()
        with tiledb.Group(path, "w", cfg) as grp:
            grp.meta["meta"] = 1

        cfg = tiledb.Config()
        with tiledb.Group(path, "w", cfg) as grp:
            grp.meta["meta"] = 2

        cfg = tiledb.Config()
        with tiledb.Group(path, "w", cfg) as grp:
            grp.meta["meta"] = 3

        meta_path = pathlib.Path(path) / "__meta"
        assert len(vfs.ls(meta_path)) == 3

        tiledb.Group.consolidate_metadata(path, cfg)
        tiledb.Group.vacuum_metadata(path, cfg)

        assert len(vfs.ls(meta_path)) == 1

    @pytest.mark.skipif(
        tiledb.libtiledb.version() < (2, 18, 0),
        reason="Group consolidation and vacuuming not available < 2.18",
    )
    def test_consolidation_and_vac_no_config(self):
        vfs = tiledb.VFS()
        path = self.path("test_consolidation_and_vac")
        tiledb.Group.create(path)

        with tiledb.Group(path, "w") as grp:
            grp.meta["meta"] = 1

        with tiledb.Group(path, "w") as grp:
            grp.meta["meta"] = 2

        with tiledb.Group(path, "w") as grp:
            grp.meta["meta"] = 3

        meta_path = pathlib.Path(path) / "__meta"
        assert len(vfs.ls(meta_path)) == 3

        tiledb.Group.consolidate_metadata(path)
        tiledb.Group.vacuum_metadata(path)

        assert len(vfs.ls(meta_path)) == 1

    def test_string_metadata(self, capfd):
        # this test ensures that string metadata is correctly stored and
        # retrieved from the metadata store. It also tests that the metadata
        # dump method works correctly for string metadata.
        uri = self.path("test_ascii_metadata")
        tiledb.Group.create(uri)

        grp = tiledb.Group(uri, "w")
        grp.meta["abc"] = "xyz"
        grp.close()

        grp = tiledb.Group(uri, "r")
        assert grp.meta["abc"] == "xyz"
        grp.meta.dump()
        assert_captured(capfd, "Type: DataType.STRING_UTF8")
        grp.close()

    def test_array_or_list_of_strings_metadata_error(self):
        # this test ensures that an error is raised when trying to store
        # an array or list of strings as metadata in a group.
        # numpy arrays of single characters are supported since we don't need
        # any extra offset information to retrieve them.
        uri = self.path("test_ascii_metadata")
        tiledb.Group.create(uri)

        grp = tiledb.Group(uri, "w")
        with pytest.raises(TypeError) as exc:
            grp.meta["abc"] = ["x", "1"]
        assert "Unsupported item type" in str(exc.value)

        with pytest.raises(TypeError) as exc:
            grp.meta["abc"] = ["foo", "foofoo"]

        with pytest.raises(TypeError) as exc:
            grp.meta["abc"] = np.array(["foo", "12345"])

        grp.meta["abc"] = np.array(["1", "2", "3", "f", "o", "o"], dtype="U1")
        grp.close()

        grp = tiledb.Group(uri, "r")
        self.assert_metadata_roundtrip(
            grp.meta, {"abc": np.array(["1", "2", "3", "f", "o", "o"], dtype="U1")}
        )
        grp.close()

        grp = tiledb.Group(uri, "w")
        grp.meta["abc"] = np.array(["T", "i", "l", "e", "D", "B", "!"], dtype="S1")
        grp.close()

        grp = tiledb.Group(uri, "r")
        self.assert_metadata_roundtrip(
            grp.meta,
            {"abc": np.array([b"T", b"i", b"l", b"e", b"D", b"B", b"!"], dtype="S1")},
        )
        grp.close()

    def test_bytes_metadata(self, capfd):
        # this test ensures that bytes metadata is correctly stored and
        # retrieved from the metadata store. It also tests that the metadata
        # dump method works correctly for bytes metadata.
        path = self.path()
        tiledb.Group.create(path)

        grp = tiledb.Group(path, "w")
        grp.meta["bytes"] = b"blob"
        grp.close()

        grp = tiledb.Group(path, "r")
        assert grp.meta["bytes"] == b"blob"
        grp.meta.dump()
        assert_captured(capfd, "Type: DataType.BLOB")
        grp.close()

    def test_group_metadata_backwards_compat(self):
        # This test ensures that metadata written with the TileDB-Py 0.32.3
        # will be read correctly in the future versions.

        # === The following code creates a group with metadata using the current version of TileDB-Py ===
        path_new = self.path("new_group")
        tiledb.Group.create(path_new)
        group = tiledb.Group(path_new, "w")

        # python primitive types
        group.meta["python_int"] = -1234
        group.meta["python_float"] = 3.14
        group.meta["python_str"] = "hello"
        group.meta["python_bytes"] = b"hello"
        group.meta["python_bool"] = False

        # numpy primitive types
        group.meta["numpy_int"] = np.int64(-93)
        group.meta["numpy_uint"] = np.uint64(42)
        group.meta["numpy_float64"] = np.float64(3.14)
        group.meta["numpy_bytes"] = np.bytes_("hello")
        group.meta["numpy_str"] = np.str_("hello")
        group.meta["numpy_bool"] = np.bool_(False)

        # lists/tuples
        group.meta["list_int"] = [7]
        group.meta["tuple_int"] = (7,)
        group.meta["list_ints"] = [1, -2, 3]
        group.meta["tuple_ints"] = (1, 2, 3)
        group.meta["list_float"] = [1.1]
        group.meta["tuple_float"] = (1.1,)
        group.meta["list_floats"] = [1.1, 2.2, 3.3]
        group.meta["tuple_floats"] = (1.1, 2.2, 3.3)
        group.meta["list_empty"] = []
        group.meta["tuple_empty"] = ()

        # numpy arrays
        group.meta["numpy_int"] = np.array([-11], dtype=np.int64)
        group.meta["numpy_ints"] = np.array([1, -2, 3], dtype=np.int64)
        group.meta["numpy_uint"] = np.array([22], dtype=np.uint64)
        group.meta["numpy_uints"] = np.array([1, 2, 3], dtype=np.uint64)
        group.meta["numpy_float"] = np.array([3.14], dtype=np.float64)
        group.meta["numpy_floats"] = np.array([1.1, 2.2, 3.3], dtype=np.float64)
        group.meta["numpy_byte"] = np.array([b"hello"], dtype="S5")
        group.meta["numpy_str"] = np.array(["hello"], dtype="U5")
        group.meta["numpy_bool"] = np.array([True, False, True])

        group.close()
        # === End of the code that creates the group with metadata ===

        # The following commented out code was used to generate the base64 encoded string of the group
        # from the TileDB-Py 0.32.3 after creating the group with metadata in the exact same way as above.
        '''
        # Compress the contents of the group folder to tgz
        with tarfile.open("test.tar.gz", "w:gz") as tar:
            with os.scandir(path_new) as entries:
                for entry in entries:
                    tar.add(entry.path, arcname=entry.name)

        # Read the .tgz file and encode it to base64
        with open("test.tar.gz", 'rb') as f:
            s = base64.encodebytes(f.read())

        # Print the base64 encoded string
        group_tgz = f"""{s.decode():>32}"""
        print(group_tgz)
        '''

        # The following base64 encoded string is the contents of the group folder compressed
        # to a tgz file using TileDB-Py 0.32.3.
        group_tgz = b"""H4sICO/+G2cC/3Rlc3QudGFyANPT19N3CEis8EhNTEktYqAJMIAAXLSBgbEJgg0SNzQwMjRiUKhg
                        oAMoLS5JLAJazzAygZGFQm5JZm6qraG5kaWFhbmlhbGekaGphbGlJRfDKBj2ID4+N7UkUZ+mdoAy
                        tbmpKYQ2g9AGRqh53tDE3MDM3Nzc2NQcmP8NDc3NGRRM6Zn/E9Mzi/GpAypLSxt+8a83KMp/Y8zy
                        33C0/KdL+W+Otfy3NBot/kdS+R8fj4h/YPSj8UxTktOSjQxMjNPMzS0MDCxTjVLNTUwS01IMzMxM
                        zJMTicj/ZiYmuMp/QwNjM9Ty38jQAFhdKBjQM/+P0PJfDIhfMULYV1khNAsjTFYITDIygAQYQbKM
                        YBYDQv0xIEcAymdEEqtgbA1x9DtsIBATrJgRpRfwgC18R8GqqqXxD1gDJwZtnTTb5YbtE0YbprhD
                        8y0KH7SwVJTnps9d9sorMOX8Met7M8+yMHzas+bz0rgbMet7z3b75kqb3mSdtisqonQnu8GrGvHI
                        6WGxX/Jm+7UW7V45+8/OVSZ3+O+Ic/0Sloo+8OKG6hqutaun9NgfXjqDz9ftBZNBwLvXt6+fX94/
                        ++EfK0X1S2nBpVv5jQ0cut7nS8T3/wn7rOpq5q9/Jn2XW8OhQ/frZTLrkycxHt1evlKvrtbsXeIX
                        2dw33D0fd0yt5vqe8T/k3d3wtO4UI5Vm8yMvspXTJE+ozFY+13ZA7e+avDertDwP+b1mcjq0JPar
                        QLS26mvFLQH6D97dDbyZlx1b8X/ZHYmHWpqMjTP6QiVvrZX/3nsqxv3WwofHjtgmbk+YGnhC/U1D
                        v5+z0SvXZ5YfmXhYiw4Ynmi727rZteXvpZULJ/jvNikQV1/tuiM73XDytc2ZVu6PRcy4NN3Cuze9
                        0GJc1KHr+mXOAxexJaUFAv/kVgi/K+FaI+2wZfqOxoYWocQPGzNeG9h9edh+3DfBJMYzOKL2l+em
                        ezc0Hyq98xaQ8eT40PDoxpYX60KKnogs7Ht2d+cf9lm5m9pGy8fhDvRG+/+j/X+M9p+JqYGJ+WgD
                        cES0/0oyc1JTkuLTi/JLC/RKUpJok//xtP+w9P+NTUD9v9H232j5P1r+D0j5b2ZoYDZa/o+I8h9c
                        8NN0AJiM8V8TA9PR8d9RMApGwSgYBaNgFIyCUTAKRsEooCYAAP1+F2wAKAAA"""

        # Ceate a new group by extracting the contents of the tgz file
        path_original = self.path("original_group")
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(group_tgz))) as tf:
            try:
                tf.extractall(path_original, filter="fully_trusted")
            except TypeError:
                tf.extractall(path_original)

        # Open both the original and the new group and compare the metadata both in values and types
        group_original = tiledb.Group(path_original, "r")
        group_new = tiledb.Group(path_new, "r")

        self.assert_metadata_roundtrip(group_new.meta, group_original.meta)

        group_original.close()
        group_new.close()

    def test_group_metadata_new_types(self):
        # This kind of data was not supported for TileDB-Py <= 0.32.3
        path_new = self.path("new_group")

        tiledb.Group.create(path_new)
        group = tiledb.Group(path_new, "w")
        test_vals = {
            "int64": np.array(-1111, dtype=np.int64),
            "uint64": np.array(2, dtype=np.uint64),
            "float64": np.array(3.14, dtype=np.float64),
            "bool": np.array(True, dtype=bool),
            "str": np.array(["a", "b", "c"], dtype="S"),
            "unicode": np.array(["a", "b", "c"], dtype="U"),
            "bytes": np.array([b"a", b"b", b"c"]),
            "datetime": np.array(
                [np.datetime64("2021-01-01"), np.datetime64("2021-01-02")]
            ),
        }
        group.meta.update(test_vals)
        group.close()

        group = tiledb.Group(path_new, "r")
        self.assert_metadata_roundtrip(group.meta, test_vals)
        group.close()
