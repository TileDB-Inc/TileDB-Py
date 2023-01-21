import os
import time
import warnings

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as st_np

import tiledb
from tiledb.main import metadata_test_aux

from .common import DiskTestCase, assert_captured, rand_utf8

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


class MetadataTest(DiskTestCase):
    def assert_equal_md_values(self, written_value, read_value):
        if isinstance(written_value, np.ndarray):
            self.assertIsInstance(read_value, np.ndarray)
            self.assertEqual(read_value.dtype, written_value.dtype)
            np.testing.assert_array_equal(read_value, written_value)
        elif not isinstance(written_value, (list, tuple)):
            self.assertEqual(read_value, written_value)
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

        # test __iter__() is consistent with keys()
        self.assertEqual(list(tdb_meta), tdb_meta.keys())

        # test keys()
        self.assertSetEqual(set(tdb_meta.keys()), set(dict_meta.keys()))

        # test values() and items()
        read_values = tdb_meta.values()
        read_items = tdb_meta.items()
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
        path = self.path("test_md_errors")
        with tiledb.from_numpy(path, np.ones((5,), np.float64)):
            pass

        # can't read from a closed array
        A = tiledb.open(path)
        A.close()
        with self.assertRaises(tiledb.TileDBError):
            A.meta["x"]

        with tiledb.Array(path) as A:
            # can't write to a mode='r' array
            with self.assertRaises(tiledb.TileDBError):
                A.meta["invalid_write"] = 1

            # missing key raises KeyError
            with self.assertRaises(KeyError):
                A.meta["xyz123nokey"]

            self.assert_not_implemented_methods(A.meta)

        # test invalid input
        with tiledb.Array(path, "w") as A:
            # keys must be strings
            with self.assertRaises(TypeError):
                A.meta[123] = 1

            # can't write an int > typemax(Int64)
            with self.assertRaises(OverflowError):
                A.meta["bigint"] = MAX_INT + 1

            # can't write mixed-type list
            with self.assertRaises(TypeError):
                A.meta["mixed_list"] = [1, 2.1]

            # can't write mixed-type tuple
            with self.assertRaises(TypeError):
                A.meta["mixed_list"] = (0, 3.1)

            # can't write objects
            with self.assertRaises(TypeError):
                A.meta["object"] = object()

            self.assert_not_implemented_methods(A.meta)

    @given(st_metadata)
    @settings(deadline=None)
    def test_basic(self, test_vals):
        path = self.path()
        with tiledb.from_numpy(path, np.ones((5,), np.float64)):
            pass

        with tiledb.Array(path, mode="w") as A:
            A.meta.update(test_vals)

        with tiledb.Array(path) as A:
            self.assert_metadata_roundtrip(A.meta, test_vals)

        # test a 1 MB blob
        blob = np.random.rand(int((1024**2) / 8)).tobytes()
        with tiledb.Array(path, "w") as A:
            test_vals["bigblob"] = blob
            A.meta["bigblob"] = blob

        with tiledb.Array(path) as A:
            self.assert_metadata_roundtrip(A.meta, test_vals)

        # test del key
        with tiledb.Array(path, "w") as A:
            del test_vals["bigblob"]
            del A.meta["bigblob"]

        with tiledb.Array(path) as A:
            self.assert_metadata_roundtrip(A.meta, test_vals)

        # test update
        with tiledb.Array(path, mode="w") as A:
            test_vals.update(foo="bar", double=3.14)
            A.meta.update(foo="bar", double=3.14)

        with tiledb.Array(path) as A:
            self.assert_metadata_roundtrip(A.meta, test_vals)

    @given(st_metadata, st_ndarray)
    @settings(deadline=None)
    def test_numpy(self, test_vals, ndarray):
        test_vals["ndarray"] = ndarray

        path = self.path()
        with tiledb.from_numpy(path, np.ones((5,), np.float64)):
            pass

        with tiledb.Array(path, mode="w") as A:
            A.meta.update(test_vals)

        with tiledb.Array(path) as A:
            self.assert_metadata_roundtrip(A.meta, test_vals)

        # test resetting a key with a ndarray value to a non-ndarray value
        time.sleep(0.001)
        with tiledb.Array(path, "w") as A:
            A.meta["ndarray"] = 42
            test_vals["ndarray"] = 42

        with tiledb.Array(path) as A:
            self.assert_metadata_roundtrip(A.meta, test_vals)

        # test resetting a key with a non-ndarray value to a ndarray value
        with tiledb.Array(path, "w") as A:
            A.meta["bytes"] = ndarray
            test_vals["bytes"] = ndarray

        with tiledb.Array(path) as A:
            self.assert_metadata_roundtrip(A.meta, test_vals)

        # test del ndarray key
        time.sleep(0.001)
        with tiledb.Array(path, "w") as A:
            del A.meta["ndarray"]
            del test_vals["ndarray"]

        with tiledb.Array(path) as A:
            self.assert_metadata_roundtrip(A.meta, test_vals)

        # test update
        time.sleep(0.001)
        with tiledb.Array(path, mode="w") as A:
            test_vals.update(ndarray=np.stack([ndarray, ndarray]), transp=ndarray.T)
            A.meta.update(ndarray=np.stack([ndarray, ndarray]), transp=ndarray.T)

        with tiledb.Array(path) as A:
            self.assert_metadata_roundtrip(A.meta, test_vals)

    @pytest.mark.filterwarnings("ignore::UserWarning")
    @tiledb.scope_ctx(
        {"sm.vacuum.mode": "array_meta", "sm.consolidation.mode": "array_meta"}
    )
    def test_consecutive(self):
        vfs = tiledb.VFS()
        path = self.path("test_md_consecutive")

        write_count = 100

        with tiledb.from_numpy(path, np.ones((5,), np.float64)):
            pass

        randints = np.random.randint(0, MAX_INT - 1, size=write_count, dtype=np.int64)
        randutf8s = [rand_utf8(i) for i in np.random.randint(1, 30, size=write_count)]

        # write 100 times, then consolidate
        for i in range(write_count):
            with tiledb.Array(path, mode="w") as A:
                A.meta["randint"] = int(randints[i])
                A.meta["randutf8"] = randutf8s[i]
                time.sleep(0.001)

        self.assertEqual(len(vfs.ls(os.path.join(path, "__meta"))), 100)

        with tiledb.Array(path) as A:
            self.assertEqual(A.meta["randint"], randints[-1])
            self.assertEqual(A.meta["randutf8"], randutf8s[-1])

        with tiledb.Array(path, mode="w") as aw:
            aw.meta.consolidate()

        try:
            self.assertEqual(len(vfs.ls(os.path.join(path, "__meta"))), 102)
        except AssertionError:
            # this test is broken under libtiledb 2.3, see ch 7449
            if tiledb.libtiledb.version() >= (2, 3):
                warnings.warn(
                    "Suppressed assertion error with libtiledb 2.3! see ch 7449"
                )
            else:
                raise

        with tiledb.Array(path) as A:
            self.assertEqual(A.meta["randint"], randints[-1])
            self.assertEqual(A.meta["randutf8"], randutf8s[-1])

        # use randutf8s as keys, then consolidate
        for _ in range(2):
            for i in range(write_count):
                with tiledb.Array(path, mode="w") as A:
                    A.meta[randutf8s[i] + "{}".format(randints[i])] = int(randints[i])
                    A.meta[randutf8s[i]] = randutf8s[i]
                    time.sleep(0.001)

        # test data
        with tiledb.Array(path) as A:
            for i in range(write_count):
                key_int = randutf8s[i] + "{}".format(randints[i])
                self.assertEqual(A.meta[key_int], randints[i])
                self.assertEqual(A.meta[randutf8s[i]], randutf8s[i])

        # test expected number of fragments before consolidating
        try:
            self.assertEqual(len(vfs.ls(os.path.join(path, "__meta"))), 302)
        except AssertionError:
            # this test is broken under libtiledb 2.3, see ch 7449
            if tiledb.libtiledb.version() >= (2, 3):
                warnings.warn(
                    "Suppressed assertion error with libtiledb 2.3! see ch 7449"
                )
            else:
                raise

        with tiledb.Array(path, mode="w") as A:
            A.meta.consolidate()

        # test expected number of fragments before vacuuming
        try:
            self.assertEqual(len(vfs.ls(os.path.join(path, "__meta"))), 304)
        except AssertionError:
            # this test is broken under libtiledb 2.3, see ch 7449
            if tiledb.libtiledb.version() >= (2, 3):
                warnings.warn(
                    "Suppressed assertion error with libtiledb 2.3! see ch 7449"
                )
            else:
                raise

        tiledb.vacuum(path)

        # should only have one fragment+'.ok' after vacuuming
        try:
            self.assertEqual(len(vfs.ls(os.path.join(path, "__meta"))), 1)
        except AssertionError:
            # this test is broken under libtiledb 2.3, see ch 7449
            if tiledb.libtiledb.version() >= (2, 3):
                warnings.warn(
                    "Suppressed assertion error with libtiledb 2.3! see ch 7449"
                )
            else:
                raise

        # test data again after consolidation
        with tiledb.Array(path) as A:
            for i in range(write_count):
                key_int = randutf8s[i] + "{}".format(randints[i])
                self.assertEqual(A.meta[key_int], randints[i])
                self.assertEqual(A.meta[randutf8s[i]], randutf8s[i])

    def test_ascii_metadata(self, capfd):
        uri = self.path("test_ascii_metadata")

        dom = tiledb.Domain(tiledb.Dim(domain=(0, 2), tile=1, dtype=np.int64))
        att = tiledb.Attr(dtype=np.int64)
        schema = tiledb.ArraySchema(sparse=True, domain=dom, attrs=(att,))
        tiledb.Array.create(uri, schema)

        metadata_test_aux.write_ascii(uri)

        with tiledb.open(uri) as A:
            assert A.meta["abc"] == b"xyz"
            A.meta.dump()
            assert_captured(capfd, "Type: STRING_ASCII")

    def test_bytes_metadata(self, capfd):
        path = self.path()
        with tiledb.from_numpy(path, np.ones((5,), np.float64)):
            pass

        with tiledb.Array(path, mode="w") as A:
            A.meta["bytes"] = b"blob"

        with tiledb.Array(path, mode="r") as A:
            assert A.meta["bytes"] == b"blob"
            A.meta.dump()
            assert_captured(capfd, "Type: BLOB")
