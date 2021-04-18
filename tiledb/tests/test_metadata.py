import os
import time

import tiledb
import numpy as np
from hypothesis import given, settings, strategies as st

from tiledb.tests.common import DiskTestCase, rand_utf8


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


class MetadataTest(DiskTestCase):
    def assert_equal_md_values(self, written_value, read_value):
        if not isinstance(written_value, (list, tuple)):
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
        blob = np.random.rand(int((1024 ** 2) / 8)).tobytes()
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

    def test_consecutive(self):
        ctx = tiledb.Ctx(
            {"sm.vacuum.mode": "array_meta", "sm.consolidation.mode": "array_meta"}
        )
        vfs = tiledb.VFS(ctx=ctx)
        path = self.path("test_md_consecutive")

        write_count = 100

        with tiledb.from_numpy(path, np.ones((5,), np.float64), ctx=ctx) as A:
            pass

        randints = np.random.randint(0, MAX_INT - 1, size=write_count, dtype=np.int64)
        randutf8s = [rand_utf8(i) for i in np.random.randint(1, 30, size=write_count)]

        # write 100 times, then consolidate
        for i in range(write_count):
            with tiledb.Array(path, mode="w", ctx=ctx) as A:
                A.meta["randint"] = int(randints[i])
                A.meta["randutf8"] = randutf8s[i]
                time.sleep(0.001)

        with tiledb.Array(path) as A:
            self.assertEqual(A.meta["randint"], randints[-1])
            self.assertEqual(A.meta["randutf8"], randutf8s[-1])

        with tiledb.Array(path, mode="w", ctx=ctx) as aw:
            aw.meta.consolidate()

        with tiledb.Array(path) as A:
            self.assertEqual(A.meta["randint"], randints[-1])
            self.assertEqual(A.meta["randutf8"], randutf8s[-1])

        # use randutf8s as keys, then consolidate
        for _ in range(2):
            for i in range(write_count):
                with tiledb.Array(path, mode="w") as A:
                    A.meta[randutf8s[i] + u"{}".format(randints[i])] = int(randints[i])
                    A.meta[randutf8s[i]] = randutf8s[i]
                    time.sleep(0.001)

        # test data
        with tiledb.Array(path, ctx=ctx) as A:
            for i in range(write_count):
                key_int = randutf8s[i] + u"{}".format(randints[i])
                self.assertEqual(A.meta[key_int], randints[i])
                self.assertEqual(A.meta[randutf8s[i]], randutf8s[i])

        # test expected number of fragments before consolidating
        self.assertEqual(len(vfs.ls(os.path.join(path, "__meta"))), 302)

        with tiledb.Array(path, mode="w", ctx=ctx) as A:
            A.meta.consolidate()

        # test expected number of fragments before vacuuming
        self.assertEqual(len(vfs.ls(os.path.join(path, "__meta"))), 304)

        tiledb.vacuum(path, ctx=ctx)

        # should only have one fragment+'.ok' after vacuuming
        self.assertEqual(len(vfs.ls(os.path.join(path, "__meta"))), 1)

        # test data again after consolidation
        with tiledb.Array(path, ctx=ctx) as A:
            for i in range(write_count):
                key_int = randutf8s[i] + u"{}".format(randints[i])
                self.assertEqual(A.meta[key_int], randints[i])
                self.assertEqual(A.meta[randutf8s[i]], randutf8s[i])

    def test_metadata_small_dtypes(self):
        path = self.path("test_md_small_dtypes")

        with tiledb.from_numpy(path, np.arange(1)) as A:
            pass

        test_vals = {
            "np.int8": np.array((-1,), dtype=np.int8),
            "np.uint8": np.array((2,), dtype=np.uint8),
            "np.int16": np.array((-3,), dtype=np.int16),
            "np.uint16": np.array((4,), dtype=np.uint16),
            "np.int32": np.array((-5,), dtype=np.int32),
            "np.uint32": np.array((6,), dtype=np.uint32),
            "np.float32": np.array((-7.0,), dtype=np.float32),
        }

        with tiledb.Array(path, "w") as A:
            for k, v in test_vals.items():
                A.meta._set_numpy(k, v)

        # note: the goal here is to test read-back of these datatypes,
        #       which is currently done as int for all types
        with tiledb.Array(path) as A:
            for k, v in test_vals.items():
                self.assertEqual(A.meta[k], int(v))
