import tempfile

import numpy as np
from numpy.testing import assert_array_equal

import tiledb

from .common import DiskTestCase


class UtilTest(DiskTestCase):
    def test_empty_like(self):
        arr = np.zeros((10, 10), dtype=np.float32)

        def check_schema(self, s):
            self.assertEqual(s.attr(0).dtype, np.float32)
            self.assertEqual(s.shape, (10, 10))
            self.assertEqual(s.domain.dim(0).shape, (10,))
            self.assertEqual(s.domain.dim(1).shape, (10,))

        with self.assertRaises(ValueError):
            tiledb.schema_like("", None)

        schema = tiledb.schema_like(arr, tile=1)
        self.assertIsInstance(schema, tiledb.ArraySchema)
        check_schema(self, schema)

        uri = self.path("empty_like")
        T = tiledb.empty_like(uri, arr)
        check_schema(self, T.schema)
        self.assertEqual(T.shape, arr.shape)
        self.assertEqual(T.dtype, arr.dtype)

        uri = self.path("empty_like_shape")
        T = tiledb.empty_like(uri, arr.shape, dtype=arr.dtype)
        check_schema(self, T.schema)
        self.assertEqual(T.shape, arr.shape)
        self.assertEqual(T.dtype, arr.dtype)

        # test a fake object with .shape, .ndim, .dtype
        class FakeArray(object):
            def __init__(self, shape, dtype):
                self.shape = shape
                self.ndim = len(shape)
                self.dtype = dtype

        fake = FakeArray((3, 3), np.int16)
        schema2 = tiledb.empty_like(self.path("fake_like"), fake)
        self.assertIsInstance(schema2, tiledb.Array)
        self.assertEqual(schema2.shape, fake.shape)
        self.assertEqual(schema2.dtype, fake.dtype)
        self.assertEqual(schema2.ndim, fake.ndim)

        # test passing shape and dtype directly
        schema3 = tiledb.schema_like(shape=(4, 4), dtype=np.float32)
        self.assertIsInstance(schema3, tiledb.ArraySchema)
        self.assertEqual(schema3.attr(0).dtype, np.float32)
        self.assertEqual(schema3.domain.dim(0).tile, 4)
        schema3 = tiledb.schema_like(shape=(4, 4), dtype=np.float32, tile=1)
        self.assertEqual(schema3.domain.dim(0).tile, 1)

    def test_open(self):
        uri = self.path("load")
        with tiledb.from_numpy(uri, np.array(np.arange(3))) as T:
            with tiledb.open(uri) as T2:
                self.assertEqual(T.schema, T2.schema)
                assert_array_equal(T, T2)

    def test_save(self):
        uri = self.path("test_save")
        arr = np.array(np.arange(3))
        with tiledb.save(uri, arr):
            with tiledb.open(uri) as T:
                assert_array_equal(arr, T)

    def test_array_exists(self):
        with tempfile.NamedTemporaryFile() as tmpfn:
            self.assertFalse(tiledb.array_exists(tmpfn.name))

        uri = self.path("test_array_exists_dense")
        with tiledb.from_numpy(uri, np.arange(0, 5)) as T:
            self.assertTrue(tiledb.array_exists(uri))
            self.assertTrue(tiledb.array_exists(uri, isdense=True))
            self.assertFalse(tiledb.array_exists(uri, issparse=True))

        uri = self.path("test_array_exists_sparse")
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 3), tile=4, dtype=int))
        att = tiledb.Attr(dtype=int)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True)
        tiledb.Array.create(uri, schema)

        with tiledb.SparseArray(uri, mode="w") as T:
            T[[0, 1]] = np.array([0, 1])

        self.assertTrue(tiledb.array_exists(uri))
        self.assertTrue(tiledb.array_exists(uri, issparse=True))
        self.assertFalse(tiledb.array_exists(uri, isdense=True))

        uri3 = self.path("test_array_exists_deleted")
        with tiledb.from_numpy(uri3, np.arange(0, 5)) as T:
            self.assertTrue(tiledb.array_exists(uri3))
        tiledb.Array.delete_array(uri3)
        self.assertFalse(tiledb.array_exists(uri3))

        # test with context
        ctx = tiledb.Ctx()
        self.assertFalse(tiledb.array_exists(uri3, ctx=ctx))
        with tiledb.from_numpy(uri3, np.arange(0, 5), ctx=ctx) as T:
            self.assertTrue(tiledb.array_exists(uri3, ctx=ctx))
