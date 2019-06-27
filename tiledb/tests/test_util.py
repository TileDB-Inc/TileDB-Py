#!/usr/bin/env python
# -*- coding: utf-8 -*-

import tiledb
from tiledb import *
from tiledb.libtiledb import index_as_tuple, replace_ellipsis
from tiledb.tests.common import DiskTestCase

import numpy as np
from numpy.testing import assert_equal, assert_approx_equal,\
                          assert_array_equal, assert_raises

import unittest
from unittest import TestCase


class UtilTest(DiskTestCase):
    def test_empty_like(self):
        arr = np.zeros((10,10), dtype=np.float32)

        def check_schema(self, s):
            self.assertEqual(s.attr(0).dtype, np.float32)
            self.assertEqual(s.shape, (10,10))
            self.assertEqual(s.domain.dim(0).shape, (10,))
            self.assertEqual(s.domain.dim(1).shape, (10,))

        with self.assertRaises(ValueError):
            schema_like('', None)

        schema = schema_like(arr, tile=1)
        self.assertIsInstance(schema, ArraySchema)
        check_schema(self, schema)

        uri = self.path("empty_like")
        T = empty_like(uri, arr)
        check_schema(self, T.schema)
        self.assertEqual(T.shape, arr.shape)
        self.assertEqual(T.dtype, arr.dtype)


        # test a fake object with .shape, .ndim, .dtype
        class FakeArray(object):
            def __init__(self, shape, dtype):
                self.shape = shape
                self.ndim = len(shape)
                self.dtype = dtype

        fake = FakeArray((3,3), np.int16)
        schema2 = empty_like(self.path('fake_like'), fake)
        self.assertIsInstance(schema2, Array)
        self.assertEqual(schema2.shape, fake.shape)
        self.assertEqual(schema2.dtype, fake.dtype)
        self.assertEqual(schema2.ndim, fake.ndim)

        # test passing shape and dtype directly
        schema3 = schema_like(shape=(4,4), dtype=np.float32)
        self.assertIsInstance(schema3, ArraySchema)
        self.assertEqual(schema3.attr(0).dtype, np.float32)
        self.assertEqual(schema3.domain.dim(0).tile, 4)
        schema3 = schema_like(shape=(4,4), dtype=np.float32, tile=1)
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
        with tiledb.save(uri, arr) as tmp:
            with tiledb.open(uri) as T:
                assert_array_equal(arr, T)

    def test_array_exists(self):
        import tempfile
        with tempfile.NamedTemporaryFile() as tmpfn:
            self.assertFalse(tiledb.array_exists(tmpfn.name))

        uri = self.path("test_array_exists_dense")
        with tiledb.from_numpy(uri, np.arange(0,5)) as T:
            self.assertTrue(tiledb.array_exists(uri))
            self.assertTrue(tiledb.array_exists(uri, isdense=True))
            self.assertFalse(tiledb.array_exists(uri, issparse=True))

        uri = self.path("test_array_exists_sparse")
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 3), tile=4, dtype=int, ctx=ctx), ctx=ctx)
        att = tiledb.Attr(dtype=int, ctx=ctx)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True, ctx=ctx)
        tiledb.Array.create(uri, schema)

        with tiledb.SparseArray(uri, mode='w') as T:
            T[[0,1]] = np.array([0,1])

        self.assertTrue(tiledb.array_exists(uri))
        self.assertTrue(tiledb.array_exists(uri, issparse=True))
        self.assertFalse(tiledb.array_exists(uri, isdense=True))
