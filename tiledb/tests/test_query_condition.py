import pytest

import numpy as np
from numpy.testing import assert_array_equal
import string

import tiledb
from tiledb import _query_condition
from tiledb.tests.common import DiskTestCase


class QueryConditionTest(DiskTestCase):
    def setUp(self):
        super().setUp()
        if not tiledb.libtiledb.version() >= (2, 2, 3):
            pytest.skip("Only run QueryCondition test with TileDB>=2.2.3")

    def test_errors(self):
        with self.assertRaises(tiledb.TileDBError):
            tiledb.QueryCondition("1.324 < 1")

        with self.assertRaises(tiledb.TileDBError):
            tiledb.QueryCondition("foo >= bar")

        with self.assertRaises(tiledb.TileDBError):
            tiledb.QueryCondition("'foo' == 'bar'")

    def test_ints_floats_bytestrings(self):
        ctx = tiledb.Ctx()
        path = self.path("test_ints_and_floats")

        dom = tiledb.Domain(
            tiledb.Dim(name="d", domain=(1, 10), tile=1, dtype=np.uint32, ctx=ctx),
            ctx=ctx,
        )
        attrs = [
            tiledb.Attr(name="U", dtype=np.uint32, ctx=ctx),
            tiledb.Attr(name="D", dtype=np.float64, ctx=ctx),
            tiledb.Attr(name="S", dtype=np.dtype("|S1"), var=False, ctx=ctx),
        ]

        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=True, ctx=ctx)
        tiledb.SparseArray.create(path, schema)

        U = np.arange(1, 11)
        D = np.arange(1, 11) / 10
        S = np.array(list(string.ascii_lowercase[:10]), dtype="|S1")

        coords = np.linspace(1, 10, num=10, dtype=np.uint32)
        data = {"U": U, "D": D, "S": S}

        with tiledb.open(path, "w") as A:
            A[coords] = data

        with tiledb.open(path) as A:
            # bytestrings with PyArrow not yet support in TileDB-Py
            qc = tiledb.QueryCondition("U < 5")
            result = A.query(attr_cond=qc, use_arrow=False).df[:]
            assert_array_equal(result["U"], np.arange(1, 5))

            qc = tiledb.QueryCondition("U >= 3 and 0.7 < D")
            result = A.query(attr_cond=qc, use_arrow=False).df[:]
            assert_array_equal(result["U"], np.arange(8, 11))
            assert_array_equal(result["D"], np.arange(8, 11) / 10)

            qc = tiledb.QueryCondition("S == 'c'")
            result = A.query(attr_cond=qc, use_arrow=False).df[:]
            assert len(result["S"]) == 1
            assert result["S"][0] == b"c"
