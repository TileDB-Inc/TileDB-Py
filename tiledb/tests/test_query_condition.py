import pytest

import numpy as np
from numpy.testing import assert_array_equal
import string

import tiledb
from tiledb import _query_condition
from tiledb.tests.common import DiskTestCase


class QueryConditionTest(DiskTestCase):
    @pytest.fixture
    def input_array_UIDS(self):
        ctx = tiledb.Ctx()
        path = self.path("input_array_UIDS")

        dom = tiledb.Domain(
            tiledb.Dim(name="d", domain=(1, 10), tile=1, dtype=np.uint32, ctx=ctx),
            ctx=ctx,
        )
        attrs = [
            tiledb.Attr(name="U", dtype=np.uint32, ctx=ctx),
            tiledb.Attr(name="I", dtype=np.int64, ctx=ctx),
            tiledb.Attr(name="D", dtype=np.float64, ctx=ctx),
            tiledb.Attr(name="S", dtype=np.dtype("|S1"), var=False, ctx=ctx),
        ]

        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=True, ctx=ctx)
        tiledb.SparseArray.create(path, schema)

        U = np.random.randint(1, 10, 10)
        I = np.random.randint(-5, 5, 10)
        D = np.random.rand(10)
        S = np.array(list(string.ascii_lowercase[:10]), dtype="|S1")

        coords = np.linspace(1, 10, num=10, dtype=np.uint32)
        data = {"U": U, "I": I, "D": D, "S": S}

        with tiledb.open(path, "w") as A:
            A[coords] = data

        return path

    def setUp(self):
        super().setUp()
        if not tiledb.libtiledb.version() >= (2, 2, 3):
            pytest.skip("Only run QueryCondition test with TileDB>=2.2.3")

    def test_errors(self, input_array_UIDS):
        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(input_array_UIDS) as A:
                qc = tiledb.QueryCondition("1.324 < 1")
                A.query(attr_cond=qc, use_arrow=False).df[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(input_array_UIDS) as A:
                qc = tiledb.QueryCondition("foo >= bar")
                A.query(attr_cond=qc, use_arrow=False).df[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(input_array_UIDS) as A:
                qc = tiledb.QueryCondition("'foo' == 'bar'")
                A.query(attr_cond=qc, use_arrow=False).df[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(input_array_UIDS) as A:
                qc = tiledb.QueryCondition("NO_CONDITION")
                A.query(attr_cond=qc, use_arrow=False).df[:]

    def test_ints_floats_bytestrings(self, input_array_UIDS):
        with tiledb.open(input_array_UIDS) as A:
            # bytestrings with PyArrow not yet support in TileDB-Py
            qc = tiledb.QueryCondition("U < 5")
            result = A.query(attr_cond=qc, attrs=["U"]).df[:]
            assert all(result["U"] < 5)

            qc = tiledb.QueryCondition("I < 1")
            result = A.query(attr_cond=qc, attrs=["I"]).df[:]
            assert all(result["I"] < 1)

            qc = tiledb.QueryCondition("I < +1")
            result = A.query(attr_cond=qc, attrs=["I"]).df[:]
            assert all(result["I"] < +1)

            qc = tiledb.QueryCondition("I < ---1")
            result = A.query(attr_cond=qc, attrs=["I"]).df[:]
            assert all(result["I"] < ---1)

            qc = tiledb.QueryCondition("D > 5.0")
            result = A.query(attr_cond=qc, attrs=["D"]).df[:]
            assert all(result["D"] > 5.0)

            qc = tiledb.QueryCondition("U >= 3 and 0.7 < D")
            result = A.query(attr_cond=qc, attrs=["U", "D"]).df[:]
            assert all(result["U"] >= 3)
            assert all(0.7 < result["D"])

            qc = tiledb.QueryCondition("S == 'c'")
            result = A.query(attr_cond=qc, attrs=["S"], use_arrow=False).df[:]
            assert len(result["S"]) == 1
            assert result["S"][0] == b"c"

    def test_check_attrs(self, input_array_UIDS):
        with tiledb.open(input_array_UIDS) as A:
            qc = tiledb.QueryCondition("U < 0.1")
            result = A.query(attr_cond=qc, attrs=["U"]).df[:]
            assert all(result["U"] < 0.1)

            qc = tiledb.QueryCondition("U < 1.0")
            result = A.query(attr_cond=qc, attrs=["U"]).df[:]
            assert all(result["U"] < 1.0)

            with self.assertRaises(tiledb.TileDBError):
                qc = tiledb.QueryCondition("U < '1'")
                A.query(attr_cond=qc, attrs=["U"]).df[:]

            with self.assertRaises(tiledb.TileDBError):
                qc = tiledb.QueryCondition("U < 'one'")
                A.query(attr_cond=qc, attrs=["U"]).df[:]

            with self.assertRaises(tiledb.TileDBError):
                qc = tiledb.QueryCondition("U < 1")
                A.query(attr_cond=qc, attrs=["D"]).df[:]

            with self.assertRaises(tiledb.TileDBError):
                qc = tiledb.QueryCondition("U < 10000000000000000000000.0")
                A.query(attr_cond=qc, attrs=["U"]).df[:]
