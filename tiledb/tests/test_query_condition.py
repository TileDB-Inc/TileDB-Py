import pytest

import numpy as np
from numpy.testing import assert_array_equal
import string

import tiledb

# from tiledb.main import PyQueryCondition
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
            tiledb.Attr(name="A", dtype="ascii", var=True, ctx=ctx),
        ]

        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=True, ctx=ctx)
        tiledb.SparseArray.create(path, schema)

        U = np.random.randint(1, 10, 10)
        I = np.random.randint(-5, 5, 10)
        D = np.random.rand(10)
        S = np.array(list(string.ascii_lowercase[:10]), dtype="|S1")
        A = np.array(list(string.ascii_lowercase[:10]), dtype="|S1")

        with tiledb.open(path, "w") as arr:
            arr[np.arange(1, 11)] = {"U": U, "I": I, "D": D, "S": S, "A": A}

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
                qc = tiledb.QueryCondition("U < 10000000000000000000000.0")
                A.query(attr_cond=qc, attrs=["U"]).df[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(input_array_UIDS) as A:
                qc = tiledb.QueryCondition("D")
                A.query(attr_cond=qc, attrs=["D"]).df[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(input_array_UIDS) as A:
                qc = tiledb.QueryCondition("D,")
                A.query(attr_cond=qc, attrs=["D"]).df[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(input_array_UIDS) as A:
                qc = tiledb.QueryCondition("D > ")
                A.query(attr_cond=qc, attrs=["D"]).df[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(input_array_UIDS) as A:
                qc = tiledb.QueryCondition("(D > 0.7) | (D < 3.5)")
                A.query(attr_cond=qc, attrs=["D"]).df[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(input_array_UIDS) as A:
                qc = tiledb.QueryCondition("U >= 3 or 0.7 < D")
                A.query(attr_cond=qc, attrs=["U", "D"]).df[:]

    def test_unsigned(self, input_array_UIDS):
        with tiledb.open(input_array_UIDS) as A:
            qc = tiledb.QueryCondition("U < 5")
            result = A.query(attr_cond=qc, attrs=["U"]).df[:]
            assert all(result["U"] < 5)

    def test_signed(self, input_array_UIDS):
        with tiledb.open(input_array_UIDS) as A:
            qc = tiledb.QueryCondition("I < 1")
            result = A.query(attr_cond=qc, attrs=["I"]).df[:]
            assert all(result["I"] < 1)

            qc = tiledb.QueryCondition("I < +1")
            result = A.query(attr_cond=qc, attrs=["I"]).df[:]
            assert all(result["I"] < +1)

            qc = tiledb.QueryCondition("I < ---1")
            result = A.query(attr_cond=qc, attrs=["I"]).df[:]
            assert all(result["I"] < ---1)

            qc = tiledb.QueryCondition("-5 < I < 5")
            result = A.query(attr_cond=qc, attrs=["I"]).df[:]
            assert all(-5 < result["I"])
            assert all(result["I"] < 5)

    def test_floats(self, input_array_UIDS):
        with tiledb.open(input_array_UIDS) as A:
            qc = tiledb.QueryCondition("D > 5.0")
            result = A.query(attr_cond=qc, attrs=["D"]).df[:]
            assert all(result["D"] > 5.0)

            qc = tiledb.QueryCondition("(D > 0.7) & (D < 3.5)")
            result = A.query(attr_cond=qc, attrs=["D"]).df[:]
            assert all((result["D"] > 0.7) & (result["D"] < 3.5))

            qc = tiledb.QueryCondition("0.2 < D < 0.75")
            result = A.query(attr_cond=qc, attrs=["D", "I"]).df[:]
            assert all(0.2 < result["D"])
            assert all(result["D"] < 0.75)

    def test_string(self, input_array_UIDS):
        with tiledb.open(input_array_UIDS) as A:
            qc = tiledb.QueryCondition("S == 'c'")
            result = A.query(attr_cond=qc, attrs=["S"], use_arrow=False).df[:]
            assert len(result["S"]) == 1
            assert result["S"][0] == b"c"

            qc = tiledb.QueryCondition("A == 'a'")
            result = A.query(attr_cond=qc, attrs=["A"], use_arrow=False).df[:]
            assert len(result["A"]) == 1
            assert result["A"][0] == b"a"

    def test_combined_types(self, input_array_UIDS):
        with tiledb.open(input_array_UIDS) as A:
            qc = tiledb.QueryCondition("(I > 0) & ((-3 < D) & (D < 3.0))")
            result = A.query(attr_cond=qc, attrs=["I", "D"]).df[:]
            assert all((result["I"] > 0) & ((-3 < result["D"]) & (result["D"] < 3.0)))

            qc = tiledb.QueryCondition("U >= 3 and 0.7 < D")
            result = A.query(attr_cond=qc, attrs=["U", "D"]).df[:]
            assert all(result["U"] >= 3)
            assert all(0.7 < result["D"])

            qc = tiledb.QueryCondition("(0.2 < D and D < 0.75) and (-5 < I < 5)")
            result = A.query(attr_cond=qc, attrs=["D", "I"]).df[:]
            assert all((0.2 < result["D"]) & (result["D"] < 0.75))
            assert all((-5 < result["I"]) & (result["I"] < 5))

            qc = tiledb.QueryCondition("(-5 < I <= -1) and (0.2 < D < 0.75)")
            result = A.query(attr_cond=qc, attrs=["D", "I"]).df[:]
            assert all((0.2 < result["D"]) & (result["D"] < 0.75))
            assert all((-5 < result["I"]) & (result["I"] <= -1))

            qc = tiledb.QueryCondition("(0.2 < D < 0.75) and (-5 < I < 5)")
            result = A.query(attr_cond=qc, attrs=["D", "I"]).df[:]
            assert all((0.2 < result["D"]) & (result["D"] < 0.75))
            assert all((-5 < result["I"]) & (result["I"] < 5))

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
