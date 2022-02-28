from cmath import atanh
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
        path = self.path("input_array_UIDS")

        dom = tiledb.Domain(
            tiledb.Dim(name="d", domain=(1, 10), tile=1, dtype=np.uint32)
        )
        attrs = [
            tiledb.Attr(name="U", dtype=np.uint32),
            tiledb.Attr(name="I", dtype=np.int64),
            tiledb.Attr(name="D", dtype=np.float64),
            tiledb.Attr(name="S", dtype=np.dtype("|S1"), var=False),
            tiledb.Attr(name="A", dtype="ascii", var=True),
        ]

        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=True)
        tiledb.Array.create(path, schema)

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
                A.query(attr_cond=qc, use_arrow=False)[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(input_array_UIDS) as A:
                qc = tiledb.QueryCondition("foo >= bar")
                A.query(attr_cond=qc, use_arrow=False)[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(input_array_UIDS) as A:
                qc = tiledb.QueryCondition("'foo' == 'bar'")
                A.query(attr_cond=qc, use_arrow=False)[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(input_array_UIDS) as A:
                qc = tiledb.QueryCondition("U < 10000000000000000000000.0")
                A.query(attr_cond=qc, attrs=["U"])[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(input_array_UIDS) as A:
                qc = tiledb.QueryCondition("D")
                A.query(attr_cond=qc, attrs=["D"])[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(input_array_UIDS) as A:
                qc = tiledb.QueryCondition("D,")
                A.query(attr_cond=qc, attrs=["D"])[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(input_array_UIDS) as A:
                qc = tiledb.QueryCondition("D > ")
                A.query(attr_cond=qc, attrs=["D"])[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(input_array_UIDS) as A:
                qc = tiledb.QueryCondition("(D > 0.7) | (D < 3.5)")
                A.query(attr_cond=qc, attrs=["D"])[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(input_array_UIDS) as A:
                qc = tiledb.QueryCondition("U >= 3 or 0.7 < D")
                A.query(attr_cond=qc, attrs=["U", "D"])[:]

    @pytest.mark.xfail(
        tiledb.libtiledb.version() >= (2, 5),
        reason="Skip fail_on_dense with libtiledb >2.5",
    )
    def test_fail_on_dense(self):
        path = self.path("test_fail_on_dense")

        dom = tiledb.Domain(
            tiledb.Dim(name="d", domain=(1, 10), tile=1, dtype=np.uint8)
        )
        attrs = [tiledb.Attr(name="a", dtype=np.uint8)]
        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False)
        tiledb.Array.create(path, schema)

        with tiledb.open(path) as A:
            with pytest.raises(tiledb.TileDBError) as excinfo:
                A.query(attr_cond=tiledb.QueryCondition("a < 5"))
            assert "QueryConditions may only be applied to sparse arrays" in str(
                excinfo.value
            )

    def test_unsigned(self, input_array_UIDS):
        with tiledb.open(input_array_UIDS) as A:
            qc = tiledb.QueryCondition("U < 5")
            result = A.query(attr_cond=qc, attrs=["U"])[:]
            assert all(result["U"] < 5)

    def test_signed(self, input_array_UIDS):
        with tiledb.open(input_array_UIDS) as A:
            qc = tiledb.QueryCondition("I < 1")
            result = A.query(attr_cond=qc, attrs=["I"])[:]
            assert all(result["I"] < 1)

            qc = tiledb.QueryCondition("I < +1")
            result = A.query(attr_cond=qc, attrs=["I"])[:]
            assert all(result["I"] < +1)

            qc = tiledb.QueryCondition("I < ---1")
            result = A.query(attr_cond=qc, attrs=["I"])[:]
            assert all(result["I"] < ---1)

            qc = tiledb.QueryCondition("-5 < I < 5")
            result = A.query(attr_cond=qc, attrs=["I"])[:]
            assert all(-5 < result["I"])
            assert all(result["I"] < 5)

    def test_floats(self, input_array_UIDS):
        with tiledb.open(input_array_UIDS) as A:
            qc = tiledb.QueryCondition("D > 5.0")
            result = A.query(attr_cond=qc, attrs=["D"])[:]
            assert all(result["D"] > 5.0)

            qc = tiledb.QueryCondition("(D > 0.7) & (D < 3.5)")
            result = A.query(attr_cond=qc, attrs=["D"])[:]
            assert all((result["D"] > 0.7) & (result["D"] < 3.5))

            qc = tiledb.QueryCondition("0.2 < D < 0.75")
            result = A.query(attr_cond=qc, attrs=["D", "I"])[:]
            assert all(0.2 < result["D"])
            assert all(result["D"] < 0.75)

    def test_string(self, input_array_UIDS):
        with tiledb.open(input_array_UIDS) as A:
            qc = tiledb.QueryCondition("S == 'c'")
            result = A.query(attr_cond=qc, attrs=["S"], use_arrow=False)[:]
            assert len(result["S"]) == 1
            assert result["S"][0] == b"c"

            qc = tiledb.QueryCondition("A == 'a'")
            result = A.query(attr_cond=qc, attrs=["A"], use_arrow=False)[:]
            assert len(result["A"]) == 1
            assert result["A"][0] == b"a"

    def test_combined_types(self, input_array_UIDS):
        with tiledb.open(input_array_UIDS) as A:
            qc = tiledb.QueryCondition("(I > 0) & ((-3 < D) & (D < 3.0))")
            result = A.query(attr_cond=qc, attrs=["I", "D"])[:]
            assert all((result["I"] > 0) & ((-3 < result["D"]) & (result["D"] < 3.0)))

            qc = tiledb.QueryCondition("U >= 3 and 0.7 < D")
            result = A.query(attr_cond=qc, attrs=["U", "D"])[:]
            assert all(result["U"] >= 3)
            assert all(0.7 < result["D"])

            qc = tiledb.QueryCondition("(0.2 < D and D < 0.75) and (-5 < I < 5)")
            result = A.query(attr_cond=qc, attrs=["D", "I"])[:]
            assert all((0.2 < result["D"]) & (result["D"] < 0.75))
            assert all((-5 < result["I"]) & (result["I"] < 5))

            qc = tiledb.QueryCondition("(-5 < I <= -1) and (0.2 < D < 0.75)")
            result = A.query(attr_cond=qc, attrs=["D", "I"])[:]
            assert all((0.2 < result["D"]) & (result["D"] < 0.75))
            assert all((-5 < result["I"]) & (result["I"] <= -1))

            qc = tiledb.QueryCondition("(0.2 < D < 0.75) and (-5 < I < 5)")
            result = A.query(attr_cond=qc, attrs=["D", "I"])[:]
            assert all((0.2 < result["D"]) & (result["D"] < 0.75))
            assert all((-5 < result["I"]) & (result["I"] < 5))

    def test_check_attrs(self, input_array_UIDS):
        with tiledb.open(input_array_UIDS) as A:
            qc = tiledb.QueryCondition("U < 0.1")
            result = A.query(attr_cond=qc, attrs=["U"])[:]
            assert all(result["U"] < 0.1)

            qc = tiledb.QueryCondition("U < 1.0")
            result = A.query(attr_cond=qc, attrs=["U"])[:]
            assert all(result["U"] < 1.0)

            with self.assertRaises(tiledb.TileDBError):
                qc = tiledb.QueryCondition("U < '1'")
                A.query(attr_cond=qc, attrs=["U"])[:]

            with self.assertRaises(tiledb.TileDBError):
                qc = tiledb.QueryCondition("U < 'one'")
                A.query(attr_cond=qc, attrs=["U"])[:]

            with self.assertRaises(tiledb.TileDBError):
                qc = tiledb.QueryCondition("U < 1")
                A.query(attr_cond=qc, attrs=["D"])[:]

    def test_error_when_using_dim(self, input_array_UIDS):
        with tiledb.open(input_array_UIDS) as A:
            with pytest.raises(tiledb.TileDBError) as excinfo:
                qc = tiledb.QueryCondition("d < 5")
                A.query(attr_cond=qc)[:]
            assert (
                "`d` is a dimension. QueryConditions currently only work on attributes."
                in str(excinfo.value)
            )

    def test_attr_and_val_casting_num(self):
        path = self.path("test_attr_and_val_casting_num")

        dom = tiledb.Domain(
            tiledb.Dim(name="dim", domain=(1, 10), tile=1, dtype=np.uint32)
        )
        attrs = [
            tiledb.Attr(name="64-bit integer", dtype=np.int64),
            tiledb.Attr(name="double", dtype=np.float64),
        ]

        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=True)
        tiledb.Array.create(path, schema)

        I = np.random.randint(-5, 5, 10)
        D = np.random.rand(10)

        with tiledb.open(path, "w") as arr:
            arr[np.arange(1, 11)] = {"64-bit integer": I, "double": D}

        with tiledb.open(path) as arr:
            qc = tiledb.QueryCondition("attr('64-bit integer') <= val(0)")
            result = arr.query(attr_cond=qc)[:]
            assert all(result["64-bit integer"] <= 0)

            qc = tiledb.QueryCondition("attr('64-bit integer') <= 0")
            result = arr.query(attr_cond=qc)[:]
            assert all(result["64-bit integer"] <= 0)

            qc = tiledb.QueryCondition("double <= 0.5")
            result = arr.query(attr_cond=qc)[:]
            assert all(result["double"] <= 0.5)

            qc = tiledb.QueryCondition("attr('double') <= 0.5")
            result = arr.query(attr_cond=qc)[:]
            assert all(result["double"] <= 0.5)

            qc = tiledb.QueryCondition("double <= val(0.5)")
            result = arr.query(attr_cond=qc)[:]
            assert all(result["double"] <= 0.5)

            qc = tiledb.QueryCondition("attr('double') <= val(0.5)")
            result = arr.query(attr_cond=qc)[:]
            assert all(result["double"] <= 0.5)

    def test_attr_and_val_casting_str(self):
        path = self.path("test_attr_and_val_casting_str")

        dom = tiledb.Domain(tiledb.Dim(name="dim", dtype="ascii"))
        attrs = [tiledb.Attr(name="attr with spaces", dtype="ascii", var=True)]

        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=True)
        tiledb.Array.create(path, schema)

        A = np.array(
            [
                "value with spaces",
                "nospaces",
                "value with spaces",
                "another value",
                "",
            ]
        )

        with tiledb.open(path, "w") as arr:
            arr[["a", "b", "c", "d", "e"]] = {"attr with spaces": A}

        with tiledb.open(path) as arr:
            qc = tiledb.QueryCondition(
                "attr('attr with spaces') == 'value with spaces'"
            )
            result = arr.query(attr_cond=qc, use_arrow=False)[:]
            assert list(result["dim"]) == [b"a", b"c"]

            qc = tiledb.QueryCondition(
                "attr('attr with spaces') == val('value with spaces')"
            )
            result = arr.query(attr_cond=qc, use_arrow=False)[:]
            assert list(result["dim"]) == [b"a", b"c"]

    @pytest.mark.skipif(
        tiledb.libtiledb.version() < (2, 7, 0),
        reason="var-length np.bytes_ query condition support introduced in 2.7.0",
    )
    def test_var_length_str(self):
        path = self.path("test_var_length_str")

        dom = tiledb.Domain(tiledb.Dim(name="d", domain=(0, 4)))
        attrs = [
            tiledb.Attr(name="ascii", dtype="ascii", var=True),
            tiledb.Attr(name="bytes", dtype=np.bytes_, var=True),
        ]

        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=True)
        tiledb.Array.create(path, schema)

        create_array = lambda func: np.array(
            [func[i - 1] * i for i in range(1, 6)], dtype=np.bytes_
        )

        ascii_data = create_array(string.ascii_lowercase)
        bytes_data = create_array(string.ascii_uppercase)

        with tiledb.open(path, "w") as arr:
            arr[np.arange(5)] = {"ascii": ascii_data, "bytes": bytes_data}

        with tiledb.open(path, "r") as arr:
            for s in ascii_data:
                qc = tiledb.QueryCondition(f"ascii == '{s.decode()}'")
                result = arr.query(attr_cond=qc, use_arrow=False)[:]
                assert result["ascii"][0] == s

            for s in bytes_data:
                qc = tiledb.QueryCondition(f"bytes == '{s.decode()}'")
                result = arr.query(attr_cond=qc, use_arrow=False)[:]
                assert result["bytes"][0] == s
