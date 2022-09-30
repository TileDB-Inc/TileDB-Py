import pytest

import numpy as np
from numpy.testing import assert_array_equal
import string

import tiledb

from tiledb.tests.common import DiskTestCase, has_pandas


class QueryConditionTest(DiskTestCase):
    def filter_dense(self, data, mask):
        if isinstance(mask, np.ndarray):
            mask = mask[0]

        if isinstance(mask, float):
            return data[np.invert(np.isnan(data))]

        if isinstance(mask, np.timedelta64):
            return data[np.invert(np.isnat(data))]

        return data[data != mask]

    def create_input_array_UIDSA(self, sparse):
        path = self.path("input_array_UIDSA")

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

        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=sparse)
        tiledb.Array.create(path, schema)

        with tiledb.open(path, "w") as arr:
            data = {
                "U": np.random.randint(1, 10, 10),
                "I": np.random.randint(-5, 5, 10),
                "D": np.random.rand(10),
                "S": np.array(list(string.ascii_lowercase[:10]), dtype="|S1"),
                "A": np.array(list(string.ascii_lowercase[:10]), dtype="|S1"),
            }

            if sparse:
                arr[np.arange(1, 11)] = data
            else:
                arr[:] = data

        return path

    def setUp(self):
        super().setUp()
        if not tiledb.libtiledb.version() >= (2, 2, 3):
            pytest.skip("Only run QueryCondition test with TileDB>=2.2.3")

    @pytest.mark.parametrize("sparse", [True, False])
    def test_errors(self, sparse):
        uri = self.create_input_array_UIDSA(sparse)

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(uri) as A:
                qc = tiledb.QueryCondition("1.324 < 1")
                A.query(attr_cond=qc)[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(uri) as A:
                qc = tiledb.QueryCondition("foo >= bar")
                A.query(attr_cond=qc)[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(uri) as A:
                qc = tiledb.QueryCondition("'foo' == 'bar'")
                A.query(attr_cond=qc)[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(uri) as A:
                qc = tiledb.QueryCondition("U < 10000000000000000000000.0")
                A.query(attr_cond=qc, attrs=["U"])[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(uri) as A:
                qc = tiledb.QueryCondition("D")
                A.query(attr_cond=qc, attrs=["D"])[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(uri) as A:
                qc = tiledb.QueryCondition("D,")
                A.query(attr_cond=qc, attrs=["D"])[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(uri) as A:
                qc = tiledb.QueryCondition("D > ")
                A.query(attr_cond=qc, attrs=["D"])[:]

    def test_qc_dense(self):
        path = self.path("test_qc_dense")

        dom = tiledb.Domain(
            tiledb.Dim(name="d", domain=(1, 10), tile=1, dtype=np.uint8)
        )
        attrs = [tiledb.Attr(name="a", dtype=np.uint8)]
        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False)
        tiledb.Array.create(path, schema)

        with tiledb.open(path) as A:
            A.query(attr_cond=tiledb.QueryCondition("a < 5"))

    def test_unsigned_sparse(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=True)) as A:
            qc = tiledb.QueryCondition("U < 5")
            result = A.query(attr_cond=qc, attrs=["U"])[:]
            assert all(result["U"] < 5)

    def test_unsigned_dense(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=False)) as A:
            mask = A.attr("U").fill

            qc = tiledb.QueryCondition("U < 5")
            result = A.query(attr_cond=qc, attrs=["U"])[:]
            assert all(self.filter_dense(result["U"], mask) < 5)

    def test_signed_sparse(self):
        uri = self.create_input_array_UIDSA(sparse=True)

        with tiledb.open(uri) as A:
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

    def test_signed_dense(self):
        uri = self.create_input_array_UIDSA(sparse=False)

        with tiledb.open(uri) as A:
            mask = A.attr("I").fill

            qc = tiledb.QueryCondition("I < 1")
            result = A.query(attr_cond=qc, attrs=["I"])[:]
            assert all(self.filter_dense(result["I"], mask) < 1)

            qc = tiledb.QueryCondition("I < +1")
            result = A.query(attr_cond=qc, attrs=["I"])[:]
            assert all(self.filter_dense(result["I"], mask) < +1)

            qc = tiledb.QueryCondition("I < ---1")
            result = A.query(attr_cond=qc, attrs=["I"])[:]
            assert all(self.filter_dense(result["I"], mask) < ---1)

            qc = tiledb.QueryCondition("-5 < I < 5")
            result = A.query(attr_cond=qc, attrs=["I"])[:]
            assert all(-5 < self.filter_dense(result["I"], mask))
            assert all(self.filter_dense(result["I"], mask) < 5)

    def test_floats_sparse(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=True)) as A:
            qc = tiledb.QueryCondition("D > 5.0")
            result = A.query(attr_cond=qc, attrs=["D"])[:]
            assert all(result["D"] > 5.0)

            qc = tiledb.QueryCondition("(D > 0.7) & (D < 3.5)")
            result = A.query(attr_cond=qc, attrs=["D"])[:]
            assert all((result["D"] > 0.7) & (result["D"] < 3.5))

            qc = tiledb.QueryCondition("0.2 < D < 0.75")
            result = A.query(attr_cond=qc, attrs=["D", "D"])[:]
            assert all(0.2 < result["D"])
            assert all(result["D"] < 0.75)

    def test_floats_dense(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=False)) as A:
            mask = A.attr("D").fill

            qc = tiledb.QueryCondition("D > 5.0")
            result = A.query(attr_cond=qc, attrs=["D"])[:]
            assert all(self.filter_dense(result["D"], mask) > 5.0)

            qc = tiledb.QueryCondition("(D > 0.7) & (D < 3.5)")
            result = A.query(attr_cond=qc, attrs=["D"])[:]
            assert all(self.filter_dense(result["D"], mask) > 0.7)
            assert all(self.filter_dense(result["D"], mask) < 3.5)

            qc = tiledb.QueryCondition("0.2 < D < 0.75")
            result = A.query(attr_cond=qc, attrs=["D", "D"])[:]
            assert all(0.2 < self.filter_dense(result["D"], mask))
            assert all(self.filter_dense(result["D"], mask) < 0.75)

    def test_string_sparse(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=True)) as A:
            qc = tiledb.QueryCondition("S == 'c'")
            result = A.query(attr_cond=qc, attrs=["S"])[:]
            assert len(result["S"]) == 1
            assert result["S"][0] == b"c"

            qc = tiledb.QueryCondition("A == 'a'")
            result = A.query(attr_cond=qc, attrs=["A"])[:]
            assert len(result["A"]) == 1
            assert result["A"][0] == b"a"

    def test_string_dense(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=False)) as A:
            qc = tiledb.QueryCondition("S == 'c'")
            result = A.query(attr_cond=qc, attrs=["S"])[:]
            assert all(self.filter_dense(result["S"], A.attr("S").fill) == b"c")

            qc = tiledb.QueryCondition("A == 'a'")
            result = A.query(attr_cond=qc, attrs=["A"])[:]
            assert all(self.filter_dense(result["A"], A.attr("A").fill) == b"a")

    def test_combined_types_sparse(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=True)) as A:
            qc = tiledb.QueryCondition("(I > 0) & ((-3 < D) & (D < 3.0))")
            result = A.query(attr_cond=qc, attrs=["I", "D"])[:]
            assert all((result["I"] > 0) & ((-3 < result["D"]) & (result["D"] < 3.0)))

            qc = tiledb.QueryCondition("U >= 3 and 0.7 < D")
            result = A.query(attr_cond=qc, attrs=["U", "D"])[:]
            assert all(result["U"] >= 3) & all(0.7 < result["D"])

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

    def test_combined_types_dense(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=False)) as A:
            mask_U = A.attr("U").fill
            mask_I = A.attr("I").fill
            mask_D = A.attr("D").fill

            qc = tiledb.QueryCondition("(I > 0) & ((-3 < D) & (D < 3.0))")
            result = A.query(attr_cond=qc, attrs=["I", "D"])[:]
            res_I = self.filter_dense(result["I"], mask_I)
            res_D = self.filter_dense(result["D"], mask_D)
            assert all(res_I > 0) & all(-3 < res_D) & all(res_D < 3.0)

            qc = tiledb.QueryCondition("U >= 3 and 0.7 < D")
            result = A.query(attr_cond=qc, attrs=["U", "D"])[:]
            res_U = self.filter_dense(result["U"], mask_U)
            res_D = self.filter_dense(result["D"], mask_D)
            assert all(res_U >= 3) & all(0.7 < res_D)

            qc = tiledb.QueryCondition("(0.2 < D and D < 0.75) and (-5 < I < 5)")
            result = A.query(attr_cond=qc, attrs=["D", "I"])[:]
            res_D = self.filter_dense(result["D"], mask_D)
            res_I = self.filter_dense(result["I"], mask_I)
            assert all((0.2 < res_D) & (res_D < 0.75))
            assert all((-5 < res_I) & (res_I < 5))

            qc = tiledb.QueryCondition("(-5 < I <= -1) and (0.2 < D < 0.75)")
            result = A.query(attr_cond=qc, attrs=["D", "I"])[:]
            res_D = self.filter_dense(result["D"], mask_D)
            res_I = self.filter_dense(result["I"], mask_I)
            assert all((0.2 < res_D) & (res_D < 0.75))
            assert all((-5 < res_I) & (res_I <= -1))

            qc = tiledb.QueryCondition("(0.2 < D < 0.75) and (-5 < I < 5)")
            result = A.query(attr_cond=qc, attrs=["D", "I"])[:]
            res_D = self.filter_dense(result["D"], mask_D)
            res_I = self.filter_dense(result["I"], mask_I)
            assert all((0.2 < res_D) & (res_D < 0.75))
            assert all((-5 < res_I) & (res_I < 5))

    def test_check_attrs_sparse(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=True)) as A:
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

    def test_check_attrs_dense(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=False)) as A:
            mask = A.attr("U").fill

            qc = tiledb.QueryCondition("U < 0.1")
            result = A.query(attr_cond=qc, attrs=["U"])[:]
            assert all(self.filter_dense(result["U"], mask) < 0.1)

            qc = tiledb.QueryCondition("U < 1.0")
            result = A.query(attr_cond=qc, attrs=["U"])[:]
            assert all(self.filter_dense(result["U"], mask) < 1.0)

            with self.assertRaises(tiledb.TileDBError):
                qc = tiledb.QueryCondition("U < '1'")
                A.query(attr_cond=qc, attrs=["U"])[:]

            with self.assertRaises(tiledb.TileDBError):
                qc = tiledb.QueryCondition("U < 'one'")
                A.query(attr_cond=qc, attrs=["U"])[:]

    @pytest.mark.parametrize("sparse", [True, False])
    def test_error_when_using_dim(self, sparse):
        with tiledb.open(self.create_input_array_UIDSA(sparse)) as A:
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
            result = arr.query(attr_cond=qc)[:]
            assert list(result["dim"]) == [b"a", b"c"]

            qc = tiledb.QueryCondition(
                "attr('attr with spaces') == val('value with spaces')"
            )
            result = arr.query(attr_cond=qc)[:]
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
                result = arr.query(attr_cond=qc)[:]
                assert result["ascii"][0] == s

            for s in bytes_data:
                qc = tiledb.QueryCondition(f"bytes == '{s.decode()}'")
                result = arr.query(attr_cond=qc)[:]
                assert result["bytes"][0] == s

    @pytest.mark.skipif(
        tiledb.libtiledb.version() < (2, 10, 0),
        reason="OR query condition operator introduced in libtiledb 2.10",
    )
    def test_or_sparse(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=True)) as A:
            qc = tiledb.QueryCondition("(D < 0.25) | (D > 0.75)")
            result = A.query(attr_cond=qc, attrs=["D"])[:]
            assert all((result["D"] < 0.25) | (result["D"] > 0.75))

            qc = tiledb.QueryCondition("(D < 0.25) or (D > 0.75)")
            result = A.query(attr_cond=qc, attrs=["D"])[:]
            assert all((result["D"] < 0.25) | (result["D"] > 0.75))

    @pytest.mark.skipif(
        tiledb.libtiledb.version() < (2, 10, 0),
        reason="OR query condition operator introduced in libtiledb 2.10",
    )
    def test_or_dense(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=False)) as A:
            mask = A.attr("D").fill

            qc = tiledb.QueryCondition("(D < 0.25) | (D > 0.75)")
            result = A.query(attr_cond=qc, attrs=["D"])[:]
            res = self.filter_dense(result["D"], mask)
            assert all((res < 0.25) | (res > 0.75))

            qc = tiledb.QueryCondition("(D < 0.25) or (D > 0.75)")
            result = A.query(attr_cond=qc, attrs=["D"])[:]
            res = self.filter_dense(result["D"], mask)
            assert all((res < 0.25) | (res > 0.75))

    @pytest.mark.skipif(
        tiledb.libtiledb.version() < (2, 10, 0),
        reason="OR query condition operator and bool type introduced in libtiledb 2.10",
    )
    def test_01(self):
        path = self.path("test_01")

        dom = tiledb.Domain(tiledb.Dim(domain=(1, 10), tile=1, dtype=np.uint32))
        attrs = [
            tiledb.Attr(name="a", dtype=np.uint8),
            tiledb.Attr(name="b", dtype=np.uint8),
            tiledb.Attr(name="c", dtype=np.uint8),
        ]
        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=True)
        tiledb.Array.create(path, schema)

        with tiledb.open(path, "w") as arr:
            arr[np.arange(1, 11)] = {
                "a": np.random.randint(0, high=2, size=10),
                "b": np.random.randint(0, high=2, size=10),
                "c": np.random.randint(0, high=2, size=10),
            }

        with tiledb.open(path) as A:
            qc = tiledb.QueryCondition("a == 1 and b == 1 and c == 1")
            result = A.query(attr_cond=qc)[:]
            assert all(result["a"] & result["b"] & result["c"])

            qc = tiledb.QueryCondition("a == 1 and b == 1 or c == 1")
            result = A.query(attr_cond=qc)[:]
            assert all(result["a"] & result["b"] | result["c"])

            qc = tiledb.QueryCondition("a == 1 or b == 1 and c == 1")
            result = A.query(attr_cond=qc)[:]
            assert all(result["a"] | result["b"] & result["c"])

            qc = tiledb.QueryCondition("a == 1 or b == 1 or c == 1")
            result = A.query(attr_cond=qc)[:]
            assert all(result["a"] | result["b"] | result["c"])

            qc = tiledb.QueryCondition("(a == 1 and b == 1) or c == 1")
            result = A.query(attr_cond=qc)[:]
            assert all(result["a"] & result["b"] | result["c"])

            qc = tiledb.QueryCondition("a == 1 and (b == 1 or c == 1)")
            result = A.query(attr_cond=qc)[:]
            assert all(result["a"] & (result["b"] | result["c"]))

            qc = tiledb.QueryCondition("(a == 1 or b == 1) and c == 1")
            result = A.query(attr_cond=qc)[:]
            assert all((result["a"] | result["b"]) & result["c"])

            qc = tiledb.QueryCondition("a == 1 or (b == 1 and c == 1)")
            result = A.query(attr_cond=qc)[:]
            assert all(result["a"] | result["b"] & result["c"])

    def test_in_operator_sparse(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=True)) as A:
            qc = tiledb.QueryCondition("U in [1, 2, 3]")
            result = A.query(attr_cond=qc, attrs=["U"])[:]
            for val in result["U"]:
                assert val in [1, 2, 3]

            qc = tiledb.QueryCondition("S in ['a', 'e', 'i', 'o', 'u']")
            result = A.query(attr_cond=qc, attrs=["S"])[:]
            for val in result["S"]:
                assert val in [b"a", b"e", b"i", b"o", b"u"]

            qc = tiledb.QueryCondition(
                "S in ['a', 'e', 'i', 'o', 'u'] and U in [5, 6, 7]"
            )
            result = A.query(attr_cond=qc)[:]
            for val in result["U"]:
                assert val in [5, 6, 7]
            for val in result["S"]:
                assert val in [b"a", b"e", b"i", b"o", b"u"]

            result = A.query(attr_cond=tiledb.QueryCondition("U in [8]"))[:]
            for val in result["U"]:
                assert val == 8

            result = A.query(attr_cond=tiledb.QueryCondition("S in ['8']"))[:]
            assert len(result["S"]) == 0

    def test_in_operator_dense(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=False)) as A:
            U_mask = A.attr("U").fill
            S_mask = A.attr("S").fill

            qc = tiledb.QueryCondition("U in [1, 2, 3]")
            result = A.query(attr_cond=qc, attrs=["U"])[:]
            for val in self.filter_dense(result["U"], U_mask):
                assert val in [1, 2, 3]

            qc = tiledb.QueryCondition("S in ['a', 'e', 'i', 'o', 'u']")
            result = A.query(attr_cond=qc, attrs=["S"])[:]
            for val in self.filter_dense(result["S"], S_mask):
                assert val in [b"a", b"e", b"i", b"o", b"u"]

            qc = tiledb.QueryCondition(
                "S in ['a', 'e', 'i', 'o', 'u'] and U in [5, 6, 7]"
            )
            result = A.query(attr_cond=qc)[:]
            for val in self.filter_dense(result["U"], U_mask):
                assert val in [5, 6, 7]
            for val in self.filter_dense(result["S"], S_mask):
                assert val in [b"a", b"e", b"i", b"o", b"u"]

            result = A.query(attr_cond=tiledb.QueryCondition("U in [8]"))[:]
            for val in self.filter_dense(result["U"], U_mask):
                assert val == 8

            result = A.query(attr_cond=tiledb.QueryCondition("S in ['8']"))[:]
            assert len(self.filter_dense(result["S"], S_mask)) == 0

    @pytest.mark.skipif(not has_pandas(), reason="pandas not installed")
    def test_dense_datetime(self):
        import pandas as pd

        uri = self.path("query-filter-dense-datetime.tdb")

        data = pd.DataFrame(
            np.random.randint(438923600, 243892360000, 20, dtype=np.int64),
            columns=["dates"],
        )

        tiledb.from_pandas(
            uri,
            data,
            column_types={"dates": "datetime64[ns]"},
        )

        with tiledb.open(uri) as A:
            idx = 5

            dt_mask = A.attr("dates").fill
            search_date = data["dates"][idx]

            qc = tiledb.QueryCondition(f"dates == {search_date}")
            result = A.query(attr_cond=qc).df[:]

            assert all(self.filter_dense(result["dates"], dt_mask) == A[idx]["dates"])

    def test_array_with_bool_but_unused(self):
        path = self.path("test_array_with_bool_but_unused")

        dom = tiledb.Domain(
            tiledb.Dim(name="d", domain=(1, 3), tile=1, dtype=np.uint32)
        )
        attrs = [
            tiledb.Attr(name="myint", dtype=int),
            tiledb.Attr(name="mystr", dtype=str),
            tiledb.Attr(name="mybool", dtype=bool),
        ]

        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=True)
        tiledb.Array.create(path, schema)

        data = {
            "myint": np.asarray([10, 20, 30]),
            "mystr": np.asarray(["apple", "ball", "cat"]),
            "mybool": np.asarray([True, False, True]),
        }

        with tiledb.open(path, "w") as A:
            A[np.arange(1, 4)] = data

        with tiledb.open(path) as A:
            qc = tiledb.QueryCondition("myint > 10")
            result = A.query(attr_cond=qc, attrs=["myint"])[:]
            assert all(result["myint"] > 10)

    def test_do_not_return_queried_attr(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=True)) as A:
            qc = tiledb.QueryCondition("U < 3")

            i_result = A.query(attr_cond=qc, attrs=["I", "U"])[:]
            assert "I" in i_result.keys()
            assert "U" in i_result.keys()
            assert all(i_result["U"] < 5)

            u_result = A.query(attr_cond=qc, attrs=["I"])[:]
            assert "I" in u_result.keys()
            assert "U" not in u_result.keys()
            assert_array_equal(i_result["I"], u_result["I"])
