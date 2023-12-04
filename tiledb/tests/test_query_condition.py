import string

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import tiledb

from .common import DiskTestCase, has_pandas, rand_utf8


class QueryConditionTest(DiskTestCase):
    def filter_dense(self, data, mask):
        if isinstance(mask, np.ndarray):
            mask = mask[0]

        if isinstance(mask, float):
            return data[np.invert(np.isnan(data))]

        if isinstance(mask, np.timedelta64):
            return data[np.invert(np.isnat(data))]

        if isinstance(mask, (str, bytes)):
            return data[np.invert(data == mask)]

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
            tiledb.Attr(name="UTF", dtype=np.dtype("U"), var=True),
        ]

        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=sparse)
        tiledb.Array.create(path, schema)

        with tiledb.open(path, "w") as arr:
            data = {
                "U": np.random.randint(1, 10, 10),
                "I": np.random.randint(-5, 5, 10),
                "D": np.random.rand(10),
                "S": np.array(list(string.ascii_lowercase[:10]), dtype="|S1"),
                "A": np.array(
                    list(string.ascii_lowercase[i] * (i + 1) for i in range(10)),
                    dtype="|S",
                ),
                "UTF": np.array(
                    ["$", "£$", "€ह£$", "한ह£", "£$𐍈"]
                    + [rand_utf8(np.random.randint(1, 100)) for _ in range(5)],
                    dtype="|U0",
                ),
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
                A.query(cond="1.324 < 1")[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(uri) as A:
                A.query(cond="foo >= bar")[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(uri) as A:
                A.query(cond="'foo' == 'bar'")[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(uri) as A:
                A.query(cond="U < 10000000000000000000000.0", attrs=["U"])[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(uri) as A:
                A.query(cond="D", attrs=["D"])[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(uri) as A:
                A.query(cond="D,", attrs=["D"])[:]

        with self.assertRaises(tiledb.TileDBError):
            with tiledb.open(uri) as A:
                A.query(cond="D > ", attrs=["D"])[:]

    def test_qc_dense(self):
        path = self.path("test_qc_dense")

        dom = tiledb.Domain(
            tiledb.Dim(name="d", domain=(1, 10), tile=1, dtype=np.uint8)
        )
        attrs = [tiledb.Attr(name="a", dtype=np.uint8)]
        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False)
        tiledb.Array.create(path, schema)

        with tiledb.open(path) as A:
            A.query(cond="a < 5")

    def test_unsigned_sparse(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=True)) as A:
            with pytest.raises(tiledb.TileDBError) as exc_info:
                A.query(cond=tiledb.QueryCondition("U < 5"), attrs=["U"])[:]
            assert (
                "Passing `tiledb.QueryCondition` to `cond` is no longer supported"
                in str(exc_info.value)
            )

            result = A.query(cond="U < 5", attrs=["U"])[:]
            assert all(result["U"] < 5)

    def test_unsigned_dense(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=False)) as A:
            mask = A.attr("U").fill

            with pytest.raises(tiledb.TileDBError) as exc_info:
                A.query(cond=tiledb.QueryCondition("U < 5"), attrs=["U"])[:]
            assert (
                "Passing `tiledb.QueryCondition` to `cond` is no longer supported"
                in str(exc_info.value)
            )

            result = A.query(cond="U < 5", attrs=["U"])[:]
            assert all(self.filter_dense(result["U"], mask) < 5)

    def test_signed_sparse(self):
        uri = self.create_input_array_UIDSA(sparse=True)

        with tiledb.open(uri) as A:
            result = A.query(cond="I < 1", attrs=["I"])[:]
            assert all(result["I"] < 1)

            result = A.query(cond="I < +1", attrs=["I"])[:]
            assert all(result["I"] < +1)

            result = A.query(cond="I < ---1", attrs=["I"])[:]
            assert all(result["I"] < ---1)

            result = A.query(cond="-5 < I < 5", attrs=["I"])[:]
            assert all(-5 < result["I"])
            assert all(result["I"] < 5)

    def test_signed_dense(self):
        uri = self.create_input_array_UIDSA(sparse=False)

        with tiledb.open(uri) as A:
            mask = A.attr("I").fill

            result = A.query(cond="I < 1", attrs=["I"])[:]
            assert all(self.filter_dense(result["I"], mask) < 1)

            result = A.query(cond="I < +1", attrs=["I"])[:]
            assert all(self.filter_dense(result["I"], mask) < +1)

            result = A.query(cond="I < ---1", attrs=["I"])[:]
            assert all(self.filter_dense(result["I"], mask) < ---1)

            result = A.query(cond="-5 < I < 5", attrs=["I"])[:]
            assert all(-5 < self.filter_dense(result["I"], mask))
            assert all(self.filter_dense(result["I"], mask) < 5)

    def test_floats_sparse(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=True)) as A:
            result = A.query(cond="D > 5.0", attrs=["D"])[:]
            assert all(result["D"] > 5.0)

            result = A.query(cond="(D > 0.7) & (D < 3.5)", attrs=["D"])[:]
            assert all((result["D"] > 0.7) & (result["D"] < 3.5))

            result = A.query(cond="0.2 < D < 0.75", attrs=["D", "D"])[:]
            assert all(0.2 < result["D"])
            assert all(result["D"] < 0.75)

    def test_floats_dense(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=False)) as A:
            mask = A.attr("D").fill

            result = A.query(cond="D > 5.0", attrs=["D"])[:]
            assert all(self.filter_dense(result["D"], mask) > 5.0)

            result = A.query(cond="(D > 0.7) & (D < 3.5)", attrs=["D"])[:]
            assert all(self.filter_dense(result["D"], mask) > 0.7)
            assert all(self.filter_dense(result["D"], mask) < 3.5)

            result = A.query(cond="0.2 < D < 0.75", attrs=["D", "D"])[:]
            assert all(0.2 < self.filter_dense(result["D"], mask))
            assert all(self.filter_dense(result["D"], mask) < 0.75)

    def test_string_sparse(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=True)) as A:
            result = A.query(cond="S == 'c'", attrs=["S"])[:]
            assert len(result["S"]) == 1
            assert result["S"][0] == b"c"

            result = A.query(cond="A == 'a'", attrs=["A"])[:]
            assert len(result["A"]) == 1
            assert result["A"][0] == b"a"

            if tiledb.libtiledb.version() > (2, 14):
                for t in A.query(attrs=["UTF"])[:]["UTF"]:
                    cond = f"""UTF == '{t}'"""
                    result = A.query(cond=cond, attrs=["UTF"])[:]
                    assert result["UTF"] == t

    def test_string_dense(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=False)) as A:
            result = A.query(cond="S == 'ccc'", attrs=["S"])[:]
            assert all(self.filter_dense(result["S"], A.attr("S").fill) == b"c")

            result = A.query(cond="A == 'ccc'", attrs=["A"])[:]
            assert all(self.filter_dense(result["A"], A.attr("A").fill) == b"ccc")

            if tiledb.libtiledb.version() > (2, 14):
                for t in A.query(attrs=["UTF"])[:]["UTF"]:
                    cond = f"""UTF == '{t}'"""
                    result = A.query(cond=cond, attrs=["UTF"])[:]
                    assert all(
                        self.filter_dense(result["UTF"], A.attr("UTF").fill) == t
                    )

    def test_combined_types_sparse(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=True)) as A:
            qc = "(I > 0) & ((-3 < D) & (D < 3.0))"
            result = A.query(cond=qc, attrs=["I", "D"])[:]
            assert all((result["I"] > 0) & ((-3 < result["D"]) & (result["D"] < 3.0)))

            qc = "U >= 3 and 0.7 < D"
            result = A.query(cond=qc, attrs=["U", "D"])[:]
            assert all(result["U"] >= 3) & all(0.7 < result["D"])

            qc = "(0.2 < D and D < 0.75) and (-5 < I < 5)"
            result = A.query(cond=qc, attrs=["D", "I"])[:]
            assert all((0.2 < result["D"]) & (result["D"] < 0.75))
            assert all((-5 < result["I"]) & (result["I"] < 5))

            qc = "(-5 < I <= -1) and (0.2 < D < 0.75)"
            result = A.query(cond=qc, attrs=["D", "I"])[:]
            assert all((0.2 < result["D"]) & (result["D"] < 0.75))
            assert all((-5 < result["I"]) & (result["I"] <= -1))

            qc = "(0.2 < D < 0.75) and (-5 < I < 5)"
            result = A.query(cond=qc, attrs=["D", "I"])[:]
            assert all((0.2 < result["D"]) & (result["D"] < 0.75))
            assert all((-5 < result["I"]) & (result["I"] < 5))

    def test_combined_types_dense(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=False)) as A:
            mask_U = A.attr("U").fill
            mask_I = A.attr("I").fill
            mask_D = A.attr("D").fill

            qc = "(I > 0) & ((-3 < D) & (D < 3.0))"
            result = A.query(cond=qc, attrs=["I", "D"])[:]
            res_I = self.filter_dense(result["I"], mask_I)
            res_D = self.filter_dense(result["D"], mask_D)
            assert all(res_I > 0) & all(-3 < res_D) & all(res_D < 3.0)

            qc = "U >= 3 and 0.7 < D"
            result = A.query(cond=qc, attrs=["U", "D"])[:]
            res_U = self.filter_dense(result["U"], mask_U)
            res_D = self.filter_dense(result["D"], mask_D)
            assert all(res_U >= 3) & all(0.7 < res_D)

            qc = "(0.2 < D and D < 0.75) and (-5 < I < 5)"
            result = A.query(cond=qc, attrs=["D", "I"])[:]
            res_D = self.filter_dense(result["D"], mask_D)
            res_I = self.filter_dense(result["I"], mask_I)
            assert all((0.2 < res_D) & (res_D < 0.75))
            assert all((-5 < res_I) & (res_I < 5))

            qc = "(-5 < I <= -1) and (0.2 < D < 0.75)"
            result = A.query(cond=qc, attrs=["D", "I"])[:]
            res_D = self.filter_dense(result["D"], mask_D)
            res_I = self.filter_dense(result["I"], mask_I)
            assert all((0.2 < res_D) & (res_D < 0.75))
            assert all((-5 < res_I) & (res_I <= -1))

            qc = "(0.2 < D < 0.75) and (-5 < I < 5)"
            result = A.query(cond=qc, attrs=["D", "I"])[:]
            res_D = self.filter_dense(result["D"], mask_D)
            res_I = self.filter_dense(result["I"], mask_I)
            assert all((0.2 < res_D) & (res_D < 0.75))
            assert all((-5 < res_I) & (res_I < 5))

    def test_check_attrs_sparse(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=True)) as A:
            result = A.query(cond="U < 0.1", attrs=["U"])[:]
            assert all(result["U"] < 0.1)

            result = A.query(cond="U < 1.0", attrs=["U"])[:]
            assert all(result["U"] < 1.0)

            with self.assertRaises(tiledb.TileDBError):
                A.query(cond="U < '1'", attrs=["U"])[:]

            with self.assertRaises(tiledb.TileDBError):
                A.query(cond="U < 'one'", attrs=["U"])[:]

    def test_check_attrs_dense(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=False)) as A:
            mask = A.attr("U").fill

            result = A.query(cond="U < 0.1", attrs=["U"])[:]
            assert all(self.filter_dense(result["U"], mask) < 0.1)

            result = A.query(cond="U < 1.0", attrs=["U"])[:]
            assert all(self.filter_dense(result["U"], mask) < 1.0)

            with self.assertRaises(tiledb.TileDBError):
                A.query(cond="U < '1'", attrs=["U"])[:]

            with self.assertRaises(tiledb.TileDBError):
                A.query(cond="U < 'one'", attrs=["U"])[:]

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

        with tiledb.open(path, "w") as arr:
            arr[np.arange(1, 11)] = {
                "64-bit integer": np.random.randint(-5, 5, 10),
                "double": np.random.rand(10),
            }

        with tiledb.open(path) as arr:
            result = arr.query(cond="attr('64-bit integer') <= val(0)")[:]
            assert all(result["64-bit integer"] <= 0)

            result = arr.query(cond="attr('64-bit integer') <= 0")[:]
            assert all(result["64-bit integer"] <= 0)

            result = arr.query(cond="double <= 0.5")[:]
            assert all(result["double"] <= 0.5)

            result = arr.query(cond="attr('double') <= 0.5")[:]
            assert all(result["double"] <= 0.5)

            result = arr.query(cond="double <= val(0.5)")[:]
            assert all(result["double"] <= 0.5)

            result = arr.query(cond="attr('double') <= val(0.5)")[:]
            assert all(result["double"] <= 0.5)

    def test_casting_str(self):
        path = self.path("test_attr_and_val_casting_str")

        dom = tiledb.Domain(tiledb.Dim(name="dim with spaces", dtype="ascii"))
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
            qc = "attr('attr with spaces') == 'value with spaces'"
            result = arr.query(cond=qc)[:]
            assert list(result["dim with spaces"]) == [b"a", b"c"]

            with pytest.raises(tiledb.TileDBError) as exc_info:
                result = arr.query(cond="dim('attr with spaces') == 'd'")[:]
            assert "is not a dimension" in str(exc_info.value)

            qc = "attr('attr with spaces') == val('value with spaces')"
            result = arr.query(cond=qc)[:]
            assert list(result["dim with spaces"]) == [b"a", b"c"]

            with pytest.raises(tiledb.TileDBError) as exc_info:
                result = arr.query(cond="attr('dim with spaces') == 'd'")[:]
            assert "is not an attribute" in str(exc_info.value)

            result = arr.query(cond="dim('dim with spaces') == 'd'")[:]
            assert list(result["dim with spaces"]) == [b"d"]

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

        def create_array(func):
            return np.array([func[i - 1] * i for i in range(1, 6)], dtype=np.bytes_)

        ascii_data = create_array(string.ascii_lowercase)
        bytes_data = create_array(string.ascii_uppercase)

        with tiledb.open(path, "w") as arr:
            arr[np.arange(5)] = {"ascii": ascii_data, "bytes": bytes_data}

        with tiledb.open(path, "r") as arr:
            for s in ascii_data:
                result = arr.query(cond=f"ascii == '{s.decode()}'")[:]
                assert result["ascii"][0] == s

            for s in bytes_data:
                result = arr.query(cond=f"bytes == '{s.decode()}'")[:]
                assert result["bytes"][0] == s

    @pytest.mark.skipif(
        tiledb.libtiledb.version() < (2, 10, 0),
        reason="OR query condition operator introduced in libtiledb 2.10",
    )
    def test_or_sparse(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=True)) as A:
            result = A.query(cond="(D < 0.25) | (D > 0.75)", attrs=["D"])[:]
            assert all((result["D"] < 0.25) | (result["D"] > 0.75))

            result = A.query(cond="(D < 0.25) or (D > 0.75)", attrs=["D"])[:]
            assert all((result["D"] < 0.25) | (result["D"] > 0.75))

    @pytest.mark.skipif(
        tiledb.libtiledb.version() < (2, 10, 0),
        reason="OR query condition operator introduced in libtiledb 2.10",
    )
    def test_or_dense(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=False)) as A:
            mask = A.attr("D").fill

            result = A.query(cond="(D < 0.25) | (D > 0.75)", attrs=["D"])[:]
            res = self.filter_dense(result["D"], mask)
            assert all((res < 0.25) | (res > 0.75))

            result = A.query(cond="(D < 0.25) or (D > 0.75)", attrs=["D"])[:]
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
            result = A.query(cond="a == 1 and b == 1 and c == 1")[:]
            assert all(result["a"] & result["b"] & result["c"])

            result = A.query(cond="a == 1 and b == 1 or c == 1")[:]
            assert all(result["a"] & result["b"] | result["c"])

            result = A.query(cond="a == 1 or b == 1 and c == 1")[:]
            assert all(result["a"] | result["b"] & result["c"])

            result = A.query(cond="a == 1 or b == 1 or c == 1")[:]
            assert all(result["a"] | result["b"] | result["c"])

            result = A.query(cond="(a == 1 and b == 1) or c == 1")[:]
            assert all(result["a"] & result["b"] | result["c"])

            result = A.query(cond="a == 1 and (b == 1 or c == 1)")[:]
            assert all(result["a"] & (result["b"] | result["c"]))

            result = A.query(cond="(a == 1 or b == 1) and c == 1")[:]
            assert all((result["a"] | result["b"]) & result["c"])

            result = A.query(cond="a == 1 or (b == 1 and c == 1)")[:]
            assert all(result["a"] | result["b"] & result["c"])

    def test_in_operator_sparse(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=True)) as A:
            result = A.query(cond="U in [1, 2, 3]", attrs=["U"])[:]
            for val in result["U"]:
                assert val in [1, 2, 3]

            result = A.query(cond="S in ['a', 'e', 'i', 'o', 'u']", attrs=["S"])[:]
            for val in result["S"]:
                assert val in [b"a", b"e", b"i", b"o", b"u"]

            qc = "S in ['a', 'e', 'i', 'o', 'u'] and U in [5, 6, 7]"
            result = A.query(cond=qc)[:]
            for val in result["U"]:
                assert val in [5, 6, 7]
            for val in result["S"]:
                assert val in [b"a", b"e", b"i", b"o", b"u"]

            result = A.query(cond="U in [8]")[:]
            for val in result["U"]:
                assert val == 8

            result = A.query(cond="S in ['8']")[:]
            assert len(result["S"]) == 0

            result = A.query(cond="U not in [5, 6, 7]")[:]
            for val in result["U"]:
                assert val not in [5, 6, 7]

            with pytest.raises(tiledb.TileDBError) as exc_info:
                A.query(cond="U not in []")[:]
            assert "At least one value must be provided to the set membership" in str(
                exc_info.value
            )

    def test_in_operator_dense(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=False)) as A:
            U_mask = A.attr("U").fill
            S_mask = A.attr("S").fill

            result = A.query(cond="U in [1, 2, 3]", attrs=["U"])[:]
            for val in self.filter_dense(result["U"], U_mask):
                assert val in [1, 2, 3]

            result = A.query(cond="S in ['a', 'e', 'i', 'o', 'u']", attrs=["S"])[:]
            for val in self.filter_dense(result["S"], S_mask):
                assert val in [b"a", b"e", b"i", b"o", b"u"]

            qc = "S in ['a', 'e', 'i', 'o', 'u'] and U in [5, 6, 7]"
            result = A.query(cond=qc)[:]
            for val in self.filter_dense(result["U"], U_mask):
                assert val in [5, 6, 7]
            for val in self.filter_dense(result["S"], S_mask):
                assert val in [b"a", b"e", b"i", b"o", b"u"]

            result = A.query(cond="U in [8]")[:]
            for val in self.filter_dense(result["U"], U_mask):
                assert val == 8

            result = A.query(cond="S in ['8']")[:]
            assert len(self.filter_dense(result["S"], S_mask)) == 0

            result = A.query(cond="U not in [5, 6, 7]")[:]
            for val in self.filter_dense(result["U"], U_mask):
                assert val not in [5, 6, 7]

            with pytest.raises(tiledb.TileDBError) as exc_info:
                A.query(cond="U not in []")[:]
            assert "At least one value must be provided to the set membership" in str(
                exc_info.value
            )

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

            result = A.query(cond=f"dates == {search_date}").df[:]

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
            result = A.query(cond="myint > 10", attrs=["myint"])[:]
            assert all(result["myint"] > 10)

    def test_do_not_return_queried_attr(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=True)) as A:
            qc = "U < 3"

            i_result = A.query(cond=qc, attrs=["I", "U"])[:]
            assert "I" in i_result.keys()
            assert "U" in i_result.keys()
            assert all(i_result["U"] < 5)

            u_result = A.query(cond=qc, attrs=["I"])[:]
            assert "I" in u_result.keys()
            assert "U" not in u_result.keys()
            assert_array_equal(i_result["I"], u_result["I"])

    def test_deprecate_attr_cond(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=True)) as A:
            qc = "U < 3"

            with pytest.raises(tiledb.TileDBError) as exc_info:
                A.query(cond=qc, attr_cond=qc)
            assert "Both `attr_cond` and `cond` were passed." in str(exc_info.value)

            with pytest.raises(tiledb.TileDBError) as exc_info:
                A.query(attr_cond=qc)
            assert "`attr_cond` is no longer supported" in str(exc_info.value)

            with pytest.raises(tiledb.TileDBError) as exc_info:
                A.query(cond=qc).attr_cond
            assert "`attr_cond` is no longer supported" in str(exc_info.value)

            with pytest.raises(tiledb.TileDBError) as exc_info:
                A.subarray(1, cond=qc, attr_cond=qc)
            assert "Both `attr_cond` and `cond` were passed." in str(exc_info.value)

            with pytest.raises(tiledb.TileDBError) as exc_info:
                A.subarray(1, attr_cond=qc)
            assert "`attr_cond` is no longer supported" in str(exc_info.value)

    def test_on_dense_dimensions(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=False)) as A:
            with pytest.raises(tiledb.TileDBError) as excinfo:
                A.query(cond="2 <= d < 6")[:]
            assert (
                "Cannot apply query condition to dimensions on dense arrays"
            ) in str(excinfo.value)

    def test_on_sparse_dimensions(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=True)) as A:
            result = A.query(cond="2 <= d < 6")[:]
            assert_array_equal(result["d"], A[2:6]["d"])

    def test_overlapping(self):
        path = self.path("test_overlapping")

        dom = tiledb.Domain(tiledb.Dim(name="dim", domain=(0, 10), dtype=np.uint32))
        attrs = [tiledb.Attr(name="data", dtype=np.uint32)]

        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=True)
        tiledb.Array.create(path, schema)

        with tiledb.open(path, "w") as A:
            A[np.arange(11)] = np.arange(11)

        with tiledb.open(path, "r") as A:
            result = A.query(cond="2 <= dim < 7 and 5 <= dim < 9")[:]
            assert_array_equal(result["dim"], A[5:7]["dim"])

            result = A.query(cond="2 <= dim < 6 or 5 <= dim < 9")[:]
            assert_array_equal(result["dim"], A[2:9]["dim"])

            result = A.query(cond="2 <= data < 7 and 5 <= data < 9")[:]
            assert_array_equal(result["data"], A[5:7]["data"])

            result = A.query(cond="2 <= data < 6 or 5 <= data < 9")[:]
            assert_array_equal(result["data"], A[2:9]["data"])

    def test_with_whitespace(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=True)) as A:
            result = A.query(cond="        d < 6")[:]
            assert_array_equal(result["d"], A[:6]["d"])

            result = A.query(cond="   (     d < 6)  ")[:]
            assert_array_equal(result["d"], A[:6]["d"])

            result = A.query(cond="   (  \n   d \n\t< 6)  ")[:]
            assert_array_equal(result["d"], A[:6]["d"])

            qc = """
                U < 5
            or
                                                I >= 5
            """
            result = A.query(cond=qc)[:]
            assert all((result["U"] < 5) | (result["U"] > 5))

            qc = """

                                                A == ' a'

            """
            result = A.query(cond=qc)[:]
            # ensures that ' a' does not match 'a'
            assert len(result["A"]) == 0

    @pytest.mark.skipif(not has_pandas(), reason="pandas not installed")
    def test_do_not_return_attrs(self):
        with tiledb.open(self.create_input_array_UIDSA(sparse=True)) as A:
            cond = None
            assert "D" in A.query(cond=cond, attrs=None)[:]
            assert "D" not in A.query(cond=cond, attrs=[])[:]
            assert "D" in A.query(cond=cond, attrs=None).df[:]
            assert "D" not in A.query(cond=cond, attrs=[]).df[:]
            assert "D" in A.query(cond=cond, attrs=None).multi_index[:]
            assert "D" not in A.query(cond=cond, attrs=[]).multi_index[:]

            cond = "D > 100"
            assert "D" in A.query(cond=cond, attrs=None)[:]
            assert "D" not in A.query(cond=cond, attrs=[])[:]
            assert "D" in A.query(cond=cond, attrs=None).df[:]
            assert "D" not in A.query(cond=cond, attrs=[]).df[:]
            assert "D" in A.query(cond=cond, attrs=None).multi_index[:]
            assert "D" not in A.query(cond=cond, attrs=[]).multi_index[:]

    def test_boolean_sparse(self):
        path = self.path("test_boolean_sparse")

        dom = tiledb.Domain(tiledb.Dim(domain=(1, 10), tile=1, dtype=np.uint32))
        attrs = [
            tiledb.Attr(name="a", dtype=np.bool_),
            tiledb.Attr(name="b", dtype=np.bool_),
            tiledb.Attr(name="c", dtype=np.bool_),
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
            result = A.query(cond="a == True")[:]
            assert all(result["a"])

            result = A.query(cond="a == False")[:]
            assert all(~result["a"])

            result = A.query(cond="a == True and b == True")[:]
            assert all(result["a"])
            assert all(result["b"])

            result = A.query(cond="a == False and c == True")[:]
            assert all(~result["a"])
            assert all(result["c"])

    def test_boolean_dense(self):
        path = self.path("test_boolean_dense")

        dom = tiledb.Domain(tiledb.Dim(domain=(1, 10), tile=1, dtype=np.uint32))
        attrs = [
            tiledb.Attr(name="a", dtype=np.bool_),
            tiledb.Attr(name="b", dtype=np.bool_),
            tiledb.Attr(name="c", dtype=np.bool_),
        ]
        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False)
        tiledb.Array.create(path, schema)

        with tiledb.open(path, "w") as arr:
            arr[:] = {
                "a": np.random.randint(0, high=2, size=10),
                "b": np.random.randint(0, high=2, size=10),
                "c": np.random.randint(0, high=2, size=10),
            }

        with tiledb.open(path) as A:
            mask = A.attr("a").fill

            result = A.query(cond="a == True")[:]
            assert all(self.filter_dense(result["a"], mask))

            result = A.query(cond="a == True and b == True")[:]
            assert all(self.filter_dense(result["a"], mask))
            assert all(self.filter_dense(result["b"], mask))

    def test_qc_enumeration(self):
        uri = self.path("test_qc_enumeration")
        dom = tiledb.Domain(tiledb.Dim(domain=(1, 8), tile=1))
        enum1 = tiledb.Enumeration("enmr1", True, [0, 1, 2])
        enum2 = tiledb.Enumeration("enmr2", False, ["a", "bb", "ccc"])
        attr1 = tiledb.Attr("attr1", dtype=np.int32, enum_label="enmr1")
        attr2 = tiledb.Attr("attr2", dtype=np.int32, enum_label="enmr2")
        schema = tiledb.ArraySchema(
            domain=dom, attrs=(attr1, attr2), enums=(enum1, enum2)
        )
        tiledb.Array.create(uri, schema)

        data1 = np.random.randint(0, 3, 8)
        data2 = np.random.randint(0, 3, 8)

        with tiledb.open(uri, "w") as A:
            A[:] = {"attr1": data1, "attr2": data2}

        with tiledb.open(uri, "r") as A:
            mask = A.attr("attr1").fill
            result = A.query(cond="attr1 < 2", attrs=["attr1"])[:]
            assert all(self.filter_dense(result["attr1"], mask) < 2)

            mask = A.attr("attr2").fill
            result = A.query(cond="attr2 == 'bb'", attrs=["attr2"])[:]
            assert all(
                self.filter_dense(result["attr2"], mask)
                == list(enum2.values()).index("bb")
            )

    def test_boolean_insert(self):
        path = self.path("test_boolean_insert")
        attr = tiledb.Attr("a", dtype=np.bool_, var=False)
        dom = tiledb.Domain(tiledb.Dim(domain=(1, 10), tile=1, dtype=np.uint32))
        schema = tiledb.ArraySchema(domain=dom, sparse=True, attrs=[attr])
        tiledb.Array.create(path, schema)
        a = np.array(
            list(
                [
                    np.array([True], dtype=np.bool_),
                    np.array([True], dtype=np.bool_),
                    np.array([True], dtype=np.bool_),
                    np.array([True], dtype=np.bool_),
                ]
            ),
            dtype=object,
        )
        with tiledb.open(path, "w") as A:
            A[range(1, len(a) + 1)] = {"a": a}

        with tiledb.open(path, "r") as A:
            for k in A[:]["a"]:
                assert k == True  # noqa: E712


class QueryDeleteTest(DiskTestCase):
    def test_basic_sparse(self):
        path = self.path("test_basic_sparse")
        dom = tiledb.Domain(tiledb.Dim(domain=(1, 10), tile=1, dtype=np.uint32))
        attrs = [tiledb.Attr("ints", dtype=np.uint32)]
        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=True)
        tiledb.Array.create(path, schema)

        data = np.random.randint(1, 10, 10)

        qc = "ints < 5"

        with tiledb.open(path, "w") as A:
            A[np.arange(1, 11)] = data

            with pytest.raises(
                tiledb.TileDBError,
                match="SparseArray must be opened in read or delete mode",
            ):
                A.query(cond=qc).submit()

        with tiledb.open(path, "r") as A:
            assert_array_equal(data, A[:]["ints"])

        with tiledb.open(path, "d") as A:
            with pytest.raises(
                tiledb.TileDBError,
                match="Cannot initialize deletes; One condition is needed",
            ):
                A.query().submit()

            A.query(cond=qc).submit()

        with tiledb.open(path, "r") as A:
            assert all(A[:]["ints"] >= 5)

    def test_basic_dense(self):
        path = self.path("test_basic_dense")

        dom = tiledb.Domain(tiledb.Dim(domain=(1, 10), tile=1))
        attrs = [tiledb.Attr("ints", dtype=np.uint8)]
        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False)
        tiledb.Array.create(path, schema)

        with tiledb.open(path, "d") as A:
            with pytest.raises(
                tiledb.TileDBError,
                match="DenseArray must be opened in read mode",
            ):
                A.query()

    def test_with_fragments(self):
        path = self.path("test_with_fragments")

        dom = tiledb.Domain(tiledb.Dim(domain=(1, 3), tile=1))
        attrs = [tiledb.Attr("ints", dtype=np.uint8)]
        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=True)
        tiledb.Array.create(path, schema)

        with tiledb.open(path, "w", timestamp=1) as A:
            A[1] = 1

        with tiledb.open(path, "w", timestamp=2) as A:
            A[2] = 2

        with tiledb.open(path, "w", timestamp=3) as A:
            A[3] = 3

        with tiledb.open(path, "r") as A:
            assert_array_equal([1, 2, 3], A[:]["ints"])

        with tiledb.open(path, "d", timestamp=3) as A:
            A.query(cond="ints == 1").submit()

        with tiledb.open(path, "r", timestamp=1) as A:
            assert_array_equal([1], A[:]["ints"])

        with tiledb.open(path, "r", timestamp=2) as A:
            assert_array_equal([1, 2], A[:]["ints"])

        with tiledb.open(path, "r", timestamp=3) as A:
            assert_array_equal([2, 3], A[:]["ints"])

        assert len(tiledb.array_fragments(path)) == 3

        tiledb.consolidate(path)
        tiledb.vacuum(path)

        assert len(tiledb.array_fragments(path)) == 1

        with tiledb.open(path, "r") as A:
            assert A.nonempty_domain() == ((1, 3),)
            assert_array_equal([2, 3], A[:]["ints"])

    def test_purge_deleted_cells(self):
        path = self.path("test_with_fragments")

        dom = tiledb.Domain(tiledb.Dim(domain=(1, 3), tile=1))
        attrs = [tiledb.Attr("ints", dtype=np.uint8)]
        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=True)
        tiledb.Array.create(path, schema)

        with tiledb.open(path, "w", timestamp=1) as A:
            A[1] = 1

        with tiledb.open(path, "w", timestamp=2) as A:
            A[2] = 2

        with tiledb.open(path, "w", timestamp=3) as A:
            A[3] = 3

        with tiledb.open(path, "r") as A:
            assert_array_equal([1, 2, 3], A[:]["ints"])

        with tiledb.open(path, "d", timestamp=3) as A:
            A.query(cond="ints == 1").submit()

        with tiledb.open(path, "r", timestamp=1) as A:
            assert_array_equal([1], A[:]["ints"])

        with tiledb.open(path, "r", timestamp=2) as A:
            assert_array_equal([1, 2], A[:]["ints"])

        with tiledb.open(path, "r", timestamp=3) as A:
            assert_array_equal([2, 3], A[:]["ints"])

        cfg = tiledb.Config({"sm.consolidation.purge_deleted_cells": "true"})
        with tiledb.scope_ctx(cfg):
            tiledb.consolidate(path)
        tiledb.vacuum(path)

        with tiledb.open(path, "r") as A:
            assert A.nonempty_domain() == ((2, 3),)
            assert_array_equal([2, 3], A[:]["ints"])

    def test_delete_with_string_dimension(self):
        path = self.path("test_delete_with_string_dimension")

        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(tiledb.Dim(name="d", dtype="|S0", var=True)),
            attrs=[tiledb.Attr(name="a", dtype="uint32")],
            sparse=True,
        )

        tiledb.Array.create(path, schema)

        with tiledb.open(path, "w") as A:
            A[["a", "b", "c"]] = [10, 20, 30]

        with tiledb.open(path, "d") as A:
            A.query(cond="a == 20").submit()

        with tiledb.open(path, "r") as A:
            assert_array_equal(A[:]["d"], [b"a", b"c"])
            assert_array_equal(A[:]["a"], [10, 30])

        with tiledb.open(path, "d") as A:
            A.query(cond="d == 'a'").submit()

        with tiledb.open(path, "r") as A:
            assert_array_equal(A[:]["d"], [b"c"])
            assert_array_equal(A[:]["a"], [30])

    def test_qc_dense_empty(self):
        path = self.path("test_qc_dense_empty")

        dom = tiledb.Domain(tiledb.Dim(name="d", domain=(1, 1), tile=1, dtype=np.uint8))
        attrs = [tiledb.Attr(name="a", dtype=np.uint8)]
        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False)
        tiledb.Array.create(path, schema)

        with tiledb.open(path, mode="w") as A:
            A[:] = np.arange(1)

        with tiledb.open(path) as A:
            assert_array_equal(A.query(cond="")[:]["a"], [0])

    def test_qc_sparse_empty(self):
        path = self.path("test_qc_sparse_empty")

        dom = tiledb.Domain(
            tiledb.Dim(name="d", domain=(1, 10), tile=1, dtype=np.uint8)
        )
        attrs = [tiledb.Attr(name="a", dtype=np.uint8)]
        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=True)
        tiledb.Array.create(path, schema)

        with tiledb.open(path, mode="w") as A:
            A[1] = {"a": np.arange(1)}

        with tiledb.open(path) as A:
            assert_array_equal(A.query(cond="")[:]["a"], [0])
