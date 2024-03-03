import numpy as np
import pytest

import tiledb

from .common import DiskTestCase


class AggregateTest(DiskTestCase):
    @pytest.mark.parametrize("sparse", [True, False])
    @pytest.mark.parametrize(
        "dtype",
        [
            np.uint8,
            np.int8,
            np.uint16,
            np.int16,
            np.uint32,
            np.int32,
            np.uint64,
            np.int64,
            np.float32,
            np.float64,
        ],
    )
    def test_basic(self, sparse, dtype):
        path = self.path("test_basic")
        dom = tiledb.Domain(tiledb.Dim(name="d", domain=(0, 9), dtype=np.int32))
        attrs = [tiledb.Attr(name="a", dtype=dtype)]
        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=sparse)
        tiledb.Array.create(path, schema)

        data = np.random.randint(1, 10, size=10)

        with tiledb.open(path, "w") as A:
            if sparse:
                A[np.arange(0, 10)] = data
            else:
                A[:] = data

        all_aggregates = ("count", "sum", "min", "max", "mean")

        with tiledb.open(path, "r") as A:
            # entire column
            q = A.query(dims=["d"])
            results = q[:]
            expected = results["a"]
            expected_dimension = results["d"]

            with pytest.raises(tiledb.TileDBError):
                q.agg("bad")[:]

            with pytest.raises(tiledb.TileDBError):
                q.agg("null_count")[:]

            with pytest.raises(NotImplementedError):
                q.agg("count").df[:]

            assert q.agg("sum")[:] == sum(expected)
            assert q.agg("min")[:] == min(expected)
            assert q.agg("max")[:] == max(expected)
            assert q.agg("mean")[:] == sum(expected) / len(expected)
            assert q.agg("count")[:] == len(expected)

            assert q.agg({"a": "sum"})[:] == sum(expected)
            assert q.agg({"a": "min"})[:] == min(expected)
            assert q.agg({"a": "max"})[:] == max(expected)
            assert q.agg({"a": "mean"})[:] == sum(expected) / len(expected)
            assert q.agg({"a": "count"})[:] == len(expected)

            actual = q.agg(all_aggregates)[:]
            assert actual["sum"] == sum(expected)
            assert actual["min"] == min(expected)
            assert actual["max"] == max(expected)
            assert actual["mean"] == sum(expected) / len(expected)
            assert actual["count"] == len(expected)

            actual = q.agg({"a": all_aggregates})[:]
            assert actual["sum"] == sum(expected)
            assert actual["min"] == min(expected)
            assert actual["max"] == max(expected)
            assert actual["mean"] == sum(expected) / len(expected)
            assert actual["count"] == len(expected)

            if sparse:
                actual = q.agg({"d": all_aggregates})[:]
                assert actual["sum"] == sum(expected_dimension)
                assert actual["min"] == min(expected_dimension)
                assert actual["max"] == max(expected_dimension)
                assert actual["mean"] == sum(expected_dimension) / len(expected_dimension)
                assert actual["count"] == len(expected_dimension)

            # subarray
            expected = A[4:7]["a"]
            expected_dimension = A.query(dims=["d"], attrs=[])[4:7]["d"]

            assert q.agg("sum")[4:7] == sum(expected)
            assert q.agg("min")[4:7] == min(expected)
            assert q.agg("max")[4:7] == max(expected)
            assert q.agg("mean")[4:7] == sum(expected) / len(expected)
            assert q.agg("count")[4:7] == len(expected)

            assert q.agg({"a": "sum"})[4:7] == sum(expected)
            assert q.agg({"a": "min"})[4:7] == min(expected)
            assert q.agg({"a": "max"})[4:7] == max(expected)
            assert q.agg({"a": "mean"})[4:7] == sum(expected) / len(expected)
            assert q.agg({"a": "count"})[4:7] == len(expected)

            actual = q.agg(all_aggregates)[4:7]
            assert actual["sum"] == sum(expected)
            assert actual["min"] == min(expected)
            assert actual["max"] == max(expected)
            assert actual["mean"] == sum(expected) / len(expected)
            assert actual["count"] == len(expected)

            actual = q.agg({"a": all_aggregates})[4:7]
            assert actual["sum"] == sum(expected)
            assert actual["min"] == min(expected)
            assert actual["max"] == max(expected)
            assert actual["mean"] == sum(expected) / len(expected)
            assert actual["count"] == len(expected)

            # dimension
            if sparse:
                actual = q.agg({"d": all_aggregates})[4:7]
                assert actual["sum"] == sum(expected_dimension)
                assert actual["min"] == min(expected_dimension)
                assert actual["max"] == max(expected_dimension)
                assert actual["mean"] == sum(expected_dimension) / len(expected_dimension)
                assert actual["count"] == len(expected_dimension)

    @pytest.mark.parametrize("sparse", [True, False])
    @pytest.mark.parametrize(
        "dtype",
        [
            np.uint8,
            np.int8,
            np.uint16,
            np.int16,
            np.uint32,
            np.int32,
            np.uint64,
            np.int64,
            np.float32,
            np.float64,
        ],
    )
    def test_multi_index(self, sparse, dtype):
        path = self.path("test_multi_index")
        dom = tiledb.Domain(tiledb.Dim(name="d", domain=(0, 9), dtype=np.int32))
        attrs = [tiledb.Attr(name="a", dtype=dtype)]
        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=sparse)
        tiledb.Array.create(path, schema)

        data = np.random.randint(1, 10, size=10)

        with tiledb.open(path, "w") as A:
            if sparse:
                A[np.arange(0, 10)] = data
            else:
                A[:] = data

        all_aggregates = ("count", "sum", "min", "max", "mean")

        with tiledb.open(path, "r") as A:
            # entire column
            q = A.query()
            expected = q.multi_index[:]["a"]
            expected_dimension = A.query(dims=["d"], attrs=[]).multi_index[:]["d"]

            with pytest.raises(tiledb.TileDBError):
                q.agg("bad")[:]

            with pytest.raises(tiledb.TileDBError):
                q.agg("null_count")[:]

            assert q.agg("sum").multi_index[:] == sum(expected)
            assert q.agg("min").multi_index[:] == min(expected)
            assert q.agg("max").multi_index[:] == max(expected)
            assert q.agg("mean").multi_index[:] == sum(expected) / len(expected)
            assert q.agg("count").multi_index[:] == len(expected)

            actual = q.agg(all_aggregates).multi_index[:]
            assert actual["sum"] == sum(expected)
            assert actual["min"] == min(expected)
            assert actual["max"] == max(expected)
            assert actual["mean"] == sum(expected) / len(expected)
            assert actual["count"] == len(expected)

            actual = q.agg({"a": all_aggregates}).multi_index[:]
            assert actual["sum"] == sum(expected)
            assert actual["min"] == min(expected)
            assert actual["max"] == max(expected)
            assert actual["mean"] == sum(expected) / len(expected)
            assert actual["count"] == len(expected)

            # dimension
            actual = q.agg({"d": all_aggregates}).multi_index[:]
            assert actual["sum"] == sum(expected_dimension)
            assert actual["min"] == min(expected_dimension)
            assert actual["max"] == max(expected_dimension)
            assert actual["mean"] == sum(expected_dimension) / len(expected_dimension)
            assert actual["count"] == len(expected_dimension)

            # subarray
            expected = A.multi_index[4:7]["a"]
            expected_dimension = A.query(dims=["d"], attrs=[]).multi_index[4:7][
                "d"
            ]

            assert q.agg("sum").multi_index[4:7] == sum(expected)
            assert q.agg("min").multi_index[4:7] == min(expected)
            assert q.agg("max").multi_index[4:7] == max(expected)
            assert q.agg("mean").multi_index[4:7] == sum(expected) / len(expected)
            assert q.agg("count").multi_index[4:7] == len(expected)

            actual = q.agg(all_aggregates).multi_index[4:7]
            assert actual["sum"] == sum(expected)
            assert actual["min"] == min(expected)
            assert actual["max"] == max(expected)
            assert actual["mean"] == sum(expected) / len(expected)
            assert actual["count"] == len(expected)

            # dimension
            actual = q.agg({"d": all_aggregates})[4:7]
            assert actual["sum"] == sum(expected_dimension)
            assert actual["min"] == min(expected_dimension)
            assert actual["max"] == max(expected_dimension)
            assert actual["mean"] == sum(expected_dimension) / len(expected_dimension)
            assert actual["count"] == len(expected_dimension)

    @pytest.mark.parametrize(
        "dtype",
        [
            np.uint8,
            np.int8,
            np.uint16,
            np.int16,
            np.uint32,
            np.int32,
            np.uint64,
            np.int64,
            np.float32,
            np.float64,
        ],
    )
    def test_with_query_condition(self, dtype):
        path = self.path("test_with_query_condition")
        dom = tiledb.Domain(tiledb.Dim(name="d", domain=(0, 9), dtype=np.int32))
        attrs = [tiledb.Attr(name="a", dtype=dtype)]
        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=True)
        tiledb.Array.create(path, schema)

        with tiledb.open(path, "w") as A:
            # hardcode the first value to be 1 to ensure that the a < 5
            # query condition always returns a non-empty result
            data = np.random.randint(1, 10, size=10)
            data[0] = 1

            A[np.arange(0, 10)] = data

        all_aggregates = ("count", "sum", "min", "max", "mean")

        with tiledb.open(path, "r") as A:
            q = A.query(cond="a < 5")

            expected = q[:]["a"]
            actual = q.agg(all_aggregates)[:]
            assert actual["sum"] == sum(expected)
            assert actual["min"] == min(expected)
            assert actual["max"] == max(expected)
            assert actual["mean"] == sum(expected) / len(expected)
            assert actual["count"] == len(expected)

            expected = q.multi_index[:]["a"]
            actual = q.agg(all_aggregates).multi_index[:]
            assert actual["sum"] == sum(expected)
            assert actual["min"] == min(expected)
            assert actual["max"] == max(expected)
            assert actual["mean"] == sum(expected) / len(expected)
            assert actual["count"] == len(expected)

            # no value matches query condition
            q = A.query(cond="a > 10")

            expected = q[:]
            actual = q.agg(all_aggregates)[:]
            assert actual["sum"] == 0
            if dtype in (np.float32, np.float64):
                assert np.isnan(actual["min"])
                assert np.isnan(actual["max"])
            else:
                assert actual["min"] is None
                assert actual["max"] is None
            assert np.isnan(actual["mean"])
            assert actual["count"] == 0

            expected = q.multi_index[:]
            actual = q.agg(all_aggregates).multi_index[:]
            assert actual["sum"] == 0
            if dtype in (np.float32, np.float64):
                assert np.isnan(actual["min"])
                assert np.isnan(actual["max"])
            else:
                assert actual["min"] is None
                assert actual["max"] is None
            assert np.isnan(actual["mean"])
            assert actual["count"] == 0

    @pytest.mark.parametrize("sparse", [True, False])
    def test_nullable(self, sparse):
        path = self.path("test_nullable")
        dom = tiledb.Domain(tiledb.Dim(name="d", domain=(0, 9), dtype=np.int32))
        attrs = [
            tiledb.Attr(name="integer", nullable=True, dtype=int),
            tiledb.Attr(name="float", nullable=True, dtype=float),
        ]
        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=sparse)
        tiledb.Array.create(path, schema)

        # set index 5 and 7 to be null
        data = np.random.rand(10)
        data[5], data[7] = np.nan, np.nan

        # write data
        with tiledb.open(path, "w") as A:
            if sparse:
                A[np.arange(0, 10)] = {"integer": data, "float": data}
            else:
                A[:] = {"integer": data, "float": data}

        with tiledb.open(path, "r") as A:
            agg = A.query().agg

            result = agg("null_count")
            assert result[0]["integer"]["null_count"] == 0
            assert result[:6]["integer"]["null_count"] == 1
            assert result[5:8]["integer"]["null_count"] == 2
            assert result[5]["integer"]["null_count"] == 1
            assert result[6:]["integer"]["null_count"] == 1
            assert result[7]["integer"]["null_count"] == 1
            assert result[:]["integer"]["null_count"] == 2

            assert result[0]["float"]["null_count"] == 0
            assert result[:6]["float"]["null_count"] == 1
            assert result[5:8]["float"]["null_count"] == 2
            assert result[5]["float"]["null_count"] == 1
            assert result[6:]["float"]["null_count"] == 1
            assert result[7]["float"]["null_count"] == 1
            assert result[:]["float"]["null_count"] == 2

            all_aggregates = ("count", "sum", "min", "max", "mean", "null_count")

            actual = agg({"integer": all_aggregates, "float": all_aggregates})[:]

            expected = A[:]["integer"]
            expected_no_null = A[:]["integer"].compressed()
            assert actual["integer"]["sum"] == sum(expected_no_null)
            assert actual["integer"]["min"] == min(expected_no_null)
            assert actual["integer"]["max"] == max(expected_no_null)
            assert actual["integer"]["mean"] == sum(expected_no_null) / len(
                expected_no_null
            )
            assert actual["integer"]["count"] == len(expected)
            assert actual["integer"]["null_count"] == np.count_nonzero(expected.mask)

            expected = A[:]["float"]
            expected_no_null = A[:]["float"].compressed()
            assert actual["float"]["sum"] == sum(expected_no_null)
            assert actual["float"]["min"] == min(expected_no_null)
            assert actual["float"]["max"] == max(expected_no_null)
            assert actual["float"]["mean"] == sum(expected_no_null) / len(
                expected_no_null
            )
            assert actual["float"]["count"] == len(expected)
            assert actual["float"]["null_count"] == np.count_nonzero(expected.mask)

            # no valid values
            actual = agg({"integer": all_aggregates, "float": all_aggregates})[5]

            assert actual["integer"]["sum"] is None
            assert actual["integer"]["min"] is None
            assert actual["integer"]["max"] is None
            assert actual["integer"]["mean"] is None
            assert actual["integer"]["count"] == 1
            assert actual["integer"]["null_count"] == 1

            assert np.isnan(actual["float"]["sum"])
            assert np.isnan(actual["float"]["min"])
            assert np.isnan(actual["float"]["max"])
            assert np.isnan(actual["float"]["mean"])
            assert actual["float"]["count"] == 1
            assert actual["float"]["null_count"] == 1

    @pytest.mark.parametrize("sparse", [True, False])
    def test_empty(self, sparse):
        path = self.path("test_empty_sparse")
        dom = tiledb.Domain(tiledb.Dim(name="d", domain=(0, 9), dtype=np.int32))
        attrs = [
            tiledb.Attr(name="integer", nullable=True, dtype=int),
            tiledb.Attr(name="float", nullable=True, dtype=float),
        ]
        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=sparse)
        tiledb.Array.create(path, schema)

        data = np.random.rand(5)

        # write data
        with tiledb.open(path, "w") as A:
            if sparse:
                A[np.arange(0, 5)] = {"integer": data, "float": data}
            else:
                A[:5] = {"integer": data, "float": data}

        with tiledb.open(path, "r") as A:
            invalid_aggregates = ("sum", "min", "max", "mean")
            actual = A.query().agg(invalid_aggregates)[6:]

            assert actual["integer"]["sum"] is None
            assert actual["integer"]["min"] is None
            assert actual["integer"]["max"] is None
            assert actual["integer"]["mean"] is None

            assert np.isnan(actual["float"]["sum"])
            assert np.isnan(actual["float"]["min"])
            assert np.isnan(actual["float"]["max"])
            assert np.isnan(actual["float"]["mean"])

    def test_multiple_attrs(self):
        path = self.path("test_multiple_attrs")
        dom = tiledb.Domain(tiledb.Dim(name="d", domain=(0, 9), dtype=np.int32))
        attrs = [
            tiledb.Attr(name="integer", dtype=int),
            tiledb.Attr(name="float", dtype=float),
            tiledb.Attr(name="string", dtype=str),
        ]
        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=True)
        tiledb.Array.create(path, schema)

        with tiledb.open(path, "w") as A:
            A[np.arange(0, 10)] = {
                "integer": np.random.randint(1, 10, size=10),
                "float": np.random.randint(1, 10, size=10),
                "string": np.random.randint(1, 10, size=10).astype(str),
            }

        with tiledb.open(path, "r") as A:
            actual = A.query()[:]
            agg = A.query().agg

            assert agg({"string": "count"})[:] == len(actual["string"])
            invalid_aggregates = ("sum", "min", "max", "mean")
            for invalid_agg in invalid_aggregates:
                with pytest.raises(tiledb.TileDBError):
                    agg({"string": invalid_agg})[:]

            result = agg("count")[:]
            assert result["integer"]["count"] == len(actual["integer"])
            assert result["float"]["count"] == len(actual["float"])
            assert result["string"]["count"] == len(actual["string"])

            with pytest.raises(tiledb.TileDBError):
                agg("sum")[:]

            result = agg({"integer": "sum", "float": "sum"})[:]
            assert "string" not in result
            assert result["integer"]["sum"] == sum(actual["integer"])
            assert result["float"]["sum"] == sum(actual["float"])

            result = agg(
                {
                    "string": ("count",),
                    "integer": "sum",
                    "float": ["max", "min", "sum", "mean"],
                }
            )[:]
            assert result["string"]["count"] == len(actual["string"])
            assert result["integer"]["sum"] == sum(actual["integer"])
            assert result["float"]["max"] == max(actual["float"])
            assert result["float"]["min"] == min(actual["float"])
            assert result["float"]["sum"] == sum(actual["float"])
            assert result["float"]["mean"] == sum(actual["float"]) / len(
                actual["float"]
            )

    def test_strings(self):
        all_dim_aggregates = ("count", "min", "max", "mean")
        all_aggregates = ("count", "min", "max", "mean", "null_count")

        path = self.path("test_string_sparse")
        dom = tiledb.Domain(tiledb.Dim(name="d", dtype="ascii"))
        attrs = [
            tiledb.Attr(name="ascii", nullable=True, dtype="ascii"),
            tiledb.Attr(name="utf8", nullable=True, dtype="S0"),
        ]
        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=True)
        tiledb.Array.create(path, schema)

        dim_data = ["1", "2", "3", "4", "5"]
        data = ["a", "bb", "ccc", "dddd", "eeeee"]

        # write data
        with tiledb.open(path, "w") as A:
            A[dim_data] = {"ascii": data, "utf8": data}

        with tiledb.open(path, "r") as A:
            invalid_aggregates = ("sum", "min", "max", "mean")
            actual = A.query().agg(invalid_aggregates)["6":]

            assert actual["ascii"]["min"] is None
            assert actual["ascii"]["max"] is None
            assert actual["ascii"]["mean"] is None

            # entire column
            q = A.query()
            expected = q.multi_index[:]
            actual = q.agg({"ascii": all_aggregates})[:]
            assert actual["min"] == min(expected["ascii"])
            assert actual["max"] == max(expected["ascii"])
            assert actual["mean"] == sum(expected["ascii"]) / len(expected["ascii"])
            assert actual["count"] == len(expected["ascii"])

            actual = q.agg({"utf8": all_aggregates})[:]
            assert actual["min"] == min(expected["utf8"])
            assert actual["max"] == max(expected["utf8"])
            assert actual["mean"] == sum(expected["utf8"]) / len(expected["utf8"])
            assert actual["count"] == len(expected["utf8"])

            actual = q.agg({"d": all_dim_aggregates})[:]
            assert actual["min"] == min(expected["utf8"])
            assert actual["max"] == max(expected["utf8"])
            assert actual["mean"] == sum(expected["utf8"]) / len(expected["utf8"])
            assert actual["count"] == len(expected["utf8"])
