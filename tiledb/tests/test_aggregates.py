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
            q = A.query()
            expected = q[:]["a"]

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

            # subarray
            expected = A[4:7]["a"]

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

            # subarray
            expected = A.multi_index[4:7]["a"]

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

            A[np.arange(0, 10)] = np.random.randint(1, 10, size=10)

        all_aggregates = ("count", "sum", "min", "max", "mean")

        with tiledb.open(path, "r") as A:
            expected = A.query(cond="a < 5")[:]["a"]
            actual = A.query(cond="a < 5").agg(all_aggregates)[:]
            assert actual["sum"] == sum(expected)
            assert actual["min"] == min(expected)
            assert actual["max"] == max(expected)
            assert actual["mean"] == sum(expected) / len(expected)
            assert actual["count"] == len(expected)

            expected = A.query(cond="a < 5").multi_index[:]["a"]
            actual = A.query(cond="a < 5").agg(all_aggregates).multi_index[:]
            assert actual["sum"] == sum(expected)
            assert actual["min"] == min(expected)
            assert actual["max"] == max(expected)
            assert actual["mean"] == sum(expected) / len(expected)
            assert actual["count"] == len(expected)

            # no value matches query condition
            invalid_aggregates = ("count", "sum", "min", "max", "mean")
            expected = A.query(cond="a > 10")[:]
            actual = A.query(cond="a > 10").agg(invalid_aggregates)[:]
            assert actual["sum"] == 0
            assert actual["min"] is None
            assert actual["max"] is None
            assert np.isnan(actual["mean"])
            assert actual["count"] == 0

            expected = A.query(cond="a > 10").multi_index[:]
            actual = A.query(cond="a > 10").agg(invalid_aggregates).multi_index[:]
            assert actual["sum"] == 0
            assert actual["min"] is None
            assert actual["max"] is None
            assert np.isnan(actual["mean"])
            assert actual["count"] == 0

    @pytest.mark.parametrize("sparse", [True, False])
    def test_nullable(self, sparse):
        path = self.path("test_nullable")
        dom = tiledb.Domain(tiledb.Dim(name="d", domain=(0, 9), dtype=np.int32))
        attrs = [tiledb.Attr(name="a", nullable=True, dtype=float)]
        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=sparse)
        tiledb.Array.create(path, schema)

        # set index 5 and 7 to be null
        data = np.random.rand(10)
        data[5], data[7] = np.nan, np.nan

        # write data
        with tiledb.open(path, "w") as A:
            if sparse:
                A[np.arange(0, 10)] = data
            else:
                A[:] = data

        with tiledb.open(path, "r") as A:
            assert A.query().agg("null_count")[0] == 0
            assert A.query().agg("null_count")[:6] == 1
            assert A.query().agg("null_count")[5:8] == 2
            assert A.query().agg("null_count")[5] == 1
            assert A.query().agg("null_count")[6:] == 1
            assert A.query().agg("null_count")[7] == 1
            assert A.query().agg("null_count")[:] == 2

            all_aggregates = ("count", "sum", "min", "max", "mean", "null_count")
            actual = A.query().agg({"a": all_aggregates})[:]
            expected = A[:]["a"]
            expected_no_null = A[:]["a"].compressed()
            assert actual["sum"] == sum(expected_no_null)
            assert actual["min"] == min(expected_no_null)
            assert actual["max"] == max(expected_no_null)
            assert actual["mean"] == sum(expected_no_null) / len(expected_no_null)
            assert actual["count"] == len(expected)
            assert actual["null_count"] == np.count_nonzero(expected.mask)

            # no valid values
            actual = A.query().agg({"a": all_aggregates})[5]
            assert actual["sum"] is None
            assert actual["min"] is None
            assert actual["max"] is None
            assert actual["mean"] is None
            assert actual["count"] == 1
            assert actual["null_count"] == 1

    def test_empty_sparse(self):
        path = self.path("test_empty_sparse")
        dom = tiledb.Domain(tiledb.Dim(name="d", domain=(0, 9), dtype=np.int32))
        attrs = [tiledb.Attr(name="a", dtype=float)]
        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=True)
        tiledb.Array.create(path, schema)

        with tiledb.open(path, "w") as A:
            A[np.arange(0, 5)] = np.random.rand(5)

        with tiledb.open(path, "r") as A:
            invalid_aggregates = ("sum", "min", "max", "mean")
            actual = A.query().agg(invalid_aggregates)[6:]
            assert actual["sum"] == 0
            assert actual["min"] is None
            assert actual["max"] is None
            assert np.isnan(actual["mean"])
            assert "count" not in actual

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

            assert A.query().agg({"string": "count"})[:] == len(actual["string"])
            invalid_aggregates = ("sum", "min", "max", "mean")
            for agg in invalid_aggregates:
                with pytest.raises(tiledb.TileDBError):
                    A.query().agg({"string": agg})[:]

            result = A.query().agg("count")[:]
            assert result["integer"]["count"] == len(actual["integer"])
            assert result["float"]["count"] == len(actual["float"])
            assert result["string"]["count"] == len(actual["string"])

            with pytest.raises(tiledb.TileDBError):
                A.query().agg("sum")[:]

            result = A.query().agg({"integer": "sum", "float": "sum"})[:]
            assert "string" not in result
            assert result["integer"]["sum"] == sum(actual["integer"])
            assert result["float"]["sum"] == sum(actual["float"])

            result = A.query().agg(
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
