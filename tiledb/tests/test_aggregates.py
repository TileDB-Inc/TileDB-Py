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

        # write data
        with tiledb.open(path, "w") as A:
            if sparse:
                A[np.arange(0, 10)] = data
            else:
                A[:] = data

        all_aggregates = ("count", "sum", "min", "max", "mean")

        with tiledb.open(path, "r") as A:
            # entire column
            expected = A[:]["a"]

            with pytest.raises(tiledb.TileDBError):
                A.query().agg("bad")[:]

            with pytest.raises(tiledb.TileDBError):
                A.query().agg("null_count")[:]

            assert A.query().agg("sum")[:] == sum(expected)
            assert A.query().agg("min")[:] == min(expected)
            assert A.query().agg("max")[:] == max(expected)
            assert A.query().agg("mean")[:] == sum(expected) / len(expected)
            assert A.query().agg("count")[:] == len(expected)

            assert A.query().agg({"a": "sum"})[:] == sum(expected)
            assert A.query().agg({"a": "min"})[:] == min(expected)
            assert A.query().agg({"a": "max"})[:] == max(expected)
            assert A.query().agg({"a": "mean"})[:] == sum(expected) / len(expected)
            assert A.query().agg({"a": "count"})[:] == len(expected)

            actual = A.query().agg(all_aggregates)[:]
            assert actual["sum"] == sum(expected)
            assert actual["min"] == min(expected)
            assert actual["max"] == max(expected)
            assert actual["mean"] == sum(expected) / len(expected)
            assert actual["count"] == len(expected)

            actual = A.query().agg({"a": all_aggregates})[:]
            assert actual["sum"] == sum(expected)
            assert actual["min"] == min(expected)
            assert actual["max"] == max(expected)
            assert actual["mean"] == sum(expected) / len(expected)
            assert actual["count"] == len(expected)

            # subarray
            expected = A[4:7]["a"]

            assert A.query().agg("sum")[4:7] == sum(expected)
            assert A.query().agg("min")[4:7] == min(expected)
            assert A.query().agg("max")[4:7] == max(expected)
            assert A.query().agg("mean")[4:7] == sum(expected) / len(expected)
            assert A.query().agg("count")[4:7] == len(expected)

            assert A.query().agg({"a": "sum"})[4:7] == sum(expected)
            assert A.query().agg({"a": "min"})[4:7] == min(expected)
            assert A.query().agg({"a": "max"})[4:7] == max(expected)
            assert A.query().agg({"a": "mean"})[4:7] == sum(expected) / len(expected)
            assert A.query().agg({"a": "count"})[4:7] == len(expected)

            actual = A.query().agg(all_aggregates)[4:7]
            assert actual["sum"] == sum(expected)
            assert actual["min"] == min(expected)
            assert actual["max"] == max(expected)
            assert actual["mean"] == sum(expected) / len(expected)
            assert actual["count"] == len(expected)

            actual = A.query().agg({"a": all_aggregates})[4:7]
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
            
            # no value matches query condition
            invalid_aggregates = ("sum", "min", "max", "mean")
            expected = A.query(cond="a > 10")[:]
            actual = A.query(cond="a > 10").agg(invalid_aggregates)[:]
            assert actual["sum"] is None
            assert actual["min"] is None
            assert actual["max"] is None
            assert actual["mean"] is None
            assert "count" not in actual
            
    @pytest.mark.parametrize("sparse", [True, False])
    def test_nullable(self, sparse):
        path = self.path("test_basic")
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
            assert actual["mean"] == sum(expected_no_null)/len(expected_no_null)
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
        path = self.path("test_basic")
        dom = tiledb.Domain(tiledb.Dim(name="d", domain=(0, 9), dtype=np.int32))
        attrs = [tiledb.Attr(name="a", dtype=float)]
        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=True)
        tiledb.Array.create(path, schema)

        with tiledb.open(path, "w") as A:
            A[np.arange(0, 5)] = np.random.rand(5)
            
        with tiledb.open(path, "r") as A:
            invalid_aggregates = ("sum", "min", "max", "mean")
            actual = A.query().agg(invalid_aggregates)[6:]
            assert actual["sum"] is None
            assert actual["min"] is None
            assert actual["max"] is None
            assert actual["mean"] is None
            assert "count" not in actual
            
    # TODO
    # test multiple attributes
    # test multiple operations
    # test incorrect dtypes
