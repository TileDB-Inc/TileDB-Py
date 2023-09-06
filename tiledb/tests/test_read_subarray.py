import numpy as np
import pytest

import tiledb

from .common import (
    DiskTestCase,
    assert_array_equal,
    assert_unordered_equal,
)

SUPPORTED_INTEGER_DTYPES = (
    np.uint8,
    np.uint16,
    np.uint32,
    np.uint64,
    np.int8,
    np.int16,
    np.int32,
    np.int64,
)

SUPPORTED_DATETIME64_RESOLUTION = ("Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns")


@pytest.mark.parametrize("dim_dtype", (np.uint32,))
@pytest.mark.parametrize("sparse", (True, False))
class TestReadSubarray1D(DiskTestCase):
    data1 = np.random.rand(101)
    data2 = np.random.randint(-1000, 1000, (101,), dtype=np.int16)
    label_data = np.linspace(-1.0, 1.0, 101)

    @pytest.fixture
    def array_uri(self, sparse, dim_dtype):
        """Create TileDB array, write data, and return the URI."""
        suffix = "1d_label_sparse" if sparse else "1d_label_dense"
        uri = self.path(f"read_subarray_1d_{suffix}")
        dim1 = tiledb.Dim(name="d1", domain=(0, 100), tile=101, dtype=np.int32)
        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(dim1),
            attrs=[
                tiledb.Attr(name="a1", dtype=np.float64),
                tiledb.Attr(name="a2", dtype=np.int16),
            ],
            dim_labels={
                0: {
                    "l1": dim1.create_label_schema("increasing", np.float64),
                    "l2": dim1.create_label_schema("decreasing", np.float64),
                }
            },
            sparse=sparse,
        )
        tiledb.Array.create(uri, schema)
        data_buffers = {
            "a1": self.data1,
            "a2": self.data2,
            "l1": self.label_data,
            "l2": np.flip(self.label_data),
        }
        with tiledb.open(uri, "w") as array:
            if sparse:
                array[np.arange(101, dtype=np.int32)] = data_buffers
            else:
                array[...] = data_buffers
        return uri

    def test_read_full_array(self, array_uri):
        with tiledb.open(array_uri, "r") as array:
            sparse = array.schema.sparse
            subarray = tiledb.Subarray(array)
            subarray.add_dim_range(0, (0, 100))
            result = array.read_subarray(subarray)
        assert_unordered_equal(result["a1"], self.data1, unordered=sparse)
        assert_unordered_equal(result["a2"], self.data2, unordered=sparse)
        if sparse:
            assert_unordered_equal(result["d1"], np.arange(101), True)
            assert len(result) == 3
        else:
            assert len(result) == 2

    def test_read_partial(self, array_uri):
        with tiledb.open(array_uri, "r") as array:
            sparse = array.schema.sparse
            subarray = tiledb.Subarray(array)
            subarray.add_dim_range(0, (10, 20))
            result = array.read_subarray(subarray)
        assert_unordered_equal(result["a1"], self.data1[10:21], unordered=sparse)
        assert_unordered_equal(result["a2"], self.data2[10:21], unordered=sparse)
        if sparse:
            assert_unordered_equal(result["d1"], np.arange(10, 21), True)
            assert len(result) == 3
        else:
            assert len(result) == 2

    def test_read_full_array_by_increasing_label(self, array_uri):
        with tiledb.open(array_uri, "r") as array:
            sparse = array.schema.sparse
            subarray = tiledb.Subarray(array)
            subarray.add_label_range("l1", (-1.0, 1.0))
            result = array.read_subarray(subarray)
        assert_unordered_equal(result["a1"], self.data1, unordered=sparse)
        assert_unordered_equal(result["a2"], self.data2, unordered=sparse)
        if sparse:
            assert_unordered_equal(result["d1"], np.arange(101), True)
            assert len(result) == 3
        else:
            assert len(result) == 2

    def test_read_partial_by_increasing_label(self, array_uri):
        with tiledb.open(array_uri, "r") as array:
            sparse = array.schema.sparse
            subarray = tiledb.Subarray(array)
            subarray.add_label_range("l1", (0.0, 1.0))
            result = array.read_subarray(subarray)
        assert_unordered_equal(result["a1"], self.data1[50:], unordered=sparse)
        assert_unordered_equal(result["a2"], self.data2[50:], unordered=sparse)
        if sparse:
            assert_unordered_equal(result["d1"], np.arange(50, 101), True)
            assert len(result) == 3
        else:
            assert len(result) == 2

    def test_read_partial_by_decreasing_label(self, array_uri):
        with tiledb.open(array_uri, "r") as array:
            sparse = array.schema.sparse
            subarray = tiledb.Subarray(array)
            subarray.add_label_range("l2", (0.0, 1.0))
            result = array.read_subarray(subarray)
        assert_unordered_equal(result["a1"], self.data1[:51], unordered=sparse)
        assert_unordered_equal(result["a2"], self.data2[:51], unordered=sparse)
        if sparse:
            assert_unordered_equal(result["d1"], np.arange(51), True)
            assert len(result) == 3
        else:
            assert len(result) == 2


@pytest.mark.parametrize("dim_res", SUPPORTED_DATETIME64_RESOLUTION)
@pytest.mark.parametrize("sparse", (True, False))
class TestReadSubarrayDenseDatetime1D(DiskTestCase):
    data = np.random.rand(100)

    @pytest.fixture
    def array_uri(self, dim_res, sparse):
        """Create TileDB array, write data, and return the URI."""
        suffix = f"datetime_{dim_res}_sparse" if sparse else f"datetime_{dim_res}_dense"
        uri = self.path(f"read_subarray_1d_datetime_{suffix}")
        start_time = np.datetime64("2000-01-01", dim_res)
        domain = (start_time, start_time + np.timedelta64(99, dim_res))
        schema = tiledb.ArraySchema(
            tiledb.Domain(
                tiledb.Dim(
                    name="d1",
                    domain=domain,
                    tile=100,
                    dtype=np.dtype(f"M8[{dim_res}]"),
                )
            ),
            [tiledb.Attr(name="a1", dtype=np.float64)],
            sparse=sparse,
        )
        tiledb.Array.create(uri, schema)
        with tiledb.open(uri, "w") as array:
            if sparse:
                array[
                    np.arange(
                        domain[0],
                        domain[1] + np.timedelta64(1, dim_res),
                        np.timedelta64(1, dim_res),
                    )
                ] = self.data
            else:
                array[...] = self.data
        return uri

    def test_read_full_array(self, array_uri, dim_res):
        with tiledb.open(array_uri, "r") as array:
            subarray = tiledb.Subarray(array)
            start_time = np.datetime64("2000-01-01", dim_res)
            domain = (start_time, start_time + np.timedelta64(99, dim_res))
            subarray.add_dim_range(0, domain)
            result = array.read_subarray(subarray)
        assert_array_equal(result["a1"], self.data)

    def test_read_partial(self, array_uri, dim_res):
        with tiledb.open(array_uri, "r") as array:
            subarray = tiledb.Subarray(array)
            start_time = np.datetime64("2000-01-01", dim_res)
            dim_range = (
                start_time + np.timedelta64(10, dim_res),
                start_time + np.timedelta64(20, dim_res),
            )
            subarray.add_dim_range(0, dim_range)
            result = array.read_subarray(subarray)
        assert_array_equal(result["a1"], self.data[10:21])
