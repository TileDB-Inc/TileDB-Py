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
    #    np.uint16,
    # np.uint32,
    # np.uint64,
    # np.int8,
    # np.int16,
    # np.int32,
    # np.int64,
)

SUPPORTED_DATETIME64_RESOLUTION = ("Y", "M", "W", "D", "h", "m", "s", "ms", "us", "ns")


@pytest.mark.parametrize("dim_dtype", SUPPORTED_INTEGER_DTYPES)
@pytest.mark.parametrize("sparse", (True, False))
class TestReadSubarray1D(DiskTestCase):
    data = np.random.rand(101)
    label_data = np.linspace(-1.0, 1.0, 101)

    @pytest.fixture
    def array_uri(self, dim_dtype, sparse):
        """Create TileDB array, write data, and return the URI."""
        suffix = (
            f"{np.dtype(dim_dtype).name}_sparse"
            if sparse
            else f"{np.dtype(dim_dtype).name}_dense"
        )
        uri = self.path(f"dense_read_subarray_1d_{suffix}")
        dim1 = tiledb.Dim(name="d1", domain=(0, 100), tile=101, dtype=dim_dtype)
        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(dim1),
            attrs=[tiledb.Attr(name="a1", dtype=np.float64)],
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
            "a1": self.data,
            "l1": self.label_data,
            "l2": np.flip(self.label_data),
        }
        with tiledb.open(uri, "w") as array:
            if sparse:
                array[np.arange(101, dtype=dim_dtype)] = data_buffers
            else:
                array[...] = data_buffers
        return uri

    def test_read_full_array(self, array_uri):
        with tiledb.open(array_uri, "r") as array:
            sparse = array.schema.sparse
            subarray = tiledb.Subarray(array)
            subarray.add_dim_range(0, (0, 100))
            result = array.read_subarray(subarray)
        assert_unordered_equal(result["a1"], self.data, unordered=sparse)

    def test_read_full_array_by_increasing_label(self, array_uri):
        with tiledb.open(array_uri, "r") as array:
            sparse = array.schema.sparse
            subarray = tiledb.Subarray(array)
            subarray.add_label_range("l1", (-1.0, 1.0))
            result = array.read_subarray(subarray)
        assert_unordered_equal(result["a1"], self.data, unordered=sparse)

    def test_read_full_array_by_decreasing_label(self, array_uri):
        with tiledb.open(array_uri, "r") as array:
            sparse = array.schema.sparse
            subarray = tiledb.Subarray(array)
            subarray.add_label_range("l2", (-1.0, 1.0))
            result = array.read_subarray(subarray)
        assert_unordered_equal(result["a1"], self.data, unordered=sparse)


# @pytest.mark.parametrize("dim_res", SUPPORTED_DATETIME64_RESOLUTION, scope="class")
@pytest.mark.parametrize("dim_res", ["Y"], scope="class")
class TestReadSubarrayDenseDatetime1D(DiskTestCase):
    data = np.random.rand(100)

    @pytest.fixture
    def array_uri(self, dim_res):
        """Create TileDB array, write data, and return the URI."""
        uri = self.path(f"dense_read_subarray_1d_datetime_{dim_res}")
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
        )
        tiledb.Array.create(uri, schema)
        with tiledb.open(uri, "w") as array:
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
