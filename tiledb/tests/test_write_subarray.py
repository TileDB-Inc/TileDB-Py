import numpy as np
import pytest

import tiledb

from .common import DiskTestCase, assert_array_equal

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


class TestWriteSubarrayDense(DiskTestCase):
    @pytest.mark.parametrize("dim_dtype", [np.uint32])
    def test_1d_full_write(self, dim_dtype):
        # Create array.
        uri = self.path(f"dense_write_subarray_1d_{np.dtype(dim_dtype).name}")
        schema = tiledb.ArraySchema(
            tiledb.Domain(
                tiledb.Dim(name="d1", domain=(0, 999), tile=1000, dtype=dim_dtype)
            ),
            [tiledb.Attr(name="", dtype=np.float64)],
        )
        tiledb.Array.create(uri, schema)

        # Write data.
        data = np.random.rand(1000)
        with tiledb.open(uri, "w") as array:
            subarray = tiledb.Subarray(array)
            subarray.add_dim_range(0, (0, 999))
            array.write_subarray(subarray, data)

        # Check results.
        with tiledb.open(uri, "r") as array:
            result = array[...]
        assert_array_equal(result, data)

    @pytest.mark.parametrize("dim_res", SUPPORTED_DATETIME64_RESOLUTION)
    def test_1d_datetime_full_write(self, dim_res):
        """Create TileDB array, write data, and return the URI."""
        # Create array.
        uri = self.path(f"write_subarray_1d_datetime_{dim_res}")
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
            [tiledb.Attr(name="", dtype=np.float64)],
        )
        tiledb.Array.create(uri, schema)

        # Write data.
        data = np.random.rand(100)
        with tiledb.open(uri, "w") as array:
            subarray = tiledb.Subarray(array)
            subarray.add_dim_range(0, domain)
            array.write_subarray(subarray, data)

        # Check results.
        with tiledb.open(uri, "r") as array:
            result = array[...]
        assert_array_equal(result, data)
