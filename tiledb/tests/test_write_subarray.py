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

    def test_1d_full_write(self):
        # Create array.
        uri = self.path("dense_write_subarray_1d_full_write")
        schema = tiledb.ArraySchema(
            tiledb.Domain(tiledb.Dim(name="d1", domain=(0, 999), tile=1000)),
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

    def test_1d_partial_write(self):
        # Create array.
        uri = self.path("dense_write_subarray_1d_multiple_partial_writes")
        schema = tiledb.ArraySchema(
            tiledb.Domain(tiledb.Dim(name="d1", domain=(0, 99), tile=100)),
            [tiledb.Attr(name="", dtype=np.float32)],
        )
        tiledb.Array.create(uri, schema)

        # Write data.
        data = np.random.rand(10).astype(np.float32)
        with tiledb.open(uri, "w") as array:
            subarray = tiledb.Subarray(array)
            subarray.add_dim_range(0, (10, 19))
            array.write_subarray(subarray, data)

        # Check results.
        with tiledb.open(uri, "r") as array:
            result = array[10:20]
        assert_array_equal(result, data)

    def test_multidim_set_all_ranges(self):
        # Create array.
        uri = self.path("dense_write_subarray_multidim_set_all_ranges")
        schema = tiledb.ArraySchema(
            tiledb.Domain(
                tiledb.Dim(name="d1", domain=(0, 99), tile=100),
                tiledb.Dim(name="d2", domain=(0, 99), tile=100),
                tiledb.Dim(name="d3", domain=(0, 99), tile=100),
            ),
            [
                tiledb.Attr(name="a1", dtype=np.float64),
                tiledb.Attr(name="a2", dtype=np.float64),
            ],
        )
        tiledb.Array.create(uri, schema)

        # Write data.
        data1 = np.random.rand(1000).reshape((10, 10, 10))
        data2 = np.random.rand(1000).reshape((10, 10, 10))

        with tiledb.open(uri, "w") as array:
            subarray = tiledb.Subarray(array)
            subarray.add_dim_range(0, (0, 9))
            subarray.add_dim_range(1, (10, 19))
            subarray.add_dim_range(2, (20, 29))
            array.write_subarray(subarray, {"a1": data1, "a2": data2})

        # Check results.
        with tiledb.open(uri, "r") as array:
            nonempty = array.nonempty_domain()
            assert nonempty[0] == (0, 9)
            assert nonempty[1] == (10, 19)
            assert nonempty[2] == (20, 29)
            result = array[0:10, 10:20, 20:30]
        assert len(result) == 2
        assert_array_equal(result["a1"], data1)
        assert_array_equal(result["a2"], data2)

    def test_multidim_set_some_ranges(self):
        # Create array.
        uri = self.path("dense_write_subarray_multidim_set_some_ranges")
        schema = tiledb.ArraySchema(
            tiledb.Domain(
                tiledb.Dim(name="d1", domain=(0, 99), tile=100),
                tiledb.Dim(name="d2", domain=(0, 99), tile=100),
            ),
            [tiledb.Attr(name="a1", dtype=np.float64)],
        )
        tiledb.Array.create(uri, schema)

        # Write data.
        data = np.random.rand(1000)
        with tiledb.open(uri, "w") as array:
            subarray = tiledb.Subarray(array)
            subarray.add_dim_range(1, (11, 20))
            array.write_subarray(subarray, {"a1": data})

        # Check results.
        with tiledb.open(uri, "r") as array:
            nonempty = array.nonempty_domain()
            assert nonempty[0] == (0, 99)
            assert nonempty[1] == (11, 20)
            result = array[:, 11:21]
        assert len(result) == 1
        assert_array_equal(result["a1"], data.reshape(100, 10))

    def test_write_by_label(self):
        # Create array.
        uri = self.path("dense_write_subarray_multidim_set_some_ranges")
        dim1 = tiledb.Dim(name="d1", domain=(0, 10), tile=11)
        dim2 = tiledb.Dim(name="d2", domain=(0, 10), tile=11)
        schema = tiledb.ArraySchema(
            tiledb.Domain(dim1, dim2),
            [tiledb.Attr(name="a1", dtype=np.float64)],
            dim_labels={
                0: {"l1": dim1.create_label_schema("increasing", np.int32)},
                1: {"l2": dim1.create_label_schema("decreasing", np.int32)},
            },
        )
        tiledb.Array.create(uri, schema)

        data = {
            "a1": np.random.rand(121),
            "l1": np.arange(-5, 6),
            "l2": np.arange(5, -6, -1),
        }

        # Write full data and label data
        with tiledb.open(uri, "w") as array:
            subarray = tiledb.Subarray(array)
            subarray.add_dim_range(0, (0, 10))
            subarray.add_dim_range(1, (0, 10))
            array.write_subarray(subarray, data)

        with tiledb.open(uri, "r") as array:
            result = array.label_index(["l1", "l2"])[-5:5, -5:5]
        assert len(result) == 3
        assert_array_equal(result["a1"], data["a1"].reshape(11, 11))
        assert_array_equal(result["l1"], data["l1"])
        assert_array_equal(result["l2"], data["l2"])
