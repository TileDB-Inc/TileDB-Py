from collections import OrderedDict

import numpy as np
import pytest

import tiledb

from .common import DiskTestCase, assert_array_equal, assert_dict_arrays_equal

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
        data = OrderedDict(
            [
                ("a1", np.random.rand(1000).reshape((10, 10, 10))),
                ("a2", np.random.rand(1000).reshape((10, 10, 10))),
            ]
        )

        with tiledb.open(uri, "w") as array:
            subarray = tiledb.Subarray(array)
            subarray.add_dim_range(0, (0, 9))
            subarray.add_dim_range(1, (10, 19))
            subarray.add_dim_range(2, (20, 29))
            array.write_subarray(subarray, data)

        # Check results.
        with tiledb.open(uri, "r") as array:
            nonempty = array.nonempty_domain()
            assert nonempty == ((0, 9), (10, 19), (20, 29))
            result = array[0:10, 10:20, 20:30]
        assert_dict_arrays_equal(result, data)

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
        assert_dict_arrays_equal(result, {"a1": data.reshape(100, 10)})

    def test_with_negative_domain(self):
        # Create array.
        uri = self.path("dense_write_subarray_by_labels")
        schema = tiledb.ArraySchema(
            tiledb.Domain(
                tiledb.Dim(name="d1", domain=(-100, 100), tile=201, dtype=np.int32)
            ),
            [tiledb.Attr(name="a1", dtype=np.float64)],
        )
        tiledb.Array.create(uri, schema)

        # Define the data.
        data = OrderedDict([("a1", np.random.rand(5))])

        # Write full data and label data
        with tiledb.open(uri, "w") as array:
            subarray = tiledb.Subarray(array)
            subarray.add_dim_range(0, (-2, 2))
            array.write_subarray(subarray, data["a1"])

        # Check results
        with tiledb.open(uri, "r") as array:
            nonempty = array.nonempty_domain()
            assert nonempty[0] == (-2, 2)
            result = array.multi_index[-2:2]

        assert_dict_arrays_equal(result, data)

    def test_with_labels(self):
        # Create array.
        uri = self.path("dense_write_subarray_with_labels")
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

        data = OrderedDict(
            [
                ("a1", np.random.rand(121).reshape(11, 11)),
                ("l1", np.arange(-5, 6)),
                ("l2", np.arange(5, -6, -1)),
            ]
        )

        # Write full data and label data
        with tiledb.open(uri, "w") as array:
            subarray = tiledb.Subarray(array)
            subarray.add_dim_range(0, (0, 10))
            subarray.add_dim_range(1, (0, 10))
            array.write_subarray(subarray, data)

        # Check results
        with tiledb.open(uri, "r") as array:
            result = array.label_index(["l1", "l2"])[-5:5, -5:5]
        assert_dict_arrays_equal(result, data)

    def test_by_labels(self):
        # Create array.
        uri = self.path("dense_write_subarray_by_labels")
        dim1 = tiledb.Dim(name="d1", domain=(0, 10), tile=11)
        schema = tiledb.ArraySchema(
            tiledb.Domain(dim1),
            [tiledb.Attr(name="a1", dtype=np.float64)],
            dim_labels={0: {"l1": dim1.create_label_schema("increasing", np.int32)}},
        )
        tiledb.Array.create(uri, schema)

        # Define the data.
        data = OrderedDict(
            [("a1", np.random.rand(5)), ("l1", np.arange(-5, 6, dtype=np.int32))]
        )

        # Reload to get the label uris and write the labels.
        schema = tiledb.ArraySchema.load(uri)
        with tiledb.open(schema.dim_label("l1").uri, mode="w") as array:
            array[:] = data["l1"]

        # Write full data and label data
        with tiledb.open(uri, "w") as array:
            subarray = tiledb.Subarray(array)
            subarray.add_label_range("l1", (-2, 2))
            with pytest.raises(tiledb.TileDBError):
                array.write_subarray(subarray, data["a1"])

    def test_with_var_label(self):
        # Create array.
        uri = self.path("dense_write_subarray_by_var_label")
        dim1 = tiledb.Dim(name="d1", domain=(0, 10), tile=11)
        schema = tiledb.ArraySchema(
            tiledb.Domain(dim1),
            [tiledb.Attr(name="a1", dtype=np.float64)],
            dim_labels={
                0: {"l1": dim1.create_label_schema("increasing", "U")},
            },
        )
        tiledb.Array.create(uri, schema)

        # Write array.
        data = OrderedDict(
            [
                ("a1", np.random.rand(5)),
                (
                    "l1",
                    np.array(
                        ["alpha", "beta", "gamma", "kappa", "sigma"], dtype=object
                    ),
                ),
            ]
        )
        with tiledb.open(uri, "w") as array:
            subarray = tiledb.Subarray(array)
            subarray.add_dim_range(0, (3, 7))
            array.write_subarray(subarray, data)

        # Check results.
        with tiledb.open(uri, "r") as array:
            nonempty = array.nonempty_domain()
            assert nonempty[0] == (3, 7)
            with tiledb.open(array.schema.dim_label("l1").uri, "r") as label_array:
                nonempty_label = label_array.nonempty_domain()
                assert nonempty_label[0] == (3, 7)
            array.label_index(["l1"])["alpha":"sigma"]
