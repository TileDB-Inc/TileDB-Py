from collections import OrderedDict

import numpy as np
import pytest

import tiledb

from .common import (
    DiskTestCase,
    assert_array_equal,
    assert_dict_arrays_equal,
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


@pytest.mark.parametrize("sparse", (True, False))
class TestReadSubarray1D(DiskTestCase):
    data1 = np.random.rand(101)
    data2 = np.random.randint(-1000, 1000, (101,), dtype=np.int16)
    label_data = np.linspace(-1.0, 1.0, 101)

    @pytest.fixture
    def array_uri(self, sparse):
        """Create TileDB array, write data, and return the URI."""
        suffix = "1d_label_sparse" if sparse else "1d_label_dense"
        uri = self.path(f"read_subarray_{suffix}")
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

        expected = OrderedDict()
        if sparse:
            expected["d1"] = np.arange(101, dtype=np.int32)
        expected["a1"] = self.data1
        expected["a2"] = self.data2

        assert_dict_arrays_equal(result, expected, not sparse)

    def test_read_partial(self, array_uri):
        with tiledb.open(array_uri, "r") as array:
            sparse = array.schema.sparse
            subarray = tiledb.Subarray(array)
            subarray.add_dim_range(0, (10, 20))
            result = array.read_subarray(subarray)

        expected = OrderedDict()
        if sparse:
            expected["d1"] = np.arange(10, 21, dtype=np.int32)
        expected["a1"] = self.data1[10:21]
        expected["a2"] = self.data2[10:21]

        assert_dict_arrays_equal(result, expected, not sparse)

    def test_read_multiple_ranges(self, array_uri):
        with tiledb.open(array_uri, "r") as array:
            sparse = array.schema.sparse
            subarray = tiledb.Subarray(array)
            subarray.add_dim_range(0, (3, 3))
            subarray.add_dim_range(0, (1, 2))
            subarray.add_dim_range(0, (5, 10))
            result = array.read_subarray(subarray)

        expected = OrderedDict()
        d1_expected = np.array([3, 1, 2, 5, 6, 7, 8, 9, 10], dtype=np.int32)
        if sparse:
            expected["d1"] = d1_expected
        expected["a1"] = self.data1[d1_expected]
        expected["a2"] = self.data2[d1_expected]

        assert_dict_arrays_equal(result, expected, not sparse)

    def test_read_single_attr(self, array_uri):
        with tiledb.open(array_uri, attr="a1", mode="r") as array:
            sparse = array.schema.sparse
            subarray = tiledb.Subarray(array)
            subarray.add_dim_range(0, (10, 20))
            result = array.read_subarray(subarray)

        expected = OrderedDict()
        if sparse:
            expected["d1"] = np.arange(10, 21, dtype=np.int32)
        expected["a1"] = self.data1[10:21]

        assert_dict_arrays_equal(result, expected, not sparse)

    def test_read_full_array_by_increasing_label(self, array_uri):
        with tiledb.open(array_uri, "r") as array:
            sparse = array.schema.sparse
            subarray = tiledb.Subarray(array)
            subarray.add_label_range("l1", (-1.0, 1.0))
            result = array.read_subarray(subarray)

        expected = OrderedDict()
        if sparse:
            expected["d1"] = np.arange(101, dtype=np.int32)
        expected["a1"] = self.data1
        expected["a2"] = self.data2

        assert_dict_arrays_equal(result, expected, not sparse)

    def test_read_partial_by_increasing_label(self, array_uri):
        with tiledb.open(array_uri, "r") as array:
            sparse = array.schema.sparse
            subarray = tiledb.Subarray(array)
            subarray.add_label_range("l1", (0.0, 1.0))
            result = array.read_subarray(subarray)

        expected = OrderedDict()
        if sparse:
            expected["d1"] = np.arange(50, 101, dtype=np.int32)
        expected["a1"] = self.data1[50:]
        expected["a2"] = self.data2[50:]

        assert_dict_arrays_equal(result, expected, not sparse)

    def test_read_partial_by_decreasing_label(self, array_uri):
        with tiledb.open(array_uri, "r") as array:
            sparse = array.schema.sparse
            subarray = tiledb.Subarray(array)
            subarray.add_label_range("l2", (0.0, 1.0))
            result = array.read_subarray(subarray)

        expected = OrderedDict()
        if sparse:
            expected["d1"] = np.arange(51, dtype=np.int32)
        expected["a1"] = self.data1[:51]
        expected["a2"] = self.data2[:51]

        assert_dict_arrays_equal(result, expected, not sparse)

    def test_read_by_label_no_data(self, array_uri):
        with tiledb.open(array_uri, "r") as array:
            subarray = tiledb.Subarray(array)
            subarray.add_label_range("l1", (0.01, 0.012))
            with pytest.raises(tiledb.TileDBError):
                array.read_subarray(subarray)


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
            sparse = array.schema.sparse
            subarray = tiledb.Subarray(array)
            start_time = np.datetime64("2000-01-01", dim_res)
            domain = (start_time, start_time + np.timedelta64(99, dim_res))
            subarray.add_dim_range(0, domain)
            result = array.read_subarray(subarray)

        expected = OrderedDict()
        if sparse:
            expected["d1"] = np.arange(
                start_time,
                start_time + np.timedelta64(100, dim_res),
                np.timedelta64(1, dim_res),
            )
        expected["a1"] = self.data

        assert_dict_arrays_equal(result, expected, not sparse)

    def test_read_partial(self, array_uri, dim_res):
        with tiledb.open(array_uri, "r") as array:
            sparse = array.schema.sparse
            subarray = tiledb.Subarray(array)
            start_time = np.datetime64("2000-01-01", dim_res)
            dim_range = (
                start_time + np.timedelta64(10, dim_res),
                start_time + np.timedelta64(20, dim_res),
            )
            subarray.add_dim_range(0, dim_range)
            result = array.read_subarray(subarray)
        assert_array_equal(result["a1"], self.data[10:21])

        expected = OrderedDict()
        if sparse:
            expected["d1"] = np.arange(
                start_time + np.timedelta64(10, dim_res),
                start_time + np.timedelta64(21, dim_res),
                np.timedelta64(1, dim_res),
            )
        expected["a1"] = self.data[10:21]

        assert_dict_arrays_equal(result, expected, not sparse)


@pytest.mark.parametrize("sparse", (True, False))
class TestReadSubarray2D(DiskTestCase):
    data_a1 = np.random.rand(16).reshape(4, 4)
    data_a2 = np.random.randint(-1000, 1000, (4, 4), dtype=np.int16)
    data_l1 = np.arange(-2, 2)
    data_l2 = np.arange(1, -3, -1)

    @pytest.fixture
    def array_uri(self, sparse):
        """Create TileDB array, write data, and return the URI."""
        suffix = "2d_sparse" if sparse else "2d_dense"
        uri = self.path(f"read_subarray_{suffix}")
        dim1 = tiledb.Dim(name="d1", domain=(0, 3), tile=4, dtype=np.int32)
        dim2 = tiledb.Dim(name="d2", domain=(0, 3), tile=4, dtype=np.int32)
        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(dim1, dim2),
            attrs=[
                tiledb.Attr(name="a1", dtype=np.float64),
                tiledb.Attr(name="a2", dtype=np.int16),
            ],
            dim_labels={
                0: {"l1": dim1.create_label_schema("increasing", np.int32)},
                1: {"l2": dim1.create_label_schema("decreasing", np.int32)},
            },
            sparse=sparse,
        )
        tiledb.Array.create(uri, schema)
        if sparse:
            _schema = tiledb.ArraySchema.load(uri)
            with tiledb.open(_schema.dim_label("l1").uri, mode="w") as label1:
                label1[:] = self.data_l1
            with tiledb.open(_schema.dim_label("l2").uri, mode="w") as label2:
                label2[:] = self.data_l2
            data_d1, data_d2 = np.meshgrid(np.arange(4), np.arange(4), indexing="ij")
            with tiledb.open(uri, "w") as array:
                array[data_d1.flatten(), data_d2.flatten()] = {
                    "a1": self.data_a1,
                    "a2": self.data_a2,
                }
        else:
            with tiledb.open(uri, "w") as array:
                array[...] = {
                    "a1": self.data_a1,
                    "a2": self.data_a2,
                    "l1": self.data_l1,
                    "l2": self.data_l2,
                }

        return uri

    def test_read_full_array(self, array_uri):
        with tiledb.open(array_uri) as array:
            sparse = array.schema.sparse
            subarray = tiledb.Subarray(array)
            subarray.add_dim_range(0, (0, 3))
            subarray.add_dim_range(1, (0, 3))
            result = array.read_subarray(subarray)
        if sparse:
            # Construct the expected result
            data_d1, data_d2 = np.meshgrid(
                np.arange(4, dtype=np.int32),
                np.arange(4, dtype=np.int32),
                indexing="ij",
            )
            expected = {
                "d1": data_d1.flatten(),
                "d2": data_d2.flatten(),
                "a1": self.data_a1.flatten(),
                "a2": self.data_a2.flatten(),
            }
        else:
            expected = {"a1": self.data_a1, "a2": self.data_a2}
        assert_dict_arrays_equal(result, expected, not sparse)

    def test_read_mixed_ranges(self, array_uri):
        with tiledb.open(array_uri) as array:
            sparse = array.schema.sparse
            subarray = tiledb.Subarray(array)
            subarray.add_dim_range(0, (0, 1))
            result = array.read_subarray(subarray)
        if sparse:
            data_d1, data_d2 = np.meshgrid(
                np.arange(2, dtype=np.int32),
                np.arange(4, dtype=np.int32),
                indexing="ij",
            )
            expected = {
                "d1": data_d1.flatten(),
                "d2": data_d2.flatten(),
                "a1": self.data_a1[0:2, :].flatten(),
                "a2": self.data_a2[0:2, :].flatten(),
            }
        else:
            expected = {"a1": self.data_a1[0:2, :], "a2": self.data_a2[0:2, :]}
        assert_dict_arrays_equal(result, expected, not sparse)


@pytest.mark.parametrize("sparse", (True, False))
class TestReadSubarrayNegativeDomain2D(DiskTestCase):
    data_a1 = np.random.rand(121).reshape(11, 11)
    data_a2 = np.random.randint(-1000, 1000, (11, 11), dtype=np.int16)

    @pytest.fixture
    def array_uri(self, sparse):
        """Create TileDB array, write data, and return the URI."""
        suffix = "_sparse" if sparse else "_dense"
        uri = self.path(f"read_subarray_{suffix}")
        dim1 = tiledb.Dim(name="d1", domain=(-5, 5), tile=4, dtype=np.int32)
        dim2 = tiledb.Dim(name="d2", domain=(-5, 5), tile=4, dtype=np.int32)
        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(dim1, dim2),
            attrs=[
                tiledb.Attr(name="a1", dtype=np.float64),
                tiledb.Attr(name="a2", dtype=np.int16),
            ],
            sparse=sparse,
        )
        tiledb.Array.create(uri, schema)
        if sparse:
            data_d1, data_d2 = np.meshgrid(
                np.arange(-5, 6), np.arange(-5, 6), indexing="ij"
            )
            with tiledb.open(uri, "w") as array:
                array[data_d1.flatten(), data_d2.flatten()] = {
                    "a1": self.data_a1,
                    "a2": self.data_a2,
                }
        else:
            with tiledb.open(uri, "w") as array:
                array[...] = {"a1": self.data_a1, "a2": self.data_a2}

        return uri

    def test_read_full_array(self, array_uri):
        with tiledb.open(array_uri) as array:
            sparse = array.schema.sparse
            subarray = tiledb.Subarray(array)
            subarray.add_dim_range(0, (-5, 5))
            subarray.add_dim_range(1, (-5, 5))
            result = array.read_subarray(subarray)
        if sparse:
            # Construct the expected result
            data_d1, data_d2 = np.meshgrid(
                np.arange(-5, 6, dtype=np.int32),
                np.arange(-5, 6, dtype=np.int32),
                indexing="ij",
            )
            expected = {
                "d1": data_d1.flatten(),
                "d2": data_d2.flatten(),
                "a1": self.data_a1.flatten(),
                "a2": self.data_a2.flatten(),
            }
        else:
            expected = {"a1": self.data_a1, "a2": self.data_a2}
        assert_dict_arrays_equal(result, expected, not sparse)

    def test_read_mixed_ranges(self, array_uri):
        with tiledb.open(array_uri) as array:
            sparse = array.schema.sparse
            subarray = tiledb.Subarray(array)
            subarray.add_dim_range(1, (-1, 2))
            result = array.read_subarray(subarray)

        if sparse:
            data_d1, data_d2 = np.meshgrid(
                np.arange(-5, 6, dtype=np.int32),
                np.arange(-1, 3, dtype=np.int32),
                indexing="ij",
            )
            expected = {
                "d1": data_d1.flatten(),
                "d2": data_d2.flatten(),
                "a1": self.data_a1[:, 4:8].flatten(),
                "a2": self.data_a2[:, 4:8].flatten(),
            }
        else:
            expected = {"a1": self.data_a1[:, 4:8], "a2": self.data_a2[:, 4:8]}
        assert_dict_arrays_equal(result, expected, not sparse)


class TestReadSubarraySparseArray1D(DiskTestCase):
    data_dim1 = np.linspace(-1.0, 1.0, 5)
    data_attr1 = np.arange(5, dtype=np.uint32)

    @pytest.fixture
    def array_uri(self):
        uri = self.path("test_read_subarray_array_1d")
        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(name="d1", domain=(-1.0, 1.0), tile=2.0, dtype=np.float64)
            ),
            attrs=[tiledb.Attr(name="a1", dtype=np.uint32)],
            sparse=True,
        )
        tiledb.Array.create(uri, schema)
        with tiledb.open(uri, "w") as array:
            array[self.data_dim1] = self.data_attr1
        return uri

    def test_read_full_array(self, array_uri):
        with tiledb.open(array_uri, "r") as array:
            subarray = tiledb.Subarray(array)
            subarray.add_dim_range(0, (-1.0, 1.0))
            result = array.read_subarray(subarray)

        expected = OrderedDict([("d1", self.data_dim1), ("a1", self.data_attr1)])

        assert_dict_arrays_equal(result, expected, False)

    def test_empty_result(self, array_uri):
        with tiledb.open(array_uri, "r") as array:
            subarray = tiledb.Subarray(array)
            subarray.add_dim_range(0, (-0.9, -0.89))
            result = array.read_subarray(subarray)

        expected = OrderedDict(
            [
                ("d1", np.array([], dtype=np.float64)),
                ("a1", np.array([], dtype=np.uint32)),
            ]
        )
        assert_dict_arrays_equal(result, expected, True)
