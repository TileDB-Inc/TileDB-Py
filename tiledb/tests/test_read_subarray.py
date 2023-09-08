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

    def test_read_multiple_ranges(self, array_uri):
        with tiledb.open(array_uri, "r") as array:
            sparse = array.schema.sparse
            subarray = tiledb.Subarray(array)
            subarray.add_dim_range(0, (3, 3))
            subarray.add_dim_range(0, (1, 2))
            subarray.add_dim_range(0, (5, 10))
            result = array.read_subarray(subarray)
        if sparse:
            assert len(result) == 3
            a1_expected = self.data1[result["d1"]]
            a2_expected = self.data2[result["d1"]]
        else:
            assert len(result) == 2
            a1_expected = np.hstack((self.data1[3], self.data1[1:3], self.data1[5:11]))
            a2_expected = np.hstack((self.data2[3], self.data2[1:3], self.data2[5:11]))

        assert_array_equal(result["a1"], a1_expected)
        assert_array_equal(result["a2"], a2_expected)

    def test_read_single_attr(self, array_uri):
        with tiledb.open(array_uri, attr="a1", mode="r") as array:
            sparse = array.schema.sparse
            subarray = tiledb.Subarray(array)
            subarray.add_dim_range(0, (10, 20))
            result = array.read_subarray(subarray)
        assert_unordered_equal(result["a1"], self.data1[10:21], unordered=sparse)
        if sparse:
            assert_unordered_equal(result["d1"], np.arange(10, 21), True)
            assert len(result) == 2
        else:
            assert len(result) == 1

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

    def test_read_by_label_no_data(self, array_uri, dim_dtype):
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


@pytest.mark.parametrize("sparse", (True, False))
class TestReadSubarray2D(DiskTestCase):
    data_a1 = np.random.rand(121).reshape(11, 11)
    data_a2 = np.random.randint(-1000, 1000, (11, 11), dtype=np.int16)
    data_l1 = np.arange(-5, 6)
    data_l2 = np.arange(5, -6, -1)

    @pytest.fixture
    def array_uri(self, sparse):
        """Create TileDB array, write data, and return the URI."""
        suffix = "2d_label_sparse" if sparse else "2d_label_dense"
        uri = self.path(f"read_subarray_{suffix}")
        dim1 = tiledb.Dim(name="d1", domain=(0, 10), tile=11, dtype=np.int32)
        dim2 = tiledb.Dim(name="d2", domain=(0, 10), tile=11, dtype=np.int32)
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
            coords_d1, coords_d2 = np.meshgrid(
                np.arange(11), np.arange(11), indexing="ij"
            )
            with tiledb.open(uri, "w") as array:
                array[coords_d1.flatten(), coords_d2.flatten()] = {
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
            subarray.add_dim_range(0, (0, 10))
            subarray.add_dim_range(0, (0, 10))
            result = array.read_subarray(subarray)
        if sparse:
            assert_unordered_equal(
                result["a1"], self.data_a1.flatten(), unordered=sparse
            )
            assert_unordered_equal(
                result["a2"], self.data_a2.flatten(), unordered=sparse
            )
            data_d1, data_d2 = np.meshgrid(np.arange(11), np.arange(11), indexing="ij")
            assert_unordered_equal(result["d1"], data_d1.flatten(), True)
            assert_unordered_equal(result["d2"], data_d2.flatten(), True)
            assert len(result) == 4
        else:
            assert len(result) == 2

    def test_read_mixed_ranges(self, array_uri):
        with tiledb.open(array_uri) as array:
            sparse = array.schema.sparse
            subarray = tiledb.Subarray(array)
            subarray.add_dim_range(0, (0, 2))
            result = array.read_subarray(subarray)
        if sparse:
            data_d1, data_d2 = np.meshgrid(np.arange(3), np.arange(11), indexing="ij")
            assert_unordered_equal(result["a1"], self.data_a1[0:3, :].flatten(), True)
            assert_unordered_equal(result["a2"], self.data_a2[0:3, :].flatten(), True)
            assert_unordered_equal(result["d1"], data_d1.flatten(), True)
            assert_unordered_equal(result["d2"], data_d2.flatten(), True)
            assert len(result) == 4
        else:
            assert_unordered_equal(result["a1"], self.data_a1[0:3, :], unordered=sparse)
            assert_unordered_equal(result["a2"], self.data_a2[0:3, :], unordered=sparse)
            assert len(result) == 2
