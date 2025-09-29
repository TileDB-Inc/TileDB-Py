import numpy as np
import pytest

import tiledb
import tiledb.main as core
from tiledb import TileDBError
from tiledb.tests.common import DiskTestCase


class SubarrayTest(DiskTestCase):
    def test_add_range(self):
        dim1 = tiledb.Dim("row", domain=(1, 10))
        dim2 = tiledb.Dim("col", domain=(1, 10))
        dom = tiledb.Domain(dim1, dim2)
        att = tiledb.Attr("val", dtype=np.uint64)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,))
        uri = self.path("dense_array")
        tiledb.Array.create(uri, schema)

        with tiledb.open(uri, "w") as array:
            array[1:5, 1:5] = np.reshape(np.arange(1, 17, dtype=np.float64), (4, 4))

        with tiledb.open(uri, "r") as array:
            subarray1 = tiledb.Subarray(array)

            # Check number of ranges: each dimension should have the default range.
            assert subarray1.num_dim_ranges(0) == 1
            assert subarray1.num_dim_ranges(1) == 1
            assert subarray1.shape() == (10, 10)

            # Add range to first dim and check still only 1 range (replace default).
            subarray1.add_dim_range(0, (1, 2))
            assert subarray1.num_dim_ranges(0) == 1
            assert subarray1.shape() == (2, 10)

            # Add additional range to first dim and check 2 ranges.
            subarray1.add_dim_range(0, (4, 4))
            assert subarray1.num_dim_ranges(0) == 2
            assert subarray1.shape() == (3, 10)

    def test_add_ranges_basic(self):
        uri = self.path("test_pyquery_basic")
        with tiledb.from_numpy(uri, np.random.rand(4)):
            pass

        with tiledb.open(uri) as array:
            subarray = tiledb.Subarray(array)

            subarray.add_ranges([[(0, 3)]])

            with self.assertRaises(TileDBError):
                subarray.add_ranges([[(0, 3.0)]])

            subarray.add_ranges([[(0, np.int32(3))]])

            with self.assertRaises(TileDBError):
                subarray.add_ranges([[(3, "a")]])

            with self.assertRaisesRegex(
                TileDBError,
                "Failed to cast dim range '\\(1.2344, 5.6789\\)' to dim type UINT64.*$",
            ):
                subarray.add_ranges([[(1.2344, 5.6789)]])

            with self.assertRaisesRegex(
                TileDBError,
                "Failed to cast dim range '\\('aa', 'bbbb'\\)' to dim type UINT64.*$",
            ):
                subarray.add_ranges([[("aa", "bbbb")]])

    @pytest.mark.skipif(
        tiledb.libtiledb.version() < (2, 15),
        reason="dimension labels requires libtiledb version 2.15 or greater",
    )
    def test_add_label_ranges_1d(self):
        # Create array schema with dimension labels
        dim = tiledb.Dim("d1", domain=(1, 10), dtype=np.uint32)
        dom = tiledb.Domain(dim)
        att = tiledb.Attr("a1", dtype=np.int64)
        dim_labels = {0: {"l1": dim.create_label_schema("increasing", np.int64)}}
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), dim_labels=dim_labels)

        # Create array
        uri = self.path("dense_array_with_label")
        tiledb.Array.create(uri, schema)

        # Add dimension label ranges
        with tiledb.open(uri, "r") as array:
            subarray1 = tiledb.Subarray(array)
            assert subarray1.num_dim_ranges(0) == 1

            subarray1.add_label_range("l1", (-1, 1))
            assert subarray1.num_dim_ranges(0) == 0
            assert subarray1.num_label_ranges("l1") == 1

    def test_get_range_fixed(self):
        dim1 = tiledb.Dim("row", domain=(1, 10))
        dim2 = tiledb.Dim("col", domain=(1, 10))
        dom = tiledb.Domain(dim1, dim2)
        att = tiledb.Attr("val", dtype=np.uint64)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,))
        uri = self.path("dense_array")
        tiledb.Array.create(uri, schema)
        with tiledb.open(uri, "w") as array:
            array[1:5, 1:5] = np.reshape(np.arange(1, 17, dtype=np.float64), (4, 4))
        with tiledb.open(uri, "r") as array:
            subarray = tiledb.Subarray(array)

            # Add range to first dim and check
            subarray.add_dim_range(0, (1, 2))
            assert subarray.get_range(0, 0) == [1, 2, 0]  # [start, end, stride]
            assert subarray.get_range(1, 0) == [1, 10, 0]

            # Add range to second dim and check
            subarray.add_dim_range(1, (3, 4))
            assert subarray.get_range(0, 0) == [1, 2, 0]
            assert subarray.get_range(1, 0) == [3, 4, 0]

    def test_get_range_var(self):
        # create array with string dimension
        dim1 = tiledb.Dim("d1", dtype="ascii")
        dom = tiledb.Domain(dim1)
        att = tiledb.Attr("a1", dtype=np.int64)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True)
        uri = self.path("var_array")
        tiledb.Array.create(uri, schema)

        with tiledb.open(uri, "w") as array:
            array["a"] = np.array([1], dtype=np.int64)
            array["b"] = np.array([2], dtype=np.int64)
            array["c"] = np.array([3], dtype=np.int64)

        with tiledb.open(uri, "r") as array:
            subarray = tiledb.Subarray(array)

            # Add range to first dim and check
            subarray.add_dim_range(0, ("a", "b"))
            assert subarray.get_range(0, 0) == ["a", "b"]  # [start, end]

            # check that assert subarray.get_range(0, 1) throws an error
            with pytest.raises(TileDBError):
                subarray.get_range(0, 1)

    def test_copy_ranges(self):
        # Create array schema with dimension labels
        d1 = tiledb.Dim("d1", domain=(1, 10), dtype=np.uint32)
        d2 = tiledb.Dim("d2", domain=(1, 10), dtype=np.uint32)
        d3 = tiledb.Dim("d3", domain=(1, 10), dtype=np.uint32)
        d4 = tiledb.Dim("d4", domain=(1, 10), dtype=np.uint32)
        dom = tiledb.Domain(d1, d2, d3, d4)
        att = tiledb.Attr("a1", dtype=np.int64)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,))

        # Create array
        uri = self.path("array4d")
        tiledb.Array.create(uri, schema)

        # Add ranges (1, 1) and (3, 3) to each dimension of the subarray.
        with tiledb.open(uri, "r") as array:
            subarray1 = tiledb.Subarray(array)
            subarray1.add_ranges(
                (((1, 1), (3, 3)), ((1, 1), (3, 3)), ((1, 1), (3, 3)), ((1, 1), (3, 3)))
            )
            assert subarray1.num_dim_ranges(0) == 2
            assert subarray1.num_dim_ranges(1) == 2
            assert subarray1.num_dim_ranges(2) == 2
            assert subarray1.num_dim_ranges(3) == 2

            # Should copy ranges from d1 and d3.
            # All other dimensions should only have default range.
            subarray2 = tiledb.Subarray(array)
            subarray2.copy_ranges(subarray1, [0, 2])
            assert subarray2.num_dim_ranges(0) == 2
            assert subarray2.num_dim_ranges(1) == 1
            assert subarray2.num_dim_ranges(2) == 2
            assert subarray2.num_dim_ranges(3) == 1

    def test_add_fixed_sized_point_ranges(self):
        # Create a 1D int dimension array
        dim = tiledb.Dim(name="d", dtype=np.int32, domain=(0, 100))
        dom = tiledb.Domain(dim)
        att = tiledb.Attr("a", dtype=np.int32)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,))
        uri = self.path("int_dim_array")
        tiledb.Array.create(uri, schema)

        # Write some data
        coords = np.array([1, 2, 3, 4], dtype=np.int32)
        data = np.array([10, 20, 30, 40], dtype=np.int32)
        with tiledb.open(uri, "w") as A:
            A[coords] = data

        with tiledb.open(uri, "r") as A:
            sub = tiledb.Subarray(A)
            # Add point ranges as numpy array
            points = np.array([1, 3], dtype=np.int32)
            sub.add_ranges([points])
            assert sub.num_dim_ranges(0) == 2

            q = core.PyQuery(A.ctx, A, ("a",), (), 0, False)
            q.set_subarray(sub)
            q.submit()

            results = q.results()
            arr = results["a"][0]
            arr.dtype = np.int32
            assert arr.shape == (2,)
            assert np.array_equal(arr, np.array([10, 30], dtype=np.int32))

    def test_add_var_sized_point_ranges(self):
        # Create a 1D string dimension array
        dim = tiledb.Dim(name="d", dtype=np.bytes_)
        dom = tiledb.Domain(dim)
        att = tiledb.Attr("a", dtype=np.int32)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True)
        uri = self.path("str_dim_array")
        tiledb.Array.create(uri, schema)

        # Write some data
        coords = np.array([b"aa", b"b", b"ccc", b"dd"], dtype=np.bytes_)
        data = np.array([10, 20, 30, 40], dtype=np.int32)
        with tiledb.open(uri, "w") as A:
            A[coords] = data

        with tiledb.open(uri, "r") as A:
            sub = tiledb.Subarray(A)
            # Add point ranges as numpy array
            points = np.array([b"aa", b"ccc"], dtype=np.bytes_)
            sub.add_ranges([points])
            assert sub.num_dim_ranges(0) == 2

            q = core.PyQuery(A.ctx, A, ("a",), (), 0, False)
            q.set_subarray(sub)
            q.submit()

            results = q.results()
            arr = results["a"][0]
            arr.dtype = np.int32
            assert arr.shape == (2,)
            assert np.array_equal(arr, np.array([10, 30], dtype=np.int32))
