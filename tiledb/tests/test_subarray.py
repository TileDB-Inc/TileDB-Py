import numpy as np
import pytest

import tiledb
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
        tiledb.libtiledb.version()[0] == 2 and tiledb.libtiledb.version()[1] < 15,
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
