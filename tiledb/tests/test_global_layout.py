import numpy as np
import pytest

import tiledb

from .common import DiskTestCase, assert_array_equal


class TestGlobalLayout(DiskTestCase):

    def test_open_with_layout_parameter(self):
        """
        Test that the order parameter is correctly accepted and stored
        when opening arrays with tiledb.open().
        """
        uri = self.path("test_global_layout_parameter")

        # Create a simple 1D sparse array
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=5, dtype=np.uint64))
        schema = tiledb.ArraySchema(
            domain=dom, sparse=True, attrs=(tiledb.Attr(dtype=np.int64),)
        )
        tiledb.Array.create(uri, schema)

        # Test different layout parameters
        test_layouts = [
            "G",
            "global",
            "U",
            "unordered",
            "C",
            "row-major",
            "R",
            "col-major",
            None,
        ]

        for layout in test_layouts:
            with tiledb.open(uri, mode="w", order=layout) as A:
                assert A.order == layout, f"Expected layout {layout}, got {A.order}"

    def test_sparse_global_layout_1d_success(self):
        """
        Test successful global layout writes for 1D sparse arrays.
        Coordinates must be provided in ascending order for 1D arrays.
        """
        uri = self.path("test_sparse_global_1d_success")

        # Create 1D sparse array
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=5, dtype=np.uint64))
        schema = tiledb.ArraySchema(
            domain=dom, sparse=True, attrs=(tiledb.Attr(dtype=np.int64),)
        )
        tiledb.Array.create(uri, schema)

        # Write data in global order (ascending coordinates for 1D)
        expected_coords = [0, 1, 3, 5, 7, 9]
        expected_data = [100, 200, 300, 400, 500, 600]

        with tiledb.open(uri, mode="w", order="G") as A:
            A[expected_coords] = expected_data

        # Verify the data was written correctly
        with tiledb.open(uri, mode="r") as A:
            result = A[:]
            actual_coords = result["__dim_0"].tolist()
            actual_data = result[""].tolist()

            assert actual_coords == expected_coords
            assert actual_data == expected_data

    def test_sparse_global_layout_1d_error_on_wrong_order(self):
        """
        Test that global layout correctly validates coordinate ordering
        and raises an error when coordinates are not in global order.
        """
        uri = self.path("test_sparse_global_1d_error")

        # Create 1D sparse array
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=5, dtype=np.uint64))
        schema = tiledb.ArraySchema(
            domain=dom, sparse=True, attrs=(tiledb.Attr(dtype=np.int64),)
        )
        tiledb.Array.create(uri, schema)

        # Try to write coordinates NOT in global order - should fail
        with pytest.raises(tiledb.TileDBError, match="global order"):
            with tiledb.open(uri, mode="w", order="G") as A:
                # Coordinates are not in ascending order
                A[[3, 1, 5]] = [30, 10, 50]

    def test_multi_attribute_global_layout(self):
        """
        Test global layout writes with multiple attributes.
        """
        uri = self.path("test_multi_attr_global")

        # Create sparse array with multiple attributes
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=5, dtype=np.uint64))
        schema = tiledb.ArraySchema(
            domain=dom,
            sparse=True,
            attrs=(
                tiledb.Attr(name="attr1", dtype=np.int64),
                tiledb.Attr(name="attr2", dtype=np.float64),
            ),
        )
        tiledb.Array.create(uri, schema)

        # Write data in global order
        coords = [0, 2, 4, 6]
        data = {
            "attr1": np.array([100, 200, 300, 400]),
            "attr2": np.array([1.1, 2.2, 3.3, 4.4]),
        }

        with tiledb.open(uri, mode="w", order="G") as A:
            A[coords] = data

        # Verify data was written correctly
        with tiledb.open(uri, mode="r") as A:
            result = A[:]
            assert result["__dim_0"].tolist() == coords
            assert result["attr1"].tolist() == data["attr1"].tolist()
            assert_array_equal(result["attr2"], data["attr2"])

    @pytest.mark.parametrize("layout_spec", ["C", "R", "row-major", "col-major"])
    def test_row_col_major_layouts(self, layout_spec):
        """
        Test dense with row-major and column-major layout specifications.

        """
        uri = self.path(f"test_layout_{layout_spec.replace('-', '_')}")

        # Create 2D dense array
        dom = tiledb.Domain(
            tiledb.Dim(domain=(0, 2), tile=3, dtype=np.uint64),
            tiledb.Dim(domain=(0, 2), tile=3, dtype=np.uint64),
        )
        schema = tiledb.ArraySchema(domain=dom, attrs=(tiledb.Attr(dtype=np.int64),))
        tiledb.Array.create(uri, schema)

        test_data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

        with tiledb.open(uri, mode="w", order=layout_spec) as A:
            assert A.order == layout_spec
            A[:] = test_data

        # Verify data was written correctly
        with tiledb.open(uri, mode="r") as A:
            result = A[:]

            if layout_spec in ("col-major", "R"):
                # Column-major layouts store data in transposed form
                expected_result = test_data.T
            else:
                # Row-major layouts preserve the original form
                expected_result = test_data

            assert_array_equal(result, expected_result)

    def test_global_vs_unordered_fragment_comparison(self):
        """
        Global layout writes all data to a single fragment.
        Unordered layout (and the rest of the layouts) create separate fragments on each query submission.
        """
        uri_global = self.path("test_fragment_comparison_global")
        uri_unordered = self.path("test_fragment_comparison_unordered")

        # Create identical sparse arrays
        for uri in [uri_global, uri_unordered]:
            dom = tiledb.Domain(tiledb.Dim(domain=(0, 99), tile=10, dtype=np.uint64))
            schema = tiledb.ArraySchema(
                domain=dom, sparse=True, attrs=(tiledb.Attr(dtype=np.int64),)
            )
            tiledb.Array.create(uri, schema)

        # Write with global layout - should append to same fragment
        with tiledb.open(uri_global, mode="w", order="G") as A:
            A[[0, 1, 2]] = [100, 200, 300]
            A[[3, 4, 5]] = [400, 500, 600]  # Continues in the same fragment
            A[[10, 15, 20]] = [1000, 1500, 2000]  # More data in the same fragment

        # Write with unordered layout - creates separate fragments
        with tiledb.open(uri_unordered, mode="w", order="U") as A:
            A[[0, 1, 2]] = [100, 200, 300]
            A[[3, 4, 5]] = [400, 500, 600]  # Separate fragment
            A[[10, 15, 20]] = [1000, 1500, 2000]  # Another separate fragment

        # Check fragment counts
        global_fragments = tiledb.array_fragments(uri_global)
        unordered_fragments = tiledb.array_fragments(uri_unordered)

        # Global writes should create one fragment
        assert (
            len(global_fragments) == 1
        ), f"Expected 1 fragment with global writes, got {len(global_fragments)}"

        # Unordered writes in create separate fragments
        assert (
            len(unordered_fragments) == 3
        ), f"Expected 3 fragments with unordered writes, got {len(unordered_fragments)}"

        # Verify both arrays have the same data despite different fragment counts
        with (
            tiledb.open(uri_global, mode="r") as A_global,
            tiledb.open(uri_unordered, mode="r") as A_unordered,
        ):

            result_global = A_global[:]
            result_unordered = A_unordered[:]

            # Both should have the same data
            assert_array_equal(result_global["__dim_0"], result_unordered["__dim_0"])
            assert_array_equal(result_global[""], result_unordered[""])
