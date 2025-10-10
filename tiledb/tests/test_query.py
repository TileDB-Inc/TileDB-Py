import numpy as np
import pytest

import tiledb
from tiledb.tests.common import DiskTestCase


class QueryTest(DiskTestCase):
    @pytest.mark.skipif(
        tiledb.libtiledb.version() < (2, 15),
        reason="dimension labels requires libtiledb version 2.15 or greater",
    )
    def test_label_range_query(self):
        # Create array schema with dimension labels
        dim = tiledb.Dim("d1", domain=(1, 10))
        dom = tiledb.Domain(dim)
        att = tiledb.Attr("a1", dtype=np.int64)
        dim_labels = {0: {"l1": dim.create_label_schema("increasing", np.int64)}}
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), dim_labels=dim_labels)

        # Create array
        uri = self.path("dense_array_with_label")
        tiledb.Array.create(uri, schema)

        # Write data to the array and the label
        attr_data = np.arange(11, 21)
        label_data = np.arange(-10, 0)
        with tiledb.open(uri, "w") as array:
            array[:] = {"a1": attr_data, "l1": label_data}

        # Read and check the data using label indexer on parent array
        with tiledb.open(uri, "r") as array:
            input_subarray = tiledb.Subarray(array)
            input_subarray.add_label_range("l1", (-10, -10))
            input_subarray.add_label_range("l1", (-8, -6))
            query = tiledb.Query(array)
            query.set_subarray(input_subarray)
            query._submit()
            output_subarray = query.subarray()
            assert output_subarray.num_dim_ranges(0) == 2

    @pytest.mark.parametrize("sparse", [True, False])
    def test_global_order_write_single_submit(self, sparse):
        """Test writing in global order with a single submit and finalize."""
        uri = self.path("test_global_order_single")

        # Create schema
        dim = tiledb.Dim("d1", domain=(1, 100), tile=10, dtype=np.int32)
        dom = tiledb.Domain(dim)
        att = tiledb.Attr("a1", dtype=np.int64)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=sparse)
        tiledb.Array.create(uri, schema)

        # Write using Query with global order
        with tiledb.open(uri, "w") as A:
            q = tiledb.Query(A, order="G")

            if sparse:
                coords = np.array([1, 5, 10, 15, 20], dtype=np.int32)
                data = np.array([100, 200, 300, 400, 500], dtype=np.int64)
                q.set_data({"d1": coords, "a1": data})
            else:
                start_coord = 1
                end_coord = 20
                data = np.arange(
                    100, 100 + (end_coord - start_coord + 1), dtype=np.int64
                )
                q.set_subarray_ranges([(start_coord, end_coord)])
                q.set_data({"a1": data})

            q.submit()
            q.finalize()

        # Verify only one fragment was created
        fragments_info = tiledb.array_fragments(uri)
        assert len(fragments_info) == 1

        # Verify data
        with tiledb.open(uri, "r") as A:
            if sparse:
                result = A[:]
                np.testing.assert_array_equal(result["a1"], data)
                np.testing.assert_array_equal(result["d1"], coords)
            else:
                result = A[start_coord:end_coord]
                np.testing.assert_array_equal(result["a1"], data[:-1])

    @pytest.mark.parametrize("sparse", [True, False])
    def test_global_order_write_multiple_submits(self, sparse):
        """Test writing in global order with multiple submits before finalize."""
        uri = self.path("test_global_order_multiple")

        # Create schema
        dim = tiledb.Dim("d1", domain=(1, 100), tile=10, dtype=np.int32)
        dom = tiledb.Domain(dim)
        att = tiledb.Attr("a1", dtype=np.int64)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=sparse)
        tiledb.Array.create(uri, schema)

        # Write using Query with global order and multiple submits
        with tiledb.open(uri, "w") as A:
            q = tiledb.Query(A, order="G")

            if sparse:
                # First submit
                coords_batch1 = np.array([1, 5, 10], dtype=np.int32)
                data_batch1 = np.array([100, 200, 300], dtype=np.int64)
                q.set_data({"d1": coords_batch1, "a1": data_batch1})
                q.submit()

                # Second submit
                coords_batch2 = np.array([15, 20], dtype=np.int32)
                data_batch2 = np.array([400, 500], dtype=np.int64)
                q.set_data({"d1": coords_batch2, "a1": data_batch2})
                q.submit()
            else:
                # For dense arrays, set subarray once to cover full range
                start_coord = 1
                end_coord = 20
                q.set_subarray_ranges([(start_coord, end_coord)])

                # First submit - first batch of cells
                mid_point = 10
                data_batch1 = np.arange(100, 100 + mid_point, dtype=np.int64)
                q.set_data({"a1": data_batch1})
                q.submit()

                # Second submit - second batch of cells
                data_batch2 = np.arange(
                    100 + mid_point, 100 + (end_coord - start_coord + 1), dtype=np.int64
                )
                q.set_data({"a1": data_batch2})
                q.submit()

            q.finalize()

        # Verify only one fragment was created
        fragments_info = tiledb.array_fragments(uri)
        assert len(fragments_info) == 1

        # Verify data
        with tiledb.open(uri, "r") as A:
            if sparse:
                result = A[:]
                expected_data = np.array([100, 200, 300, 400, 500], dtype=np.int64)
                expected_coords = np.array([1, 5, 10, 15, 20], dtype=np.int32)
                np.testing.assert_array_equal(result["a1"], expected_data)
                np.testing.assert_array_equal(result["d1"], expected_coords)
            else:
                result = A[1:20]
                expected_data = np.arange(100, 120, dtype=np.int64)
                np.testing.assert_array_equal(result["a1"], expected_data[:-1])
