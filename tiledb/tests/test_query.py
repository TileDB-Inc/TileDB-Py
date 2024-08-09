import numpy as np
import pytest

import tiledb
from tiledb.tests.common import DiskTestCase


class QueryTest(DiskTestCase):
    @pytest.mark.skipif(
        tiledb.libtiledb.version()[0] == 2 and tiledb.libtiledb.version()[1] < 15,
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
            query.submit()
            output_subarray = query.subarray()
            assert output_subarray.num_dim_ranges(0) == 2
