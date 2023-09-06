import numpy as np
import pytest

import tiledb

from .common import DiskTestCase, assert_array_equal


class TestWriteSubarrayDense(DiskTestCase):
    @pytest.mark.parametrize("dim_dtype", [np.int32, np.uint32])
    def test_1d_full_write(self, dim_dtype):
        uri = self.path(f"dense_write_subarray_1d_{np.dtype(dim_dtype).name}")
        schema = tiledb.ArraySchema(
            tiledb.Domain(
                tiledb.Dim(name="d1", domain=(0, 999), tile=1000, dtype=dim_dtype)
            ),
            [tiledb.Attr(name="", dtype=np.float64)],
        )
        tiledb.Array.create(uri, schema)

        data = np.random.rand(1000)
        with tiledb.open(uri, "w") as array:
            subarray = tiledb.Subarray(array)
            subarray.add_dim_range(0, (0, 999))
            array.write_subarray(subarray, data)

        #
        with tiledb.open(uri, "r") as array:
            result = array[...]

        assert_array_equal(result, data)
