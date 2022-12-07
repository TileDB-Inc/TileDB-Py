import numpy as np

import tiledb
from tiledb.tests.common import DiskTestCase


class DimensionLabelTestCase(DiskTestCase):
    def test_add_to_array_schema(self):
        dim = tiledb.Dim("dim", domain=(1, 10))
        dom = tiledb.Domain(dim)
        att = tiledb.Attr("val", dtype=np.uint64)
        dim_labels = {
            "dim": tiledb.DimLabelSchema(
                0,
                "increasing",
                label_dtype=dim.dtype,
                dim_dtype=dim.dtype,
                dim_tile=10,
            )
        }
        sch = tiledb.ArraySchema(domain=dom, attrs=(att,), dim_labels=dim_labels)

        assert sch.has_dim_label("dim") == True
        assert sch.has_dim_label("dne") == False
