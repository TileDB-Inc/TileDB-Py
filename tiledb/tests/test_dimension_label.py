import pytest

import numpy as np

import tiledb
from tiledb.tests.common import DiskTestCase


class DimensionLabelTestCase(DiskTestCase):
    def test_add_to_array_schema(self):
        dim = tiledb.Dim("dim", domain=(1, 10))
        dom = tiledb.Domain(dim)
        att = tiledb.Attr("val", dtype=np.uint64)
        dim_labels = {
            "label": tiledb.DimLabelSchema(
                0,
                "increasing",
                label_dtype=dim.dtype,
                dim_dtype=dim.dtype,
                dim_tile=10,
            )
        }
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), dim_labels=dim_labels)

        assert schema.has_dim_label("label")
        assert not schema.has_dim_label("fake_name")

    def test_add_to_array_schema_out_of_bounds(self):
        dim = tiledb.Dim("label", domain=(1, 10))
        dom = tiledb.Domain(dim)
        att = tiledb.Attr("val", dtype=np.uint64)
        dim_labels = {
            "label": tiledb.DimLabelSchema(
                2,
                "increasing",
                label_dtype=dim.dtype,
                dim_dtype=dim.dtype,
                dim_tile=10,
            )
        }

        with pytest.raises(tiledb.TileDBError):
            tiledb.ArraySchema(domain=dom, attrs=(att,), dim_labels=dim_labels)
