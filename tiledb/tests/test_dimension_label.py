import numpy as np
import pytest

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

        # Check the dimension label properties
        dim_label = schema.dim_label("label")
        assert dim_label.dtype == np.uint64
        assert not dim_label.isvar
        assert not dim_label.isascii
        assert dim_label.uri == "__labels/l0"

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

    def test_dimension_label_round_trip_dense_array(self):
        # Create array schema with dimension labels
        dim = tiledb.Dim("index", domain=(1, 10))
        dom = tiledb.Domain(dim)
        att = tiledb.Attr("value", dtype=np.int64)
        dim_labels = {
            "label": tiledb.DimLabelSchema(
                0,
                "increasing",
                label_dtype=np.int64,
                dim_dtype=dim.dtype,
            )
        }
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), dim_labels=dim_labels)

        # Create array
        uri = self.path("dense_array_with_label")
        tiledb.Array.create(uri, schema)

        # Write data to the array and the label
        attr_data = np.arange(11, 21)
        label_data = np.arange(-10, 0)
        with tiledb.open(uri, "w") as array:
            array[:] = {"value": attr_data, "label": label_data}

        # Load the array schema and get the URI of the dimension label
        schema = tiledb.ArraySchema.load(uri)
        dim_label = schema.dim_label("label")

        # Read and check the data directly from the dimension label
        with tiledb.open(dim_label.uri, "r") as L1:
            output_label_data = L1[:]["label"]
            np.testing.assert_array_equal(output_label_data, label_data)

    def test_dimension_label_round_trip_sparse_array(self):
        # Create array schema with dimension labels
        dim = tiledb.Dim("index", domain=(1, 10))
        dom = tiledb.Domain(dim)
        att = tiledb.Attr("value", dtype=np.int64)
        dim_labels = {
            "label": tiledb.DimLabelSchema(
                0,
                "increasing",
                label_dtype=np.int64,
                dim_dtype=dim.dtype,
            )
        }
        schema = tiledb.ArraySchema(
            domain=dom, attrs=(att,), dim_labels=dim_labels, sparse=True
        )

        # Create array
        uri = self.path("sparse_array_with_label")
        tiledb.Array.create(uri, schema)

        # Write data to the array and the label
        index_data = np.arange(1, 11)
        attr_data = np.arange(11, 21)
        label_data = np.arange(-10, 0)
        with tiledb.open(uri, "w") as array:
            array[index_data] = {"value": attr_data, "label": label_data}

        # Load the array schema and get the URI of the dimension label
        schema = tiledb.ArraySchema.load(uri)
        dim_label = schema.dim_label("label")

        # Read and check the data directly from the dimension label
        with tiledb.open(dim_label.uri, "r") as L1:
            output_label_data = L1[:]["label"]
            np.testing.assert_array_equal(output_label_data, label_data)
