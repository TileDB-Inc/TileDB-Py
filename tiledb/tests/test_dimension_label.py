import numpy as np
import pytest

import tiledb
from tiledb.tests.common import DiskTestCase


class DimensionLabelTestCase(DiskTestCase):
    def test_dim_label_schema(self):
        dim_label_schema = tiledb.DimLabelSchema(
            0, "decreasing", label_dtype=np.float64, dim_dtype=np.int32
        )
        assert dim_label_schema.label_order == "decreasing"
        assert dim_label_schema.label_dtype == np.float64
        assert dim_label_schema.dim_dtype == np.int32
        assert dim_label_schema.dim_tile is None
        assert dim_label_schema.label_filters is None

        filter = tiledb.FilterList()
        dim_label_schema = tiledb.DimLabelSchema(
            10,
            "increasing",
            label_dtype=np.float32,
            dim_dtype=np.int64,
            dim_tile=20,
            label_filters=filter,
        )
        assert dim_label_schema.label_order == "increasing"
        assert dim_label_schema.label_dtype == np.float32
        assert dim_label_schema.dim_dtype == np.int64
        assert dim_label_schema.dim_tile == 20
        assert dim_label_schema.label_filters == filter

    @pytest.mark.skipif(
        tiledb.libtiledb.version()[0] == 2 and tiledb.libtiledb.version()[1] < 15,
        reason="dimension labels requires libtiledb version 2.15 or greater",
    )
    def test_add_to_array_schema(self):
        dim = tiledb.Dim("dim", domain=(1, 10))
        dom = tiledb.Domain(dim)
        att = tiledb.Attr("val", dtype=np.uint64)
        filters = tiledb.FilterList([tiledb.ZstdFilter(10)])
        dim_labels = {
            "l1": tiledb.DimLabelSchema(
                0,
                "increasing",
                label_dtype=np.float64,
                dim_dtype=dim.dtype,
                dim_tile=10,
                label_filters=filters,
            )
        }
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), dim_labels=dim_labels)
        assert schema.has_dim_label("l1")
        assert not schema.has_dim_label("fake_name")

        # Check the dimension label properties
        dim_label = schema.dim_label("l1")
        assert dim_label.dtype == np.float64
        assert not dim_label.isvar
        assert not dim_label.isascii

        # Create array check values in dimension label schema
        uri = self.path("array_with_label")
        tiledb.Array.create(uri, schema)

        # Load the array schema for the dimension label
        base_array_schema = tiledb.ArraySchema.load(uri)
        dim_label = base_array_schema.dim_label("l1")
        label_array_schema = tiledb.ArraySchema.load(dim_label.uri)

        # Chack the array schema for the dimension label
        label_dim = label_array_schema.domain.dim(0)
        assert label_dim.tile == 10
        assert label_dim.dtype == np.uint64
        # TODO: Adjust the attr name to dim_label.label_attr_name after #1640
        # is merged
        label_attr = label_array_schema.attr("label")
        assert label_attr.dtype == np.float64
        assert label_attr.filters == filters

    @pytest.mark.skipif(
        tiledb.libtiledb.version()[0] == 2 and tiledb.libtiledb.version()[1] < 15,
        reason="dimension labels requires libtiledb version 2.15 or greater",
    )
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
            )
        }

        with pytest.raises(tiledb.TileDBError):
            tiledb.ArraySchema(domain=dom, attrs=(att,), dim_labels=dim_labels)

    @pytest.mark.skipif(
        tiledb.libtiledb.version()[0] == 2 and tiledb.libtiledb.version()[1] < 15,
        reason="dimension labels requires libtiledb version 2.15 or greater",
    )
    def test_add_to_array_schema_dim_dtype_mismatch(self):
        dim = tiledb.Dim("label", domain=(1, 10))
        dom = tiledb.Domain(dim)
        att = tiledb.Attr("val", dtype=np.uint64)
        dim_labels = {
            "label": tiledb.DimLabelSchema(
                2,
                "increasing",
                label_dtype=dim.dtype,
                dim_dtype=np.int32,
            )
        }

        with pytest.raises(tiledb.TileDBError):
            tiledb.ArraySchema(domain=dom, attrs=(att,), dim_labels=dim_labels)

    @pytest.mark.skipif(
        tiledb.libtiledb.version()[0] == 2 and tiledb.libtiledb.version()[1] < 15,
        reason="dimension labels requires libtiledb version 2.15 or greater",
    )
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

    @pytest.mark.skipif(
        tiledb.libtiledb.version()[0] == 2 and tiledb.libtiledb.version()[1] < 15,
        reason="dimension labels requires libtiledb version 2.15 or greater",
    )
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
