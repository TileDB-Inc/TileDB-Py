import numpy as np
import pytest

import tiledb
from tiledb.tests.common import DiskTestCase


class DimensionLabelTestCase(DiskTestCase):
    @pytest.mark.skipif(
        tiledb.libtiledb.version()[0] == 2 and tiledb.libtiledb.version()[1] < 15,
        reason="dimension labels requires libtiledb version 2.15 or greater",
    )
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
                dim_tile=10,
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
        dim = tiledb.Dim("d1", domain=(1, 10))
        dom = tiledb.Domain(dim)
        att = tiledb.Attr("a1", dtype=np.int64)
        dim_labels = {
            "l1": tiledb.DimLabelSchema(
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
            array[:] = {"a1": attr_data, "l1": label_data}

        # Load the array schema and get the URI of the dimension label
        schema = tiledb.ArraySchema.load(uri)
        dim_label = schema.dim_label("l1")

        # Read and check the data directly from the dimension label
        with tiledb.open(dim_label.uri, "r") as L1:
            output_data = L1[:]
            output_label_data = output_data["label"]
            np.testing.assert_array_equal(output_label_data, label_data)

        # Read and check the data using label indexer on parent array
        with tiledb.open(uri, "r") as array:
            indexer = array.label_index(["l1"])
            output_attr_data = indexer[-100:100]["a1"]
            np.testing.assert_array_equal(output_attr_data, attr_data)
            for index in range(10):
                label_index = label_data[index]
                attr_value = indexer[label_index:label_index]["a1"][0]
                assert attr_value == attr_data[index]
            np.testing.assert_array_equal(output_label_data, label_data)

    @pytest.mark.skipif(
        tiledb.libtiledb.version()[0] == 2 and tiledb.libtiledb.version()[1] < 15,
        reason="dimension labels requires libtiledb version 2.15 or greater",
    )
    def test_dimension_label_round_trip_multidim_dense_array(self):
        # Create array schema with dimension labels
        dim1 = tiledb.Dim("x_index", domain=(1, 8))
        dim2 = tiledb.Dim("y_index", domain=(1, 8))
        dom = tiledb.Domain(dim1, dim2)
        att = tiledb.Attr("value", dtype=np.int64)
        dim_labels = {
            "x1": tiledb.DimLabelSchema(
                0,
                "increasing",
                label_dtype=np.int64,
                dim_dtype=dim1.dtype,
            ),
            "x2": tiledb.DimLabelSchema(
                0,
                "decreasing",
                label_dtype=np.int64,
                dim_dtype=dim1.dtype,
            ),
            "y1": tiledb.DimLabelSchema(
                0,
                "increasing",
                label_dtype=np.int64,
                dim_dtype=dim2.dtype,
            ),
        }
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), dim_labels=dim_labels)

        # Create array
        uri = self.path("dense_array_with_label")
        tiledb.Array.create(uri, schema)

        # Write data to the array and the label
        attr_data = np.reshape(np.arange(1, 65), (8, 8))
        x1_data = np.arange(9, 17)
        x2_data = np.arange(8, 0, -1)
        y1_data = np.arange(9, 17)
        with tiledb.open(uri, "w") as array:
            array[:, :] = {
                "value": attr_data,
                "x1": x1_data,
                "y1": y1_data,
                "x2": x2_data,
            }

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
