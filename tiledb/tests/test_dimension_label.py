import numpy as np
import pytest

import tiledb
from tiledb.tests.common import DiskTestCase


class DimensionLabelTestCase(DiskTestCase):
    def test_dim_label_schema(self):
        dim_label_schema = tiledb.DimLabelSchema(
            "decreasing", label_dtype=np.float64, dim_dtype=np.int32
        )
        assert dim_label_schema.label_order == "decreasing"
        assert dim_label_schema.label_dtype == np.float64
        assert dim_label_schema.dim_dtype == np.int32
        assert dim_label_schema.dim_tile is None
        assert dim_label_schema.label_filters is None

        filter = tiledb.FilterList()
        dim_label_schema = tiledb.DimLabelSchema(
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

    def test_dim_label_schema_from_dim(self):
        dim = tiledb.Dim("dim", domain=(1, 10), dtype=np.int32, tile=10)
        dim_label_schema = dim.create_label_schema("decreasing", np.float64)
        assert dim_label_schema.label_order == "decreasing"
        assert dim_label_schema.label_dtype == np.float64
        assert dim_label_schema.dim_dtype == np.int32
        assert dim_label_schema.dim_tile == 10
        assert dim_label_schema.label_filters is None

        filter = tiledb.FilterList()
        dim_label_schema = dim.create_label_schema(
            order="increasing", dtype=np.float32, tile=5, filters=filter
        )
        assert dim_label_schema.label_order == "increasing"
        assert dim_label_schema.label_dtype == np.float32
        assert dim_label_schema.dim_dtype == np.int32
        assert dim_label_schema.dim_tile == 5
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
            0: {
                "l1": tiledb.DimLabelSchema(
                    "increasing",
                    label_dtype=np.float64,
                    dim_dtype=dim.dtype,
                    dim_tile=10,
                    label_filters=filters,
                )
            }
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
        label_attr = label_array_schema.attr(dim_label.label_attr_name)
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
            2: {
                "l1": tiledb.DimLabelSchema(
                    "increasing", label_dtype=dim.dtype, dim_dtype=dim.dtype
                )
            }
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
            2: {
                "label": tiledb.DimLabelSchema(
                    "increasing", label_dtype=dim.dtype, dim_dtype=np.int32
                )
            }
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
        dim_labels = {0: {"l1": dim.create_label_schema("increasing", np.int64)}}
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), dim_labels=dim_labels)

        # Create array
        uri = self.path("dense_array_with_label")
        tiledb.Array.create(uri, schema)

        # Write data to the array and the label
        attr_data = np.arange(1, 11)
        label_data = np.arange(-9, 10, 2)
        with tiledb.open(uri, "w") as array:
            array[:] = {"a1": attr_data, "l1": label_data}

        # Load the array schema and get the URI of the dimension label
        schema = tiledb.ArraySchema.load(uri)
        dim_label = schema.dim_label("l1")

        # Read and check the data directly from the dimension label
        with tiledb.open(dim_label.uri, "r") as L1:
            output_data = L1[:]
            output_label_data = output_data[dim_label.label_attr_name]
            np.testing.assert_array_equal(output_label_data, label_data)

        # Read and check the data using label indexer on parent array
        with tiledb.open(uri, "r") as array:
            indexer = array.label_index(["l1"])

            # Read full array
            result = indexer[-100:100]
            np.testing.assert_array_equal(result["a1"], attr_data)
            np.testing.assert_array_equal(result["l1"], label_data)

            # Read each individual index
            for index in range(10):
                label_index = label_data[index]
                result = indexer[label_index:label_index]
                assert result["a1"][0] == attr_data[index]
                assert result["l1"][0] == label_index

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
            0: {
                "x1": dim1.create_label_schema("increasing", np.float64),
                "x2": dim1.create_label_schema("decreasing", np.int64),
            },
            1: {
                "y1": dim2.create_label_schema("increasing", np.int64),
            },
        }
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), dim_labels=dim_labels)

        # Create array
        uri = self.path("dense_array_with_label")
        tiledb.Array.create(uri, schema)

        # Write data to the array and the label
        attr_data = np.reshape(np.arange(1, 65), (8, 8))
        x1_data = np.linspace(-1.0, 1.0, 8)
        x2_data = np.arange(8, 0, -1)
        y1_data = np.arange(9, 17)
        with tiledb.open(uri, "w") as array:
            array[:, :] = {
                "value": attr_data,
                "x1": x1_data,
                "y1": y1_data,
                "x2": x2_data,
            }

        # Test querying by label
        with tiledb.open(uri, "r") as array:
            # Read full array: labels on both ranges
            result = array.label_index(["x1", "y1"])[-1.0:1.0, 9:17]
            np.testing.assert_array_equal(result["value"], attr_data)
            np.testing.assert_array_equal(result["x1"], x1_data)
            np.testing.assert_array_equal(result["y1"], y1_data)
            assert "x2" not in result

            # Read full array: label only on first range
            result = array.label_index(["x2"])[0:8]
            np.testing.assert_array_equal(result["value"], attr_data)
            np.testing.assert_array_equal(result["x2"], x2_data)
            assert "x1" not in result
            assert "y1" not in result

            # Read full array: Label only on second range
            result = array.label_index(["y1"])[:, 9:17]
            np.testing.assert_array_equal(result["value"], attr_data)
            np.testing.assert_array_equal(result["y1"], y1_data)
            assert "x1" not in result
            assert "x2" not in result

            # Check conflicting labels are not allowed
            with pytest.raises(tiledb.TileDBError):
                array.label_index(["x1", "x2"])

    @pytest.mark.skipif(
        tiledb.libtiledb.version()[0] == 2 and tiledb.libtiledb.version()[1] < 15,
        reason="dimension labels requires libtiledb version 2.15 or greater",
    )
    def test_dimension_label_round_trip_sparse_array(self):
        # Create array schema with dimension labels
        dim = tiledb.Dim("index", domain=(1, 10))
        dom = tiledb.Domain(dim)
        att = tiledb.Attr("value", dtype=np.int64)
        dim_labels = {0: {"l1": dim.create_label_schema("increasing", np.int64)}}
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
            array[index_data] = {"value": attr_data, "l1": label_data}

        # Load the array schema and get the URI of the dimension label
        schema = tiledb.ArraySchema.load(uri)
        dim_label = schema.dim_label("l1")

        # Read and check the data directly from the dimension label
        with tiledb.open(dim_label.uri, "r") as label1:
            output_label_data = label1[:][dim_label.label_attr_name]
            np.testing.assert_array_equal(output_label_data, label_data)
