import xml.etree.ElementTree

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import tiledb
from tiledb.tests.common import DiskTestCase, assert_captured


class ArraySchemaTest(DiskTestCase):
    def test_schema_basic(self):
        dom = tiledb.Domain(
            tiledb.Dim("d1", (1, 4), 2, dtype="u8"),
            tiledb.Dim("d2", (1, 4), 2, dtype="u8"),
        )

        attr1 = tiledb.Attr("foo", dtype=float)
        attr2 = tiledb.Attr("foo", dtype=int)

        # test unique attributes
        with self.assertRaises(tiledb.TileDBError):
            tiledb.ArraySchema(domain=dom, attrs=(attr1, attr2))

        # test schema.check
        schema = tiledb.ArraySchema(domain=dom, attrs=(attr1,))
        # valid schema does not raise
        schema.check()

        try:
            assert xml.etree.ElementTree.fromstring(schema._repr_html_()) is not None
        except:
            pytest.fail(
                f"Could not parse schema._repr_html_(). Saw {schema._repr_html_()}"
            )

    def test_dense_array_schema(self):
        domain = tiledb.Domain(
            tiledb.Dim(domain=(1, 8), tile=2), tiledb.Dim(domain=(1, 8), tile=2)
        )
        a1 = tiledb.Attr("val", dtype="f8")
        schema = tiledb.ArraySchema(domain=domain, attrs=(a1,))
        assert schema.sparse is False
        assert schema.cell_order == "row-major"
        assert schema.tile_order == "row-major"
        assert schema.domain == domain
        assert schema.ndim == 2
        assert schema.shape == (8, 8)
        assert schema.nattr == 1
        assert schema.domain.homogeneous is True
        assert hasattr(schema, "version")  # don't pin to a specific version
        assert schema.attr(0) == a1
        assert schema.has_attr("val") is True
        assert schema.has_attr("nononoattr") is False
        assert schema == tiledb.ArraySchema(domain=domain, attrs=(a1,))
        assert schema != tiledb.ArraySchema(domain=domain, attrs=(a1,), sparse=True)

        with self.assertRaises(tiledb.TileDBError):
            schema.allows_duplicates
        # test iteration over attributes
        assert list(schema) == [a1]

        with self.assertRaisesRegex(
            tiledb.TileDBError,
            "Cannot set cell order; Hilbert order is only applicable to sparse arrays",
        ):
            tiledb.ArraySchema(
                domain=domain, attrs=(a1,), sparse=False, cell_order="hilbert"
            )

    def test_dense_array_schema_fp_domain_error(self):
        dom = tiledb.Domain(tiledb.Dim(domain=(1, 8), tile=2, dtype=np.float64))
        att = tiledb.Attr("val", dtype=np.float64)

        with self.assertRaises(tiledb.TileDBError):
            tiledb.ArraySchema(domain=dom, attrs=(att,))

    def test_sparse_schema(self):
        # create dimensions
        d1 = tiledb.Dim("d1", domain=(1, 1000), tile=10, dtype="uint64")
        d2 = tiledb.Dim("d2", domain=(101, 10000), tile=100, dtype="uint64")

        # create domain
        domain = tiledb.Domain(d1, d2)

        # create attributes
        a1 = tiledb.Attr("a1", dtype="int32,int32,int32")
        a2 = tiledb.Attr(
            "a2", filters=tiledb.FilterList([tiledb.GzipFilter(-1)]), dtype="int32"
        )

        # create sparse array with schema
        coords_filters = tiledb.FilterList([tiledb.ZstdFilter(4)])
        offsets_filters = tiledb.FilterList([tiledb.LZ4Filter(5)])
        validity_filters = tiledb.FilterList([tiledb.GzipFilter(9)])

        with pytest.warns(
            DeprecationWarning,
            match="coords_filters is deprecated; set the FilterList for each dimension",
        ):
            schema = tiledb.ArraySchema(
                domain=domain,
                attrs=(a1, a2),
                capacity=10,
                cell_order="col-major",
                tile_order="row-major",
                allows_duplicates=True,
                sparse=True,
                coords_filters=coords_filters,
                offsets_filters=offsets_filters,
                validity_filters=validity_filters,
            )

        # schema.dump()
        # assert_captured(capfd, "Array type: sparse")

        assert schema.sparse is True
        assert schema.capacity == 10
        assert schema.cell_order, "co == major"
        assert schema.tile_order, "ro == major"

        # <todo>
        # assert schema.coords_compressor, ('zstd' == 4)
        # assert schema.offsets_compressor, ('lz4' == 5)
        assert len(schema.coords_filters) == 0
        assert len(schema.offsets_filters) == 1
        assert len(schema.validity_filters) == 1

        assert schema.domain == domain
        assert schema.ndim == 2
        assert schema.shape, 1000 == 9900
        assert schema.nattr == 2
        assert schema.attr(0) == a1
        assert schema.attr("a2") == a2
        assert schema.allows_duplicates is True

        assert schema.domain.dim("d1").filters == coords_filters
        assert schema.domain.dim("d2").filters == coords_filters

        with pytest.warns(
            DeprecationWarning,
            match="coords_filters is deprecated; set the FilterList for each dimension",
        ):
            schema2 = tiledb.ArraySchema(
                domain=domain,
                attrs=(a1, a2),
                capacity=10,
                cell_order="col-major",
                tile_order="row-major",
                allows_duplicates=True,
                sparse=True,
                coords_filters=coords_filters,
                offsets_filters=offsets_filters,
                validity_filters=validity_filters,
            )
        assert schema == schema2

        # test iteration over attributes
        assert list(schema) == [a1, a2]

        with self.assertRaisesRegex(
            tiledb.TileDBError,
            "Cannot set tile order; Hilbert order is not applicable to tiles",
        ):
            tiledb.ArraySchema(
                domain=domain, attrs=(a1,), sparse=True, tile_order="hilbert"
            )

    def test_sparse_schema_filter_list(self, capfd):
        # create dimensions
        d1 = tiledb.Dim("d1", domain=(1, 1000), tile=10, dtype="uint64")
        d2 = tiledb.Dim("d2", domain=(101, 10000), tile=100, dtype="uint64")

        # create domain
        domain = tiledb.Domain(d1, d2)

        # create attributes
        a1 = tiledb.Attr("a1", dtype="int32,int32,int32")
        filter_list = tiledb.FilterList([tiledb.GzipFilter()])
        a2 = tiledb.Attr("a2", filters=filter_list, dtype="float32")

        off_filters_pylist = [tiledb.ZstdFilter(level=10)]
        off_filters = tiledb.FilterList(filters=off_filters_pylist, chunksize=2048)

        coords_filters_pylist = [tiledb.Bzip2Filter(level=5)]
        coords_filters = tiledb.FilterList(
            filters=coords_filters_pylist, chunksize=4096
        )

        validity_filters_pylist = [tiledb.GzipFilter(level=9)]
        validity_filters = tiledb.FilterList(
            filters=validity_filters_pylist, chunksize=1024
        )

        # create sparse array with schema
        with pytest.warns(
            DeprecationWarning,
            match="coords_filters is deprecated; set the FilterList for each dimension",
        ):
            schema = tiledb.ArraySchema(
                domain=domain,
                attrs=(a1, a2),
                capacity=10,
                cell_order="col-major",
                tile_order="row-major",
                coords_filters=coords_filters,
                offsets_filters=off_filters,
                validity_filters=validity_filters,
                sparse=True,
            )
        self.assertTrue(schema.sparse)

        assert len(schema.coords_filters) == 0

        assert len(schema.domain.dim("d1").filters) == 1
        assert schema.domain.dim("d1").filters[0] == tiledb.Bzip2Filter(level=5)
        assert schema.domain.dim("d2").filters[0] == tiledb.Bzip2Filter(level=5)

        assert len(schema.offsets_filters) == 1
        assert schema.offsets_filters[0] == tiledb.ZstdFilter(level=10)

        assert len(schema.validity_filters) == 1
        assert schema.validity_filters[0] == tiledb.GzipFilter(level=9)

        schema.dump()
        assert_captured(capfd, "Array type: sparse")

        # make sure we can construct ArraySchema with python lists of filters
        with pytest.warns(
            DeprecationWarning,
            match="coords_filters is deprecated; set the FilterList for each dimension",
        ):
            schema2 = tiledb.ArraySchema(
                domain=domain,
                attrs=(a1, a2),
                capacity=10,
                cell_order="col-major",
                tile_order="row-major",
                coords_filters=coords_filters_pylist,
                offsets_filters=off_filters,
                validity_filters=validity_filters,
                sparse=True,
            )
        assert len(schema2.coords_filters) == 0

        assert schema.domain.dim("d1").filters == coords_filters_pylist
        assert schema.domain.dim("d2").filters == coords_filters_pylist

        assert len(schema2.domain.dim("d1").filters) == 1
        assert schema2.domain.dim("d1").filters[0] == tiledb.Bzip2Filter(level=5)
        assert schema2.domain.dim("d2").filters[0] == tiledb.Bzip2Filter(level=5)

        assert len(schema2.offsets_filters) == 1
        assert schema2.offsets_filters[0] == tiledb.ZstdFilter(level=10)

        assert len(schema2.validity_filters) == 1
        assert schema2.validity_filters[0] == tiledb.GzipFilter(level=9)

    def test_none_filter_list(self):
        with self.assertRaises(ValueError):
            tiledb.FilterList([None])

        with self.assertRaises(ValueError):
            fl = tiledb.FilterList()
            fl.append(None)

    def test_mixed_string_schema(self):
        path = self.path("test_mixed_string_schema")

        dims = [
            tiledb.Dim(name="dpos", domain=(-100.0, 100.0), tile=10, dtype=np.float64),
            tiledb.Dim(name="str_index", tile=None, dtype=np.bytes_),
        ]
        dom = tiledb.Domain(*dims)
        attrs = [tiledb.Attr(name="val", dtype=np.float64)]

        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=True)

        self.assertTrue(schema.domain.has_dim("str_index"))
        self.assertFalse(schema.domain.has_dim("nonono_str_index"))
        self.assertTrue(schema.domain.dim("str_index").isvar)
        self.assertFalse(schema.domain.dim("dpos").isvar)
        self.assertEqual(schema.domain.dim("dpos").dtype, np.double)
        self.assertEqual(schema.domain.dim("str_index").dtype, np.bytes_)
        self.assertFalse(schema.domain.homogeneous)

        tiledb.Array.create(path, schema)
        with tiledb.open(path, "r") as arr:
            assert_array_equal(arr[:]["str_index"], np.array([], dtype="|S1"))
