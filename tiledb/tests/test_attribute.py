import sys
import xml.etree.ElementTree

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import tiledb

from .common import DiskTestCase, assert_captured, has_pandas


class AttributeTest(DiskTestCase):
    def test_minimal_attribute(self):
        attr = tiledb.Attr()
        self.assertEqual(attr, attr)
        self.assertTrue(attr.isanon)
        self.assertEqual(attr.name, "")
        self.assertEqual(attr.dtype, np.float64)
        self.assertFalse(attr.isvar)
        self.assertFalse(attr.isnullable)

        try:
            assert xml.etree.ElementTree.fromstring(attr._repr_html_()) is not None
        except:
            pytest.fail(f"Could not parse attr._repr_html_(). Saw {attr._repr_html_()}")

    def test_attribute_name_only(self, capfd):
        attr = tiledb.Attr("foo")

        attr.dump()
        assert_captured(capfd, "Name: foo")

        assert attr == attr
        assert attr.name == "foo"
        assert attr.dtype == np.float64, "default attribute type is float64"

    @pytest.mark.parametrize(
        "dtype, fill",
        [
            (np.dtype(bytes), b"abc"),
            (str, "defg"),
            (np.float32, np.float32(0.4023573667780681)),
            (np.float64, np.float64(0.0560602549760851)),
            (np.dtype("M8[ns]"), np.timedelta64(11, "ns")),
            (np.dtype([("f0", "<i4"), ("f1", "<i4"), ("f2", "<i4")]), (1, 2, 3)),
        ],
    )
    def test_attribute_fill(self, dtype, fill):
        attr = tiledb.Attr("", dtype=dtype, fill=fill)
        assert attr == attr
        assert np.array(attr.fill, dtype=dtype) == np.array(fill, dtype=dtype)

        path = self.path()
        dom = tiledb.Domain(tiledb.Dim(domain=(0, 0), tile=1, dtype=np.int64))
        schema = tiledb.ArraySchema(domain=dom, attrs=(attr,))
        tiledb.DenseArray.create(path, schema)

        with tiledb.open(path) as R:
            assert R.multi_index[0][""] == np.array(fill, dtype=dtype)
            assert R[0] == np.array(fill, dtype=dtype)
            if has_pandas() and not hasattr(dtype, "fields"):
                # record type unsupported for .df
                assert R.df[0][""].values == np.array(fill, dtype=dtype)

    def test_full_attribute(self, capfd):
        filter_list = tiledb.FilterList([tiledb.ZstdFilter(10)])
        filter_list = tiledb.FilterList([tiledb.ZstdFilter(10)])
        attr = tiledb.Attr("foo", dtype=np.int64, filters=filter_list)

        attr.dump()
        assert_captured(capfd, "Name: foo")

        self.assertEqual(attr, attr)
        self.assertEqual(attr.name, "foo")
        self.assertEqual(attr.dtype, np.int64)
        self.assertIsInstance(attr.filters[0], tiledb.ZstdFilter)
        self.assertEqual(attr.filters[0].level, 10)

    def test_ncell_attribute(self):
        dtype = np.dtype([("", np.int32), ("", np.int32), ("", np.int32)])
        attr = tiledb.Attr("foo", dtype=dtype)

        self.assertEqual(attr, attr)
        self.assertEqual(attr.dtype, dtype)
        self.assertEqual(attr.ncells, 3)

        # dtype subarrays not supported
        with self.assertRaises(TypeError):
            tiledb.Attr("foo", dtype=np.dtype((np.int32, 2)))

        # mixed type record arrays not supported
        with self.assertRaises(TypeError):
            tiledb.Attr("foo", dtype=np.dtype([("", np.float32), ("", np.int32)]))

    def test_complex64_attribute(self):
        attr = tiledb.Attr("foo", fill=(0 + 1j), dtype=np.dtype("complex64"))
        assert attr == attr
        assert attr.fill == attr.fill
        assert attr.dtype == np.complex64
        assert attr.ncells == 2

    def test_complex128_attribute(self):
        dtype = np.dtype([("", np.double), ("", np.double)])
        attr = tiledb.Attr("foo", fill=(2.0, 2.0), dtype=dtype)

        assert attr == attr
        assert attr.fill == attr.fill
        assert attr.dtype == np.complex128
        assert attr.ncells == 2

    @pytest.mark.parametrize(
        "fill", [(1.0, 1.0), np.array((1.0, 1.0), dtype=np.dtype("f4, f4"))]
    )
    def test_two_cell_float_attribute(self, fill):
        attr = tiledb.Attr("foo", fill=fill, dtype=np.dtype("f4, f4"))

        assert attr == attr
        assert attr.dtype == np.complex64
        assert attr.fill == attr.fill
        assert attr.ncells == 2

    @pytest.mark.parametrize(
        "fill", [(1.0, 1.0), np.array((1.0, 1.0), dtype=np.dtype("f8, f8"))]
    )
    def test_two_cell_double_attribute(self, fill):
        attr = tiledb.Attr("foo", fill=fill, dtype=np.dtype("f8, f8"))
        assert attr == attr
        assert attr.dtype == np.complex128
        assert attr.fill == attr.fill
        assert attr.ncells == 2

    def test_ncell_double_attribute(self):
        dtype = np.dtype([("", np.double), ("", np.double), ("", np.double)])
        fill = np.array((0, np.nan, np.inf), dtype=dtype)
        attr = tiledb.Attr("foo", dtype=dtype, fill=fill)

        self.assertEqual(attr, attr)
        self.assertEqual(attr.dtype, dtype)
        self.assertEqual(attr.ncells, 3)

    def test_ncell_not_equal_fill_attribute(self):
        dtype = np.dtype([("", np.double), ("", np.double), ("", np.double)])
        fill1 = np.array((0, np.nan, np.inf), dtype=dtype)
        fill2 = np.array((np.nan, -1, np.inf), dtype=dtype)
        attr1 = tiledb.Attr("foo", dtype=dtype, fill=fill1)
        attr2 = tiledb.Attr("foo", dtype=dtype, fill=fill2)
        assert attr1 != attr2

    def test_ncell_bytes_attribute(self):
        dtype = np.dtype((np.bytes_, 10))
        attr = tiledb.Attr("foo", dtype=dtype)
        self.assertEqual(attr, attr)
        self.assertEqual(attr.dtype, dtype)
        self.assertEqual(attr.ncells, 10)

    def test_bytes_var_attribute(self):
        with pytest.warns(DeprecationWarning, match="Attr given `var=True` but"):
            attr = tiledb.Attr("foo", var=True, dtype="S1")
            self.assertEqual(attr.dtype, np.dtype("S"))
            self.assertTrue(attr.isvar)

        with pytest.warns(DeprecationWarning, match="Attr given `var=False` but"):
            attr = tiledb.Attr("foo", var=False, dtype="S")
            self.assertEqual(attr.dtype, np.dtype("S"))
            self.assertTrue(attr.isvar)

        attr = tiledb.Attr("foo", var=True, dtype="S")
        self.assertEqual(attr, attr)
        self.assertEqual(attr.dtype, np.dtype("S"))
        self.assertTrue(attr.isvar)

        attr = tiledb.Attr("foo", var=False, dtype="S1")
        self.assertEqual(attr, attr)
        self.assertEqual(attr.dtype, np.dtype("S1"))
        self.assertFalse(attr.isvar)

        attr = tiledb.Attr("foo", dtype="S1")
        self.assertEqual(attr, attr)
        self.assertEqual(attr.dtype, np.dtype("S1"))
        self.assertFalse(attr.isvar)

        attr = tiledb.Attr("foo", dtype="S")
        self.assertEqual(attr, attr)
        self.assertEqual(attr.dtype, np.dtype("S"))
        self.assertTrue(attr.isvar)

    def test_nullable_attribute(self):
        attr = tiledb.Attr("nullable", nullable=True, dtype=np.int32)
        self.assertEqual(attr, attr)
        self.assertEqual(attr.dtype, np.dtype(np.int32))
        self.assertTrue(attr.isnullable)

    def test_blob_attribute(self):
        attr = tiledb.Attr(name="foo", dtype="blob")
        self.assertEqual(attr, attr)
        self.assertEqual(attr.dtype, np.bytes_)

    def test_blob_attribute_dump(self, capfd):
        attr = tiledb.Attr(name="foo", dtype="blob")
        attr.dump()
        assert_captured(capfd, "Type: BLOB")

    def test_datetime_attribute(self):
        attr = tiledb.Attr("foo", dtype=np.datetime64("", "D"))
        self.assertEqual(attr, attr)
        assert attr.dtype == np.dtype(np.datetime64("", "D"))
        assert attr.dtype != np.dtype(np.datetime64("", "Y"))
        assert attr.dtype != np.dtype(np.datetime64)

    @pytest.mark.parametrize("sparse", [True, False])
    def test_ascii_attribute(self, sparse, capfd):
        path = self.path("test_ascii")
        dom = tiledb.Domain(
            tiledb.Dim(name="d", domain=(1, 4), tile=1, dtype=np.uint32)
        )

        with pytest.raises(TypeError) as exc_info:
            tiledb.Attr(name="A", dtype="ascii", var=False)
        assert (
            str(exc_info.value) == "dtype is not compatible with var-length attribute"
        )

        attrs = [tiledb.Attr(name="A", dtype="ascii")]

        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=sparse)
        tiledb.Array.create(path, schema)

        ascii_data = ["a", "b", "c", "ABC"]
        unicode_data = ["±", "×", "÷", "√"]

        with tiledb.open(path, "w") as A:
            if sparse:
                with self.assertRaises(tiledb.TileDBError):
                    A[np.arange(1, 5)] = unicode_data
                A[np.arange(1, 5)] = ascii_data
            else:
                with self.assertRaises(tiledb.TileDBError):
                    A[:] = unicode_data
                A[:] = ascii_data

        with tiledb.open(path, "r") as A:
            assert A.schema.nattr == 1
            A.schema.dump()
            assert_captured(capfd, "Type: STRING_ASCII")
            assert A.schema.attr("A").isvar
            assert A.schema.attr("A").dtype == np.bytes_
            assert A.schema.attr("A").isascii
            assert_array_equal(A[:]["A"], np.asarray(ascii_data, dtype=np.bytes_))

    def test_modify_attribute_in_schema(self):
        path = self.path("test_modify_attribute_in_schema")
        tiledb.from_numpy(path, np.random.rand(10))

        with tiledb.open(path, "r") as A:
            assert A.schema.nattr == 1
            assert A.schema.attr(0).name == ""
            with pytest.raises(AttributeError) as exc:
                A.schema.attr(0).name = "can't change"

            if sys.version_info < (3, 11):
                assert "can't set attribute" in str(exc.value)
            else:
                assert "object has no setter" in str(exc.value)

    def test_wkt_attribute(self):
        A = np.array(
            ["POINT (30 10)", "POLYGON ((3 1, 4 5, 2 2, 1 2, 3 1))"],
            dtype="S",
        )

        dom = tiledb.Domain(tiledb.Dim(domain=(0, 1), tile=2))
        att = tiledb.Attr(dtype="wkt", var=True)

        schema = tiledb.ArraySchema(dom, (att,))

        tiledb.DenseArray.create(self.path("foo"), schema)
        with tiledb.DenseArray(self.path("foo"), mode="w") as T:
            T[:] = A

        # read back the data
        with tiledb.DenseArray(self.path("foo"), mode="r") as T:
            for i in range(2):
                assert_array_equal(T[i], A[i])

    def test_wkb_attribute(self):
        A = np.array(
            [
                # representation of POINT (30 10)
                b"\x01\x01\x00\x00\x00\x00\x00\x00\x00\x00\x00>@\x00\x00\x00\x00\x00\x00$@",
                # representation of POLYGON ((3 1, 4 5, 2 2, 1 2, 3 1))
                (
                    b"\x01\x03\x00\x00\x00\x01\x00\x00\x00\x05\x00\x00\x00\x00\x00\x00\x00\x00\x00\x08@"
                    b"\x00\x00\x00\x00\x00\x00\xf0?\x00\x00\x00\x00\x00\x00\x10@\x00\x00\x00\x00\x00\x00\x14@"
                    b"\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\xf0?"
                    b"\x00\x00\x00\x00\x00\x00\x00@\x00\x00\x00\x00\x00\x00\x08@\x00\x00\x00\x00\x00\x00\xf0?"
                ),
            ],
        )

        dom = tiledb.Domain(tiledb.Dim(domain=(0, 1), tile=2))
        att = tiledb.Attr(dtype="wkb", var=True)

        schema = tiledb.ArraySchema(dom, (att,))

        tiledb.DenseArray.create(self.path("foo"), schema)
        with tiledb.DenseArray(self.path("foo"), mode="w") as T:
            T[:] = A

        # read back the data
        with tiledb.DenseArray(self.path("foo"), mode="r") as T:
            for i in range(2):
                assert_array_equal(T[:][i].tobytes(), A[i])

    def test_var_numeric_attribute(self):
        uri = self.path("test_var_numeric_attribute")
        ctx = tiledb.Ctx()

        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(name="sample", domain=(None, None), dtype="ascii")
            ),
            attrs=[tiledb.Attr(name="value", dtype="uint32", var=True, nullable=False)],
            sparse=True,
            allows_duplicates=True,
        )

        data = np.array(
            [
                np.array([1, 2], dtype="uint32"),
                np.array([3], dtype="uint32"),
                np.array([4], dtype="uint32"),
            ],
            dtype="O",
        )

        tiledb.Array.create(uri, schema)

        with tiledb.open(uri, "w", ctx=ctx) as A:
            A[[b"sample1", b"sample2", b"sample3"]] = data

        with tiledb.open(uri, "r", ctx=ctx) as A:
            assert_array_equal(A[:]["value"][0], data[0])
            assert_array_equal(A[:]["value"][1], data[1])
            assert_array_equal(A[:]["value"][2], data[2])

            if has_pandas():
                import pandas as pd

                tm = pd._testing

                expected_df = pd.DataFrame(
                    {
                        "sample": [b"sample1", b"sample2", b"sample3"],
                        "value": [
                            np.array([1, 2], dtype="uint32"),
                            np.array([3], dtype="uint32"),
                            np.array([4], dtype="uint32"),
                        ],
                    }
                )

                tm.assert_frame_equal(A.df[:], expected_df)

                with pytest.raises(tiledb.TileDBError) as exc:
                    A.query(use_arrow=True).df[:]
                assert (
                    "Multi-value attributes are not currently supported when use_arrow=True. "
                    "This includes all variable-length attributes and fixed-length "
                    "attributes with more than one value. Use `query(use_arrow=False)`."
                    in str(exc.value)
                )

    def test_var_nullable_numeric_attribute(self):
        uri = self.path("test_var_nullable_numeric_attribute")
        ctx = tiledb.Ctx()

        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(name="sample", domain=(None, None), dtype="ascii")
            ),
            attrs=[tiledb.Attr(name="value", dtype="float32", var=True, nullable=True)],
            sparse=True,
            allows_duplicates=True,
        )

        data = np.array(
            [
                np.array([1.0, 2.0], dtype="float32"),
                np.array(np.nan, dtype="float32"),
                np.array([4.0], dtype="float32"),
            ],
            dtype="O",
        )

        tiledb.Array.create(uri, schema)

        with tiledb.open(uri, "w", ctx=ctx) as A:
            A[[b"sample1", b"sample2", b"sample3"]] = data

        with tiledb.open(uri, "r", ctx=ctx) as A:
            assert_array_equal(A[:]["value"][0], data[0])
            assert_array_equal(A[:]["value"][1], data[1])
            assert_array_equal(A[:]["value"][2], data[2])

            if has_pandas():
                import pandas as pd

                tm = pd._testing

                expected_df = pd.DataFrame(
                    {
                        "sample": [b"sample1", b"sample2", b"sample3"],
                        "value": [
                            np.array([1.0, 2.0], dtype="float32"),
                            np.array([np.nan], dtype="float32"),
                            np.array([4.0], dtype="float32"),
                        ],
                    }
                )

                tm.assert_frame_equal(A.df[:], expected_df)

                with pytest.raises(tiledb.TileDBError) as exc:
                    A.query(use_arrow=True).df[:]
                assert (
                    "Multi-value attributes are not currently supported when use_arrow=True. "
                    "This includes all variable-length attributes and fixed-length "
                    "attributes with more than one value. Use `query(use_arrow=False)`."
                    in str(exc.value)
                )
