import numpy as np
import pytest
from numpy.testing import assert_array_equal

import tiledb

from .common import DiskTestCase, has_pandas, has_pyarrow


class EnumerationTest(DiskTestCase):
    @pytest.mark.parametrize(
        "name,data",
        (
            ("int", np.array([0])),
            ("float", np.array([1.0, 2.2, 5.8234, 94.23])),
            ("str", np.array(["abc", "defghi", "jk"])),
            ("utf8", np.array(["abc", "defghi", "jk"], dtype=np.str_)),
            ("ascii", np.array([b"abc", b"defghi", b"jk"], dtype=np.bytes_)),
        ),
    )
    @pytest.mark.parametrize("ordered", [True, False])
    def test_enumeration_basic(self, name, ordered, data):
        enmr = tiledb.Enumeration(name, ordered, data)

        assert enmr.name == name
        assert enmr.ordered == ordered
        assert_array_equal(enmr.values(), data)
        if name in ("str", "utf8", "ascii"):
            assert enmr.cell_val_num == tiledb.cc.TILEDB_VAR_NUM()
            assert enmr.dtype.kind == data.dtype.kind
        else:
            assert enmr.cell_val_num == 1
            assert enmr.dtype.kind == data.dtype.kind

    def test_attribute_enumeration(self):
        attr = tiledb.Attr()
        attr.enum = "enum"
        assert attr.enum == "enum"

    def test_array_schema_enumeration(self):
        uri = self.path("test_array_schema_enumeration")
        dom = tiledb.Domain(tiledb.Dim(domain=(1, 8), tile=1))
        enum1 = tiledb.Enumeration("enmr1", False, np.arange(3) * 10)
        enum2 = tiledb.Enumeration("enmr2", False, ["a", "bb", "ccc"])
        attr1 = tiledb.Attr("attr1", dtype=np.int32, enum_label="enmr1")
        attr2 = tiledb.Attr("attr2", dtype=np.int32, enum_label="enmr2")
        attr3 = tiledb.Attr("attr3", dtype=np.int32)
        schema = tiledb.ArraySchema(
            domain=dom, attrs=(attr1, attr2, attr3), enums=(enum1, enum2)
        )
        tiledb.Array.create(uri, schema)

        data1 = np.random.randint(0, 3, 8)
        data2 = np.random.randint(0, 3, 8)
        data3 = np.random.randint(0, 3, 8)

        with tiledb.open(uri, "w") as A:
            A[:] = {"attr1": data1, "attr2": data2, "attr3": data3}

        with tiledb.open(uri, "r") as A:
            assert A.enum("enmr1") == enum1
            assert attr1.enum_label == "enmr1"
            assert A.attr("attr1").enum_label == "enmr1"

            assert A.enum("enmr2") == enum2
            assert attr2.enum_label == "enmr2"
            assert A.attr("attr2").enum_label == "enmr2"

            with self.assertRaises(tiledb.TileDBError) as excinfo:
                assert A.enum("enmr3") == []
            assert " No enumeration named 'enmr3'" in str(excinfo.value)
            assert attr3.enum_label is None
            assert A.attr("attr3").enum_label is None

            if has_pandas():
                assert_array_equal(A.df[:]["attr1"].cat.codes, data1)
                assert_array_equal(A.df[:]["attr2"].cat.codes, data2)

                assert_array_equal(A.df[:]["attr1"], A.multi_index[:]["attr1"])
                assert_array_equal(A.df[:]["attr2"], A.multi_index[:]["attr2"])

                assert_array_equal(A.df[:]["attr1"], A[:]["attr1"])
                assert_array_equal(A.df[:]["attr2"], A[:]["attr2"])

    @pytest.mark.skipif(
        not has_pyarrow() or not has_pandas(),
        reason="pyarrow and/or pandas not installed",
    )
    @pytest.mark.parametrize("sparse", [True, False])
    @pytest.mark.parametrize("pass_df", [True, False])
    def test_array_schema_enumeration_nullable(self, sparse, pass_df):
        import pyarrow as pa

        uri = self.path("test_array_schema_enumeration_nullable")
        enmr = tiledb.Enumeration("e", False, ["alpha", "beta", "gamma"])
        dom = tiledb.Domain(tiledb.Dim("d", domain=(1, 5), dtype="int64"))
        att = tiledb.Attr("a", dtype="int8", nullable=True, enum_label="e")
        schema = tiledb.ArraySchema(
            domain=dom, attrs=[att], enums=[enmr], sparse=sparse
        )
        tiledb.Array.create(uri, schema)

        with tiledb.open(uri, "w") as A:
            dims = pa.array([1, 2, 3, 4, 5])
            data = pa.array([1.0, 2.0, None, 0, 1.0])
            if pass_df:
                dims = dims.to_pandas()
                data = data.to_pandas()

            if sparse:
                A[dims] = data
            else:
                A[:] = data

        with tiledb.open(uri, "r") as A:
            expected_validity = [False, False, True, False, False]
            assert_array_equal(A[:]["a"].mask, expected_validity)
            assert_array_equal(A.df[:]["a"].isna(), expected_validity)
