import re

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

    def test_enumeration_repr(self):
        """Doesn't check exact string, just makes sure each component is matched, in case order is changed in the future."""
        enmr = tiledb.Enumeration("e", False, [1, 2, 3])
        # Get its string representation
        repr_str = repr(enmr)

        # Define patterns to match each component in the representation
        patterns = {
            "Enumeration": r"Enumeration",
            "name": r"name='e'",
            # use regex because it is depending on platform
            "dtype": r"dtype=int\d+",
            "dtype_name": r"dtype_name='int\d+'",
            "cell_val_num": r"cell_val_num=1",
            "ordered": r"ordered=False",
            "values": r"values=\[1, 2, 3\]",
        }

        # Check that each pattern is found in the representation string
        for key, pattern in patterns.items():
            assert re.search(pattern, repr_str), f"{key} not found or incorrect in repr"

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
        reason="pyarrow>=1.0 and/or pandas>=1.0,<3.0 not installed",
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
            assert_array_equal(A.query(attrs=["a"])[:]["a"].mask, expected_validity)

    @pytest.mark.parametrize(
        "dtype, values",
        [
            (np.int8, np.array([1, 2, 3], np.int8)),
            (np.uint8, np.array([1, 2, 3], np.uint8)),
            (np.int16, np.array([1, 2, 3], np.int16)),
            (np.uint16, np.array([1, 2, 3], np.uint16)),
            (np.int32, np.array([1, 2, 3], np.int32)),
            (np.uint32, np.array([1, 2, 3], np.uint32)),
            (np.int64, np.array([1, 2, 3], np.int64)),
            (np.uint64, np.array([1, 2, 3], np.uint64)),
            (np.dtype("S"), np.array(["a", "b", "c"], np.dtype("S"))),
            (np.dtype("U"), np.array(["a", "b", "c"], np.dtype("U"))),
        ],
    )
    def test_enum_dtypes(self, dtype, values):
        # create empty
        enmr = tiledb.Enumeration("e", False, dtype=dtype)
        if dtype in (np.dtype("S"), np.dtype("U")):
            assert enmr.dtype.kind == enmr.values().dtype.kind == dtype.kind
        else:
            assert enmr.dtype == enmr.values().dtype == dtype
            assert_array_equal(enmr.values(), [])

        # then extend with values
        enmr = enmr.extend(values)
        if dtype in (np.dtype("S"), np.dtype("U")):
            assert enmr.dtype.kind == enmr.values().dtype.kind == dtype.kind
        else:
            assert enmr.dtype == enmr.values().dtype == dtype
            assert_array_equal(enmr.values(), values)

        # create with values
        enmr = tiledb.Enumeration("e", False, values=values)
        if dtype in (np.dtype("S"), np.dtype("U")):
            assert enmr.dtype.kind == enmr.values().dtype.kind == dtype.kind
        else:
            assert enmr.dtype == enmr.values().dtype == dtype
            assert_array_equal(enmr.values(), values)

    @pytest.mark.skipif(not has_pandas(), reason="pandas>=1.0,<3.0 not installed")
    def test_from_pandas_dtype_mismatch(self):
        import pandas as pd

        schema = tiledb.ArraySchema(
            enums=[
                tiledb.Enumeration(name="enum1", values=["a", "b", "c"], ordered=False)
            ],
            domain=tiledb.Domain(
                tiledb.Dim(name="dim1", dtype=np.int32, domain=(0, 1))
            ),
            attrs=[tiledb.Attr(name="attr1", dtype=np.int32, enum_label="enum1")],
            sparse=True,
        )

        # Pandas category's categories matches the TileDB enumeration's values
        df1 = pd.DataFrame(data={"dim1": [0, 1], "attr1": ["b", "c"]})
        df1["attr1"] = pd.Categorical(values=df1.attr1, categories=["a", "b", "c"])

        array_path = self.path("arr1")
        tiledb.Array.create(array_path, schema)
        tiledb.from_pandas(array_path, df1, schema=schema, mode="append")

        actual_values = tiledb.open(array_path).df[:]["attr1"].values.tolist()
        assert actual_values == ["b", "c"]

        # Pandas category's categories does not match the TileDB enumeration's values
        df2 = pd.DataFrame(data={"dim1": [0, 1], "attr1": ["b", "c"]})
        df2["attr1"] = df2["attr1"].astype("category")

        array_path = self.path("arr2")
        tiledb.Array.create(array_path, schema)
        tiledb.from_pandas(array_path, df2, schema=schema, mode="append")

        actual_values = tiledb.open(array_path).df[:]["attr1"].values.tolist()
        assert actual_values == ["b", "c"]
