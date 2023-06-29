import numpy as np
import pytest
from numpy.testing import assert_array_equal

import tiledb

from .common import DiskTestCase


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
        dom = tiledb.Domain(tiledb.Dim(domain=(1, 8), tile=2))
        enum1 = tiledb.Enumeration("enmr1", False, np.arange(8))
        enum2 = tiledb.Enumeration("enmr2", False, range(8))
        attr1 = tiledb.Attr("attr1", dtype=np.int32, enum_label="enmr1")
        attr2 = tiledb.Attr("attr2", dtype=np.int32, enum_label="enmr2")
        schema = tiledb.ArraySchema(domain=dom, attrs=(attr1, attr2), enums=(enum1, enum2))
        tiledb.Array.create(uri, schema)

        with tiledb.open(uri, "r") as A:
            assert A.enum("enmr1") == enum1
            assert attr1.enum_label == "enmr1"
            assert A.attr("attr1").enum_label == "enmr1"
            assert A.enum("enmr2") == enum2
            assert attr2.enum_label == "enmr2"
            assert A.attr("attr2").enum_label == "enmr2"
    
