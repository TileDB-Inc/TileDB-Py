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
        enum = tiledb.Enumeration("enmr", False, np.random.rand(5))
        attr = tiledb.Attr("val", dtype="f8")
        schema = tiledb.ArraySchema(domain=dom, attrs=(attr,), enums=(enum,))
        attr.enum = "enmr"
        tiledb.Array.create(uri, schema)

        with tiledb.open(uri, "r") as A:
            assert A.enum("enmr") == enum
            assert attr.enum == "enmr"
            # assert A.attr("val").enum == "enmr"
    
