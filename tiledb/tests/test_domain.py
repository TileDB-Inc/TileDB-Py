import xml.etree.ElementTree

import numpy as np
import pytest

import tiledb

from .common import DiskTestCase, assert_captured


class DomainTest(DiskTestCase):
    def test_domain(self, capfd):
        dims = [
            tiledb.Dim("d1", (1, 4), 2, dtype="u8"),
            tiledb.Dim("d2", (1, 4), 2, dtype="u8"),
        ]
        dom = tiledb.Domain(*dims)

        # check that dumping works
        dom.dump()
        assert_captured(capfd, "Name: d1")

        self.assertEqual(dom.ndim, 2)
        self.assertEqual(dom.dtype, np.dtype("uint64"))
        self.assertEqual(dom.shape, (4, 4))

        # check that we can iterate over the dimensions
        dim_names = [dim.name for dim in dom]
        self.assertEqual(["d1", "d2"], dim_names)

        # check that we can access dim by name
        dim_d1 = dom.dim("d1")
        self.assertEqual(dim_d1, dom.dim(0))

        # check that we can construct directly from a List[Dim]
        dom2 = tiledb.Domain(dims)
        self.assertEqual(dom, dom2)

        try:
            assert xml.etree.ElementTree.fromstring(dom._repr_html_()) is not None
        except:
            pytest.fail(f"Could not parse dom._repr_html_(). Saw {dom._repr_html_()}")

    def test_datetime_domain(self):
        dim = tiledb.Dim(
            name="d1",
            domain=(np.datetime64("2010-01-01"), np.datetime64("2020-01-01")),
            tile=np.timedelta64(20, "D"),
            dtype=np.datetime64("", "D"),
        )
        dom = tiledb.Domain(dim)
        self.assertEqual(dom, dom)
        self.assertEqual(dom.dtype, np.datetime64("", "D"))

    def test_domain_mixed_names_error(self):
        with self.assertRaises(tiledb.TileDBError):
            tiledb.Domain(
                tiledb.Dim("d1", (1, 4), 2, dtype="u8"),
                tiledb.Dim("__dim_0", (1, 4), 2, dtype="u8"),
            )

    def test_ascii_domain(self, capfd):
        path = self.path("test_ascii_domain")

        dim = tiledb.Dim(name="d", dtype="ascii")
        assert dim.dtype == np.bytes_

        dom = tiledb.Domain(dim)
        self.assertEqual(dom, dom)
        dom.dump()
        assert_captured(capfd, "Type: STRING_ASCII")

        att = tiledb.Attr(name="a", dtype=np.int64)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True)
        tiledb.SparseArray.create(path, schema)

        ascii_coords = ["a", "b", "c", "ABC"]
        unicode_coords = ["±", "×", "÷", "√"]
        data = [1, 2, 3, 4]

        with tiledb.open(path, "w") as A:
            with self.assertRaises(tiledb.TileDBError):
                A[unicode_coords] = data
            A[ascii_coords] = data
