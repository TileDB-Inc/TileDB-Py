import unittest
import xml.etree.ElementTree

import numpy as np
import pytest

import tiledb


class DimensionTest(unittest.TestCase):
    def test_minimal_dimension(self):
        dim = tiledb.Dim(domain=(0, 4), tile=5)
        self.assertEqual(dim.name, "__dim_0", "automatic dimension name is incorrect")
        self.assertEqual(dim.shape, (5,))
        self.assertEqual(dim.tile, 5)
        self.assertEqual(dim, dim)

    def test_dimension(self):
        dim = tiledb.Dim(name="d1", domain=(0, 3), tile=2)
        self.assertEqual(dim.name, "d1")
        self.assertEqual(dim.shape, (4,))
        self.assertEqual(dim.tile, 2)
        self.assertEqual(dim, dim)
        try:
            assert xml.etree.ElementTree.fromstring(dim._repr_html_()) is not None
        except:
            pytest.fail(f"Could not parse dim._repr_html_(). Saw {dim._repr_html_()}")

    def test_dimension_filter(self):
        filters = [tiledb.GzipFilter(2)]
        dim = tiledb.Dim(name="df", domain=(0, 2), tile=1, filters=filters)
        self.assertEqual(dim.filters, filters)
        self.assertEqual(dim, dim)

        filter_list = tiledb.FilterList(filters)
        dim = tiledb.Dim(name="df", domain=(0, 2), tile=1, filters=filter_list)
        self.assertEqual(dim.filters, filter_list)
        self.assertEqual(dim, dim)

        with self.assertRaises(TypeError):
            tiledb.Dim(name="df", domain=(0, 2), tile=1, filters=1)

    def test_datetime_dimension(self):
        # Regular usage
        dim = tiledb.Dim(
            name="d1",
            domain=(np.datetime64("2010-01-01"), np.datetime64("2020-01-01")),
            tile=np.timedelta64(20, "D"),
            dtype=np.datetime64("", "D"),
        )
        self.assertEqual(dim, dim)
        self.assertEqual(dim.dtype, np.dtype(np.datetime64("", "D")))
        self.assertEqual(dim.tile, np.timedelta64(20, "D"))
        self.assertNotEqual(dim.tile, np.timedelta64(21, "D"))
        self.assertNotEqual(dim.tile, np.timedelta64(20, "W"))  # Sanity check unit
        self.assertTupleEqual(
            dim.domain, (np.datetime64("2010-01-01"), np.datetime64("2020-01-01"))
        )
        self.assertEqual(dim.shape, (3653,))

        # No tile extent specified: this is not an error in 2.2
        if tiledb.libtiledb.version() < (2, 2):
            with self.assertRaises(tiledb.TileDBError):
                tiledb.Dim(
                    name="d1",
                    domain=(np.datetime64("2010-01-01"), np.datetime64("2020-01-01")),
                    dtype=np.datetime64("", "D"),
                )

        # Integer tile extent is ok
        dim = tiledb.Dim(
            name="d1",
            domain=(np.datetime64("2010-01-01"), np.datetime64("2020-01-01")),
            tile=20,
            dtype=np.datetime64("", "D"),
        )
        self.assertEqual(dim, dim)
        self.assertEqual(dim.dtype, np.dtype(np.datetime64("", "D")))
        self.assertEqual(dim.tile, np.timedelta64(20, "D"))

        # Year resolution
        dim = tiledb.Dim(
            name="d1",
            domain=(np.datetime64("2010"), np.datetime64("2020")),
            tile=5,
            dtype=np.datetime64("", "Y"),
        )
        self.assertEqual(dim, dim)
        self.assertEqual(dim.dtype, np.dtype(np.datetime64("", "Y")))
        self.assertEqual(dim.tile, np.timedelta64(5, "Y"))
        self.assertTupleEqual(
            dim.domain, (np.datetime64("2010", "Y"), np.datetime64("2020", "Y"))
        )

        # End domain promoted to day resolution
        dim = tiledb.Dim(
            name="d1",
            domain=(np.datetime64("2010-01-01"), np.datetime64("2020")),
            tile=2,
            dtype=np.datetime64("", "D"),
        )
        self.assertEqual(dim, dim)
        self.assertEqual(dim.tile, np.timedelta64(2, "D"))
        self.assertTupleEqual(
            dim.domain,
            (np.datetime64("2010-01-01", "D"), np.datetime64("2020-01-01", "D")),
        )

        # Domain values can't be integral
        with self.assertRaises(TypeError):
            dim = tiledb.Dim(
                name="d1", domain=(-10, 10), tile=2, dtype=np.datetime64("", "D")
            )

    def test_shape(self):
        dim = tiledb.Dim(name="", dtype="|S0", var=True)
        with self.assertRaisesRegex(
            TypeError,
            "shape only valid for integer and datetime dimension domains",
        ):
            dim.shape

    @pytest.mark.xfail
    def test_fail_on_0_extent(self):
        tiledb.Dim(domain=(0, 10), tile=0)
