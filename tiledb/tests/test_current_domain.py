import tempfile
import unittest
import xml.etree.ElementTree

import numpy as np
import pytest

import tiledb


class CurrentDomainTest(unittest.TestCase):
    def test_current_domain_integer_dimensions(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(
            tiledb.Dim(name="x", domain=(0, 100), tile=10, dtype=np.int64),
            tiledb.Dim(name="y", domain=(0, 100), tile=10, dtype=np.int64),
        )
        ndrect = tiledb.NDRectangle(ctx, dom)

        # Set ranges
        range_one = (10, 20)
        range_two = (30, 40)

        ndrect.set_range(0, range_one[0], range_one[1])
        ndrect.set_range(1, range_two[0], range_two[1])

        # Get and check ranges
        self.assertEqual(ndrect.range(0), range_one)
        self.assertEqual(ndrect.range(1), range_two)

        # Create a currentDomain and set the NDRectangle
        current_domain = tiledb.CurrentDomain(ctx)
        current_domain.set_ndrectangle(ndrect)

        self.assertFalse(current_domain.is_empty)

        rect = current_domain.ndrectangle()

        # Get and check ranges
        range = rect.range(0)
        self.assertEqual(range, range_one)
        range = rect.range(1)
        self.assertEqual(range, range_two)

    def test_current_domain_string_dimensions(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(
            tiledb.Dim(name="d1", dtype="S0"),
            tiledb.Dim(name="d2", dtype="S0"),
        )
        ndrect = tiledb.NDRectangle(ctx, dom)

        # Set ranges
        range_one = ("a", "c")
        range_two = ("b", "db")

        ndrect.set_range(0, range_one[0], range_one[1])
        ndrect.set_range(1, range_two[0], range_two[1])

        # Get and check ranges
        self.assertEqual(ndrect.range(0), range_one)
        self.assertEqual(ndrect.range(1), range_two)

        # Create a currentDomain and set the NDRectangle
        current_domain = tiledb.CurrentDomain(ctx)
        current_domain.set_ndrectangle(ndrect)

        self.assertFalse(current_domain.is_empty)

        rect = current_domain.ndrectangle()

        # Get and check ranges
        range = rect.range(0)
        self.assertEqual(range, range_one)
        range = rect.range(1)
        self.assertEqual(range, range_two)

    def test_current_domain_add_to_array_schema(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(
            tiledb.Dim(name="d", domain=(1, 999), tile=2, dtype=np.int64),
        )
        att = tiledb.Attr(name="a", dtype=np.int64)
        schema = tiledb.ArraySchema(sparse=True, ctx=ctx, domain=dom, attrs=(att,))

        ndrect = tiledb.NDRectangle(ctx, dom)
        range_one = (10, 20)
        ndrect.set_range(0, range_one[0], range_one[1])

        current_domain = tiledb.CurrentDomain(ctx)
        current_domain.set_ndrectangle(ndrect)

        schema.set_current_domain(current_domain)

        cd = schema.current_domain()
        self.assertFalse(cd.is_empty)

        self.assertEqual(cd.ndrectangle().range(0), range_one)

        with self.assertRaises(tiledb.TileDBError):
            cd.ndrectangle().range(1)

    def test_current_domain_evolve(self):
        uri = tempfile.mkdtemp()

        ctx = tiledb.Ctx()
        dom = tiledb.Domain(
            tiledb.Dim(name="d", domain=(1, 999), tile=2, dtype=np.int64),
        )

        att = tiledb.Attr(name="a", dtype=np.int64)
        schema = tiledb.ArraySchema(sparse=True, ctx=ctx, domain=dom, attrs=(att,))

        ndrect = tiledb.NDRectangle(ctx, dom)
        range_one = (10, 20)
        ndrect.set_range(0, range_one[0], range_one[1])

        current_domain = tiledb.CurrentDomain(ctx)
        current_domain.set_ndrectangle(ndrect)

        schema.set_current_domain(current_domain)

        tiledb.Array.create(uri, schema)

        new_current_domain = tiledb.CurrentDomain(ctx)
        range_two = (5, 30)
        ndrect_two = tiledb.NDRectangle(ctx, dom)
        ndrect_two.set_range(0, range_two[0], range_two[1])
        new_current_domain.set_ndrectangle(ndrect_two)

        se = tiledb.ArraySchemaEvolution(ctx)
        se.expand_current_domain(new_current_domain)
        se.array_evolve(uri)

        array = tiledb.Array(uri, mode="r")
        s = array.schema
        cd = s.current_domain()
        n = cd.ndrectangle()
        self.assertEqual(n.range(0), range_two)
