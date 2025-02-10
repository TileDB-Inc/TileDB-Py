import tempfile
import unittest

import numpy as np
import pytest

import tiledb
import tiledb.libtiledb as lt

from .common import DiskTestCase

if not (lt.version()[0] == 2 and lt.version()[1] >= 25):
    pytest.skip(
        "CurrentDomain is only available in TileDB 2.26 and later",
        allow_module_level=True,
    )


class NDRectangleTest(DiskTestCase):
    def test_ndrectagle_standalone_string(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(
            tiledb.Dim(name="d1", dtype="S0"),
            tiledb.Dim(name="d2", dtype="S0"),
        )
        ndrect = tiledb.NDRectangle(ctx, dom)

        range_one = ("a", "c")
        range_two = ("b", "db")

        ndrect.set_range(0, range_one[0], range_one[1])
        ndrect.set_range(1, range_two[0], range_two[1])

        self.assertEqual(ndrect.range(0), range_one)
        self.assertEqual(ndrect.range(1), range_two)

        # should be the same
        self.assertEqual(ndrect.range("d1"), range_one)
        self.assertEqual(ndrect.range("d2"), range_two)

    def test_ndrectagle_standalone_integer(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(
            tiledb.Dim(name="x", domain=(0, 100), tile=10, dtype=np.int64),
            tiledb.Dim(name="y", domain=(0, 100), tile=10, dtype=np.int64),
        )
        ndrect = tiledb.NDRectangle(ctx, dom)

        range_one = (10, 20)
        range_two = (30, 40)

        ndrect.set_range(0, range_one[0], range_one[1])
        ndrect.set_range(1, range_two[0], range_two[1])

        self.assertEqual(ndrect.range(0), range_one)
        self.assertEqual(ndrect.range(1), range_two)

        # should be the same
        self.assertEqual(ndrect.range("x"), range_one)
        self.assertEqual(ndrect.range("y"), range_two)

    @pytest.mark.parametrize(
        "dtype_, range_",
        (
            (np.int32, (0, 1)),
            (np.int64, (2, 7)),
            (np.uint32, (1, 5)),
            (np.uint64, (5, 9)),
            (np.float32, (0.0, 1.0)),
            (np.float64, (2.0, 4.0)),
            (np.dtype(bytes), (b"abc", b"def")),
        ),
    )
    def test_set_range_different_types(self, dtype_, range_):
        domain = tiledb.Domain(
            *[
                tiledb.Dim(
                    name="rows",
                    domain=(0, 9),
                    dtype=dtype_,
                ),
            ]
        )

        ctx = tiledb.Ctx()
        ndrect = tiledb.NDRectangle(ctx, domain)
        ndrect.set_range("rows", range_[0], range_[1])


class CurrentDomainTest(DiskTestCase):
    def test_current_domain_with_ndrectangle_integer(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(
            tiledb.Dim(name="x", domain=(0, 100), tile=10, dtype=np.int64),
            tiledb.Dim(name="y", domain=(0, 100), tile=10, dtype=np.int64),
        )
        ndrect = tiledb.NDRectangle(ctx, dom)

        range_one = (10, 20)
        range_two = (30, 40)

        ndrect.set_range(0, range_one[0], range_one[1])
        ndrect.set_range(1, range_two[0], range_two[1])

        self.assertEqual(ndrect.range(0), range_one)
        self.assertEqual(ndrect.range(1), range_two)

        current_domain = tiledb.CurrentDomain(ctx)

        self.assertTrue(current_domain.is_empty)
        current_domain.set_ndrectangle(ndrect)
        self.assertFalse(current_domain.is_empty)

        # let's try to get the NDRectangle back from the current domain object
        rect = current_domain.ndrectangle

        range1 = rect.range(0)
        range2 = rect.range(1)

        self.assertEqual(range1, range_one)
        self.assertEqual(range2, range_two)

        range1 = rect.range("x")
        range2 = rect.range("y")

        # should be the same
        self.assertEqual(range1, range_one)
        self.assertEqual(range2, range_two)

    def test_current_domain_with_ndrectangle_string(self):
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(
            tiledb.Dim(name="d1", dtype="S0"),
            tiledb.Dim(name="d2", dtype="S0"),
        )
        ndrect = tiledb.NDRectangle(ctx, dom)

        range_one = ("a", "c")
        range_two = ("b", "db")

        ndrect.set_range(0, range_one[0], range_one[1])
        ndrect.set_range(1, range_two[0], range_two[1])

        self.assertEqual(ndrect.range(0), range_one)
        self.assertEqual(ndrect.range(1), range_two)

        current_domain = tiledb.CurrentDomain(ctx)

        self.assertTrue(current_domain.is_empty)
        current_domain.set_ndrectangle(ndrect)
        self.assertFalse(current_domain.is_empty)

        # let's try to get the NDRectangle back from the current domain object
        rect = current_domain.ndrectangle

        range1 = rect.range(0)
        range2 = rect.range(1)

        self.assertEqual(range1, range_one)
        self.assertEqual(range2, range_two)

        range1 = rect.range("d1")
        range2 = rect.range("d2")

        # should be the same
        self.assertEqual(range1, range_one)
        self.assertEqual(range2, range_two)

    def test_array_schema_with_current_domain_with_ndrectangle(self):
        uri = tempfile.mkdtemp()
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(
            tiledb.Dim(name="d", domain=(1, 999), tile=2, dtype=np.int64),
            tiledb.Dim(name="d2", domain=(1, 999), tile=2, dtype=np.int64),
        )
        att = tiledb.Attr(name="a", dtype=np.int64)
        schema = tiledb.ArraySchema(sparse=True, ctx=ctx, domain=dom, attrs=(att,))

        ndrect = tiledb.NDRectangle(ctx, dom)
        range_one = (10, 20)
        range_two = (30, 40)
        ndrect.set_range(0, range_one[0], range_one[1])
        ndrect.set_range(1, range_two[0], range_two[1])

        current_domain = tiledb.CurrentDomain(ctx)
        current_domain.set_ndrectangle(ndrect)
        schema.set_current_domain(current_domain)

        # create the array
        tiledb.Array.create(uri, schema)

        # open the array and check the current domain and the NDRectangle
        A = tiledb.Array(uri, mode="r")

        cd = A.schema.current_domain
        self.assertFalse(cd.is_empty)
        self.assertEqual(cd.type, lt.CurrentDomainType.NDRECTANGLE)

        ndr = cd.ndrectangle
        self.assertEqual(ndr.range(0), range_one)
        self.assertEqual(ndr.range(1), range_two)

        # a 3rd dimension should raise an error
        with self.assertRaises(tiledb.TileDBError):
            ndr.range(2)

        A.close()

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

        new_range = (5, 30)
        new_ndrect = tiledb.NDRectangle(ctx, dom)
        new_ndrect.set_range(0, new_range[0], new_range[1])
        new_current_domain = tiledb.CurrentDomain(ctx)
        new_current_domain.set_ndrectangle(new_ndrect)

        se = tiledb.ArraySchemaEvolution(ctx)
        se.expand_current_domain(new_current_domain)
        se.array_evolve(uri)

        A = tiledb.Array(uri, mode="r")
        s = A.schema
        cd = s.current_domain
        n = cd.ndrectangle
        self.assertEqual(n.range(0), new_range)
        A.close()
