import itertools
import tempfile

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import tiledb
import tiledb.libtiledb as lt

from .common import DiskTestCase

pd = pytest.importorskip("pandas")

if lt.version() < (2, 26):
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

    @pytest.mark.skipif(
        tiledb.libtiledb.version() < (2, 27),
        reason="Support for current domain on dense arrays was added in 2.27",
    )
    def test_take_current_domain_into_account_dense_indexing_sc61914(self):
        uri = self.path("test_sc61914")
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(
            tiledb.Dim(name="d1", domain=(0, 99), tile=20, dtype=np.int64),
            tiledb.Dim(name="d2", domain=(0, 99), tile=20, dtype=np.int64),
        )
        att = tiledb.Attr(name="a", dtype=np.int64)
        schema = tiledb.ArraySchema(sparse=False, ctx=ctx, domain=dom, attrs=(att,))

        tiledb.Array.create(uri, schema)

        data = np.arange(0, 10000).reshape(100, 100)

        with tiledb.open(uri, "w") as A:
            A[:] = data

        with tiledb.DenseArray(uri, mode="r") as A:
            ndrect = tiledb.NDRectangle(ctx, dom)
            range_one = (10, 20)
            range_two = (30, 35)
            ndrect.set_range(0, range_one[0], range_one[1])
            ndrect.set_range(1, range_two[0], range_two[1])

            current_domain = tiledb.CurrentDomain(ctx)
            current_domain.set_ndrectangle(ndrect)
            A.schema.set_current_domain(current_domain)

            # Define the expected results
            d1_values = range(10, 21)
            d2_values = range(30, 36)
            data = [
                (d1, d2, d1 * 100 + d2)
                for d1, d2 in itertools.product(d1_values, d2_values)
            ]
            expected_df = pd.DataFrame(data, columns=["d1", "d2", "a"])

            expected_array = np.array(
                [
                    [1030, 1031, 1032, 1033, 1034, 1035],
                    [1130, 1131, 1132, 1133, 1134, 1135],
                    [1230, 1231, 1232, 1233, 1234, 1235],
                    [1330, 1331, 1332, 1333, 1334, 1335],
                    [1430, 1431, 1432, 1433, 1434, 1435],
                    [1530, 1531, 1532, 1533, 1534, 1535],
                    [1630, 1631, 1632, 1633, 1634, 1635],
                    [1730, 1731, 1732, 1733, 1734, 1735],
                    [1830, 1831, 1832, 1833, 1834, 1835],
                    [1930, 1931, 1932, 1933, 1934, 1935],
                    [2030, 2031, 2032, 2033, 2034, 2035],
                ]
            )

            assert_array_equal(A, expected_array)
            assert_array_equal(A.df[:, :], expected_df)

            # check indexing the array inside the range of the current domain
            assert_array_equal(A[11:14, 33:35]["a"], expected_array[1:4, 3:5])
            filtered_df = expected_df.query(
                "d1 >= 11 and d1 <= 14 and d2 >= 33 and d2 <= 35"
            ).reset_index(drop=True)
            assert_array_equal(A.df[11:14, 33:35], filtered_df)

            # check only one side of the range
            assert_array_equal(A[11:, :35]["a"], expected_array[1:, :5])
            filtered_df = expected_df.query("d1 >= 11 and d2 <= 35").reset_index(
                drop=True
            )
            assert_array_equal(A.df[11:, :35], filtered_df)

            # check indexing the array outside the range of the current domain - should raise an error
            with self.assertRaises(tiledb.TileDBError):
                A[11:55, 33:34]

            with self.assertRaises(tiledb.TileDBError):
                A.df[11:55, 33:34]

    def test_take_current_domain_into_account_sparse_indexing_sc61914(self):
        uri = self.path("test_sc61914")
        ctx = tiledb.Ctx()
        dom = tiledb.Domain(
            tiledb.Dim(name="d1", domain=(0, 99), tile=20, dtype=np.int64),
            tiledb.Dim(name="d2", domain=(0, 99), tile=20, dtype=np.int64),
        )
        att = tiledb.Attr(name="a", dtype=np.int64)
        schema = tiledb.ArraySchema(sparse=True, ctx=ctx, domain=dom, attrs=(att,))

        tiledb.Array.create(uri, schema)

        data = np.arange(0, 2500).reshape(50, 50)

        d1 = np.linspace(10, 59, num=50, dtype=np.int64)
        d2 = np.linspace(10, 69, num=50, dtype=np.int64)
        coords_d1, coords_d2 = np.meshgrid(d1, d2, indexing="ij")

        with tiledb.open(uri, "w") as A:
            A[coords_d1.flatten(), coords_d2.flatten()] = data

        with tiledb.open(uri, mode="r") as A:
            ndrect = tiledb.NDRectangle(ctx, dom)
            range_one = (21, 23)
            range_two = (35, 38)
            ndrect.set_range(0, range_one[0], range_one[1])
            ndrect.set_range(1, range_two[0], range_two[1])

            current_domain = tiledb.CurrentDomain(ctx)
            current_domain.set_ndrectangle(ndrect)
            A.schema.set_current_domain(current_domain)

            expected_array = {
                "d1": np.array([21, 21, 21, 21, 22, 22, 22, 22, 23, 23, 23, 23]),
                "d2": np.array([35, 36, 37, 38, 35, 36, 37, 38, 35, 36, 37, 38]),
                "a": np.array(
                    [571, 572, 573, 574, 621, 622, 623, 624, 671, 672, 673, 674]
                ),
            }

            assert_array_equal(A[:]["d1"], expected_array["d1"])
            assert_array_equal(A[:]["d2"], expected_array["d2"])
            assert_array_equal(A[:]["a"], expected_array["a"])

            expected_df = pd.DataFrame(expected_array)
            assert_array_equal(A.df[:], expected_df)
