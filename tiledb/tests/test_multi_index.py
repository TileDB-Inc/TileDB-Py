"""
TODO
- # implement mock of expected behavior in pure numpy w/ test function


- implement read function and tests (single [x], multi-attribute [ ])
- implement custom indexer
- implement oindex...
"""

import random

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import tiledb
from tiledb.multirange_indexing import getitem_ranges, mr_dense_result_shape

from .common import (
    SUPPORTED_DATETIME64_DTYPES,
    DiskTestCase,
    assert_dict_arrays_equal,
    assert_tail_equal,
    has_pandas,
    has_pyarrow,
    intspace,
    rand_datetime64_array,
)


def make_1d_dense(path, attr_name="", attr_dtype=np.int64, dim_dtype=np.uint64):
    a_orig = np.arange(36)

    dom = tiledb.Domain(tiledb.Dim(domain=(0, 35), tile=35, dtype=dim_dtype))
    att = tiledb.Attr(name=attr_name, dtype=attr_dtype)
    schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=False)
    tiledb.DenseArray.create(path, schema)

    with tiledb.DenseArray(path, "w") as A:
        A[:] = a_orig


def make_2d_dense(path, attr_name="", attr_dtype=np.int64):
    a_orig = np.arange(1, 37).reshape(9, 4)

    dom = tiledb.Domain(
        tiledb.Dim(domain=(0, 8), tile=9, dtype=np.uint64),
        tiledb.Dim(domain=(0, 3), tile=4, dtype=np.uint64),
    )
    att = tiledb.Attr(name=attr_name, dtype=attr_dtype)
    schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=False)
    tiledb.DenseArray.create(path, schema)

    with tiledb.DenseArray(path, "w") as A:
        A[:] = a_orig


class TestMultiRangeAuxiliary(DiskTestCase):
    def test_shape_funcs(self):
        range1el = (((1, 1),),)
        self.assertEqual(mr_dense_result_shape(range1el), (1,))

        range1d = tuple([((1, 2), (4, 4))])
        self.assertEqual(mr_dense_result_shape(range1d), (3,))

        range2d1 = (((3, 6), (7, 7), (10, 12)), ((5, 7),))
        self.assertEqual(mr_dense_result_shape(range2d1), (8, 3))

        # range2d2 = ([(3, 6), (7, 7), (10, 12)], [(5, 7), (10, 10)])

    # def test_3d(self):
    #     range3d1 = (((2, 4),), ((3, 6),), ((1, 4), (5, 9)))
    #
    #     # self.assertEqual()

    def test_sel_to_ranges(self):
        class Obj(object):
            pass

        class IBI(object):
            def __getitem__(self, idx):
                return idx

        def make_arr(ndim):
            arr = Obj()
            arr.schema = Obj()
            arr.schema.domain = Obj()
            arr.schema.domain.ndim = ndim
            arr.schema.sparse = False
            arr.array = Obj()
            # place-holder for attribute that is not used in these tests
            arr.nonempty_domain = lambda: [()] * ndim
            return arr

        ibi = IBI()
        # ndim = 1
        arr = make_arr(1)
        self.assertEqual(getitem_ranges(arr, ibi[[1]]), (((1, 1),),))
        self.assertEqual(getitem_ranges(arr, ibi[[1, 2]]), (((1, 1), (2, 2)),))
        self.assertEqual(getitem_ranges(arr, ibi[slice(1, 2)]), (((1, 2),),))
        self.assertEqual(getitem_ranges(arr, ibi[1:2]), (((1, 2),),))

        # ndim = 2
        arr2 = make_arr(2)
        self.assertEqual(getitem_ranges(arr2, ibi[[1]]), (((1, 1),), ()))
        self.assertEqual(getitem_ranges(arr2, ibi[slice(1, 33)]), (((1, 33),), ()))
        self.assertEqual(
            getitem_ranges(arr2, ibi[[1, 2], [[1], slice(1, 3)]]),
            (((1, 1), (2, 2)), ((1, 1), (1, 3))),
        )

        # ndim = 3
        arr3 = make_arr(3)
        self.assertEqual(
            getitem_ranges(arr3, ibi[1, 2, 3]), (((1, 1),), ((2, 2),), ((3, 3),))
        )
        self.assertEqual(getitem_ranges(arr3, ibi[1, 2]), ((((1, 1),), ((2, 2),), ())))
        self.assertEqual(
            getitem_ranges(arr3, ibi[1:2, 3:4]), (((1, 2),), ((3, 4),), ())
        )
        self.assertEqual(
            getitem_ranges(arr3, ibi[1:2, 3:4, 5:6]), (((1, 2),), ((3, 4),), ((5, 6),))
        )
        self.assertEqual(
            getitem_ranges(arr3, ibi[[1], [2], [5, 6]]),
            (((1, 1),), ((2, 2),), ((5, 5), (6, 6))),
        )
        self.assertEqual(
            getitem_ranges(arr3, ibi[1, [slice(3, 6), 8], slice(4, 6)]),
            (((1, 1),), ((3, 6), (8, 8)), ((4, 6),)),
        )
        self.assertEqual(getitem_ranges(arr3, ibi[(1, 2)]), (((1, 1),), ((2, 2),), ()))
        self.assertEqual(getitem_ranges(arr3, ibi[[(1, 2)]]), (((1, 2),), (), ()))
        self.assertEqual(
            getitem_ranges(arr3, ibi[[(1, 2), 4], [slice(1, 4)]]),
            (((1, 2), (4, 4)), ((1, 4),), ()),
        )


class TestMultiRange(DiskTestCase):
    @pytest.mark.skipif(
        not has_pyarrow() or not has_pandas(),
        reason="pyarrow>=1.0 and/or pandas>=1.0,<3.0 not installed",
    )
    def test_return_arrow_indexers(self):
        uri = self.path("multirange_behavior_sparse")

        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(
                tiledb.Dim(name="idx", domain=(-5, 5), dtype=np.int64)
            ),
            attrs=[tiledb.Attr(name="data", dtype=np.int64)],
        )
        tiledb.Array.create(uri, schema)
        data = np.random.randint(-10, 10, size=11)

        with tiledb.open(uri, "w") as A:
            A[:] = data

        with tiledb.open(uri, "r") as A:
            with self.assertRaisesRegex(
                tiledb.TileDBError,
                "Cannot initialize return_arrow with use_arrow=False",
            ):
                q = A.query(return_arrow=True, use_arrow=False)

            q = A.query(return_arrow=True)

            with self.assertRaisesRegex(
                tiledb.TileDBError,
                "`return_arrow=True` requires .df indexer",
            ):
                q[:]

            with self.assertRaisesRegex(
                tiledb.TileDBError,
                "`return_arrow=True` requires .df indexer",
            ):
                q.multi_index[:]

            assert_array_equal(q.df[:]["data"], data)

    @pytest.mark.skipif(
        not has_pyarrow() or not has_pandas(),
        reason="pyarrow>=1.0 and/or pandas>=1.0,<3.0 not installed",
    )
    @pytest.mark.parametrize("sparse", [True, False])
    def test_return_large_arrow_table(self, sparse):
        num = 2**16 - 1
        uri = self.path("test_return_large_arrow_table")
        dom = tiledb.Domain(tiledb.Dim(domain=(0, num - 1), dtype=np.uint16))
        att = tiledb.Attr(dtype=np.uint64)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=sparse)
        tiledb.Array.create(uri, schema)

        expected_data = np.arange(num)

        with tiledb.open(uri, "w") as A:
            if sparse:
                A[np.arange(num)] = expected_data
            else:
                A[:] = expected_data

        with tiledb.open(uri, "r") as arr:
            actual_data = arr.query(return_arrow=True).df[:]
            assert_array_equal(actual_data[:][""], expected_data)

    def test_multirange_behavior(self):
        uri = self.path("multirange_behavior_sparse")

        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(
                *[
                    tiledb.Dim(
                        name="idx",
                        domain=(-1.0, 0.7999999999999996),
                        tile=2.0,
                        dtype="float64",
                    )
                ]
            ),
            attrs=[tiledb.Attr(name="data", dtype="float64", var=False)],
            cell_order="row-major",
            tile_order="row-major",
            capacity=10000,
            sparse=True,
            allows_duplicates=True,
        )
        tiledb.SparseArray.create(uri, schema)
        data = np.random.rand(10)
        idx = np.arange(-1, 1, 0.2)

        with tiledb.open(uri, "w") as A:
            A[idx] = {"data": data}

        with tiledb.open(uri) as A:
            res = A.multi_index[:]
            # always return data
            self.assertTrue("data" in res)
            # return coordinates for sparse
            self.assertTrue("idx" in res)
            assert_array_equal(res["data"], data)
            assert_array_equal(res["idx"], idx)

        uri = self.path("multirange_behavior_dense")
        with tiledb.from_numpy(uri, data):
            pass

        with tiledb.open(uri) as B:
            res = B.multi_index[0:9]  # TODO: this should accept [:]
            # always return data
            self.assertTrue("" in res)
            # don't return coordinates for dense
            self.assertTrue("idx" not in res)

    def test_multirange_empty(self):
        path1 = self.path("test_multirange_empty_1d")
        make_1d_dense(path1, attr_dtype=np.uint16)
        with tiledb.open(path1) as A:
            res = A.multi_index[tiledb.EmptyRange]
            assert res[""].dtype == np.uint16
            assert res[""].shape == (0,)

        path2 = self.path("test_multirange_empty_2d")
        make_2d_dense(path2, attr_dtype=np.float32)
        with tiledb.open(path2) as A:
            res = A.multi_index[tiledb.EmptyRange]
            assert res[""].dtype == np.float32
            assert res[""].shape == (0,)

    def test_multirange_1d_1dim_ranges(self):
        path = self.path("test_multirange_1d_1dim_ranges")
        attr_name = "a"

        make_1d_dense(path, attr_name=attr_name)

        with tiledb.DenseArray(path) as A:
            ranges = (((0, 0),),)
            expected = np.array([0], dtype=np.int64)
            res = tiledb.libtiledb.multi_index(A, (attr_name,), ranges)
            a = res[attr_name]
            assert_array_equal(a, expected)
            self.assertEqual(a.dtype, expected.dtype)
            self.assertEqual(len(res.keys()), 2)

            ranges2 = (((1, 1), (5, 8)),)
            expected2 = np.array([1, 5, 6, 7, 8], dtype=np.int64)
            a2 = tiledb.libtiledb.multi_index(A, (attr_name,), ranges2)[attr_name]
            assert_array_equal(a2, expected2)
            self.assertEqual(a2.dtype, expected2.dtype)

    def test_multirange_2d_1dim_ranges(self):
        path = self.path("test_multirange_1dim_ranges")
        attr_name = "a"

        make_2d_dense(path, attr_name=attr_name)

        expected = np.array(
            [
                1,
                2,
                3,
                4,
                21,
                22,
                23,
                24,
                25,
                26,
                27,
                28,
                29,
                30,
                31,
                32,
                33,
                34,
                35,
                36,
            ],
            dtype=np.uint64,
        )

        ranges = (((0, 0), (5, 8)),)

        with tiledb.DenseArray(path) as A:
            a = tiledb.libtiledb.multi_index(A, (attr_name,), ranges)[attr_name]

            assert_array_equal(a, expected)

    def test_multirange_2d_2dim_ranges(self):
        path = self.path("test_multirange_2dim_ranges")
        attr_name = "a"

        make_2d_dense(path, attr_name=attr_name)

        expected = np.arange(1, 21)

        ranges = (((0, 4),), ((0, 3),))

        with tiledb.DenseArray(path) as A:
            a = tiledb.libtiledb.multi_index(A, (attr_name,), ranges)[attr_name]
            assert_array_equal(a, expected)

            # test slicing start=end on 1st dim at 0 (bug fix)
            assert_tail_equal(
                np.array([[1, 2, 3, 4]]),
                A.multi_index[:0][attr_name],
                A.multi_index[0:0][attr_name],
            )

            # test slicing start=end on 2nd dim at 0 (bug fix)
            assert_tail_equal(
                np.arange(1, 34, 4).reshape((9, 1)),
                A.multi_index[:, :0][attr_name],
                A.multi_index[:, 0:0][attr_name],
            )

            # test slicing start=end on 1st dim at 1
            assert_array_equal(np.array([[5, 6, 7, 8]]), A.multi_index[1:1][attr_name])

            # test slicing start=end on 2nd dim at 1
            assert_array_equal(
                np.arange(2, 35, 4).reshape((9, 1)), A.multi_index[:, 1:1][attr_name]
            )

            # test slicing start=end on 1st dim at max range
            assert_array_equal(
                np.array([[33, 34, 35, 36]]), A.multi_index[8:8][attr_name]
            )

            # test slicing start=end on 2nd dim at max range
            assert_tail_equal(
                np.arange(4, 37, 4).reshape((9, 1)), A.multi_index[:, 3:3][attr_name]
            )

    def test_multirange_1d_dense_int64(self):
        attr_name = ""
        path = self.path("multi_index_1d")

        dom = tiledb.Domain(
            tiledb.Dim(name="coords", domain=(-10, 10), tile=9, dtype=np.int64)
        )
        att = tiledb.Attr(name=attr_name, dtype=np.float32)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,))
        tiledb.DenseArray.create(path, schema)

        orig_array = np.random.rand(schema.domain.dim(0).size).astype(np.float32)
        with tiledb.open(path, "w") as A:
            A[:] = orig_array

        with tiledb.open(path) as A:
            # stepped ranges are not supported
            with self.assertRaises(ValueError):
                A.query(coords=True).multi_index[1::2]

            assert_array_equal(orig_array[[0, -1]], A.multi_index[[-10, 10]][attr_name])
            self.assertEqual(orig_array[0], A.multi_index[-10][attr_name])
            self.assertEqual(
                -10, A.query(coords=True).multi_index[-10]["coords"].view("i8")
            )
            assert_array_equal(orig_array[0:], A.multi_index[[(-10, 10)]][attr_name])
            assert_array_equal(
                orig_array[0:], A.multi_index[[slice(-10, 10)]][attr_name]
            )
            assert_array_equal(
                orig_array[0:10], A.multi_index[-10 : np.int64(-1)][attr_name]
            )
            assert_array_equal(orig_array, A.multi_index[:][attr_name])
            ned = A.nonempty_domain()[0]
            assert_array_equal(
                A.multi_index[ned[0] : ned[1]][attr_name], A.multi_index[:][attr_name]
            )

    def test_multirange_1d_sparse_double(self):
        attr_name = ""
        path = self.path("mr_1d_sparse_double")

        dom = tiledb.Domain(
            tiledb.Dim(name="coords", domain=(0, 30), tile=10, dtype=np.float64)
        )
        att = tiledb.Attr(name=attr_name, dtype=np.float64)
        schema = tiledb.ArraySchema(domain=dom, sparse=True, attrs=(att,))
        tiledb.SparseArray.create(path, schema)

        coords = np.linspace(0, 30, num=31)
        orig_array = np.random.rand(coords.size)

        with tiledb.open(path, "w") as A:
            A[coords] = orig_array

        with tiledb.open(path) as A:
            assert_array_equal(orig_array[[0]], A.multi_index[[0]][attr_name])
            assert_array_equal(orig_array[-1], A.multi_index[30][attr_name])
            assert_array_equal(orig_array[-1], A.multi_index[30.0][attr_name])
            assert_array_equal(
                orig_array[coords.size - 3 : coords.size],
                A.multi_index[
                    (28.0, 30.0),
                ][attr_name],
            )

            res = A.multi_index[slice(0, 5)]
            assert_array_equal(orig_array[0:6], res[attr_name])
            assert_array_equal(coords[0:6], res["coords"].astype(np.float64))

            # test slice range indexing
            ned = A.nonempty_domain()
            res = A.multi_index[: ned[0][1]]
            assert_array_equal(coords, res["coords"].astype(np.float64))

            res = A.multi_index[ned[0][0] : coords[15]]
            assert_array_equal(coords[:16], res["coords"].astype(np.float64))

    def test_multirange_2d_sparse_domain_utypes(self):
        attr_name = "foo"

        types = (np.uint8, np.uint16, np.uint32, np.uint64)

        for dtype in types:
            min = 0
            max = int(np.iinfo(dtype).max) - 1
            path = self.path("multi_index_2d_sparse_" + str(dtype.__name__))

            dom = tiledb.Domain(tiledb.Dim(domain=(min, max), tile=1, dtype=dtype))

            att = tiledb.Attr(name=attr_name, dtype=dtype)
            schema = tiledb.ArraySchema(domain=dom, sparse=True, attrs=(att,))
            tiledb.SparseArray.create(path, schema)

            coords = intspace(min, max, num=100, dtype=dtype)

            with tiledb.open(path, "w") as A:
                A[coords] = coords

            with tiledb.open(path) as A:
                res = A.multi_index[slice(coords[0], coords[-1])]
                assert_array_equal(res[attr_name], coords)
                assert_array_equal(res["__dim_0"].astype(dtype), coords)

                res = A.multi_index[coords[0]]
                assert_array_equal(res[attr_name], coords[0])
                assert_array_equal(res["__dim_0"].astype(dtype), coords[0])

                res = A.multi_index[coords[-1]]
                assert_array_equal(res[attr_name], coords[-1])
                assert_array_equal(res["__dim_0"].astype(dtype), coords[-1])

                midpoint = len(coords) // 2
                start = midpoint - 20
                stop = midpoint + 20
                srange = slice(coords[start], coords[stop])
                res = A.multi_index[srange]
                assert_array_equal(res[attr_name], coords[start : stop + 1])
                assert_array_equal(
                    res["__dim_0"].astype(dtype), coords[start : stop + 1]
                )

    def test_multirange_2d_sparse_float(self):
        attr_name = ""
        path = self.path("mr_2d_sparse_float")

        dom = tiledb.Domain(
            tiledb.Dim(domain=(0, 10), tile=1, dtype=np.float32),
            tiledb.Dim(domain=(0, 10), tile=1, dtype=np.float32),
        )
        att = tiledb.Attr(name=attr_name, dtype=np.float64)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True)
        tiledb.SparseArray.create(path, schema)

        orig_array = np.random.rand(11, 11)
        d1 = np.linspace(0, 10, num=11, dtype=np.float32)
        d2 = np.linspace(0, 10, num=11, dtype=np.float32)
        coords_d1, coords_d2 = np.meshgrid(d1, d2, indexing="ij")

        with tiledb.open(path, "w") as A:
            A[coords_d1.flatten(), coords_d2.flatten()] = orig_array

        with tiledb.open(path) as A:
            res = A.multi_index[[0], :]
            assert_array_equal(orig_array[[0], :].squeeze(), res[attr_name])
            assert_array_equal(coords_d1[0, :], res["__dim_0"])

            # ===
            res = A.multi_index[10, :]
            assert_array_equal(orig_array[[-1], :].squeeze(), res[attr_name])
            assert_array_equal(coords_d2[[-1], :].squeeze(), res["__dim_1"])

            # ===
            res = A.multi_index[[slice(0, 2), [5]]]
            assert_array_equal(
                np.vstack([orig_array[0:3, :], orig_array[5, :]]).flatten(),
                res[attr_name],
            )
            assert_array_equal(
                np.vstack((coords_d1[0:3], coords_d1[5])).flatten(), res["__dim_0"]
            )

            # ===
            res = A.multi_index[slice(0.0, 2.0), slice(2.0, 5.0)]
            assert_array_equal(orig_array[0:3, 2:6].flatten(), res[attr_name])
            assert_array_equal(coords_d1[0:3, 2:6].flatten(), res["__dim_0"])
            assert_array_equal(coords_d2[0:3, 2:6].flatten(), res["__dim_1"])
            res = A.multi_index[
                slice(np.float32(0.0), np.float32(2.0)),
                slice(np.float32(2.0), np.float32(5.0)),
            ]
            assert_array_equal(orig_array[0:3, 2:6].flatten(), res[attr_name])
            assert_array_equal(coords_d1[0:3, 2:6].flatten(), res["__dim_0"])
            assert_array_equal(coords_d2[0:3, 2:6].flatten(), res["__dim_1"])

    def test_multirange_1d_sparse_query(self):
        path = self.path("mr_1d_sparse_query")

        dom = tiledb.Domain(
            tiledb.Dim(name="coords", domain=(-100, 100), tile=1, dtype=np.float32)
        )
        attrs = [
            tiledb.Attr(name="U", dtype=np.float64),
            tiledb.Attr(name="V", dtype=np.uint32),
        ]

        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=True)
        tiledb.SparseArray.create(path, schema)

        U = np.random.rand(11)
        V = np.random.randint(0, np.iinfo(np.uint32).max, 11, dtype=np.uint32)

        coords = np.linspace(-10, 10, num=11, dtype=np.float32)
        data = {"U": U, "V": V}

        with tiledb.open(path, "w") as A:
            A[coords] = data

        with tiledb.open(path) as A:
            for k, d in data.items():
                Q = A.query(attrs=k)

                res = Q.multi_index[[-10]]
                assert_array_equal(d[[0]], res[k])

                assert_array_equal(coords[[0]], res["coords"].view("f4"))

                res = A.multi_index[10]
                assert_array_equal(d[[-1]].squeeze(), res[k])

                assert_array_equal(coords[[-1]], res["coords"].view("f4"))

                res = A.multi_index[[slice(coords[0], coords[2]), [coords[-1]]]]
                assert_array_equal(np.hstack([d[0:3], d[-1]]), res[k])

                # make sure full slice indexing works on query
                res = Q.multi_index[:]
                assert_array_equal(coords, res["coords"])

                # TODO: this should be an error
                # res = A.multi_index[10, :]
                # assert_array_equal(
                #    d[[-1]].squeeze(),
                #    res[k]
                # )

        with tiledb.open(path) as A:
            Q = A.query(coords=False, attrs=["U"])
            res = Q.multi_index[:]
            self.assertTrue("U" in res)
            self.assertTrue("V" not in res)
            self.assertTrue("coords" not in res)
            assert_array_equal(res["U"], data["U"])

    def test_multirange_1d_dense_vectorized(self):
        path = self.path("mr_1d_dense_vectorized")

        dom = tiledb.Domain(tiledb.Dim(domain=(0, 999), tile=1000, dtype=np.uint32))
        attrs = tiledb.Attr(name="", dtype=np.float64)

        schema = tiledb.ArraySchema(domain=dom, attrs=(attrs,), sparse=False)
        tiledb.DenseArray.create(path, schema)

        data = np.random.rand(1000)
        with tiledb.DenseArray(path, "w") as A:
            A[0] = data[0]
            A[-1] = data[-1]
            A[:] = data

        for _ in range(0, 50):
            with tiledb.DenseArray(path) as A:
                idxs = random.sample(range(0, 999), k=100)
                res = A.multi_index[idxs]
                assert_array_equal(data[idxs], res[""])

    def test_multirange_2d_dense_float(self):
        attr_name = ""
        path = self.path("multirange_2d_dense_float")

        dom = tiledb.Domain(
            tiledb.Dim(domain=(0, 10), tile=1, dtype=np.int64),
            tiledb.Dim(domain=(0, 10), tile=1, dtype=np.int64),
        )
        att = tiledb.Attr(name=attr_name, dtype=np.float64)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=False)
        tiledb.DenseArray.create(path, schema)

        orig_array = np.random.rand(11, 11)

        with tiledb.open(path, "w") as A:
            A[:] = orig_array

        with tiledb.open(path) as A:
            assert_array_equal(orig_array[[0], :], A.multi_index[[0], :][attr_name])
            assert_array_equal(
                orig_array[[-1, -1], :], A.multi_index[[10, 10], :][attr_name]
            )
            assert_array_equal(
                orig_array[0:4, 7:10], A.multi_index[[(0, 3)], slice(7, 9)][attr_name]
            )
            assert_array_equal(orig_array[:, :], A.multi_index[:, :][attr_name])
            # TODO this should be an error to match NumPy 1.12 semantics
            # assert_array_equal(
            #    orig_array[0:4,7:10],
            #    A.multi_index[[(np.float64(0),np.float64(3.0))], slice(7,9)][attr_name]
            # )

    @pytest.mark.parametrize("dtype", SUPPORTED_DATETIME64_DTYPES)
    def test_multirange_1d_sparse_datetime64(self, dtype):
        path = self.path("multirange_1d_sparse_datetime64")

        dates = rand_datetime64_array(10, dtype=dtype)
        dom = tiledb.Domain(
            tiledb.Dim(domain=(dates.min(), dates.max()), dtype=dtype, tile=1)
        )

        attr_name = ""
        att = tiledb.Attr(name=attr_name, dtype=dtype)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True)
        tiledb.SparseArray.create(path, schema)

        with tiledb.SparseArray(path, mode="w") as T:
            T[dates] = dates

        with tiledb.open(path) as A:
            res = A.multi_index[:]
            # check full range
            assert_tail_equal(dates, res[""], res["__dim_0"])

            # check range pairs
            for i in range(len(dates) - 1):
                start, stop = dates[i : i + 2]
                assert_tail_equal(
                    dates[i : i + 2],
                    A.multi_index[start:stop][""],
                    A.multi_index[start:stop]["__dim_0"],
                )

    def test_fix_473_sparse_index_bug(self):
        # test of fix for issue raised in
        # https://github.com/TileDB-Inc/TileDB-Py/pull/473#issuecomment-784675012

        uri = self.path("test_fix_473_sparse_index_bug")
        dom = tiledb.Domain(
            tiledb.Dim(name="x", domain=(0, 2**64 - 2), tile=1, dtype=np.uint64)
        )
        schema = tiledb.ArraySchema(
            domain=dom, sparse=True, attrs=[tiledb.Attr(name="a", dtype=np.uint64)]
        )

        tiledb.SparseArray.create(uri, schema)

        slice_index = slice(0, 4, None)

        with tiledb.SparseArray(uri, mode="r") as A:
            data = A.multi_index[slice_index]

            assert_array_equal(data["a"], np.array([], dtype=np.uint64))
            assert_array_equal(A.multi_index[:], [])

        with tiledb.open(uri, mode="w") as A:
            A[[10]] = {"a": [10]}

        with tiledb.open(uri) as A:
            assert_tail_equal(
                A.multi_index[slice_index]["a"],
                A.multi_index[:],
                A.multi_index[0:],
                A.multi_index[1:],
                A.multi_index[:10],
                A.multi_index[:11],
                np.array([], dtype=np.uint64),
            )

    @pytest.mark.skipif(not has_pandas(), reason="pandas>=1.0,<3.0 not installed")
    def test_fixed_multi_attr_df(self):
        uri = self.path("test_fixed_multi_attr_df")
        dom = tiledb.Domain(
            tiledb.Dim(name="dim", domain=(0, 0), tile=None, dtype=np.int32)
        )
        schema = tiledb.ArraySchema(
            domain=dom,
            sparse=True,
            attrs=[
                tiledb.Attr(
                    name="111", dtype=[("", np.int32), ("", np.int32), ("", np.int32)]
                )
            ],
        )

        tiledb.SparseArray.create(uri, schema)

        data_111 = np.array(
            [(1, 1, 1)], dtype=[("", np.int32), ("", np.int32), ("", np.int32)]
        )
        with tiledb.SparseArray(uri, mode="w") as A:
            A[0] = data_111

        with tiledb.SparseArray(uri, mode="r") as A:
            result = A.query(attrs=["111"])[0]
            assert_array_equal(result["111"], data_111)

            with self.assertRaises(tiledb.TileDBError):
                result = A.query(attrs=["111"]).df[0]

            result = A.query(attrs=["111"], use_arrow=False)
            assert_array_equal(result.df[0]["111"], data_111)

    @pytest.mark.skipif(not has_pandas(), reason="pandas>=1.0,<3.0 not installed")
    def test_var_multi_attr_df(self):
        uri = self.path("test_var_multi_attr_df")
        dom = tiledb.Domain(
            tiledb.Dim(name="dim", domain=(0, 2), tile=None, dtype=np.int32)
        )
        schema = tiledb.ArraySchema(
            domain=dom,
            sparse=True,
            attrs=[tiledb.Attr(name="1s", dtype=np.int32, var=True)],
        )

        tiledb.SparseArray.create(uri, schema)

        data = np.array(
            [
                np.array([1], dtype=np.int32),
                np.array([1, 1], dtype=np.int32),
                np.array([1, 1, 1], dtype=np.int32),
            ],
            dtype="O",
        )
        with tiledb.SparseArray(uri, mode="w") as A:
            A[[0, 1, 2]] = data

        with tiledb.SparseArray(uri, mode="r") as A:
            result = A.query(attrs=["1s"])
            assert_array_equal(result[0]["1s"][0], data[0])
            assert_array_equal(result[1]["1s"][0], data[1])
            assert_array_equal(result[2]["1s"][0], data[2])

            with self.assertRaises(tiledb.TileDBError):
                result = A.query(attrs=["1s"]).df[0]

            result = A.query(attrs=["1s"], use_arrow=False)
            assert_array_equal(result.df[0]["1s"][0], data[0])
            assert_array_equal(result.df[1]["1s"][0], data[1])
            assert_array_equal(result.df[2]["1s"][0], data[2])

    def test_multi_index_with_implicit_full_string_range(self):
        uri = self.path("test_multi_index_with_implicit_full_string_range")
        dom = tiledb.Domain(
            tiledb.Dim(name="dint", domain=(0, 4), tile=5, dtype=np.int32),
            tiledb.Dim(name="dstr", domain=(None, None), tile=None, dtype=np.bytes_),
        )
        schema = tiledb.ArraySchema(
            domain=dom, sparse=True, attrs=[tiledb.Attr(name="", dtype=np.int32)]
        )

        tiledb.Array.create(uri, schema)
        with tiledb.open(uri, mode="w") as A:
            d1 = np.concatenate((np.arange(5), np.arange(5)))
            d2 = np.asarray(
                ["a", "b", "ab", "ab", "c", "c", "c", "c", "d", "e"], dtype=np.bytes_
            )
            A[d1, d2] = np.array(np.random.randint(10, size=10), dtype=np.int32)

        with tiledb.open(uri, mode="r") as A:
            assert_array_equal(A[:][""], A.multi_index[:][""])
            assert_array_equal(A.multi_index[:][""], A.multi_index[:, :][""])

            assert_array_equal(A[1:4][""], A.multi_index[1:3][""])
            assert_array_equal(A.multi_index[1:3][""], A.multi_index[1:3, :][""])

            assert_array_equal(A[0][""], A.multi_index[0][""])
            assert_array_equal(A.multi_index[0][""], A.multi_index[0, :][""])

    def test_multi_index_open_timestamp_with_empty_nonempty_domain(self):
        uri = self.path("test_multi_index_open_timestamp_with_empty_nonempty_domain")
        dom = tiledb.Domain(tiledb.Dim(domain=(1, 3)))
        attr = tiledb.Attr(name="", dtype=np.int32)
        schema = tiledb.ArraySchema(domain=dom, sparse=True, attrs=[attr])
        tiledb.Array.create(uri, schema)

        with tiledb.open(uri, mode="w", timestamp=2) as A:
            d1 = np.array(np.random.randint(1, 11, size=3, dtype=np.int32))
            A[np.arange(1, 4)] = d1

        with tiledb.open(uri, mode="r", timestamp=1) as A:
            assert A.nonempty_domain() is None
            assert_array_equal(A.multi_index[:][""], A[:][""])

    @pytest.mark.skipif(not has_pandas(), reason="pandas>=1.0,<3.0 not installed")
    def test_multi_index_query_args(self):
        uri = self.path("test_multi_index_query_args")
        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(tiledb.Dim(name="dim", domain=(0, 9), dtype=np.uint8)),
            sparse=True,
            attrs=[
                tiledb.Attr(name="a", dtype=np.uint8),
                tiledb.Attr(name="b", dtype=np.uint8),
            ],
        )
        tiledb.Array.create(uri, schema)

        a = np.array(np.random.randint(10, size=10), dtype=np.int8)
        b = np.array(np.random.randint(10, size=10), dtype=np.int8)

        with tiledb.open(uri, mode="w") as A:
            A[np.arange(10)] = {"a": a, "b": b}

        with tiledb.open(uri, mode="r") as A:
            q = A.query(cond="a >= 5", attrs=["a"])
            assert {"a", "dim"} == q.multi_index[:].keys() == q[:].keys()
            assert_array_equal(q.multi_index[:]["a"], q[:]["a"])
            assert_array_equal(q.multi_index[:]["a"], q.df[:]["a"])
            assert all(q[:]["a"] >= 5)

    @pytest.mark.skipif(not has_pandas(), reason="pandas>=1.0,<3.0 not installed")
    def test_multi_index_timing(self):
        path = self.path("test_multi_index_timing")
        attr_name = "a"

        make_1d_dense(path, attr_name=attr_name)
        tiledb.stats_enable()
        with tiledb.open(path) as A:
            assert_array_equal(A.df[:][attr_name], np.arange(36))
            internal_stats = tiledb.main.python_internal_stats()
            assert "py.getitem_time :" in internal_stats
            assert "py.getitem_time.buffer_conversion_time :" in internal_stats
            assert "py.getitem_time.pandas_index_update_time :" in internal_stats
        tiledb.stats_disable()

    @pytest.mark.skipif(not has_pandas(), reason="pandas>=1.0,<3.0 not installed")
    def test_fixed_width_char(self):
        uri = self.path("test_fixed_width_char")
        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(tiledb.Dim(name="dim", domain=(0, 2), dtype=np.uint8)),
            attrs=[tiledb.Attr(dtype="|S3")],
        )
        tiledb.Array.create(uri, schema)

        data = np.array(["cat", "dog", "hog"], dtype="|S3")

        with tiledb.open(uri, mode="w") as A:
            A[:] = data

        with tiledb.open(uri, mode="r") as A:
            assert all(A.query(use_arrow=True).df[:][""] == data)


# parametrize dtype and sparse
@pytest.mark.parametrize(
    "dim_dtype",
    [
        np.int64,
        np.uint64,
        np.int32,
        np.uint32,
        np.int16,
        np.uint16,
        np.int8,
        np.uint8,
    ],
)
class TestMultiIndexND(DiskTestCase):
    def test_multi_index_ndarray(self, dim_dtype):
        # TODO support for dense?
        sparse = True  # ndarray indexing currently only supported for sparse

        path = self.path("test_multi_index_ndarray")

        ncells = 10
        data = np.arange(ncells - 1)
        coords = np.arange(ncells - 1)

        # use negative range for sparse
        if sparse and np.issubdtype(dim_dtype, np.signedinteger):
            coords -= 4

        dom = tiledb.Domain(
            tiledb.Dim(domain=(coords.min(), coords.max()), dtype=dim_dtype)
        )
        att = tiledb.Attr(dtype=np.int8)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=sparse)
        tiledb.Array.create(path, schema)

        with tiledb.open(path, "w") as A:
            if sparse:
                A[coords] = data
            else:
                A[:] = data

        with tiledb.open(path) as A:
            assert_dict_arrays_equal(
                A.multi_index[coords.tolist()], A.multi_index[coords]
            )
            assert_dict_arrays_equal(
                A.multi_index[coords.tolist()], A.multi_index[coords]
            )

    def test_multi_index_ndarray_2d(self, dim_dtype):
        sparse = False

        path = self.path("test_multi_index_ndarray_2d")

        ncells = 10
        ext = ncells - 1
        if sparse:
            data = np.arange(ext)
        else:
            data = np.arange(ext**2).reshape(ext, ext)
        d1_coords = np.arange(ext)
        d2_coords = np.arange(ext, 0, -1)

        # use negative range for sparse
        if sparse and np.issubdtype(dim_dtype, np.signedinteger):
            d1_coords -= 4

        d1 = tiledb.Dim(
            name="d1", domain=(d1_coords.min(), d1_coords.max()), dtype=dim_dtype
        )
        d2 = tiledb.Dim(
            name="d2", domain=(d2_coords.min(), d2_coords.max()), dtype=dim_dtype
        )
        dom = tiledb.Domain([d1, d2])

        att = tiledb.Attr(dtype=np.int8)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=sparse)
        tiledb.Array.create(path, schema)

        with tiledb.open(path, "w") as A:
            if sparse:
                A[d1_coords.tolist(), d2_coords.tolist()] = {"": data}
            else:
                A[:] = data
                # raise ValueError("Test only support sparse")

        with tiledb.open(path) as A:
            assert_dict_arrays_equal(
                A.multi_index[d1_coords.tolist(), d2_coords.tolist()],
                A.multi_index[d1_coords, d2_coords],
            )

            # note: np.flip below because coords are in reverse order, which is how
            #       tiledb will return the results for the first query, but not second
            assert_dict_arrays_equal(
                A.multi_index[d1_coords.tolist(), np.flip(d2_coords.tolist())],
                A.multi_index[d1_coords, :],
            )

            slc = slice(0, ncells - 1, 2)
            assert_dict_arrays_equal(
                A.multi_index[d1_coords[slc].tolist(), :],
                A.multi_index[d1_coords[slc], :],
            )
            assert_dict_arrays_equal(
                A.multi_index[:, d2_coords[slc]], A.multi_index[:, d2_coords[slc]]
            )
