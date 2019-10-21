"""
TODO
- # implement mock of expected behavior in pure numpy w/ test function


- implement read function and tests (single [x], multi-attribute [ ])
- implement custom indexer
- implement oindex...
"""

import tiledb
from tiledb.multirange_indexing import *
import os, numpy as np
import sys

from numpy.testing import assert_array_equal
import unittest
from tiledb.tests.common import *

def make_1d_dense(ctx, path, attr_name=''):
    a_orig = np.arange(36)

    dom = tiledb.Domain(tiledb.Dim(domain=(0, 35), tile=35, dtype=np.uint64, ctx=ctx),
                        ctx=ctx)
    att = tiledb.Attr(name=attr_name, dtype=np.int64, ctx=ctx)
    schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=False, ctx=ctx)
    tiledb.DenseArray.create(path, schema)

    with tiledb.DenseArray(path, 'w') as A:
        A[:] = a_orig


def make_2d_dense(ctx, path, attr_name=''):
    a_orig = np.arange(1,37).reshape(9, 4)

    dom = tiledb.Domain(tiledb.Dim(domain=(0, 8), tile=9, dtype=np.uint64, ctx=ctx),
                        tiledb.Dim(domain=(0, 3), tile=4, dtype=np.uint64, ctx=ctx),
                        ctx=ctx)
    att = tiledb.Attr(name=attr_name, dtype=np.int64, ctx=ctx)
    schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=False, ctx=ctx)
    tiledb.DenseArray.create(path, schema)

    with tiledb.DenseArray(path, 'w') as A:
        A[:] = a_orig

class TestMultiRange(DiskTestCase):

    def test_mr_1d_1dim_ranges(self):
        path = self.path('test_mr_1d_1dim_ranges')
        attr_name = 'a'

        ctx = tiledb.Ctx()
        make_1d_dense(ctx, path, attr_name=attr_name)

        expected = np. array([0],
                             dtype=np.uint64)


        with tiledb.DenseArray(path) as A:
            ranges = ( ((0, 0),), )
            expected = np.array([0], dtype=np.int64)
            a = tiledb.libtiledb.multi_index(A, (attr_name,), ranges)[attr_name]
            assert_array_equal(a, expected)
            self.assertEqual(a.dtype, expected.dtype)

            #===
        #with tiledb.DenseArray(path) as A:
            ranges2 = ( ((1, 1), (5,8)), )
            expected2 = np.array([1,5,6,7,8], dtype=np.int64)
            a2 = tiledb.libtiledb.multi_index(A, (attr_name,), ranges2)[attr_name]
            assert_array_equal(a2, expected2)
            self.assertEqual(a2.dtype, expected2.dtype)

    def test_mr_2d_1dim_ranges(self):
        path = self.path('test_mr_1dim_ranges')
        attr_name = 'a'

        ctx = tiledb.Ctx()
        make_2d_dense(ctx, path, attr_name=attr_name)

        expected = np. array([ 1,  2,  3,  4,
                               21, 22, 23, 24,
                               25, 26, 27, 28,
                               29, 30, 31, 32, 33,
                               34, 35, 36],
                             dtype=np.uint64)

        ranges = (
            ( (0, 0), (5,8), ),
        )

        with tiledb.DenseArray(path) as A:
            a = tiledb.libtiledb.multi_index(A, (attr_name,), ranges)[attr_name]

            assert_array_equal(a, expected)

    def test_mr_2d_2dim_ranges(self):
        ctx = tiledb.Ctx()
        path = self.path('test_mr_2dim_ranges')
        attr_name = 'a'

        make_2d_dense(ctx, path, attr_name=attr_name)

        expected = np.array([1, 2, 3, 4,
                             5, 6, 7, 8,
                             9, 10, 11, 12,
                             13, 14, 15, 16,
                             17, 18, 19, 20])

        ranges = (
            ( (0,4), ),
            ( (0,3), )
        )

        with tiledb.DenseArray(path) as A:
            a = tiledb.libtiledb.multi_index(A, (attr_name,), ranges)[attr_name]
            assert_array_equal(a, expected)


    def test_shape_funcs(self):
        #-----------
        range1el = ( ((1,1),), )
        self.assertEqual(
            mr_dense_result_shape(range1el),
            (1,))
        self.assertEqual(
            mr_dense_result_numel(range1el),
            1)

        #-----------
        range1d = tuple([((1, 2), (4, 4))])
        self.assertEqual(
            mr_dense_result_shape(range1d),
            (3,))
        self.assertEqual(
            mr_dense_result_numel(range1d),
            3
        )

        #-----------
        range2d1 = (
            ( (3,6), (7,7), (10,12) ),
            ( (5,7), ),
        )

        self.assertEqual(
            mr_dense_result_shape(range2d1),
            (8,3)
        )
        self.assertEqual(
            mr_dense_result_numel(range2d1),
            24
        )

        #-----------
        range2d2 = (
            [(3,6), (7,7), (10,12)],
            [(5,7), (10,10)]
        )

    def test_3d(self):
        range3d1 = (
            ( (2,4), ),
            ( (3,6), ),
            ( (1,4), (5,9) )
        )

        #self.assertEqual()

    def test_sel_to_ranges(self):
        class Obj(object): pass
        class IBI(object):
            def __getitem__(self, idx):
                return idx

        def make_arr(ndim):
            arr = Obj()
            arr.schema = Obj()
            arr.schema.domain = Obj()
            arr.schema.domain.ndim = ndim
            return arr

        ibi = IBI()
        # ndim = 1
        arr = make_arr(1)
        m = MultiRangeIndexer(arr)
        self.assertEqual(
            m.getitem_ranges( ibi[[1]] ),
            (((1, 1),),)
        )
        self.assertEqual(
            m.getitem_ranges( ibi[[1, 2]] ),
            (((1, 1), (2, 2)),)
        )
        self.assertEqual(
            m.getitem_ranges( ibi[slice(1, 2)], ),
            (((1, 2),),)
        )
        self.assertEqual(
            m.getitem_ranges( ibi[1:2, 3:5] ),
            (((1, 2),), ((3, 5),),)
        )

        # ndim = 2
        arr2 = make_arr(2)
        m = MultiRangeIndexer(arr2)

        self.assertEqual(
            m.getitem_ranges( ibi[[1]] ),
            (((1, 1),), ())
        )
        self.assertEqual(
            m.getitem_ranges( ibi[slice(1, 33)] ),
            (((1, 33),), ())
        )
        self.assertEqual(
            m.getitem_ranges( ibi[ [1, 2], [1], slice(1, 3) ] ),
            (((1, 1), (2, 2)), ((1, 1),), ((1, 3),))
        )

        # ndim = 3
        arr3 = make_arr(3)
        m = MultiRangeIndexer(arr3)

        self.assertEqual(
            m.getitem_ranges( ibi[1, 2, 3] ),
            (
                ((1, 1),), ((2, 2,),), ((3, 3),)
            )
        )
        self.assertEqual(
            m.getitem_ranges( ibi[1, 2] ),
            (
                (((1, 1),), ((2, 2,),), ())
            )
        )
        self.assertEqual(
            m.getitem_ranges( ibi[1:2, 3:4]),
            (((1, 2),), ((3, 4),), ())
        )
        self.assertEqual(
            m.getitem_ranges( ibi[1:2, 3:4, 5:6] ),
            (((1, 2),), ((3, 4),), ((5, 6),))
        )
        self.assertEqual(
            m.getitem_ranges( ibi[[1], [2], [5, 6]] ),
            (((1, 1),), ((2, 2),), ((5, 5), (6, 6),))
        )
        self.assertEqual(
            m.getitem_ranges( ibi[1, [slice(3, 6), 8], slice(4, 6)] ),
            (
                ((1, 1),),
                ((3, 6), (8, 8)),
                ((4, 6),)
            )
        )

    def test_mr_1d_dense_int(self):
        attr_name = ''
        ctx = tiledb.Ctx()
        path = self.path('multi_index_1d')

        dom = tiledb.Domain(tiledb.Dim(domain=(-10, 10), tile=9,
                                       dtype=np.int64, ctx=ctx),
                            ctx=ctx)
        att = tiledb.Attr(name=attr_name, dtype=np.float32, ctx=ctx)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), ctx=ctx)
        tiledb.DenseArray.create(path, schema)

        orig_array = np.random.rand(schema.domain.dim(0).size).astype(np.float32)
        with tiledb.open(path, 'w') as A:
            A[:] = orig_array

        with tiledb.open(path) as A:
            # stepped ranges are not supported
            with self.assertRaises(ValueError):
                A.multi_index[ 1::2 ]

            assert_array_equal(
                orig_array[ [0,-1] ],
                A.multi_index[ [-10,10] ][attr_name]
            )
            self.assertEqual(
                orig_array[0],
                A.multi_index[-10][attr_name]
            )

    def test_mr_1d_sparse_double(self):
        attr_name = ''
        ctx = tiledb.Ctx()
        path = self.path('mr_1d_sparse_double')

        dom = tiledb.Domain(tiledb.Dim(domain=(0, 30), tile=10,
                                       dtype=np.float64, ctx=ctx),
                            ctx=ctx)
        att = tiledb.Attr(name=attr_name, dtype=np.float64, ctx=ctx)
        schema = tiledb.ArraySchema(domain=dom, sparse=True, attrs=(att,), ctx=ctx)
        tiledb.SparseArray.create(path, schema)

        coords = np.linspace(0,30, num=31)
        orig_array = np.random.rand(coords.size)

        with tiledb.open(path, 'w') as A:
            A[coords] = orig_array

        with tiledb.open(path) as A:
            assert_array_equal(
                orig_array[ [0] ],
                A.multi_index[ [0] ][attr_name]
            )
            assert_array_equal(
                orig_array[-1],
                A.multi_index[30][attr_name]
            )

            res = A.multi_index[ slice(0,5) ]
            assert_array_equal(
                orig_array[0:6],
                res[attr_name]
            )
            assert_array_equal(
                coords[0:6],
                res['coords'].astype(np.float64)
            )

    def test_mr_2d_sparse_domain_utypes(self):
        attr_name = 'foo'
        ctx = tiledb.Ctx()

        types = (np.uint8, np.uint16, np.uint32, np.uint64)

        for dtype in types:
            min = 0
            max = np.iinfo(dtype).max - 1
            path = self.path('multi_index_2d_sparse_' + str(dtype.__name__))

            dom = tiledb.Domain(tiledb.Dim(domain=(min, max), tile=1,
                                           dtype=dtype, ctx=ctx),
                                ctx=ctx)

            att = tiledb.Attr(name=attr_name, dtype=dtype, ctx=ctx)
            schema = tiledb.ArraySchema(domain=dom, sparse=True, attrs=(att,), ctx=ctx)
            tiledb.SparseArray.create(path, schema)

            coords = intspace(min, max, num=100, dtype=dtype)

            with tiledb.open(path, 'w') as A:
                A[coords] = coords

            with tiledb.open(path) as A:

                res = A.multi_index[slice(coords[0], coords[-1])]
                assert_array_equal(
                    res[attr_name],
                    coords
                )
                assert_array_equal(
                    res['coords'].astype(dtype),
                    coords
                )

                res = A.multi_index[coords[0]]
                assert_array_equal(
                    res[attr_name],
                    coords[0]
                )
                assert_array_equal(
                    res['coords'].astype(dtype),
                    coords[0]
                )

                res = A.multi_index[coords[-1]]
                assert_array_equal(
                    res[attr_name],
                    coords[-1]
                )
                assert_array_equal(
                    res['coords'].astype(dtype),
                    coords[-1]
                )

                midpoint = len(coords)//2
                start = midpoint-20
                stop = midpoint+20
                srange = slice(coords[start], coords[stop])
                res = A.multi_index[ srange ]
                assert_array_equal(
                    res[attr_name],
                    coords[start:stop+1])
                assert_array_equal(
                    res['coords'].astype(dtype),
                    coords[start:stop+1])

    def test_mr_2d_sparse_float(self):
        attr_name = ''
        ctx = tiledb.Ctx()
        path = self.path('mr_2d_sparse_float')

        dom = tiledb.Domain(tiledb.Dim(domain=(0, 10), tile=1,
                                       dtype=np.float32, ctx=ctx),
                            tiledb.Dim(domain=(0, 10), tile=1,
                                       dtype=np.float32, ctx=ctx),
                            ctx=ctx)
        att = tiledb.Attr(name=attr_name, dtype=np.float64, ctx=ctx)
        schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True, ctx=ctx)
        tiledb.SparseArray.create(path, schema)

        orig_array = np.random.rand(11,11)
        d1 = np.linspace(0, 10, num=11, dtype=np.float32)
        d2 = np.linspace(0, 10, num=11, dtype=np.float32)
        coords_d1,coords_d2 = np.meshgrid(d1,d2,indexing='ij')
        coords_d1,coords_d2 = coords_d1.flatten(), coords_d2.flatten()
        coords = np.column_stack([coords_d1, coords_d2]).reshape(11,11,2)

        with tiledb.open(path, 'w') as A:
            A[coords_d1,coords_d2] = orig_array

        with tiledb.open(path) as A:
            res = A.multi_index[[0], :]
            assert_array_equal(
                orig_array[[0], :].squeeze(),
                res[attr_name]
            )
            assert_array_equal(
                coords[[0], :].squeeze(),
                res['coords'].view('f4').reshape(-1,2)
            )

            res = A.multi_index[10, :]
            assert_array_equal(
                orig_array[[-1], :].squeeze(),
                res[attr_name]
            )
            assert_array_equal(
                coords[[-1], :].squeeze(),
                res['coords'].view('f4').reshape(-1,2)
            )

            res = A.multi_index[ [ slice(0,2), [5]] ]
            assert_array_equal(
                np.vstack([orig_array[0:3,:], orig_array[5,:]]).flatten(),
                res[attr_name]
            )
            assert_array_equal(
                np.vstack([coords[0:3,:].reshape(-1,2), coords[5,:]]),
                res['coords'].view('f4').reshape(-1,2)
            )
