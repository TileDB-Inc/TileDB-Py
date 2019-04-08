try:
    import dask, dask.array as da
    import_failed = False
except ImportError:
    import_failed = True

import unittest

import tiledb
from tiledb.tests.common import DiskTestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_approx_equal


class DaskSupport(DiskTestCase):
    def setUp(self):
        if import_failed:
            self.skipTest("Dask not available")
        else:
            super().setUp()

    @unittest.expectedFailure
    def test_dask_from_numpy_1d(self):
        uri = self.path("np_1attr")
        A = np.random.randn(50,50)
        T = tiledb.from_numpy(uri, A, tile=50)
        T.close()

        with tiledb.open(uri) as T:
            D = da.from_tiledb(T)
            assert_array_equal(D, A)

        D2 = da.from_tiledb(uri)
        assert_array_equal(D2, A)
        self.assertAlmostEqual(np.mean(A), D2.mean().compute(scheduler='single-threaded'))

    def _make_multiattr_2d(self, uri, shape=(0,100), tile=10):
        dom = tiledb.Domain(
                tiledb.Dim("x", (0,10), dtype=np.uint64, tile=tile),
                tiledb.Dim("y", (0,50), dtype=np.uint64, tile=tile))
        schema = tiledb.ArraySchema(
                    attrs=(tiledb.Attr("attr1"),
                           tiledb.Attr("attr2")),
                    domain=dom)

        tiledb.DenseArray.create(uri, schema)


    @unittest.expectedFailure
    def test_dask_multiattr_2d(self):
        uri = self.path("multiattr")

        self._make_multiattr_2d(uri)

        with tiledb.DenseArray(uri, 'w') as T:
            ar1 = np.random.randn(*T.schema.shape)
            ar2 = np.random.randn(*T.schema.shape)
            T[:] = {'attr1': ar1,
                    'attr2': ar2}
        with tiledb.DenseArray(uri, mode='r', attr='attr2') as T:
            # basic round-trip from dask.array
            D = da.from_tiledb(T, attribute='attr2')
            assert_array_equal(ar2, np.array(D))

        # smoke-test computation
        # note: re-init from_tiledb each time, or else dask just uses the cached materialization
        D = da.from_tiledb(uri, attribute='attr2')
        self.assertAlmostEqual(np.mean(ar2), D.mean().compute(scheduler='threads'))
        D = da.from_tiledb(uri, attribute='attr2')
        self.assertAlmostEqual(np.mean(ar2), D.mean().compute(scheduler='single-threaded'))
        D = da.from_tiledb(uri, attribute='attr2')
        self.assertAlmostEqual(np.mean(ar2), D.mean().compute(scheduler='processes'))

        # test dask.distributed
        from dask.distributed import Client
        D = da.from_tiledb(uri, attribute='attr2')
        with Client() as client:
            assert_approx_equal(D.mean().compute(), np.mean(ar2))

    @unittest.expectedFailure
    def test_dask_write(self):
        uri = self.path("dask_w")
        D = da.random.random(10,10)
        D.to_tiledb(uri)
        DT = da.from_tiledb(uri)
        assert_array_equal(D, DT)
