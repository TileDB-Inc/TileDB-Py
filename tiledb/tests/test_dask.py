import pytest

da = pytest.importorskip("dask.array")

import sys
import tiledb
from tiledb.tests.common import DiskTestCase

import numpy as np
from numpy.testing import assert_array_equal, assert_approx_equal

# override the no_output fixture because it conflicts with these tests
#   eg: "ResourceWarning: unclosed event loop"
@pytest.fixture(scope="function", autouse=True)
def no_output():
    pass


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
class TestDaskSupport(DiskTestCase):
    def test_dask_from_numpy_1d(self):
        uri = self.path("np_1attr")
        A = np.random.randn(50, 50)
        T = tiledb.from_numpy(uri, A, tile=50)
        T.close()

        with tiledb.open(uri) as T:
            D = da.from_tiledb(T)
            assert_array_equal(D, A)

        D2 = da.from_tiledb(uri)
        assert_array_equal(D2, A)
        self.assertAlmostEqual(
            np.mean(A), D2.mean().compute(scheduler="single-threaded")
        )

    def _make_multiattr_2d(self, uri, shape=(0, 100), tile=10):
        dom = tiledb.Domain(
            tiledb.Dim("x", (0, 10), dtype=np.uint64, tile=tile),
            tiledb.Dim("y", (0, 50), dtype=np.uint64, tile=tile),
        )
        schema = tiledb.ArraySchema(
            attrs=(tiledb.Attr("attr1"), tiledb.Attr("attr2")), domain=dom
        )

        tiledb.DenseArray.create(uri, schema)

    def test_dask_multiattr_2d(self):
        uri = self.path("multiattr")

        self._make_multiattr_2d(uri)

        with tiledb.DenseArray(uri, "w") as T:
            ar1 = np.random.randn(*T.schema.shape)
            ar2 = np.random.randn(*T.schema.shape)
            T[:] = {"attr1": ar1, "attr2": ar2}
        with tiledb.DenseArray(uri, mode="r", attr="attr2") as T:
            # basic round-trip from dask.array
            D = da.from_tiledb(T, attribute="attr2")
            assert_array_equal(ar2, np.array(D))

        # smoke-test computation
        # note: re-init from_tiledb each time, or else dask just uses the cached materialization
        D = da.from_tiledb(uri, attribute="attr2")
        self.assertAlmostEqual(np.mean(ar2), D.mean().compute(scheduler="threads"))
        D = da.from_tiledb(uri, attribute="attr2")
        self.assertAlmostEqual(
            np.mean(ar2), D.mean().compute(scheduler="single-threaded")
        )
        D = da.from_tiledb(uri, attribute="attr2")
        self.assertAlmostEqual(np.mean(ar2), D.mean().compute(scheduler="processes"))

        # test dask.distributed
        from dask.distributed import Client

        D = da.from_tiledb(uri, attribute="attr2")
        with Client() as client:
            assert_approx_equal(D.mean().compute(), np.mean(ar2))

    def test_dask_write(self):
        uri = self.path("dask_w")
        D = da.random.random(10, 10)
        D.to_tiledb(uri)
        DT = da.from_tiledb(uri)
        assert_array_equal(D, DT)

    def test_dask_overlap_blocks(self):
        uri = self.path("np_overlap_blocks")
        A = np.ones((2, 50, 50))
        T = tiledb.from_numpy(uri, A, tile=(1, 5, 5))
        T.close()

        with tiledb.open(uri) as T:
            D = da.from_tiledb(T)
            assert_array_equal(D, A)

        D2 = da.from_tiledb(uri)
        assert_array_equal(D2, A)

        D3 = D2.map_overlap(
            lambda x: x + 1, depth={0: 0, 1: 1, 2: 1}, dtype=A.dtype
        ).compute()
        assert_array_equal(D2 * 2, D3)

    def test_labeled_dask_overlap_blocks(self):
        uri = self.path("np_labeled_overlap_blocks")
        A = np.ones((2, 50, 50))

        dom = tiledb.Domain(
            tiledb.Dim(name="BANDS", domain=(0, 1), tile=1),
            tiledb.Dim(name="Y", domain=(0, 49), tile=5, dtype=np.uint64),
            tiledb.Dim(name="X", domain=(0, 49), tile=5, dtype=np.uint64),
        )

        schema = tiledb.ArraySchema(
            domain=dom,
            sparse=False,
            attrs=[tiledb.Attr(name="TDB_VALUES", dtype=A.dtype)],
        )

        tiledb.DenseArray.create(uri, schema)

        with tiledb.open(uri, "w", attr="TDB_VALUES") as T:
            T[:] = A

        D2 = da.from_tiledb(uri, attribute="TDB_VALUES")

        D3 = D2.map_overlap(
            lambda x: x + 1, depth={0: 0, 1: 1, 2: 1}, dtype=D2.dtype
        ).compute()
        assert_array_equal(D2 + 1, D3)

    def test_labeled_dask_blocks(self):
        uri = self.path("np_labeled_map_blocks")
        A = np.ones((2, 50, 50))

        dom = tiledb.Domain(
            tiledb.Dim(name="BANDS", domain=(0, 1), tile=1),
            tiledb.Dim(name="Y", domain=(0, 49), tile=5, dtype=np.uint64),
            tiledb.Dim(name="X", domain=(0, 49), tile=5, dtype=np.uint64),
        )

        schema = tiledb.ArraySchema(
            domain=dom,
            sparse=False,
            attrs=[tiledb.Attr(name="TDB_VALUES", dtype=A.dtype)],
        )

        tiledb.DenseArray.create(uri, schema)
        with tiledb.open(uri, "w", attr="TDB_VALUES") as D1:
            D1[:] = A

        D2 = da.from_tiledb(uri, attribute="TDB_VALUES")

        D3 = D2.map_blocks(lambda x: x + 1, dtype=D2.dtype).compute(
            scheduler="processes"
        )
        assert_array_equal(D2 + 1, D3)
