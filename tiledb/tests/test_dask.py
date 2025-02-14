import warnings

import numpy as np
import pytest

import tiledb

from .common import DiskTestCase

# Skip this test if dask is unavailable
da_array = pytest.importorskip("dask.array")
da_distributed = pytest.importorskip("dask.distributed")


class TestDaskSupport(DiskTestCase):
    def test_dask_from_numpy_1d(self):
        uri = self.path("np_1attr")
        A = np.random.randn(50, 50)
        T = tiledb.from_numpy(uri, A, tile=50)
        T.close()

        with tiledb.open(uri) as T:
            D = da_array.from_tiledb(T)
            np.testing.assert_array_equal(D, A)

        D2 = da_array.from_tiledb(uri)
        np.testing.assert_array_equal(D2, A)
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

    @pytest.mark.flaky(reruns=3, reruns_delay=2, rerun_except="TileDBError")
    @pytest.mark.filterwarnings("ignore:There is no current event loop")
    def test_dask_multiattr_2d(self):
        uri = self.path("multiattr")

        self._make_multiattr_2d(uri)

        with tiledb.DenseArray(uri, "w") as T:
            ar1 = np.random.randn(*T.schema.shape)
            ar2 = np.random.randn(*T.schema.shape)
            T[:] = {"attr1": ar1, "attr2": ar2}
        with tiledb.DenseArray(uri, mode="r", attr="attr2") as T:
            # basic round-trip from dask.array
            D = da_array.from_tiledb(T, attribute="attr2")
            np.testing.assert_array_equal(ar2, np.array(D))

        # smoke-test computation
        # note: re-init from_tiledb each time, or else dask just uses the cached materialization
        D = da_array.from_tiledb(uri, attribute="attr2")
        self.assertAlmostEqual(np.mean(ar2), D.mean().compute(scheduler="threads"))
        D = da_array.from_tiledb(uri, attribute="attr2")
        self.assertAlmostEqual(
            np.mean(ar2), D.mean().compute(scheduler="single-threaded")
        )
        D = da_array.from_tiledb(uri, attribute="attr2")
        self.assertAlmostEqual(np.mean(ar2), D.mean().compute(scheduler="processes"))

        # test dask.distributed
        D = da_array.from_tiledb(uri, attribute="attr2")
        with da_distributed.Client():
            np.testing.assert_approx_equal(D.mean().compute(), np.mean(ar2))

    def test_dask_write(self):
        uri = self.path("dask_w")
        D = da_array.random.random(10, 10)
        D.to_tiledb(uri)
        DT = da_array.from_tiledb(uri)
        np.testing.assert_array_equal(D, DT)

    def test_dask_overlap_blocks(self):
        uri = self.path("np_overlap_blocks")
        A = np.ones((2, 50, 50))
        T = tiledb.from_numpy(uri, A, tile=(1, 5, 5))
        T.close()

        with tiledb.open(uri) as T:
            D = da_array.from_tiledb(T)
            np.testing.assert_array_equal(D, A)

        D2 = da_array.from_tiledb(uri)
        np.testing.assert_array_equal(D2, A)

        D3 = D2.map_overlap(
            lambda x: x + 1, depth={0: 0, 1: 1, 2: 1}, dtype=A.dtype, boundary="none"
        ).compute()
        np.testing.assert_array_equal(D2 * 2, D3)

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

        D2 = da_array.from_tiledb(uri, attribute="TDB_VALUES")

        D3 = D2.map_overlap(
            lambda x: x + 1, depth={0: 0, 1: 1, 2: 1}, dtype=D2.dtype, boundary="none"
        ).compute()
        np.testing.assert_array_equal(D2 + 1, D3)

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

        D2 = da_array.from_tiledb(uri, attribute="TDB_VALUES")

        D3 = D2.map_blocks(lambda x: x + 1, dtype=D2.dtype).compute(
            scheduler="processes"
        )
        np.testing.assert_array_equal(D2 + 1, D3)


def test_sc33742_dask_array_object_dtype_conversion():
    # This test verifies that an array can be converted to buffer after serialization
    # through several dask.distributed compute steps. The original source of the issue
    # was that a `dtype == dtype("O")` check was returning false, presumably because the
    # dtype object was not === after serialization.
    import random

    import dask
    import numpy as np

    @dask.delayed
    def get_data():
        dd = dask.delayed(
            lambda x=0: {
                "Z": np.array(
                    [
                        np.zeros((random.randint(60, 100),), np.dtype("float64")),
                        np.zeros((random.randint(1, 50),), np.dtype("float64")),
                    ],
                    dtype=np.dtype("O"),
                )
            }
        )()
        return dask.delayed([dd])

    @dask.delayed
    def use_data(data):
        f = dask.compute(data, traverse=True)[0][0]

        from tiledb import main

        main.array_to_buffer(f["Z"], True, False)

    # Various warnings are raised by dask.distributed in different Python versions and
    # package combinations (eg Python 3.7 and older tornado), but they are not relevant to
    # this test.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        global client
        client = da_distributed.Client(
            da_distributed.LocalCluster(scheduler_port=9786, dashboard_address=9787)
        )

        w = []

        data = dask.delayed(get_data)()
        w.append(use_data(data))

        futures = client.compute(w)
        client.gather(futures)
