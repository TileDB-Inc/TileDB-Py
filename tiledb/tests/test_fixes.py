import numpy as np
import tiledb
import concurrent, concurrent.futures

import pytest
from tiledb.tests.common import DiskTestCase, has_pandas
from numpy.testing import assert_array_equal


class FixesTest(DiskTestCase):
    def test_ch7727_float32_dim_estimate_incorrect(self):
        # set max allocation: because windows won't overallocate
        with tiledb.scope_ctx({"py.alloc_max_bytes": 1024**2 * 100}):
            uri = self.path()
            dom = tiledb.Domain(tiledb.Dim("x", domain=(1, 100), dtype=np.float32))
            att = tiledb.Attr("", dtype=np.bytes_)
            schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True)
            tiledb.Array.create(uri, schema)

            with tiledb.open(uri, mode="w") as T:
                T[50.4] = b"hello"

            with tiledb.open(uri, mode="r") as T:
                assert T[:][""] == b"hello"
                assert T[50.4][""] == b"hello"

    def test_ch8292(self):
        # test fix for ch8292
        # We need to ensure that py.alloc_max_bytes is *not* applied to
        # dense arrays. Dense arrays should have exact estimates based
        # on the ranges, so there should be no risk of over-estimates.
        # This test sets py.alloc_max_bytes to 1 less than the expected
        # result array size, and asserts that the allocated buffers match
        # the expected result size rather than py.alloc_max_bytes.
        uri = self.path()
        max_val = 1024**2 + 1
        with tiledb.from_numpy(uri, np.uint8(range(max_val))):
            pass
        with tiledb.scope_ctx(
            {"py.init_buffer_bytes": 2 * 1024**2, "py.alloc_max_bytes": 1024**2}
        ) as ctx3:
            with tiledb.open(uri) as b:
                q = tiledb.main.PyQuery(ctx3, b, ("",), (), 0, False)
                q._return_incomplete = True
                q.set_ranges([[(0, max_val)]])
                q._allocate_buffers()
                buffers = list(*q._get_buffers().values())
                assert buffers[0].nbytes == max_val

    @pytest.mark.skipif(not has_pandas(), reason="pandas not installed")
    def test_ch10282_concurrent_multi_index(self):
        """Test concurrent access to a single tiledb.Array using
        Array.multi_index and Array.df. We pass an array and slice
        into a function run by a set of futures, along with expected
        result; then assert that the result from TileDB matches the
        expectation.
        """

        def slice_array(a: tiledb.Array, indexer, selection, expected):
            """Helper function to slice a given tiledb.Array with an indexer
            and assert that the selection matches the expected result."""
            res = getattr(a, indexer)[selection][""]
            if indexer == "df":
                res = res.values

            assert_array_equal(res, expected)

        uri = self.path()

        data = np.random.rand(100)
        with tiledb.from_numpy(uri, data):
            pass

        futures = []
        with tiledb.open(uri) as A:
            with concurrent.futures.ThreadPoolExecutor(10) as executor:
                for indexer in ["multi_index", "df"]:  #
                    for end_idx in range(1, 100, 5):
                        sel = slice(0, end_idx)
                        expected = data[sel.start : sel.stop + 1]
                        futures.append(
                            executor.submit(slice_array, A, indexer, sel, expected)
                        )

                concurrent.futures.wait(futures)

            # Important: must get each result here or else assertion
            # failures or exceptions will disappear.
            list(map(lambda x: x.result(), futures))

    # skip, does not currently work, because we cannot force use
    # of the memory estimate
    @pytest.mark.skip
    def test_sc16301_arrow_extra_estimate_dense(self):
        """
        Test that dense query of array with var-length attribute completes
        in one try. We are currently adding an extra element to the offset
        estimate from libtiledb, in order to avoid an unnecessary pair of
        query resubmits when the offsets won't fit in the estimated buffer.
        """

        uri = self.path("test_sc16301_arrow_extra_estimate_dense")

        dim1 = tiledb.Dim(name="d1", dtype="int64", domain=(1, 3))
        att = tiledb.Attr(name="a1", dtype="<U0", var=True)

        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(dim1),
            attrs=(att,),
            sparse=False,
            allows_duplicates=False,
        )
        tiledb.Array.create(uri, schema)

        with tiledb.open(uri, "w") as A:
            A[:] = np.array(["aaa", "bb", "c"])

        with tiledb.open(uri) as A:
            tiledb.stats_enable()
            r = A[:]
            assert (
                """"Context.StorageManager.Query.Reader.loop_num": 1"""
                in tiledb.stats_dump(print_out=False)
            )
            tiledb.stats_disable()
