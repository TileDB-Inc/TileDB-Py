import numpy as np
import tiledb

from tiledb.tests.common import DiskTestCase


class FixesTest(DiskTestCase):
    def test_ch7727_float32_dim_estimate_incorrect(self):
        # set max allocation: because windows won't overallocate
        with tiledb.scope_ctx({"py.alloc_max_bytes": 1024 ** 2 * 100}):
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
        max = 1024 ** 2 + 1
        with tiledb.from_numpy(uri, np.uint8(range(max))):
            pass
        with tiledb.scope_ctx(
            {"py.init_buffer_bytes": 2 * 1024 ** 2, "py.alloc_max_bytes": 1024 ** 2}
        ) as ctx3:
            with tiledb.open(uri) as b:
                q = tiledb.main.PyQuery(ctx3, b, ("",), (), 0, False)
                q._return_incomplete = True
                q.set_ranges([[(0, max)]])
                q._allocate_buffers()
                buffers = q._get_buffers()
                assert buffers[0].nbytes == max
