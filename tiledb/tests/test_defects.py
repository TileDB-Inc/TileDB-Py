import numpy as np
import tiledb

from tiledb.tests.common import DiskTestCase


class DefectTest(DiskTestCase):
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
