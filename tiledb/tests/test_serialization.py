import numpy as np
import pytest

import tiledb

from .common import DiskTestCase

ser_test = pytest.importorskip(
    "tiledb.main.test_serialization", reason="Serialization not enabled."
)


class SerializationTest(DiskTestCase):
    def test_query_deserialization(self):
        path = self.path("test_query_deserialization")
        dom = tiledb.Domain(tiledb.Dim(domain=(1, 10), dtype=np.uint32))
        attrs = [tiledb.Attr(dtype=np.int64)]
        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=True)
        tiledb.Array.create(path, schema)

        data = np.random.randint(-5, 5, 10)

        with tiledb.open(path, "w") as A:
            A[np.arange(1, 11)] = data

        with tiledb.open(path, "r") as A:
            ctx = tiledb.default_ctx()
            ser_qry = ser_test.create_serialized_test_query(ctx, A)
            np.testing.assert_array_equal(A.query()[3:8][""], A.set_query(ser_qry)[""])
