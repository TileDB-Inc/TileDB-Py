from contextlib import AbstractContextManager
import itertools

import numpy as np
from numpy.testing import assert_array_equal
import pytest

import tiledb
from tiledb.main import tiledb_serialization_type_t as ser_type
from tiledb.main import serialization as ser

from tiledb.tests.common import DiskTestCase
from tiledb.main import test_serialization as ser_test


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
            assert_array_equal(A.query()[3:8][""], A.set_query(ser_qry)[""])
