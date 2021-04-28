import pytest
import random

import tiledb
import numpy as np

from numpy.testing import assert_array_equal
from tiledb.tests.common import assert_subarrays_equal, rand_utf8


@pytest.fixture(scope="module", params=["hilbert", "row-major"])
def sparse_cell_order(request):
    yield request.param


@pytest.fixture(scope="class")
def test_incomplete_return_array(tmpdir_factory):
    tmp_path = str(tmpdir_factory.mktemp("array"))
    ncells = 20
    nvals = 10

    data = np.array([rand_utf8(nvals - i % 2) for i in range(ncells)], dtype="O")

    ctx = tiledb.default_ctx()

    dom = tiledb.Domain(
        tiledb.Dim(domain=(0, len(data) - 1), tile=len(data), ctx=ctx), ctx=ctx
    )
    att = tiledb.Attr(dtype=str, var=True, ctx=ctx)

    schema = tiledb.ArraySchema(dom, (att,), sparse=True, ctx=ctx)

    coords = np.arange(ncells)

    tiledb.SparseArray.create(tmp_path, schema)
    with tiledb.SparseArray(tmp_path, mode="w", ctx=ctx) as T:
        T[coords] = data

    with tiledb.SparseArray(tmp_path, mode="r", ctx=ctx) as T:
        assert_subarrays_equal(data, T[:][""])

    return tmp_path
