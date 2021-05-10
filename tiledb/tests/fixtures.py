import pytest

import tiledb
import numpy as np

from tiledb.tests.common import assert_subarrays_equal, rand_utf8

INTEGER_DTYPES = ["u1", "u2", "u4", "u8", "i1", "i2", "i4", "i8"]


@pytest.fixture(scope="module", params=["hilbert", "row-major"])
def sparse_cell_order(request):
    yield request.param


@pytest.fixture(scope="class")
def test_incomplete_return_array(tmpdir_factory):
    tmp_path = str(tmpdir_factory.mktemp("array"))
    ncells = 20
    nvals = 10

    data = np.array([rand_utf8(nvals - i % 2) for i in range(ncells)], dtype="O")

    dom = tiledb.Domain(tiledb.Dim(domain=(0, len(data) - 1), tile=len(data)))
    att = tiledb.Attr(dtype=str, var=True)

    schema = tiledb.ArraySchema(dom, (att,), sparse=True)

    coords = np.arange(ncells)

    tiledb.SparseArray.create(tmp_path, schema)
    with tiledb.SparseArray(tmp_path, mode="w") as T:
        T[coords] = data

    with tiledb.SparseArray(tmp_path, mode="r") as T:
        assert_subarrays_equal(data, T[:][""])

    return tmp_path
