import pytest
import tiledb
import numpy as np

from numpy.testing import assert_array_equal


def test_schema_evolution(tmp_path):
    ctx = tiledb.default_ctx()
    se = tiledb.ArraySchemaEvolution(ctx)

    uri = str(tmp_path)

    attrs = [
        tiledb.Attr(name="a1", dtype=np.float64),
        tiledb.Attr(name="a2", dtype=np.int32),
    ]
    dims = [tiledb.Dim(domain=(0, 3), dtype=np.uint64)]
    domain = tiledb.Domain(*dims)
    schema = tiledb.ArraySchema(domain=domain, attrs=attrs, sparse=False)
    tiledb.Array.create(uri, schema)

    data1 = {
        "a1": np.arange(5, 9),
        "a2": np.random.randint(0, 1e7, size=4).astype(np.int32),
    }

    with tiledb.open(uri, "w") as A:
        A[:] = data1

    with tiledb.open(uri) as A:
        res = A[:]
        assert_array_equal(res["a1"], data1["a1"])
        assert_array_equal(res["a2"], data1["a2"])
        assert "a3" not in res.keys()

    newattr = tiledb.Attr("a3", dtype=np.int8)
    se.add_attribute(newattr)
    se.array_evolve(uri)

    data2 = {
        "a1": np.arange(5, 9),
        "a2": np.random.randint(0, 1e7, size=4).astype(np.int32),
        "a3": np.random.randint(0, 255, size=4).astype(np.int8),
    }

    with tiledb.open(uri, "w") as A:
        A[:] = data2

    with tiledb.open(uri) as A:
        res = A[:]
        assert_array_equal(res["a1"], data2["a1"])
        assert_array_equal(res["a2"], data2["a2"])
        assert_array_equal(res["a3"], data2["a3"])

    se = tiledb.ArraySchemaEvolution(ctx)
    se.drop_attribute("a1")
    se.array_evolve(uri)

    with tiledb.open(uri) as A:
        res = A[:]
        assert "a1" not in res.keys()
        assert_array_equal(res["a2"], data2["a2"])
        assert_array_equal(res["a3"], data2["a3"])
