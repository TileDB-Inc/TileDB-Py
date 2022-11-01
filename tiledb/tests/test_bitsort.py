#%%
import numpy as np
import pytest
import tempfile

import hypothesis as hp
import hypothesis.strategies as st
from hypothesis import given, reproduce_failure
from numpy.testing import assert_array_equal

import tiledb, numpy as np

"""
Testing plan

- check array schema filter compat
- round-trip over range of capacity
- round-trip with incomplete queries
- round-trip with memory limit
"""


@pytest.mark.xfail(
    reason="TODO BitSortFilter ArraySchema construction should only work with supported types"
)
@pytest.mark.parametrize(
    "test_info",
    [
        (np.int64, "ascii", None),  # ascii dim should not be supported
        (np.int64, np.float64, True),  # var-size attr should not be supported
    ],
)
def test_bitsort_init(test_info):
    attr_dtype, dim_dtype, var = test_info
    with pytest.raises(tiledb.TileDBError):
        tiledb.ArraySchema(
            domain=tiledb.Domain(
                [
                    tiledb.Dim(
                        "id", domain=(0, np.iinfo(np.uint64).max - 10), dtype=dim_dtype
                    ),
                ]
            ),
            attrs=[
                tiledb.Attr(
                    "p", dtype=attr_dtype, var=var, filters=[tiledb.BitSortFilter()]
                )
            ],
            sparse=True,
        )


#%%
def is_integer_dtype(dtype):
    return issubclass(dtype.type, (np.integer,))


def is_floating_dtype(dtype):
    return issubclass(dtype.type, (np.floating,))


def make_data(dtype, size):
    if is_floating_dtype(dtype):
        return np.random.default_rng().random(size=size, dtype=dtype)
    elif is_integer_dtype(dtype):
        return np.random.default_rng().integers(
            np.iinfo(dtype).min, high=np.iinfo(dtype).max, size=size, dtype=dtype
        )
    elif dtype == "ascii":
        raise NotImplementedError()
    elif dtype == bytes:
        raise NotImplementedError()
    elif dtype == str:
        raise NotImplementedError()
    else:
        raise NotImplementedError()


def make_sorted_data(dtype, size):
    return np.sort(make_data(dtype, size))


def make_domain(dtype, tile=10):
    if dtype in (bytes, str):
        return (None, None)
    elif is_integer_dtype(dtype):
        range = np.iinfo(dtype)
        return (range.min + tile, range.max - tile)
    elif is_floating_dtype(dtype):
        range = np.finfo(np.float32)
        return (range.min, range.max)
    else:
        raise NotImplementedError(dtype)


@pytest.mark.parametrize(
    # ncells, tiling, capacity
    "bounds",
    [
        (100, (10,), 2),
        (10000, (100,), 5),
        (100000, (100,), 999),
        (100000, (1000,), 9999),
    ],
)
@pytest.mark.parametrize(
    "attr_filters",
    [
        [tiledb.BitSortFilter()],
        [tiledb.BitSortFilter(), tiledb.ZstdFilter()],
        [tiledb.BitSortFilter(), tiledb.XORFilter(), tiledb.ZstdFilter(7)],
        [tiledb.BitSortFilter(), tiledb.XORFilter(), tiledb.Bzip2Filter()],
    ],
)
def test_bitsort_roundtrip(bounds, attr_filters):
    ncells, tiling, capacity = bounds

    gen = lambda dtype: make_data(dtype, ncells)
    coords = [gen(np.dtype("float32")), gen(np.dtype("float32"))]
    input_data = [gen(np.dtype("float32")), gen(np.dtype("int64"))]
    filters = {"attrs": attr_filters, "dims": None}

    data = {}

    dims = []
    for i, c in enumerate(coords):
        name = f"d{i}"
        dim_domain = make_domain(c.dtype)
        dims.append(
            tiledb.Dim(name=name, tile=tiling, domain=dim_domain, dtype=c.dtype)
        )

        data[name] = c

    domain = tiledb.Domain(*dims)

    attrs = []
    for i, a in enumerate(input_data):
        name = f"a{i}"
        filters = filters["attrs"] if i == 0 else None
        attrs.append(tiledb.Attr(name=name, dtype=a.dtype, filters=filters))
        data[name] = a

    schema = tiledb.ArraySchema(
        domain=domain, attrs=attrs, sparse=True, capacity=capacity
    )

    uri = tempfile.mkdtemp()
    tiledb.Array.create(uri, schema)

    # print("uri: ", uri)
    # schema.dump()

    with tiledb.open(uri, "w") as B:
        attr_data = {
            name: data[name] for name in data.keys() if not name.startswith("d")
        }
        B[tuple(coords)] = attr_data

    with tiledb.open(uri) as B:
        res = B[:]
        for k in data.keys():
            assert_array_equal(np.sort(data[k]), np.sort(res[k]))


# test_bitsort_roundtrip()
# %%
