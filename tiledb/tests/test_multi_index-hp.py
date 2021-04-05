#
# Property-based tests for Array.multi_index using Hypothesis
#

import tiledb
from tiledb import SparseArray
import numpy as np
from numpy.testing import assert_array_equal

from pathlib import Path
import warnings

import pytest
from tiledb.tests.common import checked_path
from tiledb.tests.common import bounded_ntuple, ranged_slices

from hypothesis import given, assume, settings
from hypothesis import strategies as st


def is_boundserror(exc: Exception):
    assert str(exc) != ""

    vals = ["out of domain bounds", "Cannot add range to dimension"]

    return any(x in str(exc) for x in vals)


def _direct_query_ranges(array: SparseArray, ranges):
    ctx = tiledb.default_ctx()
    q = tiledb.core.PyQuery(ctx, array, ("a",), (), 0, False)
    q.set_ranges(ranges)
    q.submit()

    res = {k: v[0].view(array.attr(0).dtype) for k, v in q.results().items()}
    return res


class TestMultiIndexPropertySparse:
    dmin, dmax = -100, 100

    @classmethod
    @pytest.fixture(scope="class")
    def sparse_array_1d(cls, checked_path):
        def write_sparse_contig(uri):
            data = np.arange(cls.dmin, cls.dmax, dtype=np.int64)
            with tiledb.open(uri, "w") as A:
                A[data] = data

        def create_array(uri):
            schema = tiledb.ArraySchema(
                tiledb.Domain(
                    [tiledb.Dim(dtype=np.int64, domain=(cls.dmin, cls.dmax))]
                ),
                attrs=[
                    tiledb.Attr(name="a", dtype="float64", var=False, nullable=False),
                ],
                cell_order="row-major",
                tile_order="row-major",
                capacity=10000,
                sparse=True,
            )

            tiledb.Array.create(uri, schema)

        uri = checked_path.path()

        create_array(uri)
        write_sparse_contig(uri)

        return uri

    @given(st.lists(bounded_ntuple(length=2, min_value=-100, max_value=100)))
    def test_multi_index_two_way_query(self, sparse_array_1d, ranges):
        """This test checks the result of "direct" range queries using PyQuery
        against the result of `multi_index` on the same ranges."""

        uri = sparse_array_1d

        assert isinstance(uri, str)
        assume(v[0] <= v[1] for v in ranges)

        try:
            with tiledb.open(uri) as A:
                r1 = A.multi_index[ranges]["a"]
                r2 = _direct_query_ranges(A, [ranges])["a"]

                assert_array_equal(r1, r2)
        except tiledb.TileDBError as exc:
            exc_str = str(exc)
            if is_boundserror(exc_str):
                # out of bounds, this is ok so we tell hypothesis to ignore
                # TODO these should all be IndexError
                assume(False)
            raise
