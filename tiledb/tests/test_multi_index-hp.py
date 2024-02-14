#
# Property-based tests for Array.multi_index using Hypothesis
#

import warnings

import hypothesis as hp
import numpy as np
import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from numpy.testing import assert_array_equal

import tiledb
from tiledb import SparseArray

from .strategies import bounded_ntuple, ranged_slices


def is_boundserror(exc: Exception):
    assert str(exc) != ""

    vals = [
        "out of domain bounds",
        "Cannot add range to dimension",
        "cannot be larger than the higher bound",
    ]

    return any(x in str(exc) for x in vals)


def _direct_query_ranges(array: SparseArray, ranges, order):
    order_map = {"C": 0, "F": 1, "U": 3}
    layout = order_map[order]
    with tiledb.scope_ctx() as ctx:
        q = tiledb.main.PyQuery(ctx, array, ("a",), (), layout, False)
        subarray = tiledb.Subarray(array)
        subarray.add_ranges(ranges)
        q.set_subarray(subarray)

        q.submit()
    return {k: v[0].view(array.attr(0).dtype) for k, v in q.results().items()}


# Compound strategies to build valid inputs for multi_index
subindex_obj = st.one_of(st.integers(), ranged_slices())

index_obj = st.one_of(subindex_obj, st.tuples(st.lists(subindex_obj)))


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
                    tiledb.Attr(name="a", dtype="float64", var=False, nullable=False)
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

    @given(
        order=st.sampled_from(["C", "F", "U"]),
        ranges=st.lists(bounded_ntuple(length=2, min_value=-100, max_value=100)),
    )
    @hp.settings(deadline=None)
    def test_multi_index_two_way_query(self, order, ranges, sparse_array_1d):
        """This test checks the result of "direct" range queries using PyQuery
        against the result of `multi_index` on the same ranges."""

        uri = sparse_array_1d

        assert isinstance(uri, str)
        assume(v[0] <= v[1] for v in ranges)

        try:
            with tiledb.open(uri) as A:
                r1 = A.query(order=order).multi_index[ranges]["a"]
                r2 = _direct_query_ranges(A, [ranges], order)["a"]

                assert_array_equal(r1, r2)
        except tiledb.TileDBError as exc:
            if is_boundserror(exc):
                # out of bounds, this is ok so we tell hypothesis to ignore
                # TODO these should all be IndexError
                assume(False)
            raise

    @given(index_obj)
    @hp.settings(deadline=None)
    def test_multi_index_inputs(self, sparse_array_1d, ind):
        # TODO
        # currently we don't have a comparison target/mockup to check
        # as there is no direct numpy equivalent for this indexing mode
        # but we could still assert more details about the result
        # - coordinates are inbounds
        # - values are within known attribute range from write
        # another option for indirect testing
        # - densify slices and ranges and compare to numpy
        #   numpy vectorized indexing result

        uri = sparse_array_1d

        try:
            with tiledb.open(uri) as A:
                r1 = A.multi_index[ind]
                r1_array = r1["a"]
                r1_coords = r1["__dim_0"]

                assert isinstance(r1_array, np.ndarray)
                assert isinstance(r1_coords, np.ndarray)

                # some results may be empty
                if len(r1_array):
                    # assertions based on input data
                    assert r1_array.min() >= self.dmin
                    assert r1_array.max() <= self.dmax
                    assert r1_coords.min() >= self.dmin
                    assert r1_coords.max() <= self.dmax
        except tiledb.TileDBError as exc:
            # bounds errors are not failures
            if is_boundserror(exc):
                assume(False)
            elif "Failed to cast dim range" in str(exc):
                # TODO this should be IndexError
                assume(False)
            else:
                raise
        except ValueError as exc:
            if "Stepped slice ranges are not supported" in str(exc):
                # stepped slice errors are ok
                assume(False)
            elif "Cannot convert <class 'NoneType'> to scalar" in str(exc):
                assume(False)
            else:
                raise
        except TypeError as exc:
            if "Unsupported selection" in str(exc):
                # mostly ok but warn for cross-check
                warnings.warn(str(exc))
                assume(False)
            else:
                raise
