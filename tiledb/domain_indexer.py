import weakref

import numpy as np

import tiledb
import tiledb.libtiledb as lt


def _index_as_tuple(idx):
    """Forces scalar index objects to a tuple representation"""
    if isinstance(idx, tuple):
        return idx
    return (idx,)


class DomainIndexer:
    @staticmethod
    def with_schema(schema):
        indexer = DomainIndexer.__new__(DomainIndexer)
        indexer.array = None
        indexer.schema = schema
        return indexer

    def __init__(self, array, query=None):
        self.array_ref = weakref.ref(array)
        self.schema = array.schema
        self.query = query

    # @property
    def schema(self):
        return self.array.array_ref().schema

    @property
    def array(self):
        assert (
            self.array_ref() is not None
        ), "Internal error: invariant violation (index[] with dead array_ref)"
        return self.array_ref()

    def __getitem__(self, idx):
        from .subarray import Subarray  # prevent circular import

        # implements domain-based indexing: slice by domain coordinates, not 0-based python indexing

        schema = self.array.schema
        dom = schema.domain
        ndim = dom.ndim
        attr_names = list()

        idx = _index_as_tuple(idx)

        if len(idx) < dom.ndim:
            raise IndexError(
                "number of indices does not match domain rank: "
                "(got {!r}, expected: {!r})".format(len(idx), ndim)
            )

        new_idx = []
        for i in range(dom.ndim):
            dim = dom.dim(i)
            dim_idx = idx[i]
            if np.isscalar(dim_idx):
                start = dim_idx
                stop = dim_idx
                new_idx.append(slice(start, stop, None))
            else:
                new_idx.append(dim_idx)

        dim_ranges = list()

        for i, subidx in enumerate(new_idx):
            assert isinstance(subidx, slice)
            dim_ranges.append((subidx.start, subidx.stop))
        subarray = Subarray(self.array)
        subarray.add_ranges([list([x]) for x in dim_ranges])

        attr_names = list(schema.attr(i).name for i in range(schema.nattr))
        attr_cond = None

        order = None
        # TODO make coords optional for array.domain_index. there are no kwargs in slicing[], so
        #      one way to do this would be to overload __call__ and return a new
        #      object with a flag set. not ideal.
        coords = True

        if self.query is not None:
            # if we are called via Query object, then we need to respect Query semantics
            order = self.query.order
            attr_names = (
                self.query.attrs if self.query.attrs else attr_names
            )  # query.attrs might be None -> all
            attr_cond = self.query.attr_cond
            coords = self.query.has_coords

        if coords:
            attr_names = [
                dom.dim(idx).name for idx in range(self.schema.ndim)
            ] + attr_names

        if order is None or order == "C":
            layout = lt.LayoutType.ROW_MAJOR
        elif order == "F":
            layout = lt.LayoutType.COL_MAJOR
        elif order == "G":
            layout = lt.LayoutType.GLOBAL_ORDER
        elif order == "U":
            layout = lt.LayoutType.UNORDERED
        else:
            raise ValueError(
                "order must be 'C' (TILEDB_ROW_MAJOR), 'F' (TILEDB_COL_MAJOR), or 'G' (TILEDB_GLOBAL_ORDER)"
            )

        if isinstance(self.array, tiledb.sparse_array.SparseArrayImpl):
            return self.array._read_sparse_subarray(
                subarray, attr_names, attr_cond, layout
            )
        elif isinstance(self.array, tiledb.dense_array.DenseArrayImpl):
            return self.array._read_dense_subarray(
                subarray, attr_names, attr_cond, layout, coords
            )
        else:
            raise Exception("No handler for Array type: " + str(type(self.array)))
