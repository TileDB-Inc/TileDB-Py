IF TILEDBPY_MODULAR:
  include "common.pxi"
  from .libtiledb cimport *

import numpy as np

# ref
#   https://github.com/TileDB-Inc/TileDB-Py/issues/102
#   https://github.com/TileDB-Inc/TileDB-Py/issues/201

def _index_as_tuple(idx):
    """Forces scalar index objects to a tuple representation"""
    if isinstance(idx, tuple):
        return idx
    return (idx,)


cdef class DomainIndexer(object):

    @staticmethod
    def with_schema(ArraySchema schema):
        cdef DomainIndexer indexer = DomainIndexer.__new__(DomainIndexer)
        indexer.array = None
        indexer.schema = schema
        return indexer

    def __init__(self, Array array, query = None):
        self.array = array
        self.schema = array.schema
        self.query = query

    def __getitem__(self, object idx):
        # implements domain-based indexing: slice by domain coordinates, not 0-based python indexing

        cdef Domain dom = self.schema.domain
        cdef ndim = dom.ndim
        cdef list attr_names = list()

        idx = _index_as_tuple(idx)

        if len(idx) < dom.ndim:
            raise IndexError("number of indices does not match domain rank: "
                             "(got {!r}, expected: {!r})".format(len(idx), ndim))

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

        subarray = np.zeros(shape=(ndim, 2), dtype=dom.dtype)

        for i, subidx in enumerate(new_idx):
            assert isinstance(subidx, slice)
            subarray[i] = subidx.start, subidx.stop

        attr_names = list(self.schema.attr(i).name for i in range(self.schema.nattr))

        order = None
        # TODO make coords optional for array.domain_index. there are no kwargs in slicing[], so
        #      one way to do this would be to overload __call__ and return a new
        #      object with a flag set. not ideal.
        coords = True

        if self.query is not None:
            # if we are called via Query object, then we need to respect Query semantics
            order = self.query.order
            attr_names = self.query.attrs if self.query.attrs else attr_names # query.attrs might be None -> all
            coords = self.query.coords

        if coords:
            attr_names.insert(0, "coords")

        if order is None or order == 'C':
            layout = TILEDB_ROW_MAJOR
        elif order == 'F':
            layout = TILEDB_COL_MAJOR
        elif order == 'G':
            layout = TILEDB_GLOBAL_ORDER
        else:
            raise ValueError("order must be 'C' (TILEDB_ROW_MAJOR), 'F' (TILEDB_COL_MAJOR), or 'G' (TILEDB_GLOBAL_ORDER)")


        if isinstance(self.array, SparseArray):
            return (<SparseArrayImpl>self.array)._read_sparse_subarray(subarray, attr_names, layout)
        elif isinstance(self.array, DenseArray):
            return (<DenseArrayImpl>self.array)._read_dense_subarray(subarray, attr_names, layout)
        else:
            raise Exception("No handler for Array type: " + str(type(self.array)))