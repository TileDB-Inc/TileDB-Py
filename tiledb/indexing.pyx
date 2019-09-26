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

    def __init__(self, SparseArray array):
        self.array = array
        self.schema = array.schema

    def __getitem__(self, object idx):
        # implements domain-based indexing: slice by domain coordinates, not 0-based python indexing

        cdef Domain dom = self.schema.domain
        cdef ndim = dom.ndim

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

        # TODO...
        order = None
        if order is None or order == 'C':
            layout = TILEDB_ROW_MAJOR
        elif order == 'F':
            layout = TILEDB_COL_MAJOR
        elif order == 'G':
            layout = TILEDB_GLOBAL_ORDER
        else:
            raise ValueError("order must be 'C' (TILEDB_ROW_MAJOR), 'F' (TILEDB_COL_MAJOR), or 'G' (TILEDB_GLOBAL_ORDER)")

        # TODO make coords optional. there are no kwargs in slicing[], so
        #      one way to do this would be to overload __call__ and return a new
        #      object with a flag set. not ideal.
        attr_names = list()
        attr_names.append("coords")
        attr_names.extend(self.schema.attr(i).name for i in range(self.schema.nattr))

        return self.array._read_sparse_subarray(subarray, attr_names, layout)
