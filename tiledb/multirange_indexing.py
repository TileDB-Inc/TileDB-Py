import tiledb
from tiledb import Array, ArraySchema
import os, numpy as np
import sys

try:
    from tiledb.libtiledb import multi_index
except:
    from tiledb.indexing import multi_index

def _index_as_tuple(idx):
    """Forces scalar index objects to a tuple representation"""
    if isinstance(idx, tuple):
        return idx
    return (idx,)

def mr_dense_result_shape(ranges, base_shape = None):
    # assumptions: len(ranges) matches number of dims
    if base_shape is not None:
        assert len(ranges) == len(base_shape), "internal error: mismatched shapes"

    new_shape = list()
    for i,rr in enumerate(ranges):
        if rr is not ():
            m = list(map(lambda y: abs(y[1] - y[0]) + 1, rr))
            new_shape.append(np.sum(m))
        else:
            if base_shape is None:
                raise ValueError("Missing required base_shape for whole-dimension slices")
            # empty range covers dimension
            new_shape.append(base_shape[i])

    return tuple(new_shape)

def mr_dense_result_numel(ranges):
    return np.prod(mr_dense_result_shape(ranges))

def sel_to_subranges(dim_sel):
    if isinstance(dim_sel, list):
        dim_sel = tuple(dim_sel)
    elif not isinstance(dim_sel, tuple):
        dim_sel = (dim_sel,)

    subranges = list()
    for range in dim_sel:
        if np.isscalar(range):
            subranges.append( (range, range) )
        elif isinstance(range, slice):
            if range.step is not None:
                raise ValueError("Stepped slice ranges are not supported")
            elif range.start is None and range.stop is None:
                # ':' full slice
                pass
            else:
                subranges.append( (range.start, range.stop) )
        elif isinstance(range, list):
            for el in range:
                subranges.append( (el, el) )
        else:
            raise TypeError("Unsupported selection ")

    return tuple(subranges)


class MultiRangeIndexer(object):
    """
    Implements multi-range / outer / orthogonal indexing.

    """
    # for cython
    # comment out for Python 2 :/
    #array: Array
    #schema: ArraySchema
    #def __init__(self, array: Array, query = None):

    def __init__(self, array, query = None):
        self.array = array
        # TODO remove
        if hasattr(array, 'schema'):
            self.schema = array.schema
        self.query = query

    def getitem_ranges(self, idx):
        dom = self.schema.domain
        ndim = dom.ndim
        idx = _index_as_tuple(idx)

        ranges = list()
        for i,sel in enumerate(idx):
            subranges = sel_to_subranges(sel)
            ranges.append(subranges)

        # extend the list to ndim
        if len(ranges) < ndim:
            ranges.extend([ tuple() for _ in range(ndim-len(ranges))])

        rval = tuple(ranges)
        return rval

    def __getitem__(self, idx):
        # implements multi-range / outer / orthogonal indexing
        ranges = self.getitem_ranges(idx)

        dom = self.schema.domain
        attr_names = tuple(self.schema.attr(i).name for i in range(self.schema.nattr))

        # TODO order
        result_dict = multi_index(
            self.array,
            attr_names,
            ranges
        )

        if self.schema.sparse:
            return result_dict
        else:
            result_shape = mr_dense_result_shape(ranges, self.schema.shape)
            for arr in result_dict.values():
                # TODO check/test layout
                arr.shape = result_shape
            return result_dict