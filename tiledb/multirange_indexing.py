import tiledb
from tiledb import Array, ArraySchema, TileDBError
import os, numpy as np
import sys, weakref
from collections import OrderedDict

def mr_dense_result_shape(ranges, base_shape = None):
    # assumptions: len(ranges) matches number of dims
    if base_shape is not None:
        assert len(ranges) == len(base_shape), "internal error: mismatched shapes"

    new_shape = list()
    for i,rr in enumerate(ranges):
        if rr != ():
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
        elif isinstance(range, tuple):
            subranges.extend((range,))
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

    def __init__(self, array, query = None):
        if not issubclass(type(array), tiledb.Array):
            raise ValueError("Internal error: MultiRangeIndexer expected tiledb.Array")
        self.array_ref = weakref.ref(array)
        self.schema = array.schema
        self.query = query

    @property
    def array(self):
        assert self.array_ref() is not None, \
            "Internal error: invariant violation (indexing call w/ dead array_ref)"
        return self.array_ref()

    @classmethod
    def __test_init__(cls, array):
        """
        Internal helper method for testing getitem range calculation.
        :param array:
        :return:
        """
        m = cls.__new__(cls)
        m.array_ref = weakref.ref(array)
        m.schema = array.schema
        m.query = None
        return m

    def getitem_ranges(self, idx):
        dom = self.schema.domain
        ndim = dom.ndim

        if isinstance(idx, tuple):
            idx = list(idx)
        else:
            idx = [idx]

        ranges = list()
        for i,sel in enumerate(idx):
            if not isinstance(sel, list):
                sel = [sel]
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

        schema = self.schema
        dom = self.schema.domain
        attr_names = tuple(self.schema.attr(i).name for i in range(self.schema.nattr))

        coords = None
        if self.query is not None:
            # if we are called via Query object, then we need to respect Query semantics
            attr_names = tuple(self.query.attrs) if self.query.attrs else attr_names # query.attrs might be None -> all
            coords = self.query.coords

        from tiledb.core import PyQuery
        q = PyQuery(self.array._ctx_(), self.array, attr_names, coords)

        q.set_ranges(ranges)
        q.submit()

        result_dict = OrderedDict(q.results())

        for name, item in result_dict.items():
            if len(item[1]) > 0:
                arr = self.array._unpack_varlen_query(item, name)
            else:
                arr = item[0]
                arr.dtype = schema.attr_or_dim_dtype(name)
            result_dict[name] = arr

        if self.schema.sparse:
            return result_dict
        else:
            result_shape = mr_dense_result_shape(ranges, self.schema.shape)
            for arr in result_dict.values():
                # TODO check/test layout
                arr.shape = result_shape
            return result_dict
