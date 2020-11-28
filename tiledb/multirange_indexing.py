import tiledb
from tiledb import Array, ArraySchema, TileDBError
from tiledb.core import increment_stat, use_stats

import os, numpy as np
import sys, time, weakref, warnings
import json
from collections import OrderedDict

def mr_dense_result_shape(ranges, base_shape = None):
    # assumptions: len(ranges) matches number of dims
    if base_shape is not None:
        assert len(ranges) == len(base_shape), "internal error: mismatched shapes"

    new_shape = list()
    for i,rr in enumerate(ranges):
        if rr != ():
            m = list(map(lambda y: abs(np.uint64(y[1]) - np.uint64(y[0])) + np.uint64(1), rr))
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


try:
    import pyarrow as pa
    _have_pa = True
except ImportError:
    _have_pa = False

class MultiRangeIndexer(object):
    """
    Implements multi-range indexing.
    """
    debug=False

    def __init__(self, array, query = None, use_pa = None):
        if not issubclass(type(array), tiledb.Array):
            raise ValueError("Internal error: MultiRangeIndexer expected tiledb.Array")
        self.array_ref = weakref.ref(array)
        self.schema = array.schema
        self.query = query
        _use_pa = use_pa
        if _use_pa == None:
            # default on for .df
            _use_pa = False
        if use_pa and not _have_pa:
            warnings.warn("'use_arrow' mode requires installation of pyarrow package")

        self.use_pa = _have_pa and _use_pa

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
        attr_names = tuple(self.schema.attr(i)._internal_name for i in range(self.schema.nattr))
        coords = None
        order = 'C' # TILEDB_ROW_MAJOR
        if self.query is not None:
            # if we are called via Query object, then we need to respect Query semantics
            attr_names = tuple(self.query.attrs) if self.query.attrs else attr_names # query.attrs might be None -> all
            coords = self.query.coords
            order = self.query.order

        if order is None or order == 'C':
            layout = 0
        elif order == 'F':
            layout = 1
        elif order == 'G':
            layout = 2
        elif order == 'U':
            layout = 3
        else:
            raise ValueError("order must be 'C' (TILEDB_ROW_MAJOR), "\
                             "'F' (TILEDB_COL_MAJOR), "\
                             "or 'G' (TILEDB_GLOBAL_ORDER)")

        from tiledb.core import PyQuery
        q = PyQuery(self.array._ctx_(),
                    self.array,
                    attr_names,
                    coords,
                    layout,
                    self.use_pa)

        q.set_ranges(ranges)
        q.submit()

        if isinstance(self, DataFrameIndexer) and self.use_pa:
            return q

        result_dict = OrderedDict(q.results())

        final_names = dict()
        for name, item in result_dict.items():
            if len(item[1]) > 0:
                arr = q.unpack_buffer(name, item[0], item[1])
            else:
                arr = item[0]
                final_dtype = schema.attr_or_dim_dtype(name)
                if (len(arr) < 1 and
                        (np.issubdtype(final_dtype, np.bytes_) or
                         np.issubdtype(final_dtype, np.unicode_))):
                    # special handling to get correctly-typed empty array
                    # (expression below changes itemsize from 0 to 1)
                    arr.dtype = final_dtype.str + '1'
                else:
                    arr.dtype = schema.attr_or_dim_dtype(name)
            if name == '__attr':
                final_names[name] = ''
            result_dict[name] = arr

        for name, replacement in final_names.items():
            result_dict[replacement] = result_dict.pop(name)

        if self.schema.sparse:
            return result_dict
        else:
            result_shape = mr_dense_result_shape(ranges, self.schema.shape)
            for arr in result_dict.values():
                # TODO check/test layout
                arr.shape = result_shape
            return result_dict

class DataFrameIndexer(MultiRangeIndexer):
    """
    Implements `.df[]` indexing to directly return a dataframe
    [] operator uses multi_index semantics.
    """
    def __getitem__(self, idx):
        from .dataframe_ import _tiledb_result_as_dataframe

        idx_start = time.time()

        # we need to use a Query in order to get coords for a dense array
        if not self.query:
            self.query = tiledb.libtiledb.Query(self.array, coords=True)

        result = super(DataFrameIndexer, self).__getitem__(idx)

        # return pre-converted result directly
        if self.use_pa:
            if use_stats():
                pd_start = time.time()

            df = self.pa_to_pandas(result)

            if use_stats():
                pd_duration = time.time() - pd_start
                increment_stat("py.buffer_conversion_time", pd_duration)
                idx_duration = time.time() - idx_start
                increment_stat("py.__getitem__time", idx_duration)
            return df
        else:
            if not isinstance(result, OrderedDict):
                raise ValueError("Expected OrderedDict result, got '{}'".format(type(result)))

            if use_stats():
                pd_start = time.time()

            df = _tiledb_result_as_dataframe(self.array, result)

            if use_stats():
                pd_duration = time.time() - pd_start
                idx_duration = time.time() - idx_start
                tiledb.core.increment_stat("py.buffer_conversion_time", pd_duration)
                tiledb.core.increment_stat("py.__getitem__time", idx_duration)

            return df

    def pa_to_pandas(self, pyquery):
        if not _have_pa:
            raise TileDBError("Cannot convert to pandas via this path without pyarrow; please disable Arrow results")
        try:
            table = pyquery._buffers_to_pa_table()
        except Exception as exc:
            if MultiRangeIndexer.debug:
                print("Exception during pa.Table conversion, returning pyquery: '{}'".format(exc))
                return pyquery
            raise
        try:
            res_df = table.to_pandas()
        except Exception as exc:
            if MultiRangeIndexer.debug:
                print("Exception during Pandas conversion, returning (table,query): '{}'".format(exc))
                return table,pyquery
            raise

        if use_stats():
            pd_idx_start = time.time()

        # x-ref write path in dataframe_.py
        index_dims = None
        if '__pandas_index_dims' in self.array.meta:
            index_dims = json.loads(self.array.meta['__pandas_index_dims'])

        indexes = list()
        rename_cols = dict()
        for col_name in res_df.columns.values:
            if index_dims and col_name in index_dims:
                if col_name == '__tiledb_rows':
                    rename_cols['__tiledb_rows'] = None
                    indexes.append(None)
                else:
                    indexes.append(col_name)

        if len(rename_cols) > 0:
            res_df.rename(columns=rename_cols, inplace=True)
        if len(indexes) > 0:
            res_df.set_index(indexes, inplace=True)

        if use_stats():
            pd_idx_duration = time.time() - pd_idx_start
            tiledb.core.increment_stat("py.pandas_index_update_time", pd_idx_duration)

        return res_df
