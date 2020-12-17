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
            # modular arithmetic gives misleading overflow warning.
            with np.errstate(over="ignore"):
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

def sel_to_subranges(dim_sel, nonempty_domain=None):
    subranges = list()
    for idx,range in enumerate(dim_sel):
        if np.isscalar(range):
            subranges.append( (range, range) )
        elif isinstance(range, slice):
            if range.step is not None:
                raise ValueError("Stepped slice ranges are not supported")
            elif range.start and range.stop:
                # we have both endpoints, use them
                rstart = range.start
                rend = range.stop
            else:
                # we are missing one or both endpoints, maybe use nonempty_domain
                if nonempty_domain is None:
                    raise TileDBError("Open-ended slicing requires a valid nonempty_domain")
                rstart = range.start if range.start else nonempty_domain[0]
                rend = range.stop if range.stop else nonempty_domain[1]

            subranges.append( (rstart,rend) )
        elif isinstance(range, tuple):
            subranges.extend((range,))
        elif isinstance(range, list):
            for el in range:
                subranges.append( (el, el) )
        else:
            raise TypeError("Unsupported selection ")
    return tuple(subranges)


try:
    import pyarrow
    _have_pyarrow = True
except ImportError:
    _have_pyarrow = False

class MultiRangeIndexer(object):
    """
    Implements multi-range indexing.
    """
    debug=False

    def __init__(self, array, query = None, use_arrow = None):
        if not issubclass(type(array), tiledb.Array):
            raise ValueError("Internal error: MultiRangeIndexer expected tiledb.Array")
        self.array_ref = weakref.ref(array)
        self.schema = array.schema
        self.query = query

        use_arrow_real = isinstance(self, DataFrameIndexer) and _have_pyarrow
        if use_arrow != None:
            use_arrow_real = use_arrow_real and use_arrow

        self.use_arrow = use_arrow_real

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
        ned = self.array.nonempty_domain()

        if isinstance(idx, tuple):
            idx = list(idx)
        else:
            idx = [idx]

        ranges = list()
        for i,sel in enumerate(idx):
            if not isinstance(sel, list):
                sel = [sel]
            # don't try to index nonempty_domain if None
            ned_arg = ned[i] if ned else None
            subranges = sel_to_subranges(sel, ned_arg)

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
        if schema.sparse:
            dim_names = tuple(dom.dim(i).name for i in range(dom.ndim))
        else:
            dim_names = tuple()

        # set default order
        # - TILEDB_UNORDERED for sparse
        # - TILEDB_ROW_MAJOR for dense
        if self.schema.sparse:
            order = 'U'
        else:
            order = 'C'

        # if this indexing operation is part of a query (A.query().df)
        # then we need to respect the settings of the query
        if self.query is not None:
            # if we are called via Query object, then we need to respect Query semantics
            if self.query.attrs is not None:
                attr_names = tuple(self.query.attrs)
            else:
                pass # query.attrs might be None -> all

            if self.query.dims is False:
                dim_names = tuple()
            elif self.query.dims is not None:
                dim_names = tuple(self.query.dims)
            elif self.query.coords is False:
                dim_names = tuple()

            # set query order
            order = self.query.order

        # convert order to layout
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
                             "'F' (TILEDB_COL_MAJOR), "
                             "'U' (TILEDB_UNORDERED),"
                             "or 'G' (TILEDB_GLOBAL_ORDER)")

        # initialize the pybind11 query object
        from tiledb.core import PyQuery
        q = PyQuery(self.array._ctx_(),
                    self.array,
                    attr_names,
                    dim_names,
                    layout,
                    self.use_arrow)

        q.set_ranges(ranges)
        q.submit()

        if self.query is not None and self.query.return_arrow:
            return q._buffers_to_pa_table()

        if isinstance(self, DataFrameIndexer) and self.use_arrow:
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
        from .dataframe_ import _tiledb_result_as_dataframe, check_dataframe_deps
        check_dataframe_deps()

        idx_start = time.time()

        # we need to use a Query in order to get coords for a dense array
        if not self.query:
            self.query = tiledb.libtiledb.Query(self.array, coords=True)

        result = super(DataFrameIndexer, self).__getitem__(idx)

        if self.use_arrow:
            import pyarrow as pa
            if use_stats():
                pd_start = time.time()

            if isinstance(result, pa.Table):
                # support the `query(return_arrow=True)` mode and return Table untouched
                df = result
            else:
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
        if not _have_pyarrow:
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

                # this is an auto-created column and should be unnamed
                if col_name == '__tiledb_rows':
                    rename_cols['__tiledb_rows'] = None
                    indexes.append(None)
                else:
                    indexes.append(col_name)

        if len(rename_cols) > 0:
            res_df.rename(columns=rename_cols, inplace=True)

        if  self.query is not None:
            # if we have a query with index_col set, then override any
            # index information saved with the array.
            if self.query.index_col is not True and self.query.index_col is not None:
                res_df.set_index(self.query.index_col, inplace=True)
            elif self.query.index_col is True and len(indexes) > 0:
                # still need to set indexes here b/c df creates query every time
                res_df.set_index(indexes, inplace=True)
            else:
                # don't convert any column to a dataframe index
                pass
        elif len(indexes) > 0:
            res_df.set_index(indexes, inplace=True)

        if use_stats():
            pd_idx_duration = time.time() - pd_idx_start
            tiledb.core.increment_stat("py.pandas_index_update_time", pd_idx_duration)

        return res_df
