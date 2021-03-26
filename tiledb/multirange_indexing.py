import json
import time
import weakref
from collections import OrderedDict

import numpy as np

from tiledb import Array, TileDBError
from tiledb.core import PyQuery, increment_stat, use_stats
from tiledb.libtiledb import Query

from .dataframe_ import check_dataframe_deps

try:
    import pyarrow
except ImportError:
    pyarrow = None


def mr_dense_result_shape(ranges, base_shape=None):
    # assumptions: len(ranges) matches number of dims
    if base_shape is not None:
        assert len(ranges) == len(base_shape), "internal error: mismatched shapes"

    new_shape = list()
    for i, rr in enumerate(ranges):
        if rr != ():
            # modular arithmetic gives misleading overflow warning.
            with np.errstate(over="ignore"):
                m = list(
                    map(
                        lambda y: abs(np.uint64(y[1]) - np.uint64(y[0])) + np.uint64(1),
                        rr,
                    )
                )

            new_shape.append(np.sum(m))
        else:
            if base_shape is None:
                raise ValueError(
                    "Missing required base_shape for whole-dimension slices"
                )
            # empty range covers dimension
            new_shape.append(base_shape[i])

    return tuple(new_shape)


def sel_to_subranges(dim_sel, nonempty_domain=None):
    subranges = list()
    for idx, range in enumerate(dim_sel):
        if np.isscalar(range):
            subranges.append((range, range))
        elif isinstance(range, slice):
            if range.step is not None:
                raise ValueError("Stepped slice ranges are not supported")
            elif range.start is not None and range.stop is not None:
                # we have both endpoints, use them
                rstart = range.start
                rend = range.stop
            else:
                # we are missing one or both endpoints, maybe use nonempty_domain
                if nonempty_domain is None:
                    raise TileDBError(
                        "Open-ended slicing requires a valid nonempty_domain"
                    )
                rstart = nonempty_domain[0] if range.start is None else range.start
                rend = nonempty_domain[1] if range.stop is None else range.stop

            subranges.append((rstart, rend))
        elif isinstance(range, tuple):
            subranges.extend((range,))
        elif isinstance(range, list):
            for el in range:
                subranges.append((el, el))
        else:
            raise TypeError("Unsupported selection ")
    return tuple(subranges)


def getitem_ranges(array, idx):
    ranges = [()] * array.schema.domain.ndim
    ned = array.nonempty_domain()
    for i, sel in enumerate([idx] if not isinstance(idx, tuple) else idx):
        if not isinstance(sel, list):
            sel = [sel]
        # don't try to index nonempty_domain if None
        ranges[i] = sel_to_subranges(sel, nonempty_domain=ned[i] if ned else None)
    return tuple(ranges)


class MultiRangeIndexer(object):
    """
    Implements multi-range indexing.
    """

    def __init__(self, array, query=None):
        if not isinstance(array, Array):
            raise TypeError("Internal error: MultiRangeIndexer expected tiledb.Array")
        self.array_ref = weakref.ref(array)
        self.query = query

    @property
    def array(self):
        array = self.array_ref()
        if array is None:
            raise RuntimeError(
                "Internal error: invariant violation (indexing call w/ dead array_ref)"
            )
        return array

    def __getitem__(self, idx):
        return _run_query(idx, self.array, self.query)


class DataFrameIndexer(MultiRangeIndexer):
    """
    Implements `.df[]` indexing to directly return a dataframe
    [] operator uses multi_index semantics.
    """

    def __init__(self, array, query=None, use_arrow=None):
        super().__init__(array, query)
        if use_arrow is None:
            use_arrow = True
        self.use_arrow = bool(use_arrow and pyarrow is not None)

    def __getitem__(self, idx):
        check_dataframe_deps()

        idx_start = time.time()
        array = self.array

        # we need to use a Query in order to get coords for a dense array
        query = self.query or Query(array, coords=True)
        result = _run_query(
            idx, array, query, use_arrow=self.use_arrow, preload_metadata=True
        )

        pd_start = time.time()

        if isinstance(result, dict):
            df = _tiledb_result_as_dataframe(result, array.meta)
        elif isinstance(result, PyQuery):
            df = _pyarrow_to_pandas(array, result, query)
        elif isinstance(result, pyarrow.Table):
            # support the `query(return_arrow=True)` mode and return Table untouched
            df = result
        else:
            raise TypeError(f"Unhandled result type {type(result)}")

        if use_stats():
            end = time.time()
            increment_stat("py.buffer_conversion_time", end - pd_start)
            increment_stat("py.__getitem__time", end - idx_start)

        return df


def _run_query(idx, array, query, *, use_arrow=False, preload_metadata=False):
    pyquery = _get_pyquery(array, query, use_arrow)
    pyquery._preload_metadata = preload_metadata
    ranges = getitem_ranges(array, idx)
    pyquery.set_ranges(ranges)
    pyquery.submit()

    if query is not None and query.return_arrow:
        return pyquery._buffers_to_pa_table()

    if use_arrow:
        return pyquery

    schema = array.schema
    result_dict = _get_pyquery_results(pyquery, schema)
    if not schema.sparse:
        result_shape = mr_dense_result_shape(ranges, schema.shape)
        for arr in result_dict.values():
            # TODO check/test layout
            arr.shape = result_shape
    return result_dict


def _get_pyquery(array, query, use_arrow):
    schema = array.schema
    attr_names = tuple(schema.attr(i)._internal_name for i in range(schema.nattr))
    if schema.sparse:
        dom = schema.domain
        dim_names = tuple(dom.dim(i).name for i in range(dom.ndim))
    else:
        dim_names = ()

    # if this indexing operation is part of a query (A.query().df)
    # then we need to respect the settings of the query
    if query is not None:
        if query.attrs is not None:
            attr_names = tuple(query.attrs)

        if query.dims is not None:
            dim_names = tuple(query.dims or ())
        elif query.coords is False:
            dim_names = ()

        order = query.order
    else:
        # set default order:  TILEDB_UNORDERED for sparse,  TILEDB_ROW_MAJOR for dense
        order = "U" if schema.sparse else "C"

    try:
        layout = "CFGU".index(order)
    except ValueError:
        raise ValueError(
            "order must be 'C' (TILEDB_ROW_MAJOR), 'F' (TILEDB_COL_MAJOR),  "
            "'U' (TILEDB_UNORDERED), or 'G' (TILEDB_GLOBAL_ORDER)"
        )

    return PyQuery(
        array._ctx_(),
        array,
        attr_names,
        dim_names,
        layout,
        use_arrow,
    )


def _get_pyquery_results(pyquery, schema):
    result_dict = OrderedDict()
    for name, item in pyquery.results().items():
        if len(item[1]) > 0:
            arr = pyquery.unpack_buffer(name, item[0], item[1])
        else:
            arr = item[0]
            final_dtype = schema.attr_or_dim_dtype(name)
            if len(arr) < 1 and (
                np.issubdtype(final_dtype, np.bytes_)
                or np.issubdtype(final_dtype, np.unicode_)
            ):
                # special handling to get correctly-typed empty array
                # (expression below changes itemsize from 0 to 1)
                arr.dtype = final_dtype.str + "1"
            else:
                arr.dtype = schema.attr_or_dim_dtype(name)
        result_dict[name if name != "__attr" else ""] = arr
    return result_dict


def _tiledb_result_as_dataframe(result_dict, array_meta):
    import pandas as pd

    df = pd.DataFrame.from_dict(result_dict)

    col_dtypes = {}
    if "__pandas_attribute_repr" in array_meta:
        col_dtypes.update(json.loads(array_meta["__pandas_attribute_repr"]))

    index_names = []
    if "__pandas_index_dims" in array_meta:
        index_dtypes = json.loads(array_meta["__pandas_index_dims"])
        index_names.extend(index_dtypes.keys())
        for name in index_names:
            if name in df:
                col_dtypes[name] = index_dtypes[name]

    if col_dtypes:
        df = df.astype(col_dtypes)

    if index_names:
        index_cols = [name for name in index_names if name in df]
        if index_cols:
            df.set_index(index_cols, inplace=True)
        elif len(index_names) == 1 and index_names[0] != "__tiledb_rows":
            df.index.rename(index_names[0], inplace=True)

    return df


def _pyarrow_to_pandas(array, pyquery, query, debug=False):
    # TODO currently there is lack of support for Arrow list types.
    # This prevents multi-value attributes, asides from strings, from being
    # queried properly. Until list attributes are supported in core,
    # error with a clear message to pass use_arrow=False.
    if query.attrs is not None and any(
        (attr.isvar or len(attr.dtype) > 1) and attr.dtype != np.unicode_
        for attr in map(array.schema.attr, query.attrs)
    ):
        raise TileDBError(
            "Multi-value attributes are not currently supported when use_arrow=True. "
            "This includes all variable-length attributes and fixed-length "
            "attributes with more than one value. Use `query(use_arrow=False)`."
        )

    try:
        table = pyquery._buffers_to_pa_table()
    except Exception as exc:
        if debug:
            print(f"Exception during pa.Table conversion: '{exc}'")
            return pyquery
        raise

    try:
        res_df = table.to_pandas()
    except Exception as exc:
        if debug:
            print(f"Exception during Pandas conversion: '{exc}'")
            return table, pyquery
        raise

    pd_idx_start = time.time()

    meta = array.meta
    # see also: write path in dataframe_.py
    if "__pandas_index_dims" in meta:
        index_dims = json.loads(meta["__pandas_index_dims"])
    else:
        index_dims = {}

    indexes = []
    rename_cols = {}
    for col_name in res_df.columns.values:
        if col_name in index_dims:
            # this is an auto-created column and should be unnamed
            if col_name == "__tiledb_rows":
                rename_cols["__tiledb_rows"] = None
                indexes.append(None)
            else:
                indexes.append(col_name)

    if rename_cols:
        res_df.rename(columns=rename_cols, inplace=True)

    if query is not None:
        # if we have a query with index_col set, then override any
        # index information saved with the array.
        if query.index_col is not True and query.index_col is not None:
            res_df.set_index(query.index_col, inplace=True)
        elif query.index_col is True and len(indexes) > 0:
            # still need to set indexes here b/c df creates query every time
            res_df.set_index(indexes, inplace=True)
        # else don't convert any column to a dataframe index
    elif indexes:
        res_df.set_index(indexes, inplace=True)

    # apply type translation from TileDB-Py write path
    if "__pandas_attribute_repr" in meta:
        attr_reprs = json.loads(meta["__pandas_attribute_repr"])
        if attr_reprs:
            res_df = res_df.astype(attr_reprs)

    if use_stats():
        pd_idx_duration = time.time() - pd_idx_start
        increment_stat("py.pandas_index_update_time", pd_idx_duration)

    return res_df
