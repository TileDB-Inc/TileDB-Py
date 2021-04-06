import json
import time
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from numbers import Real
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union, cast

import numpy as np

from tiledb import Array, ArraySchema, TileDBError
from tiledb.core import PyQuery, increment_stat, use_stats
from tiledb.libtiledb import Metadata, Query

from .dataframe_ import check_dataframe_deps

try:
    import pyarrow

    Table = Union[pyarrow.Table]
except ImportError:
    pyarrow = Table = None

try:
    from pandas import DataFrame
except ImportError:
    DataFrame = None


# TODO: expand with more accepted scalar types
Scalar = Real
Range = Tuple[Scalar, Scalar]


@contextmanager
def timing(key: str) -> Iterator[None]:
    if not use_stats():
        yield
    else:
        start = time.time()
        try:
            yield
        finally:
            increment_stat(key, time.time() - start)


def mr_dense_result_shape(
    ranges: Sequence[Sequence[Range]],
    base_shape: Optional[Tuple[int, ...]] = None,
) -> Tuple[int, ...]:
    if base_shape is not None:
        assert len(ranges) == len(base_shape), "internal error: mismatched shapes"

    new_shape = []
    for i, subranges in enumerate(ranges):
        if subranges:
            total_length = np.sum(abs(stop - start) + 1 for start, stop in subranges)
            new_shape.append(np.uint64(total_length))
        elif base_shape is not None:
            # empty range covers dimension
            new_shape.append(base_shape[i])
        else:
            raise ValueError("Missing required base_shape for whole-dimension slices")

    return tuple(new_shape)


def to_scalar(obj: Any) -> Scalar:
    if np.isscalar(obj):
        return cast(Scalar, obj)
    if isinstance(obj, np.ndarray) and obj.ndim == 0:
        return cast(Scalar, obj[()])
    raise ValueError(f"Cannot convert {type(obj)} to scalar")


def iter_ranges(
    sel: Union[Scalar, slice, Range, List[Scalar]],
    nonempty_domain: Optional[Range] = None,
) -> Iterator[Range]:
    if isinstance(sel, slice):
        if sel.step is not None:
            raise ValueError("Stepped slice ranges are not supported")

        rstart = sel.start
        if rstart is None and nonempty_domain:
            rstart = nonempty_domain[0]

        rend = sel.stop
        if rend is None and nonempty_domain:
            rend = nonempty_domain[1]

        if rstart is None or rend is None:
            raise TileDBError("Open-ended slicing requires a valid nonempty_domain")

        yield to_scalar(rstart), to_scalar(rend)

    elif isinstance(sel, tuple):
        assert len(sel) == 2
        yield to_scalar(sel[0]), to_scalar(sel[1])

    elif isinstance(sel, list):
        for scalar in map(to_scalar, sel):
            yield scalar, scalar

    else:
        scalar = to_scalar(sel)
        yield scalar, scalar


def getitem_ranges(array: Array, idx: Any) -> Sequence[Sequence[Range]]:
    ranges: List[Sequence[Range]] = [()] * array.schema.domain.ndim
    ned = array.nonempty_domain()
    for i, dim_sel in enumerate([idx] if not isinstance(idx, tuple) else idx):
        # don't try to index nonempty_domain if None
        nonempty_domain = ned[i] if ned else None
        if not isinstance(dim_sel, list):
            dim_sel = [dim_sel]
        ranges[i] = tuple(
            rng for sel in dim_sel for rng in iter_ranges(sel, nonempty_domain)
        )
    return tuple(ranges)


class MultiRangeIndexer(object):
    """
    Implements multi-range indexing.
    """

    def __init__(self, array: Array, query: Optional[Query] = None) -> None:
        if not isinstance(array, Array):
            raise TypeError("Internal error: MultiRangeIndexer expected tiledb.Array")
        self.array_ref = weakref.ref(array)
        self.query = query

    @property
    def array(self) -> Array:
        array = self.array_ref()
        if array is None:
            raise RuntimeError(
                "Internal error: invariant violation (indexing call w/ dead array_ref)"
            )
        return array

    def __getitem__(self, idx: Any) -> Dict[str, np.ndarray]:
        return _run_query(self.array, getitem_ranges(self.array, idx), query=self.query)


class DataFrameIndexer(MultiRangeIndexer):
    """
    Implements `.df[]` indexing to directly return a dataframe
    [] operator uses multi_index semantics.
    """

    def __init__(
        self,
        array: Array,
        query: Optional[Query] = None,
        use_arrow: Optional[bool] = None,
    ) -> None:
        super().__init__(array, query)
        if use_arrow is None:
            use_arrow = True
        self.use_arrow = use_arrow

    def __getitem__(self, idx: Any) -> Union[DataFrame, Table]:
        check_dataframe_deps()
        with timing("py.__getitem__time"):
            array = self.array

            # we need to use a Query in order to get coords for a dense array
            query = self.query or Query(array, coords=True)
            result = _run_query(
                array,
                getitem_ranges(array, idx),
                query=query,
                use_arrow=bool(
                    pyarrow is not None and (self.use_arrow or query.return_arrow)
                ),
                preload_metadata=True,
            )
            if not isinstance(result, pyarrow.Table):
                if not isinstance(result, DataFrame):
                    result = DataFrame.from_dict(result)
                with timing("py.pandas_index_update_time"):
                    result = _update_df_from_meta(result, array.meta, query.index_col)
            return result


def _run_query(
    array: Array,
    ranges: Sequence[Sequence[Range]],
    *,
    query: Optional[Query] = None,
    use_arrow: bool = False,
    preload_metadata: bool = False,
) -> Union[Dict[str, np.ndarray], DataFrame, Table]:
    pyquery = _get_pyquery(array, query, use_arrow)
    pyquery._preload_metadata = preload_metadata
    pyquery.set_ranges(ranges)
    pyquery.submit()

    schema = array.schema
    if query is not None and use_arrow:
        # TODO currently there is lack of support for Arrow list types.
        # This prevents multi-value attributes, asides from strings, from being
        # queried properly. Until list attributes are supported in core,
        # error with a clear message to pass use_arrow=False.
        attrs = map(schema.attr, query.attrs or ())
        if any(
            (attr.isvar or len(attr.dtype) > 1) and attr.dtype != np.unicode_
            for attr in attrs
        ):
            raise TileDBError(
                "Multi-value attributes are not currently supported when use_arrow=True. "
                "This includes all variable-length attributes and fixed-length "
                "attributes with more than one value. Use `query(use_arrow=False)`."
            )
        with timing("py.buffer_conversion_time"):
            table = pyquery._buffers_to_pa_table()
            return table if query.return_arrow else table.to_pandas()

    result_dict = _get_pyquery_results(pyquery, schema)
    if not schema.sparse:
        result_shape = mr_dense_result_shape(ranges, schema.shape)
        for arr in result_dict.values():
            # TODO check/test layout
            arr.shape = result_shape
    return result_dict


def _get_pyquery(array: Array, query: Optional[Query], use_arrow: bool) -> PyQuery:
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


def _get_pyquery_results(
    pyquery: PyQuery, schema: ArraySchema
) -> Dict[str, np.ndarray]:
    result_dict = OrderedDict()
    for name, item in pyquery.results().items():
        if len(item[1]) > 0:
            arr = pyquery.unpack_buffer(name, item[0], item[1])
        else:
            arr = item[0]
            arr.dtype = schema.attr_or_dim_dtype(name)
        result_dict[name if name != "__attr" else ""] = arr
    return result_dict


def _update_df_from_meta(
    df: DataFrame, array_meta: Metadata, index_col: Union[List[str], bool, None] = True
) -> DataFrame:
    col_dtypes = {}
    if "__pandas_attribute_repr" in array_meta:
        attr_dtypes = json.loads(array_meta["__pandas_attribute_repr"])
        for name, dtype in attr_dtypes.items():
            if name in df:
                col_dtypes[name] = dtype

    index_names = []
    if "__pandas_index_dims" in array_meta:
        index_dtypes = json.loads(array_meta["__pandas_index_dims"])
        index_names.extend(index_dtypes.keys())
        for name, dtype in index_dtypes.items():
            if name in df:
                col_dtypes[name] = dtype

    if col_dtypes:
        df = df.astype(col_dtypes)

    if index_col:
        if index_col is not True:
            # if we have a query with index_col set, then override any
            # index information saved with the array.
            df.set_index(index_col, inplace=True)
        elif index_names:
            # set index the index names that exist as columns
            index_names_df = [name for name in index_names if name in df]
            if index_names_df:
                df.set_index(index_names_df, inplace=True)

            # for single index column, ensure that the index name is preserved
            # or renamed from __tiledb_rows to None
            if len(index_names) == 1:
                index_name = index_names[0]
                if index_name == "__tiledb_rows":
                    index_name = None
                if df.index.name != index_name:
                    df.index.rename(index_name, inplace=True)

    return df
