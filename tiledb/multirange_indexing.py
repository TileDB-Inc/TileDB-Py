import json
import time
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from numbers import Real
from dataclasses import dataclass
import warnings
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union, cast


import numpy as np

from tiledb import Array, ArraySchema, QueryCondition, TileDBError
from tiledb.main import PyQuery, increment_stat, use_stats
from tiledb.libtiledb import Metadata, Query

from .dataframe_ import check_dataframe_deps

current_timer: ContextVar[str] = ContextVar("timer_scope")

try:
    import pyarrow
    from pyarrow import Table
except ImportError:
    pyarrow = Table = None

try:
    from pandas import DataFrame
except ImportError:
    DataFrame = None


# sentinel value to denote selecting an empty range
EmptyRange = object()

# TODO: expand with more accepted scalar types
Scalar = Real
Range = Tuple[Scalar, Scalar]


@dataclass
class EstimatedResultSize:
    offsets_bytes: int
    data_bytes: int


@contextmanager
def timing(key: str) -> Iterator[None]:
    if not use_stats():
        yield
    else:
        scoped_name = f"{current_timer.get('py')}.{key}"
        parent_token = current_timer.set(scoped_name)
        start = time.time()
        try:
            yield
        finally:
            increment_stat(current_timer.get(), time.time() - start)
            current_timer.reset(parent_token)


def mr_dense_result_shape(
    ranges: Sequence[Sequence[Range]], base_shape: Optional[Tuple[int, ...]] = None
) -> Tuple[int, ...]:
    if base_shape is not None:
        assert len(ranges) == len(base_shape), "internal error: mismatched shapes"

    new_shape = []
    for i, subranges in enumerate(ranges):
        if isinstance(subranges, np.ndarray):
            total_length = len(subranges)
            new_shape.append(np.uint64(total_length))
        elif subranges not in (None, ()):
            total_length = sum(abs(stop - start) + 1 for start, stop in subranges)
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
    sparse: bool,
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

        if sparse and sel.start is None and sel.stop is None:
            # don't set nonempty_domain for full-domain slices w/ sparse
            # because TileDB query is faster without the constraint
            pass
        elif rstart is None or rend is None:
            pass
        else:
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
    is_sparse = array.schema.sparse
    for i, dim_sel in enumerate([idx] if not isinstance(idx, tuple) else idx):
        # don't try to index nonempty_domain if None
        nonempty_domain = ned[i] if ned else None
        if isinstance(dim_sel, np.ndarray):
            ranges[i] = dim_sel
            continue
        elif not isinstance(dim_sel, list):
            dim_sel = [dim_sel]
        ranges[i] = tuple(
            rng
            for sel in dim_sel
            for rng in iter_ranges(sel, is_sparse, nonempty_domain)
        )
    return tuple(ranges)


class _BaseIndexer(ABC):
    """
    Implements multi-range indexing.
    """

    def __init__(
        self,
        array: Array,
        query: Optional[Query] = None,
        use_arrow: bool = False,
        preload_metadata: bool = False,
    ):
        if not isinstance(array, Array):
            raise TypeError("_BaseIndexer expected tiledb.Array")
        self.array_ref = weakref.ref(array)
        self.query = query
        self.use_arrow = use_arrow
        self.preload_metadata = preload_metadata

    @property
    def array(self) -> Array:
        array = self.array_ref()
        if array is None:
            raise RuntimeError(
                "Internal error: invariant violation (indexing call w/ dead array_ref)"
            )
        return array

    @property
    def return_incomplete(self) -> bool:
        return bool(self.query and self.query.return_incomplete)

    def __getitem__(self, idx):
        with timing("getitem_time"):
            self._set_ranges(
                getitem_ranges(self.array, idx) if idx is not EmptyRange else None
            )
            return self if self.return_incomplete else self._run_query()

    def estimated_result_sizes(self):
        """
        Get the estimated result buffer sizes for a TileDB Query

        Sizes are returned in bytes as an EstimatedResultSize dataclass
        with two fields: `offset_bytes` and `data_bytes`, with buffer
        name as the OrderedDict key.
        See the corresponding TileDB Embedded API documentation for
        additional details:

        https://tiledb-inc-tiledb.readthedocs-hosted.com/en/stable/c++-api.html#query

        :return: OrderedDict of key: str -> EstimatedResultSize
        """
        if not hasattr(self, "pyquery"):
            raise TileDBError("Query not initialized")

        if self.pyquery is None:
            return {
                name: EstimatedResultSize(0, 0) for name in self._empty_results.keys()
            }
        else:
            return {
                name: EstimatedResultSize(*values)
                for name, values in self.pyquery.estimated_result_sizes().items()
            }

    def __iter__(self):
        if not hasattr(self, "pyquery"):
            raise TileDBError("Query not initialized")
        if not self.return_incomplete:
            raise TileDBError(
                "Cannot iterate unless query is initialized with return_incomplete=True"
            )
        while True:
            yield self._run_query()
            if self.pyquery is None or not self.pyquery.is_incomplete:
                break

    @property
    def _empty_results(self):
        return _get_empty_results(self.array.schema, self.query)

    def _set_ranges(self, ranges):
        self.pyquery = (
            _get_pyquery(
                self.array,
                self.query,
                ranges,
                self.use_arrow,
                self.return_incomplete,
                self.preload_metadata,
            )
            if ranges is not None
            else None
        )

    @abstractmethod
    def _run_query(self):
        """Run the query for the latest __getitem__ call and return the result"""


class MultiRangeIndexer(_BaseIndexer):
    """
    Implements multi-range indexing.
    """

    def __init__(self, array: Array, query: Optional[Query] = None):
        if query and query.return_arrow:
            raise TileDBError("`return_arrow=True` requires .df indexer`")
        super().__init__(array, query)
        self.result_shape = None

    def _set_ranges(self, ranges):
        super()._set_ranges(ranges)
        schema = self.array.schema
        if ranges is not None and not schema.sparse and len(schema.shape) > 1:
            self.result_shape = mr_dense_result_shape(ranges, schema.shape)
        else:
            self.result_shape = None

    def _run_query(self) -> Dict[str, np.ndarray]:
        if self.pyquery is None:
            return self._empty_results

        self.pyquery.submit()
        result_dict = _get_pyquery_results(self.pyquery, self.array.schema)
        if self.result_shape is not None:
            for arr in result_dict.values():
                # TODO check/test layout
                arr.shape = self.result_shape
        return result_dict


class DataFrameIndexer(_BaseIndexer):
    """
    Implements `.df[]` indexing to directly return a dataframe
    [] operator uses multi_index semantics.
    """

    def __init__(
        self,
        array: Array,
        query: Optional[Query] = None,
        use_arrow: Optional[bool] = None,
    ):
        check_dataframe_deps()
        # we need to use a Query in order to get coords for a dense array
        if not query:
            query = Query(array, coords=True)
        if use_arrow is None:
            use_arrow = pyarrow is not None
        # TODO: currently there is lack of support for Arrow list types. This prevents
        # multi-value attributes, asides from strings, from being queried properly. Until
        # list attributes are supported in core, error with a clear message.
        if use_arrow and any(
            (attr.isvar or len(attr.dtype) > 1)
            and attr.dtype not in (np.unicode_, np.bytes_)
            for attr in map(array.attr, query.attrs or ())
        ):
            raise TileDBError(
                "Multi-value attributes are not currently supported when use_arrow=True. "
                "This includes all variable-length attributes and fixed-length "
                "attributes with more than one value. Use `query(use_arrow=False)`."
            )
        super().__init__(array, query, use_arrow, preload_metadata=True)

    def _run_query(self) -> Union[DataFrame, Table]:
        if self.pyquery is not None:
            self.pyquery.submit()

        if self.pyquery is None:
            df = DataFrame(self._empty_results)
        elif self.use_arrow:
            with timing("buffer_conversion_time"):
                table = self.pyquery._buffers_to_pa_table()

            # this is a workaround to cast TILEDB_BOOL types from uint8
            # representation in Arrow to Boolean
            schema = table.schema
            for attr_or_dim in schema:
                if not self.array.schema.has_attr(attr_or_dim.name):
                    continue

                attr = self.array.attr(attr_or_dim.name)
                if attr.dtype == bool:
                    field_idx = schema.get_field_index(attr.name)
                    field = pyarrow.field(attr.name, pyarrow.bool_())
                    schema = schema.set(field_idx, field)

            table = table.cast(schema)

            if self.query.return_arrow:
                return table

            df = table.to_pandas()
        else:
            df = DataFrame(_get_pyquery_results(self.pyquery, self.array.schema))

        with timing("pandas_index_update_time"):
            return _update_df_from_meta(df, self.array.meta, self.query.index_col)


def _get_pyquery(
    array: Array,
    query: Optional[Query],
    ranges: Sequence[Sequence[Range]],
    use_arrow: bool,
    return_incomplete: bool,
    preload_metadata: bool,
) -> PyQuery:
    schema = array.schema
    if query:
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

    pyquery = PyQuery(
        array._ctx_(),
        array,
        tuple(
            [array.view_attr]
            if array.view_attr is not None
            else _iter_attr_names(schema, query)
        ),
        tuple(_iter_dim_names(schema, query)),
        layout,
        use_arrow,
    )
    with timing("add_ranges"):
        if hasattr(pyquery, "set_ranges_bulk") and any(
            isinstance(r, np.ndarray) for r in ranges
        ):
            pyquery.set_ranges_bulk(ranges)
        else:
            pyquery.set_ranges(ranges)

    pyquery._return_incomplete = return_incomplete
    pyquery._preload_metadata = preload_metadata
    if query and query.cond is not None:
        if isinstance(query.cond, str):
            pyquery.set_cond(QueryCondition(query.cond))
        elif isinstance(query.cond, QueryCondition):
            from tiledb import version as tiledbpy_version

            assert tiledbpy_version() < (0, 19, 0)
            warnings.warn(
                "Passing `tiledb.QueryCondition` to `cond` is no longer "
                "required and is slated for removal in version 0.19.0. "
                "Instead of `cond=tiledb.QueryCondition('expression')`, "
                "use `cond='expression'`.",
                DeprecationWarning,
            )
            pyquery.set_cond(query.cond)
        else:
            raise TypeError("`cond` expects type str.")

    return pyquery


def _iter_attr_names(
    schema: ArraySchema, query: Optional[Query] = None
) -> Iterator[str]:
    if query is not None and query.attrs is not None:
        return iter(query.attrs)
    return (schema.attr(i)._internal_name for i in range(schema.nattr))


def _iter_dim_names(
    schema: ArraySchema, query: Optional[Query] = None
) -> Iterator[str]:
    if query is not None:
        if query.dims is not None:
            return iter(query.dims or ())
        if query.coords is False:
            return iter(())
    if not schema.sparse:
        return iter(())
    dom = schema.domain
    return (dom.dim(i).name for i in range(dom.ndim))


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


def _get_empty_results(
    schema: ArraySchema, query: Optional[Query] = None
) -> Dict[str, np.ndarray]:
    names = []
    query_dims = frozenset(_iter_dim_names(schema, query))
    query_attrs = frozenset(_iter_attr_names(schema, query))

    # return dims first, if any
    dom = schema.domain
    for i in range(dom.ndim):
        dim = dom.dim(i).name
        # we need to also check if this is an attr for backward-compatibility
        if dim in query_dims or dim in query_attrs:
            names.append(dim)

    for i in range(schema.nattr):
        attr = schema.attr(i)._internal_name
        if attr in query_attrs:
            names.append(attr)

    result_dict = OrderedDict()
    for name in names:
        arr = np.array([], schema.attr_or_dim_dtype(name))
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

    index_cols = []
    if "__pandas_index_dims" in array_meta:
        index_dtypes = json.loads(array_meta["__pandas_index_dims"])
        index_cols.extend(col for col in index_dtypes.keys() if col in df)
        for name, dtype in index_dtypes.items():
            if name in df:
                col_dtypes[name] = dtype

    if col_dtypes:
        df = df.astype(col_dtypes, copy=False)

    if index_col:
        if index_col is not True:
            # if we have a query with index_col set, then override any
            # index information saved with the array.
            df.set_index(index_col, inplace=True)
        elif index_cols:
            # set index the index names that exist as columns
            df.set_index(index_cols, inplace=True)

            # rename __tiledb_rows to None
            if "__tiledb_rows" in index_cols:
                index_cols[index_cols.index("__tiledb_rows")] = None
                if len(index_cols) == 1:
                    df.index.rename(index_cols[0], inplace=True)
                else:
                    df.index.rename(index_cols, inplace=True)

    return df
