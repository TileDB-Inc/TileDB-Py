import dataclasses
import json
import time
import weakref
from collections import OrderedDict
from contextlib import contextmanager
from numbers import Real
from typing import Any, Dict, Iterator, List, Optional, Sequence, Tuple, Union, cast
from dataclasses import dataclass
from itertools import zip_longest

import numpy as np

from tiledb import Array, ArraySchema, TileDBError
from tiledb.main import PyQuery, increment_stat, use_stats
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
        start = time.time()
        try:
            yield
        finally:
            increment_stat(key, time.time() - start)


def mr_dense_result_shape(
    ranges: Sequence[Sequence[Range]], base_shape: Optional[Tuple[int, ...]] = None
) -> Tuple[int, ...]:
    if base_shape is not None:
        assert len(ranges) == len(base_shape), "internal error: mismatched shapes"

    new_shape = []
    for i, subranges in enumerate(ranges):
        if subranges:
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
    nonempty_domain: Optional[Range] = None,
) -> Iterator[Range]:
    if sel is None and nonempty_domain is None:
        pass

    elif isinstance(sel, slice):
        if sel.step is not None:
            raise ValueError("Stepped slice ranges are not supported")

        rstart = sel.start
        if rstart is None and nonempty_domain:
            rstart = nonempty_domain[0]

        rend = sel.stop
        if rend is None and nonempty_domain:
            rend = nonempty_domain[1]

        if rstart is None or rend is None:
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
    nonempty_domain = array.nonempty_domain() or []
    idx = [idx] if not isinstance(idx, tuple) else idx

    for i, (ned, dim_sel) in enumerate(zip_longest(nonempty_domain, idx)):
        # don't try to index nonempty_domain if None
        ned = ned or None

        if not isinstance(dim_sel, list):
            dim_sel = [dim_sel]

        ranges[i] = tuple(
            rng
            for sel in dim_sel
            for rng in iter_ranges(sel if sel is not None else ned, ned)
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
        self.pyquery = None
        self.use_arrow = None

    @property
    def array(self) -> Array:
        array = self.array_ref()
        if array is None:
            raise RuntimeError(
                "Internal error: invariant violation (indexing call w/ dead array_ref)"
            )
        return array

    def __getitem__(self, idx: Any) -> Dict[str, np.ndarray]:
        with timing("py.getitem_time"):
            if idx is EmptyRange:
                return _get_empty_results(self.array.schema, self.query)

            self.ranges = getitem_ranges(self.array, idx)

            if self.query and self.query.return_incomplete:
                return self

            return self._run_query(self.query)

    def _run_query(
        self, query: Optional[Query] = None, preload_metadata: bool = False
    ) -> Union[Dict[str, np.ndarray], DataFrame, Table]:

        if self.pyquery is None or not self.pyquery.is_incomplete:
            self.pyquery = _get_pyquery(self.array, query, self.use_arrow)
            self.pyquery._preload_metadata = preload_metadata
            self.pyquery.set_ranges(self.ranges)

            if self.query is not None:
                try:
                    self.pyquery.set_attr_cond(query.attr_cond)
                except TileDBError as e:
                    raise TileDBError(e)

            self.pyquery._return_incomplete = (
                self.query and self.query.return_incomplete
            )

        self.pyquery.submit()

        schema = self.array.schema
        if query is not None and self.use_arrow:
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
                table = self.pyquery._buffers_to_pa_table()
                return table if query.return_arrow else table.to_pandas()

        result_dict = _get_pyquery_results(self.pyquery, schema)
        if not schema.sparse:
            result_shape = mr_dense_result_shape(self.ranges, schema.shape)
            for arr in result_dict.values():
                # TODO check/test layout
                arr.shape = result_shape
        return result_dict

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
        results = {}
        if not self.pyquery:
            raise TileDBError("Query not initialized")
        tmp = self.pyquery.estimated_result_sizes()
        for name, val in tmp.items():
            results[name] = EstimatedResultSize(val[0], val[1])

        return results

    def __iter__(self):
        if not self.query.return_incomplete:
            raise TileDBError(
                "Cannot iterate unless query is initialized with return_incomplete=True"
            )
        return self

    def __next__(self):
        if self.pyquery and not self.pyquery.is_incomplete:
            raise StopIteration()
        return self._run_query(self.query)


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
        if pyarrow and use_arrow is None:
            use_arrow = True
        self.use_arrow = use_arrow

    def __getitem__(self, idx: Any) -> Union[DataFrame, Table]:
        with timing("py.getitem_time"):
            check_dataframe_deps()
            array = self.array
            # we need to use a Query in order to get coords for a dense array
            query = self.query if self.query else Query(array, coords=True)
            if idx is EmptyRange:
                result = _get_empty_results(array.schema, query)
            else:
                self.ranges = getitem_ranges(self.array, idx)

                if self.query and self.query.return_incomplete:
                    return self

                result = self._run_query(query, preload_metadata=True)
            if not (pyarrow and isinstance(result, pyarrow.Table)):
                if not isinstance(result, DataFrame):
                    result = DataFrame.from_dict(result)
                with timing("py.pandas_index_update_time"):
                    result = _update_df_from_meta(result, array.meta, query.index_col)
            return result


def _get_pyquery(array: Array, query: Optional[Query], use_arrow: bool) -> PyQuery:
    schema = array.schema
    if query is not None:
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
        tuple(_iter_attr_names(schema, query)),
        tuple(_iter_dim_names(schema, query)),
        layout,
        use_arrow,
    )


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
