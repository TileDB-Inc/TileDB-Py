import importlib.util
import json
import time
import weakref
from abc import ABC, abstractmethod
from collections import OrderedDict
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from numbers import Real
from typing import (
    TYPE_CHECKING,
    Any,
    Dict,
    Iterator,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
    cast,
)

import numpy as np

from .cc import TileDBError
from .dataframe_ import check_dataframe_deps
from .libtiledb import Array, ArraySchema, Metadata
from .libtiledb import Query as QueryProxy
from .main import PyQuery, increment_stat, use_stats
from .query import Query
from .query_condition import QueryCondition
from .subarray import Subarray

if TYPE_CHECKING:
    # We don't want to import these eagerly since importing Pandas in particular
    # can add around half a second of import time even if we never use it.
    import pandas
    import pyarrow


current_timer: ContextVar[str] = ContextVar("timer_scope")


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


def iter_label_range(sel: Union[Scalar, slice, Range, List[Scalar]]):
    if isinstance(sel, slice):
        if sel.start is None or sel.start is None:
            raise NotImplementedError(
                "partial and full indexing is not yet supported on dimension labels"
            )

        yield to_scalar(sel.start), to_scalar(sel.stop)

    elif isinstance(sel, tuple):
        assert len(sel) == 2
        yield to_scalar(sel[0]), to_scalar(sel[1])

    elif isinstance(sel, list):
        for scalar in map(to_scalar, sel):
            yield scalar, scalar

    else:
        scalar = to_scalar(sel)
        yield scalar, scalar


def dim_ranges_from_selection(selection, nonempty_domain, is_sparse):
    # don't try to index nonempty_domain if None
    if isinstance(selection, np.ndarray):
        return selection
    selection = selection if isinstance(selection, list) else [selection]
    return tuple(
        rng for sel in selection for rng in iter_ranges(sel, is_sparse, nonempty_domain)
    )


def label_ranges_from_selection(selection):
    if isinstance(selection, np.ndarray):
        return tuple(tuple(x, x) for x in selection)
    selection = selection if isinstance(selection, list) else [selection]
    return tuple(rng for sel in selection for rng in iter_label_range(sel))


def getitem_ranges(array: Array, idx: Any) -> Sequence[Sequence[Range]]:
    ranges: List[Sequence[Range]] = [()] * array.schema.domain.ndim
    ned = array.nonempty_domain()
    if ned is None:
        ned = [None] * array.schema.domain.ndim
    is_sparse = array.schema.sparse
    for i, dim_sel in enumerate([idx] if not isinstance(idx, tuple) else idx):
        ranges[i] = dim_ranges_from_selection(dim_sel, ned[i], is_sparse)
    return tuple(ranges)


def getitem_ranges_with_labels(
    array: Array, labels: Dict[int, str], idx: Any
) -> Tuple[Sequence[Sequence[Range]], Dict[str, Sequence[Range]]]:
    dim_ranges: List[Sequence[Range]] = [()] * array.schema.domain.ndim
    label_ranges: Dict[str, Sequence[Range]] = {}
    ned = array.nonempty_domain()
    if ned is None:
        ned = [None] * array.schema.domain.ndim
    is_sparse = array.schema.sparse
    for dim_idx, dim_sel in enumerate([idx] if not isinstance(idx, tuple) else idx):
        if dim_idx in labels.keys():
            label_ranges[labels[dim_idx]] = label_ranges_from_selection(dim_sel)
        else:
            dim_ranges[dim_idx] = dim_ranges_from_selection(
                dim_sel, ned[dim_idx], is_sparse
            )
    return dim_ranges, label_ranges


class _BaseIndexer(ABC):
    """
    Implements multi-range indexing.
    """

    def __init__(
        self,
        array: Array,
        query: Optional[QueryProxy] = None,
        use_arrow: bool = False,
        preload_metadata: bool = False,
    ):
        if not isinstance(array, Array):
            raise TypeError("_BaseIndexer expected tiledb.Array")
        self.array_ref = weakref.ref(array)
        self.query = query
        self.use_arrow = use_arrow
        self.preload_metadata = preload_metadata
        self.subarray = None
        self.pyquery = None

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
            if idx is EmptyRange:
                self.pyquery = None
                self.subarray = None
            else:
                self._set_pyquery()
                self.subarray = Subarray(self.array)
                self._set_ranges(idx)
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

    def _set_pyquery(self):
        self.pyquery = _get_pyquery(
            self.array,
            self.query,
            self.use_arrow,
            self.return_incomplete,
            self.preload_metadata,
        )

    def _set_ranges(self, idx):
        ranges = getitem_ranges(self.array, idx)
        self._set_shape(ranges)
        with timing("add_ranges"):
            self.subarray.add_ranges(ranges)
        self.pyquery.set_subarray(self.subarray)

    def _set_shape(self, ranges):
        pass

    @abstractmethod
    def _run_query(self):
        """Run the query for the latest __getitem__ call and return the result"""


class MultiRangeIndexer(_BaseIndexer):
    """
    Implements multi-range indexing.
    """

    def __init__(self, array: Array, query: Optional[QueryProxy] = None):
        if query and query.return_arrow:
            raise TileDBError("`return_arrow=True` requires .df indexer`")
        super().__init__(array, query)
        self.result_shape = None

    def _set_shape(self, ranges):
        schema = self.array.schema
        if not schema.sparse and len(schema.shape) > 1:
            self.result_shape = mr_dense_result_shape(ranges, schema.shape)
        else:
            self.result_shape = None

    def _run_query(self) -> Dict[str, np.ndarray]:
        if self.pyquery is None:
            return self._empty_results

        self.pyquery.submit()
        result_dict = _get_pyquery_results(self.pyquery, self.array)
        if self.result_shape is not None:
            for name, arr in result_dict.items():
                # TODO check/test layout
                if not self.array.schema.has_dim_label(name):
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
        query: Optional[QueryProxy] = None,
        use_arrow: Optional[bool] = None,
    ):
        check_dataframe_deps()
        # we need to use a Query in order to get coords for a dense array
        if not query:
            query = QueryProxy(array, coords=True)
        use_arrow = (
            bool(importlib.util.find_spec("pyarrow"))
            if use_arrow is None
            else use_arrow
        )

        # TODO: currently there is lack of support for Arrow list types. This prevents
        # multi-value attributes, asides from strings, from being queried properly.
        # Until list attributes are supported in core, error with a clear message.
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

    def _run_query(self) -> Union["pandas.DataFrame", "pyarrow.Table"]:
        import pandas
        import pyarrow

        if self.pyquery is not None:
            self.pyquery.submit()

        if self.pyquery is None:
            df = pandas.DataFrame(self._empty_results)
        elif self.use_arrow:
            with timing("buffer_conversion_time"):
                table = self.pyquery._buffers_to_pa_table()

            columns = []
            pa_schema = table.schema
            for pa_attr in pa_schema:
                if not self.array.schema.has_attr(pa_attr.name):
                    continue

                tdb_attr = self.array.attr(pa_attr.name)

                if tdb_attr.enum_label is not None:
                    enmr = self.array.enum(tdb_attr.enum_label)
                    col = pyarrow.DictionaryArray.from_arrays(
                        indices=table[pa_attr.name].combine_chunks(),
                        dictionary=enmr.values(),
                    )
                    idx = pa_schema.get_field_index(pa_attr.name)
                    table = table.set_column(idx, pa_attr.name, col)
                    pa_schema = table.schema
                    continue

                if np.issubdtype(tdb_attr.dtype, bool):
                    # this is a workaround to cast TILEDB_BOOL types from uint8
                    # representation in Arrow to Boolean
                    dtype = "uint8"
                elif tdb_attr.isnullable and np.issubdtype(tdb_attr.dtype, np.integer):
                    # this is a workaround for PyArrow's to_pandas function
                    # converting all integers with NULLs to float64:
                    # https://arrow.apache.org/docs/python/pandas.html#arrow-pandas-conversion
                    extended_dtype_mapping = {
                        pyarrow.int8(): pandas.Int8Dtype(),
                        pyarrow.int16(): pandas.Int16Dtype(),
                        pyarrow.int32(): pandas.Int32Dtype(),
                        pyarrow.int64(): pandas.Int64Dtype(),
                        pyarrow.uint8(): pandas.UInt8Dtype(),
                        pyarrow.uint16(): pandas.UInt16Dtype(),
                        pyarrow.uint32(): pandas.UInt32Dtype(),
                        pyarrow.uint64(): pandas.UInt64Dtype(),
                    }
                    dtype = extended_dtype_mapping[pa_attr.type]
                else:
                    continue

                columns.append(
                    {
                        "field_name": tdb_attr.name,
                        "name": tdb_attr.name,
                        "numpy_type": f"{dtype}",
                        "pandas_type": f"{dtype}",
                    }
                )

            metadata = {
                b"pandas": json.dumps(
                    {
                        "columns": columns,
                        "index_columns": [
                            {
                                "kind": "range",
                                "name": None,
                                "start": 0,
                                "step": 1,
                                "stop": len(table),
                            }
                        ],
                    }
                ).encode()
            }

            table = table.cast(pyarrow.schema(pa_schema).with_metadata(metadata))

            if self.query.return_arrow:
                return table

            df = table.to_pandas()
        else:
            df = pandas.DataFrame(_get_pyquery_results(self.pyquery, self.array))

        with timing("pandas_index_update_time"):
            return _update_df_from_meta(df, self.array.meta, self.query.index_col)


class LabelIndexer(MultiRangeIndexer):
    """
    Implements multi-range indexing by label.
    """

    def __init__(
        self, array: Array, labels: Sequence[str], query: Optional[QueryProxy] = None
    ):
        if array.schema.sparse:
            raise NotImplementedError(
                "querying sparse arrays by label is not yet implemented"
            )
        super().__init__(array, query)
        self.label_query: Optional[Query] = None
        self._labels: Dict[int, str] = {}
        for label_name in labels:
            dim_label = array.schema.dim_label(label_name)
            dim_idx = dim_label.dim_index
            if dim_idx in self._labels:
                raise TileDBError(
                    f"cannot set labels `{self._labels[dim_idx]}` and "
                    f"`{label_name}` defined on the same dimension"
                )
            self._labels[dim_idx] = label_name

    def _set_ranges(self, idx):
        dim_ranges, label_ranges = getitem_ranges_with_labels(
            self.array, self._labels, idx
        )
        if label_ranges is None:
            with timing("add_ranges"):
                self.subarray.add_ranges(tuple(dim_ranges))
            # No label query.
            self.label_query = None
            # All ranges are finalized: set shape and subarray now.
            self._set_shape(dim_ranges)
            self.pyquery.set_subarray(self.subarray)
        else:
            label_subarray = Subarray(self.array)
            with timing("add_ranges"):
                self.subarray.add_ranges(dim_ranges=dim_ranges)
                label_subarray.add_ranges(label_ranges=label_ranges)
            self.label_query = Query(self.array)
            self.label_query.set_subarray(label_subarray)

    def _run_query(self) -> Dict[str, np.ndarray]:
        # If querying by label and the label query is not yet complete, run the label
        # query and update the pyquery with the actual dimensions.
        if self.label_query is not None and not self.label_query.is_complete():
            self.label_query.submit()

            if not self.label_query.is_complete():
                raise TileDBError("failed to get dimension ranges from labels")
            label_subarray = self.label_query.subarray()
            # Check that the label query returned results for all dimensions.
            if any(
                label_subarray.num_dim_ranges(dim_idx) == 0 for dim_idx in self._labels
            ):
                self.pyquery = None
            else:
                # Get the ranges from the label query and set to the
                self.subarray.copy_ranges(
                    self.label_query.subarray(), self._labels.keys()
                )
                self.pyquery.set_subarray(self.subarray)
            self.result_shape = self.subarray.shape()
            for dim_idx, label_name in self._labels.items():
                if self.result_shape is None:
                    raise TileDBError("failed to compute subarray shape")
                self.pyquery.add_label_buffer(label_name, self.result_shape[dim_idx])
        return super()._run_query()


def _get_pyquery(
    array: Array,
    query: Optional[QueryProxy],
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

    pyquery._return_incomplete = return_incomplete
    pyquery._preload_metadata = preload_metadata
    if query and query.cond is not None:
        if isinstance(query.cond, str):
            pyquery.set_cond(QueryCondition(query.cond))
        elif isinstance(query.cond, QueryCondition):
            raise TileDBError(
                "Passing `tiledb.QueryCondition` to `cond` is no longer supported "
                "as of 0.19.0. Instead of `cond=tiledb.QueryCondition('expression')` "
                "you must use `cond='expression'`. This message will be "
                "removed in 0.21.0.",
            )
        else:
            raise TypeError("`cond` expects type str.")

    return pyquery


def _iter_attr_names(
    schema: ArraySchema, query: Optional[QueryProxy] = None
) -> Iterator[str]:
    if query is not None and query.attrs is not None:
        return iter(query.attrs)
    return (schema.attr(i)._internal_name for i in range(schema.nattr))


def _iter_dim_names(
    schema: ArraySchema, query: Optional[QueryProxy] = None
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


def _get_pyquery_results(pyquery: PyQuery, array: Array) -> Dict[str, np.ndarray]:
    schema = array.schema
    result_dict = OrderedDict()
    for name, item in pyquery.results().items():
        if len(item[1]) > 0:
            arr = pyquery.unpack_buffer(name, item[0], item[1])
        else:
            arr = item[0]
            arr.dtype = (
                schema.attr_or_dim_dtype(name)
                if not schema.has_dim_label(name)
                else schema.dim_label(name).dtype
            )

        if schema.has_attr(name):
            enum_label = schema.attr(name).enum_label
            if enum_label is not None:
                values = array.enum(enum_label).values()
                arr = np.array([values[idx] for idx in arr])

        result_dict[name if name != "__attr" else ""] = arr
    return result_dict


def _get_empty_results(
    schema: ArraySchema, query: Optional[QueryProxy] = None
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
    df: "pandas.DataFrame",
    array_meta: Metadata,
    index_col: Union[List[str], bool, None] = True,
) -> "pandas.DataFrame":
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
