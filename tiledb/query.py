from collections.abc import Sequence
from json import loads as json_loads
from typing import Optional, Sequence, Union

import numpy as np

import tiledb.libtiledb as lt

from .array import Array
from .ctx import Ctx, CtxMixin, default_ctx
from .datatypes import DataType
from .subarray import Subarray


class Query(CtxMixin, lt.Query):
    """
    Represents a TileDB query.
    """

    def __init__(
        self,
        array: Array,
        ctx: Optional[Ctx] = None,
        attrs: Optional[Union[Sequence[str], Sequence[int]]] = None,
        cond: Optional[str] = None,
        dims: Union[bool, Sequence[str]] = False,
        has_coords: bool = False,
        index_col: Optional[Union[bool, Sequence[int]]] = True,
        order: Optional[str] = None,
        use_arrow: Optional[bool] = None,
        return_arrow: bool = False,
        return_incomplete: bool = False,
    ):
        """Class representing a query on a TileDB Array.

        Allows easy subarray queries of cells for an item or region of the array
        across one or more attributes. Optionally subselect over attributes, return
        dense result coordinate values, and specify a layout a result layout / cell-order.

        For write mode arrays, the Query can be used to write data with explicit control
        over submit() and finalize() operations.

        :param array: the Array object to query.
        :param ctx: the TileDB context.
        :param attrs: the attributes to subselect over.
            If attrs is None (default) all array attributes will be returned.
            Array attributes can be defined by name or by positional index.
        :param cond: the str expression to filter attributes or dimensions on. The expression must be parsable by tiledb.QueryCondition(). See help(tiledb.QueryCondition) for more details.
        :param dims: the dimensions to subselect over. If dims is False (default), no specific selection is made.
            If True, all dimensions are returned. Otherwise, specify a list of dimension names.
        :param has_coords: (deprecated) if True, return array of coordinate value (default False).
        :param index_col: For dataframe queries, override the saved index information,
            and only set specified index(es) in the final dataframe, or None.
        :param order: 'C', 'F', 'G', or 'U' (row-major, col-major, global order, unordered).
        :param use_arrow: if True, return dataframes via PyArrow if applicable.
        :param return_arrow: if True, return results as a PyArrow Table if applicable.
        :param return_incomplete: if True, initialize and return an iterable Query object over the indexed range.
            Consuming this iterable returns a result set for each TileDB incomplete query.
            If False (default), queries will be internally run to completion by resizing buffers and
            resubmitting until query is complete.
        """

        if array.mode not in ("r", "d", "w"):
            raise ValueError("array mode must be read, delete, or write mode")

        if dims not in (False, None) and has_coords == True:
            raise ValueError("Cannot pass both dims and has_coords=True to Query")

        if return_incomplete and not array.schema.sparse:
            raise lt.TileDBError(
                "Incomplete queries are only supported for sparse arrays at this time"
            )

        # reference to the array we are querying
        self._array = array

        query_type_map = {
            "r": lt.QueryType.READ,
            "d": lt.QueryType.DELETE,
            "w": lt.QueryType.WRITE,
        }
        query_type = query_type_map[array.mode]

        super().__init__(
            ctx, lt.Array(ctx if ctx is not None else default_ctx(), array), query_type
        )

        if order is None:
            if array.schema.sparse:
                order = "U"  # unordered
            else:
                order = "C"  # row-major

        layout_map = {
            "C": lt.LayoutType.ROW_MAJOR,
            "F": lt.LayoutType.COL_MAJOR,
            "G": lt.LayoutType.GLOBAL_ORDER,
            "U": lt.LayoutType.UNORDERED,
        }

        if order not in layout_map:
            raise ValueError(
                f"order must be one of {list(layout_map.keys())}, got '{order}'"
            )

        self.layout = layout_map[order]
        self._order = order

        self._dims = dims

        if dims == True or has_coords == True:
            domain = array.schema.domain
            self._dims = [domain.dim(i).name for i in range(domain.ndim)]
        elif dims:
            domain = array.schema.domain
            for dname in dims:
                if not domain.has_dim(dname):
                    raise lt.TileDBError(
                        f"Selected dimension does not exist: '{dname}'"
                    )
            self._dims = dims
        else:
            self._dims = None

        if attrs is not None:
            for name in attrs:
                if not array.schema.has_attr(name):
                    raise lt.TileDBError(f"Selected attribute does not exist: '{name}'")
        self._attrs = attrs
        self._cond = cond
        self._has_coords = has_coords
        self._index_col = index_col
        self._return_arrow = return_arrow
        self._use_arrow = use_arrow
        self._return_incomplete = return_incomplete

        if array.mode in ("r", "d") and return_arrow:
            if use_arrow is None:
                use_arrow = True
            if not use_arrow:
                raise lt.TileDBError(
                    "Cannot initialize return_arrow with use_arrow=False"
                )

    def subarray(self) -> Subarray:
        """Subarray with the ranges this query is on.

        :rtype: Subarray
        """
        return Subarray.from_pybind11(self._ctx, self._subarray)

    def __getitem__(self, selection):
        if self._return_arrow:
            raise lt.TileDBError("`return_arrow=True` requires .df indexer`")

        return self._array.subarray(
            selection,
            attrs=self._attrs,
            cond=self._cond,
            coords=self._has_coords if self._has_coords else self._dims,
            order=self._order,
        )

    def agg(self, aggs):
        """
        Calculate an aggregate operation for a given attribute. Available
        operations are sum, min, max, mean, count, and null_count (for nullable
        attributes only). Aggregates may be combined with other query operations
        such as query conditions and slicing.

        The input may be a single operation, a list of operations, or a
        dictionary with attribute mapping to a single operation or list of
        operations.

        For undefined operations on max and min, which can occur when a nullable
        attribute contains only nulled data at the given coordinates or when
        there is no data read for the given query (e.g. query conditions that do
        not match any values or coordinates that contain no data)), invalid
        results are represented as np.nan for attributes of floating point types
        and None for integer types.

        >>> import tiledb, tempfile, numpy as np
        >>> path = tempfile.mkdtemp()

        >>> with tiledb.from_numpy(path, np.arange(1, 10)) as A:
        ...     pass

        >>> # Note that tiledb.from_numpy creates anonymous attributes, so the
        >>> # name of the attribute is represented as an empty string

        >>> with tiledb.open(path, 'r') as A:
        ...     A.query().agg("sum")[:]
        45

        >>> with tiledb.open(path, 'r') as A:
        ...     A.query(cond="attr('') < 5").agg(["count", "mean"])[:]
        {'count': 9, 'mean': 2.5}

        >>> with tiledb.open(path, 'r') as A:
        ...     A.query().agg({"": ["max", "min"]})[2:7]
        {'max': 7, 'min': 3}

        :param agg: The input attributes and operations to apply aggregations on
        :returns: single value for single operation on one attribute, a dictionary
            of attribute keys associated with a single value for a single operation
            across multiple attributes, or a dictionary of attribute keys that maps
            to a dictionary of operation labels with the associated value
        """
        schema = self._array.schema
        attr_to_aggs_map = {}
        if isinstance(aggs, dict):
            attr_to_aggs_map = {
                a: (tuple([aggs[a]]) if isinstance(aggs[a], str) else tuple(aggs[a]))
                for a in aggs
            }
        elif isinstance(aggs, str):
            attrs = tuple(schema.attr(i).name for i in range(schema.nattr))
            attr_to_aggs_map = {a: (aggs,) for a in attrs}
        elif isinstance(aggs, Sequence):
            attrs = tuple(schema.attr(i).name for i in range(schema.nattr))
            attr_to_aggs_map = {a: tuple(aggs) for a in attrs}

        from .aggregation import Aggregation

        return Aggregation(self, attr_to_aggs_map)

    @property
    def array(self):
        return self._array

    @array.setter
    def array(self, value):
        self._array = value

    @property
    def attrs(self):
        """List of attributes to include in Query."""
        return self._attrs

    @attrs.setter
    def attrs(self, value):
        self._attrs = value

    @property
    def cond(self):
        """QueryCondition used to filter attributes or dimensions in Query."""
        return self._cond

    @cond.setter
    def cond(self, value):
        self._cond = value

    @property
    def dims(self):
        """List of dimensions to include in Query."""
        return self._dims

    @property
    def has_coords(self):
        """
        True if query should include (return) coordinate values.

        :rtype: bool
        """
        return self._has_coords

    @property
    def order(self):
        """Return underlying Array order."""
        return self._order

    @order.setter
    def order(self, value):
        self._order = value

    @property
    def index_col(self):
        """List of columns to set as index for dataframe queries, or None."""
        return self._index_col

    @property
    def use_arrow(self):
        return self._use_arrow

    @property
    def return_arrow(self):
        return self._return_arrow

    @property
    def return_incomplete(self):
        return self._return_incomplete

    @property
    def domain_index(self):
        """Apply Array.domain_index with query parameters."""
        return self._domain_index

    def label_index(self, labels):
        """Apply Array.label_index with query parameters."""
        from .multirange_indexing import LabelIndexer

        return LabelIndexer(self._array, tuple(labels), query=self)

    @property
    def multi_index(self):
        """Apply Array.multi_index with query parameters."""
        # Delayed to avoid circular import
        from .multirange_indexing import MultiRangeIndexer

        return MultiRangeIndexer(self._array, query=self)

    @property
    def df(self):
        """Apply Array.multi_index with query parameters and return result
        as a Pandas dataframe."""
        # Delayed to avoid circular import
        from .multirange_indexing import DataFrameIndexer

        return DataFrameIndexer(self._array, query=self, use_arrow=self._use_arrow)

    def get_stats(self, print_out=True, json=False):
        """Retrieves the stats from a TileDB query.

        :param print_out: Print string to console (default True), or return as string
        :param json: Return stats JSON object (default: False)
        """
        pyquery = self._array.pyquery
        if pyquery is None:
            return ""
        stats = self._array.pyquery.get_stats()
        if json:
            stats = json_loads(stats)
        if print_out:
            print(stats)
        else:
            return stats

    def submit(self):
        """Submit the query.

        For read/delete queries: an alias for calling the regular indexer [:].
        For write queries: submits the write query with current buffers.
        """
        if self._array.mode in ("r", "d"):
            return self[:]
        else:
            # Write mode - submit the underlying query
            return self._submit()

    def finalize(self):
        """Finalize a query."""
        super().finalize()

    def set_data(self, data):
        """Set data buffers for write queries.

        :param data: Dictionary mapping attribute/dimension names to numpy arrays,
                    or a single numpy array if the array has a single attribute.
        :raises ValueError: if array is not in write mode or invalid data provided

        Example:
            >>> import tiledb, numpy as np
            >>> with tiledb.open(uri, 'w') as A:
            ...     q = tiledb.Query(A, order='G')
            ...     q.set_data({'d1': np.array([1, 5, 10]), 'a1': np.array([100, 200, 300])})
            ...     q.submit()
            ...     q.set_data({'d1': np.array([15, 20]), 'a1': np.array([400, 500])})
            ...     q.submit()
            ...     q.finalize()
        """
        if self._array.mode != "w":
            raise ValueError("set_data() is only supported for arrays in write mode")

        schema = self._array.schema

        # Convert single array to dict
        if isinstance(data, np.ndarray):
            if schema.nattr != 1:
                raise ValueError(
                    "Single array provided but schema has multiple attributes"
                )
            data = {schema.attr(0).name: data}

        if not isinstance(data, dict):
            raise ValueError("data must be a dict or numpy array")

        # Set buffers for each attribute/dimension
        for name, buffer in data.items():
            if not isinstance(buffer, np.ndarray):
                buffer = np.array(buffer)

            # Determine ncells based on datatype
            if schema.has_attr(name):
                dtype = schema.attr(name).dtype
            elif schema.domain.has_dim(name):
                dtype = schema.domain.dim(name).dtype
            else:
                raise ValueError(f"Unknown attribute or dimension: {name}")

            ncells = DataType.from_numpy(dtype).ncells
            buffer_size = np.uint64(len(buffer) * ncells)

            self.set_data_buffer(name, buffer, buffer_size)

    def set_subarray_ranges(self, ranges):
        """Set subarray for dense array writes.

        :param ranges: List of (start, end) tuples, one per dimension.
        """

        if self._array.mode != "w":
            raise ValueError(
                "set_subarray_ranges() is only supported for arrays in write mode"
            )

        subarray = Subarray(self._array, self._ctx)
        dim_ranges = [[r] for r in ranges]
        subarray.add_ranges(dim_ranges)
        self.set_subarray(subarray)
