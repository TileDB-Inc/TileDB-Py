from collections import OrderedDict

import numpy as np

import tiledb
import tiledb.cc as lt

from .array import (
    index_as_tuple,
    index_domain_subarray,
    replace_ellipsis,
    replace_scalars_slice,
)
from .libtiledb import Array


class SparseArrayImpl(Array):
    """Class representing a sparse TileDB array (internal).

    Inherits properties and methods of :py:class:`tiledb.Array`.
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if not self.schema.sparse:
            raise ValueError(f"Array at '{self.uri}' is not a sparse array")

    def __len__(self):
        raise TypeError("SparseArray length is ambiguous; use shape[0]")

    def __setitem__(self, selection, val):
        """Set / update sparse data cells

        :param tuple selection: N coordinate value arrays (dim0, dim1, ...) where N in the ndim of the SparseArray,
            The format follows numpy sparse (point) indexing semantics.
        :param val: a dictionary of nonempty array attribute values, values must able to be converted to 1-d numpy arrays.\
            if the number of attributes is one, then a 1-d numpy array is accepted.
        :type val: dict or :py:class:`numpy.ndarray`
        :raises IndexError: invalid or unsupported index selection
        :raises ValueError: value / coordinate length mismatch
        :raises: :py:exc:`tiledb.TileDBError`

        **Example:**

        >>> import tiledb, numpy as np, tempfile
        >>> # Write to multi-attribute 2D array
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     dom = tiledb.Domain(
        ...         tiledb.Dim(domain=(0, 1), tile=2, dtype=np.uint64),
        ...         tiledb.Dim(domain=(0, 1), tile=2, dtype=np.uint64))
        ...     schema = tiledb.ArraySchema(domain=dom, sparse=True,
        ...         attrs=(tiledb.Attr(name="a1", dtype=np.int64),
        ...                tiledb.Attr(name="a2", dtype=np.int64)))
        ...     tiledb.SparseArray.create(tmp + "/array", schema)
        ...     with tiledb.SparseArray(tmp + "/array", mode='w') as A:
        ...         # Write in the corner cells (0,0) and (1,1) only.
        ...         I, J = [0, 1], [0, 1]
        ...         # Write to each attribute
        ...         A[I, J] = {"a1": np.array([1, 2]),
        ...                    "a2": np.array([3, 4])}

        """
        from .libtiledb import _setitem_impl_sparse

        _setitem_impl_sparse(self, selection, val, dict())

    def __getitem__(self, selection):
        """Retrieve nonempty cell data for an item or region of the array

        :param tuple selection: An int index, slice or tuple of integer/slice objects,
            specifying the selected subarray region for each dimension of the SparseArray.
        :rtype: :py:class:`collections.OrderedDict`
        :returns: An OrderedDict is returned with dimension and attribute names as keys. \
            Nonempty attribute values are returned as Numpy 1-d arrays.
        :raises IndexError: invalid or unsupported index selection
        :raises: :py:exc:`tiledb.TileDBError`

        **Example:**

        >>> import tiledb, numpy as np, tempfile
        >>> from collections import OrderedDict
        >>> # Write to multi-attribute 2D array
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     dom = tiledb.Domain(
        ...         tiledb.Dim(name="y", domain=(0, 9), tile=2, dtype=np.uint64),
        ...         tiledb.Dim(name="x", domain=(0, 9), tile=2, dtype=np.uint64))
        ...     schema = tiledb.ArraySchema(domain=dom, sparse=True,
        ...         attrs=(tiledb.Attr(name="a1", dtype=np.int64),
        ...                tiledb.Attr(name="a2", dtype=np.int64)))
        ...     tiledb.SparseArray.create(tmp + "/array", schema)
        ...     with tiledb.SparseArray(tmp + "/array", mode='w') as A:
        ...         # Write in the twp cells (0,0) and (2,3) only.
        ...         I, J = [0, 2], [0, 3]
        ...         # Write to each attribute
        ...         A[I, J] = {"a1": np.array([1, 2]),
        ...                    "a2": np.array([3, 4])}
        ...     with tiledb.SparseArray(tmp + "/array", mode='r') as A:
        ...         # Return an OrderedDict with values and coordinates
        ...         np.testing.assert_equal(A[0:3, 0:10], OrderedDict({'a1': np.array([1, 2]),
        ...                'a2': np.array([3, 4]), 'y': np.array([0, 2], dtype=np.uint64),
        ...                 'x': np.array([0, 3], dtype=np.uint64)}))
        ...         # Return just the "x" coordinates values
        ...         A[0:3, 0:10]["x"]
        array([0, 3], dtype=uint64)

        With a floating-point array domain, index bounds are inclusive, e.g.:

        >>> # Return nonempty cells within a floating point array domain (fp index bounds are inclusive):
        >>> # A[5.0:579.9]

        """
        result = self.subarray(selection)
        for i in range(self.schema.nattr):
            attr = self.schema.attr(i)
            enum_label = attr.enum_label
            if enum_label is not None:
                values = self.enum(enum_label).values()
                if attr.isnullable:
                    data = np.array([values[idx] for idx in result[attr.name].data])
                    result[attr.name] = np.ma.array(data, mask=result[attr.name].mask)
                else:
                    result[attr.name] = np.array(
                        [values[idx] for idx in result[attr.name]]
                    )
            else:
                if attr.isnullable:
                    result[attr.name] = np.ma.array(
                        result[attr.name].data, mask=result[attr.name].mask
                    )

        return result

    def query(
        self,
        attrs=None,
        cond=None,
        dims=None,
        index_col=True,
        coords=None,
        order="U",
        use_arrow=None,
        return_arrow=None,
        return_incomplete=False,
    ):
        """
        Construct a proxy Query object for easy subarray queries of cells
        for an item or region of the array across one or more attributes.

        Optionally subselect over attributes, return dense result coordinate values,
        and specify a layout a result layout / cell-order.

        :param attrs: the SparseArray attributes to subselect over.
            If attrs is None (default) all array attributes will be returned.
            Array attributes can be defined by name or by positional index.
        :param cond: the str expression to filter attributes or dimensions on. The expression must be parsable by tiledb.QueryCondition(). See help(tiledb.QueryCondition) for more details.
        :param dims: the SparseArray dimensions to subselect over. If dims is None (default)
            then all dimensions are returned, unless coords=False.
        :param index_col: For dataframe queries, override the saved index information,
            and only set specified index(es) in the final dataframe, or None.
        :param coords: (deprecated) if True, return array of coordinate value (default False).
        :param order: 'C', 'F', or 'G' (row-major, col-major, tiledb global order)
        :param mode: "r" to read
        :param use_arrow: if True, return dataframes via PyArrow if applicable.
        :param return_arrow: if True, return results as a PyArrow Table if applicable.
        :return: A proxy Query object that can be used for indexing into the SparseArray
            over the defined attributes, in the given result layout (order).

        **Example:**

        >>> import tiledb, numpy as np, tempfile
        >>> from collections import OrderedDict
        >>> # Write to multi-attribute 2D array
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     dom = tiledb.Domain(
        ...         tiledb.Dim(name="y", domain=(0, 9), tile=2, dtype=np.uint64),
        ...         tiledb.Dim(name="x", domain=(0, 9), tile=2, dtype=np.uint64))
        ...     schema = tiledb.ArraySchema(domain=dom, sparse=True,
        ...         attrs=(tiledb.Attr(name="a1", dtype=np.int64),
        ...                tiledb.Attr(name="a2", dtype=np.int64)))
        ...     tiledb.SparseArray.create(tmp + "/array", schema)
        ...     with tiledb.SparseArray(tmp + "/array", mode='w') as A:
        ...         # Write in the twp cells (0,0) and (2,3) only.
        ...         I, J = [0, 2], [0, 3]
        ...         # Write to each attribute
        ...         A[I, J] = {"a1": np.array([1, 2]),
        ...                    "a2": np.array([3, 4])}
        ...     with tiledb.SparseArray(tmp + "/array", mode='r') as A:
        ...         np.testing.assert_equal(A.query(attrs=("a1",), coords=False, order='G')[0:3, 0:10],
        ...                    OrderedDict({'a1': np.array([1, 2])}))

        """
        if not self.isopen or self.mode not in ("r", "d"):
            raise tiledb.TileDBError(
                "SparseArray must be opened in read or delete mode"
            )

        # backwards compatibility
        _coords = coords
        if dims is False:
            _coords = False
        elif dims is None and coords is None:
            _coords = True

        return tiledb.libtiledb.Query(
            self,
            attrs=attrs,
            cond=cond,
            dims=dims,
            coords=_coords,
            index_col=index_col,
            order=order,
            use_arrow=use_arrow,
            return_arrow=return_arrow,
            return_incomplete=return_incomplete,
        )

    def read_subarray(self, subarray):
        from .main import PyQuery
        from .subarray import Subarray

        # Set layout to UNORDERED for sparse query.
        # cdef tiledb_layout_t layout = TILEDB_UNORDERED
        layout = lt.LayoutType.UNORDERED

        # Create the PyQuery and set the subarray on it.
        pyquery = PyQuery(
            self._ctx_(),
            self,
            tuple(
                [self.view_attr]
                if self.view_attr is not None
                else (attr._internal_name for attr in self.schema)
            ),
            tuple(dim.name for dim in self.schema.domain),
            layout,
            False,
        )
        pyquery.set_subarray(subarray)

        # Set the array pyquery to this pyquery and submit.
        self.pyquery = pyquery
        pyquery.submit()

        # Clean-up the results.
        result_dict = OrderedDict()
        for name, item in pyquery.results().items():
            if len(item[1]) > 0:
                arr = pyquery.unpack_buffer(name, item[0], item[1])
            else:
                arr = item[0]
                arr.dtype = (
                    self.schema.attr_or_dim_dtype(name)
                    if not self.schema.has_dim_label(name)
                    else self.schema.dim_label(name).dtype
                )
            result_dict[name if name != "__attr" else ""] = arr
        return result_dict

    def subarray(self, selection, coords=True, attrs=None, cond=None, order=None):
        """
        Retrieve dimension and data cells for an item or region of the array.

        Optionally subselect over attributes, return sparse result coordinate values,
        and specify a layout a result layout / cell-order.

        :param selection: tuple of scalar and/or slice objects
        :param cond: the str expression to filter attributes or dimensions on. The expression must be parsable by tiledb.QueryCondition(). See help(tiledb.QueryCondition) for more details.
        :param coords: if True, return array of coordinate value (default True).
        :param attrs: the SparseArray attributes to subselect over.
            If attrs is None (default) all array attributes will be returned.
            Array attributes can be defined by name or by positional index.
        :param order: 'C', 'F', or 'G' (row-major, col-major, tiledb global order)
        :returns: An OrderedDict is returned with dimension and attribute names as keys. \
            Nonempty attribute values are returned as Numpy 1-d arrays.

        **Example:**

        >>> import tiledb, numpy as np, tempfile
        >>> from collections import OrderedDict
        >>> # Write to multi-attribute 2D array
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     dom = tiledb.Domain(
        ...         tiledb.Dim(name="y", domain=(0, 9), tile=2, dtype=np.uint64),
        ...         tiledb.Dim(name="x", domain=(0, 9), tile=2, dtype=np.uint64))
        ...     schema = tiledb.ArraySchema(domain=dom, sparse=True,
        ...         attrs=(tiledb.Attr(name="a1", dtype=np.int64),
        ...                tiledb.Attr(name="a2", dtype=np.int64)))
        ...     tiledb.SparseArray.create(tmp + "/array", schema)
        ...     with tiledb.SparseArray(tmp + "/array", mode='w') as A:
        ...         # Write in the twp cells (0,0) and (2,3) only.
        ...         I, J = [0, 2], [0, 3]
        ...         # Write to each attribute
        ...         A[I, J] = {"a1": np.array([1, 2]),
        ...                    "a2": np.array([3, 4])}
        ...     with tiledb.SparseArray(tmp + "/array", mode='r') as A:
        ...         # A[0:3, 0:10], attribute a1, row-major without coordinates
        ...         np.testing.assert_equal(
        ...                    A.subarray((slice(0, 3), slice(0, 10)), attrs=("a1",), coords=False, order='G'),
        ...                    OrderedDict({'a1': np.array([1, 2])})
        ...         )

        """
        from .subarray import Subarray

        if not self.isopen or self.mode not in ("r", "d"):
            raise tiledb.TileDBError("SparseArray is not opened in read or delete mode")

        layout = lt.LayoutType.UNORDERED
        if order is None or order == "U":
            layout = lt.LayoutType.UNORDERED
        elif order == "C":
            layout = lt.LayoutType.ROW_MAJOR
        elif order == "F":
            layout = lt.LayoutType.COL_MAJOR
        elif order == "G":
            layout = lt.LayoutType.GLOBAL_ORDER
        else:
            raise ValueError(
                "order must be 'C' (TILEDB_ROW_MAJOR), "
                "'F' (TILEDB_COL_MAJOR), "
                "'G' (TILEDB_GLOBAL_ORDER), "
                "or 'U' (TILEDB_UNORDERED)"
            )

        attr_names = list()

        if attrs is None:
            attr_names.extend(
                self.schema.attr(i)._internal_name for i in range(self.schema.nattr)
            )
        else:
            attr_names.extend(self.schema.attr(a)._internal_name for a in attrs)

        if coords == True:
            attr_names.extend(
                self.schema.domain.dim(i).name for i in range(self.schema.ndim)
            )
        elif coords:
            attr_names.extend(coords)

        dom = self.schema.domain
        idx = index_as_tuple(selection)
        idx = replace_ellipsis(dom.ndim, idx)
        idx, drop_axes = replace_scalars_slice(dom, idx)
        dim_ranges = index_domain_subarray(self, dom, idx)
        subarray = Subarray(self, self._ctx_())
        subarray.add_ranges([list([x]) for x in dim_ranges])
        return self._read_sparse_subarray(subarray, attr_names, cond, layout)

    def __repr__(self):
        if self.isopen:
            return f"SparseArray(uri={self.uri}, mode={self.mode}, ndim={self.schema.ndim})"
        else:
            return "SparseArray(uri={self.uri}, mode=closed)"

    def _read_sparse_subarray(self, subarray, attr_names: list, cond, layout):
        out = OrderedDict()
        # all results are 1-d vectors
        dims = np.array([1], dtype=np.intp)

        nattr = len(attr_names)

        from .main import PyQuery
        from .subarray import Subarray

        q = PyQuery(self._ctx_(), self, tuple(attr_names), tuple(), layout, False)
        self.pyquery = q

        if cond is not None and cond != "":
            from .query_condition import QueryCondition

            if isinstance(cond, str):
                q.set_cond(QueryCondition(cond))
            else:
                raise TypeError("`cond` expects type str.")

        if self.mode == "r":
            q.set_subarray(subarray)

        q.submit()

        if self.mode == "d":
            return

        results = OrderedDict()
        results = q.results()

        # collect a list of dtypes for resulting to construct array
        dtypes = list()
        for i in range(nattr):
            name, final_name = attr_names[i], attr_names[i]
            if name == "__attr":
                final_name = ""
            if self.schema._needs_var_buffer(name):
                if len(results[name][1]) > 0:  # note: len(offsets) > 0
                    arr = q.unpack_buffer(name, results[name][0], results[name][1])
                else:
                    arr = results[name][0]
                    arr.dtype = self.schema.attr_or_dim_dtype(name)
                out[final_name] = arr
            else:
                if self.schema.domain.has_dim(name):
                    el_dtype = self.schema.domain.dim(name).dtype
                else:
                    el_dtype = self.attr(name).dtype
                arr = results[name][0]

                # this is a work-around for NumPy restrictions removed in 1.16
                if el_dtype == np.dtype("S0"):
                    out[final_name] = b""
                elif el_dtype == np.dtype("U0"):
                    out[final_name] = ""
                else:
                    arr.dtype = el_dtype
                    out[final_name] = arr

            if self.schema.has_attr(final_name) and self.attr(final_name).isnullable:
                out[final_name] = np.ma.array(
                    out[final_name], mask=~results[name][2].astype(bool)
                )

        return out

    def unique_dim_values(self, dim=None):
        if dim is not None and not isinstance(dim, str):
            raise ValueError(f"Given Dimension {dim} is not a string.")

        if dim is not None and not self.domain.has_dim(dim):
            raise ValueError(f"Array does not contain Dimension '{dim}'.")

        query = self.query(attrs=[])[:]

        if dim:
            dim_values = tuple(np.unique(query[dim]))
        else:
            dim_values = OrderedDict()
            for dim in query:
                dim_values[dim] = tuple(np.unique(query[dim]))

        return dim_values
