from collections import OrderedDict

import numpy as np

import tiledb
import tiledb.libtiledb as lt

from .array import (
    Array,
    check_for_floats,
    index_as_tuple,
    index_domain_subarray,
    replace_ellipsis,
    replace_scalars_slice,
)
from .datatypes import DataType
from .query import Query
from .subarray import Subarray


class DenseArrayImpl(Array):
    """Class representing a dense TileDB array.

    Inherits properties and methods of :py:class:`tiledb.Array`.

    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if self.schema.sparse:
            raise ValueError(f"Array at {self.uri} is not a dense array")

    def __len__(self):
        return self.domain.shape[0]

    def __getitem__(self, selection):
        """Retrieve data cells for an item or region of the array.

        :param selection: An int index, ``slice``, tuple, list/numpy array/pyarrow array
            of integer/``slice`` objects, specifying the selected subarray region
            for each dimension of the DenseArray.
        :rtype: :py:class:`numpy.ndarray` or :py:class:`collections.OrderedDict`
        :returns: If the dense array has a single attribute then a Numpy array of corresponding shape/dtype \
                is returned for that attribute.  If the array has multiple attributes, a \
                :py:class:`collections.OrderedDict` is returned with dense Numpy subarrays \
                for each attribute.
        :raises IndexError: invalid or unsupported index selection
        :raises: :py:exc:`tiledb.TileDBError`

        **Example:**

        >>> import tiledb, numpy as np, tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     # Creates array 'array' on disk.
        ...     A = tiledb.from_numpy(tmp + "/array",  np.ones((100, 100)))
        ...     # Many aspects of Numpy's fancy indexing are supported:
        ...     A[1:10, ...].shape
        ...     A[1:10, 20:99].shape
        ...     A[1, 2].shape
        (9, 100)
        (9, 79)
        ()
        >>> # Subselect on attributes when reading:
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
        ...     schema = tiledb.ArraySchema(domain=dom,
        ...         attrs=(tiledb.Attr(name="a1", dtype=np.int64),
        ...                tiledb.Attr(name="a2", dtype=np.int64)))
        ...     tiledb.DenseArray.create(tmp + "/array", schema)
        ...     with tiledb.DenseArray(tmp + "/array", mode='w') as A:
        ...         A[0:10] = {"a1": np.zeros((10)), "a2": np.ones((10))}
        ...     with tiledb.DenseArray(tmp + "/array", mode='r') as A:
        ...         # Access specific attributes individually.
        ...         A[0:5]["a1"]
        ...         A[0:5]["a2"]
        array([0, 0, 0, 0, 0])
        array([1, 1, 1, 1, 1])

        """
        if self.view_attr:
            return self.subarray(selection)

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

    def __repr__(self):
        if self.isopen:
            return f"DenseArray(uri={self.uri!r}, mode={self.mode}, ndim={self.schema.ndim})"
        else:
            return f"DenseArray(uri={self.uri!r}, mode=closed)"

    def query(
        self,
        attrs=None,
        cond=None,
        dims=None,
        coords=False,
        order="C",
        use_arrow=None,
        return_arrow=False,
        return_incomplete=False,
    ):
        """Construct a proxy Query object for easy subarray queries of cells
        for an item or region of the array across one or more attributes.

        Optionally subselect over attributes, return dense result coordinate values,
        and specify a layout a result layout / cell-order.

        :param attrs: the DenseArray attributes to subselect over.
            If attrs is None (default) all array attributes will be returned.
            Array attributes can be defined by name or by positional index.
        :param cond: the str expression to filter attributes or dimensions on. The expression must be parsable by tiledb.QueryCondition(). See help(tiledb.QueryCondition) for more details.
        :param dims: the DenseArray dimensions to subselect over. If dims is None (default)
            then no dimensions are returned, unless coords=True.
        :param coords: if True, return array of coodinate value (default False).
        :param order: 'C', 'F', 'U', or 'G' (row-major, col-major, unordered, TileDB global order)
        :param mode: "r" to read (default), "d" to delete
        :param use_arrow: if True, return dataframes via PyArrow if applicable.
        :param return_arrow: if True, return results as a PyArrow Table if applicable.
        :param return_incomplete: if True, initialize and return an iterable Query object over the indexed range.
            Consuming this iterable returns a result set for each TileDB incomplete query.
            See usage example in 'examples/incomplete_iteration.py'.
            To retrieve the estimated result sizes for the query ranges, use:
            `A.query(..., return_incomplete=True)[...].est_result_size()`
            If False (default False), queries will be internally run to completion by resizing buffers and
            resubmitting until query is complete.
        :return: A proxy Query object that can be used for indexing into the DenseArray
            over the defined attributes, in the given result layout (order).

        :raises ValueError: array is not opened for reads (mode = 'r')
        :raises: :py:exc:`tiledb.TileDBError`

        **Example:**

        >>> # Subselect on attributes when reading:
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
        ...     schema = tiledb.ArraySchema(domain=dom,
        ...         attrs=(tiledb.Attr(name="a1", dtype=np.int64),
        ...                tiledb.Attr(name="a2", dtype=np.int64)))
        ...     tiledb.DenseArray.create(tmp + "/array", schema)
        ...     with tiledb.DenseArray(tmp + "/array", mode='w') as A:
        ...         A[0:10] = {"a1": np.zeros((10)), "a2": np.ones((10))}
        ...     with tiledb.DenseArray(tmp + "/array", mode='r') as A:
        ...         # Access specific attributes individually.
        ...         np.testing.assert_equal(A.query(attrs=("a1",))[0:5],
        ...                {"a1": np.zeros(5)})

        """

        if not self.isopen:
            raise tiledb.TileDBError("Array is not opened")

        if self.mode == "w":
            raise tiledb.TileDBError(
                "Write mode is not supported for queries on Dense Arrays"
            )
        elif self.mode == "d":
            raise tiledb.TileDBError(
                "Delete mode is not supported for queries on Dense Arrays"
            )
        elif self.mode != "r":
            raise tiledb.TileDBError("Invalid mode for queries on Dense Arrays")

        return Query(
            self,
            attrs=attrs,
            cond=cond,
            dims=dims,
            has_coords=coords,
            order=order,
            use_arrow=use_arrow,
            return_arrow=return_arrow,
            return_incomplete=return_incomplete,
        )

    def subarray(self, selection, attrs=None, cond=None, coords=False, order=None):
        """Retrieve data cells for an item or region of the array.

        Optionally subselect over attributes, return dense result coordinate values,
        and specify a layout a result layout / cell-order.

        :param selection: tuple of scalar and/or slice objects
        :param cond: the str expression to filter attributes or dimensions on. The expression must be parsable by tiledb.QueryCondition(). See help(tiledb.QueryCondition) for more details.
        :param coords: if True, return array of coordinate value (default False).
        :param attrs: the DenseArray attributes to subselect over.
            If attrs is None (default) all array attributes will be returned.
            Array attributes can be defined by name or by positional index.
        :param order: 'C', 'F', 'U', or 'G' (row-major, col-major, unordered, TileDB global order)
        :returns: If the dense array has a single attribute then a Numpy array of corresponding shape/dtype \
            is returned for that attribute.  If the array has multiple attributes, a \
            :py:class:`collections.OrderedDict` is returned with dense Numpy subarrays for each attribute.
        :raises IndexError: invalid or unsupported index selection
        :raises: :py:exc:`tiledb.TileDBError`

        **Example:**

        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
        ...     schema = tiledb.ArraySchema(domain=dom,
        ...         attrs=(tiledb.Attr(name="a1", dtype=np.int64),
        ...                tiledb.Attr(name="a2", dtype=np.int64)))
        ...     tiledb.DenseArray.create(tmp + "/array", schema)
        ...     with tiledb.DenseArray(tmp + "/array", mode='w') as A:
        ...         A[0:10] = {"a1": np.zeros((10)), "a2": np.ones((10))}
        ...     with tiledb.DenseArray(tmp + "/array", mode='r') as A:
        ...         # A[0:5], attribute a1, row-major without coordinates
        ...         np.testing.assert_equal(A.subarray((slice(0, 5),), attrs=("a1",), coords=False, order='C'),
        ...                 OrderedDict({'a1': np.zeros(5)}))

        """
        if not self.isopen:
            raise tiledb.TileDBError("Array is not opened")

        if self.mode == "w":
            raise tiledb.TileDBError(
                "Write mode is not supported for subarray queries on Dense Arrays"
            )
        elif self.mode == "d":
            raise tiledb.TileDBError(
                "Delete mode is not supported for subarray queries on Dense Arrays"
            )
        elif self.mode != "r":
            raise tiledb.TileDBError("Invalid mode for subarray query on Dense Array")

        layout = lt.LayoutType.UNORDERED
        if order is None or order == "C":
            layout = lt.LayoutType.ROW_MAJOR
        elif order == "F":
            layout = lt.LayoutType.COL_MAJOR
        elif order == "G":
            layout = lt.LayoutType.GLOBAL_ORDER
        elif order == "U":
            pass
        else:
            raise ValueError(
                "order must be 'C' (TILEDB_ROW_MAJOR), "
                "'F' (TILEDB_COL_MAJOR), "
                "'G' (TILEDB_GLOBAL_ORDER), "
                "or 'U' (TILEDB_UNORDERED)"
            )
        attr_names = list()
        if coords == True:
            attr_names.extend(
                self.schema.domain.dim(i).name for i in range(self.schema.ndim)
            )
        elif coords:
            attr_names.extend(coords)

        if attrs is None:
            attr_names.extend(
                self.schema.attr(i)._internal_name for i in range(self.schema.nattr)
            )
        else:
            attr_names.extend(self.schema.attr(a).name for a in attrs)

        selection = index_as_tuple(selection)
        idx = replace_ellipsis(self.schema.domain.ndim, selection)
        idx, drop_axes = replace_scalars_slice(self.schema.domain, idx)
        dim_ranges = index_domain_subarray(self, self.schema.domain, idx)
        subarray = Subarray(self, self.ctx)
        subarray.add_ranges(dim_ranges)
        # Note: we included dims (coords) above to match existing semantics
        out = self._read_dense_subarray(subarray, attr_names, cond, layout, coords)
        if any(s.step for s in idx):
            steps = tuple(slice(None, None, s.step) for s in idx)
            for k, v in out.items():
                out[k] = v.__getitem__(steps)
        if drop_axes:
            for k, v in out.items():
                out[k] = v.squeeze(axis=drop_axes)
        # attribute is anonymous, just return the result
        if not coords and self.schema.nattr == 1:
            attr = self.schema.attr(0)
            if attr.isanon:
                return out[attr._internal_name]
        if self.view_attr is not None:
            return out[self.view_attr]
        return out

    def _read_dense_subarray(
        self, subarray, attr_names: list, cond, layout, include_coords
    ):
        from .main import PyQuery

        q = PyQuery(self.ctx, self, tuple(attr_names), tuple(), layout, False)
        self.pyquery = q

        if cond is not None and cond != "":
            if isinstance(cond, str):
                from .query_condition import QueryCondition

                q.set_cond(QueryCondition(cond))
            else:
                raise TypeError("`cond` expects type str.")

        q.set_subarray(subarray)
        q.submit()
        results = OrderedDict()
        results = q.results()

        out = OrderedDict()

        output_shape = subarray.shape()

        nattr = len(attr_names)
        for i in range(nattr):
            name = attr_names[i]
            if not self.schema.domain.has_dim(name) and self.schema.attr(name).isvar:
                # for var arrays we create an object array
                dtype = object
                out[name] = q.unpack_buffer(
                    name, results[name][0], results[name][1]
                ).reshape(output_shape)
            else:
                dtype = q.buffer_dtype(name)

                # <TODO> sanity check the TileDB buffer size against schema?
                # <TODO> add assert to verify np.require doesn't copy?
                arr = results[name][0]
                arr.dtype = dtype
                if len(arr) == 0:
                    # special case: the C API returns 0 len for blank arrays
                    arr = np.zeros(output_shape, dtype=dtype)
                elif len(arr) != np.prod(output_shape):
                    raise Exception(
                        "Mismatched output array shape! (arr.shape: {}, output.shape: {}".format(
                            arr.shape, output_shape
                        )
                    )

                if layout == lt.LayoutType.ROW_MAJOR:
                    arr.shape = output_shape
                    arr = np.require(arr, requirements="C")
                elif layout == lt.LayoutType.COL_MAJOR:
                    arr.shape = output_shape
                    arr = np.require(arr, requirements="F")
                else:
                    arr.shape = np.prod(output_shape)

                out[name] = arr

            if self.schema.has_attr(name) and self.attr(name).isnullable:
                out[name] = np.ma.array(out[name], mask=~results[name][2].astype(bool))

        return out

    def __setitem__(self, selection, val):
        """Set / update dense data cells

        :param tuple selection: An int index, slice or tuple of integer/slice objects,
            specifiying the selected subarray region for each dimension of the DenseArray.
        :param val: a dictionary of array attribute values, values must able to be converted to n-d numpy arrays.\
            if the number of attributes is one, then a n-d numpy array is accepted.
        :type val: dict or :py:class:`numpy.ndarray`
        :raises IndexError: invalid or unsupported index selection
        :raises ValueError: value / coordinate length mismatch
        :raises: :py:exc:`tiledb.TileDBError`

        **Example:**

        >>> import tiledb, numpy as np, tempfile
        >>> # Write to single-attribute 2D array
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     # Create an array initially with all zero values
        ...     with tiledb.from_numpy(tmp + "/array",  np.zeros((2, 2))) as A:
        ...         pass
        ...     with tiledb.DenseArray(tmp + "/array", mode='w') as A:
        ...         # Write to the single (anonymous) attribute
        ...         A[:] = np.array(([1,2], [3,4]))
        >>>
        >>> # Write to multi-attribute 2D array
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     dom = tiledb.Domain(
        ...         tiledb.Dim(domain=(0, 1), tile=2, dtype=np.uint64),
        ...         tiledb.Dim(domain=(0, 1), tile=2, dtype=np.uint64))
        ...     schema = tiledb.ArraySchema(domain=dom,
        ...         attrs=(tiledb.Attr(name="a1", dtype=np.int64),
        ...                tiledb.Attr(name="a2", dtype=np.int64)))
        ...     tiledb.DenseArray.create(tmp + "/array", schema)
        ...     with tiledb.DenseArray(tmp + "/array", mode='w') as A:
        ...         # Write to each attribute
        ...         A[0:2, 0:2] = {"a1": np.array(([-3, -4], [-5, -6])),
        ...                        "a2": np.array(([1, 2], [3, 4]))}

        """
        selection_tuple = (
            (selection,) if not isinstance(selection, tuple) else selection
        )
        self._setitem_impl(selection, val, dict())

    def _setitem_impl(self, selection, val, nullmaps: dict):
        """Implementation for setitem with optional support for validity bitmaps."""
        if not self.isopen or self.mode != "w":
            raise tiledb.TileDBError("DenseArray is not opened for writing")

        domain = self.domain
        idx = replace_ellipsis(domain.ndim, index_as_tuple(selection))
        idx, _drop = replace_scalars_slice(domain, idx)
        attributes = list()
        values = list()
        labels = dict()

        if isinstance(selection, Subarray):
            subarray = selection
        else:
            dim_ranges = index_domain_subarray(self, domain, idx)
            subarray = Subarray(self, self.ctx)
            subarray.add_ranges(dim_ranges)

        subarray_shape = subarray.shape()
        if isinstance(val, np.ndarray):
            try:
                np.broadcast_shapes(subarray_shape, val.shape)
            except ValueError:
                raise ValueError(
                    "shape mismatch; data dimensions do not match the domain "
                    f"given in array schema ({subarray_shape} != {val.shape})"
                )

        if isinstance(val, dict):
            # Create dictionary of label names and values
            labels = {
                name: (
                    data
                    if not type(data) is np.ndarray or data.dtype is np.dtype("O")
                    else np.ascontiguousarray(
                        data, dtype=self.schema.dim_label(name).dtype
                    )
                )
                for name, data in val.items()
                if self.schema.has_dim_label(name)
            }

            # Create list of attribute names and values
            for attr_idx in range(self.schema.nattr):
                attr = self.schema.attr(attr_idx)
                name = attr.name
                attr_val = val[name]

                attributes.append(attr._internal_name)
                # object arrays are var-len and handled later
                if type(attr_val) is np.ndarray and attr_val.dtype is not np.dtype("O"):
                    if attr.isnullable and name not in nullmaps:
                        try:
                            nullmaps[name] = ~np.ma.masked_invalid(attr_val).mask
                            attr_val = np.nan_to_num(attr_val)
                        except Exception as exc:
                            attr_val = np.asarray(attr_val)
                            nullmaps[name] = np.array(
                                [int(v is not None) for v in attr_val], dtype=np.uint8
                            )
                    attr_val = np.ascontiguousarray(attr_val, dtype=attr.dtype)

                try:
                    if attr.isvar:
                        # ensure that the value is array-convertible, for example: pandas.Series
                        attr_val = np.asarray(attr_val)
                        if attr.isnullable and name not in nullmaps:
                            nullmaps[name] = np.array(
                                [int(v is not None) for v in attr_val], dtype=np.uint8
                            )
                    else:
                        if np.issubdtype(attr.dtype, np.bytes_) and not (
                            np.issubdtype(attr_val.dtype, np.bytes_)
                            or attr_val.dtype == np.dtype("O")
                        ):
                            raise ValueError(
                                "Cannot write a string value to non-string "
                                "typed attribute '{}'!".format(name)
                            )

                        if attr.isnullable and name not in nullmaps:
                            try:
                                nullmaps[name] = ~np.ma.masked_invalid(attr_val).mask
                            except Exception as exc:
                                attr_val = np.asarray(attr_val)
                                nullmaps[name] = np.array(
                                    [int(v is not None) for v in attr_val],
                                    dtype=np.uint8,
                                )

                            if np.issubdtype(attr.dtype, np.bytes_):
                                attr_val = np.array(
                                    ["" if v is None else v for v in attr_val]
                                )
                            else:
                                attr_val = np.nan_to_num(attr_val)
                                attr_val = np.array(
                                    [0 if v is None else v for v in attr_val]
                                )
                        attr_val = np.ascontiguousarray(attr_val, dtype=attr.dtype)
                except Exception as exc:
                    raise ValueError(
                        f"NumPy array conversion check failed for attr '{name}'"
                    ) from exc

                values.append(attr_val)

        elif np.isscalar(val):
            for i in range(self.schema.nattr):
                attr = self.schema.attr(i)
                attributes.append(attr._internal_name)
                A = np.empty(subarray_shape, dtype=attr.dtype)
                A[:] = val
                values.append(A)
        elif self.schema.nattr == 1:
            attr = self.schema.attr(0)
            name = attr.name
            attributes.append(attr._internal_name)
            # object arrays are var-len and handled later
            if type(val) is np.ndarray and val.dtype is not np.dtype("O"):
                val = np.ascontiguousarray(val, dtype=attr.dtype)
            try:
                if attr.isvar:
                    # ensure that the value is array-convertible, for example: pandas.Series
                    val = np.asarray(val)
                    if attr.isnullable and name not in nullmaps:
                        nullmaps[name] = np.array(
                            [int(v is None) for v in val], dtype=np.uint8
                        )
                else:
                    if np.issubdtype(attr.dtype, np.bytes_) and not (
                        np.issubdtype(val.dtype, np.bytes_)
                        or val.dtype == np.dtype("O")
                    ):
                        raise ValueError(
                            "Cannot write a string value to non-string "
                            "typed attribute '{}'!".format(name)
                        )

                    if attr.isnullable and name not in nullmaps:
                        nullmaps[name] = ~np.ma.masked_invalid(val).mask
                        val = np.nan_to_num(val)
                    val = np.ascontiguousarray(val, dtype=attr.dtype)
            except Exception as exc:
                raise ValueError(
                    f"NumPy array conversion check failed for attr '{name}'"
                ) from exc
            values.append(val)
        elif self.view_attr is not None:
            # Support single-attribute assignment for multi-attr array
            # This is a hack pending
            #   https://github.com/TileDB-Inc/TileDB/issues/1162
            # (note: implicitly relies on the fact that we treat all arrays
            #  as zero initialized as long as query returns TILEDB_OK)
            # see also: https://github.com/TileDB-Inc/TileDB-Py/issues/128
            if self.schema.nattr == 1:
                attributes.append(self.schema.attr(0).name)
                values.append(val)
            else:
                dtype = self.schema.attr(self.view_attr).dtype
                with DenseArrayImpl(
                    self.uri, "r", ctx=tiledb.Ctx(self.ctx.config())
                ) as readable:
                    current = readable[selection]
                current[self.view_attr] = np.ascontiguousarray(val, dtype=dtype)
                # `current` is an OrderedDict
                attributes.extend(current.keys())
                values.extend(current.values())
        else:
            raise ValueError(
                "ambiguous attribute assignment, "
                "more than one array attribute "
                "(use a dict({'attr': val}) to "
                "assign multiple attributes)"
            )

        if nullmaps:
            for key, val in nullmaps.items():
                if not self.schema.has_attr(key):
                    raise tiledb.TileDBError(
                        "Cannot set validity for non-existent attribute."
                    )
                if not self.schema.attr(key).isnullable:
                    raise ValueError(
                        "Cannot set validity map for non-nullable attribute."
                    )
                if not isinstance(val, np.ndarray):
                    raise TypeError(
                        f"Expected NumPy array for attribute '{key}' "
                        f"validity bitmap, got {type(val)}"
                    )

        self._write_array(subarray, [], attributes, values, labels, nullmaps, False)

    def __array__(self, dtype=None, **kw):
        """Implementation of numpy __array__ protocol (internal).

        :return: Numpy ndarray resulting from indexing the entire array.

        """
        if self.view_attr is None and self.nattr > 1:
            raise ValueError(
                "cannot call __array__ for TileDB array with more than one attribute"
            )
        if self.view_attr:
            name = self.view_attr
        else:
            name = self.schema.attr(0).name
        array = self.read_direct(name=name)
        if dtype and array.dtype != dtype:
            return array.astype(dtype)
        return array

    def write_direct(self, array: np.ndarray, **kw):
        """
        Write directly to given array attribute with minimal checks,
        assumes that the numpy array is the same shape as the array's domain

        :param np.ndarray array: Numpy contiguous dense array of the same dtype \
            and shape and layout of the DenseArray instance
        :raises ValueError: cannot write to multi-attribute DenseArray
        :raises ValueError: array is not contiguous
        :raises: :py:exc:`tiledb.TileDBError`
        """
        append_dim = kw.pop("append_dim", None)
        mode = kw.pop("mode", "ingest")
        start_idx = kw.pop("start_idx", None)

        if not self.isopen or self.mode != "w":
            raise tiledb.TileDBError("DenseArray is not opened for writing")
        if self.schema.nattr != 1:
            raise ValueError("cannot write_direct to a multi-attribute DenseArray")
        if not array.flags.c_contiguous and not array.flags.f_contiguous:
            raise ValueError("array is not contiguous")

        use_global_order = (
            self.ctx.config().get("py.use_global_order_1d_write", False) == "true"
        )

        layout = lt.LayoutType.ROW_MAJOR
        if array.ndim == 1 and use_global_order:
            layout = lt.LayoutType.GLOBAL_ORDER
        elif array.flags.f_contiguous:
            layout = lt.LayoutType.COL_MAJOR

        range_start_idx = start_idx or 0

        subarray_ranges = np.zeros(2 * array.ndim, np.uint64)
        for n in range(array.ndim):
            subarray_ranges[n * 2] = range_start_idx
            subarray_ranges[n * 2 + 1] = array.shape[n] + range_start_idx - 1

        if mode == "append":
            with Array.load_typed(self.uri) as A:
                ned = A.nonempty_domain()

            if array.ndim <= append_dim:
                raise IndexError("`append_dim` out of range")

            if array.ndim != len(ned):
                raise ValueError(
                    "The number of dimension of the TileDB array and "
                    "Numpy array to append do not match"
                )

            for n in range(array.ndim):
                if n == append_dim:
                    if start_idx is not None:
                        range_start_idx = start_idx
                        range_end_idx = array.shape[n] + start_idx - 1
                    else:
                        range_start_idx = ned[n][1] + 1
                        range_end_idx = array.shape[n] + ned[n][1]

                    subarray_ranges[n * 2] = range_start_idx
                    subarray_ranges[n * 2 + 1] = range_end_idx
                else:
                    if array.shape[n] != ned[n][1] - ned[n][0] + 1:
                        raise ValueError(
                            "The input Numpy array must be of the same "
                            "shape as the TileDB array, exluding the "
                            "`append_dim`, but the Numpy array at index "
                            f"{n} has {array.shape[n]} dimension(s) and "
                            f"the TileDB array has {ned[n][1]-ned[n][0]}."
                        )

        ctx = lt.Context(self.ctx)
        q = lt.Query(ctx, self.array, lt.QueryType.WRITE)
        q.layout = layout

        subarray = lt.Subarray(ctx, self.array)
        for n in range(array.ndim):
            subarray._add_dim_range(
                n, (subarray_ranges[n * 2], subarray_ranges[n * 2 + 1])
            )
        q.set_subarray(subarray)

        attr = self.schema.attr(0)
        battr_name = attr._internal_name.encode("UTF-8")

        tiledb_type = DataType.from_numpy(array.dtype)

        if tiledb_type in (lt.DataType.BLOB, lt.DataType.CHAR, lt.DataType.STRING_UTF8):
            q.set_data_buffer(battr_name, array, array.nbytes)
        else:
            q.set_data_buffer(battr_name, array, tiledb_type.ncells * array.size)

        q._submit()
        q.finalize()

    def read_direct(self, name=None):
        """Read attribute directly with minimal overhead, returns a numpy ndarray over the entire domain

        :param str attr_name: read directly to an attribute name (default <anonymous>)
        :rtype: numpy.ndarray
        :return: numpy.ndarray of `attr_name` values over the entire array domain
        :raises: :py:exc:`tiledb.TileDBError`

        """
        if not self.isopen or self.mode != "r":
            raise tiledb.TileDBError("DenseArray is not opened for reading")

        if name is None and self.schema.nattr != 1:
            raise ValueError(
                "read_direct with no provided attribute is ambiguous for multi-attribute arrays"
            )
        elif name is None:
            attr = self.schema.attr(0)
            attr_name = attr._internal_name
        else:
            attr = self.schema.attr(name)
            attr_name = attr._internal_name
        order = "C"
        cell_layout = lt.LayoutType.ROW_MAJOR
        if (
            self.schema.cell_order == "col-major"
            and self.schema.tile_order == "col-major"
        ):
            order = "F"
            cell_layout = lt.LayoutType.COL_MAJOR

        schema = self.schema
        domain = schema.domain

        idx = tuple(slice(None) for _ in range(domain.ndim))
        range_index = index_domain_subarray(self, domain, idx)
        subarray = Subarray(self, self.ctx)
        subarray.add_ranges(range_index)
        out = self._read_dense_subarray(
            subarray,
            [
                attr_name,
            ],
            None,
            cell_layout,
            False,
        )
        return out[attr_name]

    def read_subarray(self, subarray):

        from .main import PyQuery
        from .query import Query

        # Precompute and label ranges: this step is only needed so the attribute
        # buffer sizes are set correctly.
        ndim = self.schema.domain.ndim
        has_labels = any(subarray.has_label_range(dim_idx) for dim_idx in range(ndim))
        if has_labels:
            label_query = Query(self, self.ctx)
            label_query.set_subarray(subarray)
            label_query._submit()
            if not label_query.is_complete():
                raise tiledb.TileDBError("Failed to get dimension ranges from labels")
            result_subarray = Subarray(self, self.ctx)
            result_subarray.copy_ranges(label_query.subarray(), range(ndim))
            return self.read_subarray(result_subarray)

        # If the subarray has shape of zero, return empty result without querying.
        if subarray.shape() == 0:
            if self.view_attr is not None:
                return OrderedDict(
                    ("" if self.view_attr == "__attr" else self.view_attr),
                    np.array(
                        [],
                        self.schema.attr_or_dim_dtype(self.view_attr),
                    ),
                )
            return OrderedDict(
                ("" if attr.name == "__attr" else attr.name, np.array([], attr.dtype))
                for attr in self.schema.attrs
            )

        # Create the pyquery and set the subarray.
        layout = lt.LayoutType.ROW_MAJOR
        pyquery = PyQuery(
            self.ctx,
            self,
            tuple(
                [self.view_attr]
                if self.view_attr is not None
                else (attr._internal_name for attr in self.schema)
            ),
            tuple(),
            layout,
            False,
        )
        pyquery.set_subarray(subarray)

        # Set the array pyquery to this pyquery and submit.
        self.pyquery = pyquery
        pyquery.submit()

        # Clean-up the results:
        result_shape = subarray.shape()
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
            arr.shape = result_shape
            result_dict[name if name != "__attr" else ""] = arr

        return result_dict

    def write_subarray(self, subarray, values):
        """Set / update dense data cells

        :param subarray: a subarray object that specifies the region to write
            data to.
        :param values: a dictionary of array attribute values, values must able to be
            converted to n-d numpy arrays. If the number of attributes is one, then a
            n-d numpy array is accepted.
        :type subarray: :py:class:`tiledb.Subarray`
        :type values: dict or :py:class:`numpy.ndarray`
        :raises ValueError: value / coordinate length mismatch
        :raises: :py:exc:`tiledb.TileDBError`

        **Example:**

        >>> # Write to multi-attribute 2D array
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     dom = tiledb.Domain(
        ...         tiledb.Dim(domain=(0, 7), tile=8, dtype=np.uint64),
        ...         tiledb.Dim(domain=(0, 7), tile=8, dtype=np.uint64))
        ...     schema = tiledb.ArraySchema(domain=dom,
        ...         attrs=(tiledb.Attr(name="a1", dtype=np.int64),
        ...                tiledb.Attr(name="a2", dtype=np.int64)))
        ...     tiledb.Array.create(tmp + "/array", schema)
        ...     with tiledb.open(tmp + "/array", mode='w') as A:
        ...         subarray = tiledb.Subarray(A)
        ...         subarray.add_dim_range(0, (0, 1))
        ...         subarray.add_dim_range(1, (0, 1))
        ...         # Write to each attribute
        ...         A.write_subarray(
        ...             subarray,
        ...             {
        ...                 "a1": np.array(([-3, -4], [-5, -6])),
        ...                 "a2": np.array(([1, 2], [3, 4])),
        ...             }
        ...         )

        """
        # Check for label ranges
        for dim_idx in range(self.schema.ndim):
            if subarray.has_label_range(dim_idx):
                raise tiledb.TileDBError(
                    f"Label range on dimension {dim_idx}. Support for writing by label "
                    f"ranges has not been implemented in the Python API."
                )
        self._setitem_impl(subarray, values, dict())
