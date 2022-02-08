from collections import OrderedDict
from dataclasses import dataclass
import numpy as np
from typing import Sequence

from .ctx import default_ctx
from .libtiledb import Array, Ctx, Dim, DenseArrayImpl, SparseArrayImpl, Query
from .schema import schema_like_numpy
from .typing import TileDBSequence, TileDBDictSequence

# Extensible (pure Python) array class definitions inheriting from the
# Cython implemention. The cloudarray mix-in adds optional functionality
# for registering arrays and executing functions on the

# NOTE: the mixin import must be inside the __new__ initializer because it
#       needs to be deferred. tiledb.cloud is not yet known to the importer
#       when this code is imported.
#       TODO: might be possible to work-around/simplify by using
#       import meta-hooks instead.
@dataclass
class DenseArray(DenseArrayImpl):
    """Class representing a dense TileDB array.

    Inherits properties and methods of :py:class:`tiledb.Array` and
    implements `__setitem__` and `__getitem__` for dense array indexing
    and assignment.
    """

    _mixin_init = False

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if self.schema.sparse:
            raise ValueError(f"Array at {self.uri} is not a dense array")

    def __new__(cls, *args, **kwargs):
        if not cls._mixin_init:
            # must set before importing, because import is not thread-safe
            #   https://github.com/TileDB-Inc/TileDB-Py/issues/244
            cls._mixin_init = True
            try:
                from tiledb.cloud import cloudarray

                DenseArray.__bases__ = DenseArray.__bases__ + (cloudarray.CloudArray,)
            except ImportError:
                pass

        return super().__new__(cls, *args, **kwargs)

    @staticmethod
    def from_numpy(uri: str, array: Array, ctx: Ctx = None, **kw):
        """Implementation of tiledb.from_numpy for dense arrays. See documentation
        of tiledb.from_numpy
        """
        if not ctx:
            ctx = default_ctx()

        # pop the write timestamp before creating schema
        timestamp = kw.pop("timestamp", None)

        schema = schema_like_numpy(array, ctx=ctx, **kw)
        Array.create(uri, schema)

        with DenseArray(uri, mode="w", ctx=ctx, timestamp=timestamp) as arr:
            # <TODO> probably need better typecheck here
            if array.dtype == object:
                arr[:] = array
            else:
                arr.write_direct(np.ascontiguousarray(array))
        return DenseArray(uri, mode="r", ctx=ctx)

    def __len__(self):
        return self.domain.shape[0]

    def query(
        self,
        attrs: str | int | Sequence[str | int] = None,
        attr_cond=None,
        dims: str | int | Sequence[str | int] = None,
        coords: bool = False,
        order: str = "C",
        use_arrow: bool = None,
        return_arrow: bool = False,
        return_incomplete: bool = False,
    ):
        """
        Construct a proxy Query object for easy subarray queries of cells
        for an item or region of the array across one or more attributes.

        Optionally subselect over attributes, return dense result coordinate values,
        and specify a layout a result layout / cell-order.

        :param attrs: the DenseArray attributes to subselect over.
            If attrs is None (default) all array attributes will be returned.
            Array attributes can be defined by name or by positional index.
        :param attr_cond: the QueryCondition to filter attributes on.
        :param dims: the DenseArray dimensions to subselect over. If dims is None (default)
            then no dimensions are returned, unless coords=True.
        :param coords: if True, return array of coodinate value (default False).
        :param order: 'C', 'F', 'U', or 'G' (row-major, col-major, unordered, TileDB global order)
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
        ...         A.query(attrs=("a1",))[0:5]
        OrderedDict([('a1', array([0, 0, 0, 0, 0]))])

        """
        if not self.isopen or self.mode != "r":
            raise TileDBError("DenseArray is not opened for reading")
        return Query(
            self,
            attrs=attrs,
            attr_cond=attr_cond,
            dims=dims,
            coords=coords,
            order=order,
            use_arrow=use_arrow,
            return_arrow=return_arrow,
            return_incomplete=return_incomplete,
        )

    def __setitem__(self, selection: TileDBSequence, val: TileDBDictSequence):
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
        ...     with tiledb.DenseArray.from_numpy(tmp + "/array",  np.zeros((2, 2))) as A:
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
        if any(isinstance(s, np.ndarray) for s in selection_tuple):
            warnings.warn(
                "Sparse writes to dense arrays is deprecated",
                DeprecationWarning,
            )
            self._setitem_impl_sparse(self, selection, val, dict())
            return

        self._setitem_impl(selection, val, dict())

    def __array__(self, dtype: np.dtype = None, **kw):
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


@dataclass
class SparseArray(SparseArrayImpl):
    """Class representing a sparse TileDB array.

    Inherits properties and methods of :py:class:`tiledb.Array` and
    implements `__setitem__` and `__getitem__` for sparse array indexing
    and assignment.
    """

    _mixin_init = False

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if not self.schema.sparse:
            raise ValueError("Array at '{}' is not a sparse array".format(self.uri))

        return

    def __len__(self):
        raise TypeError("SparseArray length is ambiguous; use shape[0]")

    def __new__(cls, *args, **kwargs):
        if not cls._mixin_init:
            cls._mixin_init = True
            try:
                from tiledb.cloud import cloudarray

                SparseArray.__bases__ = SparseArray.__bases__ + (cloudarray.CloudArray,)
            except ImportError:
                pass

        return super().__new__(cls, *args, **kwargs)

    def query(
        self,
        attrs: str | int | Sequence[str | int] = None,
        attr_cond=None,
        dims: str | int | Sequence[str | int] = None,
        index_col: int | Sequence[int] = True,
        coords: bool = None,
        order: str = "U",
        use_arrow: bool = None,
        return_arrow: bool = None,
        return_incomplete: bool = False,
    ):
        """
        Construct a proxy Query object for easy subarray queries of cells
        for an item or region of the array across one or more attributes.

        Optionally subselect over attributes, return dense result coordinate values,
        and specify a layout a result layout / cell-order.

        :param attrs: the SparseArray attributes to subselect over.
            If attrs is None (default) all array attributes will be returned.
            Array attributes can be defined by name or by positional index.
        :param attr_cond: the QueryCondition to filter attributes on.
        :param dims: the SparseArray dimensions to subselect over. If dims is None (default)
            then all dimensions are returned, unless coords=False.
        :param index_col: For dataframe queries, override the saved index information,
            and only set specified index(es) in the final dataframe, or None.
        :param coords: (deprecated) if True, return array of coordinate value (default False).
        :param order: 'C', 'F', or 'G' (row-major, col-major, tiledb global order)
        :param use_arrow: if True, return dataframes via PyArrow if applicable.
        :param return_arrow: if True, return results as a PyArrow Table if applicable.
        :return: A proxy Query object that can be used for indexing into the SparseArray
            over the defined attributes, in the given result layout (order).

        **Example:**

        >>> import tiledb, numpy as np, tempfile
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
        ...         A.query(attrs=("a1",), coords=False, order='G')[0:3, 0:10]
        OrderedDict([('a1', array([1, 2]))])

        """
        if not self.isopen:
            raise TileDBError("SparseArray is not opened")

        # backwards compatibility
        _coords = coords
        if dims is False:
            _coords = False
        elif dims is None and coords is None:
            _coords = True

        return Query(
            self,
            attrs=attrs,
            attr_cond=attr_cond,
            dims=dims,
            coords=_coords,
            index_col=index_col,
            order=order,
            use_arrow=use_arrow,
            return_arrow=return_arrow,
            return_incomplete=return_incomplete,
        )

    def __getitem__(self, selection: TileDBSequence):
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
        ...         A[0:3, 0:10]
        ...         # Return just the "x" coordinates values
        ...         A[0:3, 0:10]["x"]
        OrderedDict([('a1', array([1, 2])), ('a2', array([3, 4])), ('y', array([0, 2], dtype=uint64)), ('x', array([0, 3], dtype=uint64))])
        array([0, 3], dtype=uint64)

        With a floating-point array domain, index bounds are inclusive, e.g.:

        >>> # Return nonempty cells within a floating point array domain (fp index bounds are inclusive):
        >>> # A[5.0:579.9]

        """
        return self.subarray(selection)

    def unique_dim_values(self, dim: Dim = None):
        if dim is not None and not isinstance(dim, str):
            raise ValueError("Given Dimension {} is not a string.".format(dim))

        if dim is not None and not self.domain.has_dim(dim):
            raise ValueError("Array does not contain Dimension '{}'.".format(dim))

        query = self.query(attrs=[])[:]

        if dim:
            dim_values = tuple(np.unique(query[dim]))
        else:
            dim_values = OrderedDict()
            for dim in query:
                dim_values[dim] = tuple(np.unique(query[dim]))

        return dim_values
