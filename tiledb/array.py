import warnings
from typing import Dict, List

import numpy as np

import tiledb
import tiledb.libtiledb as lt

from .ctx import Config, Ctx, default_ctx
from .datatypes import DataType
from .domain_indexer import DomainIndexer
from .enumeration import Enumeration
from .metadata import Metadata

# Integer types supported by Python / System
_inttypes = (int, np.integer)


def _tiledb_datetime_extent(begin, end):
    """
    Returns the integer extent of a datetime range.

    :param begin: beginning of datetime range
    :type begin: numpy.datetime64
    :param end: end of datetime range
    :type end: numpy.datetime64
    :return: Extent of range, returned as an integer number of time units
    :rtype: int
    """
    extent = end - begin + 1
    date_unit = np.datetime_data(extent.dtype)[0]
    one = np.timedelta64(1, date_unit)
    # Dividing a timedelta by 1 will convert the timedelta to an integer
    return int(extent / one)


def index_as_tuple(idx):
    """Forces scalar index objects to a tuple representation"""
    if isinstance(idx, tuple):
        return idx
    return (idx,)


def replace_ellipsis(ndim: int, idx: tuple):
    """
    Replace indexing ellipsis object with slice objects to match the number
    of dimensions.
    """
    # count number of ellipsis
    n_ellip = sum(1 for i in idx if i is Ellipsis)
    if n_ellip > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    elif n_ellip == 1:
        n = len(idx)
        if (n - 1) >= ndim:
            # does nothing, strip it out
            idx = tuple(i for i in idx if i is not Ellipsis)
        else:
            # locate where the ellipse is, count the number of items to left and right
            # fill in whole dim slices up to th ndim of the array
            left = idx.index(Ellipsis)
            right = n - (left + 1)
            new_idx = idx[:left] + ((slice(None),) * (ndim - (n - 1)))
            if right:
                new_idx += idx[-right:]
            idx = new_idx
    idx_ndim = len(idx)
    if idx_ndim < ndim:
        idx += (slice(None),) * (ndim - idx_ndim)
    if len(idx) > ndim:
        raise IndexError("too many indices for array")
    return idx


def replace_scalars_slice(dom, idx: tuple):
    """Replace scalar indices with slice objects"""
    new_idx, drop_axes = [], []
    for i in range(dom.ndim):
        dim = dom.dim(i)
        dim_idx = idx[i]
        if np.isscalar(dim_idx):
            drop_axes.append(i)
            if isinstance(dim_idx, _inttypes):
                start = int(dim_idx)
                if start < 0:
                    start += int(dim.domain[1]) + 1
                stop = start + 1
            else:
                start = dim_idx
                stop = dim_idx
            new_idx.append(slice(start, stop, None))
        else:
            new_idx.append(dim_idx)
    return tuple(new_idx), tuple(drop_axes)


def check_for_floats(selection):
    """
    Check if a selection object contains floating point values

    :param selection: selection object
    :return: True if selection contains floating point values
    :rtype: bool
    """
    if isinstance(selection, float):
        return True
    if isinstance(selection, slice):
        if isinstance(selection.start, float) or isinstance(selection.stop, float):
            return True
    elif isinstance(selection, tuple):
        for s in selection:
            if check_for_floats(s):
                return True
    return False


def index_domain_subarray(array, dom, idx: tuple):
    """
    Return a numpy array representation of the tiledb subarray buffer
    for a given domain and tuple of index slices
    """
    ndim = dom.ndim
    if len(idx) != ndim:
        raise IndexError(
            "number of indices does not match domain rank: "
            "(got {!r}, expected: {!r})".format(len(idx), ndim)
        )

    subarray = list()

    for r in range(ndim):
        # extract lower and upper bounds for domain dimension extent
        dim = dom.dim(r)
        dim_dtype = dim.dtype

        if array.mode == "r" and (
            np.issubdtype(dim_dtype, np.str_) or np.issubdtype(dim_dtype, np.bytes_)
        ):
            # NED can only be retrieved in read mode
            ned = array.nonempty_domain()
            (dim_lb, dim_ub) = ned[r] if ned else (None, None)
        else:
            (dim_lb, dim_ub) = dim.domain

        dim_idx = idx[r]

        if isinstance(dim_idx, np.ndarray) or isinstance(dim_idx, list):
            if not np.issubdtype(dim_dtype, np.str_) and not np.issubdtype(
                dim_dtype, np.bytes_
            ):
                # if this is a list, convert to numpy array
                if isinstance(dim_idx, list):
                    dim_idx = np.array(dim_idx)
                subarray.append(
                    (dim_idx),
                )
            else:
                subarray.append([(x, x) for x in dim_idx])
            continue
        try:
            import pyarrow

            if isinstance(dim_idx, pyarrow.Array):
                if not np.issubdtype(dim_dtype, np.str_) and not np.issubdtype(
                    dim_dtype, np.bytes_
                ):
                    # this is zero copy by default
                    subarray.append(dim_idx.to_numpy())
                else:
                    # zero copy is not supported for string types
                    subarray.append(
                        [(x, x) for x in dim_idx.to_numpy(zero_copy_only=False)]
                    )
                continue
        except ImportError:
            pass
        if not isinstance(dim_idx, slice):
            raise IndexError(f"invalid index type: {type(dim_idx)!r}")

        start, stop, step = dim_idx.start, dim_idx.stop, dim_idx.step

        # In the case that current domain is non-empty, we need to consider it
        if (
            hasattr(array.schema, "current_domain")
            and not array.schema.current_domain.is_empty
        ):
            if start is None:
                dim_lb = array.schema.current_domain.ndrectangle.range(r)[0]
            if stop is None:
                dim_ub = array.schema.current_domain.ndrectangle.range(r)[1]

        if np.issubdtype(dim_dtype, np.str_) or np.issubdtype(dim_dtype, np.bytes_):
            if start is None or stop is None:
                if start is None:
                    start = dim_lb
                if stop is None:
                    stop = dim_ub
            elif not isinstance(start, (str, bytes)) or not isinstance(
                stop, (str, bytes)
            ):
                raise tiledb.TileDBError(
                    f"Non-string range '({start},{stop})' provided for string dimension '{dim.name}'"
                )
            subarray.append([(start, stop)])
            continue

        if step and array.schema.sparse:
            raise IndexError("steps are not supported for sparse arrays")

        # Datetimes will be treated specially
        is_datetime = dim_dtype.kind == "M"

        # Promote to a common type
        if start is not None and stop is not None:
            if type(start) != type(stop):
                promoted_dtype = np.promote_types(type(start), type(stop))
                start = np.array(start, dtype=promoted_dtype, ndmin=1)[0]
                stop = np.array(stop, dtype=promoted_dtype, ndmin=1)[0]

        if start is not None:
            if is_datetime and not isinstance(start, np.datetime64):
                raise IndexError(
                    "cannot index datetime dimension with non-datetime interval"
                )
            # don't round / promote fp slices
            if np.issubdtype(dim_dtype, np.integer):
                if isinstance(start, (np.float32, np.float64)):
                    raise IndexError(
                        "cannot index integral domain dimension with floating point slice"
                    )
                elif not isinstance(start, _inttypes):
                    raise IndexError(
                        "cannot index integral domain dimension with non-integral slice (dtype: {})".format(
                            type(start)
                        )
                    )
            # apply negative indexing (wrap-around semantics)
            if not is_datetime and start < 0:
                start += int(dim_ub) + 1
            if start < dim_lb:
                # numpy allows start value < the array dimension shape,
                # clamp to lower bound of dimension domain
                # start = dim_lb
                raise IndexError("index out of bounds <todo>")
        else:
            start = dim_lb
        if stop is not None:
            if is_datetime and not isinstance(stop, np.datetime64):
                raise IndexError(
                    "cannot index datetime dimension with non-datetime interval"
                )
            # don't round / promote fp slices
            if np.issubdtype(dim_dtype, np.integer):
                if isinstance(start, (np.float32, np.float64)):
                    raise IndexError(
                        "cannot index integral domain dimension with floating point slice"
                    )
                elif not isinstance(start, _inttypes):
                    raise IndexError(
                        "cannot index integral domain dimension with non-integral slice (dtype: {})".format(
                            type(start)
                        )
                    )
            if not is_datetime and stop < 0:
                stop = np.int64(stop) + dim_ub
            if stop > dim_ub:
                # numpy allows stop value > than the array dimension shape,
                # clamp to upper bound of dimension domain
                if is_datetime:
                    stop = dim_ub
                else:
                    stop = int(dim_ub) + 1
        else:
            if np.issubdtype(dim_dtype, np.floating) or is_datetime:
                stop = dim_ub
            else:
                stop = int(dim_ub) + 1

        if np.issubdtype(type(stop), np.floating):
            # inclusive bounds for floating point / datetime ranges
            start = dim_dtype.type(start)
            stop = dim_dtype.type(stop)
            subarray.append([(start, stop)])
        elif is_datetime:
            # need to ensure that datetime ranges are in the units of dim_dtype
            # so that add_range and output shapes work correctly
            start = start.astype(dim_dtype)
            stop = stop.astype(dim_dtype)
            subarray.append([(start, stop)])
        elif np.issubdtype(type(stop), np.integer):
            # normal python indexing semantics
            subarray.append([(start, int(stop) - 1)])
        else:
            raise IndexError(
                "domain indexing is defined for integral and floating point values"
            )
    return subarray


# this function loads the pybind Array to determine whether it is a sparse or dense array.
def preload_array(uri, mode, key, timestamp, ctx):
    if key is not None:
        config = ctx.config()
        config["sm.encryption_key"] = key
        config["sm.encryption_type"] = "AES_256_GCM"
        ctx = tiledb.Ctx(config=config)

    _mode_to_query_type = {
        "r": lt.QueryType.READ,
        "w": lt.QueryType.WRITE,
        "m": lt.QueryType.MODIFY_EXCLUSIVE,
        "d": lt.QueryType.DELETE,
    }
    try:
        query_type = _mode_to_query_type[mode]
    except KeyError:
        raise ValueError(
            f"TileDB array mode must be one of {_mode_to_query_type.keys()}"
        )

    ts_start = None
    ts_end = None
    if timestamp is not None:
        if isinstance(timestamp, tuple):
            if len(timestamp) != 2 and not (
                isinstance(timestamp[0], int) and isinstance(timestamp[1], int)
            ):
                raise ValueError(
                    "'timestamp' argument expects either int or tuple(start: int, end: int)"
                )
            ts_start, ts_end = timestamp
        elif isinstance(timestamp, int):
            # handle the existing behavior for unary timestamp
            # which is equivalent to endpoint of the range
            ts_end = timestamp
        else:
            raise TypeError("Unexpected argument type for 'timestamp' keyword argument")

    return lt.Array(ctx, uri, query_type, (ts_start, ts_end))


class Array:
    """Base class for TileDB array objects.

    Defines common properties/functionality for the different array types. When
    an Array instance is initialized, the array is opened with the specified mode.

    :param str uri: URI of array to open
    :param str mode: (default 'r') Open the array object in read 'r', write 'w', modify exclusive 'm', or delete 'd' mode
    :param str key: (default None) If not None, encryption key to decrypt the array
    :param tuple timestamp: (default None) If int, open the array at a given TileDB
        timestamp. If tuple, open at the given start and end TileDB timestamps.
    :param str attr: (default None) open one attribute of the array; indexing a
        dense array will return a Numpy ndarray directly rather than a dictionary.
    :param Ctx ctx: TileDB context
    """

    def __init__(
        self, uri, mode="r", key=None, timestamp=None, attr=None, ctx=None, **kwargs
    ):
        if ctx is None:
            ctx = default_ctx()

        if "preloaded_array" in kwargs:
            self.array = kwargs.get("preloaded_array")
        else:
            self.array = preload_array(uri, mode, key, timestamp, ctx)

        # view on a single attribute
        schema = self.array._schema()
        if attr is not None and not schema._has_attribute(attr):
            self.array._close()
            raise KeyError(f"No attribute matching '{attr}'")
        else:
            self.view_attr = attr

        self.ctx = ctx
        self.key = key
        self.uri = uri
        self.domain_index = DomainIndexer(self)
        self.pyquery = None
        self.__buffers = None

        self.last_fragment_info = dict()
        self._meta = Metadata(self.array)

    def __capsule__(self):
        return self.array.__capsule__()

    def __repr__(self):
        if self.isopen:
            array_type = "Sparse" if self.schema.sparse else "Dense"
            return f"Array(type={array_type }, uri={self.uri!r}, mode={self.mode}, ndim={self.schema.ndim})"
        else:
            return f"Array(uri={self.uri!r}, mode=closed)"

    @classmethod
    def create(cls, uri, schema, key=None, overwrite=False, ctx=None):
        """Creates a TileDB Array at the given URI

        :param str uri: URI at which to create the new empty array.
        :param ArraySchema schema: Schema for the array
        :param str key: (default None) Encryption key to use for array
        :param bool overwrite: (default False) Overwrite the array if it already exists
        :param Ctx ctx: (default None) Optional TileDB Ctx used when creating the array,
                        by default uses the ArraySchema's associated context
                        (*not* necessarily ``tiledb.default_ctx``).

        """
        if ctx is not None and not isinstance(ctx, Ctx):
            raise TypeError(
                "tiledb.Array.create() expected tiledb.Ctx " "object to argument ctx"
            )
        ctx = ctx or default_ctx()

        from .dense_array import DenseArrayImpl
        from .sparse_array import SparseArrayImpl

        if issubclass(cls, DenseArrayImpl) and schema.sparse:
            raise ValueError(
                "Array.create `schema` argument must be a dense schema for DenseArray and subclasses"
            )
        if issubclass(cls, SparseArrayImpl) and not schema.sparse:
            raise ValueError(
                "Array.create `schema` argument must be a sparse schema for SparseArray and subclasses"
            )

        config = tiledb.Config(ctx.config())
        if key is not None:
            config["sm.encryption_type"] = "AES_256_GCM"
            config["sm.encryption_key"] = key
            ctx = tiledb.Ctx(config)

        if overwrite:
            from .highlevel import object_type
            from .vfs import VFS

            if object_type(uri) == "array":
                if uri.startswith("file://") or "://" not in uri:
                    try:
                        VFS(config=config, ctx=ctx).remove_dir(uri)
                    except tiledb.TileDBError as e:
                        raise tiledb.TileDBError(
                            f"Error removing existing array at '{uri}': {e}"
                        )
                else:
                    raise TypeError("Cannot overwrite non-local array.")
            else:
                warnings.warn("Overwrite set, but array does not exist")

        lt.Array._create(ctx, uri, schema)

    @classmethod
    def load_typed(cls, uri, mode="r", key=None, timestamp=None, attr=None, ctx=None):
        """Return a {Dense,Sparse}Array instance from a pre-opened Array (internal)"""
        if ctx is None:
            ctx = default_ctx()

        tmp_array = preload_array(uri, mode, key, timestamp, ctx)

        if tmp_array._schema()._array_type == lt.ArrayType.SPARSE:
            return tiledb.SparseArray(
                uri, mode, key, timestamp, attr, ctx, preloaded_array=tmp_array
            )
        else:
            return tiledb.DenseArray(
                uri, mode, key, timestamp, attr, ctx, preloaded_array=tmp_array
            )

    def __enter__(self):
        """
        The `__enter__` and `__exit__` methods allow TileDB arrays to be opened (and auto-closed)
        using `with tiledb.open(uri) as A:` syntax.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        The `__enter__` and `__exit__` methods allow TileDB arrays to be opened (and auto-closed)
        using `with tiledb.open(uri) as A:` syntax.
        """
        self.close()

    def close(self):
        """Closes this array, flushing all buffered data."""
        self.array._close()

    def reopen(self):
        """
        Reopens this array.

        This is useful when the array is updated after it was opened.
        To sync-up with the updates, the user must either close the array and open again,
        or just use ``reopen()`` without closing. ``reopen`` will be generally faster than
        a close-then-open.
        """
        self.array._reopen()

    @property
    def schema(self):
        """The :py:class:`ArraySchema` for this array."""
        schema = self.array._schema()
        if schema is None:
            raise tiledb.TileDBError("Cannot access schema, array is closed")
        from .array_schema import ArraySchema

        return ArraySchema.from_pybind11(self.ctx, schema)

    @property
    def mode(self):
        """The mode this array was opened with."""
        mode = self.array._query_type()
        _query_type_to_mode = {
            lt.QueryType.READ: "r",
            lt.QueryType.WRITE: "w",
            lt.QueryType.MODIFY_EXCLUSIVE: "m",
            lt.QueryType.DELETE: "d",
        }
        return _query_type_to_mode[mode]

    @property
    def iswritable(self):
        """This array is currently opened as writable."""
        return self.mode == "w"

    @property
    def isopen(self):
        """True if this array is currently open."""
        return self.array._is_open()

    @property
    def ndim(self):
        """The number of dimensions of this array."""
        return self.schema.ndim

    @property
    def domain(self):
        """The :py:class:`Domain` of this array."""
        return self.schema.domain

    @property
    def dtype(self):
        """The NumPy dtype of the specified attribute"""
        if self.view_attr is None and self.schema.nattr > 1:
            raise NotImplementedError("Multi-attribute does not have single dtype!")
        return self.schema.attr(0).dtype

    @property
    def shape(self):
        """The shape of this array."""
        return self.schema.shape

    @property
    def nattr(self):
        """The number of attributes of this array."""
        return 1 if self.view_attr is not None else self.schema.nattr

    def view_attr(self):
        """The view attribute of this array."""
        return self.view_attr

    @property
    def timestamp_range(self):
        """Returns the timestamp range the array is opened at

        :rtype: tuple
        :returns: tiledb timestamp range at which point the array was opened

        """
        timestamp_start = self.array._open_timestamp_start
        timestamp_end = self.array._open_timestamp_end

        return (timestamp_start, timestamp_end)

    @property
    def meta(self) -> Metadata:
        """
        :return: The Array's metadata as a key-value structure
        :rtype: Metadata
        """
        return self._meta

    def subarray(self, selection, attrs=None, coords=False, order=None):
        raise NotImplementedError()

    def attr(self, key):
        """Returns an :py:class:`Attr` instance given an int index or string label

        :param key: attribute index (positional or associative)
        :type key: int or str
        :rtype: :py:class:`Attr`
        :return: The array attribute at index or with the given name (label)
        :raises TypeError: invalid key type"""
        return self.schema.attr(key)

    @property
    def attr_names(self):
        """Returns a list of attribute names"""
        return self.schema.attr_names

    def dim(self, dim_id):
        """Returns a :py:class:`Dim` instance given a dim index or name

        :param key: attribute index (positional or associative)
        :type key: int or str
        :rtype: :py:class:`Attr`
        :return: The array attribute at index or with the given name (label)
        :raises TypeError: invalid key type"""
        return self.schema.domain.dim(dim_id)

    def enum(self, name):
        """
        Return the Enumeration from the attribute name.

        :param name: attribute name
        :type key: str
        :rtype: `Enumeration`
        """
        enum = self.array._get_enumeration(self.ctx, name)
        return Enumeration.from_pybind11(self.ctx, enum)

    @staticmethod
    def delete_fragments(uri, timestamp_start, timestamp_end, ctx=None):
        """
        Delete a range of fragments from timestamp_start to timestamp_end.
        The array needs to be opened in 'm' mode as shown in the example below.

        :param timestamp_start: the first fragment to delete in the range
        :type timestamp_start: int
        :param timestamp_end: the last fragment to delete in the range
        :type timestamp_end: int

        **Example:**

        >>> import tiledb, tempfile, numpy as np
        >>> path = tempfile.mkdtemp()

        >>> with tiledb.open(path, 'w', timestamp=2) as A:
        ...     A[:] = np.ones(4, dtype=np.int64)

        >>> with tiledb.open(path, 'r') as A:
        ...     A[:]
        array([1., 1., 1., 1.])

        >>> tiledb.Array.delete_fragments(path, 2, 2)

        >>> with tiledb.open(path, 'r') as A:
        ...     A[:]
        array([0., 0., 0., 0.])

        """
        if isinstance(uri, str):
            lt.Array._delete_fragments(
                ctx or default_ctx(), uri, timestamp_start, timestamp_end
            )
        else:
            raise TypeError("uri must be a string")

    @staticmethod
    def delete_array(uri, ctx=None):
        """
        Delete the given array.

        :param str uri: The URI of the array
        :param Ctx ctx: TileDB context

        **Example:**

        >>> import tiledb, tempfile, numpy as np
        >>> path = tempfile.mkdtemp()

        >>> with tiledb.from_numpy(path, np.zeros(4), timestamp=1) as A:
        ...     pass
        >>> tiledb.array_exists(path)
        True

        >>> tiledb.Array.delete_array(path)

        >>> tiledb.array_exists(path)
        False

        """
        lt.Array._delete_array(ctx or default_ctx(), uri)

    def nonempty_domain(self):
        """Return the minimum bounding domain which encompasses nonempty values.

        :rtype: tuple(tuple(numpy scalar, numpy scalar), ...)
        :return: A list of (inclusive) domain extent tuples, that contain all
            nonempty cells

        """
        results = list()
        dom = self.schema.domain

        for dim_idx in range(dom.ndim):
            dim_dtype = dom.dim(dim_idx).dtype

            if self.array._non_empty_domain_is_empty(dim_idx, dim_dtype, self.ctx):
                results.append((None, None))
                continue

            res_x, res_y = self.array._non_empty_domain(dim_idx, dim_dtype)

            # convert to bytes if needed
            if dim_dtype == np.bytes_:
                if not isinstance(res_x, bytes):
                    res_x = bytes(res_x, encoding="utf8")
                if not isinstance(res_y, bytes):
                    res_y = bytes(res_y, encoding="utf8")

            if np.issubdtype(dim_dtype, np.datetime64):
                # convert to np.datetime64
                date_unit = np.datetime_data(dim_dtype)[0]
                res_x = np.datetime64(res_x, date_unit)
                res_y = np.datetime64(res_y, date_unit)

            results.append((res_x, res_y))

        # if result only contains (None, None) tuples just return None
        for x in results:
            if x != (None, None):
                return tuple(results)

        return None

    def consolidate(self, config=None, key=None, fragment_uris=None, timestamp=None):
        """
        Consolidates fragments of an array object for increased read performance.

        Overview: https://docs.tiledb.com/main/concepts/internal-mechanics/consolidation

        :param tiledb.Config config: The TileDB Config with consolidation parameters set
        :param key: (default None) encryption key to decrypt an encrypted array
        :type key: str or bytes
        :param fragment_uris: (default None) Consolidate the array using a list of fragment _names_ (note: the `__ts1_ts2_<label>_<ver>` fragment name form alone, not the full path(s))
        :param timestamp: (default None) If not None, consolidate the array using the given tuple(int, int) UNIX seconds range (inclusive). This argument will be ignored if `fragment_uris` is passed.
        :type timestamp: tuple (int, int)
        :raises: :py:exc:`tiledb.TileDBError`

        Rather than passing the timestamp into this function, it may be set with
        the config parameters `"sm.vacuum.timestamp_start"`and
        `"sm.vacuum.timestamp_end"` which takes in a time in UNIX seconds. If both
        are set then this function's `timestamp` argument will be used.

        """
        if self.mode == "r":
            raise tiledb.TileDBError(
                "cannot consolidate array opened in readonly mode (mode='r')"
            )

        config = tiledb.Config(config or self.ctx.config())

        if key is not None:
            config["sm.encryption_type"] = "AES_256_GCM"
            config["sm.encryption_key"] = key

        if fragment_uris is not None:
            if timestamp is not None:
                warnings.warn(
                    "The `timestamp` argument will be ignored and only fragments "
                    "passed to `fragment_uris` will be consolidate",
                    DeprecationWarning,
                )
            lt.Array._consolidate(self.uri, self.ctx, fragment_uris, config)
            return
        elif timestamp is not None:
            warnings.warn(
                "The `timestamp` argument is deprecated; pass a list of "
                "fragment URIs to consolidate with `fragment_uris`",
                DeprecationWarning,
            )

            if (
                not isinstance(timestamp, tuple)
                and len(timestamp) != 2
                and not (
                    isinstance(timestamp[0], int) and isinstance(timestamp[1], int)
                )
            ):
                raise TypeError(
                    "'timestamp' argument expects tuple(start: int, end: int)"
                )

            if timestamp[0] is not None:
                config["sm.consolidation.timestamp_start"] = timestamp[0]
            if timestamp[1] is not None:
                config["sm.consolidation.timestamp_end"] = timestamp[1]

        lt.Array._consolidate(self.uri, self.ctx, config)

    def upgrade_version(self, config=None):
        """
        Upgrades an array to the latest format version.

        :param config: (default None) Configuration parameters for the upgrade
            (`nullptr` means default, which will use the config from `ctx`).
        :raises: :py:exc:`tiledb.TileDBError`
        """

        lt.Array._upgrade_version(self.ctx, self.uri, config)

    @property
    def ptr(self):
        """Return the underlying C++ TileDB array object pointer"""
        return self.array.ptr

    def dump(self):
        print(self.schema._dump(), "\n")

    def domain_index(self):
        return self.domain_index

    @property
    def dindex(self):
        return self.domain_index

    def _write_array(
        self,
        subarray,
        coordinates: List,
        buffer_names: List,
        values: List,
        labels: Dict,
        nullmaps: Dict,
        issparse: bool,
    ):
        # used for buffer conversion (local import to avoid circularity)
        from .main import array_to_buffer

        isfortran = False
        nattr = len(buffer_names)
        nlabel = len(labels)

        # Create arrays to hold buffer sizes
        nbuffer = nattr + nlabel
        if issparse:
            nbuffer += self.schema.ndim
        buffer_sizes = np.zeros((nbuffer,), dtype=np.uint64)

        # Create lists for data and offset buffers
        output_values = list()
        output_offsets = list()

        # Set data and offset buffers for attributes
        for i in range(nattr):
            attr = self.schema.attr(i)
            # if dtype is ASCII, ensure all characters are valid
            if attr.isascii:
                try:
                    values[i] = np.asarray(values[i], dtype=np.bytes_)
                except Exception as exc:
                    raise tiledb.TileDBError(
                        f'dtype of attr {attr.name} is "ascii" but attr_val contains invalid ASCII characters'
                    )

            if attr.isvar:
                try:
                    if attr.isnullable:
                        if np.issubdtype(attr.dtype, np.str_) or np.issubdtype(
                            attr.dtype, np.bytes_
                        ):
                            attr_val = np.array(
                                ["" if v is None else v for v in values[i]]
                            )
                        else:
                            attr_val = np.nan_to_num(values[i])
                    else:
                        attr_val = values[i]
                    buffer, offsets = array_to_buffer(attr_val, True, False)
                except Exception as exc:
                    raise type(exc)(
                        f"Failed to convert buffer for attribute: '{attr.name}'"
                    ) from exc
            else:
                buffer, offsets = values[i], None

            buffer_sizes[i] = buffer.nbytes // (attr.dtype.itemsize or 1)
            output_values.append(buffer)
            output_offsets.append(offsets)

        # Check value layouts
        if len(values) and nattr > 1:
            value = output_values[0]
            isfortran = value.ndim > 1 and value.flags.f_contiguous
            for value in values:
                if value.ndim > 1 and value.flags.f_contiguous and not isfortran:
                    raise ValueError("mixed C and Fortran array layouts")

        # Set data and offsets buffers for dimensions (sparse arrays only)
        ibuffer = nattr
        if issparse:
            for dim_idx, coords in enumerate(coordinates):
                dim = self.schema.domain.dim(dim_idx)
                if dim.isvar:
                    buffer, offsets = array_to_buffer(coords, True, False)
                else:
                    buffer, offsets = coords, None
                buffer_sizes[ibuffer] = buffer.nbytes // (dim.dtype.itemsize or 1)
                output_values.append(buffer)
                output_offsets.append(offsets)

                name = dim.name
                buffer_names.append(name)

                ibuffer = ibuffer + 1

        for label_name, label_values in labels.items():
            # Append buffer name
            buffer_names.append(label_name)
            # Get label data buffer and offsets buffer for the labels
            dim_label = self.schema.dim_label(label_name)
            if dim_label.isvar:
                buffer, offsets = array_to_buffer(label_values, True, False)
            else:
                buffer, offsets = label_values, None
            buffer_sizes[ibuffer] = buffer.nbytes // (dim_label.dtype.itemsize or 1)
            # Append the buffers
            output_values.append(buffer)
            output_offsets.append(offsets)

            ibuffer = ibuffer + 1

        # Allocate the query
        ctx = lt.Context(self.ctx)
        q = lt.Query(ctx, self.array, lt.QueryType.WRITE)

        # Set the layout
        q.layout = (
            lt.LayoutType.UNORDERED
            if issparse
            else (lt.LayoutType.COL_MAJOR if isfortran else lt.LayoutType.ROW_MAJOR)
        )

        # Create and set the subarray for the query (dense arrays only)
        if not issparse:
            q.set_subarray(subarray)

        # Set buffers on the query
        for i, buffer_name in enumerate(buffer_names):
            # Set data buffer
            ncells = DataType.from_numpy(output_values[i].dtype).ncells
            q.set_data_buffer(
                buffer_name,
                output_values[i],
                np.uint64(buffer_sizes[i] * ncells),
            )

            # Set offsets buffer
            if output_offsets[i] is not None:
                output_offsets[i] = output_offsets[i].astype(np.uint64)
                q.set_offsets_buffer(
                    buffer_name, output_offsets[i], output_offsets[i].size
                )

            # Set validity buffer
            if buffer_name in nullmaps:
                nulmap = nullmaps[buffer_name]
                q.set_validity_buffer(buffer_name, nulmap, nulmap.size)

        q._submit()
        q.finalize()

        fragment_info = self.last_fragment_info
        if fragment_info != False:
            if not isinstance(fragment_info, dict):
                raise ValueError(
                    f"Expected fragment_info to be a dict, got {type(fragment_info)}"
                )
            fragment_info.clear()

            result = dict()
            num_fragments = q.fragment_num()

            if num_fragments < 1:
                return result

            for fragment_idx in range(0, num_fragments):
                fragment_uri = q.fragment_uri(fragment_idx)
                fragment_t1, fragment_t2 = q.fragment_timestamp_range(fragment_idx)
                result[fragment_uri] = (fragment_t1, fragment_t2)

            fragment_info.update(result)

    def label_index(self, labels):
        """Retrieve data cells with multi-range, domain-inclusive indexing by label.
        Returns the cross-product of the ranges.

        Accepts a scalar, ``slice``, or list of scalars per-label for querying on the
        corresponding dimensions. For multidimensional arrays querying by labels only on
        a subset of dimensions, ``:`` should be passed in-place for any labels preceeding
        custom ranges.

        ** Example **

        >>> import tiledb, numpy as np, tempfile
        >>> from collections import OrderedDict
        >>> dim1 = tiledb.Dim("d1", domain=(1, 4))
        >>> dim2 = tiledb.Dim("d2", domain=(1, 3))
        >>> dom = tiledb.Domain(dim1, dim2)
        >>> att = tiledb.Attr("a1", dtype=np.int64)
        >>> dim_labels = {
        ...     0: {"l1": dim1.create_label_schema("decreasing", np.int64)},
        ...     1: {
        ...         "l2": dim2.create_label_schema("increasing", np.int64),
        ...         "l3": dim2.create_label_schema("increasing", np.float64),
        ...     },
        ... }
        >>> schema = tiledb.ArraySchema(domain=dom, attrs=(att,), dim_labels=dim_labels)
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     tiledb.Array.create(tmp, schema)
        ...
        ...     a1_data = np.reshape(np.arange(1, 13), (4, 3))
        ...     l1_data = np.arange(4, 0, -1)
        ...     l2_data = np.arange(-1, 2)
        ...     l3_data = np.linspace(0, 1.0, 3)
        ...
        ...     with tiledb.open(tmp, "w") as A:
        ...         A[:] = {"a1": a1_data, "l1": l1_data, "l2": l2_data, "l3": l3_data}
        ...
        ...     with tiledb.open(tmp, "r") as A:
        ...         np.testing.assert_equal(
        ...             A.label_index(["l1"])[3:4],
        ...             OrderedDict({"l1": [4, 3], "a1": [[1, 2, 3], [4, 5, 6]]}),
        ...         )
        ...         np.testing.assert_equal(
        ...             A.label_index(["l1", "l3"])[2, 0.5:1.0],
        ...             OrderedDict(
        ...                 {"l3": [0.5, 1.0], "l1": [2], "a1": [[8, 9]]}
        ...             ),
        ...         )
        ...         np.testing.assert_equal(
        ...             A.label_index(["l2"])[:, -1:0],
        ...             OrderedDict(
        ...                 {"l2": [-1, 0],
        ...                 "a1": [[1, 2], [4, 5], [7, 8], [10, 11]]},
        ...             ),
        ...         )
        ...         np.testing.assert_equal(
        ...             A.label_index(["l3"])[:, 0.5:1.0],
        ...             OrderedDict(
        ...                 {"l3": [0.5, 1.],
        ...                 "a1": [[2, 3], [5, 6], [8, 9], [11, 12]]},
        ...             ),
        ...         )

        :param labels: List of labels to use when querying. Can only use at most one
            label per dimension.
        :param list selection: Per dimension, a scalar, ``slice``, or  list of scalars.
            Each item is iterpreted as a point (scalar) or range (``slice``) used to
            query the array on the corresponding dimension.
        :returns: dict of {'label/attribute': result}.
        :raises: :py:exc:`tiledb.TileDBError`

        """
        # Delayed to avoid circular import
        from .multirange_indexing import LabelIndexer

        return LabelIndexer(self, tuple(labels))

    @property
    def multi_index(self):
        """Retrieve data cells with multi-range, domain-inclusive indexing. Returns
        the cross-product of the ranges.

        :param list selection: Per dimension, a scalar, ``slice``,
            or a list/numpy array/pyarrow array of scalars or ``slice`` objects.
            Scalars and ``slice`` components should match the type of the underlying Dimension.
        :returns: dict of {'attribute': result}. Coords are included by default for
            Sparse arrays only (use `Array.query(coords=<>)` to select).
        :raises IndexError: invalid or unsupported index selection
        :raises: :py:exc:`tiledb.TileDBError`

        ``multi_index[]`` accepts, for each dimension, a scalar, ``slice``, or list
        of scalars or ``slice`` objects. Each item is interpreted as a point
        (scalar) or range (``slice``) used to query the array on the corresponding
        dimension.

        Unlike NumPy array indexing, ``multi_index`` respects TileDB's range semantics:
        slice ranges are *inclusive* of the start- and end-point, and negative ranges
        do not wrap around (because a TileDB dimensions may have a negative domain).

        See also: https://docs.tiledb.com/main/api-usage/reading-arrays/multi-range-subarrays

        ** Example **

        >>> import tiledb, tempfile, numpy as np
        >>>
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...    A = tiledb.from_numpy(tmp, np.eye(4) * [1,2,3,4])
        ...    A.multi_index[1]  # doctest: +ELLIPSIS
        ...    A.multi_index[1,1]  # doctest: +ELLIPSIS
        ...    # return row 0 and 2
        ...    A.multi_index[[0,2]]  # doctest: +ELLIPSIS
        ...    # return rows 0 and 2 intersecting column 2
        ...    A.multi_index[[0,2], 2]  # doctest: +ELLIPSIS
        ...    # return rows 0:2 intersecting columns 0:2
        ...    A.multi_index[slice(0,2), slice(0,2)]  # doctest: +ELLIPSIS
        OrderedDict(...''... array([[0., 2., 0., 0.]])...)
        OrderedDict(...''... array([[2.]])...)
        OrderedDict(...''... array([[1., 0., 0., 0.],
                [0., 0., 3., 0.]])...)
        OrderedDict(...''... array([[0.],
                [3.]])...)
        OrderedDict(...''... array([[1., 0., 0.],
                [0., 2., 0.],
                [0., 0., 3.]])...)

        """
        # Delayed to avoid circular import
        from .multirange_indexing import MultiRangeIndexer

        return MultiRangeIndexer(self)

    @property
    def df(self):
        """Retrieve data cells as a Pandas dataframe, with multi-range,
        domain-inclusive indexing using ``multi_index``.

        :param list selection: Per dimension, a scalar, ``slice``,
            or a list/numpy array/pyarrow array of scalars or ``slice`` objects.
            Scalars and ``slice`` components should match the type of the underlying Dimension.
        :returns: dict of {'attribute': result}. Coords are included by default for
            Sparse arrays only (use `Array.query(coords=<>)` to select).
        :raises IndexError: invalid or unsupported index selection
        :raises: :py:exc:`tiledb.TileDBError`

        ``df[]`` accepts, for each dimension, a scalar, ``slice``, or list
        of scalars or ``slice`` objects. Each item is interpreted as a point
        (scalar) or range (``slice``) used to query the array on the corresponding
        dimension.

        ** Example **

        >>> import tiledb, tempfile, numpy as np, pandas as pd
        >>>
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...    data = {'col1_f': np.arange(0.0,1.0,step=0.1), 'col2_int': np.arange(10)}
        ...    df = pd.DataFrame.from_dict(data)
        ...    tiledb.from_pandas(tmp, df)
        ...    A = tiledb.open(tmp)
        ...    A.df[1]
        ...    A.df[1:5]
              col1_f  col2_int
           1     0.1         1
              col1_f  col2_int
           1     0.1         1
           2     0.2         2
           3     0.3         3
           4     0.4         4
           5     0.5         5

        """
        # Delayed to avoid circular import
        from .multirange_indexing import DataFrameIndexer

        return DataFrameIndexer(self, use_arrow=None)

    @property
    def last_write_info(self):
        return self.last_fragment_info

    @property
    def _buffers(self):
        return self.__buffers

    def _set_buffers(self, buffers):
        """
        Helper function to set external buffers in the form of
            {'attr_name': (data_array, offsets_array)}
        Buffers will be used to satisfy the next index/query request.
        """
        self.__buffers = buffers

    def set_query(self, serialized_query):
        from .main import PyQuery

        q = PyQuery(self.ctx, self, ("",), (), 0, False)
        q.set_serialized_query(serialized_query)
        q.submit()

        from collections import OrderedDict

        results = q.results()

        out = OrderedDict()
        for name in results.keys():
            arr = results[name][0]
            arr.dtype = q.buffer_dtype(name)
            out[name] = arr
        return out

    # pickling support: this is a lightweight pickle for distributed use.
    #   simply treat as wrapper around URI, not actual data.
    def __getstate__(self):
        config_dict = self.ctx.config().dict()
        return (
            self.uri,
            self.mode,
            self.key,
            self.view_attr,
            self.timestamp_range,
            config_dict,
        )

    def __setstate__(self, state):
        uri, mode, key, view_attr, timestamp_range, config_dict = state

        if config_dict is not {}:
            config = Config(params=config_dict)
            ctx = Ctx(config)
        else:
            ctx = default_ctx()

        self.__init__(
            uri, mode=mode, key=key, attr=view_attr, timestamp=timestamp_range, ctx=ctx
        )
