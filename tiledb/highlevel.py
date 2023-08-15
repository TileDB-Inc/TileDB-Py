import numpy as np

import tiledb

from .dataframe_ import create_dim


def open(uri, mode="r", key=None, attr=None, config=None, timestamp=None, ctx=None):
    """
    Open a TileDB array at the given URI

    :param uri: any TileDB supported URI
    :param timestamp: array timestamp to open, int or None.
        See the TileDB `time traveling <https://docs.tiledb.com/main/how-to/arrays/reading-arrays/time-traveling>`_
        documentation for detailed functionality description.
    :param key: encryption key, str or None
    :param str mode: (default 'r') Open the array object in read 'r', write 'w',  modify exclusive 'm' mode, or  delete 'd' mode
    :param attr: attribute name to select from a multi-attribute array, str or None
    :param config: TileDB config dictionary, dict or None
    :return: open TileDB {Sparse,Dense}Array object
    """
    return tiledb.Array.load_typed(
        uri,
        mode=mode,
        key=key,
        timestamp=timestamp,
        attr=attr,
        ctx=_get_ctx(ctx, config),
    )


def save(uri, array, **kwargs):
    """
    Save array-like object at the given URI.

    :param uri: str or None
    :param array: array-like object convertible to NumPy
    :param kwargs: optional keyword args will be forwarded to tiledb.Array constructor
    :return:
    """
    # TODO: deprecate this in favor of from_numpy?
    return from_numpy(uri, array, **kwargs)


def empty_like(uri, arr, config=None, key=None, tile=None, ctx=None, dtype=None):
    """
    Create and return an empty, writeable DenseArray with schema based on
    a NumPy-array like object.

    :param uri: array URI
    :param arr: NumPy ndarray, or shape tuple
    :param config: (optional, deprecated) configuration to apply to *new* Ctx
    :param key: (optional) encryption key, if applicable
    :param tile: (optional) tiling of generated array
    :param ctx: (optional) TileDB Ctx
    :param dtype: (optional) required if arr is a shape tuple
    :return:
    """
    ctx = _get_ctx(ctx, config)
    if isinstance(arr, tuple):
        if dtype is None:
            raise ValueError("dtype must be valid data type (e.g. np.int32), not None")
        schema = schema_like(shape=arr, tile=tile, ctx=ctx, dtype=dtype)
    else:
        schema = schema_like(arr, tile=tile, ctx=ctx)
    tiledb.DenseArray.create(uri, schema, key=key, ctx=ctx)
    return tiledb.DenseArray(uri, mode="w", key=key, ctx=ctx)


def from_numpy(uri, array, config=None, ctx=None, **kwargs):
    """
    Write a NumPy array into a TileDB DenseArray,
    returning a readonly DenseArray instance.

    :param str uri: URI for the TileDB array (any supported TileDB URI)
    :param numpy.ndarray array: dense numpy array to persist
    :param config: TileDB config dictionary, dict or None
    :param tiledb.Ctx ctx: A TileDB Context
    :param kwargs: additional arguments to pass to the DenseArray constructor
    :rtype: tiledb.DenseArray
    :return: An open DenseArray (read mode) with a single anonymous attribute
    :raises TypeError: cannot convert ``uri`` to unicode string
    :raises: :py:exc:`tiledb.TileDBError`

    :Keyword Arguments:

        * **full_domain** - Dimensions should be created with full range of the dtype (default: False)
        * **mode** - Creation mode, one of 'ingest' (default), 'schema_only', 'append'
        * **append_dim** - The dimension along which the Numpy array is append (default: 0).
        * **start_idx** - The starting index to append to. By default, append to the end of the existing data.
        * **timestamp** - Write TileDB array at specific timestamp.

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     # Creates array 'array' on disk.
    ...     with tiledb.from_numpy(tmp + "/array",  np.array([1.0, 2.0, 3.0])) as A:
    ...         pass
    """
    if not isinstance(array, np.ndarray):
        raise Exception("from_numpy is only currently supported for numpy.ndarray")

    ctx = _get_ctx(ctx, config)
    mode = kwargs.pop("mode", "ingest")
    timestamp = kwargs.pop("timestamp", None)

    if mode not in ("ingest", "schema_only", "append"):
        raise tiledb.TileDBError(f"Invalid mode specified ('{mode}')")

    if mode in ("ingest", "schema_only"):
        try:
            with tiledb.Array.load_typed(uri):
                raise tiledb.TileDBError(f"Array URI '{uri}' already exists!")
        except tiledb.TileDBError:
            pass

    if mode == "append":
        kwargs["append_dim"] = kwargs.get("append_dim", 0)
        if tiledb.ArraySchema.load(uri).sparse:
            raise tiledb.TileDBError("Cannot append to sparse array")

    if mode in ("ingest", "schema_only"):
        schema = _schema_like_numpy(array, ctx, **kwargs)
        tiledb.Array.create(uri, schema)

    if mode in ("ingest", "append"):
        kwargs["mode"] = mode
        with tiledb.open(uri, mode="w", ctx=ctx, timestamp=timestamp) as arr:
            # <TODO> probably need better typecheck here
            if array.dtype == object:
                arr[:] = array
            else:
                arr.write_direct(np.ascontiguousarray(array), **kwargs)

    return tiledb.DenseArray(uri, mode="r", ctx=ctx)


def array_exists(uri, isdense=False, issparse=False):
    """
    Check if arrays exists and is open-able at the given URI

    Optionally restrict to `isdense` or `issparse` array types.
    """
    # note: we can't use *only* object_type here, because it returns 'array' even if
    # no files exist in the __schema directory (eg after delete). See SC-27854
    # but we need to use it first here, or else tiledb.open below will error out if
    # the array does not exist.
    if tiledb.object_type(uri) != "array":
        return False
    try:
        with tiledb.open(uri) as a:
            if isdense:
                return not a.schema.sparse
            if issparse:
                return a.schema.sparse
            return True
    except tiledb.TileDBError as exc:
        if (
            exc.args[0]
            == "[TileDB::Array] Error: Cannot open array; Array does not exist."
        ):
            return False
        else:
            raise


def array_fragments(uri, include_mbrs=False, ctx=None):
    """
    Creates a `FragmentInfoList` object, which is an ordered list of `FragmentInfo`
    objects, representing all fragments in the array at the given URI.

    The returned object contain the following attributes:
        - `uri`: URIs of fragments
        - `version`: Fragment version of each fragment
        - `nonempty_domain`: Non-empty domain of each fragment
        - `cell_num`: Number of cells in each fragment
        - `timestamp_range`: Timestamp range of when each fragment was written
        - `sparse`: For each fragment, True if fragment is sparse, else False
        - `has_consolidated_metadata`: For each fragment, True if fragment has consolidated fragment metadata, else False
        - `unconsolidated_metadata_num`: Number of unconsolidated metadata fragments in each fragment
        - `to_vacuum`: URIs of already consolidated fragments to vacuum
        - `mbrs`: The mimimum bounding rectangle of each fragment; only present when `include_mbrs=True`

    :param str uri: URI for the TileDB array (any supported TileDB URI)
    :param bool include_mbrs: Include minimum bouding rectangles in result; this is disabled by default for optimize time and space
    :param ctx: (optional) TileDB Ctx
    :return: FragmentInfoList
    """
    return tiledb.FragmentInfoList(uri, include_mbrs, ctx)


def schema_like(*args, shape=None, dtype=None, ctx=None, **kwargs):
    """
    Return an ArraySchema corresponding to a NumPy-like object or
    `shape` and `dtype` kwargs. Users are encouraged to pass 'tile'
    and 'capacity' keyword arguments as appropriate for a given
    application.

    :param A: NumPy array-like object, or TileDB reference URI, optional
    :param tuple shape: array shape, optional
    :param dtype: array dtype, optional
    :param Ctx ctx: TileDB Ctx
    :param kwargs: additional keyword arguments to pass through, optional
    :return: tiledb.ArraySchema
    """
    ctx = _get_ctx(ctx)

    def is_ndarray_like(arr):
        return hasattr(arr, "shape") and hasattr(arr, "dtype") and hasattr(arr, "ndim")

    # support override of default dimension dtype
    dim_dtype = kwargs.pop("dim_dtype", np.uint64)
    if len(args) == 1:
        arr = args[0]
        if not is_ndarray_like(arr):
            raise ValueError("expected ndarray-like object")
        schema = _schema_like_numpy(arr, ctx, dim_dtype, tile=kwargs.pop("tile", None))
    elif shape and dtype:
        if np.issubdtype(np.bytes_, dtype):
            dtype = np.dtype("S")
        elif np.issubdtype(dtype, np.unicode_):
            dtype = np.dtype("U")

        ndim = len(shape)
        tiling = _regularize_tiling(kwargs.pop("tile", None), ndim)

        dims = []
        for d in range(ndim):
            # support smaller tile extents by kwargs
            # domain is based on full shape
            tile_extent = tiling[d] if tiling else shape[d]
            domain = (0, shape[d] - 1)
            dims.append(
                tiledb.Dim(domain=domain, tile=tile_extent, dtype=dim_dtype, ctx=ctx)
            )

        att = tiledb.Attr(dtype=dtype, ctx=ctx)
        dom = tiledb.Domain(*dims, ctx=ctx)
        schema = tiledb.ArraySchema(ctx=ctx, domain=dom, attrs=(att,), **kwargs)
    elif kwargs is not None:
        raise ValueError
    else:
        raise ValueError(
            "Must provide either ndarray-like object or 'shape' "
            "and 'dtype' keyword arguments"
        )

    return schema


def _schema_like_numpy(
    array,
    ctx,
    dim_dtype=np.uint64,
    attr_name="",
    full_domain=False,
    tile=None,
    **kwargs,
):
    """
    Internal helper function for schema_like to create array schema from
    NumPy array-like object.
    """
    # create an ArraySchema from the numpy array object
    tiling = _regularize_tiling(tile, array.ndim)
    dims = [
        create_dim(
            dtype=dim_dtype,
            values=(0, array.shape[d] - 1),
            full_domain=full_domain,
            tile=tiling[d] if tiling else array.shape[d],
            ctx=ctx,
        )
        for d in range(array.ndim)
    ]
    var = False
    if array.dtype == object:
        # for object arrays, we use the dtype of the first element
        # consistency check should be done later, if needed
        el0 = array.flat[0]
        if isinstance(el0, bytes):
            el_dtype = np.dtype("S")
            var = True
        elif isinstance(el0, str):
            el_dtype = np.dtype("U")
            var = True
        elif isinstance(el0, np.ndarray):
            if len(el0.shape) != 1:
                raise TypeError(
                    "Unsupported sub-array type for Attribute: {} "
                    "(only string arrays and 1D homogeneous NumPy arrays are supported)".format(
                        type(el0)
                    )
                )
            el_dtype = el0.dtype
        else:
            raise TypeError(
                "Unsupported sub-array type for Attribute: {} "
                "(only strings and homogeneous-typed NumPy arrays are supported)".format(
                    type(el0)
                )
            )
    else:
        el_dtype = array.dtype

    att = tiledb.Attr(dtype=el_dtype, name=attr_name, var=var, ctx=ctx)
    dom = tiledb.Domain(*dims, ctx=ctx)
    return tiledb.ArraySchema(ctx=ctx, domain=dom, attrs=(att,), **kwargs)


def _regularize_tiling(tile, ndim):
    """
    Internal helper function for schema_like and _schema_like_numpy to regularize tiling.
    """
    if not tile:
        return None

    if np.isscalar(tile):
        return tuple(int(tile) for _ in range(ndim))

    if isinstance(tile, str) or len(tile) != ndim:
        raise ValueError("'tile' must be iterable and match array dimensionality")

    return tuple(tile)


def _get_ctx(ctx=None, config=None):
    if ctx:
        if config:
            raise ValueError(
                "Received extra Ctx or Config argument: either one may be provided, but not both"
            )
    elif config:
        ctx = tiledb.Ctx(tiledb.Config(config))
    else:
        ctx = tiledb.default_ctx()
    return ctx
