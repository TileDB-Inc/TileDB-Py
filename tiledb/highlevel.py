import tiledb

import numpy as np


def open(uri, mode="r", key=None, attr=None, config=None, timestamp=None, ctx=None):
    """
    Open a TileDB array at the given URI

    :param uri: any TileDB supported URI
    :param timestamp: array timestamp to open, int or None.
        See the TileDB `time traveling <https://docs.tiledb.com/main/solutions/tiledb-embedded/api-usage/reading-arrays/time-traveling>`_
        documentation for detailed functionality description.
    :param key: encryption key, str or None
    :param str mode: (default 'r') Open the array object in read 'r' or write 'w' mode
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


def empty_like(uri, arr, config=None, key=None, tile=None, ctx=None):
    """
    Create and return an empty, writeable DenseArray with schema based on
    a NumPy-array like object.

    :param uri: array URI
    :param arr: NumPy ndarray, or shape tuple
    :param config: (optional, deprecated) configuration to apply to *new* Ctx
    :param key: (optional) encryption key, if applicable
    :param tile: (optional) tiling of generated array
    :param ctx: (optional) TileDB Ctx
    :return:
    """
    ctx = _get_ctx(ctx, config)
    schema = tiledb.schema_like(arr, tile=tile, ctx=ctx)
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

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     # Creates array 'array' on disk.
    ...     with tiledb.DenseArray.from_numpy(tmp + "/array",  np.array([1.0, 2.0, 3.0])) as A:
    ...         pass
    """
    if not isinstance(array, np.ndarray):
        raise Exception("from_numpy is only currently supported for numpy.ndarray")

    return tiledb.DenseArray.from_numpy(uri, array, ctx=_get_ctx(ctx, config), **kwargs)


def array_exists(uri, isdense=False, issparse=False):
    """
    Check if arrays exists and is open-able at the given URI

    Optionally restrict to `isdense` or `issparse` array types.
    """
    try:
        with tiledb.open(uri) as a:
            if isdense:
                return not a.schema.sparse
            if issparse:
                return a.schema.sparse
            return True
    except tiledb.TileDBError:
        return False


def array_fragments(uri, ctx=None):
    """
    Creates a FragmentInfoList object, which is an ordered list of FragmentInfo
    objects, representing all fragments in the array at the URI.

    FragmentInfo objects contain fragment metadata such as the fragment URI,
    whether it is sparse or dense, timestamp, number of cells, etc.

    :param str uri: URI for the TileDB array (any supported TileDB URI)
    :param ctx: (optional) TileDB Ctx
    :return: FragmentsInfo object
    """
    return tiledb.FragmentInfoList(uri, ctx)


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
