import tiledb

import numpy as np
import os

from tiledb import fragment


def open(uri, mode="r", key=None, attr=None, config=None, timestamp=None, ctx=None):
    """
    Open a TileDB array at the given URI

    :param uri: any TileDB supported URI
    :param timestamp: array timestamp to open, int or None.
        See the TileDB `time traveling <https://docs.tiledb.com/main/how-to/arrays/reading-arrays/time-traveling>`_
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
        schema = tiledb.schema_like(shape=arr, tile=tile, ctx=ctx, dtype=dtype)
    else:
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
