import tiledb
from tiledb.libtiledb import *

import numpy as np
import warnings

def open(uri, mode='r', key=None, attr=None, config=None, timestamp=None, ctx=None):
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
    if ctx and config:
      raise ValueError("Received extra Ctx or Config argument: either one may be provided, but not both")

    if config:
        cfg = tiledb.Config(config)
        ctx = tiledb.Ctx(cfg)

    if ctx is None:
        ctx = default_ctx()

    return tiledb.Array.load_typed(uri, mode=mode, key=key, timestamp=timestamp, attr=attr, ctx=ctx)

def save(uri, array, config=None, **kw):
    """
    Save array-like object at the given URI.

    :param uri: str or None
    :param array: array-like object convertible to NumPy
    :param config: TileDB config dictionary, dict or None
    :param kw: optional keyword args will be forwarded to tiledb.Array constructor
    :return:
    """
    if not isinstance(array, np.ndarray):
        raise ValueError("expected NumPy ndarray, not '{}'".format(type(array)))
    if config:
        cfg = Config(config)
        ctx = tiledb.Ctx(cfg)
    else:
        ctx = default_ctx()

    return tiledb.from_numpy(uri, array, ctx=ctx)


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
    if config is not None:
        warnings.warn(DeprecationWarning("'config' keyword argument is deprecated; use 'ctx'"))
        cfg = tiledb.Config(config)
        ctx = tiledb.Ctx(cfg)
    elif ctx is None:
        ctx = default_ctx()

    if arr is ArraySchema:
        schema = arr
    else:
        schema = schema_like(arr, tile=tile, ctx=ctx)

    tiledb.DenseArray.create(uri, key=key, schema=schema)
    return tiledb.DenseArray(uri, mode='w', key=key, ctx=ctx)


def from_numpy(uri, array, ctx=None, **kw):
    """
    Convenience method to create a TileDB array from a Numpy array.
    See documentation of :func:`tiledb.DenseArray.from_numpy`.
    """
    if not ctx:
        ctx = default_ctx()
    if not isinstance(array, np.ndarray):
        raise Exception("from_numpy is only currently supported for numpy.ndarray")

    return DenseArray.from_numpy(uri, array, ctx=ctx, **kw)

def array_exists(uri, isdense=False, issparse=False):
    """
    Check if arrays exists and is open-able at the given URI

    Optionally restrict to `isdense` or `issparse` array types.
    """
    try:
        a = tiledb.open(uri)
    except TileDBError as exc:
        return False

    if isdense:
        rval = not a.schema.sparse
    elif issparse:
        rval = a.schema.sparse
    else:
        rval = True

    a.close()
    return rval