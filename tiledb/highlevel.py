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


def delete_fragments(
    uri, timestamp_range, config=None, ctx=None, verbose=False, dry_run=False
):
    """
    Delete fragments from an array located at uri that falls within a given
    timestamp_range.

    :param str uri: URI for the TileDB array (any supported TileDB URI)
    :param (int, int) timestamp_range: (default None) If not None, vacuum the
        array using the given range (inclusive)
    :param config: Override the context configuration. Defaults to ctx.config()
    :param ctx: (optional) TileDB Ctx
    :param verbose: (optional) Print fragments being deleted (default: False)
    :param dry_run: (optional) Preview fragments to be deleted without
        running (default: False)
    """

    if not isinstance(timestamp_range, tuple) and len(timestamp_range) != 2:
        raise TypeError(
            "'timestamp_range' argument expects tuple(start: int, end: int)"
        )

    if not ctx:
        ctx = tiledb.default_ctx()

    if config is None:
        config = tiledb.Config(ctx.config())

    vfs = tiledb.VFS(config=config, ctx=ctx)

    if verbose or dry_run:
        print("Deleting fragments:")

    for frag in tiledb.array_fragments(uri):
        if (
            timestamp_range[0] <= frag.timestamp_range[0]
            and frag.timestamp_range[1] <= timestamp_range[1]
        ):
            if verbose or dry_run:
                print(f"\t{frag.uri}")

            if not dry_run:
                vfs.remove_file(f"{frag.uri}.ok")
                vfs.remove_dir(frag.uri)


def create_array_from_fragments(
    src_uri,
    dst_uri,
    timestamp_range,
    config=None,
    ctx=None,
    verbose=False,
    dry_run=False,
):
    """
    (POSIX only). Create a new array from an already existing array by selecting
    fragments that fall withing a given timestamp_range. The original array is located
    at src_uri and the new array is created at dst_uri.

    :param str src_uri: URI for the source TileDB array (any supported TileDB URI)
    :param str dst_uri: URI for the newly created TileDB array (any supported TileDB URI)
    :param (int, int) timestamp_range: (default None) If not None, vacuum the
        array using the given range (inclusive)
    :param config: Override the context configuration. Defaults to ctx.config()
    :param ctx: (optional) TileDB Ctx
    :param verbose: (optional) Print fragments being copied (default: False)
    :param dry_run: (optional) Preview fragments to be copied without
        running (default: False)
    """
    if array_exists(dst_uri):
        raise tiledb.TileDBError(f"Array URI `{dst_uri}` already exists")

    if not isinstance(timestamp_range, tuple) and len(timestamp_range) != 2:
        raise TypeError(
            "'timestamp_range' argument expects tuple(start: int, end: int)"
        )

    if not ctx:
        ctx = tiledb.default_ctx()

    if config is None:
        config = tiledb.Config(ctx.config())

    vfs = tiledb.VFS(config=config, ctx=ctx)

    fragment_info = tiledb.array_fragments(src_uri)

    if len(fragment_info) < 1:
        print("Cannot create new array; no fragments to copy")
        return

    if verbose or dry_run:
        print(f"Creating directory for array at {dst_uri}\n")

    if not dry_run:
        vfs.create_dir(dst_uri)

    src_lock = os.path.join(src_uri, "__lock.tdb")
    dst_lock = os.path.join(dst_uri, "__lock.tdb")

    if verbose or dry_run:
        print(f"Copying lock file {dst_uri}\n")

    if not dry_run:
        vfs.copy_file(f"{src_lock}", f"{dst_lock}")

    list_new_style_schema = [ver >= 10 for ver in fragment_info.version]
    is_mixed_versions = len(set(list_new_style_schema)) > 1
    if is_mixed_versions:
        raise tiledb.TileDBError(
            "Cannot copy fragments - this array contains a mix of old and "
            "new style schemas"
        )
    is_new_style_schema = list_new_style_schema[0]

    for frag in fragment_info:
        if not (
            timestamp_range[0] <= frag.timestamp_range[0]
            and frag.timestamp_range[1] <= timestamp_range[1]
        ):
            continue

        schema_name = frag.array_schema_name
        if is_new_style_schema:
            schema_name = os.path.join("__schema", schema_name)
        src_schema = os.path.join(src_uri, schema_name)
        dst_schema = os.path.join(dst_uri, schema_name)

        if verbose or dry_run:
            print(f"Copying schema `{src_schema}` to `{dst_schema}`\n")

        if not dry_run:
            if is_new_style_schema:
                new_style_schema_uri = os.path.join(dst_uri, "__schema")
                if not vfs.is_dir(new_style_schema_uri):
                    vfs.create_dir(new_style_schema_uri)

            if not vfs.is_file(dst_schema):
                vfs.copy_file(src_schema, dst_schema)

        frag_name = os.path.basename(frag.uri)
        src_frag = frag.uri
        dst_frag = os.path.join(dst_uri, frag_name)

        if verbose or dry_run:
            print(f"Copying fragment `{src_frag}` to `{dst_frag}`\n")

        if not dry_run:
            vfs.copy_file(f"{src_frag}.ok", f"{dst_frag}.ok")
            vfs.copy_dir(src_frag, dst_frag)


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
