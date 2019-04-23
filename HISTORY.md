# In Progress

## New Features:

* Support for NumPy complex types [#142](https://github.com/TileDB-Inc/TileDB-Py/pull/142)

## Bug fixes
* fixes for several indexing bugs [#146](https://github.com/TileDB-Inc/TileDB-Py/pull/146)
  - mixed-type range indexing, where the range bounds are different types and must be promoted
    [#140](https://github.com/TileDB-Inc/TileDB-Py/issues/140)
  - dense array scalar assignment (direct report).
  - improved support for returning fixed-size strings from sparse arrays (direct report).
* fixed `VFS.is_bucket` when VFS is initialized with a Ctx object [#148](https://github.com/TileDB-Inc/TileDB-Py/pull/148)
* fixed `schema_like` to correctly forward a Ctx keyword arg [#148](https://github.com/TileDB-Inc/TileDB-Py/pull/148)


# TileDB-Py 0.4.1 Release Notes

## New Features:

* several high-level API additions (tiledb.open, .save, .empty_like, .schema_like), and serialization improvements including pickling support for DenseArray objects (#129)
* manylinux1 wheels for Python 2.7, 3.5, 3.6, and 3.7 are available on PyPI: https://pypi.org/project/tiledb

# TileDB-Py 0.4.0 Release Notes

This release builds TileDB-Py against TileDB 1.5

## New Features:

* support for variable-length arrays (#120)

## Breaking changes:

* the Ctx argument is now a keyword argument, simplifying API use in the common case (#122)

  for example: `tiledb.DenseArray(ctx, uri, ...)` becomes: tiledb.DenseArray(uri, ...)
  or optionally `tiledb.DenseArray(uri, ..., ctx=ctx)`
