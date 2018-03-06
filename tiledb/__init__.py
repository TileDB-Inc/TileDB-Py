from __future__ import absolute_import
import os

from .libtiledb import (Ctx, Config, Dim, Domain, Attr, KV, ArraySchema, DenseArray, SparseArray,
                        TileDBError, VFS, array_consolidate, group_create, object_type,
                        ls, walk, remove, move)

__all__ = [Ctx, Config, Dim, Domain, Attr, KV, ArraySchema, SparseArray, TileDBError, VFS,
           array_consolidate, group_create, object_type, ls, walk, remove, move]

if not os.environ.get("TILEDB_PY_NO_VERSION_CHECK"):
    if libtiledb.version() != (1, 2, 0):
        raise RuntimeError("The libtiledb library version does not match "
                           "the latest released version of tiledb v.1.2.0")
