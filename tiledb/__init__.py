from __future__ import absolute_import

from .libtiledb import (Ctx, Config, Dim, Domain, Attr, KV, DenseArray, SparseArray, TileDBError, VFS,
                        group_create, object_type, ls, walk, remove, move)

__all__ = [Ctx, Config, Dim, Domain, Attr, KV, DenseArray, SparseArray, TileDBError, VFS,
           group_create, object_type, ls, walk, remove, move]

if libtiledb.version() != (1, 2, 0):
    raise RuntimeError("The libtiledb library version does not match "
                       "the latest released version of tiledb v.1.2.0")
