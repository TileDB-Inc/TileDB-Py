from __future__ import absolute_import

import ctypes
import os
import sys

if os.name == "posix":
    if sys.platform == "darwin":
        lib_name = "libtiledb.dylib"
    else:
        lib_name = "libtiledb.so"
else:
    lib_name = "tiledb"

# Load the bundled TileDB dynamic library if it exists. This must occur before importing from libtiledb.
# On Windows we put the DLLs next to .pyd out of necessity, so this can be skipped.
if os.name != 'nt':
    try:
        lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "native")
        ctypes.CDLL(os.path.join(lib_dir, lib_name))
    except OSError as e:
        # Otherwise try loading by name only.
        ctypes.CDLL(lib_name)

from .libtiledb import (
     Ctx,
     Config,
     Dim,
     Domain,
     Attr,
     KVSchema,
     KV,
     ArraySchema,
     DenseArray,
     SparseArray,
     TileDBError,
     VFS,
     FileIO,
     FilterList,
     GzipFilter,
     ZstdFilter,
     LZ4Filter,
     Bzip2Filter,
     RleFilter,
     DoubleDeltaFilter,
     BitShuffleFilter,
     ByteShuffleFilter,
     BitWidthReductionFilter,
     PositiveDeltaFilter,
     consolidate,
     group_create,
     object_type,
     ls,
     walk,
     remove,
     move,
     stats_enable,
     stats_disable,
     stats_reset,
     stats_dump
)
#
# __all__ = [Ctx, Config, Dim, Domain, Attr, KV, ArraySchema, SparseArray, TileDBError, VFS,
#            array_consolidate, group_create, object_type, ls, walk, remove]
