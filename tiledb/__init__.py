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

# On Windows and whl builds, we may have a shared library already linked, or
# adjacent to, the cython .pyd shared object. In this case, we can import directly
# from .libtiledb
try:
    import tiledb
    from .libtiledb import Ctx
except:
    try:
        lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "native")
        ctypes.CDLL(os.path.join(lib_dir, lib_name))
    except OSError as e:
        # Otherwise try loading by name only.
        ctypes.CDLL(lib_name)

from .libtiledb import (
     Array,
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
     default_ctx,
     group_create,
     object_type,
     ls,
     walk,
     remove,
     move,
     stats_enable,
     stats_disable,
     stats_reset,
     stats_dump,
)

from .highlevel import *

# Note: we use a modified namespace packaging to allow continuity of existing TileDB-Py imports.
#       Therefore, 'tiledb/__init__.py' must *only* exist in this package.
#       Furthermore, in sub-packages, the `find_packages` helper will not work at the
#       root directory due to lack of 'tiledb/__init__.py'. Sub-package 'setup.py' scripts
#       must declare constituents accordingly, such as by running 'find_packages' on a sub-directory
#       and applying prefixes accordingly.
#   1) https://packaging.python.org/guides/packaging-namespace-packages/#native-namespace-packages
#   2) https://stackoverflow.com/a/53486554
#
# Note: 'pip -e' in particular will not work without this declaration:
__path__ = __import__('pkgutil').extend_path(__path__, __name__)
