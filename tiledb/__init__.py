import ctypes
import os
import sys

# un-comment this section to fix Cython backtrace line-numbers in
# IPython/Jupyter. see https://bugs.python.org/issue32797#msg323167
# ---
# try:
#    from importlib.machinery import ExtensionFileLoader
# else:
#    del ExtensionFileLoader.get_source
# ---

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

from .cc import TileDBError

from .ctx import default_ctx, scope_ctx
from .libtiledb import (
    Array,
    Ctx,
    Config,
    Dim,
    Domain,
    Attr,
    ArraySchema,
    consolidate,
    object_type,
    ls,
    walk,
    remove,
    move,
    stats_enable,
    stats_disable,
    stats_reset,
    stats_dump,
    vacuum,
)

from .array import DenseArray, SparseArray

from .filter import (
    Filter,
    FilterList,
    NoOpFilter,
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
    ChecksumMD5Filter,
    ChecksumSHA256Filter,
)

from .fragment import (
    FragmentInfoList,
    FragmentInfo,
    FragmentsInfo,
    copy_fragments_to_existing_array,
    delete_fragments,
    create_array_from_fragments,
)

from .group import Group

group_create = Group.create

from .object import Object

from .highlevel import (
    open,
    save,
    from_numpy,
    empty_like,
    array_exists,
    array_fragments,
)

from .query_condition import QueryCondition

from .schema import schema_like

from .schema_evolution import ArraySchemaEvolution

from .vfs import VFS, FileIO

# TODO restricted imports
from .dataframe_ import from_csv, from_pandas, open_dataframe
from .multirange_indexing import EmptyRange
from .parquet_ import from_parquet

from .version import version as __version__

from .version_ import VersionHelper

version = VersionHelper()

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
__path__ = __import__("pkgutil").extend_path(__path__, __name__)
