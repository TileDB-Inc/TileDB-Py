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

    del Ctx
except:
    try:
        lib_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "native")
        ctypes.CDLL(os.path.join(lib_dir, lib_name))
    except OSError:
        # Otherwise try loading by name only.
        ctypes.CDLL(lib_name)

from .array_schema import ArraySchema
from .attribute import Attr
from .cc import TileDBError
from .ctx import Config, Ctx, default_ctx, scope_ctx
from .dataframe_ import from_csv, from_pandas, open_dataframe
from .dimension import Dim
from .dimension_label import DimLabel
from .dimension_label_schema import DimLabelSchema
from .domain import Domain
from .enumeration import Enumeration
from .filestore import Filestore
from .filter import (
    BitShuffleFilter,
    BitWidthReductionFilter,
    ByteShuffleFilter,
    Bzip2Filter,
    ChecksumMD5Filter,
    ChecksumSHA256Filter,
    DeltaFilter,
    DictionaryFilter,
    DoubleDeltaFilter,
    Filter,
    FilterList,
    FloatScaleFilter,
    GzipFilter,
    LZ4Filter,
    NoOpFilter,
    PositiveDeltaFilter,
    RleFilter,
    WebpFilter,
    XORFilter,
    ZstdFilter,
)
from .fragment import (
    FragmentInfo,
    FragmentInfoList,
    FragmentsInfo,
    copy_fragments_to_existing_array,
    create_array_from_fragments,
    delete_fragments,
)
from .group import Group
from .highlevel import (
    array_exists,
    array_fragments,
    empty_like,
    from_numpy,
    open,
    save,
    schema_like,
)
from .libtiledb import (
    Array,
    consolidate,
    ls,
    move,
    object_type,
    remove,
    stats_disable,
    stats_dump,
    stats_enable,
    stats_reset,
    vacuum,
    walk,
)
from .libtiledb import DenseArrayImpl as DenseArray
from .libtiledb import SparseArrayImpl as SparseArray
from .multirange_indexing import EmptyRange
from .object import Object
from .parquet_ import from_parquet
from .query import Query
from .query_condition import QueryCondition
from .schema_evolution import ArraySchemaEvolution
from .subarray import Subarray
from .version_helper import version
from .vfs import VFS, FileIO

__version__ = version.version
group_create = Group.create

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

# If tiledb.cloud is installed, add CloudArray methods to TileDB arrays
try:
    from tiledb.cloud.cloudarray import CloudArray
except ImportError:
    pass
else:

    class DenseArray(DenseArray, CloudArray):
        pass

    class SparseArray(SparseArray, CloudArray):
        pass

    del CloudArray
