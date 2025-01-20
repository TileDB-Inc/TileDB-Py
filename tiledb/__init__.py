import ctypes
import os
import sys
import warnings

if os.name == "posix":
    if sys.platform == "darwin":
        lib_name = "libtiledb.dylib"
    else:
        lib_name = "libtiledb.so"
else:
    lib_name = "tiledb"

import numpy as np

# TODO: get rid of this - It is currently used for unified numpy printing accross numpy versions
np.set_printoptions(
    legacy="1.21" if np.lib.NumpyVersion(np.__version__) >= "1.22.0" else False
)
del np

from tiledb.libtiledb import version as libtiledb_version

if libtiledb_version()[0] == 2 and libtiledb_version()[1] >= 26:
    from .current_domain import CurrentDomain
    from .ndrectangle import NDRectangle

del libtiledb_version  # no longer needed

from .array import Array
from .array_schema import ArraySchema
from .attribute import Attr
from .consolidation_plan import ConsolidationPlan
from .ctx import Config, Ctx, default_ctx, scope_ctx
from .dataframe_ import from_csv, from_pandas, open_dataframe
from .dense_array import DenseArrayImpl
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
    CompressionFilter,
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
    copy_fragments_to_existing_array,
    create_array_from_fragments,
)
from .group import Group
from .highlevel import (
    array_exists,
    array_fragments,
    as_built,
    consolidate,
    empty_like,
    from_numpy,
    ls,
    move,
    object_type,
    open,
    remove,
    save,
    schema_like,
    vacuum,
    walk,
)
from .libtiledb import TileDBError
from .multirange_indexing import EmptyRange
from .object import Object
from .parquet_ import from_parquet
from .query import Query
from .query_condition import QueryCondition
from .schema_evolution import ArraySchemaEvolution
from .sparse_array import SparseArrayImpl
from .stats import (
    stats_disable,
    stats_dump,
    stats_enable,
    stats_reset,
)
from .subarray import Subarray
from .version_helper import version
from .vfs import VFS, FileIO

__version__ = version.version
group_create = Group.create

# Create a proxy object to wrap libtiledb and provide a `cc` alias
class CCProxy:
    def __init__(self, module):
        self._module = module

    def __getattr__(self, name):
        warnings.warn(
            "`tiledb.cc` is deprecated. Please use `tiledb.libtiledb` instead.",
        )
        return getattr(self._module, name)

    def __repr__(self):
        warnings.warn(
            "`tiledb.cc` is deprecated. Please use `tiledb.libtiledb` instead.",
        )
        return self._module.__repr__()


cc = CCProxy(libtiledb)
sys.modules["tiledb.cc"] = cc
cc = cc
del CCProxy

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

    class DenseArray(DenseArrayImpl):
        pass

    class SparseArray(SparseArrayImpl):
        pass

else:

    class DenseArray(DenseArrayImpl, CloudArray):
        pass

    class SparseArray(SparseArrayImpl, CloudArray):
        pass

    del CloudArray
