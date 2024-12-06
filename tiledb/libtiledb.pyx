#!python
#cython: embedsignature=True
#cython: auto_pickle=False

from cpython.pycapsule cimport PyCapsule_GetPointer
from cpython.version cimport PY_MAJOR_VERSION

include "common.pxi"

from .cc import TileDBError
from .ctx import Config, Ctx, default_ctx
from .array import Array

###############################################################################
#     Numpy initialization code (critical)                                    #
###############################################################################

# https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.import_array
np.import_array()

###############################################################################
#    Utility/setup                                                            #
###############################################################################

# Use unified numpy printing
np.set_printoptions(legacy="1.21" if np.lib.NumpyVersion(np.__version__) >= "1.22.0" else False)
