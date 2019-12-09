from .libtiledb import DenseArrayImpl, SparseArrayImpl

# Extensible (pure Python) array class definitions inheriting from the
# Cython implemention. The cloudarray mix-in adds optional functionality
# for registering arrays and executing functions on the

# NOTE: the mixin import must be inside the __new__ initializer because it
#       needs to be deferred. tiledb.cloud is not yet known to the importer
#       when this code is imported.
#       TODO: might be possible to work-around/simplify by using
#       import meta-hooks instead.

class DenseArray(DenseArrayImpl):
    _mixin_init = False

    def __new__(cls, *args, **kwargs):
        if not cls._mixin_init:
            # must set before importing, because import is not thread-safe
            #   https://github.com/TileDB-Inc/TileDB-Py/issues/244
            cls._mixin_init = True
            try:
                from tiledb.cloud import cloudarray
                DenseArray.__bases__ = DenseArray.__bases__ + (cloudarray.CloudArray,)
                DenseArray.__doc__ = DenseArrayImpl.__doc__
            except ImportError:
                pass

        obj = super(DenseArray, cls).__new__(cls, *args, **kwargs)
        return obj

class SparseArray(SparseArrayImpl):
    _mixin_init = False

    def __new__(cls, *args, **kwargs):
        if not cls._mixin_init:
            cls._mixin_init = True
            try:
                from tiledb.cloud import cloudarray
                SparseArray.__bases__ = SparseArray.__bases__ + (cloudarray.CloudArray,)
                SparseArray.__doc__ = DenseArrayImpl.__doc__
            except ImportError:
                pass

        obj = super(SparseArray, cls).__new__(cls, *args, **kwargs)
        return obj
