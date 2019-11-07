from .libtiledb import DenseArrayImpl, SparseArrayImpl

class DenseArray(DenseArrayImpl):
    _mixin_init = False

    def __new__(cls, *args, **kwargs):
        if not cls._mixin_init:
            try:
                from tiledb.cloud import cloudarray
                DenseArray.__bases__ = DenseArray.__bases__ + (cloudarray.CloudArray,)
            except ImportError:
                pass
            finally:
                cls._mixin_init = True

        obj = super(DenseArray, cls).__new__(cls, *args, **kwargs)
        return obj

class SparseArray(SparseArrayImpl):
    _mixin_init = False

    def __new__(cls, *args, **kwargs):
        if not cls._mixin_init:
            try:
                from tiledb.cloud import cloudarray
                SparseArray.__bases__ = SparseArray.__bases__ + (cloudarray.CloudArray,)
            except ImportError:
                pass
            finally:
                cls._mixin_init = True

        obj = super(SparseArray, cls).__new__(cls, *args, **kwargs)
        return obj
