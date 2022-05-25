from .libtiledb import DenseArrayImpl, SparseArrayImpl

# Extensible (pure Python) array class definitions inheriting from the
# Cython implementation.

# NOTE: the mixin import must be inside the __new__ initializer because it
#       needs to be deferred. tiledb.cloud is not yet known to the importer
#       when this code is imported.
#       TODO: might be possible to work-around/simplify by using
#       import meta-hooks instead.


class DenseArray(DenseArrayImpl):
    """Class representing a dense TileDB array.

    Inherits properties and methods of :py:class:`tiledb.Array` and
    implements `__setitem__` and `__getitem__` for dense array indexing
    and assignment.
    """

    def __new__(cls, *args, **kwargs):
        try:
            from tiledb.cloud import cloudarray

            cls.__bases__ += (cloudarray.CloudArray,)
        except (ImportError, TypeError):
            # If we couldn't import, tiledb.cloud isn't installed
            # If there was a TypeError, CloudArray was already added (by another thread)
            pass
        try:
            # Mixin initialization completed: delete this method for subsequent calls
            del DenseArray.__new__
        except AttributeError:
            # This method is already deleted (by another thread)
            pass
        return super().__new__(cls, *args, **kwargs)


class SparseArray(SparseArrayImpl):
    """Class representing a sparse TileDB array.

    Inherits properties and methods of :py:class:`tiledb.Array` and
    implements `__setitem__` and `__getitem__` for sparse array indexing
    and assignment.
    """

    def __new__(cls, *args, **kwargs):
        try:
            from tiledb.cloud import cloudarray

            cls.__bases__ += (cloudarray.CloudArray,)
        except (ImportError, TypeError):
            # If we couldn't import, tiledb.cloud isn't installed
            # If there was a TypeError, CloudArray was already added (by another thread)
            pass
        try:
            # Mixin initialization completed: delete this method for subsequent calls
            del SparseArray.__new__
        except AttributeError:
            # This method is already deleted (by another thread)
            pass
        return super().__new__(cls, *args, **kwargs)
