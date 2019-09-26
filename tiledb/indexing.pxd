from __future__ import absolute_import

from .libtiledb cimport *

cdef class DomainIndexer:
    cdef SparseArray array
    cdef ArraySchema schema
