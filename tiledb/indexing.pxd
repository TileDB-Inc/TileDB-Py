from .libtiledb cimport SparseArray, ArraySchema

cdef class DomainIndexer:
    cdef SparseArray array
    cdef ArraySchema schema
