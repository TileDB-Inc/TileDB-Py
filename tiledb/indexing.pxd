from .libtiledb cimport Array, ArraySchema, Query

cdef class DomainIndexer:
    cdef Array array
    cdef ArraySchema schema
    cdef Query query
