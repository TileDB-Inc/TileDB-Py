from .libtiledb cimport Array, Query

cdef class DomainIndexer:
    cdef object array_ref
    # cdef ArraySchema schema
    cdef Query query