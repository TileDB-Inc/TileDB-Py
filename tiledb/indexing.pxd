from .libtiledb cimport Array, Query


cdef class DomainIndexer:
    cdef object array_ref
    cdef object schema
    cdef Query query