from .libtiledb cimport Array, Query

import tiledb.cc as lt

cdef class DomainIndexer:
    cdef object array_ref
    # def lt.ArraySchema schema
    cdef Query query