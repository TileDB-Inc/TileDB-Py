from libc.stdio cimport printf

import weakref

import numpy as np


def _index_as_tuple(idx):
    """Forces scalar index objects to a tuple representation"""
    if isinstance(idx, tuple):
        return idx
    return (idx,)

# ref
#   https://github.com/TileDB-Inc/TileDB-Py/issues/102
#   https://github.com/TileDB-Inc/TileDB-Py/issues/201

cdef class DomainIndexer(object):

    @staticmethod
    def with_schema(schema):
        cdef DomainIndexer indexer = DomainIndexer.__new__(DomainIndexer)
        indexer.array = None
        indexer.schema = schema
        return indexer

    def __init__(self, Array array, query = None):
        self.array_ref = weakref.ref(array)
        self.schema = array.schema
        self.query = query

    @property
    def schema(self):
        return self.array.array_ref().schema

    @property
    def array(self):
        assert self.array_ref() is not None, \
            "Internal error: invariant violation (index[] with dead array_ref)"
        return self.array_ref()

    def __getitem__(self, object idx):
        from .subarray import Subarray # prevent circular import
        # implements domain-based indexing: slice by domain coordinates, not 0-based python indexing

        schema = self.array.schema
        dom = schema.domain
        cdef ndim = dom.ndim
        cdef list attr_names = list()

        idx = _index_as_tuple(idx)

        if len(idx) < dom.ndim:
            raise IndexError("number of indices does not match domain rank: "
                             "(got {!r}, expected: {!r})".format(len(idx), ndim))

        new_idx = []
        for i in range(dom.ndim):
            dim = dom.dim(i)
            dim_idx = idx[i]
            if np.isscalar(dim_idx):
                start = dim_idx
                stop = dim_idx
                new_idx.append(slice(start, stop, None))
            else:
                new_idx.append(dim_idx)

        dim_ranges = list()

        for i, subidx in enumerate(new_idx):
            assert isinstance(subidx, slice)
            dim_ranges.append((subidx.start, subidx.stop))
        subarray = Subarray(self.array)
        subarray.add_ranges([list([x]) for x in dim_ranges])

        attr_names = list(schema.attr(i).name for i in range(schema.nattr))
        attr_cond = None

        order = None
        # TODO make coords optional for array.domain_index. there are no kwargs in slicing[], so
        #      one way to do this would be to overload __call__ and return a new
        #      object with a flag set. not ideal.
        coords = True

        if self.query is not None:
            # if we are called via Query object, then we need to respect Query semantics
            order = self.query.order
            attr_names = self.query.attrs if self.query.attrs else attr_names # query.attrs might be None -> all
            attr_cond = self.query.attr_cond
            coords = self.query.coords

        if coords:
            attr_names = [dom.dim(idx).name for idx in range(self.schema.ndim)] + attr_names

        if order is None or order == 'C':
            layout = TILEDB_ROW_MAJOR
        elif order == 'F':
            layout = TILEDB_COL_MAJOR
        elif order == 'G':
            layout = TILEDB_GLOBAL_ORDER
        elif order == 'U':
            layout = TILEDB_UNORDERED
        else:
            raise ValueError("order must be 'C' (TILEDB_ROW_MAJOR), 'F' (TILEDB_COL_MAJOR), or 'G' (TILEDB_GLOBAL_ORDER)")

        if isinstance(self.array, SparseArrayImpl):
            return (<SparseArrayImpl>self.array)._read_sparse_subarray(subarray, attr_names, attr_cond, layout)
        elif isinstance(self.array, DenseArrayImpl):
            return (<DenseArrayImpl>self.array)._read_dense_subarray(subarray, attr_names, attr_cond, layout, coords)
        else:
            raise Exception("No handler for Array type: " + str(type(self.array)))

cdef class QueryAttr(object):
    cdef unicode name
    cdef np.dtype dtype

    def __init__(self, name, dtype):
        self.name = name
        self.dtype = dtype

cdef dict execute_multi_index(Array array,
                              tiledb_query_t* query_ptr,
                              tuple attr_names,
                              return_coord):

    # NOTE: query_ptr *must* only be freed in caller

    cdef:
        tiledb_ctx_t* ctx_ptr = <tiledb_ctx_t*>PyCapsule_GetPointer(
            array.ctx.__capsule__(), "ctx")
        tiledb_query_status_t query_status

    cdef:
        uint64_t result_bytes = 0
        size_t result_elements
        float result_elements_f, result_rem
        uint64_t el_count = 0
        bint repeat_query = True
        uint64_t repeat_count = 0
        uint64_t buffer_bytes_remaining = 0
        uint64_t* buffer_sizes_ptr = NULL

    cdef:
        np.dtype coords_dtype

    cdef:
        Py_ssize_t attr_idx
        bytes battr_name
        unicode attr_name
        np.ndarray attr_array
        np.dtype attr_dtype
        QueryAttr qattr

    cdef list attrs = list()

    # Coordinate attribute buffers must be set first
    if return_coord:
        dims = tuple(array.schema.domain.dim(dim_idx) for dim_idx in \
                     range(array.schema.ndim))
        attrs += [QueryAttr(dim.name, dim.dtype) for dim in dims]

    # Get the attributes
    attrs += [QueryAttr(a.name, a.dtype)
                       for a in [array.schema.attr(name)
                                 for name in attr_names]]


    # Create and assign attribute result buffers

    cdef Py_ssize_t nattr = len(attrs)
    cdef uint64_t ndim = array.ndim

    cdef dict result_dict = dict()
    cdef np.ndarray buffer_sizes = np.zeros(nattr, np.uint64)
    cdef np.ndarray result_bytes_read = np.zeros(nattr, np.uint64)

    cdef uint64_t init_buffer_size = 1310720 * 8 # 10 MB int64
    if 'py.init_buffer_bytes' in array.ctx.config():
        init_buffer_size = int(array.ctx.config()['py.init_buffer_bytes'])
    # switch from exponential to linear (+4GB) allocation
    cdef uint64_t linear_alloc_bytes = 4 * (2**30) # 4 GB

    # There are two different conditions which may cause incomplete queries,
    # requiring retries and potentially reallocation to complete the read.
    # 1) user-allocated buffer is not large enough. In this case, we need to
    #    allocate more memory and retry. This is accomplished below by resizing
    #    the array in-place (preserving the existing data), then bumping the
    #    query buffer pointer.
    # 2) internal memory limit is exceeded: the libtiledb parameter
    #    'sm.memory_budget' governs internal memory allocation. If libtiledb's
    #    internal allocation exceeds this budget, the query may need to be
    #    retried, but we do not necessarily need to bump the user buffer allocation.
    while repeat_query:
        for attr_idx in range(nattr):
            qattr = attrs[attr_idx]
            attr_name = qattr.name
            attr_dtype = qattr.dtype

            # allocate initial array
            if repeat_count == 0:
                result_dict[attr_name] = np.zeros(int(init_buffer_size / attr_dtype.itemsize),
                                                  dtype=attr_dtype)

            # Get the array here in order to save a lookup
            attr_array = result_dict[attr_name]
            if repeat_count > 0:
                buffer_bytes_remaining = attr_array.nbytes - result_bytes_read[attr_idx]
                if buffer_sizes[attr_idx] > (.25 * buffer_bytes_remaining):
                    # Check number of bytes read during the *last* pass.
                    # The conditional above handles situation (2) in order to avoid re-allocation
                    # on every repeat, in case we are reading small chunks at a time due to libtiledb
                    # memory budget.
                    # TODO make sure 'refcheck=False' is always safe
                    if attr_array.nbytes < linear_alloc_bytes:
                        attr_array.resize(attr_array.size * 2, refcheck=False)
                    else:
                        new_size = attr_array.size + linear_alloc_bytes / attr_dtype.itemsize
                        attr_array.resize(new_size, refcheck=False)

            battr_name = attr_name.encode('UTF-8')
            attr_array_ptr = np.PyArray_DATA(attr_array)

            # we need to give the pointer to the current starting point after reallocation
            attr_array_ptr = \
                <void*>(<char*>attr_array_ptr + <ptrdiff_t>result_bytes_read[attr_idx])

            buffer_sizes[attr_idx] = attr_array.nbytes - result_bytes_read[attr_idx]
            buffer_sizes_ptr = <uint64_t*>np.PyArray_DATA(buffer_sizes)

            rc = tiledb_query_set_data_buffer(
                    ctx_ptr, query_ptr, battr_name, attr_array_ptr,
                    &(buffer_sizes_ptr[attr_idx]))

            if rc != TILEDB_OK:
                # NOTE: query_ptr *must* only be freed in caller
                _raise_ctx_err(ctx_ptr, rc)

        with nogil:
            rc = tiledb_query_submit(ctx_ptr, query_ptr)

        if rc != TILEDB_OK:
            # NOTE: query_ptr *must* only be freed in caller
            _raise_ctx_err(ctx_ptr, rc)

        # update bytes-read count
        for attr_idx in range(nattr):
            result_bytes_read[attr_idx] += buffer_sizes[attr_idx]

        rc = tiledb_query_get_status(ctx_ptr, query_ptr, &query_status)
        if rc != TILEDB_OK:
            # NOTE: query_ptr *must* only be freed in caller
            _raise_ctx_err(ctx_ptr, rc)

        if query_status == TILEDB_INCOMPLETE:
            #printf("%s\n", <const char*>"got incomplete!")
            repeat_query = True
            repeat_count += 1
        elif query_status == TILEDB_COMPLETED:
            repeat_query = False
            break
        elif query_status == TILEDB_FAILED:
            raise TileDBError("Query returned TILEDB_FAILED")
        elif query_status == TILEDB_INPROGRESS:
            raise TileDBError("Query returned TILEDB_INPROGRESS")
        elif query_status == TILEDB_INCOMPLETE:
            raise TileDBError("Query returned TILEDB_INCOMPLETE")
        else:
            raise TileDBError("internal error: unknown query status")

    # resize arrays to final bytes-read
    for attr_idx in range(nattr):
        qattr = attrs[attr_idx]
        attr_name = qattr.name
        attr_dtype = qattr.dtype

        attr_item_size = attr_dtype.itemsize
        attr_array = result_dict[attr_name]
        attr_array.resize(int(result_bytes_read[attr_idx] / attr_item_size), refcheck=False)

    return result_dict

cpdef multi_index(Array array, tuple attr_names, tuple ranges,
                  order = None, coords = None):

    cdef tiledb_layout_t layout = TILEDB_UNORDERED
    if order is None or order == 'C':
        layout = TILEDB_ROW_MAJOR
    elif order == 'F':
        layout = TILEDB_COL_MAJOR
    elif order == 'G':
        layout = TILEDB_GLOBAL_ORDER
    else:
        raise ValueError("order must be 'C' (TILEDB_ROW_MAJOR), "\
                         "'F' (TILEDB_COL_MAJOR), "\
                         "or 'G' (TILEDB_GLOBAL_ORDER)")

    cdef tiledb_ctx_t* ctx_ptr = <tiledb_ctx_t*>PyCapsule_GetPointer(
        array.ctx.__capsule__(), "ctx")
    cdef tiledb_array_t* array_ptr = array.ptr
    cdef tiledb_query_t* query_ptr = NULL

    cdef int rc = TILEDB_OK
    rc = tiledb_query_alloc(ctx_ptr, array_ptr, TILEDB_READ, &query_ptr)
    if rc != TILEDB_OK:
        tiledb_query_free(&query_ptr)
        _raise_ctx_err(ctx_ptr, rc)
    rc = tiledb_query_set_layout(ctx_ptr, query_ptr, layout)
    if rc != TILEDB_OK:
        tiledb_query_free(&query_ptr)
        _raise_ctx_err(ctx_ptr, rc)

    dim = array.schema.domain.dim(0)
    cdef uint32_t c_dim_idx
    cdef void* start_ptr = NULL
    cdef void* end_ptr = NULL
    cdef tuple cur_range

    cdef np.ndarray start
    cdef np.ndarray end

    # Add ranges to query
    #####################
    # we loop over the range tuple left to right and apply
    # (unspecified dimensions are excluded)
    cdef Py_ssize_t dim_idx, range_idx
    cdef tiledb_subarray_t* subarray_ptr = NULL
    cdef bint is_default = True

    rc = tiledb_subarray_alloc(ctx_ptr, array_ptr, &subarray_ptr)
    if rc != TILEDB_OK:
        tiledb_subarray_free(&subarray_ptr)
        tiledb_query_free(&query_ptr)
        _raise_ctx_err(ctx_ptr, rc)

    for dim_idx in range(len(ranges)):
        c_dim_idx = <uint32_t>dim_idx
        dim_ranges = ranges[dim_idx]

        # skip empty dimensions
        if len(dim_ranges) == 0:
            continue

        is_default = False
        for range_idx in range(len(dim_ranges)):
            if len(dim_ranges[range_idx]) != 2:
                tiledb_subarray_free(&subarray_ptr)
                tiledb_query_free(&query_ptr)
                raise TileDBError("internal error: invalid sub-range: ", dim_ranges[range_idx])

            start = np.array(dim_ranges[range_idx][0], dtype=dim.dtype)
            end = np.array(dim_ranges[range_idx][1], dtype=dim.dtype)

            start_ptr = np.PyArray_DATA(start)
            end_ptr = np.PyArray_DATA(end)

            rc = tiledb_subarray_add_range(
                    ctx_ptr, subarray_ptr, dim_idx, start_ptr, end_ptr, NULL)

            if rc != TILEDB_OK:
                tiledb_subarray_free(&subarray_ptr)
                tiledb_query_free(&query_ptr)
                _raise_ctx_err(ctx_ptr, rc)

        if not is_default:
            rc = tiledb_query_set_subarray_t(ctx_ptr, query_ptr, subarray_ptr)
            if rc != TILEDB_OK:
                tiledb_subarray_free(&subarray_ptr)
                tiledb_query_free(&query_ptr)
                _raise_ctx_err(ctx_ptr, rc)

    try:
        if coords is None:
            coords = True
        result = execute_multi_index(array, query_ptr, attr_names, coords)
    finally:
        tiledb_query_free(&query_ptr)

    return result
