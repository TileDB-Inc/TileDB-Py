IF TILEDBPY_MODULAR:
  include "common.pxi"
  from .libtiledb cimport *

import numpy as np
from .array import DenseArray, SparseArray

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
    def with_schema(ArraySchema schema):
        cdef DomainIndexer indexer = DomainIndexer.__new__(DomainIndexer)
        indexer.array = None
        indexer.schema = schema
        return indexer

    def __init__(self, Array array, query = None):
        self.array = array
        self.schema = array.schema
        self.query = query

    def __getitem__(self, object idx):
        # implements domain-based indexing: slice by domain coordinates, not 0-based python indexing

        cdef Domain dom = self.schema.domain
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

        subarray = np.zeros(shape=(ndim, 2), dtype=dom.dtype)

        for i, subidx in enumerate(new_idx):
            assert isinstance(subidx, slice)
            subarray[i] = subidx.start, subidx.stop

        attr_names = list(self.schema.attr(i).name for i in range(self.schema.nattr))

        order = None
        # TODO make coords optional for array.domain_index. there are no kwargs in slicing[], so
        #      one way to do this would be to overload __call__ and return a new
        #      object with a flag set. not ideal.
        coords = True

        if self.query is not None:
            # if we are called via Query object, then we need to respect Query semantics
            order = self.query.order
            attr_names = self.query.attrs if self.query.attrs else attr_names # query.attrs might be None -> all
            coords = self.query.coords

        if coords:
            attr_names.insert(0, "coords")

        if order is None or order == 'C':
            layout = TILEDB_ROW_MAJOR
        elif order == 'F':
            layout = TILEDB_COL_MAJOR
        elif order == 'G':
            layout = TILEDB_GLOBAL_ORDER
        else:
            raise ValueError("order must be 'C' (TILEDB_ROW_MAJOR), 'F' (TILEDB_COL_MAJOR), or 'G' (TILEDB_GLOBAL_ORDER)")


        if isinstance(self.array, SparseArray):
            return (<SparseArrayImpl>self.array)._read_sparse_subarray(subarray, attr_names, layout)
        elif isinstance(self.array, DenseArray):
            return (<DenseArrayImpl>self.array)._read_dense_subarray(subarray, attr_names, layout)
        else:
            raise Exception("No handler for Array type: " + str(type(self.array)))

cdef dict execute_dense(tiledb_ctx_t* ctx_ptr,
                        tiledb_query_t* query_ptr,
                        DenseArrayImpl array,
                        tuple attr_names):

    # Create and assign attribute result buffers
    cdef uint64_t attr_idx
    cdef dict res = dict()
    cdef Attr attr
    cdef bytes battr_name
    cdef unicode attr_name
    cdef uint64_t result_bytes = 0
    cdef size_t result_elements
    cdef float result_elements_f, result_rem
    cdef np.ndarray attr_array
    cdef uint64_t attr_array_size
    cdef void* attr_array_ptr = NULL

    for attr_idx in range(array.schema.nattr):
        attr = array.schema.attr(attr_idx)
        attr_name = attr.name
        battr_name = attr_name.encode('UTF-8')
        rc = tiledb_query_get_est_result_size(ctx_ptr, query_ptr,
                                              battr_name, &result_bytes)

        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        if result_bytes == 0:
            raise TileDBError("Multi-range query returned size estimate result 0!")

        result_elements_f, result_rem = divmod(result_bytes, attr.dtype.itemsize)

        if result_rem != 0:
            raise TileDBError("Multi-range query size estimate "
                              "is not integral multiple of dtype bytes"
                              " (result_bytes: '{}', result_rem: '{}'".format(
                              result_bytes, result_rem))

        # TODO check that size matches cross-product of ranges (for dense)?

        result_elements = <size_t>result_elements_f

        attr_array = np.zeros(result_elements, dtype=attr.dtype)
        attr_array_size = attr_array.nbytes
        attr_array_ptr = np.PyArray_DATA(attr_array)

        rc = tiledb_query_set_buffer(ctx_ptr, query_ptr, battr_name,
                                     attr_array_ptr, &attr_array_size)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        # store the result
        res[attr_name] = attr_array

    with nogil:
        rc = tiledb_query_submit(ctx_ptr, query_ptr)

    if rc != TILEDB_OK:
        _raise_ctx_err(ctx_ptr, rc)

    return res

cdef dict execute_sparse(tiledb_ctx_t* ctx_ptr,
                        tiledb_query_t* query_ptr,
                        SparseArrayImpl array,
                        tuple attr_names,
                        return_coord):

    # NOTE: query_ptr *must* be freed in caller

    cdef np.dtype coords_dtype
    cdef unicode coord_name = (tiledb_coords()).decode('UTF-8')
    # coordinate attribute must be first
    if return_coord:
        attr_names = (coord_name, *attr_names)
        coords_dtype = array.coords_dtype

    # Create and assign attribute result buffers
    cdef uint64_t attr_idx
    cdef Attr attr
    cdef bytes battr_name
    cdef unicode attr_name
    cdef uint64_t result_bytes = 0
    cdef size_t result_elements
    cdef float result_elements_f, result_rem
    cdef np.ndarray attr_array
    cdef uint64_t el_count

    cdef void* attr_array_ptr = NULL
    cdef uint64_t* buffer_sizes_ptr = NULL

    cdef bint repeat_query = True
    cdef tiledb_query_status_t query_status
    cdef uint64_t repeat_count = 0

    cdef Py_ssize_t nattr = len(attr_names)
    cdef uint64_t ndim = array.ndim

    cdef dict result_dict = dict()
    cdef np.ndarray buffer_sizes = np.zeros(nattr, np.uint64)
    cdef np.ndarray result_bytes_read = np.zeros(nattr, np.uint64)

    # TODO ... make this nicer
    cdef uint64_t init_element_count = 1310720 # 10 MB int64
    cdef uint64_t max_element_count  = 6553600 # 50 MB int64

    while repeat_query:
        for attr_idx in range(nattr):
            if return_coord and attr_idx == 0:
                # coords
                attr_name = coord_name
                attr_dtype = coords_dtype
            else:
                # attributes
                attr = array.schema.attr(attr_idx - (1 if return_coord else 0))
                attr_dtype = attr.dtype
                attr_name = attr.name

            if repeat_count < 1:
                # coords_dtype is a record with 1 element per ndim coords
                result_dict[attr_name] = np.zeros(init_element_count,
                                                  dtype=attr_dtype)
            else:
                new_el_count = init_element_count if (repeat_count < 2) else max_element_count

                # resize in place
                attr_array = result_dict[attr_name]
                # TODO make sure 'refcheck=False' is always safe
                attr_array.resize(attr_array.size + new_el_count, refcheck=False)

            attr_item_size = coords_dtype.itemsize
            battr_name = attr_name.encode('UTF-8')

            attr_array = result_dict[attr_name]
            attr_array_ptr = np.PyArray_DATA(attr_array)

            # we need to give the pointer to the current starting point after reallocation
            attr_array_ptr = \
                <void*>(<char*>attr_array_ptr + <ptrdiff_t>result_bytes_read[attr_idx])

            buffer_sizes[attr_idx] = attr_array.nbytes - result_bytes_read[attr_idx]
            buffer_sizes_ptr = <uint64_t*>np.PyArray_DATA(buffer_sizes)

            rc = tiledb_query_set_buffer(ctx_ptr, query_ptr,
                                         battr_name,
                                         attr_array_ptr,
                                         &(buffer_sizes_ptr[attr_idx]))

            if rc != TILEDB_OK:
                _raise_ctx_err(ctx_ptr, rc)

        with nogil:
            rc = tiledb_query_submit(ctx_ptr, query_ptr)

        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        # update bytes-read count
        for attr_idx in range(nattr):
            result_bytes_read[attr_idx] += buffer_sizes[attr_idx]

        rc = tiledb_query_get_status(ctx_ptr, query_ptr, &query_status)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        if query_status == TILEDB_INCOMPLETE:
            repeat_query = True
            repeat_count += 1
        elif query_status == TILEDB_COMPLETED:
            repeat_query = False
            break
        elif query_status == TILEDB_FAILED:
            raise TileDBError("Query returned TILEDB_FAILED")
        elif query_status == TILEDB_INPROGRESS:
            raise TileDBError("Query return TILEDB_INPROGRESS")
        elif query_status == TILEDB_INCOMPLETE:
            raise TileDBError("Query returned TILEDB_INCOMPLETE")
        else:
            raise TileDBError("internal error: unknown query status")

    # resize arrays to final bytes-read
    for attr_idx in range(nattr):
        if return_coord and attr_idx == 0:
            attr_name = coord_name
            attr_dtype = coords_dtype
        else:
            attr = array.schema.attr(attr_idx - (1 if return_coord else 0))
            attr_dtype = attr.dtype
            attr_name = attr.name

        attr_item_size = attr_dtype.itemsize
        attr_array = result_dict[attr_name]
        attr_array.resize(int(result_bytes_read[attr_idx] / attr_item_size), refcheck=False)

    if return_coord:
        # replace internal name identifier
        result_dict["coords"] = result_dict.pop(coord_name)

    return result_dict

cpdef multi_index(Array array, tuple attr_names, tuple ranges,
                  order = None, coords = True):

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

    cdef tiledb_ctx_t* ctx_ptr = array.ctx.ptr
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

    cdef Dim dim = array.schema.domain.dim(0)
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
    cdef size_t dim_idx, range_idx

    for dim_idx in range(len(ranges)):
        c_dim_idx = <uint32_t>dim_idx
        dim_ranges = ranges[dim_idx]

        # skip empty dimensions
        if len(dim_ranges) == 0:
            continue

        for range_idx in range(len(dim_ranges)):
            cur_range = dim_ranges[range_idx]
            if len(cur_range) != 2:
                raise TileDBError("internal error: invalid sub-range: ", cur_range)

            start = np.array(cur_range[0], dtype=dim.dtype)
            end = np.array(cur_range[1], dtype=dim.dtype)

            start_ptr = np.PyArray_DATA(start)
            end_ptr = np.PyArray_DATA(end)

            rc = tiledb_query_add_range(ctx_ptr, query_ptr,
                                        dim_idx,
                                        start_ptr,
                                        end_ptr,
                                        NULL)

            if rc != TILEDB_OK:
                _raise_ctx_err(ctx_ptr, rc)
    try:
        if array.schema.sparse:
            result = execute_sparse(ctx_ptr, query_ptr, array, attr_names, coords)
        else:
            result = execute_dense(ctx_ptr, query_ptr, array, attr_names)
    finally:
        tiledb_query_free(&query_ptr)

    return result
