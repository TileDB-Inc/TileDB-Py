#!python
#cython: embedsignature=True
#cython: auto_pickle=False

from cpython.pycapsule cimport PyCapsule_GetPointer
from cpython.version cimport PY_MAJOR_VERSION

include "common.pxi"
include "indexing.pyx"

from .cc import TileDBError
from .ctx import Config, Ctx, default_ctx
from .array import Array

###############################################################################
#     Numpy initialization code (critical)                                    #
###############################################################################

# https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.import_array
np.import_array()

###############################################################################
#    Utility/setup                                                            #
###############################################################################

# Use unified numpy printing
np.set_printoptions(legacy="1.21" if np.lib.NumpyVersion(np.__version__) >= "1.22.0" else False)


cdef tiledb_ctx_t* safe_ctx_ptr(object ctx):
    if ctx is None:
        raise TileDBError("internal error: invalid Ctx object")
    return <tiledb_ctx_t*>PyCapsule_GetPointer(ctx.__capsule__(), "ctx")

def version():
    """Return the version of the linked ``libtiledb`` shared library

    :rtype: tuple
    :return: Semver version (major, minor, rev)

    """
    cdef:
        int major = 0
        int minor = 0
        int rev = 0
    tiledb_version(&major, &minor, &rev)
    return major, minor, rev


# note: this function is cdef, so it must return a python object in order to
#       properly forward python exceptions raised within the function. See:
#       https://cython.readthedocs.io/en/latest/src/userguide/language_basics.html#error-return-values
cdef dict get_query_fragment_info(tiledb_ctx_t* ctx_ptr,
                                   tiledb_query_t* query_ptr):

    cdef int rc = TILEDB_OK
    cdef uint32_t num_fragments
    cdef Py_ssize_t fragment_idx
    cdef const char* fragment_uri_ptr
    cdef unicode fragment_uri
    cdef uint64_t fragment_t1, fragment_t2
    cdef dict result = dict()

    rc = tiledb_query_get_fragment_num(ctx_ptr, query_ptr, &num_fragments)
    if rc != TILEDB_OK:
        _raise_ctx_err(ctx_ptr, rc)

    if (num_fragments < 1):
        return result

    for fragment_idx in range(0, num_fragments):

        rc = tiledb_query_get_fragment_uri(ctx_ptr, query_ptr, fragment_idx, &fragment_uri_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        rc = tiledb_query_get_fragment_timestamp_range(
                ctx_ptr, query_ptr, fragment_idx, &fragment_t1, &fragment_t2)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        fragment_uri = fragment_uri_ptr.decode('UTF-8')
        result[fragment_uri] = (fragment_t1, fragment_t2)

    return result

def _write_array_wrapper(
        object tiledb_array,
        object subarray,
        list coordinates,
        list buffer_names,
        list values,
        dict labels,
        dict nullmaps,
        bint issparse,
    ):

    cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(tiledb_array.ctx)
    cdef tiledb_array_t* array_ptr = <tiledb_array_t*>PyCapsule_GetPointer(tiledb_array.array.__capsule__(), "array")
    cdef dict fragment_info = tiledb_array.last_fragment_info
    _write_array(ctx_ptr, array_ptr, tiledb_array, subarray, coordinates, buffer_names, values, labels, nullmaps, fragment_info, issparse)

cdef _write_array(
        tiledb_ctx_t* ctx_ptr,
        tiledb_array_t* array_ptr,
        object tiledb_array,
        object subarray,
        list coordinates,
        list buffer_names,
        list values,
        dict labels,
        dict nullmaps,
        dict fragment_info,
        bint issparse,
    ):

    # used for buffer conversion (local import to avoid circularity)
    from .main import array_to_buffer

    cdef bint isfortran = False
    cdef Py_ssize_t nattr = len(buffer_names)
    cdef Py_ssize_t nlabel = len(labels)

    # Create arrays to hold buffer sizes
    cdef Py_ssize_t nbuffer = nattr + nlabel
    if issparse:
        nbuffer += tiledb_array.schema.ndim
    cdef np.ndarray buffer_sizes = np.zeros((nbuffer,), dtype=np.uint64)
    cdef np.ndarray buffer_offsets_sizes = np.zeros((nbuffer,),  dtype=np.uint64)
    cdef np.ndarray nullmaps_sizes = np.zeros((nbuffer,), dtype=np.uint64)

    # Create lists for data and offset buffers
    output_values = list()
    output_offsets = list()

    # Set data and offset buffers for attributes
    for i in range(nattr):
        # if dtype is ASCII, ensure all characters are valid
        if tiledb_array.schema.attr(i).isascii:
            try:
                values[i] = np.asarray(values[i], dtype=np.bytes_)
            except Exception as exc:
                raise TileDBError(f'dtype of attr {tiledb_array.schema.attr(i).name} is "ascii" but attr_val contains invalid ASCII characters')

        attr = tiledb_array.schema.attr(i)

        if attr.isvar:
            try:
                if attr.isnullable:
                    if(np.issubdtype(attr.dtype, np.str_) 
                        or np.issubdtype(attr.dtype, np.bytes_)):
                        attr_val = np.array(["" if v is None else v for v in values[i]])
                    else:
                        attr_val = np.nan_to_num(values[i])
                else:
                    attr_val = values[i]
                buffer, offsets = array_to_buffer(attr_val, True, False)
            except Exception as exc:
                raise type(exc)(f"Failed to convert buffer for attribute: '{attr.name}'") from exc
            buffer_offsets_sizes[i] = offsets.nbytes
        else:
            buffer, offsets = values[i], None

        buffer_sizes[i] = buffer.nbytes
        output_values.append(buffer)
        output_offsets.append(offsets)

    # Check value layouts
    if len(values) and nattr > 1:
        value = output_values[0]
        isfortran = value.ndim > 1 and value.flags.f_contiguous
        for value in values:
            if value.ndim > 1 and value.flags.f_contiguous and not isfortran:
                raise ValueError("mixed C and Fortran array layouts")

    # Set data and offsets buffers for dimensions (sparse arrays only)
    ibuffer = nattr
    if issparse:
        for dim_idx, coords in enumerate(coordinates):
            if tiledb_array.schema.domain.dim(dim_idx).isvar:
                buffer, offsets = array_to_buffer(coords, True, False)
                buffer_sizes[ibuffer] = buffer.nbytes
                buffer_offsets_sizes[ibuffer] = offsets.nbytes
            else:
                buffer, offsets = coords, None
                buffer_sizes[ibuffer] = buffer.nbytes
            output_values.append(buffer)
            output_offsets.append(offsets)

            name = tiledb_array.schema.domain.dim(dim_idx).name
            buffer_names.append(name)

            ibuffer = ibuffer + 1

    for label_name, label_values in labels.items():
        # Append buffer name
        buffer_names.append(label_name)
        # Get label data buffer and offsets buffer for the labels
        dim_label = tiledb_array.schema.dim_label(label_name)
        if dim_label.isvar:
            buffer, offsets = array_to_buffer(label_values, True, False)
            buffer_sizes[ibuffer] = buffer.nbytes
            buffer_offsets_sizes[ibuffer] = offsets.nbytes
        else:
            buffer, offsets = label_values, None
            buffer_sizes[ibuffer] = buffer.nbytes
        # Append the buffers
        output_values.append(buffer)
        output_offsets.append(offsets)

        ibuffer = ibuffer + 1


    # Allocate the query
    cdef int rc = TILEDB_OK
    cdef tiledb_query_t* query_ptr = NULL
    rc = tiledb_query_alloc(ctx_ptr, array_ptr, TILEDB_WRITE, &query_ptr)
    if rc != TILEDB_OK:
        _raise_ctx_err(ctx_ptr, rc)

    # Set layout
    cdef tiledb_layout_t layout = (
            TILEDB_UNORDERED
            if issparse
            else (TILEDB_COL_MAJOR if isfortran else TILEDB_ROW_MAJOR)
    )
    rc = tiledb_query_set_layout(ctx_ptr, query_ptr, layout)
    if rc != TILEDB_OK:
        tiledb_query_free(&query_ptr)
        _raise_ctx_err(ctx_ptr, rc)

    # Create and set the subarray for the query (dense arrays only)
    cdef np.ndarray s_start
    cdef np.ndarray s_end
    cdef np.dtype dim_dtype = None
    cdef void* s_start_ptr = NULL
    cdef void* s_end_ptr = NULL
    cdef tiledb_subarray_t* subarray_ptr = NULL
    if not issparse:
        subarray_ptr = <tiledb_subarray_t*>PyCapsule_GetPointer(
                subarray.__capsule__(), "subarray")
        # Set the subarray on the query
        rc = tiledb_query_set_subarray_t(ctx_ptr, query_ptr, subarray_ptr)
        if rc != TILEDB_OK:
            tiledb_query_free(&query_ptr)
            _raise_ctx_err(ctx_ptr, rc)

    # Set buffers on the query
    cdef bytes bname
    cdef void* buffer_ptr = NULL
    cdef uint64_t* offsets_buffer_ptr = NULL
    cdef uint8_t* nulmap_buffer_ptr = NULL
    cdef uint64_t* buffer_sizes_ptr = <uint64_t*> np.PyArray_DATA(buffer_sizes)
    cdef uint64_t* offsets_buffer_sizes_ptr = <uint64_t*> np.PyArray_DATA(buffer_offsets_sizes)
    cdef uint64_t* nullmaps_sizes_ptr = <uint64_t*> np.PyArray_DATA(nullmaps_sizes)
    try:
        for i, buffer_name in enumerate(buffer_names):
            # Get utf-8 version of the name for C-API calls
            bname = buffer_name.encode('UTF-8')

            # Set data buffer
            buffer_ptr = np.PyArray_DATA(output_values[i])
            rc = tiledb_query_set_data_buffer(
                    ctx_ptr, query_ptr, bname, buffer_ptr, &(buffer_sizes_ptr[i]))
            if rc != TILEDB_OK:
                _raise_ctx_err(ctx_ptr, rc)

            # Set offsets buffer
            if output_offsets[i] is not None:
                offsets_buffer_ptr = <uint64_t*>np.PyArray_DATA(output_offsets[i])
                rc = tiledb_query_set_offsets_buffer(
                        ctx_ptr,
                        query_ptr,
                        bname,
                        offsets_buffer_ptr,
                        &(offsets_buffer_sizes_ptr[i])
                )
                if rc != TILEDB_OK:
                    _raise_ctx_err(ctx_ptr, rc)

            # Set validity buffer
            if buffer_name in nullmaps:
                # NOTE: validity map is owned *by the caller*
                nulmap = nullmaps[buffer_name]
                nullmaps_sizes[i] = len(nulmap)
                nulmap_buffer_ptr = <uint8_t*>np.PyArray_DATA(nulmap)
                rc = tiledb_query_set_validity_buffer(
                    ctx_ptr,
                    query_ptr,
                    bname,
                    nulmap_buffer_ptr,
                    &(nullmaps_sizes_ptr[i])
                )
                if rc != TILEDB_OK:
                    _raise_ctx_err(ctx_ptr, rc)

        with nogil:
            rc = tiledb_query_submit(ctx_ptr, query_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        rc = tiledb_query_finalize(ctx_ptr, query_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        if fragment_info is not False:
            assert(type(fragment_info) is dict)
            fragment_info.clear()
            fragment_info.update(get_query_fragment_info(ctx_ptr, query_ptr))

    finally:
        tiledb_query_free(&query_ptr)
    return

cdef _raise_tiledb_error(tiledb_error_t* err_ptr):
    cdef const char* err_msg_ptr = NULL
    ret = tiledb_error_message(err_ptr, &err_msg_ptr)
    if ret != TILEDB_OK:
        tiledb_error_free(&err_ptr)
        if ret == TILEDB_OOM:
            raise MemoryError()
        raise TileDBError("error retrieving error message")
    cdef unicode message_string
    try:
        message_string = err_msg_ptr.decode('UTF-8', 'strict')
    finally:
        tiledb_error_free(&err_ptr)
    raise TileDBError(message_string)


cdef _raise_ctx_err(tiledb_ctx_t* ctx_ptr, int rc):
    if rc == TILEDB_OK:
        return
    if rc == TILEDB_OOM:
        raise MemoryError()
    cdef tiledb_error_t* err_ptr = NULL
    cdef int ret = tiledb_ctx_get_last_error(ctx_ptr, &err_ptr)
    if ret != TILEDB_OK:
        tiledb_error_free(&err_ptr)
        if ret == TILEDB_OOM:
            raise MemoryError()
        raise TileDBError("error retrieving error object from ctx")
    _raise_tiledb_error(err_ptr)


def write_direct_dense(self: Array, np.ndarray array not None, **kw):
    """
    Write directly to given array attribute with minimal checks,
    assumes that the numpy array is the same shape as the array's domain

    :param np.ndarray array: Numpy contiguous dense array of the same dtype \
        and shape and layout of the DenseArray instance
    :raises ValueError: array is not contiguous
    :raises: :py:exc:`tiledb.TileDBError`

    """
    append_dim = kw.pop("append_dim", None)
    mode = kw.pop("mode", "ingest")
    start_idx = kw.pop("start_idx", None)

    if not self.isopen or self.mode != 'w':
        raise TileDBError("DenseArray is not opened for writing")
    if self.schema.nattr != 1:
        raise ValueError("cannot write_direct to a multi-attribute DenseArray")
    if not array.flags.c_contiguous and not array.flags.f_contiguous:
        raise ValueError("array is not contiguous")

    cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)
    cdef tiledb_array_t* array_ptr = <tiledb_array_t*>PyCapsule_GetPointer(self.array.__capsule__(), "array")

    # attr name
    attr = self.schema.attr(0)
    cdef bytes battr_name = attr._internal_name.encode('UTF-8')
    cdef const char* attr_name_ptr = PyBytes_AS_STRING(battr_name)

    cdef void* buff_ptr = np.PyArray_DATA(array)
    cdef uint64_t buff_size = array.nbytes
    cdef np.ndarray subarray = np.zeros(2*array.ndim, np.uint64)

    try:
        use_global_order = self.ctx.config().get(
            "py.use_global_order_1d_write") == "true"
    except KeyError:
        use_global_order = False

    cdef tiledb_layout_t layout = TILEDB_ROW_MAJOR
    if array.ndim == 1 and use_global_order:
        layout = TILEDB_GLOBAL_ORDER
    elif array.flags.f_contiguous:
        layout = TILEDB_COL_MAJOR

    cdef tiledb_query_t* query_ptr = NULL
    cdef tiledb_subarray_t* subarray_ptr = NULL
    cdef int rc = TILEDB_OK
    rc = tiledb_query_alloc(ctx_ptr, array_ptr, TILEDB_WRITE, &query_ptr)
    if rc != TILEDB_OK:
        tiledb_query_free(&query_ptr)
        _raise_ctx_err(ctx_ptr, rc)
    try:
        rc = tiledb_query_set_layout(ctx_ptr, query_ptr, layout)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        range_start_idx = start_idx or 0
        for n in range(array.ndim):
            subarray[n*2] = range_start_idx
            subarray[n*2 + 1] = array.shape[n] + range_start_idx - 1

        if mode == "append":
            with Array.load_typed(self.uri) as A:
                ned = A.nonempty_domain()

            if array.ndim <= append_dim:
                raise IndexError("`append_dim` out of range")

            if array.ndim != len(ned):
                raise ValueError(
                    "The number of dimension of the TileDB array and "
                    "Numpy array to append do not match"
                )

            for n in range(array.ndim):
                if n == append_dim:
                    if start_idx is not None:
                        range_start_idx = start_idx
                        range_end_idx = array.shape[n] + start_idx -1
                    else:
                        range_start_idx = ned[n][1] + 1
                        range_end_idx = array.shape[n] + ned[n][1]

                    subarray[n*2] = range_start_idx
                    subarray[n*2 + 1] = range_end_idx
                else:
                    if array.shape[n] != ned[n][1] - ned[n][0] + 1:
                        raise ValueError(
                            "The input Numpy array must be of the same "
                            "shape as the TileDB array, exluding the "
                            "`append_dim`, but the Numpy array at index "
                            f"{n} has {array.shape[n]} dimension(s) and "
                            f"the TileDB array has {ned[n][1]-ned[n][0]}."
                        )

        rc = tiledb_subarray_alloc(ctx_ptr, array_ptr, &subarray_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        rc = tiledb_subarray_set_subarray(
                ctx_ptr,
                subarray_ptr,
                <void*>np.PyArray_DATA(subarray)
        )
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        rc = tiledb_query_set_subarray_t(ctx_ptr, query_ptr, subarray_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        rc = tiledb_query_set_data_buffer(
                ctx_ptr,
                query_ptr,
                attr_name_ptr,
                buff_ptr,
                &buff_size
        )
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        with nogil:
            rc = tiledb_query_submit(ctx_ptr, query_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        with nogil:
            rc = tiledb_query_finalize(ctx_ptr, query_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
    finally:
        tiledb_subarray_free(&subarray_ptr)
        tiledb_query_free(&query_ptr)
    return
