from common cimport *

cdef extern from "numpy/arrayobject.h":
    # Steals a reference to dtype, need to incref the dtype
    object PyArray_NewFromDescr(PyTypeObject* subtype,
                                np.dtype descr,
                                int nd,
                                np.npy_intp* dims,
                                np.npy_intp* strides,
                                void* data,
                                int flags,
                                object obj)
    # Steals a reference to dtype, need to incref the dtype
    object PyArray_Scalar(void* ptr, np.dtype descr, object itemsize)
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
    void* PyDataMem_NEW(size_t nbytes)
    void* PyDataMem_RENEW(void* data, size_t nbytes)
    void PyDataMem_FREE(void* data)

cdef _varlen_dtype_itemsize(object item):
    if (isinstance(item, np.dtype) and np.issubdtype(item, np.bytes_)):
        return sizeof(char)
    elif isinstance(item, np.dtype):
        return item.itemsize
    elif item == np.bytes_:
        return sizeof(char)
    elif item == np.unicode_:
        # Note this is just a place-holder, we call CPython API to get actual size
        return sizeof(char)

    raise TypeError("Unknown dtype itemsize for '{}'.".format(item))

cdef _varlen_cell_dtype(object var):
    cdef np.dtype _dtype
    if isinstance(var, np.ndarray):
        _dtype = var.dtype
        if np.issubdtype(_dtype, np.bytes_):
            # handles 'S[n]' dtypes for all n
            return np.bytes_
        elif np.issubdtype(_dtype, np.unicode_):
            # handles 'U[n]' dtypes for all n
            return np.unicode_
        else:
            return _dtype
    elif isinstance(var, bytes):
        return np.bytes_
    elif isinstance(var, unicode):
        return np.unicode_

    raise TypeError("Unsupported varlen cell datatype")

cdef packvar(object val):
    cdef arr = <np.ndarray?>val

    if len(arr) == 0:
        raise Exception("Empty arrays are not supported.")

    assert((arr.dtype == np.dtype('O') or
            np.issubdtype(arr.dtype, np.bytes_) or
            np.issubdtype(arr.dtype, np.unicode_)),
           "_pack_varlen_bytes: input array must be np.object or np.bytes!")


    first_dtype = _varlen_cell_dtype(arr[0])
    # item size
    cdef uint64_t el_size = _varlen_dtype_itemsize(first_dtype)

    if el_size==0:
        raise TypeError("Zero-size cell elements are not supported.")

    # total buffer size
    cdef uint64_t buffer_size = 0
    cdef np.ndarray buffer_offsets = np.empty(len(arr), dtype=np.uint64)
    cdef uint64_t el_buffer_size = 0

    # first pass: check types and calculate offsets
    for (i, item) in enumerate(arr):
        if first_dtype != _varlen_cell_dtype(item):
            msg = ("Data types of variable-length sub-arrays must be consistent. "
                   "Type '{}', of 1st sub-array, is inconsistent with type '{}', of item {}."
                   ).format(first_dtype, _varlen_cell_dtype(item), i)

            raise TypeError(msg)

        # current offset is last buffer_size
        buffer_offsets[i] = buffer_size

        if first_dtype == np.unicode_:
            # this will cache the materialized (if any) UTF8 object
            if PY_MAJOR_VERSION >= 3:
                utf8 = (<str>item).encode('UTF-8')
            else:
                utf8 = (<unicode>item).encode('UTF-8')
            el_buffer_size = len(utf8)
        else:
            el_buffer_size = el_size * len(item)
        assert(el_buffer_size > 0)

        # *running total* buffer size
        buffer_size += el_buffer_size

    # return a numpy buffer because that is what the caller uses for non-varlen buffers
    cdef np.ndarray buffer = np.zeros(shape=buffer_size, dtype=np.uint8)
    # <TODO> should be np.empty(shape=buffer_size, dtype=np.uint8)
    cdef char* buffer_ptr = <char*>np.PyArray_DATA(buffer)
    cdef char* input_ptr = NULL
    cdef object tmp_utf8 = None

    # bytes to copy in this block
    cdef uint64_t nbytes = 0
    # loop over sub-items and copy into buffer
    for (i, subarray) in enumerate(val):
        if (isinstance(subarray, bytes) or
                (isinstance(subarray, np.ndarray) and np.issubdtype(subarray.dtype, np.bytes_))):
            input_ptr = <char*>PyBytes_AS_STRING(subarray)
        elif (isinstance(subarray, str) or (isinstance(subarray, unicode)) or
              (isinstance(subarray, np.ndarray) and np.issubdtype(subarray.dtype, np.unicode_))):
            tmp_utf8 = subarray.encode("UTF-8")
            input_ptr = <char*>tmp_utf8
        else:
            input_ptr = <char*>np.PyArray_DATA(subarray)

        if i == len(val)-1:
            nbytes = buffer_size - buffer_offsets[i]
        else:
            nbytes = buffer_offsets[i+1] - buffer_offsets[i]

        memcpy(buffer_ptr, input_ptr, nbytes)
        buffer_ptr += nbytes
        # clean up the encoded object *after* storing
        if tmp_utf8:
            del tmp_utf8

    return buffer, buffer_offsets