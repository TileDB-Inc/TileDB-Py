# Set true to enable modular compilation
IF TILEDBPY_MODULAR:
    include "common.pxi"
    from .libtiledb cimport *
    from cpython.version cimport PY_MAJOR_VERSION

from collections import deque

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
    cdef np.dtype dtype
    if isinstance(var, np.ndarray):
        dtype = var.dtype
        if np.issubdtype(dtype, np.bytes_):
            # handles 'S[n]' dtypes for all n
            return np.bytes_
        elif np.issubdtype(dtype, np.unicode_):
            # handles 'U[n]' dtypes for all n
            return np.unicode_
        else:
            return dtype
    elif isinstance(var, bytes):
        return np.bytes_
    elif isinstance(var, unicode):
        return np.unicode_

    try:
        actual_type = str(type(var))
    except:
        actual_type = "[failed to get type]"

    raise TypeError(f"Unsupported varlen cell datatype ('{actual_type}')")

def array_to_buffer(object val):
    cdef arr = <np.ndarray?>val

    if len(arr) == 0:
        raise Exception("Empty arrays are not supported.")

    assert((arr.dtype == np.dtype('O') or
            np.issubdtype(arr.dtype, np.bytes_) or
            np.issubdtype(arr.dtype, np.unicode_)),
           "array_to_buffer: input array must be np.object or np.bytes!")

    firstdtype = _varlen_cell_dtype(arr.flat[0])
    # item size
    cdef uint64_t el_size = _varlen_dtype_itemsize(firstdtype)

    if el_size==0:
        raise TypeError("Zero-size cell elements are not supported.")

    # total buffer size
    cdef uint64_t buffer_size = 0
    cdef uint64_t buffer_n_elem = np.prod(arr.shape)
    cdef np.ndarray buffer_offsets = np.empty(buffer_n_elem, dtype=np.uint64)
    cdef uint64_t el_buffer_size = 0
    cdef uint64_t item_len = 0

    # first pass: check types and calculate offsets
    for (i, item) in enumerate(arr.flat):
        if firstdtype != _varlen_cell_dtype(item):
            msg = ("Data types of variable-length sub-arrays must be consistent. "
                   "Type '{}', of 1st sub-array, is inconsistent with type '{}', of item {}."
                   ).format(firstdtype, _varlen_cell_dtype(item), i)

            raise TypeError(msg)

        # current offset is last buffer_size
        buffer_offsets[i] = buffer_size

        if firstdtype == np.unicode_:
            # this will cache the materialized (if any) UTF8 object
            if PY_MAJOR_VERSION >= 3:
                utf8 = (<str>item).encode('UTF-8')
            else:
                utf8 = (<unicode>item).encode('UTF-8')
            el_buffer_size = len(utf8)
        else:
            if hasattr(item, '__len__'):
                item_len = len(item)
            else:
                item_len = 1
            el_buffer_size = el_size * item_len

        if (el_buffer_size == 0) and (
                (firstdtype == np.bytes_) or
                (firstdtype == np.unicode_)):
            el_buffer_size = 1

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
    for (i, subitem) in enumerate(val.flat):
        if (isinstance(subitem, bytes) or
                (isinstance(subitem, np.ndarray) and np.issubdtype(subitem.dtype, np.bytes_))):
            input_ptr = <char*>PyBytes_AS_STRING(subitem)
        elif (isinstance(subitem, str) or (isinstance(subitem, unicode)) or
              (isinstance(subitem, np.ndarray) and np.issubdtype(subitem.dtype, np.unicode_))):
            tmp_utf8 = subitem.encode("UTF-8")
            input_ptr = <char*>tmp_utf8
        else:
            input_ptr = <char*>np.PyArray_DATA(subitem)

        if i == buffer_n_elem - 1:
            nbytes = buffer_size - buffer_offsets[i]
        else:
            nbytes = buffer_offsets[i+1] - buffer_offsets[i]

        memcpy(buffer_ptr, input_ptr, nbytes)
        buffer_ptr += nbytes
        # clean up the encoded object *after* storing
        if tmp_utf8:
            del tmp_utf8

    return buffer, buffer_offsets

cdef tiledb_datatype_t c_dtype_to_tiledb(np.dtype dtype) except? TILEDB_CHAR:
    """Return tiledb_datatype_t enum value for a given numpy dtype object
    """
    if dtype == np.int32:
        return TILEDB_INT32
    elif dtype == np.uint32:
        return TILEDB_UINT32
    elif dtype == np.int64:
        return TILEDB_INT64
    elif dtype == np.uint64:
        return TILEDB_UINT64
    elif dtype == np.float32:
        return TILEDB_FLOAT32
    elif dtype == np.float64:
        return TILEDB_FLOAT64
    elif dtype == np.int8:
        return TILEDB_INT8
    elif dtype == np.uint8:
        return TILEDB_UINT8
    elif dtype == np.int16:
        return TILEDB_INT16
    elif dtype == np.uint16:
        return TILEDB_UINT16
    elif dtype == np.unicode_:
        return TILEDB_STRING_UTF8
    elif dtype == np.bytes_:
        return TILEDB_CHAR
    elif dtype == np.complex64:
        return TILEDB_FLOAT32
    elif dtype == np.complex128:
        return TILEDB_FLOAT64
    elif dtype.kind == 'M':
        return _tiledb_dtype_datetime(dtype)
    raise TypeError("data type {0!r} not understood".format(dtype))

def dtype_to_tiledb(np.dtype dtype):
    return c_dtype_to_tiledb(dtype)

def array_type_ncells(np.dtype dtype):
    """
    Returns the TILEDB_{TYPE} and ncells corresponding to a given numpy dtype
    """

    cdef np.dtype checked_dtype = np.dtype(dtype)
    cdef uint32_t ncells

    # - flexible datatypes of unknown size have an itemsize of 0 (str, bytes, etc.)
    # - unicode and string types are always stored as VAR because we don't want to
    #   store the pad (numpy pads to max length for 'S' and 'U' dtypes)

    if np.issubdtype(checked_dtype, np.bytes_):
        tdb_type = TILEDB_CHAR
        if checked_dtype.itemsize == 0:
            ncells = TILEDB_VAR_NUM
        else:
            ncells = checked_dtype.itemsize

    elif np.issubdtype(checked_dtype, np.unicode_):
        np_unicode_size = np.dtype("U1").itemsize

        # TODO depending on np_unicode_size, tdb_type may be UTF16 or UTF32
        tdb_type = TILEDB_STRING_UTF8

        if checked_dtype.itemsize == 0:
            ncells = TILEDB_VAR_NUM
        else:    
            ncells = checked_dtype.itemsize // np_unicode_size

    elif np.issubdtype(checked_dtype, np.complexfloating):
        # handle complex dtypes
        tdb_type = dtype_to_tiledb(checked_dtype)
        ncells = 2

    elif checked_dtype.kind == 'V':
        # handles n fixed-size record dtypes
        if checked_dtype.shape != ():
            raise TypeError("nested sub-array numpy dtypes are not supported")
        # check that types are the same
        # TODO: make sure this is not too slow for large record types
        deq = deque(checked_dtype.fields.values())
        typ0, _ = deq.popleft()
        nfields = 1
        for (typ, _) in deq:
            nfields += 1
            if typ != typ0:
                raise TypeError('heterogenous record numpy dtypes are not supported')

        tdb_type = dtype_to_tiledb(typ0)
        ncells = <uint32_t>(len(checked_dtype.fields.values()))

    else:
        # scalar cell type
        tdb_type = c_dtype_to_tiledb(checked_dtype)
        ncells = 1

    return tdb_type, ncells
