#!python
#cython: embedsignature=True
#cython: auto_pickle=False

from cpython.version cimport PY_MAJOR_VERSION
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_IsValid, PyCapsule_GetPointer

include "common.pxi"
import io
import html
import sys
import warnings
from collections import OrderedDict
from collections.abc import Sequence

from .attribute import Attr
from .ctx import default_ctx, Ctx, Config
from .dimension import Dim
from .domain import Domain
from .filter import FilterList
from .vfs import VFS
from .version import version_tuple as tiledbpy_version
from .util import sparse_array_from_numpy, dtype_range

import tiledb.cc as lt
from tiledb.cc import TileDBError


###############################################################################
#     Numpy initialization code (critical)                                    #
###############################################################################

# https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.import_array
np.import_array()


###############################################################################
#    MODULAR IMPORTS                                                 #
###############################################################################

IF TILEDBPY_MODULAR:
    from .indexing import DomainIndexer
    from .libmetadata import get_metadata, load_metadata, put_metadata
    from .np2buf import array_type_ncells, dtype_to_tiledb
ELSE:
    include "indexing.pyx"
    include "np2buf.pyx"
    include "libmetadata.pyx"

###############################################################################
#    Utility/setup                                                            #
###############################################################################

# KB / MB in bytes
_KB = 1024
_MB = 1024 * _KB

# Maximum number of retries for incomplete query
_MAX_QUERY_RETRIES = 3

# The native int type for this platform
IntType = np.dtype(np.int_)

# Integer types supported by Python / System
_inttypes = (int, np.integer)

# Numpy initialization code (critical)
# https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.import_array
np.import_array()

# Conversion from TileDB dtype to Numpy datetime
_tiledb_dtype_to_datetime_convert = {
    TILEDB_DATETIME_YEAR: np.datetime64('', 'Y'),
    TILEDB_DATETIME_MONTH: np.datetime64('', 'M'),
    TILEDB_DATETIME_WEEK: np.datetime64('', 'W'),
    TILEDB_DATETIME_DAY: np.datetime64('', 'D'),
    TILEDB_DATETIME_HR: np.datetime64('', 'h'),
    TILEDB_DATETIME_MIN: np.datetime64('', 'm'),
    TILEDB_DATETIME_SEC: np.datetime64('', 's'),
    TILEDB_DATETIME_MS: np.datetime64('', 'ms'),
    TILEDB_DATETIME_US: np.datetime64('', 'us'),
    TILEDB_DATETIME_NS: np.datetime64('', 'ns'),
    TILEDB_DATETIME_PS: np.datetime64('', 'ps'),
    TILEDB_DATETIME_FS: np.datetime64('', 'fs'),
    TILEDB_DATETIME_AS: np.datetime64('', 'as')
}

# Conversion from Numpy datetime to TileDB dtype
_datetime_tiledb_dtype_convert = {
    'Y': TILEDB_DATETIME_YEAR,
    'M': TILEDB_DATETIME_MONTH,
    'W': TILEDB_DATETIME_WEEK,
    'D': TILEDB_DATETIME_DAY,
    'h': TILEDB_DATETIME_HR,
    'm': TILEDB_DATETIME_MIN,
    's': TILEDB_DATETIME_SEC,
    'ms': TILEDB_DATETIME_MS,
    'us': TILEDB_DATETIME_US,
    'ns': TILEDB_DATETIME_NS,
    'ps': TILEDB_DATETIME_PS,
    'fs': TILEDB_DATETIME_FS,
    'as': TILEDB_DATETIME_AS
}

# Conversion from TileDB dtype to Numpy typeid
_tiledb_dtype_to_numpy_typeid_convert ={
    TILEDB_INT32: np.NPY_INT32,
    TILEDB_UINT32: np.NPY_UINT32,
    TILEDB_INT64: np.NPY_INT64,
    TILEDB_UINT64: np.NPY_UINT64,
    TILEDB_FLOAT32: np.NPY_FLOAT32,
    TILEDB_FLOAT64: np.NPY_FLOAT64,
    TILEDB_INT8: np.NPY_INT8,
    TILEDB_UINT8: np.NPY_UINT8,
    TILEDB_INT16: np.NPY_INT16,
    TILEDB_UINT16: np.NPY_UINT16,
    TILEDB_CHAR: np.NPY_STRING,
    TILEDB_STRING_ASCII: np.NPY_STRING,
    TILEDB_STRING_UTF8: np.NPY_UNICODE,
}
IF LIBTILEDB_VERSION_MAJOR >= 2:
    IF LIBTILEDB_VERSION_MINOR >= 9:
        _tiledb_dtype_to_numpy_typeid_convert[TILEDB_BLOB] = np.NPY_BYTE
IF LIBTILEDB_VERSION_MAJOR >= 2:
    IF LIBTILEDB_VERSION_MINOR >= 10:
        _tiledb_dtype_to_numpy_typeid_convert[TILEDB_BOOL] = np.NPY_BOOL

# Conversion from TileDB dtype to Numpy dtype
_tiledb_dtype_to_numpy_dtype_convert = {
    TILEDB_INT32: np.int32,
    TILEDB_UINT32: np.uint32,
    TILEDB_INT64: np.int64,
    TILEDB_UINT64: np.uint64,
    TILEDB_FLOAT32: np.float32,
    TILEDB_FLOAT64: np.float64,
    TILEDB_INT8: np.int8,
    TILEDB_UINT8: np.uint8,
    TILEDB_INT16: np.int16,
    TILEDB_UINT16: np.uint16,
    TILEDB_CHAR: np.dtype('S1'),
    TILEDB_STRING_ASCII: np.dtype('S'),
    TILEDB_STRING_UTF8: np.dtype('U1'),
}
IF LIBTILEDB_VERSION_MAJOR >= 2:
    IF LIBTILEDB_VERSION_MINOR >= 9:
        _tiledb_dtype_to_numpy_dtype_convert[TILEDB_BLOB] = np.byte
IF LIBTILEDB_VERSION_MAJOR >= 2:
    IF LIBTILEDB_VERSION_MINOR >= 10:
        _tiledb_dtype_to_numpy_dtype_convert[TILEDB_BOOL] = np.bool_

cdef tiledb_ctx_t* unsafe_ctx_ptr(object ctx):
    return <tiledb_ctx_t*>PyCapsule_GetPointer(
            ctx.__capsule__(), "ctx")

cdef tiledb_ctx_t* safe_ctx_ptr(object ctx):
    if ctx is None:
        raise TileDBError("internal error: invalid Ctx object")
    return unsafe_ctx_ptr(ctx)

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

def offset_size():
    """Return the offset size (TILEDB_OFFSET_SIZE)"""
    return tiledb_offset_size()

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

cdef _write_array(tiledb_ctx_t* ctx_ptr,
                  tiledb_array_t* array_ptr,
                  object tiledb_array,
                  list coords_or_subarray,
                  list buffer_names,
                  list values,
                  dict nullmaps,
                  dict fragment_info,
                  bint issparse):

    # used for buffer conversion (local import to avoid circularity)
    import tiledb.main

    cdef bint isfortran = False
    cdef Py_ssize_t nattr = len(buffer_names)

    # Create arrays to hold buffer sizes
    cdef Py_ssize_t nbuffer = nattr
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
                buffer, offsets = tiledb.main.array_to_buffer(values[i], True, False)
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
    if issparse:
        for dim_idx, coords in enumerate(coords_or_subarray):
            if tiledb_array.schema.domain.dim(dim_idx).isvar:
                buffer, offsets = tiledb.main.array_to_buffer(coords, True, False)
                buffer_sizes[nattr + dim_idx] = buffer.nbytes
                buffer_offsets_sizes[nattr + dim_idx] = offsets.nbytes
            else:
                buffer, offsets = coords, None
                buffer_sizes[nattr + dim_idx] = buffer.nbytes
            output_values.append(buffer)
            output_offsets.append(offsets)

            name = tiledb_array.schema.domain.dim(dim_idx).name
            buffer_names.append(name)

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
        # Allocate the subarray
        rc = tiledb_subarray_alloc(ctx_ptr, array_ptr, &subarray_ptr)
        if rc != TILEDB_OK:
            tiledb_subarray_free(&subarray_ptr)
            tiledb_query_free(&query_ptr)
            _raise_ctx_err(ctx_ptr, rc)

        # Set the ranges on the subarray
        dom = tiledb_array.schema.domain
        for dim_idx, s_range in enumerate(coords_or_subarray):
            dim = dom.dim(dim_idx)
            dim_dtype = dim.dtype
            s_start = np.asarray(s_range[0], dtype=dim_dtype)
            s_end = np.asarray(s_range[1], dtype=dim_dtype)
            s_start_ptr = np.PyArray_DATA(s_start)
            s_end_ptr = np.PyArray_DATA(s_end)
            if dim.isvar:
                rc = tiledb_subarray_add_range_var(
                    ctx_ptr, subarray_ptr, dim_idx,
                    s_start_ptr,  s_start.nbytes,
                    s_end_ptr, s_end.nbytes)
            else:
                rc = tiledb_subarray_add_range(
                    ctx_ptr, subarray_ptr, dim_idx,
                    s_start_ptr, s_end_ptr, NULL)
            if rc != TILEDB_OK:
                tiledb_subarray_free(&subarray_ptr)
                tiledb_query_free(&query_ptr)
                _raise_ctx_err(ctx_ptr, rc)

        # Set the subarray on the query
        rc = tiledb_query_set_subarray_t(ctx_ptr, query_ptr, subarray_ptr)
        if rc != TILEDB_OK:
            tiledb_subarray_free(&subarray_ptr)
            tiledb_query_free(&query_ptr)
            _raise_ctx_err(ctx_ptr, rc)
        tiledb_subarray_free(&subarray_ptr)


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


cpdef check_error(object ctx, int rc):
    cdef tiledb_ctx_t* ctx_ptr = <tiledb_ctx_t*>PyCapsule_GetPointer(
            ctx.__capsule__(), "ctx")
    _raise_ctx_err(ctx_ptr, rc)

def stats_enable():
    """Enable TileDB internal statistics."""
    tiledb_stats_enable()

    import tiledb.main
    tiledb.main.init_stats()

def stats_disable():
    """Disable TileDB internal statistics."""
    tiledb_stats_disable()

    import tiledb.main
    tiledb.main.disable_stats()

def stats_reset():
    """Reset all TileDB internal statistics to 0."""
    tiledb_stats_reset()

    import tiledb.main
    tiledb.main.init_stats()

def stats_dump(version=True, print_out=True, include_python=True, json=False, verbose=True):
    """Return TileDB internal statistics as a string.

    :param include_python: Include TileDB-Py statistics
    :param print_out: Print string to console (default True), or return as string
    :param version: Include TileDB Embedded and TileDB-Py versions (default: True)
    :param json: Return stats JSON object (default: False)
    :param verbose: Print extended internal statistics (default: True)
    """
    cdef char* stats_str_ptr = NULL;

    if json or not verbose:
        if tiledb_stats_raw_dump_str(&stats_str_ptr) == TILEDB_ERR:
            raise TileDBError("Unable to dump stats to stats_str_ptr.")
    else:
        if tiledb_stats_dump_str(&stats_str_ptr) == TILEDB_ERR:
            raise TileDBError("Unable to dump stats to stats_str_ptr.")

    stats_str_core = stats_str_ptr.decode("UTF-8", "strict").strip()

    if json or not verbose:
        from json import loads as json_loads
        stats_json_core = json_loads(stats_str_core)[0]

        if include_python:
            from json import dumps as json_dumps
            import tiledb.main
            stats_json_core["python"] = json_dumps(tiledb.main.python_internal_stats(True))
        if json:
            return stats_json_core

    if tiledb_stats_free_str(&stats_str_ptr) == TILEDB_ERR:
        raise TileDBError("Unable to free stats_str_ptr.")

    stats_str = ""

    if version:
        import tiledb
        stats_str += f"TileDB Embedded Version: {tiledb.libtiledb.version()}\n"
        stats_str += f"TileDB-Py Version: {tiledb.version.version}\n"

    if not verbose:
        stats_str += "\n==== READ ====\n\n"

        import tiledb

        if tiledb.libtiledb.version() < (2, 3):
            stats_str += "- Number of read queries: {}\n".format(
                stats_json_core["READ_NUM"]
            )
            stats_str += "- Number of attributes read: {}\n".format(
                stats_json_core["READ_ATTR_FIXED_NUM"]
                + stats_json_core["READ_ATTR_VAR_NUM"]
            )
            stats_str += "- Time to compute estimated result size: {}\n".format(
                stats_json_core["READ_COMPUTE_EST_RESULT_SIZE"]
            )
            stats_str += "- Read time: {}\n".format(stats_json_core["READ"])
            stats_str += (
                "- Total read query time (array open + init state + read): {}\n".format(
                    stats_json_core["READ"] + stats_json_core["READ_INIT_STATE"]
                )
            )
        else:
            loop_num = stats_json_core["counters"][
                "Context.StorageManager.Query.Reader.loop_num"
            ]
            stats_str += f"- Number of read queries: {loop_num}\n"

            attr_num = (
                stats_json_core["counters"]["Context.StorageManager.Query.Reader.attr_num"]
                + stats_json_core["counters"][
                    "Context.StorageManager.Query.Reader.attr_fixed_num"
                ]
            )
            stats_str += f"- Number of attributes read: {attr_num}\n"

            read_compute_est_result_size = stats_json_core["timers"][
                "Context.StorageManager.Query.Subarray.read_compute_est_result_size.sum"
            ]
            stats_str += (
                f"- Time to compute estimated result size: {read_compute_est_result_size}\n"
            )

            read_tiles = stats_json_core["timers"][
                "Context.StorageManager.Query.Reader.read_tiles.sum"
            ]
            stats_str += f"- Read time: {read_tiles}\n"

            total_read = (
                stats_json_core["timers"]["Context.StorageManager.array_open_for_reads.sum"]
                + stats_json_core["timers"][
                    "Context.StorageManager.Query.Reader.init_state.sum"
                ]
                + stats_json_core["timers"][
                    "Context.StorageManager.Query.Reader.read_tiles.sum"
                ]
            )
            stats_str += (
                f"- Total read query time (array open + init state + read): {total_read}\n"
            )
    else:
        stats_str += "\n"
        stats_str += stats_str_core
        stats_str += "\n"

    if include_python:
        import tiledb.main
        stats_str += tiledb.main.python_internal_stats()

    if print_out:
        print(stats_str)
    else:
        return stats_str

cpdef unicode ustring(object s):
    """Coerce a python object to a unicode string"""

    if type(s) is unicode:
        return <unicode> s
    elif PY_MAJOR_VERSION < 3 and isinstance(s, bytes):
        return (<bytes> s).decode('ascii')
    elif isinstance(s, unicode):
        return unicode(s)
    raise TypeError(
        "ustring() must be a string or a bytes-like object"
        ", not {0!r}".format(type(s)))


cdef bytes unicode_path(object path):
    """Returns a UTF-8 encoded byte representation of a given URI path string"""
    return ustring(path).encode('UTF-8')


def safe_repr(obj):
    """repr an object, without raising exception. Return placeholder string on failure"""

    try:
        return repr(obj)
    except:
        return "<repr failed>"

###############################################################################
#                                                                             #
#    CLASS DEFINITIONS                                                        #
#                                                                             #
###############################################################################

def _tiledb_datetime_extent(begin, end):
    """
    Returns the integer extent of a datetime range.

    :param begin: beginning of datetime range
    :type begin: numpy.datetime64
    :param end: end of datetime range
    :type end: numpy.datetime64
    :return: Extent of range, returned as an integer number of time units
    :rtype: int
    """
    extent = end - begin + 1
    date_unit = np.datetime_data(extent.dtype)[0]
    one = np.timedelta64(1, date_unit)
    # Dividing a timedelta by 1 will convert the timedelta to an integer
    return int(extent / one)


cdef bint _tiledb_type_is_datetime(tiledb_datatype_t tiledb_type) except? False:
    """Returns True if the tiledb type is a datetime type"""
    return tiledb_type in (TILEDB_DATETIME_YEAR, TILEDB_DATETIME_MONTH,
        TILEDB_DATETIME_WEEK, TILEDB_DATETIME_DAY, TILEDB_DATETIME_HR,
        TILEDB_DATETIME_MIN, TILEDB_DATETIME_SEC, TILEDB_DATETIME_MS,
        TILEDB_DATETIME_US, TILEDB_DATETIME_NS, TILEDB_DATETIME_PS,
        TILEDB_DATETIME_FS, TILEDB_DATETIME_AS)

def _tiledb_type_to_datetime(tiledb_datatype_t tiledb_type):
    """
    Return a datetime64 with appropriate unit for the given
    tiledb_datetype_t enum value
    """
    tdb_type = _tiledb_dtype_to_datetime_convert.get(tiledb_type, None)
    if tdb_type is None:
        raise TypeError("tiledb type is not a datetime {0!r}".format(tiledb_type))
    return tdb_type

cdef tiledb_datatype_t _tiledb_dtype_datetime(np.dtype dtype) except? TILEDB_DATETIME_YEAR:
    """Return tiledb_datetype_t enum value for a given np.datetime64 dtype"""
    if dtype.kind != 'M':
        raise TypeError("data type {0!r} not a datetime".format(dtype))

    date_unit = np.datetime_data(dtype)[0]
    if date_unit == 'generic':
        raise TypeError("datetime {0!r} does not specify a date unit".format(dtype))

    tdb_dt = _datetime_tiledb_dtype_convert.get(date_unit, None)
    if tdb_dt is None:
        raise TypeError("np type is not a datetime {0!r}".format(date_unit))
    return tdb_dt

cdef int _numpy_typeid(tiledb_datatype_t tiledb_dtype):
    """Return a numpy type num (int) given a tiledb_datatype_t enum value."""
    np_id_type = _tiledb_dtype_to_numpy_typeid_convert.get(tiledb_dtype, None)
    if np_id_type is not None:
        return np_id_type
    return np.NPY_DATETIME if _tiledb_type_is_datetime(tiledb_dtype) else np.NPY_NOTYPE

cdef _numpy_dtype(tiledb_datatype_t tiledb_dtype, cell_size = 1):
    """Return a numpy type given a tiledb_datatype_t enum value."""
    cdef base_dtype
    cdef uint32_t cell_val_num = cell_size

    if tiledb_dtype == TILEDB_BLOB:
        return np.bytes_

    elif cell_val_num == 1:
        if tiledb_dtype in _tiledb_dtype_to_numpy_dtype_convert:
            return _tiledb_dtype_to_numpy_dtype_convert[tiledb_dtype]
        elif _tiledb_type_is_datetime(tiledb_dtype):
            return _tiledb_type_to_datetime(tiledb_dtype)

    elif cell_val_num == 2 and tiledb_dtype == TILEDB_FLOAT32:
        return np.complex64

    elif cell_val_num == 2 and tiledb_dtype == TILEDB_FLOAT64:
        return np.complex128

    elif tiledb_dtype in (TILEDB_CHAR, TILEDB_STRING_UTF8):
        if tiledb_dtype == TILEDB_CHAR:
            dtype_str = '|S'
        elif tiledb_dtype == TILEDB_STRING_UTF8:
            dtype_str = '|U'
        if cell_val_num != TILEDB_VAR_NUM:
            dtype_str += str(cell_val_num)
        return np.dtype(dtype_str)

    elif cell_val_num == TILEDB_VAR_NUM:
        base_dtype = _numpy_dtype(tiledb_dtype, cell_size=1)
        return base_dtype

    elif cell_val_num > 1:
        # construct anonymous record dtype
        base_dtype = _numpy_dtype(tiledb_dtype, cell_size=1)
        rec = np.dtype([('', base_dtype)] * cell_val_num)
        return  rec

    raise TypeError("tiledb datatype not understood")

"""
cdef _numpy_scalar(tiledb_datatype_t typ, void* data, uint64_t nbytes):
    # Return a numpy scalar object from a tiledb_datatype_t enum type value and void pointer to scalar data
    if typ == TILEDB_CHAR:
        # bytes type, ensure a full copy
        return PyBytes_FromStringAndSize(<char*> data, nbytes)
    # fixed size numeric type
    cdef int type_num = _numpy_type_num(typ)
    return PyArray_Scalar(data, np.PyArray_DescrFromType(type_num), None)
"""

cdef tiledb_layout_t _tiledb_layout(object order) except TILEDB_UNORDERED:
    """Return the tiledb_layout_t enum value given a layout string label."""
    if order == "row-major" or order == 'C':
        return TILEDB_ROW_MAJOR
    elif order == "col-major" or order == 'F':
        return TILEDB_COL_MAJOR
    elif order == "global":
        return TILEDB_GLOBAL_ORDER
    elif order == "hilbert" or order == 'H':
        return TILEDB_HILBERT
    elif order == None or order == "unordered" or order == 'U':
        return TILEDB_UNORDERED

    raise ValueError("unknown tiledb layout: {0!r}".format(order))


cdef unicode _tiledb_layout_string(tiledb_layout_t order):
    """
    Return the unicode string label given a tiledb_layout_t enum value
    """
    tiledb_order_to_string ={
        TILEDB_ROW_MAJOR: u"row-major",
        TILEDB_COL_MAJOR: u"col-major",
        TILEDB_GLOBAL_ORDER: u"global",
        TILEDB_UNORDERED: u"unordered",
        TILEDB_HILBERT: u"hilbert"
    }

    if order not in tiledb_order_to_string:
        raise ValueError("unknown tiledb order: {0!r}".format(order))

    return tiledb_order_to_string[order]

def index_as_tuple(idx):
    """Forces scalar index objects to a tuple representation"""
    if isinstance(idx, tuple):
        return idx
    return (idx,)


def replace_ellipsis(ndim: int, idx: tuple):
    """
    Replace indexing ellipsis object with slice objects to match the number
    of dimensions.
    """
    # count number of ellipsis
    n_ellip = sum(1 for i in idx if i is Ellipsis)
    if n_ellip > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    elif n_ellip == 1:
        n = len(idx)
        if (n - 1) >= ndim:
            # does nothing, strip it out
            idx = tuple(i for i in idx if i is not Ellipsis)
        else:
            # locate where the ellipse is, count the number of items to left and right
            # fill in whole dim slices up to th ndim of the array
            left = idx.index(Ellipsis)
            right = n - (left + 1)
            new_idx = idx[:left] + ((slice(None),) * (ndim - (n - 1)))
            if right:
                new_idx += idx[-right:]
            idx = new_idx
    idx_ndim = len(idx)
    if idx_ndim < ndim:
        idx += (slice(None),) * (ndim - idx_ndim)
    if len(idx) > ndim:
        raise IndexError("too many indices for array")
    return idx


def replace_scalars_slice(dom, idx: tuple):
    """Replace scalar indices with slice objects"""
    new_idx, drop_axes = [], []
    for i in range(dom.ndim):
        dim = dom.dim(i)
        dim_idx = idx[i]
        if np.isscalar(dim_idx):
            drop_axes.append(i)
            if isinstance(dim_idx, _inttypes):
                start = int(dim_idx)
                if start < 0:
                    start += int(dim.domain[1]) + 1
                stop = start + 1
            else:
                start = dim_idx
                stop = dim_idx
            new_idx.append(slice(start, stop, None))
        else:
            new_idx.append(dim_idx)
    return tuple(new_idx), tuple(drop_axes)


def index_domain_subarray(array: Array, dom, idx: tuple):
    """
    Return a numpy array representation of the tiledb subarray buffer
    for a given domain and tuple of index slices
    """
    ndim = dom.ndim
    if len(idx) != ndim:
        raise IndexError("number of indices does not match domain rank: "
                         "(got {!r}, expected: {!r})".format(len(idx), ndim))

    subarray = list()
    for r in range(ndim):
        # extract lower and upper bounds for domain dimension extent
        dim = dom.dim(r)
        dim_dtype = dim.dtype

        if array.mode == 'r' and (np.issubdtype(dim_dtype, np.unicode_) or np.issubdtype(dim_dtype, np.bytes_)):
            # NED can only be retrieved in read mode
            ned = array.nonempty_domain()
            (dim_lb, dim_ub) = ned[r] if ned else (None, None)
        else:
            (dim_lb, dim_ub) = dim.domain

        dim_slice = idx[r]
        if not isinstance(dim_slice, slice):
            raise IndexError("invalid index type: {!r}".format(type(dim_slice)))

        start, stop, step = dim_slice.start, dim_slice.stop, dim_slice.step

        if np.issubdtype(dim_dtype, np.str_) or np.issubdtype(dim_dtype, np.bytes_):
            if start is None or stop is None:
                if start is None:
                    start = dim_lb
                if stop is None:
                    stop = dim_ub
            elif not isinstance(start, (bytes,unicode)) or not isinstance(stop, (bytes,unicode)):
                raise TileDBError(f"Non-string range '({start},{stop})' provided for string dimension '{dim.name}'")
            subarray.append((start,stop))
            continue

        if step and array.schema.sparse:
           raise IndexError("steps are not supported for sparse arrays")

        # Datetimes will be treated specially
        is_datetime = (dim_dtype.kind == 'M')

        # Promote to a common type
        if start is not None and stop is not None:
            if type(start) != type(stop):
                promoted_dtype = np.promote_types(type(start), type(stop))
                start = np.array(start, dtype=promoted_dtype, ndmin=1)[0]
                stop = np.array(stop, dtype=promoted_dtype, ndmin=1)[0]

        if start is not None:
            if is_datetime and not isinstance(start, np.datetime64):
                raise IndexError('cannot index datetime dimension with non-datetime interval')
            # don't round / promote fp slices
            if np.issubdtype(dim_dtype, np.integer):
                if isinstance(start, (np.float32, np.float64)):
                    raise IndexError("cannot index integral domain dimension with floating point slice")
                elif not isinstance(start, _inttypes):
                    raise IndexError("cannot index integral domain dimension with non-integral slice (dtype: {})".format(type(start)))
            # apply negative indexing (wrap-around semantics)
            if not is_datetime and start < 0:
                start += int(dim_ub) + 1
            if start < dim_lb:
                # numpy allows start value < the array dimension shape,
                # clamp to lower bound of dimension domain
                #start = dim_lb
                raise IndexError("index out of bounds <todo>")
        else:
            start = dim_lb
        if stop is not None:
            if is_datetime and not isinstance(stop, np.datetime64):
                raise IndexError('cannot index datetime dimension with non-datetime interval')
            # don't round / promote fp slices
            if np.issubdtype(dim_dtype, np.integer):
                if isinstance(start, (np.float32, np.float64)):
                    raise IndexError("cannot index integral domain dimension with floating point slice")
                elif not isinstance(start, _inttypes):
                    raise IndexError("cannot index integral domain dimension with non-integral slice (dtype: {})".format(type(start)))
            if not is_datetime and stop < 0:
                stop += dim_ub
            if stop > dim_ub:
                # numpy allows stop value > than the array dimension shape,
                # clamp to upper bound of dimension domain
                if is_datetime:
                    stop = dim_ub
                else:
                    stop = int(dim_ub) + 1
        else:
            if np.issubdtype(dim_dtype, np.floating) or is_datetime:
                stop = dim_ub
            else:
                stop = int(dim_ub) + 1

        if np.issubdtype(type(stop), np.floating):
            # inclusive bounds for floating point / datetime ranges
            start = dim_dtype.type(start)
            stop = dim_dtype.type(stop)
            subarray.append((start, stop))
        elif is_datetime:
            # need to ensure that datetime ranges are in the units of dim_dtype
            # so that add_range and output shapes work correctly
            start = start.astype(dim_dtype)
            stop = stop.astype(dim_dtype)
            subarray.append((start,stop))
        elif np.issubdtype(type(stop), np.integer):
            # normal python indexing semantics
            subarray.append((start, int(stop) - 1))
        else:
            raise IndexError("domain indexing is defined for integral and floating point values")
    return subarray


cdef class ArraySchema(object):
    """
    Schema class for TileDB dense / sparse array representations

    :param domain: Domain of schema
    :type attrs: tuple(tiledb.Attr, ...)
    :param cell_order:  TileDB label for cell layout
    :type cell_order: 'row-major' (default) or 'C', 'col-major' or 'F' or 'hilbert'
    :param tile_order:  TileDB label for tile layout
    :type tile_order: 'row-major' (default) or 'C', 'col-major' or 'F'
    :param int capacity: tile cell capacity
    :param offsets_filters: (default None) offsets filter list
    :type offsets_filters: tiledb.FilterList
    :param validity_filters: (default None) validity filter list
    :type validity_filters: tiledb.FilterList
    :param bool allows_duplicates: True if duplicates are allowed
    :param bool sparse: True if schema is sparse, else False \
        (set by SparseArray and DenseArray derived classes)
    :param tiledb.Ctx ctx: A TileDB Context
    :raises: :py:exc:`tiledb.TileDBError`

    """
    def __init__(self,
                 domain=None,
                 attrs=(),
                 cell_order='row-major',
                 tile_order='row-major',
                 capacity=0,
                 coords_filters=None,
                 offsets_filters=None,
                 validity_filters=None,
                 allows_duplicates=False,
                 sparse=False,
                 ctx=None):
        if ctx is None:
            ctx = default_ctx()

        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(ctx)
        cdef tiledb_array_type_t array_type =\
            TILEDB_SPARSE if sparse else TILEDB_DENSE
        cdef tiledb_array_schema_t* schema_ptr = NULL
        check_error(ctx,
                    tiledb_array_schema_alloc(ctx_ptr, array_type, &schema_ptr))
        cdef tiledb_layout_t cell_layout = TILEDB_ROW_MAJOR
        cdef tiledb_layout_t tile_layout = TILEDB_ROW_MAJOR
        try:
            cell_layout = _tiledb_layout(cell_order if cell_order else 'row-major')
            tile_layout = _tiledb_layout(tile_order if tile_order else 'row-major')
            check_error(ctx, tiledb_array_schema_set_cell_order(ctx_ptr, schema_ptr, cell_layout))
            check_error(ctx, tiledb_array_schema_set_tile_order(ctx_ptr, schema_ptr, tile_layout))
        except:
            tiledb_array_schema_free(&schema_ptr)
            raise
        cdef uint64_t _capacity = 0
        if capacity > 0:
            try:
                _capacity = <uint64_t> capacity
                check_error(ctx,
                    tiledb_array_schema_set_capacity(ctx_ptr, schema_ptr, _capacity))
            except:
                tiledb_array_schema_free(&schema_ptr)
                raise

        cdef bint ballows_dups = 0
        if allows_duplicates:
            ballows_dups = 1
            tiledb_array_schema_set_allows_dups(ctx_ptr, schema_ptr, ballows_dups)

        if not isinstance(domain, Domain):
            raise TypeError("'domain' must be an instance of Domain (domain is: '{}')".format(domain))
        cdef tiledb_domain_t* domain_ptr = <tiledb_domain_t *>PyCapsule_GetPointer(
                        domain.__capsule__(), "dom")

        cdef tiledb_domain_t* dom_with_coords_filters_ptr = NULL;
        cdef unsigned int ndim = 0
        cdef tiledb_dimension_t* dim_ptr = NULL
        try:
            if offsets_filters is not None:
                if not isinstance(offsets_filters, FilterList):
                    offsets_filters = FilterList(offsets_filters, ctx=ctx)
                filter_list = offsets_filters
                filter_list_ptr = <tiledb_filter_list_t *>PyCapsule_GetPointer(
                        filter_list.__capsule__(), "fl")
                check_error(ctx,
                    tiledb_array_schema_set_offsets_filter_list(ctx_ptr, schema_ptr, filter_list_ptr))

            if coords_filters is not None:
                warnings.warn(
                    "coords_filters is deprecated; "
                    "set the FilterList for each dimension",
                    DeprecationWarning,
                )

                filter_list = FilterList()
                filter_list_ptr = <tiledb_filter_list_t *>PyCapsule_GetPointer(
                        filter_list.__capsule__(), "fl")
                check_error(ctx,
                    tiledb_array_schema_set_coords_filter_list(ctx_ptr, schema_ptr, filter_list_ptr))

                check_error(ctx,
                    tiledb_domain_get_ndim(ctx_ptr, domain_ptr, &ndim))

                if not isinstance(coords_filters, FilterList):
                    coords_filters = FilterList(coords_filters, ctx=ctx)
                filter_list = coords_filters
                filter_list_ptr = <tiledb_filter_list_t *>PyCapsule_GetPointer(
                filter_list.__capsule__(), "fl")

                tiledb_domain_alloc(ctx_ptr, &dom_with_coords_filters_ptr)
                for dim_id in range(ndim):
                    check_error(ctx,
                            tiledb_domain_get_dimension_from_index(
                                ctx_ptr, domain_ptr, dim_id, &dim_ptr))
                    check_error(ctx,
                        tiledb_dimension_set_filter_list(
                            ctx_ptr, dim_ptr, filter_list_ptr))
                    check_error(ctx,
                        tiledb_domain_add_dimension(
                            ctx_ptr, dom_with_coords_filters_ptr, dim_ptr))
                domain_ptr = dom_with_coords_filters_ptr

            if validity_filters is not None:
                if not isinstance(validity_filters, FilterList):
                    validity_filters = FilterList(validity_filters, ctx=ctx)
                filter_list = validity_filters
                filter_list_ptr = <tiledb_filter_list_t *>PyCapsule_GetPointer(
                        filter_list.__capsule__(), "fl")
                check_error(ctx,
                    tiledb_array_schema_set_validity_filter_list(ctx_ptr, schema_ptr, filter_list_ptr))
        except:
            tiledb_array_schema_free(&schema_ptr)
            raise

        rc = tiledb_array_schema_set_domain(ctx_ptr, schema_ptr, domain_ptr)
        if rc != TILEDB_OK:
            tiledb_array_schema_free(&schema_ptr)
            _raise_ctx_err(ctx_ptr, rc)

        cdef tiledb_attribute_t* attr_ptr = NULL
        for attr in attrs:
            if not isinstance(attr, Attr):
                raise TypeError("Cannot create schema with non-Attr value for 'attrs' argument")
            attribute = attr
            attr_ptr = <tiledb_attribute_t *>PyCapsule_GetPointer(
                attr.__capsule__(), "attr")
            rc = tiledb_array_schema_add_attribute(ctx_ptr, schema_ptr, attr_ptr)
            if rc != TILEDB_OK:
                tiledb_array_schema_free(&schema_ptr)
                _raise_ctx_err(ctx_ptr, rc)
        rc = tiledb_array_schema_check(ctx_ptr, schema_ptr)
        if rc != TILEDB_OK:
            tiledb_array_schema_free(&schema_ptr)
            _raise_ctx_err(ctx_ptr, rc)

        self.ctx = ctx
        self.ptr = schema_ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            tiledb_array_schema_free(&self.ptr)

    @staticmethod
    cdef from_ptr(const tiledb_array_schema_t* schema_ptr, ctx=None):
        """
        Constructs a ArraySchema class instance from a
        Ctx and tiledb_array_schema_t pointer
        """
        if not ctx:
            ctx = default_ctx()
        cdef ArraySchema schema = ArraySchema.__new__(ArraySchema)
        schema.ctx = ctx
        # cast away const
        schema.ptr = <tiledb_array_schema_t*> schema_ptr
        return schema

    @staticmethod
    def load(uri, ctx=None, key=None):
        if not ctx:
            ctx = default_ctx()
        cdef bytes buri = uri.encode('UTF-8')
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(ctx)
        cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
        cdef tiledb_array_schema_t* array_schema_ptr = NULL
        # encryption key
        cdef bytes bkey
        cdef tiledb_encryption_type_t key_type = TILEDB_NO_ENCRYPTION
        cdef void* key_ptr = NULL
        cdef unsigned int key_len = 0
        if key is not None:
            if isinstance(key, str):
                bkey = key.encode('ascii')
            else:
                bkey = bytes(key)
            key_type = TILEDB_AES_256_GCM
            key_ptr = <void *> PyBytes_AS_STRING(bkey)
            #TODO: unsafe cast here ssize_t -> uint64_t
            key_len = <unsigned int> PyBytes_GET_SIZE(bkey)
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_array_schema_load_with_key(
                ctx_ptr, uri_ptr, key_type, key_ptr, key_len, &array_schema_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return ArraySchema.from_ptr(array_schema_ptr, ctx=ctx)

    @staticmethod
    def from_file(uri=None, ctx=None):
        """Create an ArraySchema for a Filestore Array from a given file.
        If a uri is not given, then create a default schema."""
        if not ctx:
            ctx = default_ctx()
        cdef tiledb_array_schema_t* schema_ptr = NULL
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(ctx)

        cdef const char* uri_ptr = NULL
        if uri is not None:
            uri_ptr = PyBytes_AS_STRING(uri.encode('UTF-8'))

        check_error(ctx, tiledb_filestore_schema_create(
                         ctx_ptr, uri_ptr, &schema_ptr))

        return ArraySchema.from_ptr(schema_ptr, ctx=ctx)

    def __eq__(self, other):
        """Instance is equal to another ArraySchema"""
        if not isinstance(other, ArraySchema):
            return False
        nattr = self.nattr
        if nattr != other.nattr:
            return False
        if (self.sparse != other.sparse or
            self.cell_order != other.cell_order or
            self.tile_order != other.tile_order):
            return False
        if (self.capacity != other.capacity):
            return False
        if self.domain != other.domain:
            return False
        if self.coords_filters != other.coords_filters:
            return False
        for i in range(nattr):
            if self.attr(i) != other.attr(i):
                return False
        return True

    def __len__(self):
        """Returns the number of Attributes in the ArraySchema"""
        return self.nattr

    def __iter__(self):
        """Returns a generator object that iterates over the ArraySchema's Attribute objects"""
        return (self.attr(i) for i in range(self.nattr))

    def check(self):
        """Checks the correctness of the array schema

        :rtype: None
        :raises: :py:exc:`tiledb.TileDBError` if invalid
        """
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)
        check_error(self.ctx,
                    tiledb_array_schema_check(ctx_ptr, self.ptr))

    @property
    def sparse(self):
        """True if the array is a sparse array representation

        :rtype: bool
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef tiledb_array_type_t typ = TILEDB_DENSE
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)
        check_error(self.ctx,
                    tiledb_array_schema_get_array_type(ctx_ptr, self.ptr, &typ))
        return typ == TILEDB_SPARSE

    @property
    def allows_duplicates(self):
        """Returns True if the (sparse) array allows duplicates."""
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)

        if not self.sparse:
            raise TileDBError("ArraySchema.allows_duplicates does not apply to dense arrays")

        cdef int ballows_dups
        tiledb_array_schema_get_allows_dups(ctx_ptr, self.ptr, &ballows_dups)
        return bool(ballows_dups)

    @property
    def capacity(self):
        """The array capacity

        :rtype: int
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef uint64_t cap = 0
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)
        check_error(self.ctx,
                    tiledb_array_schema_get_capacity(ctx_ptr, self.ptr, &cap))
        return cap

    cdef _cell_order(ArraySchema self, tiledb_layout_t* cell_order_ptr):
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)
        check_error(self.ctx,
            tiledb_array_schema_get_cell_order(ctx_ptr, self.ptr, cell_order_ptr))

    @property
    def cell_order(self):
        """The cell order layout of the array."""
        cdef tiledb_layout_t order = TILEDB_UNORDERED
        self._cell_order(&order)
        return _tiledb_layout_string(order)

    cdef _tile_order(ArraySchema self, tiledb_layout_t* tile_order_ptr):
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)
        check_error(self.ctx,
            tiledb_array_schema_get_tile_order(ctx_ptr, self.ptr, tile_order_ptr))

    @property
    def tile_order(self):
        """The tile order layout of the array.

        :rtype: str
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef tiledb_layout_t order = TILEDB_UNORDERED
        self._tile_order(&order)

        layout_string = _tiledb_layout_string(order)
        if self.cell_order == "hilbert":
            layout_string = None

        return layout_string

    @property
    def coords_compressor(self):
        """The compressor label and level for the array's coordinates.

        :rtype: tuple(str, int)
        :raises: :py:exc:`tiledb.TileDBError`

        """
        # <todo> reimplement on top of filter API?
        pass

    @property
    def offsets_compressor(self):
        """The compressor label and level for the array's variable-length attribute offsets.

        :rtype: tuple(str, int)
        :raises: :py:exc:`tiledb.TileDBError`

        """
        # <todo> reimplement on top of filter API?
        pass

    @property
    def offsets_filters(self):
        """The FilterList for the array's variable-length attribute offsets

        :rtype: tiledb.FilterList
        :raises: :py:exc:`tiledb.TileDBError`
        """
        cdef tiledb_filter_list_t* filter_list_ptr = NULL
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)
        check_error(self.ctx,
            tiledb_array_schema_get_offsets_filter_list(
                ctx_ptr, self.ptr, &filter_list_ptr))
        return FilterList(self.ctx,
            _capsule=PyCapsule_New(filter_list_ptr, "fl", NULL))

    @property
    def coords_filters(self):
        """The FilterList for the array's coordinates

        :rtype: tiledb.FilterList
        :raises: :py:exc:`tiledb.TileDBError`
        """
        cdef tiledb_filter_list_t* filter_list_ptr = NULL
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)
        check_error(self.ctx,
            tiledb_array_schema_get_coords_filter_list(
                ctx_ptr, self.ptr, &filter_list_ptr))
        return FilterList(self.ctx,
            _capsule=PyCapsule_New(filter_list_ptr, "fl", NULL))

    @coords_filters.setter
    def coords_filters(self, value):
        warnings.warn(
            "coords_filters is deprecated; "
            "set the FilterList for each dimension",
            DeprecationWarning,
        )

    @property
    def validity_filters(self):
        """The FilterList for the array's validity

        :rtype: tiledb.FilterList
        :raises: :py:exc:`tiledb.TileDBError`
        """
        cdef tiledb_filter_list_t* validity_list_ptr = NULL
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)
        check_error(self.ctx,
            tiledb_array_schema_get_validity_filter_list(
                ctx_ptr, self.ptr, &validity_list_ptr))
        return FilterList(self.ctx,
            _capsule=PyCapsule_New(validity_list_ptr, "fl", NULL))

    @property
    def domain(self):
        """The Domain associated with the array.

        :rtype: tiledb.Domain
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef tiledb_domain_t* dom = NULL
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)
        check_error(self.ctx,
                    tiledb_array_schema_get_domain(ctx_ptr, self.ptr, &dom))
        return Domain(self.ctx, _capsule=PyCapsule_New(dom, "dom", NULL))

    @property
    def nattr(self):
        """The number of array attributes.

        :rtype: int
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef unsigned int nattr = 0
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)
        check_error(self.ctx,
                    tiledb_array_schema_get_attribute_num(ctx_ptr, self.ptr, &nattr))
        return nattr

    @property
    def ndim(self):
        """The number of array domain dimensions.

        :rtype: int
        """
        return self.domain.ndim

    @property
    def shape(self):
        """The array's shape

        :rtype: tuple(numpy scalar, numpy scalar)
        :raises TypeError: floating point (inexact) domain
        """
        return self.domain.shape

    @property
    def version(self):
        """The array's scehma version.

        :rtype: int
        :raises :py:exc:`tiledb.TileDBError`
        """
        cdef uint32_t version
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)
        check_error(self.ctx,
                    tiledb_array_schema_get_version(ctx_ptr, self.ptr, &version))
        return version

    def _make_invalid(self):
        """This is a helper function for testing schema.check: resets schema
        in order to make the schema invalid."""
        cdef tiledb_array_schema_t* schema_ptr = self.ptr
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)
        tiledb_array_schema_free(&schema_ptr)
        check_error(self.ctx,
                    tiledb_array_schema_alloc(ctx_ptr, TILEDB_DENSE, &self.ptr))

    def _needs_var_buffer(self, unicode name):
        """
        Returns true if the given attribute or dimension is var-sized
        :param name:
        :rtype: bool
        """
        if self.has_attr(name):
            return self.attr(name).isvar
        elif self.domain.has_dim(name):
            return self.domain.dim(name).isvar
        else:
            raise ValueError(f"Requested name '{name}' is not an attribute or dimension")

    cdef _attr_name(self, name):
        cdef bytes bname = ustring(name).encode('UTF-8')
        cdef tiledb_attribute_t* attr_ptr = NULL
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)
        check_error(self.ctx,
                    tiledb_array_schema_get_attribute_from_name(
                        ctx_ptr, self.ptr, bname, &attr_ptr))
        return Attr(self.ctx, _capsule=PyCapsule_New(attr_ptr, "attr", NULL))

    cdef _attr_idx(self, int idx):
        cdef tiledb_attribute_t* attr_ptr = NULL
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)
        check_error(self.ctx,
                    tiledb_array_schema_get_attribute_from_index(
                        ctx_ptr, self.ptr, idx, &attr_ptr))
        return Attr(self.ctx, _capsule=PyCapsule_New(attr_ptr, "attr", NULL))

    def attr(self, object key not None):
        """Returns an Attr instance given an int index or string label

        :param key: attribute index (positional or associative)
        :type key: int or str
        :rtype: tiledb.Attr
        :return: The ArraySchema attribute at index or with the given name (label)
        :raises TypeError: invalid key type

        """
        if isinstance(key, (str, unicode)):
            return self._attr_name(key)
        elif isinstance(key, _inttypes):
            return self._attr_idx(int(key))
        raise TypeError("attr indices must be a string name, "
                        "or an integer index, not {0!r}".format(type(key)))

    def has_attr(self, name):
        """Returns true if the given name is an Attribute of the ArraySchema

        :param name: attribute name
        :rtype: bool
        """
        cdef:
            int32_t has_attr = 0
            int32_t rc = TILEDB_OK
            bytes bname = name.encode("UTF-8")
            cdef tiledb_ctx_t* ctx_ptr = NULL

        ctx_ptr = safe_ctx_ptr(self.ctx)

        rc = tiledb_array_schema_has_attribute(
            ctx_ptr,
            self.ptr,
            bname,
            &has_attr
        )
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        return bool(has_attr)

    def has_dim(self, name):
        """Returns True if the given name is a Dim of the ArraySchema

        :param name: dimension name
        :rtype: bool
        """
        return self.domain.has_dim(name)

    def attr_or_dim_dtype(self, unicode name):
        if self.has_attr(name):
            dtype = self.attr(name).dtype
        elif self.has_dim(name):
            dtype = self.domain.dim(name).dtype
        else:
            raise TileDBError(f"Unknown attribute or dimension ('{name}')")

        if dtype.itemsize == 0:
            # special handling for flexible numpy dtypes: change itemsize from 0 to 1
            dtype = np.dtype((dtype, 1))
        return dtype


    def dump(self):
        """Dumps a string representation of the array object to standard output (stdout)"""
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)
        check_error(self.ctx,
                    tiledb_array_schema_dump(ctx_ptr, self.ptr, stdout))
        print("\n")
        return

    def __repr__(self):
        # TODO support/use __qualname__
        output = io.StringIO()
        output.write("ArraySchema(\n")
        output.write("  domain=Domain(*[\n")
        for i in range(self.domain.ndim):
            output.write(f"    {repr(self.domain.dim(i))},\n")
        output.write("  ]),\n")
        output.write("  attrs=[\n")
        for i in range(self.nattr):
            output.write(f"    {repr(self.attr(i))},\n")
        output.write("  ],\n")
        output.write(
            f"  cell_order='{self.cell_order}',\n"
            f"  tile_order={repr(self.tile_order)},\n"
        )
        output.write(f"  capacity={self.capacity},\n")
        output.write(f"  sparse={self.sparse},\n")
        if self.sparse:
            output.write(f"  allows_duplicates={self.allows_duplicates},\n")

        output.write(")\n")

        return output.getvalue()

    def _repr_html_(self):
        output = io.StringIO()

        output.write("<table>")

        output.write("<tr><th>Domain</th></tr>")
        output.write(f"<tr><td>{self.domain._repr_html_()}</td></tr>")

        output.write("<tr><th>Attributes</th></tr>")
        output.write("<tr>")
        output.write("<td>")
        output.write("<table>")
        output.write("<tr>")
        output.write("<th>Name</th>")
        output.write("<th>Data Type</th>")
        output.write("<th>Is Var-Len</th>")
        output.write("<th>Is Nullable</th>")
        output.write("<th>Filters</th>")
        output.write("</tr>")
        for i in range(self.nattr):
            output.write(f"{self.attr(i)._repr_html_row_only_()}")
        output.write("</table>")
        output.write("</td>")
        output.write("</tr>")

        output.write("<tr><th>Cell Order</th></tr>")
        output.write(f"<tr><td>{self.cell_order}</td></tr>")

        output.write("<tr><th>Tile Order</th></tr>")
        output.write(f"<tr><td>{self.tile_order}</td></tr>")

        if not self.sparse:
            output.write("<tr><th>Capacity</th></tr>")
            output.write(f"<tr><td>{self.capacity}</td></tr>")

        output.write("<tr><th>Sparse</th></tr>")
        output.write(f"<tr><td>{self.sparse}</td></tr>")

        if self.sparse:
            output.write("<tr><th>Allows Duplicates</th></tr>")
            output.write(f"<tr><td>{self.allows_duplicates}</td></tr>")

        output.write("</table>")

        return output.getvalue()


# Wrapper class to allow returning a Python object so that exceptions work correctly
# within preload_array
cdef class ArrayPtr(object):
    cdef tiledb_array_t* ptr

cdef ArrayPtr preload_array(uri, mode, key, timestamp, ctx=None):
    """Open array URI without constructing specific type of Array object (internal)."""
    if not ctx:
        ctx = default_ctx()
    # ctx
    cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(ctx)
    # uri
    cdef bytes buri = unicode_path(uri)
    cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
    # mode
    cdef tiledb_query_type_t query_type = TILEDB_READ
    # key
    cdef bytes bkey
    cdef tiledb_encryption_type_t key_type = TILEDB_NO_ENCRYPTION
    cdef void* key_ptr = NULL
    cdef unsigned int key_len = 0

    # convert python mode string to a query type
    mode_to_query_type = {
        "r": TILEDB_READ,
        "w": TILEDB_WRITE,
        "m": TILEDB_MODIFY_EXCLUSIVE,
        "d": TILEDB_DELETE
    }
    if mode not in mode_to_query_type:
        raise ValueError("TileDB array mode must be 'r', 'w', 'm', or 'd'")
    query_type = mode_to_query_type[mode]

    # check the key, and convert the key to bytes
    if key is not None:
        if isinstance(key, str):
            bkey = key.encode('ascii')
        else:
            bkey = bytes(key)
        key_type = TILEDB_AES_256_GCM
        key_ptr = <void *> PyBytes_AS_STRING(bkey)
        #TODO: unsafe cast here ssize_t -> uint64_t
        key_len = <unsigned int> PyBytes_GET_SIZE(bkey)
    cdef uint64_t ts_start = 0
    cdef uint64_t ts_end = 0
    cdef bint set_start = False, set_end = False

    if timestamp is not None:
        if isinstance(timestamp, tuple):
            if len(timestamp) != 2:
                raise ValueError("'timestamp' argument expects either int or tuple(start: int, end: int)")
            if timestamp[0] is not None:
                ts_start = <uint64_t>timestamp[0]
                set_start = True
            if timestamp[1] is not None:
                ts_end = <uint64_t>timestamp[1]
                set_end = True
        elif isinstance(timestamp, int):
            # handle the existing behavior for unary timestamp
            # which is equivalent to endpoint of the range
            ts_end = <uint64_t> timestamp
            set_end = True
        else:
            raise TypeError("Unexpected argument type for 'timestamp' keyword argument")

    # allocate and then open the array
    cdef tiledb_array_t* array_ptr = NULL
    cdef int rc = TILEDB_OK
    rc = tiledb_array_alloc(ctx_ptr, uri_ptr, &array_ptr)
    if rc != TILEDB_OK:
        _raise_ctx_err(ctx_ptr, rc)

    try:
        if set_start:
            check_error(ctx,
                tiledb_array_set_open_timestamp_start(ctx_ptr, array_ptr, ts_start)
            )
        if set_end:
            check_error(ctx,
                tiledb_array_set_open_timestamp_end(ctx_ptr, array_ptr, ts_end)
            )
    except:
        tiledb_array_free(&array_ptr)
        raise

    with nogil:
       rc = tiledb_array_open_with_key(
           ctx_ptr, array_ptr, query_type, key_type, key_ptr, key_len)

    if rc != TILEDB_OK:
        tiledb_array_free(&array_ptr)
        _raise_ctx_err(ctx_ptr, rc)

    cdef ArrayPtr retval = ArrayPtr()
    retval.ptr = array_ptr
    return retval

cdef class Array(object):
    """Base class for TileDB array objects.

    Defines common properties/functionality for the different array types. When
    an Array instance is initialized, the array is opened with the specified mode.

    :param str uri: URI of array to open
    :param str mode: (default 'r') Open the array object in read 'r', write 'w', or delete 'd' mode
    :param str key: (default None) If not None, encryption key to decrypt the array
    :param tuple timestamp: (default None) If int, open the array at a given TileDB
        timestamp. If tuple, open at the given start and end TileDB timestamps.
    :param str attr: (default None) open one attribute of the array; indexing a
        dense array will return a Numpy ndarray directly rather than a dictionary.
    :param Ctx ctx: TileDB context
    """
    def __init__(self, uri, mode='r', key=None, timestamp=None,
                 attr=None, ctx=None):
        if not ctx:
            ctx = default_ctx()
        # ctx
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(ctx)
        # array
        cdef ArrayPtr preload_ptr

        if not self._isopen:
            preload_ptr = preload_array(uri, mode, key, timestamp, ctx)
            self.ptr =  preload_ptr.ptr

        assert self.ptr != NULL, "internal error: unexpected null tiledb_array_t pointer in Array.__init__"
        cdef tiledb_array_t* array_ptr = self.ptr

        cdef ArraySchema schema
        cdef tiledb_array_schema_t* array_schema_ptr = NULL
        try:
            rc = TILEDB_OK
            with nogil:
                rc = tiledb_array_get_schema(ctx_ptr, array_ptr, &array_schema_ptr)
            if rc != TILEDB_OK:
              _raise_ctx_err(ctx_ptr, rc)
            schema = ArraySchema.from_ptr(array_schema_ptr, ctx=ctx)
        except:
            tiledb_array_close(ctx_ptr, array_ptr)
            tiledb_array_free(&array_ptr)
            self.ptr = NULL
            raise

        # view on a single attribute
        if attr and not any(attr == schema.attr(i).name for i in range(schema.nattr)):
            tiledb_array_close(ctx_ptr, array_ptr)
            tiledb_array_free(&array_ptr)
            self.ptr = NULL
            raise KeyError("No attribute matching '{}'".format(attr))
        else:
            self.view_attr = unicode(attr) if (attr is not None) else None

        self.ctx = ctx
        self.uri = unicode(uri)
        self.mode = unicode(mode)
        self.schema = schema
        self.key = key
        self.domain_index = DomainIndexer(self)

        self.last_fragment_info = dict()
        self.meta = Metadata(self)

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            tiledb_array_free(&self.ptr)

    def __capsule__(self):
        if self.ptr == NULL:
            raise TileDBError("internal error: cannot create capsule for uninitialized Ctx!")
        cdef const char* name = "ctx"
        cap = PyCapsule_New(<void *>(self.ptr), name, NULL)
        return cap

    def __repr__(self):
        if self.isopen:
            return "Array(type={0}, uri={1!r}, mode={2}, ndim={3})"\
                .format("Sparse" if self.schema.sparse else "Dense", self.uri, self.mode, self.schema.ndim)
        else:
            return "Array(uri={0!r}, mode=closed)"

    def _ctx_(self) -> Ctx:
        """
        Get Ctx object associated with the array (internal).
        This method exists for serialization.

        :return: Ctx object used to open the array.
        :rtype: Ctx
        """
        return self.ctx

    @classmethod
    def create(cls, uri, ArraySchema schema, key=None, overwrite=False, ctx=None):
        """Creates a TileDB Array at the given URI

        :param str uri: URI at which to create the new empty array.
        :param ArraySchema schema: Schema for the array
        :param str key: (default None) Encryption key to use for array
        :param bool overwrite: (default False) Overwrite the array if it already exists
        :param ctx Ctx: (default None) Optional TileDB Ctx used when creating the array,
                        by default uses the ArraySchema's associated context
                        (*not* necessarily ``tiledb.default_ctx``).

        """
        if issubclass(cls, DenseArrayImpl) and schema.sparse:
            raise ValueError("Array.create `schema` argument must be a dense schema for DenseArray and subclasses")
        if issubclass(cls, SparseArrayImpl) and not schema.sparse:
            raise ValueError("Array.create `schema` argument must be a sparse schema for SparseArray and subclasses")

        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(schema.ctx)
        cdef bytes buri = unicode_path(uri)
        cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
        cdef tiledb_array_schema_t* schema_ptr = schema.ptr

        cdef bytes bkey
        cdef tiledb_encryption_type_t key_type = TILEDB_NO_ENCRYPTION
        cdef void* key_ptr = NULL
        cdef unsigned int key_len = 0

        cdef int rc = TILEDB_OK

        if key is not None:
            if isinstance(key, str):
                bkey = key.encode('ascii')
            else:
                bkey = bytes(key)
            key_type = TILEDB_AES_256_GCM
            key_ptr = <void *> PyBytes_AS_STRING(bkey)
            #TODO: unsafe cast here ssize_t -> uint64_t
            key_len = <unsigned int> PyBytes_GET_SIZE(bkey)

        if overwrite:
            if object_type(uri) == "array":
                if uri.startswith("file://") or "://" not in uri:
                    if VFS().remove_dir(uri) != TILEDB_OK:
                        _raise_ctx_err(ctx_ptr, rc)
                else:
                    raise TypeError("Cannot overwrite non-local array.")
            else:
                warnings.warn("Overwrite set, but array does not exist")

        if ctx is not None:
            if not isinstance(ctx, Ctx):
                raise TypeError("tiledb.Array.create() expected tiledb.Ctx "
                                "object to argument ctx")
            ctx_ptr = safe_ctx_ptr(ctx)
        with nogil:
            rc = tiledb_array_create_with_key(ctx_ptr, uri_ptr, schema_ptr, key_type, key_ptr, key_len)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return

    @staticmethod
    def load_typed(uri, mode='r', key=None, timestamp=None, attr=None, ctx=None):
        """Return a {Dense,Sparse}Array instance from a pre-opened Array (internal)"""
        if not ctx:
            ctx = default_ctx()
        cdef int32_t rc = TILEDB_OK
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(ctx)
        cdef tiledb_array_schema_t* schema_ptr = NULL
        cdef tiledb_array_type_t array_type
        cdef Array new_array
        cdef object new_array_typed

        # *** preload_array owns array_ptr until it returns ***
        #     and will free array_ptr upon exception
        cdef ArrayPtr tmp_array = preload_array(uri, mode, key, timestamp, ctx)
        assert tmp_array.ptr != NULL, "Internal error, array loading return nullptr"
        cdef tiledb_array_t* array_ptr = tmp_array.ptr
        # *** now we own array_ptr -- free in the try..except clause ***
        try:
            rc = tiledb_array_get_schema(ctx_ptr, array_ptr, &schema_ptr)
            if rc != TILEDB_OK:
                _raise_ctx_err(ctx_ptr, rc)

            rc = tiledb_array_schema_get_array_type(ctx_ptr, schema_ptr, &array_type)
            if rc != TILEDB_OK:
                _raise_ctx_err(ctx_ptr, rc)

            tiledb_array_schema_free(&schema_ptr)

            if array_type == TILEDB_DENSE:
                new_array_typed = DenseArray.__new__(DenseArray)
            else:
                new_array_typed = SparseArray.__new__(SparseArray)

        except:
            tiledb_array_free(&array_ptr)
            raise

        # *** this assignment must happen outside the try block ***
        # *** because the array destructor will free array_ptr  ***
        # note: must use the immediate form `(<cast>x).m()` here
        #       do not assign a temporary Array object
        if array_type == TILEDB_DENSE:
            (<DenseArrayImpl>new_array_typed).ptr = array_ptr
            (<DenseArrayImpl>new_array_typed)._isopen = True
        else:
            (<SparseArrayImpl>new_array_typed).ptr = array_ptr
            (<SparseArrayImpl>new_array_typed)._isopen = True
        # *** new_array_typed now owns array_ptr ***

        new_array_typed.__init__(uri, mode=mode, key=key, timestamp=timestamp, attr=attr, ctx=ctx)
        return new_array_typed

    def __enter__(self):
        """
        The `__enter__` and `__exit__` methods allow TileDB arrays to be opened (and auto-closed)
        using `with tiledb.open(uri) as A:` syntax.
        """
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """
        The `__enter__` and `__exit__` methods allow TileDB arrays to be opened (and auto-closed)
        using `with tiledb.open(uri) as A:` syntax.
        """
        self.close()

    def close(self):
        """Closes this array, flushing all buffered data."""
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)
        cdef tiledb_array_t* array_ptr = self.ptr
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_array_close(ctx_ptr, array_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        self.schema = None
        return

    def reopen(self, timestamp=None):
        """
        Reopens this array.

        This is useful when the array is updated after it was opened.
        To sync-up with the updates, the user must either close the array and open again,
        or just use ``reopen()`` without closing. ``reopen`` will be generally faster than
        a close-then-open.
        """
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)
        cdef tiledb_array_t* array_ptr = self.ptr
        cdef uint64_t _timestamp = 0
        cdef int rc = TILEDB_OK
        if timestamp is None:
            with nogil:
                rc = tiledb_array_reopen(ctx_ptr, array_ptr)
        else:
            _timestamp = <uint64_t> timestamp
            with nogil:
                rc = tiledb_array_reopen_at(ctx_ptr, array_ptr, _timestamp)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return

    @property
    def meta(self):
        """
        Return array metadata instance

        :rtype: tiledb.Metadata
        """
        return self.meta

    @property
    def schema(self):
        """The :py:class:`ArraySchema` for this array."""
        schema = self.schema
        if schema is None:
            raise TileDBError("Cannot access schema, array is closed")
        return schema

    @property
    def mode(self):
        """The mode this array was opened with."""
        return self.mode

    @property
    def iswritable(self):
        """This array is currently opened as writable."""
        return self.mode == 'w'

    @property
    def isopen(self):
        """True if this array is currently open."""
        cdef int isopen = 0
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)
        cdef tiledb_array_t* array_ptr = self.ptr
        cdef int rc = TILEDB_OK
        rc = tiledb_array_is_open(ctx_ptr, array_ptr, &isopen)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return isopen == 1

    @property
    def ndim(self):
        """The number of dimensions of this array."""
        return self.schema.ndim

    @property
    def domain(self):
        """The :py:class:`Domain` of this array."""
        return self.schema.domain

    @property
    def dtype(self):
        """The NumPy dtype of the specified attribute"""
        if self.view_attr is None and self.schema.nattr > 1:
            raise NotImplementedError("Multi-attribute does not have single dtype!")
        return self.schema.attr(0).dtype

    @property
    def shape(self):
        """The shape of this array."""
        return self.schema.shape

    @property
    def nattr(self):
        """The number of attributes of this array."""
        if self.view_attr:
            return 1
        else:
           return self.schema.nattr

    @property
    def view_attr(self):
        """The view attribute of this array."""
        return self.view_attr

    @property
    def timestamp(self):
        """Deprecated in 0.9.2.

        Use `timestamp_range`
        """
        raise TileDBError(
            "timestamp is deprecated; you must use timestamp_range. "
            "This message will be removed in 0.21.0.",
        )

    @property
    def timestamp_range(self):
        """Returns the timestamp range the array is opened at

        :rtype: tuple
        :returns: tiledb timestamp range at which point the array was opened

        """
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)
        cdef tiledb_array_t* array_ptr = self.ptr
        cdef uint64_t timestamp_start = 0
        cdef uint64_t timestamp_end = 0
        cdef int rc = TILEDB_OK

        rc = tiledb_array_get_open_timestamp_start(ctx_ptr, array_ptr, &timestamp_start)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        rc = tiledb_array_get_open_timestamp_end(ctx_ptr, array_ptr, &timestamp_end)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        return (int(timestamp_start), int(timestamp_end))

    @property
    def coords_dtype(self):
        """
        Deprecated in 0.8.10
        """
        raise TileDBError(
            "`coords_dtype` is deprecated because combined coords have been "
            "removed from libtiledb. This message will be removed in 0.21.0.",
        )

    @property
    def uri(self):
        """Returns the URI of the array"""
        return self.uri

    def subarray(self, selection, attrs=None, coords=False, order=None):
        raise NotImplementedError()

    def attr(self, key):
        """Returns an :py:class:`Attr` instance given an int index or string label

        :param key: attribute index (positional or associative)
        :type key: int or str
        :rtype: :py:class:`Attr`
        :return: The array attribute at index or with the given name (label)
        :raises TypeError: invalid key type"""
        return self.schema.attr(key)

    def dim(self, dim_id):
        """Returns a :py:class:`Dim` instance given a dim index or name

        :param key: attribute index (positional or associative)
        :type key: int or str
        :rtype: :py:class:`Attr`
        :return: The array attribute at index or with the given name (label)
        :raises TypeError: invalid key type"""
        return self.schema.domain.dim(dim_id)

    def delete_fragments(self, timestamp_start, timestamp_end):
        """
        Delete a range of fragments from timestamp_start to timestamp_end.
        The array needs to be opened in 'm' mode as shown in the example below.

        :param timestamp_start: the first fragment to delete in the range
        :type timestamp_start: int
        :param timestamp_end: the last fragment to delete in the range
        :type timestamp_end: int

        **Example:**

        >>> import tiledb, tempfile, numpy as np
        >>> path = tempfile.mkdtemp()

        >>> with tiledb.from_numpy(path, np.zeros(4), timestamp=1) as A:
        ...     pass
        >>> with tiledb.open(path, 'w', timestamp=2) as A:
        ...     A[:] = np.ones(4, dtype=np.int64)

        >>> with tiledb.open(path, 'r') as A:
        ...     A[:]
        array([1., 1., 1., 1.])

        >>> with tiledb.open(path, 'm') as A:
        ...     A.delete_fragments(2, 2)

        >>> with tiledb.open(path, 'r') as A:
        ...     A[:]
        array([0., 0., 0., 0.])

        """
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)
        cdef tiledb_array_t* array_ptr = self.ptr
        cdef tiledb_query_t* query_ptr = NULL
        cdef bytes buri = self.uri.encode('UTF-8')

        cdef int rc = TILEDB_OK

        rc = tiledb_array_delete_fragments(
                ctx_ptr,
                array_ptr,
                buri,
                timestamp_start,
                timestamp_end
        )
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

    @staticmethod
    def delete_array(uri, ctx=None):
        """
        Delete the given array.

        :param str uri: The URI of the array
        :param Ctx ctx: TileDB context

        **Example:**

        >>> import tiledb, tempfile, numpy as np
        >>> path = tempfile.mkdtemp()

        >>> with tiledb.from_numpy(path, np.zeros(4), timestamp=1) as A:
        ...     pass
        >>> tiledb.array_exists(path)
        True

        >>> tiledb.Array.delete_array(path)

        >>> tiledb.array_exists(path)
        False

        """
        if not ctx:
            ctx = default_ctx()

        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(ctx)
        cdef bytes buri = uri.encode('UTF-8')
        cdef ArrayPtr preload_ptr = preload_array(uri, 'm', None, None)
        cdef tiledb_array_t* array_ptr = preload_ptr.ptr

        cdef int rc = TILEDB_OK

        rc = tiledb_array_delete_array(ctx_ptr, array_ptr, buri)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

    def nonempty_domain(self):
        """Return the minimum bounding domain which encompasses nonempty values.

        :rtype: tuple(tuple(numpy scalar, numpy scalar), ...)
        :return: A list of (inclusive) domain extent tuples, that contain all
            nonempty cells

        """
        cdef list results = list()
        dom = self.schema.domain

        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)
        cdef tiledb_array_t* array_ptr = self.ptr
        cdef int rc = TILEDB_OK
        cdef uint32_t dim_idx

        cdef uint64_t start_size
        cdef uint64_t end_size
        cdef int32_t is_empty
        cdef np.ndarray start_buf
        cdef np.ndarray end_buf
        cdef void* start_buf_ptr
        cdef void* end_buf_ptr
        cdef np.dtype dim_dtype

        for dim_idx in range(dom.ndim):
            dim_dtype = dom.dim(dim_idx).dtype

            if np.issubdtype(dim_dtype, np.str_) or np.issubdtype(dim_dtype, np.bytes_):
                rc = tiledb_array_get_non_empty_domain_var_size_from_index(
                    ctx_ptr, array_ptr, dim_idx, &start_size, &end_size, &is_empty)
                if rc != TILEDB_OK:
                    _raise_ctx_err(ctx_ptr, rc)

                if is_empty:
                    results.append((None, None))
                    continue

                buf_dtype = 'S'
                start_buf = np.empty(start_size, 'S' + str(start_size))
                end_buf = np.empty(end_size, 'S' + str(end_size))
                start_buf_ptr = np.PyArray_DATA(start_buf)
                end_buf_ptr = np.PyArray_DATA(end_buf)
            else:
                # this one is contiguous
                start_buf = np.empty(2, dim_dtype)
                start_buf_ptr = np.PyArray_DATA(start_buf)

            if np.issubdtype(dim_dtype, np.str_) or np.issubdtype(dim_dtype, np.bytes_):
                rc = tiledb_array_get_non_empty_domain_var_from_index(
                            ctx_ptr, array_ptr, dim_idx, start_buf_ptr, end_buf_ptr, &is_empty
                )
                if rc != TILEDB_OK:
                    _raise_ctx_err(ctx_ptr, rc)
                if is_empty:
                    return None

                if start_size > 0 and end_size > 0:
                    results.append((start_buf.item(0), end_buf.item(0)))
                else:
                    results.append((None, None))
            else:
                rc = tiledb_array_get_non_empty_domain_from_index(
                        ctx_ptr, array_ptr, dim_idx, start_buf_ptr, &is_empty
                )
                if rc != TILEDB_OK:
                    _raise_ctx_err(ctx_ptr, rc)
                if is_empty:
                    return None

                res_x, res_y = start_buf.item(0), start_buf.item(1)

                if np.issubdtype(dim_dtype, np.datetime64):
                    # Convert to np.datetime64
                    date_unit = np.datetime_data(dim_dtype)[0]
                    res_x = np.datetime64(res_x, date_unit)
                    res_y = np.datetime64(res_y, date_unit)

                results.append((res_x, res_y))

        return tuple(results)

    def consolidate(self, config=None, key=None, fragment_uris=None, timestamp=None):
        """
        Consolidates fragments of an array object for increased read performance.

        Overview: https://docs.tiledb.com/main/concepts/internal-mechanics/consolidation

        :param tiledb.Config config: The TileDB Config with consolidation parameters set
        :param key: (default None) encryption key to decrypt an encrypted array
        :type key: str or bytes
        :param fragment_uris: (default None) Consolidate the array using a list of fragment file names
        :param timestamp: (default None) If not None, consolidate the array using the given tuple(int, int) UNIX seconds range (inclusive). This argument will be ignored if `fragment_uris` is passed.
        :type timestamp: tuple (int, int)
        :raises: :py:exc:`tiledb.TileDBError`

        Rather than passing the timestamp into this function, it may be set with
        the config parameters `"sm.vacuum.timestamp_start"`and
        `"sm.vacuum.timestamp_end"` which takes in a time in UNIX seconds. If both
        are set then this function's `timestamp` argument will be used.

        """
        if self.mode == 'r':
            raise TileDBError("cannot consolidate array opened in readonly mode (mode='r')")
        return consolidate(uri=self.uri, key=key, config=config, ctx=self.ctx,
                           fragment_uris=fragment_uris, timestamp=timestamp)

    def upgrade_version(self, config=None):
        """
        Upgrades an array to the latest format version.

        :param ctx The TileDB context.
        :param array_uri The uri of the array.
        :param config Configuration parameters for the upgrade
            (`nullptr` means default, which will use the config from `ctx`).
        :raises: :py:exc:`tiledb.TileDBError`
        """
        cdef int rc = TILEDB_OK
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)
        cdef bytes buri = self.uri.encode('UTF-8')
        cdef tiledb_config_t* config_ptr = NULL
        if config is not None:
            config_ptr = <tiledb_config_t*>PyCapsule_GetPointer(
                config.__capsule__(), "config")

        rc = tiledb_array_upgrade_version(
            ctx_ptr, buri, config_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

    def dump(self):
        self.schema.dump()

    cdef _ndarray_is_varlen(self, np.ndarray array):
        return  (np.issubdtype(array.dtype, np.bytes_) or
                 np.issubdtype(array.dtype, np.unicode_) or
                 array.dtype == object)

    @property
    def domain_index(self):
        return self.domain_index

    @property
    def dindex(self):
        return self.domain_index

    @property
    def multi_index(self):
        """Retrieve data cells with multi-range, domain-inclusive indexing. Returns
        the cross-product of the ranges.

        :param list selection: Per dimension, a scalar, ``slice``, or list of scalars
            or ``slice`` objects. Scalars and ``slice`` components should match the
            type of the underlying Dimension.
        :returns: dict of {'attribute': result}. Coords are included by default for
            Sparse arrays only (use `Array.query(coords=<>)` to select).
        :raises IndexError: invalid or unsupported index selection
        :raises: :py:exc:`tiledb.TileDBError`

        ``multi_index[]`` accepts, for each dimension, a scalar, ``slice``, or list
        of scalars or ``slice`` objects. Each item is interpreted as a point
        (scalar) or range (``slice``) used to query the array on the corresponding
        dimension.

        Unlike NumPy array indexing, ``multi_index`` respects TileDB's range semantics:
        slice ranges are *inclusive* of the start- and end-point, and negative ranges
        do not wrap around (because a TileDB dimensions may have a negative domain).

        See also: https://docs.tiledb.com/main/api-usage/reading-arrays/multi-range-subarrays

        ** Example **

        >>> import tiledb, tempfile, numpy as np
        >>>
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...    A = tiledb.from_numpy(tmp, np.eye(4) * [1,2,3,4])
        ...    A.multi_index[1]
        ...    A.multi_index[1,1]
        ...    # return row 0 and 2
        ...    A.multi_index[[0,2]]
        ...    # return rows 0 and 2 intersecting column 2
        ...    A.multi_index[[0,2], 2]
        ...    # return rows 0:2 intersecting columns 0:2
        ...    A.multi_index[slice(0,2), slice(0,2)]
        OrderedDict([('', array([[0., 2., 0., 0.]]))])
        OrderedDict([('', array([[2.]]))])
        OrderedDict([('', array([[1., 0., 0., 0.], [0., 0., 3., 0.]]))])
        OrderedDict([('', array([[0.], [3.]]))])
        OrderedDict([('', array([[1., 0., 0.], [0., 2., 0.], [0., 0., 3.]]))])

        """
        # Delayed to avoid circular import
        from .multirange_indexing import MultiRangeIndexer
        return MultiRangeIndexer(self)

    @property
    def df(self):
        """Retrieve data cells as a Pandas dataframe, with multi-range,
        domain-inclusive indexing using ``multi_index``.

        :param list selection: Per dimension, a scalar, ``slice``, or list of scalars
            or ``slice`` objects. Scalars and ``slice`` components should match the
            type of the underlying Dimension.
        :returns: dict of {'attribute': result}. Coords are included by default for
            Sparse arrays only (use `Array.query(coords=<>)` to select).
        :raises IndexError: invalid or unsupported index selection
        :raises: :py:exc:`tiledb.TileDBError`

        ``df[]`` accepts, for each dimension, a scalar, ``slice``, or list
        of scalars or ``slice`` objects. Each item is interpreted as a point
        (scalar) or range (``slice``) used to query the array on the corresponding
        dimension.

        ** Example **

        >>> import tiledb, tempfile, numpy as np, pandas as pd
        >>>
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...    data = {'col1_f': np.arange(0.0,1.0,step=0.1), 'col2_int': np.arange(10)}
        ...    df = pd.DataFrame.from_dict(data)
        ...    tiledb.from_pandas(tmp, df)
        ...    A = tiledb.open(tmp)
        ...    A.df[1]
        ...    A.df[1:5]
              col1_f  col2_int
           1     0.1         1
              col1_f  col2_int
           1     0.1         1
           2     0.2         2
           3     0.3         3
           4     0.4         4
           5     0.5         5

        """
        # Delayed to avoid circular import
        from .multirange_indexing import DataFrameIndexer
        return DataFrameIndexer(self, use_arrow=None)

    @property
    def last_write_info(self):
        return self.last_fragment_info

    @property
    def _buffers(self):
        return self._buffers

    def _set_buffers(self, object buffers):
        """
        Helper function to set external buffers in the form of
            {'attr_name': (data_array, offsets_array)}
        Buffers will be used to satisfy the next index/query request.
        """
        self._buffers = buffers

    def set_query(self, serialized_query):
        from tiledb.main import PyQuery
        q = PyQuery(self._ctx_(), self, ("",), (), 0, False)
        q.set_serialized_query(serialized_query)
        q.submit()

        cdef object results = OrderedDict()
        results = q.results()

        out = OrderedDict()
        for name in results.keys():
            arr = results[name][0]
            arr.dtype = q.buffer_dtype(name)
            out[name] = arr
        return out

    # pickling support: this is a lightweight pickle for distributed use.
    #   simply treat as wrapper around URI, not actual data.
    def __getstate__(self):
        config_dict = self._ctx_().config().dict()
        return (self.uri, self.mode, self.key, self.view_attr, self.timestamp_range, config_dict)

    def __setstate__(self, state):
        cdef:
            unicode uri, mode
            object view_attr = None
            object timestamp_range = None
            object key = None
            dict config_dict = {}
        uri, mode, key, view_attr, timestamp_range, config_dict = state

        if config_dict is not {}:
            config_dict = state[5]
            config = Config(params=config_dict)
            ctx = Ctx(config)
        else:
            ctx = default_ctx()

        self.__init__(uri, mode=mode, key=key, attr=view_attr,
                      timestamp=timestamp_range, ctx=ctx)

cdef class Query(object):
    """
    Proxy object returned by query() to index into original array
    on a subselection of attribute in a defined layout order

    See documentation of Array.query
    """

    def __init__(self, array, attrs=None, cond=None, dims=None,
                 coords=False, index_col=True, order=None,
                 use_arrow=None, return_arrow=False, return_incomplete=False):
        if array.mode not in  ('r', 'd'):
            raise ValueError("array mode must be read or delete mode")

        if dims is not None and coords == True:
            raise ValueError("Cannot pass both dims and coords=True to Query")

        cdef list dims_to_set = list()

        if dims is False:
            self.dims = False
        elif dims != None and dims != True:
            domain = array.schema.domain
            for dname in dims:
                if not domain.has_dim(dname):
                    raise TileDBError(f"Selected dimension does not exist: '{dname}'")
            self.dims = [unicode(dname) for dname in dims]
        elif coords == True or dims == True:
            domain = array.schema.domain
            self.dims = [domain.dim(i).name for i in range(domain.ndim)]

        if attrs is not None:
            for name in attrs:
                if not array.schema.has_attr(name):
                    raise TileDBError(f"Selected attribute does not exist: '{name}'")
        self.attrs = attrs
        self.cond = cond

        if order == None:
            if array.schema.sparse:
                self.order = 'U' # unordered
            else:
                self.order = 'C' # row-major
        else:
            self.order = order

        # reference to the array we are querying
        self.array = array
        self.coords = coords
        self.index_col = index_col
        self.return_arrow = return_arrow
        if return_arrow:
            if use_arrow is None:
                use_arrow = True
            if not use_arrow:
                raise TileDBError("Cannot initialize return_arrow with use_arrow=False")
        self.use_arrow = use_arrow
        self.return_incomplete = return_incomplete

        self.domain_index = DomainIndexer(array, query=self)

    def __getitem__(self, object selection):
        if self.return_arrow:
            raise TileDBError("`return_arrow=True` requires .df indexer`")

        return self.array.subarray(selection,
                                   attrs=self.attrs,
                                   cond=self.cond,
                                   coords=self.coords if self.coords else self.dims,
                                   order=self.order)

    @property
    def attrs(self):
        """List of attributes to include in Query."""
        return self.attrs

    @property
    def attr_cond(self):
        raise TileDBError(
            "`attr_cond` is no longer supported. You must use `cond`. "
            "This message will be removed in 0.21.0."
        )

    @property
    def cond(self):
        """QueryCondition used to filter attributes or dimensions in Query."""
        return self.cond

    @property
    def dims(self):
        """List of dimensions to include in Query."""
        return self.dims

    @property
    def coords(self):
        """
        True if query should include (return) coordinate values.

        :rtype: bool
        """
        return self.coords

    @property
    def order(self):
        """Return underlying Array order."""
        return self.order

    @property
    def index_col(self):
        """List of columns to set as index for dataframe queries, or None."""
        return self.index_col

    @property
    def use_arrow(self):
        return self.use_arrow

    @property
    def return_arrow(self):
        return self.return_arrow

    @property
    def return_incomplete(self):
        return self.return_incomplete

    @property
    def domain_index(self):
        """Apply Array.domain_index with query parameters."""
        return self.domain_index

    @property
    def multi_index(self):
        """Apply Array.multi_index with query parameters."""
        # Delayed to avoid circular import
        from .multirange_indexing import MultiRangeIndexer
        return MultiRangeIndexer(self.array, query=self)

    @property
    def df(self):
        """Apply Array.multi_index with query parameters and return result
           as a Pandas dataframe."""
        # Delayed to avoid circular import
        from .multirange_indexing import DataFrameIndexer
        return DataFrameIndexer(self.array, query=self, use_arrow=self.use_arrow)

    def get_stats(self, print_out=True, json=False):
        """Retrieves the stats from a TileDB query.

        :param print_out: Print string to console (default True), or return as string
        :param json: Return stats JSON object (default: False)
        """
        pyquery = self.array.pyquery
        if pyquery is None:
            return ""
        stats = self.array.pyquery.get_stats()

        if json:
            import json
            output = json.loads(stats)
        else:
            output = stats

        if print_out:
            print(output)
        else:
            return output

    def submit(self):
        """An alias for calling the regular indexer [:]"""
        return self[:]


cdef class DenseArrayImpl(Array):
    """Class representing a dense TileDB array.

    Inherits properties and methods of :py:class:`tiledb.Array`.

    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if self.schema.sparse:
            raise ValueError(f"Array at {self.uri} is not a dense array")
        return

    @staticmethod
    def from_numpy(uri, np.ndarray array, ctx=None, **kw):
        warnings.warn(
            "tiledb.DenseArray.from_numpy is deprecated and is slated for "
            "removal in version 0.20.0. Use tiledb.from_numpy",
            DeprecationWarning,
        )
        assert tiledbpy_version < (0, 20, 0)

        return sparse_array_from_numpy(uri=uri, array=array, ctx=ctx, **kw)

    def __len__(self):
        return self.domain.shape[0]

    def __getitem__(self, object selection):
        """Retrieve data cells for an item or region of the array.

        :param tuple selection: An int index, slice or tuple of integer/slice objects,
            specifying the selected subarray region for each dimension of the DenseArray.
        :rtype: :py:class:`numpy.ndarray` or :py:class:`collections.OrderedDict`
        :returns: If the dense array has a single attribute then a Numpy array of corresponding shape/dtype \
                is returned for that attribute.  If the array has multiple attributes, a \
                :py:class:`collections.OrderedDict` is returned with dense Numpy subarrays \
                for each attribute.
        :raises IndexError: invalid or unsupported index selection
        :raises: :py:exc:`tiledb.TileDBError`

        **Example:**

        >>> import tiledb, numpy as np, tempfile
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     # Creates array 'array' on disk.
        ...     A = tiledb.from_numpy(tmp + "/array",  np.ones((100, 100)))
        ...     # Many aspects of Numpy's fancy indexing are supported:
        ...     A[1:10, ...].shape
        ...     A[1:10, 20:99].shape
        ...     A[1, 2].shape
        (9, 100)
        (9, 79)
        ()
        >>> # Subselect on attributes when reading:
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
        ...     schema = tiledb.ArraySchema(domain=dom,
        ...         attrs=(tiledb.Attr(name="a1", dtype=np.int64),
        ...                tiledb.Attr(name="a2", dtype=np.int64)))
        ...     tiledb.DenseArray.create(tmp + "/array", schema)
        ...     with tiledb.DenseArray(tmp + "/array", mode='w') as A:
        ...         A[0:10] = {"a1": np.zeros((10)), "a2": np.ones((10))}
        ...     with tiledb.DenseArray(tmp + "/array", mode='r') as A:
        ...         # Access specific attributes individually.
        ...         A[0:5]["a1"]
        ...         A[0:5]["a2"]
        array([0, 0, 0, 0, 0])
        array([1, 1, 1, 1, 1])

        """
        if self.view_attr:
            result = self.subarray(selection, attrs=(self.view_attr,))
            return result[self.view_attr]
        else:
            result = self.subarray(selection)
            return result

    def __repr__(self):
        if self.isopen:
            return "DenseArray(uri={0!r}, mode={1}, ndim={2})"\
                .format(self.uri, self.mode, self.schema.ndim)
        else:
            return "DenseArray(uri={0!r}, mode=closed)".format(self.uri)

    def query(self, attrs=None, attr_cond=None, cond=None, dims=None,
              coords=False, order='C', use_arrow=None, return_arrow=False,
              return_incomplete=False):
        """
        Construct a proxy Query object for easy subarray queries of cells
        for an item or region of the array across one or more attributes.

        Optionally subselect over attributes, return dense result coordinate values,
        and specify a layout a result layout / cell-order.

        :param attrs: the DenseArray attributes to subselect over.
            If attrs is None (default) all array attributes will be returned.
            Array attributes can be defined by name or by positional index.
        :param cond: the str expression to filter attributes or dimensions on. The expression must be parsable by tiledb.QueryCondition(). See help(tiledb.QueryCondition) for more details.
        :param dims: the DenseArray dimensions to subselect over. If dims is None (default)
            then no dimensions are returned, unless coords=True.
        :param coords: if True, return array of coodinate value (default False).
        :param order: 'C', 'F', 'U', or 'G' (row-major, col-major, unordered, TileDB global order)
        :param mode: "r" to read (default), "d" to delete
        :param use_arrow: if True, return dataframes via PyArrow if applicable.
        :param return_arrow: if True, return results as a PyArrow Table if applicable.
        :param return_incomplete: if True, initialize and return an iterable Query object over the indexed range.
            Consuming this iterable returns a result set for each TileDB incomplete query.
            See usage example in 'examples/incomplete_iteration.py'.
            To retrieve the estimated result sizes for the query ranges, use:
                `A.query(..., return_incomplete=True)[...].est_result_size()`
            If False (default False), queries will be internally run to completion by resizing buffers and
            resubmitting until query is complete.
        :return: A proxy Query object that can be used for indexing into the DenseArray
            over the defined attributes, in the given result layout (order).

        :raises ValueError: array is not opened for reads (mode = 'r')
        :raises: :py:exc:`tiledb.TileDBError`

        **Example:**

        >>> # Subselect on attributes when reading:
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
        ...     schema = tiledb.ArraySchema(domain=dom,
        ...         attrs=(tiledb.Attr(name="a1", dtype=np.int64),
        ...                tiledb.Attr(name="a2", dtype=np.int64)))
        ...     tiledb.DenseArray.create(tmp + "/array", schema)
        ...     with tiledb.DenseArray(tmp + "/array", mode='w') as A:
        ...         A[0:10] = {"a1": np.zeros((10)), "a2": np.ones((10))}
        ...     with tiledb.DenseArray(tmp + "/array", mode='r') as A:
        ...         # Access specific attributes individually.
        ...         A.query(attrs=("a1",))[0:5]
        OrderedDict([('a1', array([0, 0, 0, 0, 0]))])

        """
        if not self.isopen or self.mode != 'r':
            raise TileDBError("DenseArray must be opened in read mode")

        if attr_cond is not None:
            if cond is not None:
                raise TileDBError("Both `attr_cond` and `cond` were passed. "
                    "Only use `cond`."
                )

            raise TileDBError(
                "`attr_cond` is no longer supported. You must use `cond`. "
                "This message will be removed in 0.21.0."
            )

        return Query(self, attrs=attrs, cond=cond, dims=dims,
                      coords=coords, order=order,
                      use_arrow=use_arrow, return_arrow=return_arrow,
                      return_incomplete=return_incomplete)

    def subarray(self, selection, attrs=None, cond=None, attr_cond=None,
                 coords=False, order=None):
        """Retrieve data cells for an item or region of the array.

        Optionally subselect over attributes, return dense result coordinate values,
        and specify a layout a result layout / cell-order.

        :param selection: tuple of scalar and/or slice objects
        :param cond: the str expression to filter attributes or dimensions on. The expression must be parsable by tiledb.QueryCondition(). See help(tiledb.QueryCondition) for more details.
        :param coords: if True, return array of coordinate value (default False).
        :param attrs: the DenseArray attributes to subselect over.
            If attrs is None (default) all array attributes will be returned.
            Array attributes can be defined by name or by positional index.
        :param order: 'C', 'F', 'U', or 'G' (row-major, col-major, unordered, TileDB global order)
        :returns: If the dense array has a single attribute then a Numpy array of corresponding shape/dtype \
            is returned for that attribute.  If the array has multiple attributes, a \
            :py:class:`collections.OrderedDict` is returned with dense Numpy subarrays for each attribute.
        :raises IndexError: invalid or unsupported index selection
        :raises: :py:exc:`tiledb.TileDBError`

        **Example:**

        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
        ...     schema = tiledb.ArraySchema(domain=dom,
        ...         attrs=(tiledb.Attr(name="a1", dtype=np.int64),
        ...                tiledb.Attr(name="a2", dtype=np.int64)))
        ...     tiledb.DenseArray.create(tmp + "/array", schema)
        ...     with tiledb.DenseArray(tmp + "/array", mode='w') as A:
        ...         A[0:10] = {"a1": np.zeros((10)), "a2": np.ones((10))}
        ...     with tiledb.DenseArray(tmp + "/array", mode='r') as A:
        ...         # A[0:5], attribute a1, row-major without coordinates
        ...         A.subarray((slice(0, 5),), attrs=("a1",), coords=False, order='C')
        OrderedDict([('a1', array([0, 0, 0, 0, 0]))])

        """
        if not self.isopen or self.mode != 'r':
            raise TileDBError("DenseArray must be opened in read mode")

        if attr_cond is not None:
            if cond is not None:
                raise TileDBError("Both `attr_cond` and `cond` were passed. "
                    "Only use `cond`."
                )

            raise TileDBError(
                "`attr_cond` is no longer supported. You must use `cond`. "
                "This message will be removed in 0.21.0."
            )

        cdef tiledb_layout_t layout = TILEDB_UNORDERED
        if order is None or order == 'C':
            layout = TILEDB_ROW_MAJOR
        elif order == 'F':
            layout = TILEDB_COL_MAJOR
        elif order == 'G':
            layout = TILEDB_GLOBAL_ORDER
        elif order == 'U':
            pass
        else:
            raise ValueError("order must be 'C' (TILEDB_ROW_MAJOR), "\
                             "'F' (TILEDB_COL_MAJOR), "\
                             "'G' (TILEDB_GLOBAL_ORDER), "\
                             "or 'U' (TILEDB_UNORDERED)")
        attr_names = list()
        if coords == True:
            attr_names.extend(self.schema.domain.dim(i).name for i in range(self.schema.ndim))
        elif coords:
            attr_names.extend(coords)

        if attrs is None:
            attr_names.extend(
                self.schema.attr(i)._internal_name for i in range(self.schema.nattr)
            )
        else:
            attr_names.extend(self.schema.attr(a).name for a in attrs)

        selection = index_as_tuple(selection)
        idx = replace_ellipsis(self.schema.domain.ndim, selection)
        idx, drop_axes = replace_scalars_slice(self.schema.domain, idx)
        subarray = index_domain_subarray(self, self.schema.domain, idx)
        # Note: we included dims (coords) above to match existing semantics
        out = self._read_dense_subarray(subarray, attr_names, cond, layout,
                                        coords)
        if any(s.step for s in idx):
            steps = tuple(slice(None, None, s.step) for s in idx)
            for (k, v) in out.items():
                out[k] = v.__getitem__(steps)
        if drop_axes:
            for (k, v) in out.items():
                out[k] = v.squeeze(axis=drop_axes)
        # attribute is anonymous, just return the result
        if not coords and self.schema.nattr == 1:
            attr = self.schema.attr(0)
            if attr.isanon:
                return out[attr._internal_name]
        return out


    cdef _read_dense_subarray(self, list subarray, list attr_names,
                              object cond, tiledb_layout_t layout,
                              bint include_coords):

        from tiledb.main import PyQuery

        q = PyQuery(self._ctx_(), self, tuple(attr_names), tuple(), <int32_t>layout, False)
        self.pyquery = q

        if cond is not None:
            from .query_condition import QueryCondition

            if isinstance(cond, str):
                q.set_cond(QueryCondition(cond))
            elif isinstance(cond, QueryCondition):
                raise TileDBError(
                    "Passing `tiledb.QueryCondition` to `cond` is no longer "
                    "supported as of 0.19.0. Instead of "
                    "`cond=tiledb.QueryCondition('expression')` "
                    "you must use `cond='expression'`. This message will be "
                    "removed in 0.21.0.",
                )
            else:
                raise TypeError("`cond` expects type str.")

        q.set_ranges([list([x]) for x in subarray])
        q.submit()
        cdef object results = OrderedDict()
        results = q.results()

        out = OrderedDict()

        cdef tuple output_shape
        domain_dtype = self.domain.dtype
        is_datetime = domain_dtype.kind == 'M'
        # Using the domain check is valid because dense arrays are homogeneous
        if is_datetime:
            output_shape = \
                tuple(_tiledb_datetime_extent(subarray[r][0], subarray[r][1])
                      for r in range(self.schema.ndim))
        else:
            output_shape = \
                tuple(int(subarray[r][1]) - int(subarray[r][0]) + 1
                      for r in range(self.schema.ndim))

        cdef Py_ssize_t nattr = len(attr_names)
        cdef int i
        for i in range(nattr):
            name = attr_names[i]
            if not self.schema.domain.has_dim(name) and self.schema.attr(name).isvar:
                # for var arrays we create an object array
                dtype = object
                out[name] = q.unpack_buffer(name, results[name][0], results[name][1]).reshape(output_shape)
            else:
                dtype = q.buffer_dtype(name)

                # <TODO> sanity check the TileDB buffer size against schema?
                # <TODO> add assert to verify np.require doesn't copy?
                arr = results[name][0]
                arr.dtype = dtype
                if len(arr) == 0:
                    # special case: the C API returns 0 len for blank arrays
                    arr = np.zeros(output_shape, dtype=dtype)
                elif len(arr) != np.prod(output_shape):
                    raise Exception("Mismatched output array shape! (arr.shape: {}, output.shape: {}".format(arr.shape, output_shape))

                if layout == TILEDB_ROW_MAJOR:
                    arr.shape = output_shape
                    arr = np.require(arr, requirements='C')
                elif layout == TILEDB_COL_MAJOR:
                    arr.shape = output_shape
                    arr = np.require(arr, requirements='F')
                else:
                    arr.shape = np.prod(output_shape)

                out[name] = arr
        return out

    def __setitem__(self, object selection, object val):
        """Set / update dense data cells

        :param tuple selection: An int index, slice or tuple of integer/slice objects,
            specifiying the selected subarray region for each dimension of the DenseArray.
        :param value: a dictionary of array attribute values, values must able to be converted to n-d numpy arrays.\
            if the number of attributes is one, then a n-d numpy array is accepted.
        :type value: dict or :py:class:`numpy.ndarray`
        :raises IndexError: invalid or unsupported index selection
        :raises ValueError: value / coordinate length mismatch
        :raises: :py:exc:`tiledb.TileDBError`

        **Example:**

        >>> import tiledb, numpy as np, tempfile
        >>> # Write to single-attribute 2D array
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     # Create an array initially with all zero values
        ...     with tiledb.from_numpy(tmp + "/array",  np.zeros((2, 2))) as A:
        ...         pass
        ...     with tiledb.DenseArray(tmp + "/array", mode='w') as A:
        ...         # Write to the single (anonymous) attribute
        ...         A[:] = np.array(([1,2], [3,4]))
        >>>
        >>> # Write to multi-attribute 2D array
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     dom = tiledb.Domain(
        ...         tiledb.Dim(domain=(0, 1), tile=2, dtype=np.uint64),
        ...         tiledb.Dim(domain=(0, 1), tile=2, dtype=np.uint64))
        ...     schema = tiledb.ArraySchema(domain=dom,
        ...         attrs=(tiledb.Attr(name="a1", dtype=np.int64),
        ...                tiledb.Attr(name="a2", dtype=np.int64)))
        ...     tiledb.DenseArray.create(tmp + "/array", schema)
        ...     with tiledb.DenseArray(tmp + "/array", mode='w') as A:
        ...         # Write to each attribute
        ...         A[0:2, 0:2] = {"a1": np.array(([-3, -4], [-5, -6])),
        ...                        "a2": np.array(([1, 2], [3, 4]))}

        """
        selection_tuple = (selection,) if not isinstance(selection, tuple) else selection
        self._setitem_impl(selection, val, dict())

    def _setitem_impl(self, object selection, object val, dict nullmaps):
        """Implementation for setitem with optional support for validity bitmaps."""
        if not self.isopen or self.mode != 'w':
            raise TileDBError("DenseArray is not opened for writing")

        domain = self.domain
        cdef tuple idx = replace_ellipsis(domain.ndim, index_as_tuple(selection))
        idx,_drop = replace_scalars_slice(domain, idx)
        cdef object subarray = index_domain_subarray(self, domain, idx)
        cdef list attributes = list()
        cdef list values = list()
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)

        if isinstance(val, np.ndarray):
            subarray_shape = tuple(int(dim[1]-dim[0]+1) for dim in subarray)
            try:
                np.broadcast_shapes(subarray_shape, val.shape)
            except ValueError:
                raise ValueError(
                    "shape mismatch; data dimensions do not match the domain "
                    f"given in array schema ({subarray_shape} != {val.shape})"
                )

        if isinstance(val, dict):
            for attr_idx in range(self.schema.nattr):
                attr = self.schema.attr(attr_idx)
                k = attr.name
                v = val[k]
                attr = self.schema.attr(k)
                attributes.append(attr._internal_name)
                # object arrays are var-len and handled later
                if type(v) is np.ndarray and v.dtype is not np.dtype('O'):
                    v = np.ascontiguousarray(v, dtype=attr.dtype)
                values.append(v)
        elif np.isscalar(val):
            for i in range(self.schema.nattr):
                attr = self.schema.attr(i)
                subarray_shape = tuple(int(subarray[r][1] - subarray[r][0]) + 1
                                       for r in range(len(subarray)))
                attributes.append(attr._internal_name)
                A = np.empty(subarray_shape, dtype=attr.dtype)
                A[:] = val
                values.append(A)
        elif self.schema.nattr == 1:
            attr = self.schema.attr(0)
            attributes.append(attr._internal_name)
            # object arrays are var-len and handled later
            if type(val) is np.ndarray and val.dtype is not np.dtype('O'):
                val = np.ascontiguousarray(val, dtype=attr.dtype)
            values.append(val)
        elif self.view_attr is not None:
            # Support single-attribute assignment for multi-attr array
            # This is a hack pending
            #   https://github.com/TileDB-Inc/TileDB/issues/1162
            # (note: implicitly relies on the fact that we treat all arrays
            #  as zero initialized as long as query returns TILEDB_OK)
            # see also: https://github.com/TileDB-Inc/TileDB-Py/issues/128
            if self.schema.nattr == 1:
                attributes.append(self.schema.attr(0).name)
                values.append(val)
            else:
                dtype = self.schema.attr(self.view_attr).dtype
                with DenseArrayImpl(self.uri, 'r', ctx=Ctx(self.ctx.config())) as readable:
                    current = readable[selection]
                current[self.view_attr] = \
                    np.ascontiguousarray(val, dtype=dtype)
                # `current` is an OrderedDict
                attributes.extend(current.keys())
                values.extend(current.values())
        else:
            raise ValueError("ambiguous attribute assignment, "
                             "more than one array attribute "
                             "(use a dict({'attr': val}) to "
                             "assign multiple attributes)")

        if nullmaps:
            for key,val in nullmaps.items():
                if not self.schema.has_attr(key):
                    raise TileDBError("Cannot set validity for non-existent attribute.")
                if not self.schema.attr(key).isnullable:
                    raise ValueError("Cannot set validity map for non-nullable attribute.")
                if not isinstance(val, np.ndarray):
                    raise TypeError(f"Expected NumPy array for attribute '{key}' "
                                    f"validity bitmap, got {type(val)}")
                if val.dtype != np.uint8:
                    raise TypeError(f"Expected NumPy uint8 array for attribute '{key}' "
                                    f"validity bitmap, got {val.dtype}")

        _write_array(ctx_ptr,
                     self.ptr, self,
                     subarray,
                     attributes,
                     values,
                     nullmaps,
                     self.last_fragment_info,
                     False)
        return

    def __array__(self, dtype=None, **kw):
        """Implementation of numpy __array__ protocol (internal).

        :return: Numpy ndarray resulting from indexing the entire array.

        """
        if self.view_attr is None and self.nattr > 1:
            raise ValueError("cannot call __array__ for TileDB array with more than one attribute")
        cdef unicode name
        if self.view_attr:
            name = self.view_attr
        else:
            name = self.schema.attr(0).name
        array = self.read_direct(name=name)
        if dtype and array.dtype != dtype:
            return array.astype(dtype)
        return array

    def write_direct(self, np.ndarray array not None, **kw):
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
        cdef tiledb_array_t* array_ptr = self.ptr

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

    def read_direct(self, unicode name=None):
        """Read attribute directly with minimal overhead, returns a numpy ndarray over the entire domain

        :param str attr_name: read directly to an attribute name (default <anonymous>)
        :rtype: numpy.ndarray
        :return: numpy.ndarray of `attr_name` values over the entire array domain
        :raises: :py:exc:`tiledb.TileDBError`

        """
        if not self.isopen or self.mode != 'r':
            raise TileDBError("DenseArray is not opened for reading")
        ctx = self.ctx
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)
        cdef tiledb_array_t* array_ptr = self.ptr

        cdef unicode attr_name
        if name is None and self.schema.nattr != 1:
            raise ValueError(
                "read_direct with no provided attribute is ambiguous for multi-attribute arrays")
        elif name is None:
            attr = self.schema.attr(0)
            attr_name = attr._internal_name
        else:
            attr = self.schema.attr(name)
            attr_name = attr._internal_name
        order = 'C'
        cdef tiledb_layout_t cell_layout = TILEDB_ROW_MAJOR
        if self.schema.cell_order == 'col-major' and self.schema.tile_order == 'col-major':
            order = 'F'
            cell_layout = TILEDB_COL_MAJOR

        cdef ArraySchema schema = self.schema
        domain = schema.domain

        idx = tuple(slice(None) for _ in range(domain.ndim))
        subarray = index_domain_subarray(self, domain, idx)
        out = self._read_dense_subarray(subarray, [attr_name,], None, cell_layout, False)
        return out[attr_name]

# point query index a tiledb array (zips) columnar index vectors
def index_domain_coords(dom: Domain, idx: tuple, check_ndim: bool):
    """
    Returns a (zipped) coordinate array representation
    given coordinate indices in numpy's point indexing format
    """
    ndim = len(idx)

    if check_ndim:
        if ndim != dom.ndim:
            raise IndexError("sparse index ndim must match domain ndim: "
                            "{0!r} != {1!r}".format(ndim, dom.ndim))

    domain_coords = []
    for dim, sel in zip(dom, idx):
        dim_is_string = (np.issubdtype(dim.dtype, np.str_) or
            np.issubdtype(dim.dtype, np.bytes_))

        if dim_is_string:
            try:
                # ensure strings contain only ASCII characters
                domain_coords.append(np.array(sel, dtype=np.bytes_, ndmin=1))
            except Exception as exc:
                raise TileDBError(f'Dim\' strings may only contain ASCII characters')
        else:
            domain_coords.append(np.array(sel, dtype=dim.dtype, ndmin=1))

    idx = tuple(domain_coords)

    # check that all sparse coordinates are the same size and dtype
    dim0 = dom.dim(0)
    dim0_type = dim0.dtype
    len0 = len(idx[0])
    for dim_idx in range(ndim):
        dim_dtype = dom.dim(dim_idx).dtype
        if len(idx[dim_idx]) != len0:
            raise IndexError("sparse index dimension length mismatch")

        if np.issubdtype(dim_dtype, np.str_) or np.issubdtype(dim_dtype, np.bytes_):
            if not (np.issubdtype(idx[dim_idx].dtype, np.str_) or \
                    np.issubdtype(idx[dim_idx].dtype, np.bytes_)):
                raise IndexError("sparse index dimension dtype mismatch")
        elif idx[dim_idx].dtype != dim_dtype:
            raise IndexError("sparse index dimension dtype mismatch")

    return idx

def _setitem_impl_sparse(self: Array, selection, val, dict nullmaps):
    cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)

    if not self.isopen or self.mode != 'w':
        raise TileDBError("SparseArray is not opened for writing")

    set_dims_only = val is None
    sparse_attributes = list()
    sparse_values = list()
    idx = index_as_tuple(selection)
    sparse_coords = list(index_domain_coords(self.schema.domain, idx, not set_dims_only))

    if set_dims_only:
        _write_array(
            ctx_ptr, self.ptr, self,
            sparse_coords,
            sparse_attributes,
            sparse_values,
            nullmaps,
            self.last_fragment_info,
            True
        )
        return

    if not isinstance(val, dict):
        if self.nattr > 1:
            raise ValueError("Expected dict-like object {name: value} for multi-attribute "
                             "array.")
        val = dict({self.attr(0).name: val})

    # must iterate in Attr order to ensure that value order matches
    for attr_idx in range(self.schema.nattr):
        attr = self.attr(attr_idx)
        name = attr.name
        attr_val = val[name]

        try:
            if attr.isvar:
                # ensure that the value is array-convertible, for example: pandas.Series
                attr_val = np.asarray(attr_val)
            else:
                if (np.issubdtype(attr.dtype, np.string_) and not
                    (np.issubdtype(attr_val.dtype, np.string_) or attr_val.dtype == np.dtype('O'))):
                    raise ValueError("Cannot write a string value to non-string "
                                     "typed attribute '{}'!".format(name))

                attr_val = np.ascontiguousarray(attr_val, dtype=attr.dtype)

            if attr.isnullable and attr.name not in nullmaps:
                nullmaps[attr.name] = np.array([int(v is not None) for v in attr_val], dtype=np.uint8)

        except Exception as exc:
            raise ValueError(f"NumPy array conversion check failed for attr '{name}'") from exc

        # set nullmap if nullable attribute does not have a nullmap already set
        if attr.isnullable and attr.name not in nullmaps:
            nullmaps[attr.name] = np.ones(attr_val.shape)

        # if dtype is ASCII, ensure all characters are valid
        if attr.isascii:
            try:
                np.asarray(attr_val, dtype=np.bytes_)
            except Exception as exc:
                raise TileDBError(f'dtype of attr {attr.name} is "ascii" but attr_val contains invalid ASCII characters')

        ncells = sparse_coords[0].shape[0]
        if attr_val.size != ncells:
           raise ValueError("value length ({}) does not match "
                             "coordinate length ({})".format(attr_val.size, ncells))
        sparse_attributes.append(attr._internal_name)
        sparse_values.append(attr_val)

    if (len(sparse_attributes) != len(val.keys())) \
        or (len(sparse_values) != len(val.values())):
        raise TileDBError("Sparse write input data count does not match number of attributes")

    _write_array(
        ctx_ptr, self.ptr, self,
        sparse_coords,
        sparse_attributes,
        sparse_values,
        nullmaps,
        self.last_fragment_info,
        True
    )
    return

cdef class SparseArrayImpl(Array):
    """Class representing a sparse TileDB array (internal).

    Inherits properties and methods of :py:class:`tiledb.Array`.
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if not self.schema.sparse:
            raise ValueError("Array at '{}' is not a sparse array".format(self.uri))

        return

    def __len__(self):
        raise TypeError("SparseArray length is ambiguous; use shape[0]")

    def __setitem__(self, selection, val):
        """Set / update sparse data cells

        :param tuple selection: N coordinate value arrays (dim0, dim1, ...) where N in the ndim of the SparseArray,
            The format follows numpy sparse (point) indexing semantics.
        :param value: a dictionary of nonempty array attribute values, values must able to be converted to 1-d numpy arrays.\
            if the number of attributes is one, then a 1-d numpy array is accepted.
        :type value: dict or :py:class:`numpy.ndarray`
        :raises IndexError: invalid or unsupported index selection
        :raises ValueError: value / coordinate length mismatch
        :raises: :py:exc:`tiledb.TileDBError`

        **Example:**

        >>> import tiledb, numpy as np, tempfile
        >>> # Write to multi-attribute 2D array
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     dom = tiledb.Domain(
        ...         tiledb.Dim(domain=(0, 1), tile=2, dtype=np.uint64),
        ...         tiledb.Dim(domain=(0, 1), tile=2, dtype=np.uint64))
        ...     schema = tiledb.ArraySchema(domain=dom, sparse=True,
        ...         attrs=(tiledb.Attr(name="a1", dtype=np.int64),
        ...                tiledb.Attr(name="a2", dtype=np.int64)))
        ...     tiledb.SparseArray.create(tmp + "/array", schema)
        ...     with tiledb.SparseArray(tmp + "/array", mode='w') as A:
        ...         # Write in the corner cells (0,0) and (1,1) only.
        ...         I, J = [0, 1], [0, 1]
        ...         # Write to each attribute
        ...         A[I, J] = {"a1": np.array([1, 2]),
        ...                    "a2": np.array([3, 4])}

        """
        _setitem_impl_sparse(self, selection, val, dict())

    def __getitem__(self, object selection):
        """Retrieve nonempty cell data for an item or region of the array

        :param tuple selection: An int index, slice or tuple of integer/slice objects,
            specifying the selected subarray region for each dimension of the SparseArray.
        :rtype: :py:class:`collections.OrderedDict`
        :returns: An OrderedDict is returned with dimension and attribute names as keys. \
            Nonempty attribute values are returned as Numpy 1-d arrays.
        :raises IndexError: invalid or unsupported index selection
        :raises: :py:exc:`tiledb.TileDBError`

        **Example:**

        >>> import tiledb, numpy as np, tempfile
        >>> # Write to multi-attribute 2D array
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     dom = tiledb.Domain(
        ...         tiledb.Dim(name="y", domain=(0, 9), tile=2, dtype=np.uint64),
        ...         tiledb.Dim(name="x", domain=(0, 9), tile=2, dtype=np.uint64))
        ...     schema = tiledb.ArraySchema(domain=dom, sparse=True,
        ...         attrs=(tiledb.Attr(name="a1", dtype=np.int64),
        ...                tiledb.Attr(name="a2", dtype=np.int64)))
        ...     tiledb.SparseArray.create(tmp + "/array", schema)
        ...     with tiledb.SparseArray(tmp + "/array", mode='w') as A:
        ...         # Write in the twp cells (0,0) and (2,3) only.
        ...         I, J = [0, 2], [0, 3]
        ...         # Write to each attribute
        ...         A[I, J] = {"a1": np.array([1, 2]),
        ...                    "a2": np.array([3, 4])}
        ...     with tiledb.SparseArray(tmp + "/array", mode='r') as A:
        ...         # Return an OrderedDict with values and coordinates
        ...         A[0:3, 0:10]
        ...         # Return just the "x" coordinates values
        ...         A[0:3, 0:10]["x"]
        OrderedDict([('a1', array([1, 2])), ('a2', array([3, 4])), ('y', array([0, 2], dtype=uint64)), ('x', array([0, 3], dtype=uint64))])
        array([0, 3], dtype=uint64)

        With a floating-point array domain, index bounds are inclusive, e.g.:

        >>> # Return nonempty cells within a floating point array domain (fp index bounds are inclusive):
        >>> # A[5.0:579.9]

        """
        return self.subarray(selection)

    def query(self, attrs=None, cond=None, attr_cond=None, dims=None,
              index_col=True, coords=None, order='U', use_arrow=None,
              return_arrow=None, return_incomplete=False):
        """
        Construct a proxy Query object for easy subarray queries of cells
        for an item or region of the array across one or more attributes.

        Optionally subselect over attributes, return dense result coordinate values,
        and specify a layout a result layout / cell-order.

        :param attrs: the SparseArray attributes to subselect over.
            If attrs is None (default) all array attributes will be returned.
            Array attributes can be defined by name or by positional index.
        :param cond: the str expression to filter attributes or dimensions on. The expression must be parsable by tiledb.QueryCondition(). See help(tiledb.QueryCondition) for more details.
        :param dims: the SparseArray dimensions to subselect over. If dims is None (default)
            then all dimensions are returned, unless coords=False.
        :param index_col: For dataframe queries, override the saved index information,
            and only set specified index(es) in the final dataframe, or None.
        :param coords: (deprecated) if True, return array of coordinate value (default False).
        :param order: 'C', 'F', or 'G' (row-major, col-major, tiledb global order)
        :param mode: "r" to read
        :param use_arrow: if True, return dataframes via PyArrow if applicable.
        :param return_arrow: if True, return results as a PyArrow Table if applicable.
        :return: A proxy Query object that can be used for indexing into the SparseArray
            over the defined attributes, in the given result layout (order).

        **Example:**

        >>> import tiledb, numpy as np, tempfile
        >>> # Write to multi-attribute 2D array
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     dom = tiledb.Domain(
        ...         tiledb.Dim(name="y", domain=(0, 9), tile=2, dtype=np.uint64),
        ...         tiledb.Dim(name="x", domain=(0, 9), tile=2, dtype=np.uint64))
        ...     schema = tiledb.ArraySchema(domain=dom, sparse=True,
        ...         attrs=(tiledb.Attr(name="a1", dtype=np.int64),
        ...                tiledb.Attr(name="a2", dtype=np.int64)))
        ...     tiledb.SparseArray.create(tmp + "/array", schema)
        ...     with tiledb.SparseArray(tmp + "/array", mode='w') as A:
        ...         # Write in the twp cells (0,0) and (2,3) only.
        ...         I, J = [0, 2], [0, 3]
        ...         # Write to each attribute
        ...         A[I, J] = {"a1": np.array([1, 2]),
        ...                    "a2": np.array([3, 4])}
        ...     with tiledb.SparseArray(tmp + "/array", mode='r') as A:
        ...         A.query(attrs=("a1",), coords=False, order='G')[0:3, 0:10]
        OrderedDict([('a1', array([1, 2]))])

        """
        if not self.isopen or self.mode not in  ('r', 'd'):
            raise TileDBError("SparseArray must be opened in read or delete mode")

        if attr_cond is not None:
            if cond is not None:
                raise TileDBError("Both `attr_cond` and `cond` were passed. "
                    "Only use `cond`."
                )

            raise TileDBError(
                "`attr_cond` is no longer supported. You must use `cond`. "
                "This message will be removed in 0.21.0."
            )

        # backwards compatibility
        _coords = coords
        if dims is False:
            _coords = False
        elif dims is None and coords is None:
            _coords = True

        return Query(self, attrs=attrs, cond=cond, dims=dims,
                      coords=_coords, index_col=index_col, order=order,
                      use_arrow=use_arrow, return_arrow=return_arrow,
                      return_incomplete=return_incomplete)

    def subarray(self, selection, coords=True, attrs=None, cond=None,
                 attr_cond=None, order=None):
        """
        Retrieve dimension and data cells for an item or region of the array.

        Optionally subselect over attributes, return sparse result coordinate values,
        and specify a layout a result layout / cell-order.

        :param selection: tuple of scalar and/or slice objects
        :param cond: the str expression to filter attributes or dimensions on. The expression must be parsable by tiledb.QueryCondition(). See help(tiledb.QueryCondition) for more details.
        :param coords: if True, return array of coordinate value (default True).
        :param attrs: the SparseArray attributes to subselect over.
            If attrs is None (default) all array attributes will be returned.
            Array attributes can be defined by name or by positional index.
        :param order: 'C', 'F', or 'G' (row-major, col-major, tiledb global order)
        :returns: An OrderedDict is returned with dimension and attribute names as keys. \
            Nonempty attribute values are returned as Numpy 1-d arrays.

        **Example:**

        >>> import tiledb, numpy as np, tempfile
        >>> # Write to multi-attribute 2D array
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     dom = tiledb.Domain(
        ...         tiledb.Dim(name="y", domain=(0, 9), tile=2, dtype=np.uint64),
        ...         tiledb.Dim(name="x", domain=(0, 9), tile=2, dtype=np.uint64))
        ...     schema = tiledb.ArraySchema(domain=dom, sparse=True,
        ...         attrs=(tiledb.Attr(name="a1", dtype=np.int64),
        ...                tiledb.Attr(name="a2", dtype=np.int64)))
        ...     tiledb.SparseArray.create(tmp + "/array", schema)
        ...     with tiledb.SparseArray(tmp + "/array", mode='w') as A:
        ...         # Write in the twp cells (0,0) and (2,3) only.
        ...         I, J = [0, 2], [0, 3]
        ...         # Write to each attribute
        ...         A[I, J] = {"a1": np.array([1, 2]),
        ...                    "a2": np.array([3, 4])}
        ...     with tiledb.SparseArray(tmp + "/array", mode='r') as A:
        ...         # A[0:3, 0:10], attribute a1, row-major without coordinates
        ...         A.subarray((slice(0, 3), slice(0, 10)), attrs=("a1",), coords=False, order='G')
        OrderedDict([('a1', array([1, 2]))])

        """
        if not self.isopen or self.mode not in ('r', 'd'):
            raise TileDBError("SparseArray is not opened in read or delete mode")

        if attr_cond is not None:
            if cond is not None:
                raise TileDBError("Both `attr_cond` and `cond` were passed. "
                    "`attr_cond` is no longer supported. You must use `cond`. "
                )

            raise TileDBError(
                "`attr_cond` is no longer supported. You must use `cond`. "
                "This message will be removed in 0.21.0.",
            )

        cdef tiledb_layout_t layout = TILEDB_UNORDERED
        if order is None or order == 'U':
            layout = TILEDB_UNORDERED
        elif order == 'C':
            layout = TILEDB_ROW_MAJOR
        elif order == 'F':
            layout = TILEDB_COL_MAJOR
        elif order == 'G':
            layout = TILEDB_GLOBAL_ORDER
        else:
            raise ValueError("order must be 'C' (TILEDB_ROW_MAJOR), "\
                             "'F' (TILEDB_COL_MAJOR), "\
                             "'G' (TILEDB_GLOBAL_ORDER), "\
                             "or 'U' (TILEDB_UNORDERED)")

        attr_names = list()

        if attrs is None:
            attr_names.extend(self.schema.attr(i)._internal_name for i in range(self.schema.nattr))
        else:
            attr_names.extend(self.schema.attr(a)._internal_name for a in attrs)

        if coords == True:
            attr_names.extend(self.schema.domain.dim(i).name for i in range(self.schema.ndim))
        elif coords:
            attr_names.extend(coords)

        dom = self.schema.domain
        idx = index_as_tuple(selection)
        idx = replace_ellipsis(dom.ndim, idx)
        idx, drop_axes = replace_scalars_slice(dom, idx)
        subarray = index_domain_subarray(self, dom, idx)
        return self._read_sparse_subarray(subarray, attr_names, cond, layout)

    def __repr__(self):
        if self.isopen:
            return "SparseArray(uri={0!r}, mode={1}, ndim={2})"\
                .format(self.uri, self.mode, self.schema.ndim)
        else:
            return "SparseArray(uri={0!r}, mode=closed)".format(self.uri)

    cdef _read_sparse_subarray(self, list subarray, list attr_names,
                               object cond, tiledb_layout_t layout):
        cdef object out = OrderedDict()
        # all results are 1-d vectors
        cdef np.npy_intp dims[1]
        cdef Py_ssize_t nattr = len(attr_names)

        from tiledb.main import PyQuery

        q = PyQuery(self._ctx_(), self, tuple(attr_names), tuple(), <int32_t>layout, False)
        self.pyquery = q

        if cond is not None:
            from .query_condition import QueryCondition

            if isinstance(cond, str):
                q.set_cond(QueryCondition(cond))
            elif isinstance(cond, QueryCondition):
                raise TileDBError(
                    "Passing `tiledb.QueryCondition` to `cond` is no longer "
                    "supported as of 0.19.0. Instead of "
                    "`cond=tiledb.QueryCondition('expression')` "
                    "you must use `cond='expression'`. This message will be "
                    "removed in 0.21.0.",
                )
            else:
                raise TypeError("`cond` expects type str.")

        if self.mode == "r":
            q.set_ranges([list([x]) for x in subarray])

        q.submit()

        if self.mode == 'd':
            return

        cdef object results = OrderedDict()
        results = q.results()

        # collect a list of dtypes for resulting to construct array
        dtypes = list()
        for i in range(nattr):
            name, final_name = attr_names[i], attr_names[i]
            if name == '__attr':
                final_name = ''
            if self.schema._needs_var_buffer(name):
                if len(results[name][1]) > 0: # note: len(offsets) > 0
                    arr = q.unpack_buffer(name, results[name][0], results[name][1])
                else:
                    arr = results[name][0]
                    arr.dtype = self.schema.attr_or_dim_dtype(name)
                out[final_name] = arr
            else:
                if self.schema.domain.has_dim(name):
                    el_dtype = self.schema.domain.dim(name).dtype
                else:
                    el_dtype = self.attr(name).dtype
                arr = results[name][0]

                # this is a work-around for NumPy restrictions removed in 1.16
                if el_dtype == np.dtype('S0'):
                    out[final_name] = b''
                elif el_dtype == np.dtype('U0'):
                    out[final_name] = u''
                else:
                    arr.dtype = el_dtype
                    out[final_name] = arr

        return out

    def unique_dim_values(self, dim=None):
        if dim is not None and not isinstance(dim, str):
            raise ValueError("Given Dimension {} is not a string.".format(dim))

        if dim is not None and not self.domain.has_dim(dim):
            raise ValueError("Array does not contain Dimension '{}'.".format(dim))

        query = self.query(attrs=[])[:]

        if dim:
            dim_values = tuple(np.unique(query[dim]))
        else:
            dim_values = OrderedDict()
            for dim in query:
                dim_values[dim] = tuple(np.unique(query[dim]))

        return dim_values


def consolidate(uri, key=None, config=None, ctx=None, fragment_uris=None, timestamp=None):
    """Consolidates TileDB array fragments for improved read performance

    :param str uri: URI to the TileDB Array
    :param str key: (default None) Key to decrypt array if the array is encrypted
    :param tiledb.Config config: The TileDB Config with consolidation parameters set
    :param tiledb.Ctx ctx: (default None) The TileDB Context
    :param fragment_uris: (default None) Consolidate the array using a list of fragment file names
    :param timestamp: (default None) If not None, consolidate the array using the given tuple(int, int) UNIX seconds range (inclusive). This argument will be ignored if `fragment_uris` is passed.
    :rtype: str or bytes
    :return: path (URI) to the consolidated TileDB Array
    :raises TypeError: cannot convert path to unicode string
    :raises: :py:exc:`tiledb.TileDBError`

    Rather than passing the timestamp into this function, it may be set with
    the config parameters `"sm.vacuum.timestamp_start"`and
    `"sm.vacuum.timestamp_end"` which takes in a time in UNIX seconds. If both
    are set then this function's `timestamp` argument will be used.

    **Example:**

    >>> import tiledb, tempfile, numpy as np, os
    >>> path = tempfile.mkdtemp()

    >>> with tiledb.from_numpy(path, np.zeros(4), timestamp=1) as A:
    ...     pass
    >>> with tiledb.open(path, 'w', timestamp=2) as A:
    ...     A[:] = np.ones(4, dtype=np.int64)
    >>> with tiledb.open(path, 'w', timestamp=3) as A:
    ...     A[:] = np.ones(4, dtype=np.int64)
    >>> with tiledb.open(path, 'w', timestamp=4) as A:
    ...     A[:] = np.ones(4, dtype=np.int64)
    >>> len(tiledb.array_fragments(path))
    4

    >>> fragment_names = [
    ...     os.path.basename(f) for f in tiledb.array_fragments(path).uri
    ... ]
    >>> array_uri = tiledb.consolidate(
    ...    path, fragment_uris=[fragment_names[1], fragment_names[3]]
    ... )
    >>> len(tiledb.array_fragments(path))
    3

    """
    if not ctx:
        ctx = default_ctx()

    if fragment_uris is not None:
        if timestamp is not None:
            warnings.warn(
                "The `timestamp` argument will be ignored and only fragments "
                "passed to `fragment_uris` will be consolidate",
                DeprecationWarning,
            )
        return _consolidate_uris(
            uri=uri, key=key, config=config, ctx=ctx, fragment_uris=fragment_uris)
    else:
        return _consolidate_timestamp(
            uri=uri, key=key, config=config, ctx=ctx, timestamp=timestamp)

def _consolidate_uris(uri, key=None, config=None, ctx=None, fragment_uris=None):
    cdef int rc = TILEDB_OK

    cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(ctx)

    if config is None:
        config = ctx.config()

    cdef tiledb_config_t* config_ptr = NULL
    if config is not None:
        config_ptr = <tiledb_config_t*>PyCapsule_GetPointer(
            config.__capsule__(), "config")
    cdef bytes buri = unicode_path(uri)
    cdef const char* array_uri_ptr = PyBytes_AS_STRING(buri)

    cdef const char **fragment_uri_buf = <const char **>malloc(
        len(fragment_uris) * sizeof(char *))

    for i, frag_uri in enumerate(fragment_uris):
        fragment_uri_buf[i] = PyUnicode_AsUTF8(frag_uri)

    if key is not None:
        config["sm.encryption_key"] = key

    rc = tiledb_array_consolidate_fragments(
        ctx_ptr, array_uri_ptr, fragment_uri_buf, len(fragment_uris), config_ptr)
    if rc != TILEDB_OK:
        _raise_ctx_err(ctx_ptr, rc)

    free(fragment_uri_buf)

    return uri

def _consolidate_timestamp(uri, key=None, config=None, ctx=None, timestamp=None):
    cdef int rc = TILEDB_OK

    cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(ctx)

    if timestamp is not None:
        warnings.warn(
            "The `timestamp` argument is deprecated; pass a list of "
            "fragment URIs to consolidate with `fragment_uris`",
            DeprecationWarning,
        )

        if config is None:
            config = ctx.config()

        if not isinstance(timestamp, tuple) and len(timestamp) != 2:
            raise TypeError("'timestamp' argument expects tuple(start: int, end: int)")

        if timestamp[0] is not None:
            config["sm.consolidation.timestamp_start"] = timestamp[0]
        if timestamp[1] is not None:
            config["sm.consolidation.timestamp_end"] = timestamp[1]

    cdef tiledb_config_t* config_ptr = NULL
    if config is not None:
        config_ptr = <tiledb_config_t*>PyCapsule_GetPointer(
            config.__capsule__(), "config")
    cdef bytes buri = unicode_path(uri)
    cdef const char* array_uri_ptr = PyBytes_AS_STRING(buri)

    # encryption key
    cdef:
        bytes bkey
        tiledb_encryption_type_t key_type = TILEDB_NO_ENCRYPTION
        void* key_ptr = NULL
        unsigned int key_len = 0

    if key is not None:
        if isinstance(key, str):
            bkey = key.encode('ascii')
        else:
            bkey = bytes(key)
        key_type = TILEDB_AES_256_GCM
        key_ptr = <void *> PyBytes_AS_STRING(bkey)
        #TODO: unsafe cast here ssize_t -> uint64_t
        key_len = <unsigned int> PyBytes_GET_SIZE(bkey)

    with nogil:
        rc = tiledb_array_consolidate_with_key(
            ctx_ptr, array_uri_ptr, key_type, key_ptr, key_len, config_ptr)
    if rc != TILEDB_OK:
        _raise_ctx_err(ctx_ptr, rc)
    return uri

def object_type(uri, ctx=None):
    """Returns the TileDB object type at the specified path (URI)

    :param str path: path (URI) of the TileDB resource
    :rtype: str
    :param tiledb.Ctx ctx: The TileDB Context
    :return: object type string
    :raises TypeError: cannot convert path to unicode string

    """
    if not ctx:
        ctx = default_ctx()
    cdef int rc = TILEDB_OK
    cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(ctx)
    cdef bytes buri = unicode_path(uri)
    cdef const char* path_ptr = PyBytes_AS_STRING(buri)
    cdef tiledb_object_t obj = TILEDB_INVALID
    with nogil:
        rc = tiledb_object_type(ctx_ptr, path_ptr, &obj)
    if rc != TILEDB_OK:
        check_error(ctx, rc)
    objtype = None
    if obj == TILEDB_ARRAY:
        objtype = "array"
    # removed in libtiledb 1.7
    #elif obj == TILEDB_KEY_VALUE:
    #    objtype = "kv"
    elif obj == TILEDB_GROUP:
        objtype = "group"
    return objtype


def remove(uri, ctx=None):
    """Removes (deletes) the TileDB object at the specified path (URI)

    :param str uri: URI of the TileDB resource
    :param tiledb.Ctx ctx: The TileDB Context
    :raises TypeError: uri cannot be converted to a unicode string
    :raises: :py:exc:`tiledb.TileDBError`

    """
    if not ctx:
        ctx = default_ctx()
    cdef int rc = TILEDB_OK
    cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(ctx)
    cdef bytes buri = unicode_path(uri)
    cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
    with nogil:
        rc = tiledb_object_remove(ctx_ptr, uri_ptr)
    if rc != TILEDB_OK:
        check_error(ctx, rc)
    return


def move(old_uri, new_uri, ctx=None):
    """Moves a TileDB resource (group, array, key-value).

    :param tiledb.Ctx ctx: The TileDB Context
    :param str old_uri: path (URI) of the TileDB resource to move
    :param str new_uri: path (URI) of the destination
    :raises TypeError: uri cannot be converted to a unicode string
    :raises: :py:exc:`TileDBError`
    """
    if not ctx:
        ctx = default_ctx()
    cdef int rc = TILEDB_OK
    cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(ctx)
    cdef bytes b_old_path = unicode_path(old_uri)
    cdef bytes b_new_path = unicode_path(new_uri)
    cdef const char* old_path_ptr = PyBytes_AS_STRING(b_old_path)
    cdef const char* new_path_ptr = PyBytes_AS_STRING(b_new_path)
    with nogil:
        rc = tiledb_object_move(ctx_ptr, old_path_ptr, new_path_ptr)
    if rc != TILEDB_OK:
        check_error(ctx, rc)
    return

cdef int vfs_ls_callback(const char* path_ptr, void* py_list):
    cdef list result_list
    cdef unicode path
    try:
        result_list = <list?>py_list
        path = path_ptr.decode('UTF-8')
        result_list.append(path)
    except StopIteration:
        return 0
    return 1

cdef int walk_callback(const char* path_ptr, tiledb_object_t obj, void* pyfunc):
    objtype = None
    if obj == TILEDB_GROUP:
        objtype = "group"
    if obj == TILEDB_ARRAY:
        objtype = "array"
    # removed in 1.7
    #elif obj == TILEDB_KEY_VALUE:
    #    objtype = "kv"
    try:
        (<object> pyfunc)(path_ptr.decode('UTF-8'), objtype)
    except StopIteration:
        return 0
    return 1


def ls(path, func, ctx=None):
    """Lists TileDB resources and applies a callback that have a prefix of ``path`` (one level deep).

    :param str path: URI of TileDB group object
    :param function func: callback to execute on every listed TileDB resource,\
            URI resource path and object type label are passed as arguments to the callback
    :param tiledb.Ctx ctx: TileDB context
    :raises TypeError: cannot convert path to unicode string
    :raises: :py:exc:`tiledb.TileDBError`

    """
    cdef tiledb_ctx_t* ctx_ptr = NULL

    if not ctx:
        ctx = default_ctx()
    cdef bytes bpath = unicode_path(path)
    ctx_ptr = safe_ctx_ptr(ctx)
    check_error(ctx,
                tiledb_object_ls(ctx_ptr, bpath, walk_callback, <void*> func))
    return


def walk(path, func, order="preorder", ctx=None):
    """Recursively visits TileDB resources and applies a callback to resources that have a prefix of ``path``

    :param str path: URI of TileDB group object
    :param function func: callback to execute on every listed TileDB resource,\
            URI resource path and object type label are passed as arguments to the callback
    :param tiledb.Ctx ctx: The TileDB context
    :param str order: 'preorder' (default) or 'postorder' tree traversal
    :raises TypeError: cannot convert path to unicode string
    :raises ValueError: unknown order
    :raises: :py:exc:`tiledb.TileDBError`

    """
    cdef tiledb_ctx_t* ctx_ptr = NULL
    if not ctx:
        ctx = default_ctx()
    cdef bytes bpath = unicode_path(path)
    cdef tiledb_walk_order_t walk_order
    if order == "postorder":
        walk_order = TILEDB_POSTORDER
    elif order == "preorder":
        walk_order = TILEDB_PREORDER
    else:
        raise ValueError("unknown walk order {}".format(order))
    ctx_ptr = safe_ctx_ptr(ctx)
    check_error(ctx,
                tiledb_object_walk(ctx_ptr, bpath, walk_order, walk_callback, <void*> func))
    return

def vacuum(uri, config=None, ctx=None, timestamp=None):
    """
    Vacuum underlying array fragments after consolidation.

    :param str uri: URI of array to be vacuumed
    :param config: Override the context configuration for vacuuming.
        Defaults to None, inheriting the context parameters.
    :param (ctx: tiledb.Ctx, optional): Context. Defaults to
        `tiledb.default_ctx()`.
    :raises TypeError: cannot convert `uri` to unicode string
    :raises: :py:exc:`tiledb.TileDBError`

    This operation of this function is controlled by
    the `"sm.vacuum.mode"` parameter, which accepts the values ``fragments``,
    ``fragment_meta``, and ``array_meta``. Rather than passing the timestamp
    into this function, it may be set by using `"sm.vacuum.timestamp_start"`and
    `"sm.vacuum.timestamp_end"` which takes in a time in UNIX seconds. If both
    are set then this function's `timestamp` argument will be used.

    **Example:**

    >>> import tiledb, numpy as np
    >>> import tempfile
    >>> path = tempfile.mkdtemp()
    >>> with tiledb.from_numpy(path, np.random.rand(4)) as A:
    ...     pass # make sure to close
    >>> with tiledb.open(path, 'w') as A:
    ...     for i in range(4):
    ...         A[:] = np.ones(4, dtype=np.int64) * i
    >>> paths = tiledb.VFS().ls(path)
    >>> # should be 12 (2 base files + 2*5 fragment+ok files)
    >>> (); len(paths); () # doctest:+ELLIPSIS
    (...)
    >>> () ; tiledb.consolidate(path) ; () # doctest:+ELLIPSIS
    (...)
    >>> tiledb.vacuum(path)
    >>> paths = tiledb.VFS().ls(path)
    >>> # should now be 4 ( base files + 2 fragment+ok files)
    >>> (); len(paths); () # doctest:+ELLIPSIS
    (...)

    """
    cdef tiledb_ctx_t* ctx_ptr = NULL
    cdef tiledb_config_t* config_ptr = NULL

    if not ctx:
        ctx = default_ctx()

    if timestamp is not None:
        warnings.warn("Partial vacuuming via timestamp will be deprecrated in "
                      "a future release and replaced by passing in fragment URIs.",
                      DeprecationWarning)

        if config is None:
            config = Config()

        if not isinstance(timestamp, tuple) and len(timestamp) != 2:
            raise TypeError("'timestamp' argument expects tuple(start: int, end: int)")

        if timestamp[0] is not None:
            config["sm.vacuum.timestamp_start"] = timestamp[0]
        if timestamp[1] is not None:
            config["sm.vacuum.timestamp_end"] = timestamp[1]

    ctx_ptr = safe_ctx_ptr(ctx)
    config_ptr = <tiledb_config_t*>PyCapsule_GetPointer(
        config.__capsule__(), "config") if config is not None else NULL
    cdef bytes buri = unicode_path(uri)
    cdef const char* uri_ptr = PyBytes_AS_STRING(buri)

    check_error(ctx, tiledb_array_vacuum(ctx_ptr, uri_ptr, config_ptr))
