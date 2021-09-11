#!python
#cython: embedsignature=True
#cython: auto_pickle=False

from cpython.version cimport PY_MAJOR_VERSION
from cpython.pycapsule cimport PyCapsule_New, PyCapsule_IsValid, PyCapsule_GetPointer

include "common.pxi"
import io
import sys
import warnings
from collections import OrderedDict

from .array import DenseArray, SparseArray
from .ctx import default_ctx

###############################################################################
#     Numpy initialization code (critical)                                    #
###############################################################################

# https://docs.scipy.org/doc/numpy/reference/c-api.array.html#c.import_array
np.import_array()


###############################################################################
#    MODULAR IMPORTS                                                          #
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
    TILEDB_STRING_UTF8: np.NPY_UNICODE
}

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
    TILEDB_STRING_ASCII: np.bytes_,
    TILEDB_STRING_UTF8: np.dtype('U1')
}

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


def regularize_tiling(tile, ndim):
    if not tile:
        return None
    elif np.isscalar(tile):
        tiling = tuple(int(tile) for _ in range(ndim))
    elif (tile is str) or (len(tile) != ndim):
        raise ValueError("'tile' argument must be iterable "
                         "and match array dimensionality")
    else:
        tiling = tuple(tile)
    return tiling


def schema_like(*args, shape=None, dtype=None, ctx=None, **kw):
    """
    Return an ArraySchema corresponding to a NumPy-like object or
    `shape` and `dtype` kwargs. Users are encouraged to pass 'tile'
    and 'capacity' keyword arguments as appropriate for a given
    application.

    :param A: NumPy array-like object, or TileDB reference URI, optional
    :param tuple shape: array shape, optional
    :param dtype: array dtype, optional
    :param Ctx ctx: TileDB Ctx
    :param kwargs: additional keyword arguments to pass through, optional
    :return: tiledb.ArraySchema
    """
    if not ctx:
        ctx = default_ctx()
    def is_ndarray_like(obj):
        return hasattr(arr, 'shape') and hasattr(arr, 'dtype') and hasattr(arr, 'ndim')

    # support override of default dimension dtype
    dim_dtype = kw.pop('dim_dtype', np.uint64)
    if len(args) == 1:
        arr = args[0]
        if is_ndarray_like(arr):
            tiling = regularize_tiling(kw.pop('tile', None), arr.ndim)
            schema = schema_like_numpy(arr, tile=tiling, dim_dtype=dim_dtype, ctx=ctx)
        else:
            raise ValueError("expected ndarray-like object")
    elif shape and dtype:
        if np.issubdtype(np.bytes_, dtype):
            dtype = np.dtype('S')
        elif np.issubdtype(dtype, np.unicode_):
            dtype = np.dtype('U')

        ndim = len(shape)
        tiling = regularize_tiling(kw.pop('tile', None), ndim)

        dims = []
        for d in range(ndim):
            # support smaller tile extents by kw
            # domain is based on full shape
            tile_extent = tiling[d] if tiling else shape[d]
            domain = (0, shape[d] - 1)
            dims.append(Dim(domain=domain, tile=tile_extent, dtype=dim_dtype, ctx=ctx))

        att = Attr(dtype=dtype, ctx=ctx)
        dom = Domain(*dims, ctx=ctx)
        schema = ArraySchema(ctx=ctx, domain=dom, attrs=(att,), **kw)
    elif kw is not None:
        raise ValueError
    else:
        raise ValueError("Must provide either ndarray-like object or 'shape' "
                         "and 'dtype' keyword arguments")

    return schema

def schema_like_numpy(array, ctx=None, **kw):
    """create array schema from Numpy array-like object
    internal function. tiledb.schema_like is exported and recommended
    """
    if not ctx:
        ctx = default_ctx()
    # create an ArraySchema from the numpy array object
    tiling = regularize_tiling(kw.pop('tile', None), array.ndim)

    attr_name = kw.pop('attr_name', '')
    dim_dtype = kw.pop('dim_dtype', np.uint64)
    dims = []
    for (dim_num,d) in enumerate(range(array.ndim)):
        # support smaller tile extents by kw
        # domain is based on full shape
        tile_extent = tiling[d] if tiling else array.shape[d]
        domain = (0, array.shape[d] - 1)
        dims.append(Dim(domain=domain, tile=tile_extent, dtype=dim_dtype, ctx=ctx))

    var = False
    if array.dtype == object:
        # for object arrays, we use the dtype of the first element
        # consistency check should be done later, if needed
        el0 = array.flat[0]
        if type(el0) is bytes:
            el_dtype = np.dtype('S')
            var = True
        elif type(el0) is str:
            el_dtype = np.dtype('U')
            var = True
        elif type(el0) == np.ndarray:
            if len(el0.shape) != 1:
                raise TypeError("Unsupported sub-array type for Attribute: {} " \
                                "(only string arrays and 1D homogeneous NumPy arrays are supported)".
                                format(type(el0)))
            el_dtype = el0.dtype
        else:
            raise TypeError("Unsupported sub-array type for Attribute: {} " \
                            "(only strings and homogeneous-typed NumPy arrays are supported)".
                            format(type(el0)))
    else:
        el_dtype = array.dtype

    att = Attr(dtype=el_dtype, name=attr_name, var=var, ctx=ctx)
    dom = Domain(*dims, ctx=ctx)
    return ArraySchema(ctx=ctx, domain=dom, attrs=(att,), **kw)

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
                  list attributes,
                  list values,
                  dict nullmaps,
                  dict fragment_info,
                  bint issparse):

    # used for buffer conversion (local import to avoid circularity)
    import tiledb.main

    cdef bint isfortran = False
    cdef Py_ssize_t nattr = len(attributes)
    cdef Py_ssize_t nattr_alloc = nattr

    # add 1 to nattr for sparse coordinates
    if issparse:
        nattr_alloc += tiledb_array.schema.ndim

    # Set up buffers
    cdef np.ndarray buffer_sizes = np.zeros((nattr_alloc,), dtype=np.uint64)
    cdef np.ndarray buffer_offsets_sizes = np.zeros((nattr_alloc,),  dtype=np.uint64)
    cdef np.ndarray nullmaps_sizes = np.zeros((nattr_alloc,), dtype=np.uint64)
    output_values = list()
    output_offsets = list()

    for i in range(nattr):
        if tiledb_array.schema.attr(i).isvar:
            try:
                buffer, offsets = tiledb.main.array_to_buffer(values[i], True, False)
            except Exception as exc:
                raise type(exc)(f"Failed to convert buffer for attribute: '{tiledb_array.schema.attr(i).name}'") from exc
            buffer_offsets_sizes[i] = offsets.nbytes
        else:
            buffer, offsets = values[i], None

        buffer_sizes[i] = buffer.nbytes
        output_values.append(buffer)
        output_offsets.append(offsets)

    # Check value layouts
    value = output_values[0]
    isfortran = value.ndim > 1 and value.flags.f_contiguous
    if nattr > 1:
        for i in range(1, nattr):
            value = values[i]
            if value.ndim > 1 and value.flags.f_contiguous and not isfortran:
                raise ValueError("mixed C and Fortran array layouts")

    #### Allocate and fill query ####

    cdef tiledb_query_t* query_ptr = NULL
    cdef int rc = TILEDB_OK
    rc = tiledb_query_alloc(ctx_ptr, array_ptr, TILEDB_WRITE, &query_ptr)
    if rc != TILEDB_OK:
        _raise_ctx_err(ctx_ptr, rc)

    cdef tiledb_layout_t layout = TILEDB_COL_MAJOR if isfortran else TILEDB_ROW_MAJOR

    # Set coordinate buffer size and name, and layout for sparse writes
    if issparse:
        for dim_idx in range(tiledb_array.schema.ndim):
            name = tiledb_array.schema.domain.dim(dim_idx).name
            val = coords_or_subarray[dim_idx]
            if tiledb_array.schema.domain.dim(dim_idx).isvar:
                buffer, offsets = tiledb.main.array_to_buffer(val, True, False)
                buffer_sizes[nattr + dim_idx] = buffer.nbytes
                buffer_offsets_sizes[nattr + dim_idx] = offsets.nbytes
            else:
                buffer, offsets = val, None
                buffer_sizes[nattr + dim_idx] = buffer.nbytes

            attributes.append(name)
            output_values.append(buffer)
            output_offsets.append(offsets)
        nattr += tiledb_array.schema.ndim
        layout = TILEDB_UNORDERED

    # Create nullmaps sizes array if necessary

    # Set layout
    rc = tiledb_query_set_layout(ctx_ptr, query_ptr, layout)
    if rc != TILEDB_OK:
        tiledb_query_free(&query_ptr)
        _raise_ctx_err(ctx_ptr, rc)

    cdef void* buffer_ptr = NULL
    cdef uint8_t* nulmap_buffer_ptr = NULL
    cdef uint
    cdef bytes battr_name
    cdef uint64_t* offsets_buffer_ptr = NULL
    cdef uint64_t* buffer_sizes_ptr = <uint64_t*> np.PyArray_DATA(buffer_sizes)
    cdef uint64_t* offsets_buffer_sizes_ptr = <uint64_t*> np.PyArray_DATA(buffer_offsets_sizes)
    cdef uint64_t* nullmaps_sizes_ptr = <uint64_t*> np.PyArray_DATA(nullmaps_sizes)

    # set subarray (ranges)
    cdef np.ndarray s_start
    cdef np.ndarray s_end
    cdef void* s_start_ptr = NULL
    cdef void* s_end_ptr = NULL
    cdef Domain dom = None
    cdef Dim dim = None
    cdef np.dtype dim_dtype = None
    if not issparse:
        dom = tiledb_array.schema.domain
        for dim_idx,s_range in enumerate(coords_or_subarray):
            dim = dom.dim(dim_idx)
            dim_dtype = dim.dtype
            s_start = np.asarray(s_range[0], dtype=dim_dtype)
            s_end = np.asarray(s_range[1], dtype=dim_dtype)
            s_start_ptr = np.PyArray_DATA(s_start)
            s_end_ptr = np.PyArray_DATA(s_end)
            if dim.isvar:
                rc = tiledb_query_add_range_var(
                    ctx_ptr, query_ptr, dim_idx,
                    s_start_ptr,  s_start.nbytes,
                    s_end_ptr, s_end.nbytes)

            else:
                rc = tiledb_query_add_range(
                    ctx_ptr, query_ptr, dim_idx,
                    s_start_ptr, s_end_ptr, NULL)

            if rc != TILEDB_OK:
                tiledb_query_free(&query_ptr)
                _raise_ctx_err(ctx_ptr, rc)

    try:
        for i in range(0, nattr):
            battr_name = attributes[i].encode('UTF-8')
            buffer_ptr = np.PyArray_DATA(output_values[i])

            if output_offsets[i] is not None:
                # VAR_NUM attribute
                offsets_buffer_ptr = <uint64_t*>np.PyArray_DATA(output_offsets[i])
                rc = tiledb_query_set_buffer_var(ctx_ptr, query_ptr, battr_name,
                                                 offsets_buffer_ptr, &(offsets_buffer_sizes_ptr[i]),
                                                 buffer_ptr, &(buffer_sizes_ptr[i]))
            elif attributes[i] in nullmaps:
                # NOTE: validity map is owned *by the caller*
                nulmap = nullmaps[attributes[i]]
                nullmaps_sizes[i] = len(nulmap)
                nulmap_buffer_ptr = <uint8_t*>np.PyArray_DATA(nulmap)
                rc = tiledb_query_set_buffer_nullable(
                    ctx_ptr,
                    query_ptr,
                    battr_name,
                    buffer_ptr,
                    &(buffer_sizes_ptr[i]),
                    nulmap_buffer_ptr,
                    &(nullmaps_sizes_ptr[i])
                )
            else:
                rc = tiledb_query_set_buffer(ctx_ptr, query_ptr, battr_name,
                                             buffer_ptr, &(buffer_sizes_ptr[i]))
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


cdef class TileDBError(Exception):
    """TileDB Error Exception

    Captures and raises error return code (``TILEDB_ERR``) messages when calling ``libtiledb``
    functions.  The error message that is raised is the last error set for the :py:class:`tiledb.Ctx`

    A Python :py:class:`MemoryError` is raised on ``TILEDB_OOM``

    """

    @property
    def message(self):
        """The TileDB error message string

        :rtype: str
        :return: error message

        """
        return self.args[0]

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


cpdef check_error(Ctx ctx, int rc):
    _raise_ctx_err(ctx.ptr, rc)

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
        import json
        stats_json_core = json.loads(stats_str_core)

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
        stats_str += "- Number of read queries: {}\n".format(
            stats_json_core["READ_NUM"])
        stats_str += "- Number of attributes read: {}\n".format(
            stats_json_core["READ_ATTR_FIXED_NUM"]
            + stats_json_core["READ_ATTR_VAR_NUM"])
        stats_str += "- Time to compute estimated result size: {}\n".format(
            stats_json_core["READ_COMPUTE_EST_RESULT_SIZE"])
        stats_str += "- Read time: {}\n".format(stats_json_core["READ"])
        stats_str += "- Total read query time (array open + init state + read): {}\n".format(
            stats_json_core["READ"] + stats_json_core["READ_INIT_STATE"])
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

def dtype_range(np.dtype dtype):
    """Return the range of a Numpy dtype"""

    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        dtype_min, dtype_max = info.min, info.max
    elif np.issubdtype(dtype, np.floating):
        info = np.finfo(dtype)
        dtype_min, dtype_max = info.min, info.max
    elif dtype.kind == 'M':
        info = np.iinfo(np.int64)
        date_unit = np.datetime_data(dtype)[0]
        # +1 to exclude NaT
        dtype_min = np.datetime64(info.min + 1, date_unit)
        dtype_max = np.datetime64(info.max, date_unit)
    else:
        raise TypeError("invalid Dim dtype {0!r}".format(dtype))
    return (dtype_min, dtype_max)

###############################################################################
#                                                                             #
#    CLASS DEFINITIONS                                                        #
#                                                                             #
###############################################################################

cdef class Config(object):
    """TileDB Config class

    The Config object stores configuration parameters for both TileDB Embedded
    and TileDB-Py.

    For TileDB Embedded parameters, see:

        https://docs.tiledb.com/main/solutions/tiledb-embedded/configuration-parameters

    The following configuration options are supported by TileDB-Py:

        - `py.init_buffer_bytes`:

           Initial allocation size in bytes for attribute and dimensions buffers.
           If result size exceed the pre-allocated buffer(s), then the query will return
           incomplete and TileDB-Py will allocate larger buffers and resubmit.
           Specifying a sufficiently large buffer size will often improve performance.
           Default 10 MB (1024**2 * 10).

        - `py.use_arrow`:

           Use `pyarrow` from the Apache Arrow project to convert
           query results into Pandas dataframe format when requested.
           Default `True`.

        - `py.deduplicate`:

           Attempt to deduplicate Python objects during buffer
           conversion to Python. Deduplication may reduce memory usage for datasets
           with many identical strings, at the cost of some performance reduction
           due to hash calculation/lookup for each object.

    Unknown parameters will be ignored!

    :param dict params: Set parameter values from dict like object
    :param str path: Set parameter values from persisted Config parameter file
    """


    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            tiledb_config_free(&self.ptr)

    def __init__(self, params=None, path=None):
        cdef tiledb_config_t* config_ptr = NULL
        cdef tiledb_error_t* err_ptr = NULL
        cdef int rc = tiledb_config_alloc(&config_ptr, &err_ptr)
        if rc == TILEDB_OOM:
            raise MemoryError()
        elif rc == TILEDB_ERR:
            _raise_tiledb_error(err_ptr)
        assert(config_ptr != NULL)
        self.ptr = config_ptr
        if path is not None:
            self.load(path)
        if params is not None:
            self.update(params)

    @staticmethod
    cdef from_ptr(tiledb_config_t* ptr):
        """Constructs a Config class instance from a (non-null) tiledb_config_t pointer"""
        assert(ptr != NULL)
        cdef Config config = Config.__new__(Config)
        config.ptr = ptr
        return config

    @staticmethod
    def load(object uri):
        """Constructs a Config class instance from config parameters loaded from a local Config file

        :parameter str uri: a local URI config file path
        :rtype: tiledb.Config
        :return: A TileDB Config instance with persisted parameter values
        :raises TypeError: `uri` cannot be converted to a unicode string
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef bytes buri = unicode_path(uri)
        cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
        cdef Config config = Config.__new__(Config)
        cdef tiledb_config_t* config_ptr = NULL
        cdef tiledb_error_t* err_ptr = NULL
        cdef int rc = tiledb_config_alloc(&config_ptr, &err_ptr)
        if rc == TILEDB_OOM:
            raise MemoryError()
        if rc == TILEDB_ERR:
            _raise_tiledb_error(err_ptr)
        with nogil:
            rc = tiledb_config_load_from_file(config_ptr, uri_ptr, &err_ptr)
        if rc == TILEDB_OOM:
            tiledb_config_free(&config_ptr)
            raise MemoryError()
        if rc == TILEDB_ERR:
            tiledb_config_free(&config_ptr)
            _raise_tiledb_error(err_ptr)
        assert(config_ptr != NULL)
        config.ptr = config_ptr
        return config

    def __setitem__(self, object key, object value):
        """Sets a config parameter value.

        :param str key: Name of parameter to set
        :param str value: Value of parameter to set
        :raises TypeError: `key` or `value` cannot be encoded into a UTF-8 string
        :raises: :py:exc:`tiledb.TileDBError`

        """
        key, value = unicode(key), unicode(value)
        cdef bytes bparam = key.encode('UTF-8')
        cdef bytes bvalue = value.encode('UTF-8')
        cdef tiledb_error_t* err_ptr = NULL
        cdef int rc = tiledb_config_set(self.ptr, bparam, bvalue, &err_ptr)
        if rc == TILEDB_OOM:
            raise MemoryError()
        elif rc == TILEDB_ERR:
            _raise_tiledb_error(err_ptr)
        return

    def get(self, object key, raise_keyerror = True):
        key = unicode(key)
        cdef bytes bparam = key.encode('UTF-8')
        cdef const char* value_ptr = NULL
        cdef tiledb_error_t* err_ptr = NULL
        cdef int rc = tiledb_config_get(self.ptr, bparam, &value_ptr, &err_ptr)
        if rc == TILEDB_OOM:
            raise MemoryError()
        elif rc == TILEDB_ERR:
            _raise_tiledb_error(err_ptr)
        if value_ptr == NULL:
            if raise_keyerror:
                raise KeyError(key)
            else:
                return None
        cdef bytes value = PyBytes_FromString(value_ptr)
        return value.decode('UTF-8')

    def __getitem__(self, object key):
        """Gets a config parameter value.

        :param str key: Name of parameter to get
        :return: Config parameter value string
        :rtype str:
        :raises TypeError: `key` cannot be encoded into a UTF-8 string
        :raises KeyError: Config parameter not found
        :raises: :py:exc:`tiledb.TileDBError`

        """
        return self.get(key, True)

    def __delitem__(self, object key):
        """
        Removes a configured parameter (resetting it to its default).

        :param str key: Name of parameter to reset.
        :raises TypeError: `key` cannot be encoded into a UTF-8 string

        """
        key = unicode(key)
        cdef bytes bkey = ustring(key).encode("UTF-8")
        cdef tiledb_error_t* err_ptr = NULL
        cdef int rc = tiledb_config_unset(self.ptr, bkey, &err_ptr)
        if rc == TILEDB_OOM:
            raise MemoryError()
        elif rc == TILEDB_ERR:
            _raise_tiledb_error(err_ptr)
        return

    def __iter__(self):
        """Returns an iterator over the Config parameters (keys)"""
        return ConfigKeys(self)

    def __len__(self):
        """Returns the number of parameters (keys) held by the Config object"""
        return sum(1 for _ in self)

    def __eq__(self, object config):
        if not isinstance(config, Config):
            return False
        keys = set(self.keys())
        okeys = set(config.keys())
        if keys != okeys:
            return False
        for k in keys:
            val, oval = self[k], config[k]
            if val != oval:
                return False
        return True

    def __repr__(self):
        colnames = ["Parameter", "Value"]
        params = list(self.keys())
        values = list(map(repr, self.values()))
        colsizes = [max(len(colnames[0]), *map(len, (p for p in params))),
                    max(len(colnames[1]), *map(len, (v for v in values)))]
        format_str = ' | '.join("{{:<{}}}".format(i) for i in colsizes)
        output = []
        output.append(format_str.format(colnames[0], colnames[1]))
        output.append(format_str.format('-' * colsizes[0], '-' * colsizes[1]))
        output.extend(format_str.format(p, v) for p, v in zip(params, values))
        return "\n".join(output)

    def items(self, prefix=u""):
        """Returns an iterator object over Config parameters, values

        :param str prefix: return only parameters with a given prefix
        :rtype: ConfigItems
        :returns: iterator over Config parameter, value tuples

        """
        return ConfigItems(self, prefix=prefix)

    def keys(self, prefix=u""):
        """Returns an iterator object over Config parameters (keys)

        :param str prefix: return only parameters with a given prefix
        :rtype: ConfigKeys
        :returns: iterator over Config parameter string keys

        """
        return ConfigKeys(self, prefix=prefix)

    def values(self, prefix=u""):
        """Returns an iterator object over Config values

        :param str prefix: return only parameters with a given prefix
        :rtype: ConfigValues
        :returns: iterator over Config string values

        """
        return ConfigValues(self, prefix=prefix)

    def dict(self, prefix=u""):
        """Returns a dict representation of a Config object

        :param str prefix: return only parameters with a given prefix
        :rtype: dict
        :return: Config parameter / values as a a Python dict

        """
        return dict(ConfigItems(self, prefix=prefix))

    def clear(self):
        """Unsets all Config parameters (returns them to their default values)"""
        for k in self.keys():
            del self[k]

    def get(self, key, *args):
        """Gets the value of a config parameter, or a default value.

        :param str key: Config parameter
        :param args: return `arg` if Config does not contain parameter `key`
        :return: Parameter value, `arg` or None.

        """
        nargs = len(args)
        if nargs > 1:
            raise TypeError("get expected at most 2 arguments, got {}".format(nargs))
        try:
            return self[key]
        except KeyError:
            return args[0] if nargs == 1 else None

    def update(self, object odict):
        """Update a config object with parameter, values from a dict like object

        :param odict: dict-like object containing parameter, values to update Config.

        """
        for (key, value) in odict.items():
            self[key] = value
        return

    def from_file(self, path):
        """Update a Config object with from a persisted config file

        :param path: A local Config file path

        """
        config = Config.load(path)
        self.update(config)

    def save(self, uri):
        """Persist Config parameter values to a config file

        :parameter str uri: a local URI config file path
        :raises TypeError: `uri` cannot be converted to a unicode string
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef bytes buri = unicode_path(uri)
        cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
        cdef tiledb_config_t* config_ptr = self.ptr
        cdef tiledb_error_t* err_ptr = NULL
        cdef int rc
        with nogil:
            rc = tiledb_config_save_to_file(config_ptr, uri_ptr, &err_ptr)
        if rc == TILEDB_OOM:
            raise MemoryError()
        elif rc == TILEDB_ERR:
            _raise_tiledb_error(err_ptr)
        return


cdef class ConfigKeys(object):
    """
    An iterator object over Config parameter strings (keys)
    """

    def __init__(self, Config config, prefix=u""):
        self.config_items = ConfigItems(config, prefix=prefix)

    def __iter__(self):
        return self

    def __next__(self):
        (k, _) = self.config_items.__next__()
        return k


cdef class ConfigValues(object):
    """
    An iterator object over Config parameter value strings
    """

    def __init__(self, Config config, prefix=u""):
        self.config_items = ConfigItems(config, prefix=prefix)

    def __iter__(self):
        return self

    def __next__(self):
        (_, v) = self.config_items.__next__()
        return v


cdef class ConfigItems(object):
    """
    An iterator object over Config parameter, values

    :param config: TileDB Config object
    :type config: tiledb.Config
    :param prefix: (default "") Filter paramter names with given prefix
    :type prefix: str

    """

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            tiledb_config_iter_free(&self.ptr)

    def __init__(self, Config config, prefix=u""):
        cdef bytes bprefix = prefix.encode("UTF-8")
        cdef const char* prefix_ptr = PyBytes_AS_STRING(bprefix)
        cdef tiledb_config_iter_t* config_iter_ptr = NULL
        cdef tiledb_error_t* err_ptr = NULL
        cdef rc = tiledb_config_iter_alloc(
            config.ptr, prefix_ptr, &config_iter_ptr, &err_ptr)
        if rc == TILEDB_OOM:
            raise MemoryError()
        elif rc == TILEDB_ERR:
            _raise_tiledb_error(err_ptr)
        assert (config_iter_ptr != NULL)
        self.config = config
        self.ptr = config_iter_ptr

    def __iter__(self):
        return self

    def __next__(self):
        cdef int done = 0
        cdef tiledb_error_t* err_ptr = NULL
        cdef int rc = tiledb_config_iter_done(self.ptr, &done, &err_ptr)
        if rc == TILEDB_OOM:
            raise MemoryError()
        elif rc == TILEDB_ERR:
            _raise_tiledb_error(err_ptr)
        if done > 0:
            raise StopIteration()
        cdef const char* param_ptr = NULL
        cdef const char* value_ptr = NULL
        rc = tiledb_config_iter_here(self.ptr, &param_ptr, &value_ptr, &err_ptr)
        if rc == TILEDB_OOM:
            raise MemoryError()
        elif rc == TILEDB_ERR:
            _raise_tiledb_error(err_ptr)
        cdef bytes bparam
        cdef bytes bvalue
        if param_ptr == NULL:
            bparam = b''
        else:
            bparam = PyBytes_FromString(param_ptr)
        if value_ptr == NULL:
            bvalue = b''
        else:
            bvalue = PyBytes_FromString(value_ptr)
        rc = tiledb_config_iter_next(self.ptr, &err_ptr)
        if rc == TILEDB_OOM:
            raise MemoryError()
        elif rc == TILEDB_ERR:
            _raise_tiledb_error(err_ptr)
        return (bparam.decode('UTF-8'), bvalue.decode('UTF-8'))


cdef class Ctx(object):
    """Class representing a TileDB context.

    A TileDB context wraps a TileDB storage manager.

    :param config: Initialize Ctx with given config parameters
    :type config: tiledb.Config or dict

    """

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            tiledb_ctx_free(&self.ptr)

    def __capsule__(self):
        if self.ptr == NULL:
            raise TileDBError("internal error: cannot create capsule for uninitialized Ctx!")
        cdef const char* name = "ctx"
        cap = PyCapsule_New(<void *>(self.ptr), name, NULL)
        return cap

    def __init__(self, config=None):
        cdef Config _config = Config()
        if config is not None:
            if isinstance(config, Config):
                _config = config
            else:
                _config.update(config)
        cdef tiledb_ctx_t* ctx_ptr = NULL
        cdef int rc = tiledb_ctx_alloc(_config.ptr, &ctx_ptr)
        if rc == TILEDB_OOM:
            raise MemoryError()
        elif rc == TILEDB_ERR:
            # we assume that the ctx pointer is valid if not OOM
            # the ctx object will be free'd when it goes out of scope
            # after the exception is raised
            _raise_ctx_err(ctx_ptr, rc)
        self.ptr = ctx_ptr
        self._set_default_tags()

    def __repr__(self):
        return "tiledb.Ctx() [see Ctx.config() for configuration]"

    def config(self):
        """Returns the Config instance associated with the Ctx."""
        cdef tiledb_config_t* config_ptr = NULL
        check_error(self,
                    tiledb_ctx_get_config(self.ptr, &config_ptr))
        return Config.from_ptr(config_ptr)

    def set_tag(self, key, value):
        """Sets a (string, string) "tag" on the Ctx (internal)."""
        cdef tiledb_ctx_t* ctx_ptr = self.ptr
        bkey = key.encode('UTF-8')
        bvalue = value.encode('UTF-8')
        cdef int rc = TILEDB_OK
        rc = tiledb_ctx_set_tag(ctx_ptr, bkey, bvalue)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

    def _set_default_tags(self):
        """Sets all default tags on the Ctx"""
        self.set_tag('x-tiledb-api-language', 'python')
        self.set_tag('x-tiledb-api-language-version', '{}.{}.{}'.format(*sys.version_info))
        self.set_tag('x-tiledb-api-sys-platform', sys.platform)


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

def _tiledb_cast_tile_extent(tile_extent, dtype):
    """Given a tile extent value, cast it to np.array of the given numpy dtype."""
    # Special handling for datetime domains
    if dtype.kind == 'M':
        date_unit = np.datetime_data(dtype)[0]
        if isinstance(tile_extent, np.timedelta64):
            extent_value = int(tile_extent / np.timedelta64(1, date_unit))
            tile_size_array = np.array(np.int64(extent_value), dtype=np.int64)
        else:
            tile_size_array = np.array(tile_extent, dtype=dtype)
    else:
        tile_size_array = np.array(tile_extent, dtype=dtype)

    if tile_size_array.size != 1:
        raise ValueError("tile extent must be a scalar")
    return tile_size_array


cdef int _numpy_typeid(tiledb_datatype_t tiledb_dtype):
    """Return a numpy type num (int) given a tiledb_datatype_t enum value."""
    np_id_type = _tiledb_dtype_to_numpy_typeid_convert.get(tiledb_dtype, None)
    if np_id_type:
        return np_id_type
    return np.NPY_DATETIME if _tiledb_type_is_datetime(tiledb_dtype) else np.NPY_NOTYPE

cdef _numpy_dtype(tiledb_datatype_t tiledb_dtype, cell_size = 1):
    """Return a numpy type given a tiledb_datatype_t enum value."""
    cdef base_dtype
    cdef uint32_t cell_val_num = cell_size

    if cell_val_num == 1:
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

cdef class Filter(object):
    """Base class for all TileDB filters."""

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            tiledb_filter_free(&self.ptr)

    def __init__(self, tiledb_filter_type_t filter_type, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        cdef tiledb_ctx_t* ctx_ptr = ctx.ptr
        cdef tiledb_filter_t* filter_ptr = NULL
        cdef int rc = TILEDB_OK
        rc = tiledb_filter_alloc(ctx_ptr, filter_type, &filter_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        self.ctx = ctx
        self.ptr = filter_ptr
        return

    def __repr__(self):
        output = io.StringIO()
        output.write(f"{type(self).__name__}(")
        if hasattr(self, '_attrs_'):
            for f in self._attrs_():
                a = getattr(self, f)
                output.write(f"{f}={a}")
        output.write(")")
        return output.getvalue()

    def __eq__(self, other):
        if type(self) != type(other):
            return False
        for f in self._attrs_():
           left = getattr(self, f)
           right = getattr(other, f)
           if left != right:
               return False
        return True

cdef class CompressionFilter(Filter):
    """
    Base class for filters performing compression.

    All compression filters support a compression level option, although
    some (such as RLE) ignore it.

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList([tiledb.GzipFilter(level=10)]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    def __init__(self, tiledb_filter_type_t filter_type, level, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        super().__init__(filter_type, ctx)
        if level is None:
            return
        cdef tiledb_ctx_t* ctx_ptr = ctx.ptr
        cdef int clevel = int(level)
        cdef int rc = TILEDB_OK
        rc = tiledb_filter_set_option(ctx_ptr, self.ptr, TILEDB_COMPRESSION_LEVEL, &clevel)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

    @property
    def level(self):
        """The compression level setting for the filter.

        Every compressor interprets this value differently (some ignore it, such as RLE).

        :return: compression level
        :rtype: int

        """
        cdef int32_t rc = TILEDB_OK
        cdef tiledb_filter_option_t option = TILEDB_COMPRESSION_LEVEL
        cdef int32_t level = -1

        rc = tiledb_filter_get_option(self.ctx.ptr, self.ptr, option, &level)
        if rc != TILEDB_OK:
            _raise_ctx_err(self.ctx.ptr, rc)
        return level


cdef class NoOpFilter(Filter):
    """A filter that does nothing."""

    @staticmethod
    cdef from_ptr(const tiledb_filter_t* filter_ptr, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        assert(filter_ptr != NULL)
        cdef NoOpFilter filter_obj = NoOpFilter.__new__(NoOpFilter)
        filter_obj.ctx = ctx
        # need to cast away the const
        filter_obj.ptr = <tiledb_filter_t*> filter_ptr
        return filter_obj

    def __init__(self, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        super().__init__(TILEDB_FILTER_NONE, ctx=ctx)

    def _attrs_(self):
        return {}

cdef class GzipFilter(CompressionFilter):
    """
    Filter that compresses using gzip.

    :param ctx: TileDB Ctx
    :type ctx: tiledb.Ctx
    :param level: (default None) If not None set the compressor level
    :type level: int

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList([tiledb.GzipFilter()]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    @staticmethod
    cdef from_ptr(const tiledb_filter_t* filter_ptr, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        assert(filter_ptr != NULL)
        cdef GzipFilter filter_obj = GzipFilter.__new__(GzipFilter)
        filter_obj.ctx = ctx
        # need to cast away the const
        filter_obj.ptr = <tiledb_filter_t*> filter_ptr
        return filter_obj

    def __init__(self, level=None, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        super().__init__(TILEDB_FILTER_GZIP, level, ctx=ctx)

    def _attrs_(self):
        return {'level': self.level}

cdef class ZstdFilter(CompressionFilter):
    """
    Filter that compresses using zstd.

    :param ctx: TileDB Ctx
    :type ctx: tiledb.Ctx
    :param level: (default None) If not None set the compressor level
    :type level: int

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList([tiledb.ZstdFilter()]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    @staticmethod
    cdef from_ptr(const tiledb_filter_t* filter_ptr, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        assert(filter_ptr != NULL)
        cdef ZstdFilter filter_obj = ZstdFilter.__new__(ZstdFilter)
        filter_obj.ctx = ctx
        # need to cast away the const
        filter_obj.ptr = <tiledb_filter_t*> filter_ptr
        return filter_obj

    def __init__(self, level=None, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        super().__init__(TILEDB_FILTER_ZSTD, level, ctx=ctx)

    def _attrs_(self):
        return {'level': self.level}

cdef class LZ4Filter(CompressionFilter):
    """
    Filter that compresses using lz4.

    :param ctx: TileDB Ctx
    :type ctx: tiledb.Ctx
    :param level: (default None) If not None set the compressor level
    :type level: int

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList([tiledb.LZ4Filter()]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    @staticmethod
    cdef from_ptr(const tiledb_filter_t* filter_ptr, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        assert(filter_ptr != NULL)
        cdef LZ4Filter filter_obj = LZ4Filter.__new__(LZ4Filter)
        filter_obj.ctx = ctx
        # need to cast away the const
        filter_obj.ptr = <tiledb_filter_t*> filter_ptr
        return filter_obj

    def __init__(self, level=None, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        super().__init__(TILEDB_FILTER_LZ4, level, ctx)

    def _attrs_(self):
        return {'level': self.level}

cdef class Bzip2Filter(CompressionFilter):
    """
    Filter that compresses using bzip2.

    :param level: (default None) If not None set the compressor level
    :type level: int

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList([tiledb.Bzip2Filter()]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    @staticmethod
    cdef from_ptr(const tiledb_filter_t* filter_ptr, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        assert(filter_ptr != NULL)
        cdef Bzip2Filter filter_obj = Bzip2Filter.__new__(Bzip2Filter)
        filter_obj.ctx = ctx
        # need to cast away the const
        filter_obj.ptr = <tiledb_filter_t*> filter_ptr
        return filter_obj

    def __init__(self, level=None, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        super().__init__(TILEDB_FILTER_BZIP2, level, ctx=ctx)

    def _attrs_(self):
        return {'level': self.level}

cdef class RleFilter(CompressionFilter):
    """
    Filter that compresses using run-length encoding (RLE).

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList([tiledb.RleFilter()]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    @staticmethod
    cdef from_ptr(const tiledb_filter_t* filter_ptr, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        assert(filter_ptr != NULL)
        cdef RleFilter filter_obj = RleFilter.__new__(RleFilter)
        filter_obj.ctx = ctx
        # need to cast away the const
        filter_obj.ptr = <tiledb_filter_t*> filter_ptr
        return filter_obj

    def __init__(self, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        super().__init__(TILEDB_FILTER_RLE, None, ctx=ctx)

    def _attrs_(self):
        return {}

cdef class DoubleDeltaFilter(CompressionFilter):
    """
    Filter that performs double-delta encoding.

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList([tiledb.DoubleDeltaFilter()]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    @staticmethod
    cdef from_ptr(const tiledb_filter_t* filter_ptr, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        assert(filter_ptr != NULL)
        cdef DoubleDeltaFilter filter_obj = DoubleDeltaFilter.__new__(DoubleDeltaFilter)
        filter_obj.ctx = ctx
        # need to cast away the const
        filter_obj.ptr = <tiledb_filter_t*> filter_ptr
        return filter_obj

    def __init__(self, Ctx ctx=None):
        if not ctx:
            ctx = None
        super().__init__(TILEDB_FILTER_DOUBLE_DELTA, None, ctx)

    def _attrs_(self):
        return {}

cdef class BitShuffleFilter(Filter):
    """
    Filter that performs a bit shuffle transformation.

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList([tiledb.BitShuffleFilter()]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    @staticmethod
    cdef from_ptr(const tiledb_filter_t* filter_ptr, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        assert(filter_ptr != NULL)
        cdef BitShuffleFilter filter_obj = BitShuffleFilter.__new__(BitShuffleFilter)
        filter_obj.ctx = ctx
        # need to cast away the const
        filter_obj.ptr = <tiledb_filter_t*> filter_ptr
        return filter_obj

    def __init__(self, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        super().__init__(TILEDB_FILTER_BITSHUFFLE, ctx=ctx)

    def _attrs_(self):
        return {}

cdef class ByteShuffleFilter(Filter):
    """
    Filter that performs a byte shuffle transformation.

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList([tiledb.ByteShuffleFilter()]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    @staticmethod
    cdef from_ptr(const tiledb_filter_t* filter_ptr, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        assert(filter_ptr != NULL)
        cdef ByteShuffleFilter filter_obj = ByteShuffleFilter.__new__(ByteShuffleFilter)
        filter_obj.ctx = ctx
        # need to cast away the const
        filter_obj.ptr = <tiledb_filter_t*> filter_ptr
        return filter_obj

    def __init__(self, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        super().__init__(TILEDB_FILTER_BYTESHUFFLE, ctx=ctx)

    def _attrs_(self):
        return {}

cdef class BitWidthReductionFilter(Filter):
    """Filter that performs bit-width reduction.

     :param ctx: A TileDB Context
     :type ctx: tiledb.Ctx
     :param window: (default None) max window size for the filter
     :type window: int

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList([tiledb.BitWidthReductionFilter()]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    @staticmethod
    cdef from_ptr(const tiledb_filter_t* filter_ptr, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        assert(filter_ptr != NULL)
        cdef BitWidthReductionFilter filter_obj = BitWidthReductionFilter.__new__(BitWidthReductionFilter)
        filter_obj.ctx = ctx
        # need to cast away the const
        filter_obj.ptr = <tiledb_filter_t*> filter_ptr
        return filter_obj

    def __init__(self, window=None, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        super().__init__(TILEDB_FILTER_BIT_WIDTH_REDUCTION, ctx)
        if window is None:
            return
        cdef tiledb_ctx_t* ctx_ptr = ctx.ptr
        cdef unsigned int cwindow = window
        cdef int rc = TILEDB_OK
        rc = tiledb_filter_set_option(ctx_ptr, self.ptr, TILEDB_BIT_WIDTH_MAX_WINDOW, &cwindow)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

    def _attrs_(self):
        return {'window': self.window}

    @property
    def window(self):
        """
        :return: The maximum window size used for the filter
        :rtype: int

        """
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_filter_t* filter_ptr = self.ptr
        cdef unsigned int cwindow = 0
        cdef int rc = TILEDB_OK
        rc = tiledb_filter_get_option(ctx_ptr, filter_ptr, TILEDB_BIT_WIDTH_MAX_WINDOW, &cwindow)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return int(cwindow)


cdef class PositiveDeltaFilter(Filter):
    """
    Filter that performs positive-delta encoding.

    :param ctx: A TileDB Context
    :type ctx: tiledb.Ctx
    :param window: (default None) the max window for the filter
    :type window: int

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList([tiledb.PositiveDeltaFilter()]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    @staticmethod
    cdef from_ptr(const tiledb_filter_t* filter_ptr, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        assert(filter_ptr != NULL)
        cdef PositiveDeltaFilter filter_obj = PositiveDeltaFilter.__new__(PositiveDeltaFilter)
        filter_obj.ctx = ctx
        # need to cast away the const
        filter_obj.ptr = <tiledb_filter_t*> filter_ptr
        return filter_obj

    def __init__(self, window=None, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        super().__init__(TILEDB_FILTER_POSITIVE_DELTA, ctx=ctx)
        if window is None:
            return
        cdef tiledb_ctx_t* ctx_ptr = ctx.ptr
        cdef unsigned int cwindow = window
        cdef int rc = TILEDB_OK
        rc = tiledb_filter_set_option(ctx_ptr, self.ptr, TILEDB_POSITIVE_DELTA_MAX_WINDOW, &cwindow)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

    def _attrs_(self):
        return {'window': self.window}

    @property
    def window(self):
        """
        :return: The maximum window size used for the filter
        :rtype: int

        """
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_filter_t* filter_ptr = self.ptr
        cdef unsigned int cwindow = 0
        cdef int rc = TILEDB_OK
        rc = tiledb_filter_get_option(ctx_ptr, filter_ptr, TILEDB_POSITIVE_DELTA_MAX_WINDOW, &cwindow)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return int(cwindow)

cdef class ChecksumMD5Filter(Filter):
    """
    MD5 checksum filter.

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList([tiledb.ChecksumMD5Filter()]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    @staticmethod
    cdef from_ptr(const tiledb_filter_t* filter_ptr, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        assert(filter_ptr != NULL)
        cdef ChecksumMD5Filter filter_obj = ChecksumMD5Filter.__new__(ChecksumMD5Filter)
        filter_obj.ctx = ctx
        # need to cast away the const
        filter_obj.ptr = <tiledb_filter_t*> filter_ptr
        return filter_obj

    def __init__(self, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        super().__init__(TILEDB_FILTER_CHECKSUM_MD5, ctx=ctx)

    def _attrs_(self):
        return {}

cdef class ChecksumSHA256Filter(Filter):
    """
    SHA256 checksum filter.

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList([tiledb.ChecksumSHA256Filter()]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    @staticmethod
    cdef from_ptr(const tiledb_filter_t* filter_ptr, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        assert(filter_ptr != NULL)
        cdef ChecksumSHA256Filter filter_obj = ChecksumSHA256Filter.__new__(ChecksumSHA256Filter)
        filter_obj.ctx = ctx
        # need to cast away the const
        filter_obj.ptr = <tiledb_filter_t*> filter_ptr
        return filter_obj

    def __init__(self, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        super().__init__(TILEDB_FILTER_CHECKSUM_SHA256, ctx=ctx)

    def _attrs_(self):
        return {}

cdef Filter _filter_type_ptr_to_filter(Ctx ctx, tiledb_filter_type_t filter_type,
                                       tiledb_filter_t* filter_ptr):
    """
    Return a filter instance for the given type.
    """
    if filter_type == TILEDB_FILTER_NONE:
        return NoOpFilter.from_ptr(filter_ptr, ctx=ctx)
    elif filter_type == TILEDB_FILTER_GZIP:
       return GzipFilter.from_ptr(filter_ptr, ctx=ctx)
    elif filter_type == TILEDB_FILTER_ZSTD:
        return ZstdFilter.from_ptr(filter_ptr, ctx=ctx)
    elif filter_type == TILEDB_FILTER_LZ4:
        return LZ4Filter.from_ptr(filter_ptr, ctx=ctx)
    elif filter_type == TILEDB_FILTER_RLE:
        return RleFilter.from_ptr(filter_ptr, ctx=ctx)
    elif filter_type == TILEDB_FILTER_BZIP2:
        return Bzip2Filter.from_ptr(filter_ptr, ctx=ctx)
    elif filter_type == TILEDB_FILTER_DOUBLE_DELTA:
        return DoubleDeltaFilter.from_ptr(filter_ptr, ctx=ctx)
    elif filter_type == TILEDB_FILTER_BIT_WIDTH_REDUCTION:
        return BitWidthReductionFilter.from_ptr(filter_ptr, ctx=ctx)
    elif filter_type == TILEDB_FILTER_BITSHUFFLE:
        return BitShuffleFilter.from_ptr(filter_ptr, ctx=ctx)
    elif filter_type == TILEDB_FILTER_BYTESHUFFLE:
        return ByteShuffleFilter.from_ptr(filter_ptr, ctx=ctx)
    elif filter_type == TILEDB_FILTER_POSITIVE_DELTA:
        return PositiveDeltaFilter.from_ptr(filter_ptr, ctx=ctx)
    elif filter_type == TILEDB_FILTER_CHECKSUM_MD5:
        return ChecksumSHA256Filter.from_ptr(filter_ptr, ctx=ctx)
    elif filter_type == TILEDB_FILTER_CHECKSUM_SHA256:
        return ChecksumMD5Filter.from_ptr(filter_ptr, ctx=ctx)
    else:
        raise ValueError("unknown filter type tag: {:s}".format(filter_type))


cdef class FilterList(object):
    """
    An ordered list of Filter objects for filtering TileDB data.

    FilterLists contain zero or more Filters, used for filtering attribute data, the array coordinate data, etc.

    :param ctx: A TileDB context
    :type ctx: tiledb.Ctx
    :param filters: An iterable of Filter objects to add.
    :param chunksize: (default None) chunk size used by the filter list in bytes
    :type chunksize: int

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(tiledb.Dim(domain=(0, 9), tile=2, dtype=np.uint64))
    ...     # Create several filters
    ...     gzip_filter = tiledb.GzipFilter()
    ...     bw_filter = tiledb.BitWidthReductionFilter()
    ...     # Create a filter list that will first perform bit width reduction, then gzip compression.
    ...     filters = tiledb.FilterList([bw_filter, gzip_filter])
    ...     a1 = tiledb.Attr(name="a1", dtype=np.int64, filters=filters)
    ...     # Create a second attribute filtered only by gzip compression.
    ...     a2 = tiledb.Attr(name="a2", dtype=np.int64,
    ...                      filters=tiledb.FilterList([gzip_filter]))
    ...     schema = tiledb.ArraySchema(domain=dom, attrs=(a1, a2))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """
    def __init__(self, filters=None, chunksize=None, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        if filters is not None:
            filters = list(filters)
            for f in filters:
                if not isinstance(f, Filter):
                    raise ValueError("filters argument must be an iterable of TileDB filter objects")
        if chunksize is not None:
            if not isinstance(chunksize, int):
                raise TypeError("chunksize keyword argument must be an integer or None")
            if chunksize <= 0:
               raise ValueError("chunksize arugment must be > 0")
        cdef tiledb_ctx_t* ctx_ptr = ctx.ptr
        cdef tiledb_filter_list_t* filter_list_ptr = NULL
        cdef int rc = TILEDB_OK
        rc = tiledb_filter_list_alloc(ctx_ptr, &filter_list_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        cdef tiledb_filter_t* filter_ptr = NULL
        cdef Filter filter

        if filters is not None:
            try:
                    for f in filters:
                        filter_ptr = (<Filter> f).ptr
                        rc = tiledb_filter_list_add_filter(ctx_ptr, filter_list_ptr, filter_ptr)
                        if rc != TILEDB_OK:
                            _raise_ctx_err(ctx_ptr, rc)
            except:
                tiledb_filter_list_free(&filter_list_ptr)
                raise
        if chunksize is not None:
            rc = tiledb_filter_list_set_max_chunk_size(ctx_ptr, filter_list_ptr, chunksize)
            if rc != TILEDB_OK:
                tiledb_filter_list_free(&filter_list_ptr)
        self.ctx = ctx
        self.ptr = filter_list_ptr

    def __cint__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            tiledb_filter_list_free(&self.ptr)

    @staticmethod
    cdef FilterList from_ptr(tiledb_filter_list_t* ptr, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        assert(ptr != NULL)
        cdef FilterList filter_list = FilterList.__new__(FilterList)
        filter_list.ctx = ctx
        # need to cast away the const
        filter_list.ptr = <tiledb_filter_list_t*> ptr
        return filter_list

    def __repr__(self):
        filters = ",\n       ".join(
            [repr(self._getfilter(i)) for i in range(self.nfilters)])
        return "FilterList([{0!s}])".format(filters)

    def __eq__(self, other):
        if other is None:
            return False
        if len(self) != len(other):
            return False
        for i,f in enumerate(self):
            if f != other[i]:
                return False
        return True

    @property
    def chunksize(self):
        """The chunk size used by the filter list."""

        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_filter_list_t* filter_list_ptr = self.ptr
        cdef unsigned int chunksize = 0
        cdef int rc = TILEDB_OK
        rc = tiledb_filter_list_get_max_chunk_size(ctx_ptr, filter_list_ptr, &chunksize)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return chunksize

    @property
    def nfilters(self):
        """
        :return: Number of filters in the filter list
        :rtype: int
        """

        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_filter_list_t* filter_list_ptr = self.ptr
        cdef unsigned int nfilters = 0
        cdef int rc = TILEDB_OK
        rc = tiledb_filter_list_get_nfilters(ctx_ptr, filter_list_ptr, &nfilters)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return nfilters

    cdef Filter _getfilter(FilterList self, int idx):
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_filter_list_t* filter_list_ptr = self.ptr
        cdef tiledb_filter_t* filter_ptr = NULL
        cdef int rc = TILEDB_OK
        rc = tiledb_filter_list_get_filter_from_index(ctx_ptr, filter_list_ptr, idx, &filter_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        cdef tiledb_filter_type_t filter_type = TILEDB_FILTER_NONE
        rc = tiledb_filter_get_type(ctx_ptr, filter_ptr, &filter_type)
        if rc != TILEDB_OK:
            tiledb_filter_free(&filter_ptr)
            _raise_ctx_err(ctx_ptr, rc)
        return _filter_type_ptr_to_filter(self.ctx, filter_type, filter_ptr)

    def __len__(self):
        """Returns the number of filters in the list."""
        return self.nfilters

    def __getitem__(self, idx):
        """Gets a copy of the filter in the list at the given index

        :param idx: index into the
        :type idx: int or slice
        :returns: A filter at given index / slice
        :raises IndexError: invalid index
        :raises: :py:exc:`tiledb.TileDBError`

        """
        if not isinstance(idx, (int, slice)):
            raise TypeError("FilterList indices must be integers or slices, not {:s}".format(type(idx).__name__))
        nfilters = self.nfilters
        if isinstance(idx, int):
            if idx < 0 or idx > (nfilters - 1):
                raise IndexError("FilterList index out of range")
            idx = slice(idx, idx + 1)
        else:
            if not isinstance(idx.start, int) or not isinstance(idx.stop, int) or not isinstance(idx.step, int):
                raise IndexError("FilterList slice indices must be integers or None")
        filters = []
        (start, stop, step) = idx.indices(nfilters)
        for i in range(start, stop, step):
            filters.append(self._getfilter(i))
        if len(filters) == 1:
            return filters[0]
        return filters

    def append(self, Filter filter):
        """Appends `filter` to the end of filter list

        :param filter: filter object to add
        :type filter: Filter
        :returns: None
        """
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_filter_list_t* filter_list_ptr = self.ptr
        assert(filter_list_ptr != NULL)

        if not isinstance(filter, Filter):
            raise ValueError("filter argument must be a TileDB filter objects")

        cdef tiledb_filter_t* filter_ptr = filter.ptr

        cdef int rc = TILEDB_OK
        rc = tiledb_filter_list_add_filter(ctx_ptr, filter_list_ptr, filter_ptr)
        if rc != TILEDB_OK:
             _raise_ctx_err(ctx_ptr, rc)


cdef class Attr(object):
    """Class representing a TileDB array attribute.

    :param tiledb.Ctx ctx: A TileDB Context
    :param str name: Attribute name, empty if anonymous
    :param dtype: Attribute value datatypes
    :type dtype: numpy.dtype object or type or string
    :param nullable: Attribute is nullable
    :type bool:
    :param fill: Fill value for unset cells.
    :param var: Attribute is variable-length (automatic for byte/string types)
    :type dtype: bool
    :param filters: List of filters to apply
    :type filters: FilterList
    :raises TypeError: invalid dtype
    :raises: :py:exc:`tiledb.TileDBError`

    """

    cdef unicode _get_name(Attr self):
        cdef const char* c_name = NULL
        check_error(self.ctx,
                    tiledb_attribute_get_name(self.ctx.ptr, self.ptr, &c_name))
        cdef unicode name = c_name.decode('UTF-8', 'strict')
        return name

    cdef unsigned int _cell_val_num(Attr self) except? 0:
        cdef unsigned int ncells = 0
        check_error(self.ctx,
                    tiledb_attribute_get_cell_val_num(self.ctx.ptr, self.ptr, &ncells))
        return ncells

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            tiledb_attribute_free(&self.ptr)

    def __capsule__(self):
        if self.ptr == NULL:
            raise TileDBError("internal error: cannot create capsule for uninitialized Attr!")
        cdef const char* name = "ctx"
        cap = PyCapsule_New(<void *>(self.ptr), name, NULL)
        return cap

    @staticmethod
    cdef from_ptr(const tiledb_attribute_t* ptr, Ctx ctx=None):
        """Constructs an Attr class instance from a (non-null) tiledb_attribute_t pointer
        """
        if not ctx:
            ctx = default_ctx()
        assert(ptr != NULL)
        cdef Attr attr = Attr.__new__(Attr)
        attr.ctx = ctx
        # need to cast away the const
        attr.ptr = <tiledb_attribute_t*> ptr
        return attr

    def __init__(self,
                 name=u"",
                 dtype=np.float64,
                 fill=None,
                 var=None,
                 nullable=False,
                 filters=None,
                 Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        cdef bytes bname = ustring(name).encode('UTF-8')
        cdef const char* name_ptr = PyBytes_AS_STRING(bname)
        cdef np.dtype _dtype
        cdef tiledb_datatype_t tiledb_dtype
        cdef uint32_t ncells

        if isinstance(dtype, str) and dtype == "ascii":
            tiledb_dtype = TILEDB_STRING_ASCII
            ncells = TILEDB_VAR_NUM
        else:
            _dtype = np.dtype(dtype)
            tiledb_dtype, ncells = array_type_ncells(_dtype)

        # ensure that all unicode strings are var-length
        if var or _dtype.kind == 'U':
            var = True
            ncells = TILEDB_VAR_NUM

        if _dtype.kind == 'S':
            if var and 0 < _dtype.itemsize:
                raise TypeError("dtype is not compatible with var-length attribute")
            
            if _dtype.itemsize == 0:
                if var == False:
                    raise TypeError("dtype is not compatible with fixed-length attribute")
            
                var = True
                ncells = TILEDB_VAR_NUM
        
        var = var or False
            
        # variable-length cell type
        if ncells == TILEDB_VAR_NUM and not var:
            raise TypeError("dtype is not compatible with var-length attribute")

        cdef FilterList filter_list
        if filters is not None:
            if not isinstance(filters, FilterList):
                try:
                    filters = iter(filters)
                except:
                    raise TypeError("filters argument must be a tiledb.FilterList or iterable of Filters")
                else:
                    # we want this to raise a specific error if construction fails
                    filters = FilterList(filters)
            filter_list = filters
        # alloc attribute object and set cell num / compressor
        cdef tiledb_attribute_t* attr_ptr = NULL
        cdef int rc = TILEDB_OK
        rc = tiledb_attribute_alloc(ctx.ptr, name_ptr, tiledb_dtype, &attr_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx.ptr, rc)
        rc = tiledb_attribute_set_cell_val_num(ctx.ptr, attr_ptr, ncells)
        if rc != TILEDB_OK:
            tiledb_attribute_free(&attr_ptr)
            _raise_ctx_err(ctx.ptr, rc)

        if nullable:
            rc = tiledb_attribute_set_nullable(ctx.ptr, attr_ptr, 1)
            if rc != TILEDB_OK:
                tiledb_attribute_free(&attr_ptr)
                _raise_ctx_err(ctx.ptr, rc)

        cdef tiledb_filter_list_t* filter_list_ptr = NULL
        if filters is not None:
            filter_list_ptr = filter_list.ptr
            rc = tiledb_attribute_set_filter_list(ctx.ptr, attr_ptr, filter_list_ptr)
            if rc != TILEDB_OK:
                tiledb_attribute_free(&attr_ptr)
                _raise_ctx_err(ctx.ptr, rc)

        cdef void* fill_ptr
        cdef uint64_t fill_nbytes
        if fill is not None:
            fill_array = np.array(fill, dtype=dtype)
            fill_nbytes = fill_array.nbytes
            fill_ptr = np.PyArray_DATA(fill_array)
            rc = tiledb_attribute_set_fill_value(ctx.ptr,
                                                 attr_ptr,
                                                 fill_ptr,
                                                 fill_nbytes)
            if rc != TILEDB_OK:
                tiledb_attribute_free(&attr_ptr)
                _raise_ctx_err(ctx.ptr, rc)

        self.ctx = ctx
        self.ptr = attr_ptr

    def __eq__(self, other):
        if not isinstance(other, Attr):
            return False
        if (self.name != other.name or
            self.dtype != other.dtype):
            return False
        return True

    cdef tiledb_datatype_t _get_type(Attr self) except? TILEDB_CHAR:
        cdef tiledb_datatype_t typ
        check_error(self.ctx,
                    tiledb_attribute_get_type(self.ctx.ptr, self.ptr, &typ))
        return typ

    def dump(self):
        """Dumps a string representation of the Attr object to standard output (stdout)"""
        check_error(self.ctx,
                    tiledb_attribute_dump(self.ctx.ptr, self.ptr, stdout))
        print('\n')
        return

    @property
    def dtype(self):
        """Return numpy dtype object representing the Attr type

        :rtype: numpy.dtype

        """
        cdef tiledb_datatype_t typ
        check_error(self.ctx,
                    tiledb_attribute_get_type(self.ctx.ptr, self.ptr, &typ))
        cdef uint32_t ncells = 0
        check_error(self.ctx,
                    tiledb_attribute_get_cell_val_num(self.ctx.ptr, self.ptr, &ncells))

        return np.dtype(_numpy_dtype(typ, ncells))

    @property
    def name(self):
        """Attribute string name, empty string if the attribute is anonymous

        :rtype: str
        :raises: :py:exc:`tiledb.TileDBError`

        """
        internal_name = self._get_name()
        # handle __attr names from arrays written with libtiledb < 2
        if internal_name == "__attr":
            return u""
        return internal_name

    @property
    def _internal_name(self):
        return self._get_name()

    @property
    def isanon(self):
        """True if attribute is an anonymous attribute

        :rtype: bool

        """
        cdef unicode name = self._get_name()
        return name == u"" or name.startswith(u"__attr")

    @property
    def compressor(self):
        """String label of the attributes compressor and compressor level

        :rtype: tuple(str, int)
        :raises: :py:exc:`tiledb.TileDBError`

        """
        # <todo> do we want to reimplement this on top of new API?
        pass

    @property
    def filters(self):
        """FilterList of the TileDB attribute

        :rtype: tiledb.FilterList
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef tiledb_filter_list_t* filter_list_ptr = NULL
        cdef int rc = TILEDB_OK
        check_error(self.ctx,
                    tiledb_attribute_get_filter_list(self.ctx.ptr, self.ptr, &filter_list_ptr))

        return FilterList.from_ptr(filter_list_ptr, self.ctx)

    @property
    def fill(self):
        """Fill value for unset cells of this attribute

        :rtype: depends on dtype
        :raises: :py:exc:`tiledb.TileDBERror`
        """
        cdef const uint8_t* value_ptr = NULL
        cdef uint64_t size
        check_error(self.ctx,
            tiledb_attribute_get_fill_value(
                self.ctx.ptr, self.ptr, <const void**>&value_ptr, &size))

        if value_ptr == NULL:
            return None

        if size == 0:
            raise TileDBError("Unexpected zero-length non-null fill value")

        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> 1
        cdef tiledb_datatype_t tiledb_type = self._get_type()
        cdef int typeid = _numpy_typeid(tiledb_type)
        assert(typeid != np.NPY_NOTYPE)
        cdef np.ndarray fill_array

        if np.issubdtype(self.dtype, np.bytes_):
            return (<char*>value_ptr)[:size]
        elif np.issubdtype(self.dtype, np.unicode_):
            return (<char*>value_ptr)[:size].decode('utf-8')
        else:
            fill_array = np.empty(1, dtype=self.dtype)
            memcpy(np.PyArray_DATA(fill_array), value_ptr, size)

        if _tiledb_type_is_datetime(tiledb_type):
            # Coerce to np.int64
            fill_array.dtype = np.int64
            datetime_dtype = _tiledb_type_to_datetime(tiledb_type).dtype
            date_unit = np.datetime_data(datetime_dtype)[0]
            tmp_val = None
            if fill_array[0] == 0:
                # undefined should span the whole dimension domain
                tmp_val = int(self.shape[0])
            else:
                tmp_val = int(fill_array[0])
            return np.timedelta64(tmp_val, date_unit)

        return fill_array

    @property
    def isnullable(self):
        """True if the attribute is nullable

        :rtype: bool
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef uint8_t nullable = 0
        cdef int rc = TILEDB_OK
        check_error(
            self.ctx,
            tiledb_attribute_get_nullable(self.ctx.ptr, self.ptr, &nullable))

        return <bint>nullable

    @property
    def isvar(self):
        """True if the attribute is variable length

        :rtype: bool
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef unsigned int ncells = self._cell_val_num()
        return ncells == TILEDB_VAR_NUM

    @property
    def ncells(self):
        """The number of cells (scalar values) for a given attribute value

        :rtype: int
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef unsigned int ncells = self._cell_val_num()
        assert (ncells != 0)
        return int(ncells)


    def __repr__(self):
        filters_str = ""
        if self.filters:
            filters_str = ", filters=FilterList(["
            for f in self.filters:
                filters_str +=  repr(f) + ", "
            filters_str += "])"

        attr_dtype = self.dtype if (self._get_type() != TILEDB_STRING_ASCII) else "ascii"

        # filters_str must be last with no spaces
        return (f"""Attr(name={repr(self.name)}, dtype='{attr_dtype!s}', """
                f"""var={self.isvar!s}, nullable={self.isnullable!s}"""
                f"""{filters_str})""")


cdef class Dim(object):
    """Class representing a dimension of a TileDB Array.

    :param str name: the dimension name, empty if anonymous
    :param domain:
    :type domain: tuple(int, int) or tuple(float, float)
    :param tile: Tile extent
    :type tile: int or float
    :param filters: List of filters to apply
    :type filters: FilterList
    :dtype: the Dim numpy dtype object, type object, or string \
        that can be corerced into a numpy dtype object
    :raises ValueError: invalid domain or tile extent
    :raises TypeError: invalid domain, tile extent, or dtype type
    :raises: :py:exc:`TileDBError`
    :param tiledb.Ctx ctx: A TileDB Context

    """

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            tiledb_dimension_free(&self.ptr)

    @staticmethod
    cdef from_ptr(const tiledb_dimension_t* ptr, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        assert(ptr != NULL)
        cdef Dim dim = Dim.__new__(Dim)
        dim.ctx = ctx
        # need to cast away the const
        dim.ptr = <tiledb_dimension_t*> ptr
        return dim

    def __init__(self, name=u"__dim_0", domain=None, tile=None,
                 filters=None, dtype=np.uint64, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()

        if domain is not None and len(domain) != 2:
            raise ValueError('invalid domain extent, must be a pair')

        # argument conversion
        cdef bytes bname = ustring(name).encode('UTF-8')
        cdef const char* name_ptr = PyBytes_AS_STRING(bname)
        cdef tiledb_datatype_t dim_datatype
        cdef const void* domain_ptr = NULL
        cdef tiledb_dimension_t* dim_ptr = NULL
        cdef void* tile_size_ptr = NULL
        cdef np.dtype domain_dtype

        if dtype == np.dtype('S'):
            # Handle var-len domain type
            #  (currently only TILEDB_STRING_ASCII)
            # The dimension's domain is implicitly formed as
            # coordinates are written.
            dim_datatype = TILEDB_STRING_ASCII
        else:
            if domain is None or len(domain) != 2:
                raise ValueError('invalid domain extent, must be a pair')

            if dtype is not None:
                dtype = np.dtype(dtype)
                dtype_min, dtype_max = dtype_range(dtype)

                if domain == (None, None):
                    # this means to use the full extent of the type
                    domain = (dtype_min, dtype_max)
                elif (domain[0] < dtype_min or domain[0] > dtype_max or
                        domain[1] < dtype_min or domain[1] > dtype_max):
                    raise TypeError(
                        "invalid domain extent, domain cannot be safely cast to dtype {0!r}".format(dtype))

            domain_array = np.asarray(domain, dtype=dtype)
            domain_ptr = np.PyArray_DATA(domain_array)
            domain_dtype = domain_array.dtype
            dim_datatype = dtype_to_tiledb(domain_dtype)
            # check that the domain type is a valid dtype (integer / floating)
            if (not np.issubdtype(domain_dtype, np.integer) and
                    not np.issubdtype(domain_dtype, np.floating) and
                    not domain_dtype.kind == 'M'):
                raise TypeError("invalid Dim dtype {0!r}".format(domain_dtype))
            # if the tile extent is specified, cast
            if tile is not None:
                tile_size_array = _tiledb_cast_tile_extent(tile, domain_dtype)
                if tile_size_array.size != 1:
                    raise ValueError("tile extent must be a scalar")
                tile_size_ptr = np.PyArray_DATA(tile_size_array)

        cdef FilterList filter_list
        cdef tiledb_filter_list_t* filter_list_ptr = NULL
        try:
            check_error(ctx,
                        tiledb_dimension_alloc(ctx.ptr,
                                               name_ptr,
                                               dim_datatype,
                                               domain_ptr,
                                               tile_size_ptr,
                                               &dim_ptr))

            assert dim_ptr != NULL, "internal error: tiledb_dimension_alloc null dim_ptr"

            if filters is not None:
                if not isinstance(filters, FilterList):
                    filters = FilterList(filters)
                filter_list = filters
                filter_list_ptr = filter_list.ptr
                check_error(ctx,
                    tiledb_dimension_set_filter_list(ctx.ptr, dim_ptr, filter_list_ptr))
        except:
            raise

        self.ctx = ctx
        self.ptr = dim_ptr

    def __repr__(self):
        filters_str = ""
        if self.filters:
            filters_str = ", filters=FilterList(["
            for f in self.filters:
                filters_str +=  repr(f) + ", "
            filters_str += "])"

        return "Dim(name={0!r}, domain={1!s}, tile='{2!s}', dtype='{3!s}'{4})" \
            .format(self.name, self.domain, self.tile, self.dtype, filters_str)

    def __len__(self):
        return self.size

    def __eq__(self, other):
        if not isinstance(other, Dim):
            return False
        if (self.name != other.name or
            self.domain != other.domain or
            self.tile != other.tile or
            self.dtype != other.dtype):
            return False
        return True

    def __array__(self, dtype=None, **kw):
        if not self._integer_domain():
            raise TypeError("conversion to numpy ndarray only valid for integer dimension domains")
        lb, ub = self.domain
        return np.arange(int(lb), int(ub) + 1,
                         dtype=dtype if dtype else self.dtype)

    cdef tiledb_datatype_t _get_type(Dim self) except? TILEDB_CHAR:
        cdef tiledb_datatype_t typ
        check_error(self.ctx,
                    tiledb_dimension_get_type(self.ctx.ptr, self.ptr, &typ))
        return typ

    @property
    def dtype(self):
        """Numpy dtype representation of the dimension type.

        :rtype: numpy.dtype

        """
        return np.dtype(_numpy_dtype(self._get_type()))

    @property
    def name(self):
        """The dimension label string.

        Anonymous dimensions return a default string representation based on the dimension index.

        :rtype: str

        """
        cdef const char* name_ptr = NULL
        check_error(self.ctx,
                    tiledb_dimension_get_name(self.ctx.ptr, self.ptr, &name_ptr))
        return name_ptr.decode('UTF-8', 'strict')

    @property
    def isvar(self):
        """True if the dimension is variable length

        :rtype: bool
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef unsigned int ncells = self._cell_val_num()
        return ncells == TILEDB_VAR_NUM

    @property
    def isanon(self):
        """True if the dimension is anonymous

        :rtype: bool

        """
        name = self.name
        return name == u"" or name.startswith("__dim")

    @property
    def filters(self):
        """FilterList of the TileDB dimension

        :rtype: tiledb.FilterList
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef tiledb_filter_list_t* filter_list_ptr = NULL
        cdef int rc = TILEDB_OK
        check_error(self.ctx,
                    tiledb_dimension_get_filter_list(self.ctx.ptr, self.ptr, &filter_list_ptr))

        return FilterList.from_ptr(filter_list_ptr, self.ctx)

    cdef unsigned int _cell_val_num(Dim self) except? 0:
        cdef unsigned int ncells = 0
        check_error(self.ctx,
                    tiledb_dimension_get_cell_val_num(
                        self.ctx.ptr,
                        self.ptr,
                        &ncells))
        return ncells

    cdef _integer_domain(self):
        cdef tiledb_datatype_t typ = self._get_type()
        if typ == TILEDB_FLOAT32 or typ == TILEDB_FLOAT64:
            return False
        return True

    cdef _datetime_domain(self):
        cdef tiledb_datatype_t typ = self._get_type()
        return _tiledb_type_is_datetime(typ)

    cdef _shape(self):
        domain = self.domain
        if self._datetime_domain():
            return (_tiledb_datetime_extent(domain[0], domain[1]),)
        else:
            return ((domain[1].item() -
                     domain[0].item() + 1),)

    @property
    def shape(self):
        """The shape of the dimension given the dimension's domain.

        **Note**: The shape is only valid for integer and datetime dimension domains.

        :rtype: tuple(numpy scalar, numpy scalar)
        :raises TypeError: floating point (inexact) domain

        """
        if not self._integer_domain() and not self._datetime_domain():
            raise TypeError("shape only valid for integer and datetime dimension domains")
        return self._shape()

    @property
    def size(self):
        """The size of the dimension domain (number of cells along dimension).

        :rtype: int
        :raises TypeError: floating point (inexact) domain

        """
        if not self._integer_domain():
            raise TypeError("size only valid for integer dimension domains")
        return int(self._shape()[0])

    @property
    def tile(self):
        """The tile extent of the dimension.

        :rtype: numpy scalar or np.timedelta64

        """
        cdef const void* tile_ptr = NULL
        check_error(self.ctx,
                    tiledb_dimension_get_tile_extent(self.ctx.ptr, self.ptr, &tile_ptr))
        if tile_ptr == NULL:
            return None
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> 1
        cdef tiledb_datatype_t tiledb_type = self._get_type()
        cdef int typeid = _numpy_typeid(tiledb_type)
        assert(typeid != np.NPY_NOTYPE)
        cdef np.ndarray tile_array =\
            np.PyArray_SimpleNewFromData(1, shape, typeid, <void*>tile_ptr)

        if _tiledb_type_is_datetime(tiledb_type):
            # Coerce to np.int64
            tile_array.dtype = np.int64
            datetime_dtype = _tiledb_type_to_datetime(tiledb_type).dtype
            date_unit = np.datetime_data(datetime_dtype)[0]
            extent = None
            if tile_array[0] == 0:
                # undefined tiles should span the whole dimension domain
                extent = int(self.shape[0])
            else:
                extent = int(tile_array[0])
            return np.timedelta64(extent, date_unit)
        else:
            if tile_array[0] == 0:
                # undefined tiles should span the whole dimension domain
                return self.shape[0]
            return tile_array[0]

    @property
    def domain(self):
        """The dimension (inclusive) domain.

        The dimension's domain is defined by a (lower bound, upper bound) tuple.

        :rtype: tuple(numpy scalar, numpy scalar)

        """
        if self.dtype == np.dtype('S'):
            return None, None
        cdef const void* domain_ptr = NULL
        check_error(self.ctx,
                    tiledb_dimension_get_domain(self.ctx.ptr,
                                                self.ptr,
                                                &domain_ptr))
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> 2
        cdef tiledb_datatype_t tiledb_type = self._get_type()
        cdef int typeid = _numpy_typeid(tiledb_type)
        assert (typeid != np.NPY_NOTYPE)
        cdef np.ndarray domain_array = \
            np.PyArray_SimpleNewFromData(1, shape, typeid, <void*>domain_ptr)

        if _tiledb_type_is_datetime(tiledb_type):
            domain_array.dtype = _tiledb_type_to_datetime(tiledb_type).dtype

        return domain_array[0], domain_array[1]


def clone_dim_with_name(Dim dim, name):
    return Dim(name=name, domain=dim.domain, tile=dim.tile, dtype=dim.dtype, ctx=dim.ctx)

cdef class Domain(object):
    """Class representing the domain of a TileDB Array.

    :param *dims*: one or more tiledb.Dim objects up to the Domain's ndim
    :raises TypeError: All dimensions must have the same dtype
    :raises: :py:exc:`TileDBError`
    :param tiledb.Ctx ctx: A TileDB Context

    """

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            tiledb_domain_free(&self.ptr)

    @staticmethod
    cdef from_ptr(const tiledb_domain_t* ptr, Ctx ctx=None):
        """Constructs an Domain class instance from a (non-null) tiledb_domain_t pointer"""
        if not ctx:
            ctx = default_ctx()
        assert(ptr != NULL)
        cdef Domain dom = Domain.__new__(Domain)
        dom.ctx = ctx
        dom.ptr = <tiledb_domain_t*> ptr
        return dom

    cdef tiledb_datatype_t _get_type(Domain self) except? TILEDB_CHAR:
        cdef tiledb_datatype_t typ
        check_error(self.ctx,
                    tiledb_domain_get_type(self.ctx.ptr, self.ptr, &typ))
        return typ

    cdef _integer_domain(Domain self):
        if not self._is_homogeneous():
            return False
        cdef tiledb_datatype_t typ = self._get_type()
        if typ == TILEDB_FLOAT32 or typ == TILEDB_FLOAT64:
            return False
        return True

    cdef _is_homogeneous(Domain self):
        cdef np.dtype dtype0 = self.dim(0).dtype
        return all(self.dim(i).dtype == dtype0 for i in range(1,self.ndim))

    cdef _shape(Domain self):
        return tuple(self.dim(i).shape[0] for i in range(self.ndim))

    def __init__(self, *dims, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()

        # support passing a list of dims without splatting
        if len(dims) == 1 and isinstance(dims[0], list):
            dims = dims[0]

        cdef Py_ssize_t ndim = len(dims)
        if ndim == 0:
            raise TileDBError("Domain must have ndim >= 1")

        if (ndim > 1):
            if all(dim.name == '__dim_0' for dim in dims):
                # rename anonymous dimensions sequentially
                dims = [clone_dim_with_name(dims[i], name=f'__dim_{i}') for i in range(ndim)]
            elif any(dim.name.startswith('__dim_0') for dim in dims[1:]):
                raise TileDBError("Mixed dimension naming: dimensions must be either all anonymous or all named.")

        cdef tiledb_domain_t* domain_ptr = NULL
        cdef int rc = tiledb_domain_alloc(ctx.ptr, &domain_ptr)
        if rc != TILEDB_OK:
            check_error(ctx, rc)
        assert(domain_ptr != NULL)

        cdef Dim dimension
        for i in range(ndim):
            if not isinstance(dims[i], Dim):
                raise TypeError("Cannot create Domain with non-Dim value for 'dims' argument")

            dimension = dims[i]
            rc = tiledb_domain_add_dimension(
                ctx.ptr, domain_ptr, dimension.ptr)
            if rc != TILEDB_OK:
                tiledb_domain_free(&domain_ptr)
                check_error(ctx, rc)
        self.ctx = ctx
        self.ptr = domain_ptr

    def __repr__(self):
        dims = ",\n       ".join(
            [repr(self.dim(i)) for i in range(self.ndim)])
        return "Domain({0!s})".format(dims)

    def __len__(self):
        """Returns the number of dimensions of the domain"""
        return self.ndim

    def __iter__(self):
        """Returns a generator object that iterates over the domain's dimension objects"""
        return (self.dim(i) for i in range(self.ndim))

    def __eq__(self, other):
        """Returns true if Domain is equal to self.

        :rtype: bool
        """
        if not isinstance(other, Domain):
            return False

        cdef bint same_dtype = self._is_homogeneous()

        if (same_dtype and
            self.shape != other.shape):
            return False

        ndim = self.ndim
        if (ndim != other.ndim):
            return False

        for i in range(ndim):
            if self.dim(i) != other.dim(i):
                return False
        return True

    @property
    def ndim(self):
        """The number of dimensions of the domain.

        :rtype: int

        """
        cdef unsigned int ndim = 0
        check_error(self.ctx,
                    tiledb_domain_get_ndim(self.ctx.ptr, self.ptr, &ndim))
        return ndim

    @property
    def dtype(self):
        """The numpy dtype of the domain's dimension type.

        :rtype: numpy.dtype

        """
        cdef tiledb_datatype_t typ = self._get_type()
        return np.dtype(_numpy_dtype(typ))

    @property
    def shape(self):
        """The domain's shape, valid only for integer domains.

        :rtype: tuple
        :raises TypeError: floating point (inexact) domain

        """
        if not self._integer_domain():
            raise TypeError("shape valid only for integer domains")
        return self._shape()

    @property
    def size(self):
        """The domain's size (number of cells), valid only for integer domains.

        :rtype: int
        :raises TypeError: floating point (inexact) domain

        """
        if not self._integer_domain():
            raise TypeError("shape valid only for integer domains")
        return np.product(self._shape())

    @property
    def homogeneous(self):
        """Returns True if the domain's dimension types are homogeneous."""
        return self._is_homogeneous()

    def dim(self, dim_id):
        """Returns a Dim object from the domain given the dimension's index or name.

        :param dim_d: dimension index (int) or name (str)
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef tiledb_dimension_t* dim_ptr = NULL
        cdef bytes uname
        cdef const char* name_ptr = NULL

        if isinstance(dim_id, (str, unicode)):
            uname = ustring(dim_id).encode('UTF-8')
            name_ptr = uname
            check_error(self.ctx,
                        tiledb_domain_get_dimension_from_name(
                            self.ctx.ptr, self.ptr, name_ptr, &dim_ptr))
        elif isinstance(dim_id, int):
            check_error(self.ctx,
                        tiledb_domain_get_dimension_from_index(
                            self.ctx.ptr, self.ptr, dim_id, &dim_ptr))
        else:
            raise ValueError("Unsupported dim identifier: '{}' (expected int or str)".format(
                safe_repr(dim_id)
            ))

        assert(dim_ptr != NULL)
        return Dim.from_ptr(dim_ptr, self.ctx)

    def has_dim(self, name):
        """
        Returns true if the Domain has a Dimension with the given name

        :param name: name of Dimension
        :rtype: bool
        :return:
        """
        cdef:
            cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
            cdef tiledb_domain_t* dom_ptr = self.ptr
            int32_t has_dim = 0
            int32_t rc = TILEDB_OK
            bytes bname = name.encode("UTF-8")

        rc = tiledb_domain_has_dimension(
            ctx_ptr,
            dom_ptr,
            bname,
            &has_dim
        )
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return bool(has_dim)


    def dump(self):
        """Dumps a string representation of the domain object to standard output (STDOUT)"""
        check_error(self.ctx,
                    tiledb_domain_dump(self.ctx.ptr, self.ptr, stdout))
        print("\n")
        return

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


def replace_scalars_slice(dom: Domain, idx: tuple):
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


def index_domain_subarray(array: Array, dom: Domain, idx: tuple):
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

        if np.issubdtype(dim_dtype, np.unicode_) or np.issubdtype(dim_dtype, np.bytes_):
            (dim_lb, dim_ub) = array.nonempty_domain()[r]
        else:
            (dim_lb, dim_ub) = dim.domain

        dim_slice = idx[r]
        if not isinstance(dim_slice, slice):
            raise IndexError("invalid index type: {!r}".format(type(dim_slice)))

        start, stop, step = dim_slice.start, dim_slice.stop, dim_slice.step

        if np.issubdtype(dim_dtype, np.str_) or np.issubdtype(dim_dtype, np.bytes_):
            if start is None:
                start = dim_lb
            if stop is None:
                stop = dim_ub
            if not isinstance(start, (bytes,unicode)) or not isinstance(stop, (bytes,unicode)):
                raise TileDBError(f"Non-string range '({start},{stop})' provided for string dimension '{dim.name}'")
            subarray.append((start,stop))
            continue

        #if step and step < 0:
        #    raise IndexError("only positive slice steps are supported")

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
    :param coords_filters: (default None) coordinate filter list
    :type coords_filters: tiledb.FilterList
    :param offsets_filters: (default None) offsets filter list
    :type offsets_filters: tiledb.FilterList
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
                 allows_duplicates=False,
                 sparse=False,
                 Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        cdef tiledb_array_type_t array_type =\
            TILEDB_SPARSE if sparse else TILEDB_DENSE
        cdef tiledb_array_schema_t* schema_ptr = NULL
        check_error(ctx,
                    tiledb_array_schema_alloc(ctx.ptr, array_type, &schema_ptr))
        cdef tiledb_layout_t cell_layout = TILEDB_ROW_MAJOR
        cdef tiledb_layout_t tile_layout = TILEDB_ROW_MAJOR
        try:
            cell_layout = _tiledb_layout(cell_order if cell_order else 'row-major')
            tile_layout = _tiledb_layout(tile_order if tile_order else 'row-major')
            check_error(ctx, tiledb_array_schema_set_cell_order(ctx.ptr, schema_ptr, cell_layout))
            check_error(ctx, tiledb_array_schema_set_tile_order(ctx.ptr, schema_ptr, tile_layout))
        except:
            tiledb_array_schema_free(&schema_ptr)
            raise
        cdef uint64_t _capacity = 0
        if capacity > 0:
            try:
                _capacity = <uint64_t> capacity
                check_error(ctx,
                    tiledb_array_schema_set_capacity(ctx.ptr, schema_ptr, _capacity))
            except:
                tiledb_array_schema_free(&schema_ptr)
                raise

        cdef bint ballows_dups = 0
        if allows_duplicates:
            ballows_dups = 1
            tiledb_array_schema_set_allows_dups(ctx.ptr, schema_ptr, ballows_dups)

        cdef FilterList filter_list
        cdef tiledb_filter_list_t* filter_list_ptr = NULL
        try:
            if offsets_filters is not None:
                if not isinstance(offsets_filters, FilterList):
                    offsets_filters = FilterList(offsets_filters)
                filter_list = offsets_filters
                filter_list_ptr = filter_list.ptr
                check_error(ctx,
                    tiledb_array_schema_set_offsets_filter_list(ctx.ptr, schema_ptr, filter_list_ptr))
            if coords_filters is not None:
                if not isinstance(coords_filters, FilterList):
                    coords_filters = FilterList(coords_filters)
                filter_list = coords_filters
                filter_list_ptr = filter_list.ptr
                check_error(ctx,
                    tiledb_array_schema_set_coords_filter_list(ctx.ptr, schema_ptr, filter_list_ptr))
        except:
            tiledb_array_schema_free(&schema_ptr)
            raise

        if  not isinstance(domain, Domain):
            raise TypeError("'domain' must be an instance of Domain (domain is: '{}')".format(domain))
        cdef tiledb_domain_t* domain_ptr = (<Domain> domain).ptr
        rc = tiledb_array_schema_set_domain(ctx.ptr, schema_ptr, domain_ptr)
        if rc != TILEDB_OK:
            tiledb_array_schema_free(&schema_ptr)
            _raise_ctx_err(ctx.ptr, rc)
        cdef tiledb_attribute_t* attr_ptr = NULL
        cdef Attr attribute
        for attr in attrs:
            if not isinstance(attr, Attr):
                raise TypeError("Cannot create schema with non-Attr value for 'attrs' argument")
            attribute = attr
            attr_ptr = attribute.ptr
            rc = tiledb_array_schema_add_attribute(ctx.ptr, schema_ptr, attr_ptr)
            if rc != TILEDB_OK:
                tiledb_array_schema_free(&schema_ptr)
                _raise_ctx_err(ctx.ptr, rc)
        rc = tiledb_array_schema_check(ctx.ptr, schema_ptr)
        if rc != TILEDB_OK:
            tiledb_array_schema_free(&schema_ptr)
            _raise_ctx_err(ctx.ptr, rc)
        self.ctx = ctx
        self.ptr = schema_ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            tiledb_array_schema_free(&self.ptr)

    @staticmethod
    cdef from_ptr(const tiledb_array_schema_t* schema_ptr, Ctx ctx=None):
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
    def load(uri, Ctx ctx=None, key=None):
        if not ctx:
            ctx = default_ctx()
        cdef bytes buri = uri.encode('UTF-8')
        cdef tiledb_ctx_t* ctx_ptr = ctx.ptr
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
        check_error(self.ctx,
                    tiledb_array_schema_check(self.ctx.ptr, self.ptr))

    @property
    def sparse(self):
        """True if the array is a sparse array representation

        :rtype: bool
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef tiledb_array_type_t typ = TILEDB_DENSE
        check_error(self.ctx,
                    tiledb_array_schema_get_array_type(self.ctx.ptr, self.ptr, &typ))
        return typ == TILEDB_SPARSE

    @property
    def allows_duplicates(self):
        """Returns True if the (sparse) array allows duplicates."""

        if not self.sparse:
            raise TileDBError("ArraySchema.allows_duplicates does not apply to dense arrays")

        cdef int ballows_dups
        tiledb_array_schema_get_allows_dups(self.ctx.ptr, self.ptr, &ballows_dups)
        return bool(ballows_dups)

    @property
    def capacity(self):
        """The array capacity

        :rtype: int
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef uint64_t cap = 0
        check_error(self.ctx,
                    tiledb_array_schema_get_capacity(self.ctx.ptr, self.ptr, &cap))
        return cap

    cdef _cell_order(ArraySchema self, tiledb_layout_t* cell_order_ptr):
        check_error(self.ctx,
            tiledb_array_schema_get_cell_order(self.ctx.ptr, self.ptr, cell_order_ptr))

    @property
    def cell_order(self):
        """The cell order layout of the array."""
        cdef tiledb_layout_t order = TILEDB_UNORDERED
        self._cell_order(&order)
        return _tiledb_layout_string(order)

    cdef _tile_order(ArraySchema self, tiledb_layout_t* tile_order_ptr):
        check_error(self.ctx,
            tiledb_array_schema_get_tile_order(self.ctx.ptr, self.ptr, tile_order_ptr))

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
        check_error(self.ctx,
            tiledb_array_schema_get_offsets_filter_list(
                self.ctx.ptr, self.ptr, &filter_list_ptr))
        return FilterList.from_ptr(filter_list_ptr, self.ctx)

    @property
    def coords_filters(self):
        """The FilterList for the array's coordinates

        :rtype: tiledb.FilterList
        :raises: :py:exc:`tiledb.TileDBError`
        """
        cdef tiledb_filter_list_t* filter_list_ptr = NULL
        check_error(self.ctx,
            tiledb_array_schema_get_coords_filter_list(
                self.ctx.ptr, self.ptr, &filter_list_ptr))
        return FilterList.from_ptr(filter_list_ptr, self.ctx)

    @property
    def domain(self):
        """The Domain associated with the array.

        :rtype: tiledb.Domain
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef tiledb_domain_t* dom = NULL
        check_error(self.ctx,
                    tiledb_array_schema_get_domain(self.ctx.ptr, self.ptr, &dom))
        return Domain.from_ptr(dom, self.ctx)

    @property
    def nattr(self):
        """The number of array attributes.

        :rtype: int
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef unsigned int nattr = 0
        check_error(self.ctx,
                    tiledb_array_schema_get_attribute_num(self.ctx.ptr, self.ptr, &nattr))
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

    def _make_invalid(self):
        """This is a helper function for testing schema.check: resets schema
        in order to make the schema invalid."""
        cdef tiledb_array_schema_t* schema_ptr = self.ptr
        tiledb_array_schema_free(&schema_ptr)
        check_error(self.ctx,
                    tiledb_array_schema_alloc(self.ctx.ptr, TILEDB_DENSE, &self.ptr))

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
        check_error(self.ctx,
                    tiledb_array_schema_get_attribute_from_name(
                        self.ctx.ptr, self.ptr, bname, &attr_ptr))
        return Attr.from_ptr(attr_ptr, self.ctx)

    cdef _attr_idx(self, int idx):
        cdef tiledb_attribute_t* attr_ptr = NULL
        check_error(self.ctx,
                    tiledb_array_schema_get_attribute_from_index(
                        self.ctx.ptr, self.ptr, idx, &attr_ptr))
        return Attr.from_ptr(attr_ptr, ctx=self.ctx)

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
        :rtype: boolean
        """
        cdef:
            int32_t has_attr = 0
            int32_t rc = TILEDB_OK
            bytes bname = name.encode("UTF-8")

        rc = tiledb_array_schema_has_attribute(
            self.ctx.ptr,
            self.ptr,
            bname,
            &has_attr
        )
        if rc != TILEDB_OK:
            _raise_ctx_err(self.ctx.ptr, rc)

        return bool(has_attr)

    def attr_or_dim_dtype(self, unicode name):
        if self.has_attr(name):
            dtype = self.attr(name).dtype
        elif self.domain.has_dim(name):
            dtype = self.domain.dim(name).dtype
        else:
            raise TileDBError(f"Unknown attribute or dimension ('{name}')")

        if dtype.itemsize == 0:
            # special handling for flexible numpy dtypes: change itemsize from 0 to 1
            dtype = np.dtype((dtype, 1))
        return dtype


    def dump(self):
        """Dumps a string representation of the array object to standard output (stdout)"""
        check_error(self.ctx,
                    tiledb_array_schema_dump(self.ctx.ptr, self.ptr, stdout))
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

        if self.coords_filters is not None:
            output.write(f"  coords_filters={self.coords_filters},\n")

        output.write(")\n")

        return output.getvalue()


# Wrapper class to allow returning a Python object so that exceptions work correctly
# within preload_array
cdef class ArrayPtr(object):
    cdef tiledb_array_t* ptr

cdef ArrayPtr preload_array(uri, mode, key, timestamp, Ctx ctx=None):
    """Open array URI without constructing specific type of Array object (internal)."""
    if not ctx:
        ctx = default_ctx()
    # ctx
    cdef tiledb_ctx_t* ctx_ptr = ctx.ptr
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
    if mode == 'r':
        query_type = TILEDB_READ
    elif mode == 'w':
        query_type = TILEDB_WRITE
    else:
        raise ValueError("TileDB array mode must be 'r' or 'w'")
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
    :param str mode: (default 'r') Open the array object in read 'r' or write 'w' mode
    :param str key: (default None) If not None, encryption key to decrypt the array
    :param tuple timestamp: (default None) If int, open the array at a given TileDB
        timestamp. If tuple, open at the given start and end TileDB timestamps.
    :param str attr: (default None) open one attribute of the array; indexing a
        dense array will return a Numpy ndarray directly rather than a dictionary.
    :param Ctx ctx: TileDB context
    """
    def __init__(self, uri, mode='r', key=None, timestamp=None,
                 attr=None, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        # ctx
        cdef tiledb_ctx_t* ctx_ptr = ctx.ptr
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

        # Delayed to avoid circular import
        from .multirange_indexing import DataFrameIndexer, MultiRangeIndexer
        self.multi_index = MultiRangeIndexer(self)
        self.df = DataFrameIndexer(self, use_arrow=None)
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
    def create(cls, uri, ArraySchema schema, key=None, Ctx ctx=None):
        """Creates a TileDB Array at the given URI

        :param str uri: URI at which to create the new empty array.
        :param ArraySchema schema: Schema for the array
        :param str key: (default None) Encryption key to use for array
        :param ctx Ctx: (default None) Optional TileDB Ctx used when creating the array,
                        by default uses the ArraySchema's associated context
                        (*not* necessarily ``tiledb.default_ctx``).

        """
        if issubclass(cls, DenseArrayImpl) and schema.sparse:
            raise ValueError("Array.create `schema` argument must be a dense schema for DenseArray and subclasses")
        if issubclass(cls, SparseArrayImpl) and not schema.sparse:
            raise ValueError("Array.create `schema` argument must be a sparse schema for SparseArray and subclasses")

        cdef tiledb_ctx_t* ctx_ptr = schema.ctx.ptr
        cdef bytes buri = unicode_path(uri)
        cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
        cdef tiledb_array_schema_t* schema_ptr = schema.ptr

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

        if ctx is not None:
            ctx_ptr = ctx.ptr

        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_array_create_with_key(ctx_ptr, uri_ptr, schema_ptr, key_type, key_ptr, key_len)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return

    @staticmethod
    def load_typed(uri, mode='r', key=None, timestamp=None, attr=None, Ctx ctx=None):
        """Return a {Dense,Sparse}Array instance from a pre-opened Array (internal)"""
        if not ctx:
            ctx = default_ctx()
        cdef int32_t rc = TILEDB_OK
        cdef tiledb_ctx_t* ctx_ptr = ctx.ptr
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
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        """Closes this array, flushing all buffered data."""
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
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
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
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
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
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
    def timestamp(self):
        """Deprecated in 0.9.2.

        Use `timestamp_range`

        Returns the timestamp the array is opened at

        :rtype: int
        :returns: tiledb timestamp at which point the array was opened

        """
        warnings.warn(
            "timestamp is deprecated; please use timestamp_range",
            DeprecationWarning,
        )

        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_array_t* array_ptr = self.ptr
        cdef uint64_t timestamp = 0
        cdef int rc = TILEDB_OK
        rc = tiledb_array_get_timestamp(ctx_ptr, array_ptr, &timestamp)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return int(timestamp)


    @property
    def timestamp_range(self):
        """Returns the timestamp range the array is opened at

        :rtype: tuple
        :returns: tiledb timestamp range at which point the array was opened

        """
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
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
        """Returns the numpy record array dtype of the array coordinates

        :rtype: numpy.dtype
        :returns: coord array record dtype

        """
        # deprecated in 0.8.10
        warnings.warn(
            """`coords_dtype` is deprecated because combined coords have been removed from libtiledb.
            Currently it returns a record array of each individual dimension dtype, but it will
            be removed because that is not applicable to split dimensions.""",
            DeprecationWarning,
        )
        # returns the record array dtype of the coordinate array
        return np.dtype([(str(dim.name), dim.dtype) for dim in self.schema.domain])

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

    def nonempty_domain(self):
        """Return the minimum bounding domain which encompasses nonempty values.

        :rtype: tuple(tuple(numpy scalar, numpy scalar), ...)
        :return: A list of (inclusive) domain extent tuples, that contain all
            nonempty cells

        """
        cdef list results = list()
        cdef Domain dom = self.schema.domain

        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
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
                start_buf = np.empty(end_size, 'S' + str(start_size))
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
                results.append((start_buf.item(0), end_buf.item(0)))
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

    def consolidate(self, Config config=None, key=None, timestamp=None):
        """
        Consolidates fragments of an array object for increased read performance.

        Overview: https://docs.tiledb.com/main/solutions/tiledb-embedded/internal-mechanics/consolidation

        :param tiledb.Config config: The TileDB Config with consolidation parameters set
        :param key: (default None) encryption key to decrypt an encrypted array
        :type key: str or bytes
        :param timestamp: (default None) If not None, vacuum the array using the
        given tuple(int, int) UNIX seconds range (inclusive)
        :type timestamp: tuple (int, int)
        :raises: :py:exc:`tiledb.TileDBError`

        Rather than passing the timestamp into this function, it may be set with
        the config parameters `"sm.vacuum.timestamp_start"`and
        `"sm.vacuum.timestamp_end"` which takes in a time in UNIX seconds. If both
        are set then this function's `timestamp` argument will be used.

        """
        if self.mode == 'r':
            raise TileDBError("cannot consolidate array opened in readonly mode (mode='r')")
        return consolidate(uri=self.uri, key=key, config=config, ctx=self.ctx, timestamp=timestamp)

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
        ...    A = tiledb.DenseArray.from_numpy(tmp, np.eye(4) * [1,2,3,4])
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
        return self.multi_index

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

        return self.df


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

    def __init__(self, array, attrs=None, attr_cond=None, dims=None,
                 coords=False, index_col=True,
                 order=None, use_arrow=None, return_arrow=False,
                 return_incomplete=False):
        if array.mode != 'r':
            raise ValueError("array mode must be read-only")

        if dims is not None and coords == True:
            raise ValueError("Cannot pass both dims and coords=True to Query")

        cdef list dims_to_set = list()

        if dims is False:
            self.dims = False
        elif dims != None and dims != True:
            domain = array.schema.domain
            for dname in dims:
                if not domain.has_dim(dname):
                    raise TileDBError("Selected dimension does not exist: '{name}")
            self.dims = [unicode(dname) for dname in dims]
        elif coords == True or dims == True:
            domain = array.schema.domain
            self.dims = [domain.dim(i).name for i in range(domain.ndim)]

        if attrs is not None:
            for name in attrs:
                if not array.schema.has_attr(name):
                    raise TileDBError(f"Selected attribute does not exist: '{name}'")
        self.attrs = attrs
        self.attr_cond = attr_cond

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
        return self.array.subarray(selection,
                                   attrs=self.attrs,
                                   coords=self.coords if self.coords else self.dims,
                                   order=self.order)

    @property
    def attrs(self):
        """List of attributes to include in Query."""
        return self.attrs

    @property
    def attr_cond(self):
        """QueryCondition used to filter attributes in Query."""
        return self.attr_cond

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


cdef class DenseArrayImpl(Array):
    """Class representing a dense TileDB array.

    Inherits properties and methods of :py:class:`tiledb.Array`.

    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if self.schema.sparse:
            raise ValueError("Array at {} is not a dense array".format(self.uri))
        return

    @staticmethod
    def from_numpy(uri, np.ndarray array, Ctx ctx=None, **kw):
        """Implementation of tiledb.from_numpy for dense arrays. See documentation
        of tiledb.from_numpy
        """
        if not ctx:
            ctx = default_ctx()
        schema = schema_like_numpy(array, ctx=ctx, **kw)
        Array.create(uri, schema)

        with DenseArray(uri, mode='w', ctx=ctx) as arr:
            # <TODO> probably need better typecheck here
            if array.dtype == object:
                arr[:] = array
            else:
                arr.write_direct(np.ascontiguousarray(array))
        return DenseArray(uri, mode='r', ctx=ctx)

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
        ...     A = tiledb.DenseArray.from_numpy(tmp + "/array",  np.ones((100, 100)))
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

    def query(self, attrs=None, attr_cond=None, dims=None, coords=False, order='C',
              use_arrow=None, return_arrow=False, return_incomplete=False):
        """
        Construct a proxy Query object for easy subarray queries of cells
        for an item or region of the array across one or more attributes.

        Optionally subselect over attributes, return dense result coordinate values,
        and specify a layout a result layout / cell-order.

        :param attrs: the DenseArray attributes to subselect over.
            If attrs is None (default) all array attributes will be returned.
            Array attributes can be defined by name or by positional index.
        :param attr_cond: the QueryCondition to filter attributes on.
        :param dims: the DenseArray dimensions to subselect over. If dims is None (default)
            then no dimensions are returned, unless coords=True.
        :param coords: if True, return array of coodinate value (default False).
        :param order: 'C', 'F', 'U', or 'G' (row-major, col-major, unordered, TileDB global order)
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
            raise TileDBError("DenseArray is not opened for reading")
        return Query(self, attrs=attrs, attr_cond=attr_cond, dims=dims,
                     coords=coords, order=order, use_arrow=use_arrow,
                     return_arrow=return_arrow,
                     return_incomplete=return_incomplete)

    def subarray(self, selection, attrs=None, attr_cond=None, coords=False, order=None):
        """Retrieve data cells for an item or region of the array.

        Optionally subselect over attributes, return dense result coordinate values,
        and specify a layout a result layout / cell-order.

        :param selection: tuple of scalar and/or slice objects
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
            raise TileDBError("DenseArray is not opened for reading")
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
        out = self._read_dense_subarray(subarray, attr_names, layout, coords)
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
                              tiledb_layout_t layout, bint include_coords):

        from tiledb.main import PyQuery
        q = PyQuery(self._ctx_(), self, tuple(attr_names), tuple(), <int32_t>layout, False)
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
        ...     with tiledb.DenseArray.from_numpy(tmp + "/array",  np.zeros((2, 2))) as A:
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
        if any(isinstance(s, np.ndarray) for s in selection_tuple):
            _setitem_impl_sparse(self, selection, val, dict())
            return

        self._setitem_impl(selection, val, dict())

    def _setitem_impl(self, object selection, object val, dict nullmaps):
        """Implementation for setitem with optional support for validity bitmaps."""
        if not self.isopen or self.mode != 'w':
            raise TileDBError("DenseArray is not opened for writing")

        cdef Domain domain = self.domain
        cdef tuple idx = replace_ellipsis(domain.ndim, index_as_tuple(selection))
        idx,_drop = replace_scalars_slice(domain, idx)
        cdef object subarray = index_domain_subarray(self, domain, idx)
        cdef Attr attr
        cdef list attributes = list()
        cdef list values = list()

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
                if self.schema.attr(key).isvar:
                    raise TileDBError("<todo> Var-length nullable arrays are unimplemented in TileDB-Py")
                if not isinstance(val, np.ndarray):
                    raise TypeError(f"Expected NumPy array for attribute '{key}' "
                                    f"validity bitmap, got {type(val)}")
                if val.dtype != np.uint8:
                    raise TypeError(f"Expected NumPy uint8 array for attribute '{key}' "
                                    f"validity bitmap, got {val.dtype}")

        _write_array(self.ctx.ptr,
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

    def write_direct(self, np.ndarray array not None):
        """
        Write directly to given array attribute with minimal checks,
        assumes that the numpy array is the same shape as the array's domain

        :param np.ndarray array: Numpy contiguous dense array of the same dtype \
            and shape and layout of the DenseArray instance
        :raises ValueError: array is not contiguous
        :raises: :py:exc:`tiledb.TileDBError`

        """
        if not self.isopen or self.mode != 'w':
            raise TileDBError("DenseArray is not opened for writing")
        if self.schema.nattr != 1:
            raise ValueError("cannot write_direct to a multi-attribute DenseArray")
        if not array.flags.c_contiguous and not array.flags.f_contiguous:
            raise ValueError("array is not contiguous")

        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_array_t* array_ptr = self.ptr

        # attr name
        cdef Attr attr = self.schema.attr(0)
        cdef bytes battr_name = attr._internal_name.encode('UTF-8')
        cdef const char* attr_name_ptr = PyBytes_AS_STRING(battr_name)


        cdef void* buff_ptr = np.PyArray_DATA(array)
        cdef uint64_t buff_size = array.nbytes

        cdef tiledb_layout_t layout = TILEDB_ROW_MAJOR
        if array.ndim == 1:
            layout = TILEDB_GLOBAL_ORDER
        elif array.ndim > 1 and array.flags.f_contiguous:
            layout = TILEDB_COL_MAJOR

        cdef tiledb_query_t* query_ptr = NULL
        cdef int rc = TILEDB_OK
        rc = tiledb_query_alloc(ctx_ptr, array_ptr, TILEDB_WRITE, &query_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        try:
            rc = tiledb_query_set_layout(ctx_ptr, query_ptr, layout)
            if rc != TILEDB_OK:
                _raise_ctx_err(ctx_ptr, rc)
            rc = tiledb_query_set_buffer(ctx_ptr, query_ptr, attr_name_ptr, buff_ptr, &buff_size)
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
        cdef Ctx ctx = self.ctx
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_array_t* array_ptr = self.ptr

        cdef Attr attr
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
        cdef Domain domain = schema.domain

        idx = tuple(slice(None) for _ in range(domain.ndim))
        subarray = index_domain_subarray(self, domain, idx)
        out = self._read_dense_subarray(subarray, [attr_name,], cell_layout, False)
        return out[attr_name]

# point query index a tiledb array (zips) columnar index vectors
def index_domain_coords(dom: Domain, idx: tuple):
    """
    Returns a (zipped) coordinate array representation
    given coordinate indices in numpy's point indexing format
    """
    ndim = len(idx)
    if ndim != dom.ndim:
        raise IndexError("sparse index ndim must match "
                         "domain ndim: {0!r} != {1!r}".format(ndim, dom.ndim))
    idx = tuple(np.array(idx[i], dtype=dom.dim(i).dtype, ndmin=1)
                for i in range(ndim))

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
    if not self.isopen or self.mode != 'w':
        raise TileDBError("SparseArray is not opened for writing")
    idx = index_as_tuple(selection)
    sparse_coords = list(index_domain_coords(self.schema.domain, idx))
    dim0_dtype = self.schema.domain.dim(0).dtype
    ncells = sparse_coords[0].shape[0]

    sparse_attributes = list()
    sparse_values = list()

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
        except Exception as exc:
            raise ValueError(f"NumPy array conversion check failed for attr '{name}'") from exc

        # if dtype is ASCII, ensure all characters are valid
        if attr.dtype == "|S0":
            try:
                np.asarray(attr_val, dtype=np.bytes_)
            except Exception as exc:
                raise TileDBError(f'Attr\'s dtype is "ascii" but attr_val contains invalid ASCII characters')

        if attr_val.size != ncells:
           raise ValueError("value length ({}) does not match "
                             "coordinate length ({})".format(attr_val.size, ncells))
        sparse_attributes.append(attr._internal_name)
        sparse_values.append(attr_val)

    if (len(sparse_attributes) != len(val.keys())) \
        or (len(sparse_values) != len(val.values())):
        raise TileDBError("Sparse write input data count does not match number of attributes")

    _write_array(
        self.ctx.ptr, self.ptr, self,
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

    def query(self, attrs=None, attr_cond=None, dims=None, index_col=True,
              coords=None, order='U', use_arrow=None, return_arrow=None, return_incomplete=False):
        """
        Construct a proxy Query object for easy subarray queries of cells
        for an item or region of the array across one or more attributes.

        Optionally subselect over attributes, return dense result coordinate values,
        and specify a layout a result layout / cell-order.

        :param attrs: the SparseArray attributes to subselect over.
            If attrs is None (default) all array attributes will be returned.
            Array attributes can be defined by name or by positional index.
        :param attr_cond: the QueryCondition to filter attributes on.
        :param dims: the SparseArray dimensions to subselect over. If dims is None (default)
            then all dimensions are returned, unless coords=False.
        :param index_col: For dataframe queries, override the saved index information,
            and only set specified index(es) in the final dataframe, or None.
        :param coords: (deprecated) if True, return array of coordinate value (default False).
        :param order: 'C', 'F', or 'G' (row-major, col-major, tiledb global order)
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
        if not self.isopen:
            raise TileDBError("SparseArray is not opened")

        # backwards compatibility
        _coords = coords
        if dims is False:
            _coords = False
        elif dims is None and coords is None:
            _coords = True

        return Query(self, attrs=attrs, attr_cond=attr_cond, dims=dims,
                     coords=_coords, index_col=index_col, order=order,
                     use_arrow=use_arrow, return_arrow=return_arrow,
                     return_incomplete=return_incomplete)

    def subarray(self, selection, coords=True, attrs=None, order=None):
        """
        Retrieve dimension and data cells for an item or region of the array.

        Optionally subselect over attributes, return sparse result coordinate values,
        and specify a layout a result layout / cell-order.

        :param selection: tuple of scalar and/or slice objects
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

        if not self.isopen or self.mode != 'r':
            raise TileDBError("SparseArray is not opened for reading")

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
        return self._read_sparse_subarray(subarray, attr_names, layout)

    def __repr__(self):
        if self.isopen:
            return "SparseArray(uri={0!r}, mode={1}, ndim={2})"\
                .format(self.uri, self.mode, self.schema.ndim)
        else:
            return "SparseArray(uri={0!r}, mode=closed)".format(self.uri)

    cdef _read_sparse_subarray(self, list subarray, list attr_names,
                               tiledb_layout_t layout):
        cdef object out = OrderedDict()
        # all results are 1-d vectors
        cdef np.npy_intp dims[1]
        cdef Py_ssize_t nattr = len(attr_names)

        from tiledb.main import PyQuery
        q = PyQuery(self._ctx_(), self, tuple(attr_names), tuple(), <int32_t>layout, False)
        q.set_ranges([list([x]) for x in subarray])
        q.submit()

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


def consolidate(uri, key=None, Config config=None, Ctx ctx=None, timestamp=None):
    """Consolidates TileDB array fragments for improved read performance

    :param str uri: URI to the TileDB Array
    :param str key: (default None) Key to decrypt array if the array is encrypted
    :param tiledb.Config config: The TileDB Config with consolidation parameters set
    :param tiledb.Ctx ctx: (default None) The TileDB Context
    :param timestamp: (default None) If not None, vacuum the array using the given
    tuple(int, int) UNIX seconds range (inclusive)
    :rtype: str or bytes
    :return: path (URI) to the consolidated TileDB Array
    :raises TypeError: cannot convert path to unicode string
    :raises: :py:exc:`tiledb.TileDBError`

    Rather than passing the timestamp into this function, it may be set with
    the config parameters `"sm.vacuum.timestamp_start"`and
    `"sm.vacuum.timestamp_end"` which takes in a time in UNIX seconds. If both
    are set then this function's `timestamp` argument will be used.

    """
    if not ctx:
        ctx = default_ctx()

    cdef tiledb_ctx_t* ctx_ptr = ctx.ptr

    if timestamp:
        if config is None:
            config = Config()

        if not isinstance(timestamp, tuple) and len(timestamp) != 2:
            raise TypeError("'timestamp' argument expects tuple(start: int, end: int)")

        if timestamp[0] is not None:
            config["sm.consolidation.timestamp_start"] = timestamp[0]
        if timestamp[1] is not None:
            config["sm.consolidation.timestamp_end"] = timestamp[1]

    cdef tiledb_config_t* config_ptr = NULL
    if config is not None:
        config_ptr = config.ptr
    cdef bytes buri = unicode_path(uri)
    cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
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

    cdef int rc = TILEDB_OK
    with nogil:
        rc = tiledb_array_consolidate_with_key(ctx_ptr, uri_ptr, key_type, key_ptr, key_len, config_ptr)
    if rc != TILEDB_OK:
        _raise_ctx_err(ctx_ptr, rc)
    return uri


def group_create(uri, Ctx ctx=None):
    """
    Create a TileDB Group object at the specified path (URI)

    :param str uri: URI of the TileDB Group to be created
    :rtype: str
    :param tiledb.Ctx ctx: The TileDB Context
    :return: The URI of the created TileDB Group
    :raises TypeError: cannot convert path to unicode string
    :raises: :py:exc:`tiledb.TileDBError`

    """
    if not ctx:
        ctx = default_ctx()
    cdef int rc = TILEDB_OK
    cdef tiledb_ctx_t* ctx_ptr = ctx.ptr
    cdef bytes buri = unicode_path(uri)
    cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
    with nogil:
        rc = tiledb_group_create(ctx_ptr, uri_ptr)
    if rc != TILEDB_OK:
        check_error(ctx, rc)
    return uri


def object_type(uri, Ctx ctx=None):
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
    cdef tiledb_ctx_t* ctx_ptr = ctx.ptr
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


def remove(uri, Ctx ctx=None):
    """Removes (deletes) the TileDB object at the specified path (URI)

    :param str uri: URI of the TileDB resource
    :param tiledb.Ctx ctx: The TileDB Context
    :raises TypeError: uri cannot be converted to a unicode string
    :raises: :py:exc:`tiledb.TileDBError`

    """
    if not ctx:
        ctx = default_ctx()
    cdef int rc = TILEDB_OK
    cdef tiledb_ctx_t* ctx_ptr = ctx.ptr
    cdef bytes buri = unicode_path(uri)
    cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
    with nogil:
        rc = tiledb_object_remove(ctx_ptr, uri_ptr)
    if rc != TILEDB_OK:
        check_error(ctx, rc)
    return


def move(old_uri, new_uri, Ctx ctx=None):
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
    cdef tiledb_ctx_t* ctx_ptr = ctx.ptr
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


def ls(path, func, Ctx ctx=None):
    """Lists TileDB resources and applies a callback that have a prefix of ``path`` (one level deep).

    :param str path: URI of TileDB group object
    :param function func: callback to execute on every listed TileDB resource,\
            URI resource path and object type label are passed as arguments to the callback
    :param tiledb.Ctx ctx: TileDB context
    :raises TypeError: cannot convert path to unicode string
    :raises: :py:exc:`tiledb.TileDBError`

    """
    if not ctx:
        ctx = default_ctx()
    cdef bytes bpath = unicode_path(path)
    check_error(ctx,
                tiledb_object_ls(ctx.ptr, bpath, walk_callback, <void*> func))
    return


def walk(path, func, order="preorder", Ctx ctx=None):
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
    check_error(ctx,
                tiledb_object_walk(ctx.ptr, bpath, walk_order, walk_callback, <void*> func))
    return


cdef class FileHandle(object):
    """
    Wraps a TileDB VFS file handle object

    Instances of this class are returned by TileDB VFS methods and are not instantiated directly
    """

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            tiledb_vfs_fh_free(&self.ptr)


    @staticmethod
    cdef from_ptr(VFS vfs, unicode uri, tiledb_vfs_fh_t* fh_ptr):
        """Constructs a FileHandle class instance from a URI and a tiledb_vfs_fh_t pointer"""
        assert(fh_ptr != NULL)
        cdef FileHandle fh = FileHandle.__new__(FileHandle)
        fh.vfs = vfs
        fh.uri = uri
        fh.ptr = fh_ptr
        return fh

    cpdef closed(self):
        """Returns true if the file handle is closed"""
        cdef Ctx ctx = self.vfs.ctx
        cdef int isclosed = 0
        check_error(ctx,
                    tiledb_vfs_fh_is_closed(ctx.ptr, self.ptr, &isclosed))
        return bool(isclosed)


cdef class VFS(object):
    """TileDB VFS class

    Encapsulates the TileDB VFS module instance with a specific configuration (config).

    :param tiledb.Ctx ctx: The TileDB Context
    :param config: Override `ctx` VFS configurations with updated values in config.
    :type config: tiledb.Config or dict

    """

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            tiledb_vfs_free(&self.ptr)

    def __init__(self, Config config=None, Ctx ctx=None):
        if not ctx:
            ctx = default_ctx()
        cdef Config _config = Config(ctx.config())
        if config is not None:
            if isinstance(config, Config):
                _config = config
            else:
                _config.update(config)
        cdef tiledb_vfs_t* vfs_ptr = NULL
        check_error(ctx,
                    tiledb_vfs_alloc(ctx.ptr, _config.ptr, &vfs_ptr))
        self.ctx = ctx
        self.ptr = vfs_ptr

    def create_bucket(self, uri):
        """Create an object store bucket at the given URI

        :param str uri: full URI of bucket resource to be created.
        :rtype: str
        :returns: created bucket URI
        :raises TypeError: cannot convert `uri` to unicode string
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef bytes buri = unicode_path(uri)
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_vfs_t* vfs_ptr = self.ptr
        cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_vfs_create_bucket(ctx_ptr, vfs_ptr, uri_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return uri

    def remove_bucket(self, uri):
        """Remove an object store bucket at the given URI

        :param str uri: URI of bucket resource to be removed.
        :raises TypeError: cannot convert `uri` to unicode string
        :raises: :py:exc:`tiledb.TileDBError`

        ..note:
            Consistency is not enforced for bucket removal
            so although this function will return immediately on success,
            the actual removal of the bucket make take some (indeterminate) amount of time.

        """
        cdef bytes buri = unicode_path(uri)
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_vfs_t* vfs_ptr = self.ptr
        cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_vfs_remove_bucket(ctx_ptr, vfs_ptr, uri_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return uri

    def empty_bucket(self, uri):
        """Empty an object store bucket of all objects at the given URI

        This function blocks until all objects are verified to be removed from the given bucket.

        :param str uri: URI of bucket resource to be emptied
        :raises TypeError: cannot convert `uri` to unicode string
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef bytes buri = unicode_path(uri)
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_vfs_t* vfs_ptr = self.ptr
        cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_vfs_empty_bucket(ctx_ptr, vfs_ptr, uri_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return

    def is_empty_bucket(self, uri):
        """Returns true if the object store bucket is empty (contains no objects).

        If the bucket is versioned, this returns the status of the latest bucket version state.

        :param str uri: URI of bucket resource
        :rtype: bool
        :return: True if bucket at given URI is empty, False otherwise
        :raises TypeError: cannot convert `uri` to unicode string
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef bytes buri = unicode_path(uri)
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_vfs_t* vfs_ptr = self.ptr
        cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
        cdef int isempty = 0
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_vfs_is_empty_bucket(ctx_ptr, vfs_ptr, uri_ptr, &isempty)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return bool(isempty)

    def is_bucket(self, uri):
        """Returns True if the URI resource is a valid object store bucket

        :param str uri: URI of bucket resource
        :rtype: bool
        :return: True if given URI is a valid object store bucket, False otherwise
        :raises TypeError: cannot convert `uri` to unicode string
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef bytes buri = unicode_path(uri)
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_vfs_t* vfs_ptr = self.ptr
        cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
        cdef int is_bucket = 0
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_vfs_is_bucket(ctx_ptr, vfs_ptr, uri_ptr, &is_bucket)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return bool(is_bucket)

    def create_dir(self, uri):
        """Create a VFS directory at the given URI

        :param str uri: URI of directory to be created
        :rtype: str
        :return: URI of created VFS directory
        :raises TypeError: cannot convert `uri` to unicode string
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef bytes buri = unicode_path(uri)
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_vfs_t* vfs_ptr = self.ptr
        cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_vfs_create_dir(ctx_ptr, vfs_ptr, uri_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return uri

    def is_dir(self, uri):
        """Returns True if the given URI is a VFS directory object

        :param str uri: URI of the directory resource
        :rtype: bool
        :return: True if `uri` is a VFS directory, False otherwise
        :raises TypeError: cannot convert `uri` to unicode string
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef bytes buri = unicode_path(uri)
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_vfs_t* vfs_ptr = self.ptr
        cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
        cdef int is_dir = 0
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_vfs_is_dir(ctx_ptr, vfs_ptr, uri_ptr, &is_dir)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return bool(is_dir)

    def remove_dir(self, uri):
        """Removes a VFS directory at the given URI

        :param str uri: URI of the directory resource to remove
        :raises TypeError: cannot convert `uri` to unicode string
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef bytes buri = unicode_path(uri)
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_vfs_t* vfs_ptr = self.ptr
        cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_vfs_remove_dir(ctx_ptr, vfs_ptr, uri_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return

    def is_file(self, uri):
        """Returns True if the given URI is a VFS file object

        :param str uri: URI of the file resource
        :rtype: bool
        :return: True if `uri` is a VFS file, False otherwise
        :raises TypeError: cannot convert `uri` to unicode string
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef bytes buri = unicode_path(uri)
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_vfs_t* vfs_ptr = self.ptr
        cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
        cdef int is_file = 0
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_vfs_is_file(ctx_ptr, vfs_ptr, uri_ptr, &is_file)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return bool(is_file)

    def remove_file(self, uri):
        """Removes a VFS file at the given URI

        :param str uri: URI of a VFS file resource
        :raises TypeError: cannot convert `uri` to unicode string
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef bytes buri = unicode_path(uri)
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_vfs_t* vfs_ptr = self.ptr
        cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_vfs_remove_file(ctx_ptr, vfs_ptr, uri_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return

    def ls(self, uri):
        """Lists contents of directory at the given URI. Raises TileDBError
        for non-existent directory.

        :param str uri: URI of a VFS directory resource
        :raises TypeError: cannot convert `uri` to unicode string
        :raises: :py:exc:`tiledb.TileDBError`
        """
        cdef bytes buri = unicode_path(uri)
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_vfs_t* vfs_ptr = self.ptr
        cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
        cdef list result_list = list()

        cdef int rc = TILEDB_OK

        check_error(self.ctx,
                    tiledb_vfs_ls(ctx_ptr, vfs_ptr, uri_ptr, vfs_ls_callback, <void*>result_list))

        return result_list

    def file_size(self, uri):
        """Returns the size (in bytes) of a VFS file at the given URI

        :param str uri: URI of a VFS file resource
        :rtype: int
        :return: file size in number of bytes
        :raises TypeError: cannot convert `uri` to unicode string
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef bytes buri = unicode_path(uri)
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_vfs_t* vfs_ptr = self.ptr
        cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
        cdef uint64_t nbytes = 0
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_vfs_file_size(ctx_ptr, vfs_ptr, uri_ptr, &nbytes)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return int(nbytes)

    def dir_size(self, uri):
        """Returns the size (in bytes) of a VFS directory at the given URI

        :param str uri: URI of a VFS directory resource
        :rtype: int
        :return: dir size in number of bytes
        :raises TypeError: cannot convert `uri` to unicode string
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef bytes buri = unicode_path(uri)
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_vfs_t* vfs_ptr = self.ptr
        cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
        cdef uint64_t nbytes = 0
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_vfs_dir_size(ctx_ptr, vfs_ptr, uri_ptr, &nbytes)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return int(nbytes)

    def move_file(self, old_uri, new_uri):
        """ Moves a VFS file from old URI to new URI

        :param str old_uri: Existing VFS file or directory resource URI
        :param str new_uri: URI to move existing VFS resource to
        :param bool force: if VFS resource at `new_uri` exists, delete the resource and overwrite
        :rtype: str
        :return: new URI of VFS resource
        :raises TypeError: cannot convert `old_uri`/`new_uri` to unicode string
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef bytes bold_uri = unicode_path(old_uri)
        cdef bytes bnew_uri = unicode_path(new_uri)
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_vfs_t* vfs_ptr = self.ptr
        cdef const char* old_uri_ptr = PyBytes_AS_STRING(bold_uri)
        cdef const char* new_uri_ptr = PyBytes_AS_STRING(bnew_uri)
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_vfs_move_file(ctx_ptr, vfs_ptr, old_uri_ptr, new_uri_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return new_uri

    def move_dir(self, old_uri, new_uri):
        """ Moves a VFS dir from old URI to new URI

        :param str old_uri: Existing VFS file or directory resource URI
        :param str new_uri: URI to move existing VFS resource to
        :param bool force: if VFS resource at `new_uri` exists, delete the resource and overwrite
        :rtype: str
        :return: new URI of VFS resource
        :raises TypeError: cannot convert `old_uri`/`new_uri` to unicode string
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef bytes bold_uri = unicode_path(old_uri)
        cdef bytes bnew_uri = unicode_path(new_uri)
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_vfs_t* vfs_ptr = self.ptr
        cdef const char* old_uri_ptr = PyBytes_AS_STRING(bold_uri)
        cdef const char* new_uri_ptr = PyBytes_AS_STRING(bnew_uri)
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_vfs_move_dir(ctx_ptr, vfs_ptr, old_uri_ptr, new_uri_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return new_uri

    def copy_file(self, old_uri, new_uri):
        """ Copiies a VFS file from old URI to new URI

        :param str old_uri: Existing VFS file or directory resource URI
        :param str new_uri: URI to copy existing VFS resource to
        :param bool force: if VFS resource at `new_uri` exists, delete the resource and overwrite
        :rtype: str
        :return: new URI of VFS resource
        :raises TypeError: cannot convert `old_uri`/`new_uri` to unicode string
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef bytes bold_uri = unicode_path(old_uri)
        cdef bytes bnew_uri = unicode_path(new_uri)
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_vfs_t* vfs_ptr = self.ptr
        cdef const char* old_uri_ptr = PyBytes_AS_STRING(bold_uri)
        cdef const char* new_uri_ptr = PyBytes_AS_STRING(bnew_uri)
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_vfs_copy_file(ctx_ptr, vfs_ptr, old_uri_ptr, new_uri_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return new_uri

    def copy_dir(self, old_uri, new_uri):
        """ Copiies a VFS dir from old URI to new URI

        :param str old_uri: Existing VFS file or directory resource URI
        :param str new_uri: URI to copy existing VFS resource to
        :param bool force: if VFS resource at `new_uri` exists, delete the resource and overwrite
        :rtype: str
        :return: new URI of VFS resource
        :raises TypeError: cannot convert `old_uri`/`new_uri` to unicode string
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef bytes bold_uri = unicode_path(old_uri)
        cdef bytes bnew_uri = unicode_path(new_uri)
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_vfs_t* vfs_ptr = self.ptr
        cdef const char* old_uri_ptr = PyBytes_AS_STRING(bold_uri)
        cdef const char* new_uri_ptr = PyBytes_AS_STRING(bnew_uri)
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_vfs_copy_dir(ctx_ptr, vfs_ptr, old_uri_ptr, new_uri_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return new_uri

    def open(self, uri, mode='rb'):
        """Opens a VFS file resource for reading / writing / appends at URI

        If the file did not exist upon opening, a new file is created.

        :param str uri: URI of VFS file resource
        :param mode str: 'rb' for opening the file to read, 'wb' to write, 'ab' to append
        :rtype: FileHandle
        :return: VFS FileHandle
        :raises TypeError: cannot convert `uri` to unicode string
        :raises ValueError: invalid mode
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef tiledb_vfs_mode_t vfs_mode
        if mode == "rb":
            vfs_mode = TILEDB_VFS_READ
        elif mode == "wb":
            vfs_mode = TILEDB_VFS_WRITE
        elif mode == "ab":
            vfs_mode = TILEDB_VFS_APPEND
        else:
            raise ValueError("invalid mode {0!r}".format(mode))
        cdef bytes buri = unicode_path(uri)
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_vfs_t* vfs_ptr = self.ptr
        cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
        cdef tiledb_vfs_fh_t* fh_ptr = NULL
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_vfs_open(ctx_ptr, vfs_ptr, uri_ptr, vfs_mode, &fh_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return FileHandle.from_ptr(self, buri.decode('UTF-8'), fh_ptr)

    def close(self, FileHandle fh):
        """Closes a VFS FileHandle object

        :param FileHandle fh: An opened VFS FileHandle
        :rtype: FileHandle
        :return: closed VFS FileHandle
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_vfs_fh_t* fh_ptr = fh.ptr
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_vfs_close(ctx_ptr, fh_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return fh

    def readinto(self, FileHandle fh, const unsigned char[:] buffer, offset, nbytes):
        """Read nbytes from an opened VFS FileHandle at a given offset into a preallocated bytes buffer

        :param FileHandle fh: An opened VFS FileHandle in 'r' mode
        :param bytes buffer: A preallocated bytes buffer object
        :param int offset: offset position in bytes to read from
        :param int nbytes: number of bytes to read
        :return: bytes `buffer`
        :raises ValueError: invalid `offset` or `nbytes` values
        :raises: :py:exc:`tiledb.TileDBError`

        """
        if offset < 0:
            raise ValueError("read offset must be >= 0")
        if nbytes < 0:
            raise ValueError("read nbytes but be >= 0")
        if nbytes > len(buffer):
            raise ValueError("read buffer is smaller than nbytes")
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_vfs_fh_t* fh_ptr = fh.ptr
        cdef uint64_t _offset = offset
        cdef uint64_t _nbytes = nbytes
        cdef const unsigned char* buffer_ptr = &buffer[0]
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_vfs_read(ctx_ptr, fh_ptr, _offset, <void*>buffer_ptr, _nbytes)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        # TileDB will error if the requested bytes are not read exactly
        return nbytes

    def read(self, FileHandle fh, offset, nbytes):
        """Read nbytes from an opened VFS FileHandle at a given offset

        :param FileHandle fh: An opened VFS FileHandle in 'r' mode
        :param int offset: offset position in bytes to read from
        :param int nbytes: number of bytes to read
        :rtype: :py:func:`bytes`
        :return: read bytes
        :raises: :py:exc:`tiledb.TileDBError`

        """
        if nbytes == 0:
            return b''
        cdef Py_ssize_t _nbytes = nbytes
        cdef bytes buffer = PyBytes_FromStringAndSize(NULL, _nbytes)
        cdef Py_ssize_t res_nbytes = self.readinto(fh, buffer, offset, nbytes)
        return buffer

    def write(self, FileHandle fh, buff):
        """Writes buffer to opened VFS FileHandle

        :param FileHandle fh: An opened VFS FileHandle in 'w' mode
        :param buff: a Python object that supports the byte buffer protocol
        :raises TypeError: cannot convert buff to bytes
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef bytes buffer = bytes(buff)
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_vfs_fh_t* fh_ptr = fh.ptr
        cdef const char* buffer_ptr = PyBytes_AS_STRING(buffer)
        cdef Py_ssize_t _nbytes = PyBytes_GET_SIZE(buffer)
        assert(_nbytes >= 0)
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_vfs_write(ctx_ptr, fh_ptr,
                                  <const void*> buffer_ptr,
                                  <uint64_t> _nbytes)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return

    def sync(self, FileHandle fh):
        """Sync / flush an opened VFS FileHandle to storage backend

        :param FileHandle fh: An opened VFS FileHandle in 'w' or 'a' mode
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_vfs_fh_t* fh_ptr = fh.ptr
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_vfs_sync(ctx_ptr, fh_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return fh

    def touch(self, uri):
        """Creates an empty VFS file at the given URI

        :param str uri: URI of a VFS file resource
        :rtype: str
        :return: URI of touched VFS file
        :raises TypeError: cannot convert `uri` to unicode string
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef bytes buri = unicode_path(uri)
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_vfs_t* vfs_ptr = self.ptr
        cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_vfs_touch(ctx_ptr, vfs_ptr, uri_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return uri

    def supports(self, scheme):
        """Returns true if the given URI scheme (storage backend) is supported

        :param str scheme: scheme component of a VFS resource URI (ex. 'file' / 'hdfs' / 's3')
        :rtype: bool
        :return: True if the linked libtiledb version supports the storage backend, False otherwise
        :raises ValueError: VFS storage backend is not supported

        """
        cdef tiledb_filesystem_t fs
        cdef int supports = 0
        if scheme == "file":
            return True
        elif scheme == "s3":
            check_error(self.ctx,
                        tiledb_ctx_is_supported_fs(self.ctx.ptr, TILEDB_S3, &supports))
            return bool(supports)
        elif scheme == "azure":
            check_error(self.ctx,
                        tiledb_ctx_is_supported_fs(self.ctx.ptr, TILEDB_AZURE, &supports))
            return bool(supports)
        elif scheme == "gcs":
            check_error(self.ctx,
                        tiledb_ctx_is_supported_fs(self.ctx.ptr, TILEDB_GCS, &supports))
            return bool(supports)
        elif scheme == "hdfs":
            check_error(self.ctx,
                        tiledb_ctx_is_supported_fs(self.ctx.ptr, TILEDB_HDFS, &supports))
            return bool(supports)
        else:
            raise ValueError("unsupported vfs scheme '{0!s}://'".format(scheme))

    def config(self):
        """Returns the Config instance associated with the VFS."""
        cdef tiledb_config_t* config_ptr = NULL
        check_error(self.ctx,
                    tiledb_vfs_get_config(self.ctx.ptr, self.ptr, &config_ptr))
        return Config.from_ptr(config_ptr)


class FileIO(io.RawIOBase):
    def __init__(self, VFS vfs, uri, mode="rb"):
        self.fh = vfs.open(uri, mode=mode)
        self.vfs = vfs
        self._offset = 0
        self._closed = False
        self._readonly = True
        if mode == "rb":
            try:
                self._nbytes = vfs.file_size(uri)
            except:
                raise IOError("URI {0!r} is not a valid file")
            self._readonly = True
        elif mode == "wb":
            self._readonly = False
            self._nbytes = 0
        else:
            raise ValueError("invalid mode {0!r}".format(mode))
        self._mode = mode
        return

    def __len__(self):
        return self._nbytes

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.flush()
        self.close()
        return

    @property
    def mode(self):
        return self._mode

    @property
    def closed(self):
        return self.fh.closed()

    def close(self):
        self.vfs.close(self.fh)

    def flush(self):
        self.vfs.sync(self.fh)

    def seekable(self):
        return True

    def readable(self):
        return self._readonly

    def seek(self, offset, whence=0):
        if not isinstance(offset, (int, long)):
            raise TypeError(f"Offset must be an integer or None (got {safe_repr(offset)})")
        if whence == 0:
            if offset < 0:
                raise ValueError("offset must be a positive or zero value when SEEK_SET")
            self._offset = offset
        elif whence == 1:
            self._offset += offset
        elif whence == 2:
            self._offset = self._nbytes + offset
        else:
            raise ValueError('whence must be equal to SEEK_SET, SEEK_START, SEEK_END')
        if self._offset < 0:
            self._offset = 0
        elif self._offset > self._nbytes:
            self._offset = self._nbytes

        return self._offset

    def tell(self):
        return self._offset

    def writable(self):
        return not self._readonly

    def read(self, size=-1):
        if not isinstance(size, (int, long)):
            raise TypeError(f"size must be an integer or None (got {safe_repr(size)})")
        if self._mode == "wb":
            raise IOError("Cannot read from write-only FileIO handle")
        if self.closed:
            raise IOError("Cannot read from closed FileIO handle")
        nbytes_remaining = self._nbytes - self._offset
        cdef Py_ssize_t nbytes
        if size < 0:
            nbytes = nbytes_remaining
        elif size > nbytes_remaining:
            nbytes = nbytes_remaining
        else:
            nbytes = size

        if nbytes == 0:
            return b''

        cdef bytes buff = PyBytes_FromStringAndSize(NULL, nbytes)
        self.vfs.readinto(self.fh, buff, self._offset, nbytes)
        self._offset += nbytes
        return buff

    def read1(self, size=-1):
        return self.read(size)

    def readall(self):
        if self._mode == "wb":
            raise IOError("cannot read from a write-only FileIO handle")
        if self.closed:
            raise IOError("cannot read from closed FileIO handle")
        cdef Py_ssize_t nbytes = self._nbytes - self._offset
        if nbytes == 0:
            return PyBytes_FromStringAndSize(NULL, 0)
        cdef bytes buff = PyBytes_FromStringAndSize(NULL, nbytes)
        self.vfs.readinto(self.fh, buff, self._offset, nbytes)
        self._offset += nbytes
        return buff

    def readinto(self, buff):
        if self._mode == "wb":
            raise IOError("cannot read from a write-only FileIO handle")
        if self.closed:
            raise IOError("cannot read from closed FileIO handle")
        nbytes = len(buff)
        if nbytes > self._nbytes:
            nbytes = self._nbytes
        if nbytes == 0:
            return 0
        self.vfs.readinto(self.fh, buff, self._offset, nbytes)
        self._offset += nbytes

        # RawIOBase contract is to return the number of bytes read
        return nbytes

    def write(self, buff):
        if not self.writable():
            raise IOError("cannot write to read-only FileIO handle")
        if isinstance(buff, str):
            buff = buff.encode()
        nbytes = len(buff)
        self.vfs.write(self.fh, buff)
        self._nbytes += nbytes
        self._offset += nbytes
        return nbytes

def vacuum(uri, Config config=None, Ctx ctx=None, timestamp=None):
    """
    Vacuum underlying array fragments after consolidation.

    :param str uri: URI of array to be vacuumed
    :param config: Override the context configuration for vacuuming.
        Defaults to None, inheriting the context parameters.
    :param (ctx: tiledb.Ctx, optional): Context. Defaults to
        `tiledb.default_ctx()`.
    :param (int, int) timestamp: (default None) If not None, vacuum the array
        using the given range (inclusive)
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
    >>> len(paths) # should be 13 (3 base files + 2*5 fragment+ok files)
    13
    >>> () ; tiledb.consolidate(path) ; () # doctest:+ELLIPSIS
    (...)
    >>> tiledb.vacuum(path)
    >>> paths = tiledb.VFS().ls(path)
    >>> len(paths) # should now be 5 (3 base files + 2 fragment+ok files)
    5

    """
    cdef tiledb_ctx_t* ctx_ptr = NULL
    cdef tiledb_config_t* config_ptr = NULL

    if not ctx:
        ctx = default_ctx()

    if timestamp:
        if config is None:
            config = Config()

        if not isinstance(timestamp, tuple) and len(timestamp) != 2:
            raise TypeError("'timestamp' argument expects tuple(start: int, end: int)")

        if timestamp[0] is not None:
            config["sm.vacuum.timestamp_start"] = timestamp[0]
        if timestamp[1] is not None:
            config["sm.vacuum.timestamp_end"] = timestamp[1]

    ctx_ptr = ctx.ptr
    config_ptr = config.ptr if config is not None else NULL
    cdef bytes buri = unicode_path(uri)
    cdef const char* uri_ptr = PyBytes_AS_STRING(buri)

    check_error(ctx, tiledb_array_vacuum(ctx_ptr, uri_ptr, config_ptr))
