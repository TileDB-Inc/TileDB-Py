#!python
#cython: embedsignature=True
#cython: auto_pickle=False

from cpython.pycapsule cimport PyCapsule_GetPointer, PyCapsule_IsValid, PyCapsule_New
from cpython.version cimport PY_MAJOR_VERSION
from .domain_indexer import DomainIndexer

include "common.pxi"
include "indexing.pyx"
include "libmetadata.pyx"
import io
import warnings
import collections.abc
from collections import OrderedDict
from json import dumps as json_dumps, loads as json_loads

from ._generated_version import version_tuple as tiledbpy_version
from .cc import TileDBError
from .ctx import Config, Ctx, default_ctx
from .vfs import VFS
from .sparse_array import SparseArrayImpl
from .dense_array import DenseArrayImpl

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
    cdef tiledb_array_t* array_ptr = <tiledb_array_t*>(<Array>tiledb_array).ptr
    cdef dict fragment_info = (<Array>tiledb_array).last_fragment_info
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


cpdef check_error(object ctx, int rc):
    cdef tiledb_ctx_t* ctx_ptr = <tiledb_ctx_t*>PyCapsule_GetPointer(
            ctx.__capsule__(), "ctx")
    _raise_ctx_err(ctx_ptr, rc)


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


###############################################################################
#                                                                             #
#    CLASS DEFINITIONS                                                        #
#                                                                             #
###############################################################################

from .array import _tiledb_datetime_extent, index_as_tuple, replace_ellipsis, replace_scalars_slice, check_for_floats, index_domain_subarray

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
    cdef const char* key_ptr = NULL
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
        key_ptr = <const char *> PyBytes_AS_STRING(bkey)
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

    cdef tiledb_config_t* config_ptr = NULL
    cdef tiledb_error_t* err_ptr = NULL
    if key is not None:
        rc = tiledb_config_alloc(&config_ptr, &err_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        rc = tiledb_config_set(config_ptr, "sm.encryption_type", "AES_256_GCM", &err_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        rc = tiledb_config_set(config_ptr, "sm.encryption_key", key_ptr, &err_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        try:
          # note: tiledb_array_set_config copies the config
          rc = tiledb_array_set_config(ctx_ptr, array_ptr, config_ptr)
          if rc != TILEDB_OK:
              _raise_ctx_err(ctx_ptr, rc)
        finally:
          tiledb_config_free(&config_ptr)

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
       rc = tiledb_array_open(ctx_ptr, array_ptr, query_type)

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

        cdef tiledb_array_schema_t* array_schema_ptr = NULL
        try:
            rc = TILEDB_OK
            with nogil:
                rc = tiledb_array_get_schema(ctx_ptr, array_ptr, &array_schema_ptr)
            if rc != TILEDB_OK:
              _raise_ctx_err(ctx_ptr, rc)
            from .array_schema import ArraySchema
            schema = ArraySchema.from_capsule(ctx, PyCapsule_New(array_schema_ptr, "schema", NULL))
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
        self.pyquery = None

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
    def create(cls, uri, schema, key=None, overwrite=False, ctx=None):
        """Creates a TileDB Array at the given URI

        :param str uri: URI at which to create the new empty array.
        :param ArraySchema schema: Schema for the array
        :param str key: (default None) Encryption key to use for array
        :param bool overwrite: (default False) Overwrite the array if it already exists
        :param Ctx ctx: (default None) Optional TileDB Ctx used when creating the array,
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
        cdef tiledb_array_schema_t* schema_ptr = <tiledb_array_schema_t *>PyCapsule_GetPointer(
            schema.__capsule__(), "schema")

        cdef bytes bkey
        cdef tiledb_encryption_type_t key_type = TILEDB_NO_ENCRYPTION
        cdef const char* key_ptr = NULL
        cdef unsigned int key_len = 0

        cdef tiledb_config_t* config_ptr = NULL
        cdef tiledb_error_t* err_ptr = NULL
        cdef int rc = TILEDB_OK

        if key is not None:
            if isinstance(key, str):
                bkey = key.encode('ascii')
            else:
                bkey = bytes(key)
            key_type = TILEDB_AES_256_GCM
            key_ptr = <const char *> PyBytes_AS_STRING(bkey)
            #TODO: unsafe cast here ssize_t -> uint64_t
            key_len = <unsigned int> PyBytes_GET_SIZE(bkey)

            rc = tiledb_config_alloc(&config_ptr, &err_ptr)
            if rc != TILEDB_OK:
                _raise_ctx_err(ctx_ptr, rc)

            rc = tiledb_config_set(config_ptr, "sm.encryption_type", "AES_256_GCM", &err_ptr)
            if rc != TILEDB_OK:
                _raise_ctx_err(ctx_ptr, rc)

            rc = tiledb_config_set(config_ptr, "sm.encryption_key", key_ptr, &err_ptr)
            if rc != TILEDB_OK:
                _raise_ctx_err(ctx_ptr, rc)
            rc = tiledb_ctx_alloc(config_ptr, &ctx_ptr)
            if rc != TILEDB_OK:
                _raise_ctx_err(ctx_ptr, rc)

        if overwrite:
            from .highlevel import object_type
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
            rc = tiledb_array_create(ctx_ptr, uri_ptr, schema_ptr)
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

            from . import DenseArray, SparseArray
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
        (<Array>new_array_typed).ptr = array_ptr
        (<Array>new_array_typed)._isopen = True
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
        if timestamp is not None:
            _timestamp = <uint64_t> timestamp
            rc = tiledb_array_set_open_timestamp_start(ctx_ptr, array_ptr, _timestamp)
            if rc != TILEDB_OK:
                _raise_ctx_err(ctx_ptr, rc)

        with nogil:
            rc = tiledb_array_reopen(ctx_ptr, array_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return

    @property
    def pyquery(self):
        return self.pyquery

    @pyquery.setter
    def pyquery(self, value):
        self.pyquery = value

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

    def enum(self, name):
        """
        Return the Enumeration from the attribute name.

        :param name: attribute name
        :type key: str
        :rtype: `Enumeration`
        """
        cdef tiledb_ctx_t* ctx_ptr = safe_ctx_ptr(self.ctx)
        cdef tiledb_array_t* array_ptr = self.ptr
        cdef bytes bname = unicode_path(name)
        cdef const char* name_ptr = PyBytes_AS_STRING(bname)
        cdef tiledb_enumeration_t* enum_ptr = NULL
        rc = tiledb_array_get_enumeration(ctx_ptr, array_ptr, name_ptr, &enum_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        from .enumeration import Enumeration
        return Enumeration.from_capsule(self.ctx, PyCapsule_New(enum_ptr, "enum", NULL))

    def delete_fragments(self_or_uri, timestamp_start, timestamp_end, ctx=None):
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

        >>> tiledb.Array.delete_fragments(path, 2, 2)

        >>> with tiledb.open(path, 'r') as A:
        ...     A[:]
        array([0., 0., 0., 0.])

        """
        cdef tiledb_ctx_t* ctx_ptr
        cdef tiledb_array_t* array_ptr
        cdef tiledb_query_t* query_ptr
        cdef bytes buri
        cdef int rc = TILEDB_OK

        if isinstance(self_or_uri, str):
            uri = self_or_uri
            if not ctx:
                ctx = default_ctx()

            ctx_ptr = safe_ctx_ptr(ctx)
            buri = uri.encode('UTF-8')

            rc = tiledb_array_delete_fragments_v2(
                    ctx_ptr,
                    buri,
                    timestamp_start,
                    timestamp_end
            )
        else:
            # TODO: Make this method static and entirely remove the conditional.
            raise TypeError(
                "The `tiledb.Array.delete_fragments` instance method is deprecated and removed. Use the static method with the same name instead.")
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

        cdef int rc = TILEDB_OK

        rc = tiledb_array_delete(ctx_ptr, buri)
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
        :param fragment_uris: (default None) Consolidate the array using a list of fragment _names_ (note: the `__ts1_ts2_<label>_<ver>` fragment name form alone, not the full path(s))
        :param timestamp: (default None) If not None, consolidate the array using the given tuple(int, int) UNIX seconds range (inclusive). This argument will be ignored if `fragment_uris` is passed.
        :type timestamp: tuple (int, int)
        :raises: :py:exc:`tiledb.TileDBError`

        Rather than passing the timestamp into this function, it may be set with
        the config parameters `"sm.vacuum.timestamp_start"`and
        `"sm.vacuum.timestamp_end"` which takes in a time in UNIX seconds. If both
        are set then this function's `timestamp` argument will be used.

        """
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
                const char* key_ptr = NULL
                unsigned int key_len = 0
                tiledb_error_t* err_ptr = NULL

            if key is not None:
                if isinstance(key, str):
                    bkey = key.encode('ascii')
                else:
                    bkey = bytes(key)
                key_type = TILEDB_AES_256_GCM
                key_ptr = <const char *> PyBytes_AS_STRING(bkey)
                #TODO: unsafe cast here ssize_t -> uint64_t
                key_len = <unsigned int> PyBytes_GET_SIZE(bkey)

                rc = tiledb_config_alloc(&config_ptr, &err_ptr)
                if rc != TILEDB_OK:
                    _raise_ctx_err(ctx_ptr, rc)

                rc = tiledb_config_set(config_ptr, "sm.encryption_type", "AES_256_GCM", &err_ptr)
                if rc != TILEDB_OK:
                    _raise_ctx_err(ctx_ptr, rc)

                rc = tiledb_config_set(config_ptr, "sm.encryption_key", key_ptr, &err_ptr)
                if rc != TILEDB_OK:
                    _raise_ctx_err(ctx_ptr, rc)

            with nogil:
                rc = tiledb_array_consolidate(
                    ctx_ptr, array_uri_ptr, config_ptr)
            if rc != TILEDB_OK:
                _raise_ctx_err(ctx_ptr, rc)
            return uri
    
        if self.mode == 'r':
            raise TileDBError("cannot consolidate array opened in readonly mode (mode='r')")

        if not self.ctx:
            self.ctx = default_ctx()

        if fragment_uris is not None:
            if timestamp is not None:
                warnings.warn(
                    "The `timestamp` argument will be ignored and only fragments "
                    "passed to `fragment_uris` will be consolidate",
                    DeprecationWarning,
                )
            return _consolidate_uris(
                uri=self.uri, key=key, config=config, ctx=self.ctx, fragment_uris=fragment_uris)
        else:
            return _consolidate_timestamp(
                uri=self.uri, key=key, config=config, ctx=self.ctx, timestamp=timestamp)

    def upgrade_version(self, config=None):
        """
        Upgrades an array to the latest format version.

        :param config: (default None) Configuration parameters for the upgrade
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
                 np.issubdtype(array.dtype, np.str_) or
                 array.dtype == object)

    @property
    def domain_index(self):
        return self.domain_index

    @property
    def dindex(self):
        return self.domain_index

    def label_index(self, labels):
        """Retrieve data cells with multi-range, domain-inclusive indexing by label.
        Returns the cross-product of the ranges.

        Accepts a scalar, ``slice``, or list of scalars per-label for querying on the
        corresponding dimensions. For multidimensional arrays querying by labels only on
        a subset of dimensions, ``:`` should be passed in-place for any labels preceeding
        custom ranges.

        ** Example **

        >>> import tiledb, numpy as np, tempfile
        >>> from collections import OrderedDict
        >>> dim1 = tiledb.Dim("d1", domain=(1, 4))
        >>> dim2 = tiledb.Dim("d2", domain=(1, 3))
        >>> dom = tiledb.Domain(dim1, dim2)
        >>> att = tiledb.Attr("a1", dtype=np.int64)
        >>> dim_labels = {
        ...     0: {"l1": dim1.create_label_schema("decreasing", np.int64)},
        ...     1: {
        ...         "l2": dim2.create_label_schema("increasing", np.int64),
        ...         "l3": dim2.create_label_schema("increasing", np.float64),
        ...     },
        ... }
        >>> schema = tiledb.ArraySchema(domain=dom, attrs=(att,), dim_labels=dim_labels)
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     tiledb.Array.create(tmp, schema)
        ...
        ...     a1_data = np.reshape(np.arange(1, 13), (4, 3))
        ...     l1_data = np.arange(4, 0, -1)
        ...     l2_data = np.arange(-1, 2)
        ...     l3_data = np.linspace(0, 1.0, 3)
        ...
        ...     with tiledb.open(tmp, "w") as A:
        ...         A[:] = {"a1": a1_data, "l1": l1_data, "l2": l2_data, "l3": l3_data}
        ...
        ...     with tiledb.open(tmp, "r") as A:
        ...         np.testing.assert_equal(
        ...             A.label_index(["l1"])[3:4],
        ...             OrderedDict({"l1": [4, 3], "a1": [[1, 2, 3], [4, 5, 6]]}),
        ...         )
        ...         np.testing.assert_equal(
        ...             A.label_index(["l1", "l3"])[2, 0.5:1.0],
        ...             OrderedDict(
        ...                 {"l3": [0.5, 1.0], "l1": [2], "a1": [[8, 9]]}
        ...             ),
        ...         )
        ...         np.testing.assert_equal(
        ...             A.label_index(["l2"])[:, -1:0],
        ...             OrderedDict(
        ...                 {"l2": [-1, 0],
        ...                 "a1": [[1, 2], [4, 5], [7, 8], [10, 11]]},
        ...             ),
        ...         )
        ...         np.testing.assert_equal(
        ...             A.label_index(["l3"])[:, 0.5:1.0],
        ...             OrderedDict(
        ...                 {"l3": [0.5, 1.],
        ...                 "a1": [[2, 3], [5, 6], [8, 9], [11, 12]]},
        ...             ),
        ...         )

        :param labels: List of labels to use when querying. Can only use at most one
            label per dimension.
        :param list selection: Per dimension, a scalar, ``slice``, or  list of scalars.
            Each item is iterpreted as a point (scalar) or range (``slice``) used to
            query the array on the corresponding dimension.
        :returns: dict of {'label/attribute': result}.
        :raises: :py:exc:`tiledb.TileDBError`

        """
        # Delayed to avoid circular import
        from .multirange_indexing import LabelIndexer
        return LabelIndexer(self, tuple(labels))

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
        ...    A.multi_index[1]  # doctest: +ELLIPSIS
        ...    A.multi_index[1,1]  # doctest: +ELLIPSIS
        ...    # return row 0 and 2
        ...    A.multi_index[[0,2]]  # doctest: +ELLIPSIS
        ...    # return rows 0 and 2 intersecting column 2
        ...    A.multi_index[[0,2], 2]  # doctest: +ELLIPSIS
        ...    # return rows 0:2 intersecting columns 0:2
        ...    A.multi_index[slice(0,2), slice(0,2)]  # doctest: +ELLIPSIS
        OrderedDict(...''... array([[0., 2., 0., 0.]])...)
        OrderedDict(...''... array([[2.]])...)
        OrderedDict(...''... array([[1., 0., 0., 0.],
                [0., 0., 3., 0.]])...)
        OrderedDict(...''... array([[0.],
                [3.]])...)
        OrderedDict(...''... array([[1., 0., 0.],
                [0., 2., 0.],
                [0., 0., 3.]])...)

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
        from .main import PyQuery
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

        if return_incomplete and not array.schema.sparse:
            raise TileDBError("Incomplete queries are only supported for sparse arrays at this time")

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
    
    def agg(self, aggs):
        """
        Calculate an aggregate operation for a given attribute. Available 
        operations are sum, min, max, mean, count, and null_count (for nullable
        attributes only). Aggregates may be combined with other query operations 
        such as query conditions and slicing.

        The input may be a single operation, a list of operations, or a 
        dictionary with attribute mapping to a single operation or list of 
        operations.

        For undefined operations on max and min, which can occur when a nullable
        attribute contains only nulled data at the given coordinates or when 
        there is no data read for the given query (e.g. query conditions that do
        not match any values or coordinates that contain no data)), invalid
        results are represented as np.nan for attributes of floating point types
        and None for integer types.

        >>> import tiledb, tempfile, numpy as np
        >>> path = tempfile.mkdtemp()

        >>> with tiledb.from_numpy(path, np.arange(1, 10)) as A:
        ...     pass

        >>> # Note that tiledb.from_numpy creates anonymous attributes, so the
        >>> # name of the attribute is represented as an empty string

        >>> with tiledb.open(path, 'r') as A:
        ...     A.query().agg("sum")[:]
        45

        >>> with tiledb.open(path, 'r') as A:
        ...     A.query(cond="attr('') < 5").agg(["count", "mean"])[:]
        {'count': 9, 'mean': 2.5}

        >>> with tiledb.open(path, 'r') as A:
        ...     A.query().agg({"": ["max", "min"]})[2:7]
        {'max': 7, 'min': 3}

        :param agg: The input attributes and operations to apply aggregations on
        :returns: single value for single operation on one attribute, a dictionary
            of attribute keys associated with a single value for a single operation
            across multiple attributes, or a dictionary of attribute keys that maps
            to a dictionary of operation labels with the associated value
        """
        schema = self.array.schema
        attr_to_aggs_map = {}
        if isinstance(aggs, dict):
            attr_to_aggs_map = {
                a: (
                    tuple([aggs[a]]) 
                    if isinstance(aggs[a], str) 
                    else tuple(aggs[a])
                )
                for a in aggs
            }
        elif isinstance(aggs, str):
            attrs = tuple(schema.attr(i).name for i in range(schema.nattr))
            attr_to_aggs_map = {a: (aggs,) for a in attrs}
        elif isinstance(aggs, collections.abc.Sequence):
            attrs = tuple(schema.attr(i).name for i in range(schema.nattr))
            attr_to_aggs_map = {a: tuple(aggs) for a in attrs}

        from .aggregation import Aggregation
        return Aggregation(self, attr_to_aggs_map)

    @property
    def array(self):
        return self.array

    @property
    def attrs(self):
        """List of attributes to include in Query."""
        return self.attrs

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

    def label_index(self, labels):
        """Apply Array.label_index with query parameters."""
        from .multirange_indexing import LabelIndexer
        return LabelIndexer(self.array, tuple(labels), query=self)

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
            stats = json_loads(stats)
        if print_out:
            print(stats)
        else:
            return stats

    def submit(self):
        """An alias for calling the regular indexer [:]"""
        return self[:]

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

# point query index a tiledb array (zips) columnar index vectors
def index_domain_coords(dom, idx, check_ndim):
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
    cdef dict labels = dict()

    if not self.isopen or self.mode != 'w':
        raise TileDBError("SparseArray is not opened for writing")

    set_dims_only = val is None
    sparse_attributes = list()
    sparse_values = list()
    idx = index_as_tuple(selection)
    sparse_coords = list(index_domain_coords(self.schema.domain, idx, not set_dims_only))

    if set_dims_only:
        _write_array(
            ctx_ptr,
            self.ptr,
            self,
            None,
            sparse_coords,
            sparse_attributes,
            sparse_values,
            labels,
            nullmaps,
            self.last_fragment_info,
            True,
        )
        return

    if not isinstance(val, dict):
        if self.nattr > 1:
            raise ValueError("Expected dict-like object {name: value} for multi-attribute "
                             "array.")
        val = dict({self.attr(0).name: val})

    # Create dictionary for label names and values from the dictionary
    labels = {
        name:
        (data
        if not type(data) is np.ndarray or data.dtype is np.dtype('O')
        else np.ascontiguousarray(data, dtype=self.schema.dim_label(name).dtype))
        for name, data in val.items()
        if self.schema.has_dim_label(name)
    }

    # must iterate in Attr order to ensure that value order matches
    for attr_idx in range(self.schema.nattr):
        attr = self.attr(attr_idx)
        name = attr.name
        attr_val = val[name]

        try:
            # ensure that the value is array-convertible, for example: pandas.Series
            attr_val = np.asarray(attr_val)

            if attr.isvar:
                if attr.isnullable and name not in nullmaps:
                    nullmaps[name] = np.array(
                        [int(v is not None) for v in attr_val], dtype=np.uint8)
            else:
                if (np.issubdtype(attr.dtype, np.bytes_) 
                    and not (np.issubdtype(attr_val.dtype, np.bytes_) 
                    or attr_val.dtype == np.dtype('O'))):
                    raise ValueError("Cannot write a string value to non-string "
                                        "typed attribute '{}'!".format(name))
                
                if attr.isnullable and name not in nullmaps:
                    try:
                        nullmaps[name] = ~np.ma.masked_invalid(attr_val).mask
                    except Exception as exc:
                        nullmaps[name] = np.array(
                            [int(v is not None) for v in attr_val], dtype=np.uint8)

                    if np.issubdtype(attr.dtype, np.bytes_):
                        attr_val = np.array(["" if v is None else v for v in attr_val])
                    else:
                        attr_val = np.nan_to_num(attr_val)
                        attr_val = np.array([0 if v is None else v for v in attr_val])
                attr_val = np.ascontiguousarray(attr_val, dtype=attr.dtype)
            
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

    if (len(sparse_attributes) + len(labels) != len(val.keys())) \
        or (len(sparse_values) + len(labels) != len(val.values())):
        raise TileDBError("Sparse write input data count does not match number of attributes")

    _write_array(
        ctx_ptr,
        self.ptr,
        self,
        None,
        sparse_coords,
        sparse_attributes,
        sparse_values,
        labels,
        nullmaps,
        self.last_fragment_info,
        True,
    )
    return
