IF TILEDBPY_MODULAR:
    include "common.pxi"
    from .libtiledb import *
    from .libtiledb cimport *

import weakref
from collections.abc import MutableMapping

from cython.operator cimport dereference as deref
from libcpp.limits cimport numeric_limits


cdef class PackedBuffer:
    cdef bytes data
    cdef tiledb_datatype_t tdbtype
    cdef uint32_t value_num

    def __init__(self, data, tdbtype, value_num):
        self.data = data
        self.tdbtype = tdbtype
        self.value_num = value_num


cdef PackedBuffer pack_metadata_val(value):
    if isinstance(value, bytes):
        return PackedBuffer(value, TILEDB_CHAR, len(value))

    if isinstance(value, str):
        value = value.encode('UTF-8')
        return PackedBuffer(value, TILEDB_STRING_UTF8, len(value))

    if not isinstance(value, (list, tuple)):
        value = (value,)

    if not value:
        # special case for empty values
        return PackedBuffer(b'', TILEDB_INT32, 0)

    val0 = value[0]
    if not isinstance(val0, (int, float)):
        raise TypeError(f"Unsupported item type '{type(value)}'")

    cdef:
        uint32_t value_num = len(value)
        tiledb_datatype_t tiledb_type = TILEDB_INT64 if isinstance(val0, int) else TILEDB_FLOAT64
        bytearray data = bytearray(value_num * tiledb_datatype_size(tiledb_type))
        char[:] data_view = data
        Py_ssize_t pack_idx = 0
        double * double_ptr
        int64_t * int64_ptr

    if tiledb_type == TILEDB_INT64:
        int64_ptr = <int64_t*>&data_view[0]
        while pack_idx < value_num:
            # TODO ideally we would support numpy scalars here
            value_item = value[pack_idx]
            if not isinstance(value_item, int):
                raise TypeError(f"Mixed-type sequences are not supported: {value}")
            int64_ptr[pack_idx] = value_item
            pack_idx += 1
    else:
        double_ptr = <double*>&data_view[0]
        while pack_idx < value_num:
            # TODO ideally we would support numpy scalars here
            value_item = value[pack_idx]
            if not isinstance(value_item, float):
                raise TypeError(f"Mixed-type sequences are not supported: {value}")
            double_ptr[pack_idx] = value_item
            pack_idx += 1

    return PackedBuffer(bytes(data), tiledb_type, value_num)


cdef object unpack_metadata_val(
        tiledb_datatype_t value_type, uint32_t value_num, const char* value_ptr
    ):
    assert value_num != 0, "internal error: unexpected value_num==0"

    if value_type == TILEDB_STRING_UTF8:
        return value_ptr[:value_num].decode('UTF-8')  if value_ptr != NULL else ''

    if value_type == TILEDB_CHAR:
        return value_ptr[:value_num] if value_ptr != NULL else b''

    if value_ptr == NULL:
        return ()

    unpacked = [None] * value_num
    cdef uint64_t itemsize = tiledb_datatype_size(value_type)
    for i in range(value_num):
        if value_type == TILEDB_INT64:
            unpacked[i] = deref(<int64_t *> value_ptr)
        elif value_type == TILEDB_FLOAT64:
            unpacked[i] = deref(<double *> value_ptr)
        elif value_type == TILEDB_FLOAT32:
            unpacked[i] = deref(<float *> value_ptr)
        elif value_type == TILEDB_INT32:
            unpacked[i] = deref(<int32_t *> value_ptr)
        elif value_type == TILEDB_UINT32:
            unpacked[i] = deref(<uint32_t *> value_ptr)
        elif value_type == TILEDB_UINT64:
            unpacked[i] = deref(<uint64_t *> value_ptr)
        elif value_type == TILEDB_INT8:
            unpacked[i] = deref(<int8_t *> value_ptr)
        elif value_type == TILEDB_UINT8:
            unpacked[i] = deref(<uint8_t *> value_ptr)
        elif value_type == TILEDB_INT16:
            unpacked[i] = deref(<int16_t *> value_ptr)
        elif value_type == TILEDB_UINT16:
            unpacked[i] = deref(<uint16_t *> value_ptr)
        else:
            raise NotImplementedError(f"TileDB datatype '{value_type}' not supported")
        value_ptr += itemsize

    # we don't differentiate between length-1 sequences and scalars
    return unpacked[0] if value_num == 1 else tuple(unpacked)


cdef put_metadata(Array array, key, value):
    cdef PackedBuffer packed_buf = pack_metadata_val(value)
    cdef const unsigned char[:] data_view = packed_buf.data
    cdef const void* data_ptr = NULL

    if packed_buf.value_num > 0:
        data_ptr = <void*>&data_view[0]

    cdef int rc = tiledb_array_put_metadata(
        array.ctx.ptr,
        array.ptr,
        PyBytes_AS_STRING(key.encode('UTF-8')),
        packed_buf.tdbtype,
        packed_buf.value_num,
        data_ptr,
    )
    if rc != TILEDB_OK:
        _raise_ctx_err(array.ctx.ptr, rc)


cdef object get_metadata(Array array, key):
    cdef:
        tiledb_datatype_t value_type
        uint32_t value_num = 0
        const char* value_ptr = NULL
        bytes key_utf8 = key.encode('UTF-8')

    cdef int32_t rc = tiledb_array_get_metadata(
        array.ctx.ptr,
        array.ptr,
        <const char*>key_utf8,
        &value_type,
        &value_num,
        <const void**>&value_ptr,
    )
    if rc != TILEDB_OK:
        _raise_ctx_err(array.ctx.ptr, rc)

    if value_ptr == NULL and value_num != 1:
        raise KeyError(key)

    return unpack_metadata_val(value_type, value_num, value_ptr)

def iter_metadata(Array array, keys_only):
    """
    Iterate over array metadata keys or (key, value) tuples

    :param array: tiledb_array_t
    :param keys_only: whether to yield just keys or values too
    """
    cdef:
        tiledb_ctx_t* ctx_ptr = array.ctx.ptr
        tiledb_array_t* array_ptr = array.ptr
        uint64_t metadata_num
        const char* key_ptr = NULL
        uint32_t key_len
        tiledb_datatype_t value_type
        uint32_t value_num
        const char* value_ptr = NULL

    cdef int32_t rc = tiledb_array_get_metadata_num(ctx_ptr, array_ptr, &metadata_num)
    if rc != TILEDB_OK:
        _raise_ctx_err(ctx_ptr, rc)

    for i in range(metadata_num):
        rc = tiledb_array_get_metadata_from_index(
            ctx_ptr,
            array_ptr,
            i,
            &key_ptr,
            &key_len,
            &value_type,
            &value_num,
            <const void**>&value_ptr,
        )
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        key = key_ptr[:key_len].decode('UTF-8')

        if keys_only:
            yield key
        elif value_ptr != NULL or value_num == 1:
            yield key, unpack_metadata_val(value_type, value_num, value_ptr)
        else:
            raise KeyError(key)


cdef class Metadata:
    def __init__(self, array):
        self.array_ref = weakref.ref(array)

    @property
    def array(self):
        assert self.array_ref() is not None, \
            "Internal error: invariant violation ([] from gc'd Array)"
        return self.array_ref()

    def __setitem__(self, key, value):
        """
        Implementation of [key] <- val (dict item assignment)

        :param key: key to set
        :param value: corresponding value
        :return: None
        """
        if not isinstance(key, str):
            raise TypeError(f"Unexpected key type '{type(key)}': expected str")

        put_metadata(self.array, key, value)

    def __getitem__(self, key):
        """
        Implementation of [key] -> val (dict item retrieval)
        :param key:
        :return:
        """
        if not isinstance(key, str):
            raise TypeError(f"Unexpected key type '{type(key)}': expected str")

        return get_metadata(self.array, key)

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError(f"Unexpected key type '{type(key)}': expected str")

        cdef:
            tiledb_ctx_t* ctx_ptr = (<Array>self.array).ctx.ptr
            tiledb_array_t* array_ptr = (<Array>self.array).ptr
            bytes key_utf8 = key.encode('UTF-8')
            tiledb_datatype_t value_type
            int32_t has_key

        cdef int32_t rc = tiledb_array_has_metadata_key(
            ctx_ptr,
            array_ptr,
            <const char*>key_utf8,
            &value_type,
            &has_key,
        )
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        return bool(has_key)

    def consolidate(self):
        """
        Consolidate array metadata. Array must be closed.

        :return:
        """
        # TODO: ensure that the array is not x-locked?
        cdef:
            uint32_t rc = 0
            tiledb_ctx_t* ctx_ptr = (<Array?> self.array).ctx.ptr
            tiledb_encryption_type_t key_type = TILEDB_NO_ENCRYPTION
            void* key_ptr = NULL
            uint32_t key_len = 0
            bytes bkey
            bytes buri = unicode_path(self.array.uri)
            str key = (<Array?>self.array).key

        if key is not None:
            if isinstance(key, str):
                bkey = key.encode('ascii')
            else:
                bkey = bytes(self.array.key)
            key_type = TILEDB_AES_256_GCM
            key_ptr = <void *> PyBytes_AS_STRING(bkey)
            #TODO: unsafe cast here ssize_t -> uint64_t
            key_len = <uint32_t> PyBytes_GET_SIZE(bkey)

        rc = tiledb_array_consolidate_metadata_with_key(
                ctx_ptr,
                buri,
                key_type,
                key_ptr,
                key_len,
                NULL)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

    def __delitem__(self, key):
        """
        Remove key from metadata.

        **Example:**

        >>> # given A = tiledb.open(uri, ...)
        >>> del A.meta['key']

        :param key:
        :return:
        """
        cdef:
            tiledb_ctx_t* ctx_ptr = (<Array>self.array).ctx.ptr
            tiledb_array_t* array_ptr = (<Array>self.array).ptr

        key_utf8 = key.encode('UTF-8')
        cdef int32_t rc = tiledb_array_delete_metadata(ctx_ptr, array_ptr,
                                                       <const char*>key_utf8)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

    get = MutableMapping.get
    update = MutableMapping.update

    def setdefault(self, key, default=None):
        raise NotImplementedError("Metadata.setdefault requires read-write access to array")

    def pop(self, key, default=None):
        raise NotImplementedError("Metadata.pop requires read-write access to array")

    def popitem(self):
        raise NotImplementedError("Metadata.popitem requires read-write access to array")

    def clear(self):
        raise NotImplementedError("Metadata.clear requires read-write access to array")

    def __len__(self):
        cdef:
            tiledb_ctx_t* ctx_ptr = (<Array>self.array).ctx.ptr
            tiledb_array_t* array_ptr = (<Array>self.array).ptr
            uint64_t num

        cdef int32_t rc = tiledb_array_get_metadata_num(ctx_ptr, array_ptr, &num)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        return <int>num

    def __iter__(self):
        return iter_metadata(self.array, keys_only=True)

    def keys(self):
        """
        Return metadata keys as list.

        :return: List of keys
        """
        # TODO this should be an iterator/view
        return list(self)

    def values(self):
        """
        Return metadata values as list.

        :return: List of values
        """
        # TODO this should be an iterator
        return [v for k, v in iter_metadata(self.array, keys_only=False)]

    def items(self):
        # TODO this should be an iterator
        return tuple(iter_metadata(self.array, keys_only=False))

    def _set_numpy(self, key, np.ndarray arr, datatype = None):
        """
        Escape hatch to directly set meta key-value from a NumPy array.
        Key type and array dimensionality are checked, but no other type-checking
        is done. Not intended for routine use.

        :param key: key
        :param arr: 1d NumPy ndarray
        :return:
        """
        cdef:
            tiledb_ctx_t* ctx_ptr = (<Array> self.array).ctx.ptr
            tiledb_array_t* array_ptr = (<Array> self.array).ptr
            int32_t rc = TILEDB_OK
            const char* key_ptr = NULL
            bytes key_utf8
            tiledb_datatype_t tiledb_type
            uint32_t value_num = 0

        if not isinstance(key, str):
            raise ValueError(f"Unexpected key type '{type(key)}': expected str")

        if not arr.ndim == 1:
            raise ValueError("Expected 1d NumPy array")

        if arr.nbytes > numeric_limits[uint32_t].max():
            raise ValueError("Byte count exceeds capacity of uint32_t")

        if datatype is None:
            tiledb_type = dtype_to_tiledb(arr.dtype)
            value_num = len(arr)
        else:
            tiledb_type = datatype
            value_num = int(arr.nbytes / tiledb_datatype_size(tiledb_type))

        key_utf8 = key.encode('UTF-8')
        key_cstr = PyBytes_AS_STRING(key_utf8)

        cdef const char* data_ptr = <const char*>np.PyArray_DATA(arr)

        rc = tiledb_array_put_metadata(
                ctx_ptr,
                array_ptr,
                key_cstr,
                tiledb_type,
                value_num,
                data_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
