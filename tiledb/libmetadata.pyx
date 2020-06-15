IF TILEDBPY_MODULAR:
    include "common.pxi"
    from .libtiledb import *
    from .libtiledb cimport *

import weakref

from cython.operator cimport dereference as deref
from cpython.long cimport PyLong_AsLongLong
from cpython.float cimport PyFloat_AsDouble
from libc.string cimport memcpy
from libcpp.memory cimport unique_ptr, make_unique
from libcpp.vector cimport vector
from libcpp.limits cimport numeric_limits

cdef extern from "<utility>" namespace "std" nogil:
    cdef unique_ptr[vector[char]] move(unique_ptr[vector[char]])
    #cdef unique_ptr[vector[char]] make_unique(vector[char])

cdef class PackedBuffer:
    cdef bytes data
    cdef tiledb_datatype_t tdbtype
    cdef uint32_t value_num

    def __init__(self, data,tdbtype,value_num):
        self.data = data
        self.tdbtype = tdbtype
        self.value_num = value_num

cdef PackedBuffer pack_metadata_val(value):

    cdef tiledb_datatype_t tiledb_type

    if isinstance(value, (str, bytes, unicode)):
        pass
    elif not isinstance(value, (list, tuple)):
        value = (value,)

    cdef:
        char[:] char_ptr
        double[:] double_buf
        int64_t[:] int64_buf
        double* double_ptr
        int64_t* int64_ptr
        Py_ssize_t pack_idx = 0
        Py_ssize_t value_len = 0
        uint64_t itemsize = 0
        bytearray data
        char[:] buf_view
        object value_item


    # NOTE: string types must not check val0: it is a char
    if isinstance(value, unicode):
        value = value.encode('UTF-8')
        tiledb_type = TILEDB_STRING_UTF8
    elif isinstance(value, bytes):
        tiledb_type = TILEDB_CHAR
    else:
        val0 = value[0]
        if isinstance(val0, int):
            tiledb_type = TILEDB_INT64
        elif isinstance(val0, long):
            tiledb_type = TILEDB_INT64
        elif isinstance(val0, float):
            # Note: all python floats are doubles
            tiledb_type = TILEDB_FLOAT64
        elif isinstance(value, np.ndarray):
            # TODO support np.array as metadata with type tag
            raise ValueError("Unsupported type: numpy array")
        else:
            raise ValueError("Unsupported item type '{}'".format(type(value)))

    value_len = len(value)
    itemsize = tiledb_datatype_size(tiledb_type)
    data = bytearray(itemsize * len(value))
    pack_idx = 0

    if tiledb_type == TILEDB_INT64:
        buf_view = data
        int64_ptr = <int64_t*>&buf_view[0]
        for value_item in value:
            # TODO ideally we would support numpy scalars here
            if not isinstance(value_item, (int, long)):
                raise TypeError(f"Inconsistent type in 'int' list ('{type(value_item)}')")
            int64_ptr[pack_idx] = int(value_item)
            pack_idx += 1

    elif tiledb_type == TILEDB_FLOAT64:
        buf_view = data
        double_ptr = <double*>&buf_view[0]
        for value_item in value:
            # TODO ideally we would support numpy scalars here
            if not isinstance(value_item, float):
                raise TypeError(f"Inconsistent type in 'float' list ('{type(value_item)}')")
            double_ptr[pack_idx] = value_item
            pack_idx += 1

    elif tiledb_type == TILEDB_CHAR:
        # already bytes
        data = bytearray(value)
    elif tiledb_type == TILEDB_STRING_UTF8:
        # already bytes
        data = bytearray(value)
    else:
        assert False, "internal error: unhandled type in pack routine!"

    return PackedBuffer(bytes(data), tiledb_type, len(value))

cdef object unpack_metadata_val(tiledb_datatype_t value_type,
                                uint32_t value_num,
                                const char* value_ptr):

    cdef:
        double o_float64
        int64_t o_int64

    if value_num == 0:
        raise TileDBError("internal error: unexpected value_num==0")

    elif value_type == TILEDB_STRING_UTF8:
        new_obj = value_ptr[:value_num].decode('UTF-8')
        return new_obj

    elif value_type == TILEDB_CHAR:
        new_obj = bytes(value_ptr[:value_num])
        return new_obj

    elif value_num > 1:

        # unpack sequence
        # should this return tuple instead of list?
        new_obj = list()
        for i in range(0, value_num):
            item = unpack_metadata_val(value_type, 1, value_ptr)
            new_obj.append(item)
            value_ptr += tiledb_datatype_size(value_type)

        new_obj = tuple(new_obj)
        return new_obj

    elif value_type == TILEDB_INT64:
        return deref(<int64_t*>value_ptr)

    elif value_type == TILEDB_FLOAT64:
        return deref(<double*>value_ptr)

    elif value_type == TILEDB_FLOAT32:
        return deref(<float*>value_ptr)

    elif value_type == TILEDB_INT32:
        return deref(<int32_t*>value_ptr)

    elif value_type == TILEDB_UINT32:
        return deref(<uint32_t*>value_ptr)

    elif value_type == TILEDB_UINT64:
        return deref(<uint64_t*>value_ptr)

    elif value_type == TILEDB_INT8:
        return deref(<int8_t*>value_ptr)

    elif value_type == TILEDB_UINT8:
        return deref(<uint8_t*>value_ptr)

    elif value_type == TILEDB_INT16:
        return deref(<int16_t*>value_ptr)

    elif value_type == TILEDB_UINT16:
        return deref(<uint16_t*>value_ptr)

    else:
        raise NotImplementedError("unimplemented type")


def put_metadata(Array array,
                 key, value):

    cdef tiledb_array_t* array_ptr = array.ptr
    cdef tiledb_ctx_t* ctx_ptr = array.ctx.ptr

    cdef int rc = TILEDB_OK
    cdef bytes key_utf8
    cdef const char* key_ptr
    cdef tiledb_datatype_t ret_type
    cdef tiledb_datatype_t ttype
    cdef PackedBuffer packed_buf
    cdef const void* data_ptr

    key_utf8 = key.encode('UTF-8')
    key_ptr = PyBytes_AS_STRING(key_utf8)

    if (isinstance(value, (bytes, unicode)) or isinstance(value, tuple))\
            and len(value) == 0:
        # special case for empty values
        if isinstance(value, bytes):
            ttype = TILEDB_CHAR
        elif isinstance(value, unicode):
            ttype = TILEDB_STRING_UTF8
        else:
            ttype = TILEDB_INT32
        packed_buf = PackedBuffer(b'', ttype, 0)
    else:
        packed_buf = pack_metadata_val(value)
        if (len(packed_buf.data) < 1):
            raise ValueError("Unsupported zero-length metadata value")

    cdef bytes data = packed_buf.data
    cdef const unsigned char[:] data_view = data

    if packed_buf.value_num == 0:
        data_ptr = NULL
    else:
        data_ptr = <void*>&data_view[0]

    rc = tiledb_array_put_metadata(
            ctx_ptr,
            array_ptr,
            key_ptr,
            packed_buf.tdbtype,
            packed_buf.value_num,
            data_ptr)

    if rc != TILEDB_OK:
        _raise_ctx_err(ctx_ptr, rc)

    return None

cdef object get_metadata(array: Array,
                         key: unicode):

    cdef tiledb_array_t* array_ptr = array.ptr
    cdef tiledb_ctx_t* ctx_ptr = array.ctx.ptr

    cdef:
        int32_t rc
        object key_utf8
        const char* key_ptr
        uint32_t key_len
        tiledb_datatype_t value_type
        uint32_t value_num = 0
        const char* value = NULL
        bint has_key

    key_utf8 = key.encode('UTF-8')
    key_ptr = <const char*>key_utf8

    rc = tiledb_array_get_metadata(
            ctx_ptr, array_ptr,
            key_ptr,
            &value_type, &value_num, <const void**>&value)

    if rc != TILEDB_OK:
        _raise_ctx_err(ctx_ptr, rc)

    if (value == NULL):
        if value_num == 1:
            # in this case, the key exists with empty value
            if value_type == TILEDB_CHAR:
                return b''
            elif value_type == TILEDB_STRING_UTF8:
                return u''
            else:
                return ()
        raise KeyError(key)

    return unpack_metadata_val(value_type, value_num, value)


cdef object load_metadata(Array array, unpack=True):
    """
    Load array metadata dict or keys

    :param ctx: tiledb_ctx_t
    :param array: tiledb_array_t
    :param unpack: unpack the values into dictionary (True)

    :return: dict(k: v) if unpack, else list
    """

    cdef tiledb_ctx_t* ctx_ptr = array.ctx.ptr
    cdef tiledb_array_t* array_ptr = array.ptr
    cdef uint64_t metadata_num

    rc = tiledb_array_get_metadata_num(ctx_ptr, array_ptr, &metadata_num)
    if rc != TILEDB_OK:
        _raise_ctx_err(ctx_ptr, rc)

    cdef:
        const char* key_ptr = NULL
        uint32_t key_len
        tiledb_datatype_t value_type
        uint32_t value_num
        const char* value = NULL
    cdef:
        object new_obj

    if unpack:
        ret_val = dict()
    else:
        ret_val = list()

    for i in range(metadata_num):
        rc = tiledb_array_get_metadata_from_index(
            ctx_ptr, array_ptr,
            i,
            &key_ptr, &key_len,
            &value_type, &value_num, <const void**>&value)

        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        key = key_ptr[:key_len].decode('UTF-8')

        if not unpack:
            ret_val.append(key)
        else:
            # in this case, the key might exist with empty value
            if (value == NULL) and (value_num == 1):
                   # in this case, the key exists with empty value
                   if value_type == TILEDB_CHAR:
                       unpacked_val = b''
                   elif value_type == TILEDB_STRING_UTF8:
                       unpacked_val = u''
                   else:
                       unpacked_val = ()
            else:
                unpacked_val = unpack_metadata_val(value_type, value_num, value)

            if unpacked_val is None:
                raise TileDBError("internal error: no unpackable value for ", key)

            ret_val[key] = unpacked_val

    return ret_val

def len_metadata(Array array):
    cdef:
        int32_t rc
        uint64_t num

    rc = tiledb_array_get_metadata_num(
            array.ctx.ptr,
            array.ptr,
            &num)

    if rc != TILEDB_OK:
        _raise_ctx_err(array.ctx.ptr, rc)

    return <int>num

def del_metadata(Array array, key):
    cdef:
        tiledb_ctx_t* ctx_ptr = array.ctx.ptr
        const char* key_ptr
        object key_utf8
        int32_t rc

    key_utf8 = key.encode('UTF-8')
    key_ptr = <const char*>key_utf8

    rc = tiledb_array_delete_metadata(array.ctx.ptr, array.ptr, key_ptr)
    if rc != TILEDB_OK:
        _raise_ctx_err(ctx_ptr, rc)


def consolidate_metadata(Array array):

        cdef uint32_t rc = 0

        cdef:
            tiledb_encryption_type_t key_type = TILEDB_NO_ENCRYPTION
            void* key_ptr = NULL
            uint32_t key_len = 0

        cdef bytes bkey
        cdef bytes buri = unicode_path(array.uri)

        if array.key is not None:
            if isinstance(array.key, str):
                bkey = array.key.encode('ascii')
            else:
                bkey = bytes(array.key)
            key_type = TILEDB_AES_256_GCM
            key_ptr = <void *> PyBytes_AS_STRING(bkey)
            #TODO: unsafe cast here ssize_t -> uint64_t
            key_len = <uint32_t> PyBytes_GET_SIZE(bkey)

        rc = tiledb_array_consolidate_metadata_with_key(
                array.ctx.ptr,
                buri,
                key_type,
                key_ptr,
                key_len,
                NULL)

        if rc != TILEDB_OK:
            _raise_ctx_err(array.ctx.ptr, rc)


cdef class Metadata(object):
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
        if not (isinstance(key, str) or isinstance(key, unicode)):
            raise ValueError("Unexpected key type '{}': expected str "
                             "type".format(type(key)))

        put_metadata(self.array, key, value)

    def __getitem__(self, key):
        """
        Implementation of [key] -> val (dict item retrieval)
        :param key:
        :return:
        """
        if not (isinstance(key, str) or isinstance(key, unicode)):
            raise ValueError("Unexpected key type '{}': expected str "
                             "type".format(type(key)))

        # `get_metadata` expects unicode
        key = ustring(key)
        v = get_metadata(self.array, key)

        if v is None:
            raise TileDBError("Failed to unpack value for key: '{}'".format(key))

        return v

    def __contains__(self, key):
        """
        Returns True if 'key' is found in metadata store.
        Provides support for python 'in' syntax ('k in A.meta')

        :param key: Target key to check against self.
        :return:
        """

        try:
            self[key]
        except KeyError:
            return False

        return True

    def consolidate(self):
        """
        Consolidate array metadata. Array must be closed.

        :return:
        """

        # TODO: ensure that the array is not x-locked?

        cdef uint32_t rc = 0
        cdef tiledb_ctx_t* ctx_ptr = (<Array?>self.array).ctx.ptr
        cdef:
            tiledb_encryption_type_t key_type = TILEDB_NO_ENCRYPTION
            void* key_ptr = NULL
            uint32_t key_len = 0
        cdef bytes bkey

        cdef bytes buri = unicode_path(self.array.uri)
        cdef unicode key = (<Array?>self.array).key
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
        cdef tiledb_ctx_t*  ctx_ptr = (<Array>self.array).ctx.ptr
        cdef tiledb_array_t*  array_ptr = (<Array>self.array).ptr
        cdef const char* key_ptr
        cdef object key_utf8
        cdef int32_t rc

        key_utf8 = key.encode('UTF-8')
        key_ptr = <const char*>key_utf8

        rc = tiledb_array_delete_metadata(ctx_ptr, array_ptr, key_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

    def __len__(self):
        cdef tiledb_ctx_t*  ctx_ptr = (<Array>self.array).ctx.ptr
        cdef tiledb_array_t*  array_ptr = (<Array>self.array).ptr
        cdef int32_t rc
        cdef uint64_t num

        rc = tiledb_array_get_metadata_num(
                ctx_ptr, array_ptr, &num)

        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        return <int>num

    def keys(self):
        """
        Return metadata keys as list.

        :return: List of keys
        """
        return load_metadata(self.array, unpack=False)

    def values(self):
        # TODO this should be an iterator
        data = load_metadata(self.array, unpack=False)
        return data.values()

    def pop(self, key, default=None):
        raise NotImplementedError("dict.pop requires read-write access to array")

    def items(self):
        # TODO this should be an iterator
        data = load_metadata(self.array, unpack=True)
        return tuple( (k, data[k]) for k in data.keys() )

    def _set_numpy(self, key, np.ndarray arr, datatype = None):
        """
        Escape hatch to directly set meta key-value from a NumPy array.
        Key type and array dimensionality are checked, but no other type-checking
        is done. Not intended for routine use.

        :param key: key
        :param arr: 1d NumPy ndarray
        :return:
        """
        cdef tiledb_ctx_t*  ctx_ptr = (<Array>self.array).ctx.ptr
        cdef tiledb_array_t*  array_ptr = (<Array>self.array).ptr

        cdef:
            int32_t rc = TILEDB_OK
            const char* key_ptr = NULL
            bytes key_utf8
            tiledb_datatype_t tiledb_type
            uint32_t value_num = 0

        if not (isinstance(key, str) or isinstance(key, unicode)):
            raise ValueError("Unexpected key type '{}': expected str "
                             "type".format(type(key)))

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
