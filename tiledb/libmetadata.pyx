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

cdef object metadata_value_check(val):
    if isinstance(val, int):
        if val > numeric_limits[int64_t].max():
            raise OverflowError("Overflow integer values not supported!")

    elif (PY_MAJOR_VERSION < 3) and isinstance(val, long):
        if val > numeric_limits[int64_t].max():
            raise OverflowError("Overflow integer values not supported!")

cdef object pack_metadata_val(val, vector[char]& data_buf,
                              tiledb_datatype_t* ret_type, uint32_t* value_num):

    cdef:
        tiledb_datatype_t first_type
        tiledb_datatype_t next_type
        object utf8
        const char* data_ptr = NULL

    cdef:
        double o_float64
        int64_t o_int64

    metadata_value_check(val)

    if isinstance(val, (list, tuple)):
        value_num[0] = <uint32_t>len(val)
        raise NotImplementedError("list, tuple")

        #first_type = tiledb_item_type(val[0])
        #type_nbytes = tiledb_datatype_size(first_type)
        #buffer_nbytes = type_nbytes * len(val)
        #data.reserve(buffer_nbytes)

        #for v in val:
        #    next_type = tiledb_item_type(v)
        #    if first_type != next_type:
        #        raise ValueError("list/tuple elements must be homogeneous!")

        #    data.insert(data.end(), type_nbytes, <char*>&v_raw)

    elif isinstance(val, int) or (PY_MAJOR_VERSION < 3 and isinstance(val, long)):

        first_type = TILEDB_INT64
        data_nbytes = tiledb_datatype_size(first_type)
        value_num[0] = <uint32_t>1
        o_int64 = PyLong_AsLongLong(val)
        data_ptr = <char*>&o_int64

    elif isinstance(val, float):

        first_type = TILEDB_FLOAT64
        data_nbytes = tiledb_datatype_size(first_type)
        value_num[0] = <uint32_t>1
        o_double = PyFloat_AsDouble(val)
        data_ptr = <char*>&o_double

    elif isinstance(val, bytes):

        first_type = TILEDB_CHAR
        data_nbytes = len(val)
        value_num[0] = len(val)
        data_ptr = PyBytes_AS_STRING(val)

    elif isinstance(val, unicode):

        first_type = TILEDB_STRING_UTF8
        utf8 = (<str>val).encode('UTF-8')
        data_ptr = <char*>utf8
        data_nbytes = len(utf8)
        value_num[0] = data_nbytes

    elif isinstance(val, np.ndarray):

        raise ValueError("Unsupported type: numpy array")

    else:
        # TODO serialize as JSON?
        raise ValueError("Unsupported item type '{}'".format(type(val)))

    data_buf.resize(data_nbytes)
    # note: cython doesn't support C++11 back_inserter (which supports len+pointer)
    memcpy(data_buf.data(), data_ptr, data_nbytes)

    # note: cython pass-by-reference bug
    # https://github.com/cython/cython/issues/1863
    ret_type[0] = first_type
    return None


cdef object unpack_metadata_val(tiledb_datatype_t value_type,
                                uint32_t value_num,
                                const char* value_ptr):

    cdef:
        double o_float64
        int64_t o_int64

    if value_type == TILEDB_STRING_UTF8:
        new_obj = value_ptr[:value_num].decode('UTF-8')
        return new_obj

    elif value_type == TILEDB_CHAR:
        new_obj = bytes(value_ptr[:value_num])
        return new_obj

    elif value_num > 1:

        # unpack sequence
        # should this return tuple instead or list?
        new_obj = list()
        for i in range(0, value_num):
            item = unpack_metadata_val(value_type, 1, value_ptr)
            new_obj.append(item)
            value_ptr += tiledb_datatype_size(value_type)

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

    return None


cdef object put_metadata(Array array,
                         key, val):

    cdef tiledb_array_t* array_ptr = array.ptr
    cdef tiledb_ctx_t* ctx_ptr = array.ctx.ptr

    cdef int rc = TILEDB_OK
    cdef vector[char] data_buf = vector[char]()
    cdef const char* key_cstr
    cdef bytes key_utf8
    cdef tiledb_datatype_t ret_type
    cdef uint32_t value_num

    pack_metadata_val(val, data_buf, &ret_type, &value_num)

    if (data_buf.size() < 1):
        raise ValueError("Failed to serialize pyobject value '{}'".format(val))

    key_utf8 = key.encode('UTF-8')
    key_cstr = PyBytes_AS_STRING(key_utf8)

    rc = tiledb_array_put_metadata(
            ctx_ptr,
            array_ptr,
            key_cstr,
            ret_type,
            value_num,
            data_buf.data())

    if rc != TILEDB_OK:
        _raise_ctx_err(ctx_ptr, rc)

    return None

cdef object get_metadata(array: Array,
                         key: unicode):

    cdef tiledb_array_t* array_ptr = array.ptr
    cdef tiledb_ctx_t* ctx_ptr = array.ctx.ptr

    cdef:
        const char* key_ptr
        object key_utf8
        int32_t rc

    cdef:
        uint32_t key_len
        tiledb_datatype_t value_type
        uint32_t value_num
        const char* value = NULL

    key_utf8 = key.encode('UTF-8')
    key_ptr = <const char*>key_utf8

    rc = tiledb_array_get_metadata(
            ctx_ptr, array_ptr,
            key_ptr,
            &value_type, &value_num, <const void**>&value)

    if rc != TILEDB_OK:
        _raise_ctx_err(ctx_ptr, rc)

    if (value == NULL):
        raise KeyError(key)

    return unpack_metadata_val(value_type, value_num, value)

cdef put_metadata_dict(Array array, kv):

    for k,v in kv.iteritems():
        put_metadata(array, k, v)

cdef object load_metadata(Array array, unpack=True):
    """
    Load array metadata dict or keys

    :param ctx: tiledb_ctx_t
    :param array: tiledb_array_t
    :param unpack: unpack the values into dictionary
    :return: dict(k: v) if unpack else list(k)
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

        if value_num == 0:
            # TODO warning
            raise ValueError("Unexpected 0-length value")

        key = PyUnicode_FromStringAndSize(key_ptr, key_len)

        unpacked_val = unpack_metadata_val(value_type, value_num, value)

        if unpacked_val is None:
            raise KeyError("key not found: ", key)

        if unpack:
            ret_val[key] = unpacked_val
        else:
            ret_val.append(key)

    return ret_val


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
            if isinstance(self.array.key, str):
                bkey = self.array.key.encode('ascii')
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