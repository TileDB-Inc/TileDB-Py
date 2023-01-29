IF TILEDBPY_MODULAR:
    include "common.pxi"
    from .libtiledb import *
    from .libtiledb cimport *

import weakref
from collections.abc import MutableMapping

from cython.operator cimport dereference as deref

from .datatypes import DataType

_NP_DATA_PREFIX = "__np_flat_"
_NP_SHAPE_PREFIX = "__np_shape_"


cdef extern from "Python.h":
    int PyBUF_READ
    object PyMemoryView_FromMemory(char*, Py_ssize_t, int)


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
        return PackedBuffer(value, TILEDB_BLOB, len(value))

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
        raise TypeError(f"Unsupported item type '{type(val0)}'")

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
            value_item = value[pack_idx]
            if not isinstance(value_item, int):
                raise TypeError(f"Mixed-type sequences are not supported: {value}")
            int64_ptr[pack_idx] = value_item
            pack_idx += 1
    else:
        double_ptr = <double*>&data_view[0]
        while pack_idx < value_num:
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

    if value_type in (TILEDB_BLOB, TILEDB_CHAR, TILEDB_STRING_ASCII):
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


cdef np.ndarray unpack_metadata_ndarray(
        tiledb_datatype_t value_type, uint32_t value_num, const char* value_ptr
    ):
    cdef np.dtype dtype = DataType.from_tiledb(value_type).np_dtype
    if value_ptr == NULL:
        return np.array((), dtype=dtype)

    # special case for TILEDB_STRING_UTF8: TileDB assumes size=1
    if value_type != TILEDB_STRING_UTF8:
        value_num *= dtype.itemsize

    return np.frombuffer(PyMemoryView_FromMemory(<char*>value_ptr, value_num, PyBUF_READ),
                         dtype=dtype).copy()


cdef object unpack_metadata(
        bint is_ndarray,
        tiledb_datatype_t value_type,
        uint32_t value_num,
        const char * value_ptr
    ):
    if value_ptr == NULL and value_num != 1:
        raise KeyError

    if is_ndarray:
        return unpack_metadata_ndarray(value_type, value_num, value_ptr)
    else:
        return unpack_metadata_val(value_type, value_num, value_ptr)


cdef put_metadata(Array array, key, value):
    cdef:
        PackedBuffer packed_buf
        tiledb_datatype_t tiledb_type
        uint32_t value_num
        cdef const unsigned char[:] data_view
        cdef const void* data_ptr
        tiledb_ctx_t* ctx_ptr = NULL

    if isinstance(value, np.ndarray):
        if value.ndim != 1:
            raise TypeError(f"Only 1D Numpy arrays can be stored as metadata")

        dt = DataType.from_numpy(value.dtype)
        if dt.ncells != 1:
            raise TypeError(f"Unsupported dtype '{value.dtype}'")

        tiledb_type = dt.tiledb_type
        value_num = len(value)
        # special case for TILEDB_STRING_UTF8: TileDB assumes size=1
        if tiledb_type == TILEDB_STRING_UTF8:
            value_num *= value.itemsize
        data_ptr = np.PyArray_DATA(value)
    else:
        packed_buf = pack_metadata_val(value)
        tiledb_type = packed_buf.tdbtype
        value_num = packed_buf.value_num
        data_view = packed_buf.data
        data_ptr = &data_view[0] if value_num > 0 else NULL

    key_utf8 = key.encode('UTF-8')
    cdef const char* key_utf8_ptr = <const char*>key_utf8
    cdef int rc = TILEDB_OK
    ctx_ptr = <tiledb_ctx_t*>PyCapsule_GetPointer(array.ctx.__capsule__(), "ctx")
    with nogil:
        rc = tiledb_array_put_metadata(
            ctx_ptr,
            array.ptr,
            key_utf8_ptr,
            tiledb_type,
            value_num,
            data_ptr,
        )
    if rc != TILEDB_OK:
        _raise_ctx_err(ctx_ptr, rc)


cdef object get_metadata(Array array, key, is_ndarray=False):
    cdef:
        tiledb_datatype_t value_type
        uint32_t value_num = 0
        const char* value_ptr = NULL
        bytes key_utf8 = key.encode('UTF-8')
        const char* key_utf8_ptr = <const char*>key_utf8
        tiledb_ctx_t* ctx_ptr = NULL

    cdef int32_t rc = TILEDB_OK
    ctx_ptr = <tiledb_ctx_t*>PyCapsule_GetPointer(array.ctx.__capsule__(), "ctx")
    with nogil:
        rc = tiledb_array_get_metadata(
            ctx_ptr,
            array.ptr,
            key_utf8_ptr,
            &value_type,
            &value_num,
            <const void**>&value_ptr,
        )
    if rc != TILEDB_OK:
        _raise_ctx_err(ctx_ptr, rc)

    return unpack_metadata(is_ndarray, value_type, value_num, value_ptr)


def iter_metadata(Array array, keys_only, dump=False):
    """
    Iterate over array metadata keys or (key, value) tuples

    :param array: tiledb_array_t
    :param keys_only: whether to yield just keys or values too
    """
    cdef:
        tiledb_ctx_t* ctx_ptr = <tiledb_ctx_t*>PyCapsule_GetPointer(
            array.ctx.__capsule__(), "ctx")
        tiledb_array_t* array_ptr = array.ptr
        uint64_t metadata_num
        const char* key_ptr = NULL
        uint32_t key_len
        tiledb_datatype_t value_type
        uint32_t value_num
        const char* value_ptr = NULL
        const char* value_type_str = NULL
    
    if keys_only and dump:
        raise ValueError("keys_only and dump cannot both be True")

    cdef int32_t rc = TILEDB_OK
    with nogil:
        rc = tiledb_array_get_metadata_num(ctx_ptr, array_ptr, &metadata_num)
    if rc != TILEDB_OK:
        _raise_ctx_err(ctx_ptr, rc)

    for i in range(metadata_num):
        with nogil:
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
        else:
            value = unpack_metadata(key.startswith(_NP_DATA_PREFIX),
                                    value_type, value_num, value_ptr)

            if dump:
                rc = tiledb_datatype_to_str(value_type, &value_type_str)
                if rc != TILEDB_OK:
                    _raise_ctx_err(ctx_ptr, rc)

                yield (
                    "### Array Metadata ###\n"
                    f"- Key: {key}\n"
                    f"- Value: {value}\n"
                    f"- Type: {value_type_str.decode('UTF-8')}\n"
                )
            else:
                yield key, value


cdef class Metadata:
    def __init__(self, array):
        self.array_ref = weakref.ref(array)

    @property
    def array(self):
        assert self.array_ref() is not None, \
            "Internal error: invariant violation ([] from gc'd Array)"
        return self.array_ref()

    def __setitem__(self, key, value):
        if not isinstance(key, str):
            raise TypeError(f"Unexpected key type '{type(key)}': expected str")

        # ensure previous key(s) are deleted (e.g. in case of replacing a
        # non-numpy value with a numpy value or vice versa)
        del self[key]

        if isinstance(value, np.ndarray):
            flat_value = value.ravel()
            put_metadata(self.array, _NP_DATA_PREFIX + key, flat_value)
            if value.shape != flat_value.shape:
                put_metadata(self.array, _NP_SHAPE_PREFIX + key, value.shape)
        else:
            put_metadata(self.array, key, value)

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise TypeError(f"Unexpected key type '{type(key)}': expected str")

        array = self.array
        try:
            return get_metadata(array, key)
        except KeyError as ex:
            try:
                np_array = get_metadata(array, _NP_DATA_PREFIX + key, is_ndarray=True)
            except KeyError:
                raise KeyError(key) from None

            try:
                shape = get_metadata(array, _NP_SHAPE_PREFIX + key)
            except KeyError:
                return np_array
            else:
                return np_array.reshape(shape)

    def __delitem__(self, key):
        if not isinstance(key, str):
            raise TypeError(f"Unexpected key type '{type(key)}': expected str")

        cdef:
            tiledb_ctx_t* ctx_ptr = <tiledb_ctx_t*>PyCapsule_GetPointer((
                <Array>self.array).ctx.__capsule__(), "ctx")
            tiledb_array_t* array_ptr = (<Array>self.array).ptr
            const char* key_utf8_ptr
            int32_t rc

        # key may be stored as is or it may be prefixed (for numpy values)
        # we don't know this here so delete all potential internal keys
        for k in key, _NP_DATA_PREFIX + key, _NP_SHAPE_PREFIX + key:
            key_utf8 = k.encode('UTF-8')
            key_utf8_ptr = <const char*>key_utf8
            with nogil:
                rc = tiledb_array_delete_metadata(ctx_ptr, array_ptr, key_utf8_ptr)
            if rc != TILEDB_OK:
                _raise_ctx_err(ctx_ptr, rc)

    def __contains__(self, key):
        if not isinstance(key, str):
            raise TypeError(f"Unexpected key type '{type(key)}': expected str")

        cdef:
            tiledb_ctx_t* ctx_ptr = <tiledb_ctx_t*>PyCapsule_GetPointer((
                <Array>self.array).ctx.__capsule__(), "ctx")
            tiledb_array_t* array_ptr = (<Array>self.array).ptr
            bytes key_utf8 = key.encode('UTF-8')
            const char* key_utf8_ptr = <const char*>key_utf8
            tiledb_datatype_t value_type
            int32_t has_key

        cdef int32_t rc = TILEDB_OK
        with nogil:
            rc = tiledb_array_has_metadata_key(
                ctx_ptr,
                array_ptr,
                key_utf8_ptr,
                &value_type,
                &has_key,
            )
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        # if key doesn't exist, check the _NP_DATA_PREFIX prefixed key
        if not has_key and not key.startswith(_NP_DATA_PREFIX):
            has_key = self.__contains__(_NP_DATA_PREFIX + key)

        return bool(has_key)

    def consolidate(self):
        """
        Consolidate array metadata. Array must be closed.

        :return:
        """
        # TODO: ensure that the array is not x-locked?
        ctx = (<Array?> self.array).ctx
        config = ctx.config()
        cdef:
            uint32_t rc = 0
            tiledb_ctx_t* ctx_ptr = <tiledb_ctx_t*>PyCapsule_GetPointer(
                ctx.__capsule__(), "ctx")
            tiledb_config_t* config_ptr = NULL
            tiledb_encryption_type_t key_type = TILEDB_NO_ENCRYPTION
            void* key_ptr = NULL
            uint32_t key_len = 0
            bytes bkey
            bytes buri = unicode_path(self.array.uri)
            str key = (<Array?>self.array).key

        if config:
            config_ptr = <tiledb_config_t*>PyCapsule_GetPointer(
                config.__capsule__(), "config")

        if key is not None:
            if isinstance(key, str):
                bkey = key.encode('ascii')
            else:
                bkey = bytes(self.array.key)
            key_type = TILEDB_AES_256_GCM
            key_ptr = <void *> PyBytes_AS_STRING(bkey)
            #TODO: unsafe cast here ssize_t -> uint64_t
            key_len = <uint32_t> PyBytes_GET_SIZE(bkey)

        cdef const char* buri_ptr = <const char*>buri

        with nogil:
            rc = tiledb_array_consolidate_with_key(
                    ctx_ptr,
                    buri_ptr,
                    key_type,
                    key_ptr,
                    key_len,
                    config_ptr)
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
            tiledb_ctx_t* ctx_ptr = <tiledb_ctx_t*>PyCapsule_GetPointer((
                <Array>self.array).ctx.__capsule__(), "ctx")
            tiledb_array_t* array_ptr = (<Array>self.array).ptr
            uint64_t num

        cdef int32_t rc = TILEDB_OK
        with nogil:
            rc = tiledb_array_get_metadata_num(ctx_ptr, array_ptr, &num)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        # subtract the _NP_SHAPE_PREFIX prefixed keys
        for key in iter_metadata(self.array, keys_only=True):
            if key.startswith(_NP_SHAPE_PREFIX):
                num -= 1

        return num

    def __iter__(self):
        np_data_prefix_len = len(_NP_DATA_PREFIX)
        for key in iter_metadata(self.array, keys_only=True):
            if key.startswith(_NP_DATA_PREFIX):
                yield key[np_data_prefix_len:]
            elif not key.startswith(_NP_SHAPE_PREFIX):
                yield key
            # else: ignore the shape keys

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
        # TODO this should be an iterator/view
        return [v for k, v in self._iteritems()]

    def items(self):
        # TODO this should be an iterator/view
        return tuple(self._iteritems())

    def _iteritems(self):
        np_data_prefix_len = len(_NP_DATA_PREFIX)
        np_shape_prefix_len = len(_NP_SHAPE_PREFIX)
        ndarray_items = []
        np_shape_map = {}

        # 1. yield all non-ndarray (key, value) pairs and keep track of
        # the ndarray data and shape to assemble them later
        for key, value in iter_metadata(self.array, keys_only=False):
            if key.startswith(_NP_DATA_PREFIX):
                ndarray_items.append((key[np_data_prefix_len:], value))
            elif key.startswith(_NP_SHAPE_PREFIX):
                np_shape_map[key[np_shape_prefix_len:]] = value
            else:
                yield key, value

        # 2. yield all ndarray (key, value) pairs after reshaping (if necessary)
        for key, value in ndarray_items:
            shape = np_shape_map.get(key)
            if shape is not None:
                value = value.reshape(shape)
            yield key, value

    def dump(self):
        for metadata in iter_metadata(self.array, keys_only=False, dump=True):
            print(metadata)