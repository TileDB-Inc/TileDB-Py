#!python
#cython: embedsignature=True
#cython: auto_pickle=False

from cpython.version cimport PY_MAJOR_VERSION

from cpython.bytes cimport (PyBytes_GET_SIZE,
                            PyBytes_AS_STRING,
                            PyBytes_Size,
                            PyBytes_FromString,
                            PyBytes_FromStringAndSize)

from cpython.mem cimport (PyMem_Malloc,
                          PyMem_Realloc,
                          PyMem_Free)

from cpython.ref cimport (Py_INCREF, PyTypeObject)

from libc.stdio cimport (FILE, stdout)
from libc.stdio cimport stdout
from libc.stdlib cimport malloc, calloc, free
from libc.stdint cimport (uint64_t, int64_t, uintptr_t)
from libc cimport limits

# Numpy imports
"""
cdef extern from "numpyFlags.h":
    # Include 'numpyFlags.h' into the generated C code to disable warning.
    # This must be included before numpy is cimported
    pass
"""

import numpy as np
cimport numpy as np

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

import sys
from os.path import abspath
from collections import OrderedDict

# Integer types supported by Python / System
if sys.version_info >= (3, 0):
    _MAXINT = 2 ** 31 - 1
    _inttypes = (int, np.integer)
else:
    _MAXINT = sys.maxint
    _inttypes = (int, long, np.integer)

# KB / MB in bytes
_KB = 1024
_MB = 1024 * _KB

# The native int type for this platform
IntType = np.dtype(np.int_)

# Numpy initialization code
np.import_array()

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

class TileDBError(Exception):
    """TileDB Error Exception

    Captures and raises error return code (``TILEDB_ERR``) messages when calling ``libtiledb``
    functions.  The error message that is raised is the last error set for the :py:class:`tiledb.Ctx`.

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
            return MemoryError()
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


def stats_disable():
    """Disable TileDB internal statistics."""
    tiledb_stats_disable()


def stats_reset():
    """Reset all TileDB internal statistics to 0."""
    tiledb_stats_reset()


def stats_dump():
    """Prints all TileDB internal statistics values to standard output."""
    tiledb_stats_dump(stdout)


cdef unicode ustring(object s):
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


cdef class Config(object):
    """TileDB Config class

    Valid parameters (unknown parameters will be ignored):

    - ``sm.tile_cache_size``
       The tile cache size in bytes. Any ``uint64_t`` value is acceptable.
       **Default**: 10,000,000
    - ``sm.array_schema_cache_size``
       The array schema cache size in bytes. Any ``uint64_t`` value is acceptable.
       **Default**: 10,000,000
    - ``sm.fragment_metadata_cache_size``
       The fragment metadata cache size in bytes. Any ``uint64_t`` value is
       acceptable.
    - ``sm.enable_signal_handlers``
       Whether or not TileDB will install signal handlers.
       **Default**: true
       **Default**: 10,000,000
    - ``sm.number_of_threads``
       The number of allocated threads per TileDB context.
       **Default**: number of cores
    - ``vfs.max_parallel_ops``
       The maximum number of VFS parallel operations.
       **Default**: number of cores
    - ``vfs.min_parallel_size``
       The minimum number of bytes in a parallel VFS operation. (Does not
       affect parallel S3 writes.)
       **Default**: 10MB
    - ``vfs.s3.region``
       The S3 region, if S3 is enabled.
       **Default**: us-east-1
    - ``vfs.s3.scheme``
       The S3 scheme (``http`` or ``https``), if S3 is enabled.
       **Default**: https
    - ``vfs.s3.endpoint_override``
       The S3 endpoint, if S3 is enabled.
       **Default**: ""
    - ``vfs.s3.use_virtual_addressing``
       The S3 use of virtual addressing (``true`` or ``false``), if S3 is
       enabled.
       **Default**: true
    - ``vfs.s3.multipart_part_size``
       The part size (in bytes) used in S3 multipart writes, if S3 is enabled.
       Any ``uint64_t`` value is acceptable. Note: ``vfs.s3.multipart_part_size *
       vfs.max_parallel_ops`` bytes will be buffered before issuing multipart
       uploads in parallel.
       **Default**: 5*1024*1024
    - ``vfs.s3.connect_timeout_ms``
       The connection timeout in ms. Any ``long`` value is acceptable.
       **Default**: 3000
    - ``vfs.s3.connect_max_tries``
       The maximum tries for a connection. Any ``long`` value is acceptable.
       **Default**: 5
    - ``vfs.s3.connect_scale_factor``
       The scale factor for exponential backofff when connecting to S3.
       Any ``long`` value is acceptable.
       **Default**: 25
    - ``vfs.s3.request_timeout_ms``
       The request timeout in ms. Any ``long`` value is acceptable.
       **Default**: 3000
    - ``vfs.hdfs.name_node"``
       Name node for HDFS.
       **Default**: ""
    - ``vfs.hdfs.username``
       HDFS username.
       **Default**: ""
    - ``vfs.hdfs.kerb_ticket_cache_path``
       HDFS kerb ticket cache path.
       **Default**: ""

    :param dict params: Set parameter values from dict like object
    :param str path: Set parameter values from persisted Config parameter file
    """

    cdef tiledb_config_t* ptr

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

    def __getitem__(self, object key):
        """Gets a config parameter value.

        :param str key: Name of parameter to get
        :return: Config parameter value string
        :rtype str:
        :raises TypeError: `key` cannot be encoded into a UTF-8 string
        :raises KeyError: Config parameter not found
        :raises: :py:exc:`tiledb.TileDBError`

        """
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
            raise KeyError(key)
        cdef bytes value = PyBytes_FromString(value_ptr)
        return value.decode('UTF-8')

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
        """Returns an iterator object over Config paramters, values

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
    cdef ConfigItems config_items

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
    cdef ConfigItems config_items

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
    cdef Config config
    cdef tiledb_config_iter_t* ptr

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

    cdef tiledb_ctx_t* ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            tiledb_ctx_free(&self.ptr)

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

    def config(self):
        """Returns the Config instance associated with the Ctx
        """
        cdef tiledb_config_t* config_ptr = NULL
        check_error(self,
                    tiledb_ctx_get_config(self.ptr, &config_ptr))
        return Config.from_ptr(config_ptr)


cdef tiledb_datatype_t _tiledb_dtype(np.dtype dtype) except? TILEDB_CHAR:
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
    elif dtype == np.str_ or dtype == np.bytes_:  # or bytes
        return TILEDB_CHAR
    raise TypeError("data type {0!r} not understood".format(dtype))


cdef int _numpy_typeid(tiledb_datatype_t tiledb_dtype):
    """
    Return a numpy type num (int) given a tiledb_datatype_t enum value
    """
    if tiledb_dtype == TILEDB_INT32:
        return np.NPY_INT32
    elif tiledb_dtype == TILEDB_UINT32:
        return np.NPY_UINT32
    elif tiledb_dtype == TILEDB_INT64:
        return np.NPY_INT64
    elif tiledb_dtype == TILEDB_UINT64:
        return np.NPY_UINT64
    elif tiledb_dtype == TILEDB_FLOAT32:
        return np.NPY_FLOAT32
    elif tiledb_dtype == TILEDB_FLOAT64:
        return np.NPY_FLOAT64
    elif tiledb_dtype == TILEDB_INT8:
        return np.NPY_INT8
    elif tiledb_dtype == TILEDB_UINT8:
        return np.NPY_UINT8
    elif tiledb_dtype == TILEDB_INT16:
        return np.NPY_INT16
    elif tiledb_dtype == TILEDB_UINT16:
        return np.NPY_UINT16
    elif tiledb_dtype == TILEDB_CHAR:
        return np.NPY_STRING
    else:
        return np.NPY_NOTYPE


cdef _numpy_type(tiledb_datatype_t tiledb_dtype):
    """
    Return a numpy *type* (not dtype) object given a tiledb_datatype_t enum value
    """
    if tiledb_dtype == TILEDB_INT32:
        return np.int32
    elif tiledb_dtype == TILEDB_UINT32:
        return np.uint32
    elif tiledb_dtype == TILEDB_INT64:
        return np.int64
    elif tiledb_dtype == TILEDB_UINT64:
        return np.uint64
    elif tiledb_dtype == TILEDB_FLOAT32:
        return np.float32
    elif tiledb_dtype == TILEDB_FLOAT64:
        return np.float64
    elif tiledb_dtype == TILEDB_INT8:
        return np.int8
    elif tiledb_dtype == TILEDB_UINT8:
        return np.uint8
    elif tiledb_dtype == TILEDB_INT16:
        return np.int16
    elif tiledb_dtype == TILEDB_UINT16:
        return np.uint16
    elif tiledb_dtype == TILEDB_CHAR:
        return np.bytes_
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

cdef tiledb_compressor_t _tiledb_compressor(object c) except TILEDB_NO_COMPRESSION:
    """
    Return a tiledb_compressor_t enum value from a string label, or None for no compression
    """
    if c is None:
        return TILEDB_NO_COMPRESSION
    elif c == "gzip":
        return TILEDB_GZIP
    elif c == "zstd":
        return TILEDB_ZSTD
    elif c == "lz4":
        return TILEDB_LZ4
    elif c == "rle":
        return TILEDB_RLE
    elif c == "bzip2":
        return TILEDB_BZIP2
    elif c == "double-delta":
        return TILEDB_DOUBLE_DELTA
    raise ValueError("unknown compressor: {0!r}".format(c))


cdef unicode _tiledb_compressor_string(tiledb_compressor_t c):
    """
    Return the (unicode) string representation of a tiledb_compressor_t enum value
    """
    if c == TILEDB_NO_COMPRESSION:
        return u"none"
    elif c == TILEDB_GZIP:
        return u"gzip"
    elif c == TILEDB_ZSTD:
        return u"zstd"
    elif c == TILEDB_LZ4:
        return u"lz4"
    elif c == TILEDB_RLE:
        return u"rle"
    elif c == TILEDB_BZIP2:
        return u"bzip2"
    elif c == TILEDB_DOUBLE_DELTA:
        return u"double-delta"


cdef tiledb_layout_t _tiledb_layout(object order) except TILEDB_UNORDERED:
    """
    Return the tiledb_layout_t enum value given a layout string label
    """
    if order == "row-major" or order == 'C':
        return TILEDB_ROW_MAJOR
    elif order == "col-major" or order == 'F':
        return TILEDB_COL_MAJOR
    elif order == "global":
        return TILEDB_GLOBAL_ORDER
    elif order == None or order == "unordered":
        return TILEDB_UNORDERED
    raise ValueError("unknown tiledb layout: {0!r}".format(order))


cdef unicode _tiledb_layout_string(tiledb_layout_t order):
    """
    Return the unicode string label given a tiledb_layout_t enum value
    """
    if order == TILEDB_ROW_MAJOR:
        return u"row-major"
    elif order == TILEDB_COL_MAJOR:
        return u"col-major"
    elif order == TILEDB_GLOBAL_ORDER:
        return u"global"
    elif order == TILEDB_UNORDERED:
        return u"unordered"


cdef class Filter(object):
    """Base class for all TileDB filters."""

    cdef Ctx ctx
    cdef tiledb_filter_t* ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            tiledb_filter_free(&self.ptr)

    def __init__(self, Ctx ctx, tiledb_filter_type_t filter_type):
        cdef tiledb_ctx_t* ctx_ptr = ctx.ptr
        cdef tiledb_filter_t* filter_ptr = NULL
        cdef int rc = TILEDB_OK
        rc = tiledb_filter_alloc(ctx_ptr, filter_type, &filter_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        self.ctx = ctx
        self.ptr = filter_ptr
        return


cdef class CompressionFilter(Filter):
    """Base class for filters performing compression.

    All compression filters support a compression level option, although some (such as RLE) ignore it.

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> ctx = tiledb.Ctx()
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(ctx, tiledb.Dim(ctx, domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(ctx, name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList(ctx, [tiledb.GzipFilter(ctx, level=10)]))
    ...     schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    def __init__(self, Ctx ctx, tiledb_filter_type_t filter_type, level):
        super().__init__(ctx, filter_type)
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
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef int rc = TILEDB_OK
        cdef int clevel = -1
        rc = tiledb_filter_get_option(ctx_ptr, self.ptr, TILEDB_COMPRESSION_LEVEL, &clevel)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return clevel


cdef class NoOpFilter(Filter):
    """
    A filter that does nothing.
    """

    @staticmethod
    cdef from_ptr(Ctx ctx, const tiledb_filter_t* filter_ptr):
        assert(filter_ptr != NULL)
        cdef NoOpFilter filter_obj = NoOpFilter.__new__(NoOpFilter)
        filter_obj.ctx = ctx
        # need to cast away the const
        filter_obj.ptr = <tiledb_filter_t*> filter_ptr
        return filter_obj

    def __init__(self, Ctx ctx):
        super().__init__(ctx, TILEDB_FILTER_NONE)


cdef class GzipFilter(CompressionFilter):
    """Filter that compresses using gzip.

    :param ctx: TileDB Ctx
    :type ctx: tiledb.Ctx
    :param level: (default None) If not None set the compressor level
    :type level: int

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> ctx = tiledb.Ctx()
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(ctx, tiledb.Dim(ctx, domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(ctx, name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList(ctx, [tiledb.GzipFilter(ctx)]))
    ...     schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    @staticmethod
    cdef from_ptr(Ctx ctx, const tiledb_filter_t* filter_ptr):
        assert(filter_ptr != NULL)
        cdef GzipFilter filter_obj = GzipFilter.__new__(GzipFilter)
        filter_obj.ctx = ctx
        # need to cast away the const
        filter_obj.ptr = <tiledb_filter_t*> filter_ptr
        return filter_obj

    def __init__(self, Ctx ctx, level=None):
        super().__init__(ctx, TILEDB_FILTER_GZIP, level)


cdef class ZstdFilter(CompressionFilter):
    """Filter that compresses using zstd.

    :param ctx: TileDB Ctx
    :type ctx: tiledb.Ctx
    :param level: (default None) If not None set the compressor level
    :type level: int

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> ctx = tiledb.Ctx()
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(ctx, tiledb.Dim(ctx, domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(ctx, name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList(ctx, [tiledb.ZstdFilter(ctx)]))
    ...     schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    @staticmethod
    cdef from_ptr(Ctx ctx, const tiledb_filter_t* filter_ptr):
        assert(filter_ptr != NULL)
        cdef ZstdFilter filter_obj = ZstdFilter.__new__(ZstdFilter)
        filter_obj.ctx = ctx
        # need to cast away the const
        filter_obj.ptr = <tiledb_filter_t*> filter_ptr
        return filter_obj

    def __init__(self, Ctx ctx, level=None):
        super().__init__(ctx, TILEDB_FILTER_ZSTD, level)


cdef class LZ4Filter(CompressionFilter):
    """Filter that compresses using lz4.

    :param ctx: TileDB Ctx
    :type ctx: tiledb.Ctx
    :param level: (default None) If not None set the compressor level
    :type level: int

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> ctx = tiledb.Ctx()
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(ctx, tiledb.Dim(ctx, domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(ctx, name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList(ctx, [tiledb.LZ4Filter(ctx)]))
    ...     schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    @staticmethod
    cdef from_ptr(Ctx ctx, const tiledb_filter_t* filter_ptr):
        assert(filter_ptr != NULL)
        cdef LZ4Filter filter_obj = LZ4Filter.__new__(LZ4Filter)
        filter_obj.ctx = ctx
        # need to cast away the const
        filter_obj.ptr = <tiledb_filter_t*> filter_ptr
        return filter_obj

    def __init__(self, Ctx ctx, level=None):
        super().__init__(ctx, TILEDB_FILTER_LZ4, level)


cdef class Bzip2Filter(CompressionFilter):
    """Filter that compresses using bzip2.

    :param level: (default None) If not None set the compressor level
    :type level: int

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> ctx = tiledb.Ctx()
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(ctx, tiledb.Dim(ctx, domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(ctx, name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList(ctx, [tiledb.Bzip2Filter(ctx)]))
    ...     schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    @staticmethod
    cdef from_ptr(Ctx ctx, const tiledb_filter_t* filter_ptr):
        assert(filter_ptr != NULL)
        cdef Bzip2Filter filter_obj = Bzip2Filter.__new__(Bzip2Filter)
        filter_obj.ctx = ctx
        # need to cast away the const
        filter_obj.ptr = <tiledb_filter_t*> filter_ptr
        return filter_obj

    def __init__(self, Ctx ctx, level=None):
        super().__init__(ctx, TILEDB_FILTER_BZIP2, level)


cdef class RleFilter(CompressionFilter):
    """Filter that compresses using run-length encoding (RLE).

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> ctx = tiledb.Ctx()
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(ctx, tiledb.Dim(ctx, domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(ctx, name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList(ctx, [tiledb.RleFilter(ctx)]))
    ...     schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    @staticmethod
    cdef from_ptr(Ctx ctx, const tiledb_filter_t* filter_ptr):
        assert(filter_ptr != NULL)
        cdef RleFilter filter_obj = RleFilter.__new__(RleFilter)
        filter_obj.ctx = ctx
        # need to cast away the const
        filter_obj.ptr = <tiledb_filter_t*> filter_ptr
        return filter_obj

    def __init__(self, Ctx ctx):
        super().__init__(ctx, TILEDB_FILTER_RLE, None)


cdef class DoubleDeltaFilter(CompressionFilter):
    """Filter that performs double-delta encoding.

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> ctx = tiledb.Ctx()
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(ctx, tiledb.Dim(ctx, domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(ctx, name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList(ctx, [tiledb.DoubleDeltaFilter(ctx)]))
    ...     schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    @staticmethod
    cdef from_ptr(Ctx ctx, const tiledb_filter_t* filter_ptr):
        assert(filter_ptr != NULL)
        cdef DoubleDeltaFilter filter_obj = DoubleDeltaFilter.__new__(DoubleDeltaFilter)
        filter_obj.ctx = ctx
        # need to cast away the const
        filter_obj.ptr = <tiledb_filter_t*> filter_ptr
        return filter_obj

    def __init__(self, Ctx ctx):
        super().__init__(ctx, TILEDB_FILTER_DOUBLE_DELTA, None)


cdef class BitShuffleFilter(Filter):
    """Filter that performs a bit shuffle transformation.

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> ctx = tiledb.Ctx()
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(ctx, tiledb.Dim(ctx, domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(ctx, name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList(ctx, [tiledb.BitShuffleFilter(ctx)]))
    ...     schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    @staticmethod
    cdef from_ptr(Ctx ctx, const tiledb_filter_t* filter_ptr):
        assert(filter_ptr != NULL)
        cdef BitShuffleFilter filter_obj = BitShuffleFilter.__new__(BitShuffleFilter)
        filter_obj.ctx = ctx
        # need to cast away the const
        filter_obj.ptr = <tiledb_filter_t*> filter_ptr
        return filter_obj

    def __init__(self, Ctx ctx):
        super().__init__(ctx, TILEDB_FILTER_BITSHUFFLE)


cdef class ByteShuffleFilter(Filter):
    """Filter that performs a byte shuffle transformation.

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> ctx = tiledb.Ctx()
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(ctx, tiledb.Dim(ctx, domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(ctx, name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList(ctx, [tiledb.ByteShuffleFilter(ctx)]))
    ...     schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    @staticmethod
    cdef from_ptr(Ctx ctx, const tiledb_filter_t* filter_ptr):
        assert(filter_ptr != NULL)
        cdef ByteShuffleFilter filter_obj = ByteShuffleFilter.__new__(ByteShuffleFilter)
        filter_obj.ctx = ctx
        # need to cast away the const
        filter_obj.ptr = <tiledb_filter_t*> filter_ptr
        return filter_obj

    def __init__(self, Ctx ctx):
        super().__init__(ctx, TILEDB_FILTER_BYTESHUFFLE)


cdef class BitWidthReductionFilter(Filter):
    """Filter that performs bit-width reduction.

     :param ctx: A TileDB Context
     :type ctx: tiledb.Ctx
     :param window: (default None) max window size for the filter
     :type window: int

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> ctx = tiledb.Ctx()
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(ctx, tiledb.Dim(ctx, domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(ctx, name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList(ctx, [tiledb.BitWidthReductionFilter(ctx)]))
    ...     schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    @staticmethod
    cdef from_ptr(Ctx ctx, const tiledb_filter_t* filter_ptr):
        assert(filter_ptr != NULL)
        cdef BitWidthReductionFilter filter_obj = BitWidthReductionFilter.__new__(BitWidthReductionFilter)
        filter_obj.ctx = ctx
        # need to cast away the const
        filter_obj.ptr = <tiledb_filter_t*> filter_ptr
        return filter_obj

    def __init__(self, Ctx ctx, window=None):
        super().__init__(ctx, TILEDB_FILTER_BIT_WIDTH_REDUCTION)
        if window is None:
            return
        cdef tiledb_ctx_t* ctx_ptr = ctx.ptr
        cdef unsigned int cwindow = window
        cdef int rc = TILEDB_OK
        rc = tiledb_filter_set_option(ctx_ptr, self.ptr, TILEDB_BIT_WIDTH_MAX_WINDOW, &cwindow)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

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
    """Filter that performs positive-delta encoding.

    :param ctx: A TileDB Context
    :type ctx: tiledb.Ctx
    :param window: (default None) the max window for the filter
    :type window: int

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> ctx = tiledb.Ctx()
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(ctx, tiledb.Dim(ctx, domain=(0, 9), tile=2, dtype=np.uint64))
    ...     a1 = tiledb.Attr(ctx, name="a1", dtype=np.int64,
    ...                      filters=tiledb.FilterList(ctx, [tiledb.PositiveDeltaFilter(ctx)]))
    ...     schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(a1,))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    @staticmethod
    cdef from_ptr(Ctx ctx, const tiledb_filter_t* filter_ptr):
        assert(filter_ptr != NULL)
        cdef PositiveDeltaFilter filter_obj = PositiveDeltaFilter.__new__(PositiveDeltaFilter)
        filter_obj.ctx = ctx
        # need to cast away the const
        filter_obj.ptr = <tiledb_filter_t*> filter_ptr
        return filter_obj

    def __init__(self, Ctx ctx, window=None):
        super().__init__(ctx, TILEDB_FILTER_POSITIVE_DELTA)
        if window is None:
            return
        cdef tiledb_ctx_t* ctx_ptr = ctx.ptr
        cdef unsigned int cwindow = window
        cdef int rc = TILEDB_OK
        rc = tiledb_filter_set_option(ctx_ptr, self.ptr, TILEDB_POSITIVE_DELTA_MAX_WINDOW, &cwindow)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

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


cdef Filter _filter_type_ptr_to_filter(Ctx ctx, tiledb_filter_type_t filter_type, tiledb_filter_t* filter_ptr):
    if filter_type == TILEDB_FILTER_NONE:
        return NoOpFilter.from_ptr(ctx, filter_ptr)
    elif filter_type == TILEDB_FILTER_GZIP:
       return GzipFilter.from_ptr(ctx, filter_ptr)
    elif filter_type == TILEDB_FILTER_ZSTD:
        return ZstdFilter.from_ptr(ctx, filter_ptr)
    elif filter_type == TILEDB_FILTER_LZ4:
        return LZ4Filter.from_ptr(ctx, filter_ptr)
    elif filter_type == TILEDB_FILTER_RLE:
        return RleFilter.from_ptr(ctx, filter_ptr)
    elif filter_type == TILEDB_FILTER_BZIP2:
        return Bzip2Filter.from_ptr(ctx, filter_ptr)
    elif filter_type == TILEDB_FILTER_DOUBLE_DELTA:
        return DoubleDeltaFilter.from_ptr(ctx, filter_ptr)
    elif filter_type == TILEDB_FILTER_BIT_WIDTH_REDUCTION:
        return BitWidthReductionFilter.from_ptr(ctx, filter_ptr)
    elif filter_type == TILEDB_FILTER_BITSHUFFLE:
        return BitShuffleFilter.from_ptr(ctx, filter_ptr)
    elif filter_type == TILEDB_FILTER_BYTESHUFFLE:
        return ByteShuffleFilter.from_ptr(ctx, filter_ptr)
    elif filter_type == TILEDB_FILTER_POSITIVE_DELTA:
        return PositiveDeltaFilter.from_ptr(ctx,  filter_ptr)
    else:
        raise ValueError("unknown filter type tag: {:s}".format(filter_type))


cdef class FilterList(object):
    """An ordered list of Filter objects for filtering TileDB data.

    FilterLists contain zero or more Filters, used for filtering attribute data, the array coordinate data, etc.

    :param ctx: A TileDB context
    :type ctx: tiledb.Ctx
    :param filters: An iterable of Filter objects to add.
    :param chunksize: (default None) chunk size used by the filter list in bytes
    :type chunksize: int

    **Example:**

    >>> import tiledb, numpy as np, tempfile
    >>> ctx = tiledb.Ctx()
    >>> with tempfile.TemporaryDirectory() as tmp:
    ...     dom = tiledb.Domain(ctx, tiledb.Dim(ctx, domain=(0, 9), tile=2, dtype=np.uint64))
    ...     # Create several filters
    ...     gzip_filter = tiledb.GzipFilter(ctx)
    ...     bw_filter = tiledb.BitWidthReductionFilter(ctx)
    ...     # Create a filter list that will first perform bit width reduction, then gzip compression.
    ...     filters = tiledb.FilterList(ctx, [bw_filter, gzip_filter])
    ...     a1 = tiledb.Attr(ctx, name="a1", dtype=np.int64, filters=filters)
    ...     # Create a second attribute filtered only by gzip compression.
    ...     a2 = tiledb.Attr(ctx, name="a2", dtype=np.int64,
    ...                      filters=tiledb.FilterList(ctx, [gzip_filter]))
    ...     schema = tiledb.ArraySchema(ctx, domain=dom, attrs=(a1, a2))
    ...     tiledb.DenseArray.create(tmp + "/array", schema)

    """

    cdef Ctx ctx
    cdef tiledb_filter_list_t* ptr

    def __cint__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            tiledb_filter_list_free(&self.ptr)

    @staticmethod
    cdef FilterList from_ptr(Ctx ctx, tiledb_filter_list_t* ptr):
        assert(ptr != NULL)
        cdef FilterList filter_list = FilterList.__new__(FilterList)
        filter_list.ctx = ctx
        # need to cast away the const
        filter_list.ptr = <tiledb_filter_list_t*> ptr
        return filter_list

    def __init__(self, Ctx ctx, filters=None, chunksize=None):
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


cdef class Attr(object):
    """Class representing a TileDB array attribute.

    :param tiledb.Ctx ctx: A TileDB Context
    :param str name: Attribute name, empty if anonymous
    :param dtype: Attribute value datatypes
    :type dtype: numpy.dtype object or type or string
    :param compressor: The compressor name and level for attribute values.
                       Available compressors:
                         - "gzip"
                         - "zstd"
                         - "lz4"
                         - "rle"
                         - "bzip2"
                         - "double-delta"
    :type compressor: tuple(str, int)
    :raises TypeError: invalid dtype
    :raises: :py:exc:`tiledb.TileDBError`

    """

    cdef Ctx ctx
    cdef tiledb_attribute_t* ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            tiledb_attribute_free(&self.ptr)

    @staticmethod
    cdef from_ptr(Ctx ctx, const tiledb_attribute_t* ptr):
        """Constructs an Attr class instance from a (non-null) tiledb_attribute_t pointer
        """
        assert(ptr != NULL)
        cdef Attr attr = Attr.__new__(Attr)
        attr.ctx = ctx
        # need to cast away the const
        attr.ptr = <tiledb_attribute_t*> ptr
        return attr

    def __init__(self,
                 Ctx ctx,
                 name=u"",
                 dtype=np.float64,
                 compressor=None,
                 filters=None):
        cdef bytes bname = ustring(name).encode('UTF-8')
        cdef const char* name_ptr = PyBytes_AS_STRING(bname)
        cdef np.dtype _dtype = np.dtype(dtype)
        cdef tiledb_datatype_t tiledb_dtype
        cdef unsigned int ncells
        if np.issubdtype(_dtype, np.bytes_):
            # flexible datatypes of unknown size have an itemsize of 0 (str, bytes, etc.)
            if _dtype.itemsize == 0:
                tiledb_dtype = TILEDB_CHAR
                ncells = TILEDB_VAR_NUM
            else:
                tiledb_dtype = TILEDB_CHAR
                ncells = _dtype.itemsize
        # handles n fixed size dtypes
        elif _dtype.kind == 'V':
            if _dtype.shape != ():
                raise TypeError("nested sub-array numpy dtypes are not supported")
            # check that types are the same
            typs = [t for (t, _) in _dtype.fields.values()]
            typ, ntypes = typs[0], len(typs)
            if typs.count(typ) != ntypes:
                raise TypeError('heterogenous record numpy dtypes are not supported')
            tiledb_dtype = _tiledb_dtype(typ)
            ncells = <unsigned int>(ntypes)
        # scalar cell type
        else:
            tiledb_dtype = _tiledb_dtype(_dtype)
            ncells = 1
        # compression and compression level
        cdef tiledb_compressor_t _compressor = TILEDB_NO_COMPRESSION
        cdef int _level = -1
        if compressor is not None:
            _compressor = _tiledb_compressor(ustring(compressor[0]))
            _level = int(compressor[1])
        cdef FilterList filter_list
        if filters is not None:
            if not isinstance(filters, FilterList):
                raise TypeError("filters argument must be a tiledb.FilterList")
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
        if _compressor != TILEDB_NO_COMPRESSION:
            rc = tiledb_attribute_set_compressor(ctx.ptr, attr_ptr, _compressor, _level)
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
        self.ctx = ctx
        self.ptr = attr_ptr

    def __eq__(self, other):
        if not isinstance(other, Attr):
            return False
        if (self.name != other.name or
            self.dtype != other.dtype or
            self.compressor != other.compressor):
            return False
        return True

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
        cdef unsigned int ncells = 0
        check_error(self.ctx,
                    tiledb_attribute_get_cell_val_num(self.ctx.ptr, self.ptr, &ncells))
        # flexible types with itemsize 0 are interpreted as VARNUM cells
        if ncells == TILEDB_VAR_NUM:
            return np.dtype((_numpy_type(typ), 0))
        elif ncells > 1:
            nptyp = _numpy_type(typ)
            # special case for fixed sized bytes arguments
            if typ == TILEDB_CHAR:
                return np.dtype((nptyp, ncells))
            # create an anon record dtype
            return np.dtype([('', nptyp)] * ncells)
        assert (ncells == 1)
        return np.dtype(_numpy_type(typ))

    cdef unicode _get_name(Attr self):
        cdef const char* c_name = NULL
        check_error(self.ctx,
                    tiledb_attribute_get_name(self.ctx.ptr, self.ptr, &c_name))
        cdef unicode name = c_name.decode('UTF-8', 'strict')
        if name.startswith("__attr"):
            return u""
        return name

    @property
    def name(self):
        """Attribute string name, empty string if the attribute is anonymous

        :rtype: str
        :raises: :py:exc:`tiledb.TileDBError`

        """
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
        cdef int level = -1
        cdef tiledb_compressor_t compr = TILEDB_NO_COMPRESSION
        check_error(self.ctx,
                    tiledb_attribute_get_compressor(self.ctx.ptr, self.ptr, &compr, &level))
        if compr == TILEDB_NO_COMPRESSION:
            return (None, -1)
        return (_tiledb_compressor_string(compr), int(level))


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

        return FilterList.from_ptr(self.ctx, filter_list_ptr)

    cdef unsigned int _cell_val_num(Attr self) except? 0:
        cdef unsigned int ncells = 0
        check_error(self.ctx,
                    tiledb_attribute_get_cell_val_num(self.ctx.ptr, self.ptr, &ncells))
        return ncells

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


cdef class Dim(object):
    """Class representing a dimension of a TileDB Array.

    :param tiledb.Ctx ctx: A TileDB Context
    :param str name: the dimension name, empty if anonymous
    :param domain:
    :type domain: tuple(int, int) or tuple(float, float)
    :param tile: Tile extent
    :type tile: int or float
    :dtype: the Dim numpy dtype object, type object, or string \
        that can be corerced into a numpy dtype object
    :raises ValueError: invalid domain or tile extent
    :raises TypeError: invalid domain, tile extent, or dtype type
    :raises: :py:exc:`TileDBError`

    """
    cdef Ctx ctx
    cdef tiledb_dimension_t* ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            tiledb_dimension_free(&self.ptr)

    @staticmethod
    cdef from_ptr(Ctx ctx, const tiledb_dimension_t* ptr):
        assert(ptr != NULL)
        cdef Dim dim = Dim.__new__(Dim)
        dim.ctx = ctx
        # need to cast away the const
        dim.ptr = <tiledb_dimension_t*> ptr
        return dim

    def __init__(self, Ctx ctx, name=u"", domain=None, tile=0, dtype=np.uint64):
        if len(domain) != 2:
            raise ValueError('invalid domain extent, must be a pair')
        if dtype is not None:
            dtype = np.dtype(dtype)
            if np.issubdtype(dtype, np.integer):
                info = np.iinfo(dtype)
            elif np.issubdtype(dtype, np.floating):
                info = np.finfo(dtype)
            else:
                raise TypeError("invalid Dim dtype {0!r}".format(dtype))
            if (domain[0] < info.min or domain[0] > info.max or
                    domain[1] < info.min or domain[1] > info.max):
                raise TypeError(
                    "invalid domain extent, domain cannot be safely cast to dtype {0!r}".format(dtype))
        domain_array = np.asarray(domain, dtype=dtype)
        domain_dtype = domain_array.dtype
        # check that the domain type is a valid dtype (intger / floating)
        if (not np.issubdtype(domain_dtype, np.integer) and
                not np.issubdtype(domain_dtype, np.floating)):
            raise TypeError("invalid Dim dtype {0!r}".format(domain_dtype))
        # if the tile extent is specified, cast
        cdef void* tile_size_ptr = NULL
        if tile > 0:
            tile_size_array = np.array(tile, dtype=domain_dtype)
            if tile_size_array.size != 1:
                raise ValueError("tile extent must be a scalar")
            tile_size_ptr = np.PyArray_DATA(tile_size_array)
        # argument conversion
        cdef bytes bname = ustring(name).encode('UTF-8')
        cdef const char* name_ptr = PyBytes_AS_STRING(bname)
        cdef tiledb_datatype_t dim_datatype = _tiledb_dtype(domain_dtype)
        cdef const void* domain_ptr = np.PyArray_DATA(domain_array)
        cdef tiledb_dimension_t* dim_ptr = NULL
        check_error(ctx,
                    tiledb_dimension_alloc(ctx.ptr,
                                           name_ptr,
                                           dim_datatype,
                                           domain_ptr,
                                           tile_size_ptr,
                                           &dim_ptr))
        assert(dim_ptr != NULL)
        self.ctx = ctx
        self.ptr = dim_ptr

    def __repr__(self):
        return 'Dim(name={0!r}, domain={1!s}, tile={2!s}, dtype={3!s})' \
            .format(self.name, self.domain, self.tile, self.dtype)

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
        return np.dtype(_numpy_type(self._get_type()))

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
    def isanon(self):
        """True if the dimension is anonymous

        :rtype: bool

        """
        name = self.name
        return name == u"" or name.startswith("__dim")

    cdef _integer_domain(self):
        cdef tiledb_datatype_t typ = self._get_type()
        if typ == TILEDB_FLOAT32 or typ == TILEDB_FLOAT64:
            return False
        return True

    cdef _shape(self):
        domain = self.domain
        return ((np.asscalar(domain[1]) -
                 np.asscalar(domain[0]) + 1),)

    @property
    def shape(self):
        """The shape of the dimension given the dimension's domain.

        **Note**: The shape is only valid for integer dimension domains.

        :rtype: tuple(numpy scalar, numpy scalar)
        :raises TypeError: floating point (inexact) domain

        """
        if not self._integer_domain():
            raise TypeError("shape only valid for integer dimension domains")
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

        :rtype: numpy scalar

        """
        cdef void* tile_ptr = NULL
        check_error(self.ctx,
                    tiledb_dimension_get_tile_extent(self.ctx.ptr, self.ptr, &tile_ptr))
        if tile_ptr == NULL:
            return None
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> 1
        cdef int typeid = _numpy_typeid(self._get_type())
        assert(typeid != np.NPY_NOTYPE)
        cdef np.ndarray tile_array =\
            np.PyArray_SimpleNewFromData(1, shape, typeid, tile_ptr)
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
        cdef void* domain_ptr = NULL
        check_error(self.ctx,
                    tiledb_dimension_get_domain(self.ctx.ptr,
                                                self.ptr,
                                                &domain_ptr))
        assert(domain_ptr != NULL)
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> 2
        cdef int typeid = _numpy_typeid(self._get_type())
        assert (typeid != np.NPY_NOTYPE)
        cdef np.ndarray domain_array = \
            np.PyArray_SimpleNewFromData(1, shape, typeid, domain_ptr)
        return domain_array[0], domain_array[1]


cdef class Domain(object):
    """Class representing the domain of a TileDB Array.

    :param tiledb.Ctx ctx: A TileDB Context
    :param *dims: one or more tiledb.Dim objects up to the Domain's ndim
    :raises TypeError: All dimensions must have the same dtype
    :raises: :py:exc:`TileDBError`

    """

    cdef Ctx ctx
    cdef tiledb_domain_t* ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            tiledb_domain_free(&self.ptr)

    @staticmethod
    cdef from_ptr(Ctx ctx, const tiledb_domain_t* ptr):
        """Constructs an Domain class instance from a (non-null) tiledb_domain_t pointer"""
        assert(ptr != NULL)
        cdef Domain dom = Domain.__new__(Domain)
        dom.ctx = ctx
        dom.ptr = <tiledb_domain_t*> ptr
        return dom

    def __init__(self, Ctx ctx, *dims):
        cdef Py_ssize_t ndim = len(dims)
        if ndim == 0:
            raise TileDBError("Domain must have ndim >= 1")
        cdef Dim dimension = dims[0]
        cdef tiledb_datatype_t domain_type = dimension._get_type()
        for i in range(1, ndim):
            dimension = dims[i]
            if dimension._get_type() != domain_type:
                raise TypeError("all dimensions must have the same dtype")
        cdef tiledb_domain_t* domain_ptr = NULL
        cdef int rc = tiledb_domain_alloc(ctx.ptr, &domain_ptr)
        if rc != TILEDB_OK:
            check_error(ctx, rc)
        assert(domain_ptr != NULL)
        for i in range(ndim):
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
        if not isinstance(other, Domain):
            return False
        ndim = self.ndim
        if (ndim != other.ndim or
            self.dtype != other.dtype or
            self.shape != other.shape):
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

    cdef tiledb_datatype_t _get_type(Domain self) except? TILEDB_CHAR:
        cdef tiledb_datatype_t typ
        check_error(self.ctx,
                    tiledb_domain_get_type(self.ctx.ptr, self.ptr, &typ))
        return typ

    @property
    def dtype(self):
        """The numpy dtype of the domain's dimension type.

        :rtype: numpy.dtype

        """
        cdef tiledb_datatype_t typ = self._get_type()
        return np.dtype(_numpy_type(typ))

    cdef _integer_domain(Domain self):
        cdef tiledb_datatype_t typ = self._get_type()
        if typ == TILEDB_FLOAT32 or typ == TILEDB_FLOAT64:
            return False
        return True

    cdef _shape(Domain self):
        return tuple(self.dim(i).shape[0] for i in range(self.ndim))

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

    def dim(self, int idx):
        """Returns a Dim object from the domain given the dimension's index.

        :param int idx: dimension index
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef tiledb_dimension_t* dim_ptr = NULL
        check_error(self.ctx,
                    tiledb_domain_get_dimension_from_index(
                        self.ctx.ptr, self.ptr, idx, &dim_ptr))
        assert(dim_ptr != NULL)
        return Dim.from_ptr(self.ctx, dim_ptr)

    def dim(self, unicode name):
        """Returns a Dim object from the domain given the dimension's index.

        :param str name: dimension name (label)
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef bytes uname = ustring(name).encode('UTF-8')
        cdef const char* name_ptr = uname
        cdef tiledb_dimension_t* dim_ptr = NULL
        check_error(self.ctx,
                    tiledb_domain_get_dimension_from_name(
                        self.ctx.ptr, self.ptr, name_ptr, &dim_ptr))
        assert(dim_ptr != NULL)
        return Dim.from_ptr(self.ctx, dim_ptr)

    def dump(self):
        """Dumps a string representation of the domain object to standard output (STDOUT)"""
        check_error(self.ctx,
                    tiledb_domain_dump(self.ctx.ptr, self.ptr, stdout))
        print("\n")
        return

cdef class KVSchema(object):
    """
    Schema class for TileDB key-value (assocative) arrays.

    **Note**: Only string-valued attributes are currently supported on KVs.

    :param tiledb.Ctx ctx: A TileDB Context
    :param attrs: one or more array attributes
    :type attrs: tuple(tiledb.Attr, ...)
    :param int capacity: tile cell capacity
    :raises: :py:exc:`tiledb.TileDBError`

    """
    cdef Ctx ctx
    cdef tiledb_kv_schema_t* ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            tiledb_kv_schema_free(&self.ptr)

    @staticmethod
    cdef from_ptr(Ctx ctx, const tiledb_kv_schema_t* ptr):
        """Constructs an KV class instance from a URI and a tiledb_kv_schema_t pointer
        """
        assert(ptr != NULL)
        cdef KVSchema schema = KVSchema.__new__(KVSchema)
        schema.ctx = ctx
        # need to cast away const
        schema.ptr = <tiledb_kv_schema_t*> ptr
        return schema

    @staticmethod
    def load(Ctx ctx, uri, key=None):
        """Loads a persisted KV array at given URI, returns an KV class instance
        """
        cdef tiledb_ctx_t* ctx_ptr = ctx.ptr
        cdef bytes buri = unicode_path(uri)
        cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
        cdef tiledb_kv_schema_t* schema_ptr = NULL
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
            #TODO: unsafe cast here ssize_t -> uint64_t;t
            key_len = <unsigned int> PyBytes_GET_SIZE(bkey)
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_kv_schema_load_with_key(ctx_ptr, uri_ptr, key_type, key_ptr, key_len, &schema_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return KVSchema.from_ptr(ctx, schema_ptr)

    def __init__(self, Ctx ctx,
                 domain=None,
                 attrs=(),
                 capacity=None):
        cdef tiledb_ctx_t* ctx_ptr = ctx.ptr
        cdef tiledb_kv_schema_t* schema_ptr = NULL
        cdef int rc = tiledb_kv_schema_alloc(ctx.ptr, &schema_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        # add attributes, if no defined then _schema_check() will raise a TileDBError
        cdef tiledb_attribute_t* attr_ptr = NULL
        try:
            for attr in attrs:
                if not isinstance(attr, Attr):
                    tiledb_kv_schema_free(&schema_ptr)
                    raise TypeError("invalid attribute type {0!r}".format(type(attr)))
                attr_ptr = (<Attr> attr).ptr
                check_error(ctx,
                    tiledb_kv_schema_add_attribute(ctx_ptr, schema_ptr, attr_ptr))
        except:
            tiledb_kv_schema_free(&schema_ptr)
            raise
        # set the (sparse array) capacity if it is defined
        cdef uint64_t _capacity = 0
        if capacity is not None:
            try:
                val = int(capacity)
                if val <= 0:
                    raise ValueError("TileDB KVSchema capacity must be >= 0")
                # checked cast
                _capacity = val
                check_error(ctx,
                    tiledb_kv_schema_set_capacity(ctx_ptr, schema_ptr, _capacity))
            except:
                tiledb_kv_schema_free(&schema_ptr)
                raise
        rc = tiledb_kv_schema_check(ctx.ptr, schema_ptr)
        if rc != TILEDB_OK:
            tiledb_kv_schema_free(&schema_ptr)
            check_error(ctx, rc)
        self.ctx = ctx
        self.ptr = schema_ptr

    def __eq__(self, other):
        if not isinstance(other, KVSchema):
            return False
        nattr = self.nattr
        if nattr != other.nattr or self.capacity != other.capacity:
            return False
        for i in range(nattr):
            if self.attr(i) != other.attr(i):
                return False
        return True

    @property
    def capacity(self):
        """The KV array capacity

        :rtype: int
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef uint64_t cap = 0
        check_error(self.ctx,
                    tiledb_kv_schema_get_capacity(self.ctx.ptr, self.ptr, &cap))
        return int(cap)

    @property
    def nattr(self):
        """The number of KV attributes

        :rtype: int
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef unsigned int nattr = 0
        check_error(self.ctx,
                    tiledb_kv_schema_get_attribute_num(self.ctx.ptr, self.ptr, &nattr))
        return nattr

    cdef _attr_name(self, name):
        cdef bytes bname = ustring(name).encode('UTF-8')
        cdef tiledb_attribute_t* attr_ptr = NULL
        check_error(self.ctx,
                    tiledb_kv_schema_get_attribute_from_name(
                        self.ctx.ptr, self.ptr, bname, &attr_ptr))
        return Attr.from_ptr(self.ctx, attr_ptr)

    cdef _attr_idx(self, int idx):
        cdef tiledb_attribute_t* attr_ptr = NULL
        check_error(self.ctx,
                    tiledb_kv_schema_get_attribute_from_index(
                        self.ctx.ptr, self.ptr, idx, &attr_ptr))
        return Attr.from_ptr(self.ctx, attr_ptr)

    def attr(self, object key not None):
        """Returns an Attr instance given an int index or string label

        :param key: attribute index (positional or associative)
        :type key: int or str
        :rtype: tiledb.Attr
        :return: The KVSchema attribute at index or with the given name (label)
        :raises TypeError: invalid key type

        """
        if isinstance(key, (str, unicode)):
            return self._attr_name(key)
        elif isinstance(key, _inttypes):
            return self._attr_idx(int(key))
        raise TypeError("attr indices must be a string name, "
                        "or an integer index, not {0!r}".format(type(key)))

    def dump(self):
        """Dumps a string representation of the array object to standard output (STDOUT)"""
        check_error(self.ctx,
                    tiledb_kv_schema_dump(self.ctx.ptr, self.ptr, stdout))
        print("\n")
        return

cdef class KV(object):
    """Class representing a TileDB KV (key-value) array.

    :param Ctx ctx: A TileDB Context
    :param str uri: URI to persistent KV resource
    :param str mode: (default 'r') Open the KV object in read 'r' or write 'w' mode
    :param str key: (default None) If not None, encryption key to decrypt the KV array
    :param int timestamp: (default None) If not None, open the KV array at a given TileDB timestamp
    :raises TypeError: invalid `uri` type
    :raises: :py:exc:`tiledb.TileDBError`
    """

    cdef Ctx ctx
    cdef unicode uri
    cdef unicode mode
    cdef KVSchema schema
    cdef tiledb_kv_t* ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            tiledb_kv_free(&self.ptr)

    @staticmethod
    cdef from_ptr(Ctx ctx, unicode uri, const tiledb_kv_t* ptr):
        """Constructs an KV class instance from a URI and a tiledb_kv_schema_t pointer
        """
        assert(ptr != NULL)
        cdef KV kv = KV.__new__(KV)
        kv.ctx = ctx
        kv.uri = uri
        # need to cast away const
        kv.ptr = <tiledb_kv_t*> ptr
        return kv

    @staticmethod
    def create(Ctx ctx, uri, KVSchema schema, key=None):
        """Creates a persistent KV at the given URI, returns a KV class instance
        """
        cdef tiledb_ctx_t* ctx_ptr = ctx.ptr
        cdef bytes buri = unicode_path(uri)
        cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
        cdef tiledb_kv_schema_t* schema_ptr = schema.ptr
        # encyrption key
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
            #TODO: unsafe cast here ssize_t -> uint64_t;t
            key_len = <unsigned int> PyBytes_GET_SIZE(bkey)
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_kv_create_with_key(
                ctx_ptr, uri_ptr, schema_ptr, key_type, key_ptr, key_len)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return

    def __init__(self, Ctx ctx, uri, mode='r', key=None, timestamp=None):
        cdef tiledb_ctx_t* ctx_ptr = ctx.ptr
        cdef bytes buri = unicode_path(uri)
        cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
        cdef tiledb_query_type_t query_type
        if mode == 'r':
            query_type = TILEDB_READ
        elif mode == 'w':
            query_type = TILEDB_WRITE
        else:
            raise ValueError("TileDB array mode must be 'r' or 'w'")
        # encyrption key
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
            #TODO: unsafe cast here ssize_t -> uint64_t;t
            key_len = <unsigned int> PyBytes_GET_SIZE(bkey)
        cdef uint64_t _timestamp = 0
        if timestamp is not None:
            _timestamp = <uint64_t> timestamp
        # allocate and then open the array
        cdef tiledb_kv_t* kv_ptr = NULL
        cdef int rc = TILEDB_OK
        rc = tiledb_kv_alloc(ctx_ptr, uri_ptr, &kv_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        if timestamp is None:
            with nogil:
                rc = tiledb_kv_open_with_key(ctx_ptr, kv_ptr, query_type, key_type, key_ptr, key_len)
        else:
            with nogil:
                rc = tiledb_kv_open_at_with_key(ctx_ptr, kv_ptr, query_type, key_type, key_ptr, key_len, _timestamp)
        if rc != TILEDB_OK:
            tiledb_kv_free(&kv_ptr)
            _raise_ctx_err(ctx_ptr, rc)
        cdef KVSchema schema
        try:
            schema = KVSchema.load(ctx, uri, key=key)
        except:
            tiledb_kv_free(&kv_ptr)
            raise
        self.ctx = ctx
        self.uri = unicode(uri)
        self.mode = unicode(mode)
        self.schema = schema
        self.ptr = kv_ptr

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    @property
    def schema(self):
        """The :py:class:`KVSchema` for this key-value store."""
        schema = self.schema
        if schema is None:
            raise TileDBError("Cannot access schema, key-value store is closed")
        return schema

    @property
    def mode(self):
        """The mode this key-value store was opened with."""
        return self.mode

    @property
    def isopen(self):
        """True if this key-value store is currently open."""
        cdef int isopen = 0
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_kv_t* kv_ptr = self.ptr
        cdef int rc = TILEDB_OK
        rc = tiledb_kv_is_open(ctx_ptr, kv_ptr, &isopen)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return isopen == 1

    @property
    def nattr(self):
        """The number of KV array attributes

        :rtype: int
        :raises: :py:exc:`tiledb.TileDBError`

        """
        return self.schema.nattr

    @property
    def timestamp(self):
        """The timestamp the KV was opened at"""
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_kv_t* kv_ptr = self.ptr
        cdef uint64_t timestamp = 0
        cdef int rc = TILEDB_OK
        check_error(self.ctx,
                    tiledb_kv_get_timestamp(ctx_ptr, kv_ptr, &timestamp))
        return int(timestamp)

    def attr(self, object key not None):
        """Returns an Attr instance given an int index or string label

        :param key: attribute index (positional or associative)
        :type key: int or str
        :rtype: tiledb.Attr
        :return: The KV attribute at index or with the given name (label)
        :raises TypeError: invalid key type

        """
        return self.schema.attr(key)

    def close(self):
        """Closes this key-value store"""
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_kv_t* kv_ptr = self.ptr
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_kv_close(ctx_ptr, kv_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        self.schema = None
        return

    def reopen(self, timestamp=None):
        """Reopens a key-value store

        Reopening the array is useful when there were updates to the key-value store after it got opened.

        :raises: :py:exc:`tiledb.TileDBError`
        """
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_kv_t* kv_ptr = self.ptr
        cdef uint64_t _timestamp = 0
        cdef int rc = TILEDB_OK
        if timestamp is None:
            with nogil:
                rc = tiledb_kv_reopen(ctx_ptr, kv_ptr)
        else:
            _timestamp = <uint64_t> timestamp
            with nogil:
                rc = tiledb_kv_reopen_at(ctx_ptr, kv_ptr, _timestamp)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return

    def consolidate(self, key=None):
        """Consolidates KV array updates for increased read performance

        :param key: (default None) If key is not None, consolidate KV with a given key
        :type key: str or bytes
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef bytes buri = unicode_path(self.uri)
        cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
        # encyrption key
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
            rc = tiledb_kv_consolidate_with_key(ctx_ptr, uri_ptr, key_type, key_ptr, key_len)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return

    def dict(self):
        """Return a dict representation of the KV array object

        :rtype: dict
        :return: Python dict of keys and attribute value (tuples)

        """
        return dict(self)

    def __iter__(self):
        """Return an iterator object over KV key, values"""
        return KVIter(self, self.attr(0).name)

    def __setitem__(self, object key, object value):
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_kv_t* kv_ptr = self.ptr
        cdef bytes buri = unicode_path(self.uri)
        cdef bytes bkey = key.encode('UTF-8')
        cdef bytes bvalue = value.encode('UTF-8')

        cdef Attr attr = self.attr(0)
        cdef bytes battr = attr.name.encode('UTF-8')
        cdef const char* battr_ptr = PyBytes_AS_STRING(battr)

        # Create KV item object
        cdef int rc
        cdef tiledb_kv_item_t* kv_item_ptr = NULL
        rc = tiledb_kv_item_alloc(ctx_ptr, &kv_item_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        # add key
        # TODO: specialized for strings
        cdef const void* bkey_ptr = PyBytes_AS_STRING(bkey)
        cdef uint64_t bkey_size = PyBytes_GET_SIZE(bkey)
        rc = tiledb_kv_item_set_key(ctx_ptr, kv_item_ptr,
                                    bkey_ptr, TILEDB_CHAR, bkey_size)
        if rc != TILEDB_OK:
            tiledb_kv_item_free(&kv_item_ptr)
            _raise_ctx_err(ctx_ptr, rc)

        # add value
        # TODO: specialized for strings
        cdef const void* bvalue_ptr = PyBytes_AS_STRING(bvalue)
        cdef uint64_t bvalue_size = PyBytes_GET_SIZE(bvalue)
        rc = tiledb_kv_item_set_value(ctx_ptr, kv_item_ptr, battr_ptr,
                                      bvalue_ptr, TILEDB_CHAR, bvalue_size)
        if rc != TILEDB_OK:
            tiledb_kv_item_free(&kv_item_ptr)
            _raise_ctx_err(ctx_ptr, rc)

        # save items
        rc = tiledb_kv_add_item(ctx_ptr, kv_ptr, kv_item_ptr)
        if rc != TILEDB_OK:
            tiledb_kv_item_free(&kv_item_ptr)
            _raise_ctx_err(ctx_ptr, rc)
        rc = tiledb_kv_flush(ctx_ptr, kv_ptr)
        tiledb_kv_item_free(&kv_item_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return

    def __getitem__(self, object key):
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_kv_t* kv_ptr = self.ptr
        cdef bytes buri = unicode_path(self.uri)
        cdef bytes bkey = key.encode('UTF-8')

        cdef Attr attr = self.attr(0)
        cdef bytes battr = attr.name.encode('UTF-8')
        cdef const char* battr_ptr = PyBytes_AS_STRING(battr)

        # add key
        # TODO: specialized for strings
        cdef const void* bkey_ptr = PyBytes_AS_STRING(bkey)
        cdef uint64_t bkey_size = PyBytes_GET_SIZE(bkey)
        cdef tiledb_kv_item_t* kv_item_ptr = NULL
        cdef int rc = TILEDB_OK
        rc = tiledb_kv_get_item(ctx_ptr, kv_ptr,
                                bkey_ptr, TILEDB_CHAR, bkey_size, &kv_item_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        if kv_item_ptr == NULL:
            tiledb_kv_item_free(&kv_item_ptr)
            raise KeyError(key)

        cdef const void* value_ptr = NULL
        cdef tiledb_datatype_t value_type = TILEDB_CHAR
        cdef uint64_t value_size = 0
        rc = tiledb_kv_item_get_value(ctx_ptr, kv_item_ptr, battr_ptr,
                                      &value_ptr, &value_type, &value_size)
        if rc != TILEDB_OK:
            tiledb_kv_item_free(&kv_item_ptr)
            _raise_ctx_err(ctx_ptr, rc)

        cdef bytes val
        try:
            val = PyBytes_FromStringAndSize(<char*> value_ptr, <Py_ssize_t> value_size)
        finally:
            tiledb_kv_item_free(&kv_item_ptr)
        return val.decode('UTF-8')

    def __contains__(self, key):
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_kv_t* kv_ptr = self.ptr
        cdef bytes bkey = key.encode('UTF-8')
        cdef const void* key_ptr = <void*> PyBytes_AS_STRING(bkey)
        cdef uint64_t key_size = PyBytes_GET_SIZE(bkey)
        cdef int has_key = -1
        cdef int rc = TILEDB_OK
        rc = tiledb_kv_has_key(
            ctx_ptr, kv_ptr, key_ptr, TILEDB_CHAR, key_size, &has_key)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return has_key == 1

    def flush(self):
        """Flush any buffered writes to the KV array."""
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_kv_t* kv_ptr = self.ptr
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_kv_flush(ctx_ptr, kv_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return

    def update(self, *args, **kw):
        """Update a KV object from dict/iterable,

        Has the same semantics as Python dict's `.update()` method
        """
        # add stub dict update implementation for now
        items = dict()
        items.update(*args, **kw)
        for (k, v) in items.items():
            self[k] = v


cdef class KVIter(object):
    """KV iterator object iterates over KV items
    """

    cdef KV kv
    cdef bytes battr
    cdef tiledb_kv_iter_t* ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            tiledb_kv_iter_free(&self.ptr)

    def __init__(self, KV kv, attr):
        cdef bytes battr = attr.encode('UTF-8')

        cdef tiledb_ctx_t* ctx_ptr = kv.ctx.ptr
        cdef tiledb_kv_t* kv_ptr = kv.ptr
        cdef tiledb_kv_iter_t* kv_iter_ptr = NULL
        cdef int rc = TILEDB_OK
        rc = tiledb_kv_iter_alloc(ctx_ptr, kv_ptr, &kv_iter_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        assert(kv_iter_ptr != NULL)
        self.kv = kv
        self.ptr = kv_iter_ptr
        self.battr = battr

    def __iter__(self):
        return self

    def __next__(self):
        cdef tiledb_ctx_t* ctx_ptr = self.kv.ctx.ptr
        cdef tiledb_kv_iter_t* iter_ptr = self.ptr
        cdef int done = 0
        cdef int rc = TILEDB_OK
        rc = tiledb_kv_iter_done(ctx_ptr, iter_ptr, &done)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        if done > 0:
            raise StopIteration()
        cdef tiledb_kv_item_t* kv_item_ptr = NULL
        rc = tiledb_kv_iter_here(ctx_ptr, iter_ptr, &kv_item_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        cdef tiledb_datatype_t dtype
        cdef const char* key_ptr = NULL
        cdef uint64_t key_size = 0
        rc = tiledb_kv_item_get_key(ctx_ptr, kv_item_ptr,
                                    <const void**> (&key_ptr), &dtype, &key_size)
        cdef bytes bkey = PyBytes_FromStringAndSize(key_ptr, <Py_ssize_t> key_size)
        cdef const char* val_ptr = NULL
        cdef uint64_t val_size = 0
        cdef bytes attr_ptr = PyBytes_AS_STRING(self.battr)
        rc = tiledb_kv_item_get_value(ctx_ptr, kv_item_ptr, attr_ptr,
                                      <const void**> (&val_ptr), &dtype, &val_size)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        cdef bytes bval = PyBytes_FromStringAndSize(val_ptr, <Py_ssize_t> val_size)
        rc = tiledb_kv_iter_next(ctx_ptr, iter_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return bkey.decode('UTF-8'), bval.decode('UTF-8')


def index_as_tuple(idx):
    """Forces scalar index objects to a tuple representation"""
    if isinstance(idx, tuple):
        return idx
    return (idx,)


def replace_ellipsis(Domain dom, tuple idx):
    """Replace indexing ellipsis object with slice objects to match the number of dimensions"""
    ndim = dom.ndim
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


def replace_scalars_slice(Domain dom, tuple idx):
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


def index_domain_subarray(Domain dom, tuple idx):
    """
    Return a numpy array representation of the tiledb subarray buffer
    for a given domain and tuple of index slices
    """
    ndim = dom.ndim
    if len(idx) != ndim:
        raise IndexError("number of indices does not match domain raank: "
                         "({:!r} expected {:!r]".format(len(idx), ndim))
    # populate a subarray array / buffer to pass to tiledb
    subarray = np.zeros(shape=(ndim, 2), dtype=dom.dtype)
    for r in range(ndim):
        # extract lower and upper bounds for domain dimension extent
        dim = dom.dim(r)
        (dim_lb, dim_ub) = dim.domain

        dim_slice = idx[r]
        if not isinstance(dim_slice, slice):
            raise IndexError("invalid index type: {!r}".format(dim_slice))

        start, stop, step = dim_slice.start, dim_slice.stop, dim_slice.step
        #if step and step < 0:
        #    raise IndexError("only positive slice steps are supported")

        # Promote to a common type
        if start is not None and stop is not None:
            if type(start) != type(stop):
                promoted_dtype = np.promote_types(type(start), type(stop))
                start = np.array(start, dtype=promoted_dtype)[0]
                stop = np.array(stop, dtype=promoted_dtype)[0]

        if start is not None:
            # don't round / promote fp slices
            if np.issubdtype(dim.dtype, np.integer):
                if not isinstance(start, _inttypes):
                    raise IndexError("cannot index integral domain dimension with floating point slice")
            # apply negative indexing (wrap-around semantics)
            if start < 0:
                start += int(dim_ub) + 1
            if start < dim_lb:
                # numpy allows start value < the array dimension shape,
                # clamp to lower bound of dimension domain
                #start = dim_lb
                raise IndexError("index out of bounds <todo>")
        else:
            start = dim_lb
        if stop is not None:
            # don't round / promote fp slices
            if np.issubdtype(dim.dtype, np.integer):
                if not isinstance(stop, _inttypes):
                    raise IndexError("cannot index integral domain dimension with floating point slice")
            if stop < 0:
                stop += dim_ub
            if stop > dim_ub:
                # numpy allows stop value > than the array dimension shape,
                # clamp to upper bound of dimension domain
                stop = int(dim_ub) + 1
        else:
            if np.issubdtype(dim.dtype, np.floating):
                stop = dim_ub
            else:
                stop = int(dim_ub) + 1
        if np.issubdtype(type(stop), np.floating):
            # inclusive bounds for floating point ranges
            subarray[r, 0] = start
            subarray[r, 1] = stop
        elif np.issubdtype(type(stop), np.integer):
            # normal python indexing semantics
            subarray[r, 0] = start
            subarray[r, 1] = int(stop) - 1
        else:
            raise IndexError("domain indexing is defined for integral and floating point values")
    return subarray


cdef class ArraySchema(object):
    """
    Schema class for TileDB dense / sparse array representations

    :param tiledb.Ctx ctx: A TileDB Context
    :param attrs: one or more array attributes
    :type attrs: tuple(tiledb.Attr, ...)
    :param cell_order:  TileDB label for cell layout
    :type cell_order: 'row-major' or 'C', 'col-major' or 'F'
    :param tile_order:  TileDB label for tile layout
    :type tile_order: 'row-major' or 'C', 'col-major' or 'F', 'unordered'
    :param int capacity: tile cell capacity
    :param coords_compressor: compressor label, level for (sparse) coordinates
    :type coords_compressor: tuple(str, int)
    :param offsets_compressor: compressor label, level for varnum attribute cells
    :type coords_compressor: tuple(str, int)
    :param coords_filters: (default None) coordinate filter list
    :type coords_filters: tiledb.FilterList
    :param offsets_filters: (default None) offsets filter list
    :type offsets_filters: tiledb.FilterList
    :param bool sparse: True if schema is sparse, else False \
        (set by SparseArray and DenseArray derived classes)
    :raises TypeError: cannot convert uri to unicode string
    :raises: :py:exc:`tiledb.TileDBError`

    """
    cdef Ctx ctx
    cdef tiledb_array_schema_t* ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            tiledb_array_schema_free(&self.ptr)

    @staticmethod
    cdef from_ptr(Ctx ctx, const tiledb_array_schema_t* schema_ptr):
        """
        Constructs a ArraySchema class instance from a 
        Ctx and tiledb_array_schema_t pointer
        """
        cdef ArraySchema schema = ArraySchema.__new__(ArraySchema)
        schema.ctx = ctx
        # cast away const
        schema.ptr = <tiledb_array_schema_t*> schema_ptr
        return schema

    @staticmethod
    def load(Ctx ctx, uri, key=None):
        cdef bytes buri = uri.encode('UTF-8')
        cdef tiledb_ctx_t* ctx_ptr = ctx.ptr
        cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
        cdef tiledb_array_schema_t* array_schema_ptr = NULL
        # encyrption key
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
        return ArraySchema.from_ptr(ctx, array_schema_ptr)

    def __init__(self, Ctx ctx,
                 domain=None,
                 attrs=(),
                 cell_order='row-major',
                 tile_order='row-major',
                 capacity=0,
                 coords_compressor=None,
                 offsets_compressor=None,
                 coords_filters=None,
                 offsets_filters=None,
                 sparse=False):
        cdef tiledb_array_type_t array_type =\
            TILEDB_SPARSE if sparse else TILEDB_DENSE
        cdef tiledb_array_schema_t* schema_ptr = NULL
        check_error(ctx,
                    tiledb_array_schema_alloc(ctx.ptr, array_type, &schema_ptr))
        cdef tiledb_layout_t cell_layout = TILEDB_ROW_MAJOR
        cdef tiledb_layout_t tile_layout = TILEDB_ROW_MAJOR
        try:
            cell_layout = _tiledb_layout(cell_order)
            tile_layout = _tiledb_layout(tile_order)
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
        cdef int _level = -1
        cdef tiledb_compressor_t _compressor = TILEDB_NO_COMPRESSION
        if coords_compressor is not None:
            try:
                compressor, level = coords_compressor
                _compressor = _tiledb_compressor(compressor)
                _level = int(level)
                check_error(ctx,
                    tiledb_array_schema_set_coords_compressor(ctx.ptr, schema_ptr, _compressor, _level))
            except:
                tiledb_array_schema_free(&schema_ptr)
                raise
        if offsets_compressor is not None:
            try:
                compressor, level = offsets_compressor
                _compressor = _tiledb_compressor(compressor)
                _level = int(level)
                check_error(ctx,
                    tiledb_array_schema_set_offsets_compressor(ctx.ptr, schema_ptr, _compressor, _level))
            except:
                tiledb_array_schema_free(&schema_ptr)
                raise
        cdef FilterList filter_list
        cdef tiledb_filter_list_t* filter_list_ptr = NULL
        try:

            if offsets_filters is not None:
                if not isinstance(offsets_filters, FilterList):
                    raise TypeError("offsets_filters must be a tiledb.FilterList instance")
                filter_list = offsets_filters
                filter_list_ptr = filter_list.ptr
                check_error(ctx,
                    tiledb_array_schema_set_offsets_filter_list(ctx.ptr, schema_ptr, filter_list_ptr))
            if coords_filters is not None:
                if not isinstance(coords_filters, FilterList):
                    raise TypeError("coords_filters must be a tiledb.FilterList instance")
                filter_list = coords_filters
                filter_list_ptr = filter_list.ptr
                check_error(ctx,
                    tiledb_array_schema_set_coords_filter_list(ctx.ptr, schema_ptr, filter_list_ptr))
        except:
            tiledb_array_schema_free(&schema_ptr)
            raise

        cdef tiledb_domain_t* domain_ptr = (<Domain> domain).ptr
        rc = tiledb_array_schema_set_domain(ctx.ptr, schema_ptr, domain_ptr)
        if rc != TILEDB_OK:
            tiledb_array_schema_free(&schema_ptr)
            _raise_ctx_err(ctx.ptr, rc)
        cdef tiledb_attribute_t* attr_ptr = NULL
        for attr in attrs:
            attr_ptr = (<Attr> attr).ptr
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

    def __eq__(self, other):
        if not isinstance(other, ArraySchema):
            return False
        nattr = self.nattr
        if nattr != other.nattr:
            return False
        if (self.sparse != other.sparse or
            self.cell_order != other.cell_order or
            self.tile_order != other.tile_order):
            return False
        if (self.capacity != other.capacity or
            self.coords_compressor != other.coords_compressor or
            self.offsets_compressor != other.offsets_compressor):
            return False
        if self.domain != other.domain:
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
        return _tiledb_layout_string(order)

    @property
    def coords_compressor(self):
        """The compressor label and level for the array's coordinates.

        :rtype: tuple(str, int)
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef tiledb_compressor_t comp = TILEDB_NO_COMPRESSION
        cdef int level = -1
        check_error(self.ctx,
                    tiledb_array_schema_get_coords_compressor(
                        self.ctx.ptr, self.ptr, &comp, &level))
        return (_tiledb_compressor_string(comp), level)

    @property
    def offsets_compressor(self):
        """The compressor label and level for the array's variable-length attribute offsets.

        :rtype: tuple(str, int)
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef tiledb_compressor_t comp = TILEDB_NO_COMPRESSION
        cdef int level = -1
        check_error(self.ctx,
                    tiledb_array_schema_get_offsets_compressor(
                        self.ctx.ptr, self.ptr, &comp, &level))
        return (_tiledb_compressor_string(comp), level)

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
        return FilterList.from_ptr(self.ctx, filter_list_ptr)

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
        return FilterList.from_ptr(self.ctx, filter_list_ptr)

    @property
    def domain(self):
        """The Domain associated with the array.

        :rtype: tiledb.Domain
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef tiledb_domain_t* dom = NULL
        check_error(self.ctx,
                    tiledb_array_schema_get_domain(self.ctx.ptr, self.ptr, &dom))
        return Domain.from_ptr(self.ctx, dom)

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

    cdef _attr_name(self, name):
        cdef bytes bname = ustring(name).encode('UTF-8')
        cdef tiledb_attribute_t* attr_ptr = NULL
        check_error(self.ctx,
                    tiledb_array_schema_get_attribute_from_name(
                        self.ctx.ptr, self.ptr, bname, &attr_ptr))
        return Attr.from_ptr(self.ctx, attr_ptr)

    cdef _attr_idx(self, int idx):
        cdef tiledb_attribute_t* attr_ptr = NULL
        check_error(self.ctx,
                    tiledb_array_schema_get_attribute_from_index(
                        self.ctx.ptr, self.ptr, idx, &attr_ptr))
        return Attr.from_ptr(self.ctx, attr_ptr)

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

    def dump(self):
        """Dumps a string representation of the array object to standard output (stdout)"""
        check_error(self.ctx,
                    tiledb_array_schema_dump(self.ctx.ptr, self.ptr, stdout))
        print("\n")
        return

cdef class Array(object):
    """Base class for TileDB array objects.

    Defines common properties/functionality for the different array types. When an Array instance is initialized,
    the array is opened with the specified mode.

    :param Ctx ctx: TileDB context
    :param str uri: URI of array to open
    :param str mode: (default 'r') Open the KV object in read 'r' or write 'w' mode
    :param str key: (default None) If not None, encryption key to decrypt the KV array
    :param int timestamp: (default None) If not None, open the KV array at a given TileDB timestamp
    """

    cdef Ctx ctx
    cdef unicode uri
    cdef unicode mode
    cdef object schema
    cdef tiledb_array_t* ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            tiledb_array_free(&self.ptr)

    @staticmethod
    def create(uri, ArraySchema schema, key=None):
        """Creates a persistent TileDB Array at the given URI

        :param str uri: URI at which to create the new empty array.
        :param ArraySchema schema: Schema for the array
        """
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
            #TODO: unsafe cast here ssize_t -> uint64_t;t
            key_len = <unsigned int> PyBytes_GET_SIZE(bkey)

        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_array_create_with_key(ctx_ptr, uri_ptr, schema_ptr, key_type, key_ptr, key_len)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return

    def __init__(self, Ctx ctx, uri, mode='r', key=None, timestamp=None):
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
        cdef uint64_t _timestamp = 0
        if timestamp is not None:
            _timestamp = <uint64_t> timestamp
        # allocate and then open the array
        cdef tiledb_array_t* array_ptr = NULL
        cdef int rc = TILEDB_OK
        rc = tiledb_array_alloc(ctx_ptr, uri_ptr, &array_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        if timestamp is None:
            with nogil:
                rc = tiledb_array_open_with_key(
                    ctx_ptr, array_ptr, query_type, key_type, key_ptr, key_len)
        else:
            with nogil:
                rc = tiledb_array_open_at_with_key(
                    ctx_ptr, array_ptr, query_type, key_type, key_ptr, key_len, _timestamp)
        if rc != TILEDB_OK:
            tiledb_array_free(&array_ptr)
            _raise_ctx_err(ctx_ptr, rc)
        cdef ArraySchema schema
        try:
            schema = ArraySchema.load(ctx, uri, key=key)
        except:
            tiledb_array_free(&array_ptr)
            raise
        self.ctx = ctx
        self.uri = unicode(uri)
        self.mode = unicode(mode)
        self.schema = schema
        self.ptr = array_ptr

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
        """Reopens this array.

        This is useful when the array is updated after it was opened.
        To sync-up with the updates, the user must either close the array and open again, or just use ``reopen()``
        without closing. Reopening will be generally faster than the former alternative.
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
    def shape(self):
        """The shape of this array."""
        return self.schema.shape

    @property
    def nattr(self):
        """The number of attributes of this array."""
        return self.schema.nattr

    @property
    def timestamp(self):
        """Returns the timestamp the array is opened at

        :rtype: int
        :returns: tiledb timestamp at which point the array was opened

        """
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_array_t* array_ptr = self.ptr
        cdef uint64_t timestamp = 0
        cdef int rc = TILEDB_OK
        rc = tiledb_array_get_timestamp(ctx_ptr, array_ptr, &timestamp)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return int(timestamp)

    @property
    def coords_dtype(self):
        """Returns the numpy record array dtype of the array coordinates

        :rtype: numpy.dtype
        :returns: coord array record dtype

        """
        # returns the record array dtype of the coordinate array
        return np.dtype([(str(dim.name), dim.dtype) for dim in self.schema.domain])

    def subarray(self, selection, coords=False, attrs=None, order=None):
        raise NotImplementedError()

    def attr(self, key):
        """Returns an :py:class:`Attr` instance given an int index or string label

        :param key: attribute index (positional or associative)
        :type key: int or str
        :rtype: :py:class:`Attr`
        :return: The array attribute at index or with the given name (label)
        :raises TypeError: invalid key type"""
        return self.schema.attr(key)

    def nonempty_domain(self):
        """Return the minimum bounding domain which encompasses nonempty values.

        :rtype: tuple(tuple(numpy scalar, numpy scalar), ...)
        :return: A list of (inclusive) domain extent tuples, that contain all nonempty cells

        """
        cdef Domain dom = self.schema.domain
        cdef np.ndarray extents = np.zeros(shape=(dom.ndim, 2), dtype=dom.dtype)

        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_array_t* array_ptr = self.ptr
        cdef void* extents_ptr = np.PyArray_DATA(extents)
        cdef int empty = 0
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_array_get_non_empty_domain(ctx_ptr, array_ptr, extents_ptr, &empty)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        if empty > 0:
            return None
        return tuple((extents[i, 0].item(), extents[i, 1].item())
                     for i in range(dom.ndim))

    def consolidate(self, key=None):
        """Consolidates fragments of an array object for increased read performance.

        :param key: (default None) encryption key to decrypt an encrypted array
        :type key: str or bytes
        :raises: :py:exc:`tiledb.TileDBError`

        """
        return consolidate(self.ctx, uri=self.uri, key=key)

    def dump(self):
        self.schema.dump()


cdef class Query(object):
    """
    Proxy object returned by query() to index into original array
    on a subselection of attribution in a defined layout order

    """

    cdef Array array
    cdef object attrs
    cdef object coords
    cdef object order

    def __init__(self, array, attrs=None, coords=False, order='C'):
        if array.mode != 'r':
            raise ValueError("array mode must be read-only")
        self.array = array
        self.attrs = attrs
        self.coords = coords
        self.order = order

    def __getitem__(self, object selection):
        return self.array.subarray(selection,
                                   attrs=self.attrs,
                                   coords=self.coords,
                                   order=self.order)

cdef class DenseArray(Array):
    """Class representing a dense TileDB array.

    Inherits properties and methods of :py:class:`tiledb.Array`.

    """

    @staticmethod
    def from_numpy(Ctx ctx, uri, np.ndarray array, **kw):
        """
        Persists a given numpy array as a TileDB DenseArray,
        returns a readonly DenseArray class instance.

        :param tiledb.Ctx ctx: A TileDB Context
        :param str uri: URI for the TileDB array resource
        :param numpy.ndarray array: dense numpy array to persist
        :param \*\*kw: additional arguments to pass to the DenseArray constructor
        :rtype: tiledb.DenseArray
        :return: A DenseArray with a single anonymous attribute
        :raises TypeError: cannot convert `uri` to unicode string
        :raises: :py:exc:`tiledb.TileDBError`

        **Example:**

        >>> import tiledb, numpy as np, tempfile
        >>> ctx = tiledb.Ctx()
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     # Creates array 'array' on disk.
        ...     A = tiledb.DenseArray.from_numpy(ctx, tmp + "/array",  np.array([1.0, 2.0, 3.0]))

        """
        # create an ArraySchema from the numpy array object
        dims = []
        for d in range(array.ndim):
            extent = array.shape[d]
            domain = (0, extent - 1)
            dims.append(Dim(ctx, "", domain=domain, tile=extent, dtype=np.uint64))
        dom = Domain(ctx, *dims)
        att = Attr(ctx, dtype=array.dtype)
        schema = ArraySchema(ctx, domain=dom, attrs=(att,), **kw)
        Array.create(uri, schema)
        with DenseArray(ctx, uri, mode='w') as arr:
            arr.write_direct(np.ascontiguousarray(array))
        return DenseArray(ctx, uri, mode='r')

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if self.schema.sparse:
            raise ValueError("Array at {} is not a dense array".format(self.uri))
        return

    def __len__(self):
        return self.domain.shape[0]

    def __getitem__(self, object selection):
        """Retrieve data cells for an item or region of the array.

        :param tuple selection: An int index, slice or tuple of integer/slice objects,
            specifiying the selected subarray region for each dimension of the DenseArray.
        :rtype: :py:class:`numpy.ndarray` or :py:class:`collections.OrderedDict`
        :returns: If the dense array has a single attribute than a Numpy array of corresponding shape/dtype \
                is returned for that attribute.  If the array has multiple attributes, a \
                :py:class:`collections.OrderedDict` is with dense Numpy subarrays for each attribute.
        :raises IndexError: invalid or unsupported index selection
        :raises: :py:exc:`tiledb.TileDBError`

        **Example:**

        >>> import tiledb, numpy as np, tempfile
        >>> ctx = tiledb.Ctx()
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     # Creates array 'array' on disk.
        ...     A = tiledb.DenseArray.from_numpy(ctx, tmp + "/array",  np.ones((100, 100)))
        ...     # Many aspects of Numpy's fancy indexing are supported:
        ...     A[1:10, ...].shape
        ...     A[1:10, 20:99].shape
        ...     A[1, 2].shape
        (9, 100)
        (9, 79)
        ()
        >>> # Subselect on attributes when reading:
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     dom = tiledb.Domain(ctx, tiledb.Dim(ctx, domain=(0, 9), tile=2, dtype=np.uint64))
        ...     schema = tiledb.ArraySchema(ctx, domain=dom,
        ...         attrs=(tiledb.Attr(ctx, name="a1", dtype=np.int64),
        ...                tiledb.Attr(ctx, name="a2", dtype=np.int64)))
        ...     tiledb.DenseArray.create(tmp + "/array", schema)
        ...     with tiledb.DenseArray(ctx, tmp + "/array", mode='w') as A:
        ...         A[0:10] = {"a1": np.zeros((10)), "a2": np.ones((10))}
        ...     with tiledb.DenseArray(ctx, tmp + "/array", mode='r') as A:
        ...         # Access specific attributes individually.
        ...         A[0:5]["a1"]
        ...         A[0:5]["a2"]
        array([0, 0, 0, 0, 0])
        array([1, 1, 1, 1, 1])

        """
        return self.subarray(selection)


    def query(self, attrs=None, coords=False, order='C'):
        """
        Construct a proxy Query object for easy subarray queries of cells
        for an item or region of the array across one or more attributes.

        Optionally subselect over attributes, return dense result coordinate values,
        and specify a layout a result layout / cell-order.

        :param attrs: the DenseArray attributes to subselect over.
            If attrs is None (default) all array attributes will be returned.
            Array attributes can be defined by name or by positional index.
        :param coords: if True, return array of coodinate value (default False).
        :param order: 'C', 'F', or 'G' (row-major, col-major, tiledb global order)
        :return: A proxy Query object that can be used for indexing into the DenseArray
            over the defined attributes, in the given result layout (order).

        :raises ValueError: array is not opened for reads (mode = 'r')
        :raises: :py:exc:`tiledb.TileDBError`

        **Example:**

        >>> # Subselect on attributes when reading:
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     dom = tiledb.Domain(ctx, tiledb.Dim(ctx, domain=(0, 9), tile=2, dtype=np.uint64))
        ...     schema = tiledb.ArraySchema(ctx, domain=dom,
        ...         attrs=(tiledb.Attr(ctx, name="a1", dtype=np.int64),
        ...                tiledb.Attr(ctx, name="a2", dtype=np.int64)))
        ...     tiledb.DenseArray.create(tmp + "/array", schema)
        ...     with tiledb.DenseArray(ctx, tmp + "/array", mode='w') as A:
        ...         A[0:10] = {"a1": np.zeros((10)), "a2": np.ones((10))}
        ...     with tiledb.DenseArray(ctx, tmp + "/array", mode='r') as A:
        ...         # Access specific attributes individually.
        ...         A.query(attrs=("a1",))[0:5]
        OrderedDict([('a1', array([0, 0, 0, 0, 0]))])

        """
        if not self.isopen or self.mode != 'r':
            raise TileDBError("DenseArray is not opened for reading")
        return Query(self, attrs=attrs, coords=coords, order=order)


    def subarray(self, selection, coords=False, attrs=None, order=None):
        """Retrieve data cells for an item or region of the array.

        Optionally subselect over attributes, return dense result coordinate values,
        and specify a layout a result layout / cell-order.

        :param selection: tuple of scalar and/or slice objects
        :param coords: if True, return array of coordinate value (default False).
        :param attrs: the DenseArray attributes to subselect over.
            If attrs is None (default) all array attributes will be returned.
            Array attributes can be defined by name or by positional index.
        :param order: 'C', 'F', or 'G' (row-major, col-major, tiledb global order)
        :returns: If the dense array has a single attribute than a Numpy array of corresponding shape/dtype \
            is returned for that attribute.  If the array has multiple attributes, a \
            :py:class:`collections.OrderedDict` is with dense Numpy subarrays for each attribute.
        :raises IndexError: invalid or unsupported index selection
        :raises: :py:exc:`tiledb.TileDBError`

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
        else:
            raise ValueError("order must be 'C' (TILEDB_ROW_MAJOR), 'F' (TILEDB_COL_MAJOR), or 'G' (TILEDB_GLOBAL_ORDER)")
        attr_names = list()
        if coords:
            attr_names.append("coords")
        if attrs is None:
            attr_names.extend(self.schema.attr(i).name for i in range(self.schema.nattr))
        else:
            attr_names.extend(self.schema.attr(a).name for a in attrs)
        selection = index_as_tuple(selection)
        idx = replace_ellipsis(self.schema.domain, selection)
        idx, drop_axes = replace_scalars_slice(self.schema.domain, idx)
        subarray = index_domain_subarray(self.schema.domain, idx)
        out = self._read_dense_subarray(subarray, attr_names, layout)
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
                return out[attr.name]
        return out
 
    cdef _read_dense_subarray(self, np.ndarray subarray, list attr_names, tiledb_layout_t layout):
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_array_t* array_ptr = self.ptr

        cdef Py_ssize_t nattr = len(attr_names)
        cdef tuple shape = \
            tuple(int(subarray[r, 1]) - int(subarray[r, 0]) + 1
                  for r in range(self.schema.ndim))
        
        cdef np.ndarray buffer_sizes = np.zeros((nattr,),  dtype=np.uint64)
        out = OrderedDict()
        for i in range(nattr):
            name = attr_names[i]
            if name == "coords":
                dtype = self.coords_dtype
            else:
                dtype = self.schema.attr(name).dtype
            if layout == TILEDB_ROW_MAJOR:
                buffer = np.empty(shape=shape, dtype=dtype, order='C')
            elif layout == TILEDB_COL_MAJOR:
                buffer = np.empty(shape=shape, dtype=dtype, order='F')
            else:
                buffer = np.empty(shape=np.prod(shape), dtype=dtype)
            buffer_sizes[i] = buffer.nbytes
            out[name] = buffer

        cdef tiledb_query_t* query_ptr = NULL
        cdef int rc = TILEDB_OK
        rc = tiledb_query_alloc(ctx_ptr, array_ptr, TILEDB_READ, &query_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        rc = tiledb_query_set_layout(ctx_ptr, query_ptr, layout)
        if rc != TILEDB_OK:
            tiledb_query_free(&query_ptr)
            _raise_ctx_err(ctx_ptr, rc)

        cdef void* subarray_ptr = np.PyArray_DATA(subarray)
        rc = tiledb_query_set_subarray(ctx_ptr, query_ptr, subarray_ptr)
        if rc != TILEDB_OK:
            tiledb_query_free(&query_ptr)
            _raise_ctx_err(ctx_ptr, rc)

        cdef bytes battr_name
        cdef void* buffer_ptr = NULL
        cdef uint64_t* buffer_sizes_ptr = <uint64_t*> np.PyArray_DATA(buffer_sizes)
        try:
            for i, (name, buffer) in enumerate(out.items()):
                buffer_ptr = np.PyArray_DATA(buffer)
                if name == "coords":
                    rc = tiledb_query_set_buffer(ctx_ptr, query_ptr, tiledb_coords(),
                                                 buffer_ptr, &(buffer_sizes_ptr[i]))
                else:
                    battr_name = name.encode('UTF-8')
                    rc = tiledb_query_set_buffer(ctx_ptr, query_ptr, battr_name,
                                                 buffer_ptr, &(buffer_sizes_ptr[i]))
                if rc != TILEDB_OK:
                    _raise_ctx_err(ctx_ptr, rc)
        except:
            tiledb_query_free(&query_ptr)
            raise
        with nogil:
            rc = tiledb_query_submit(ctx_ptr, query_ptr)
        tiledb_query_free(&query_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return out

    def __setitem__(self, object selection, object val):
        """Set / update dense data cells

        :param tuple selection: An int index, slice or tuple of integer/slice objects,
            specifiying the selected subarray region for each dimension of the DenseArray.
        :param value: a dictionary of array attribute values, values must able to be converted to n-d numpy arrays.\
            if the number of attributes is one, than a n-d numpy array is accepted.
        :type value: dict or :py:class:`numpy.ndarray`
        :raises IndexError: invalid or unsupported index selection
        :raises ValueError: value / coordinate length mismatch
        :raises: :py:exc:`tiledb.TileDBError`

        **Example:**

        >>> import tiledb, numpy as np, tempfile
        >>> ctx = tiledb.Ctx()
        >>> # Write to single-attribute 2D array
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     # Create an array initially with all zero values
        ...     with tiledb.DenseArray.from_numpy(ctx, tmp + "/array",  np.zeros((2, 2))) as A:
        ...         pass
        ...     with tiledb.DenseArray(ctx, tmp + "/array", mode='w') as A:
        ...         # Write to the single (anonymous) attribute
        ...         A[:] = np.array(([1,2], [3,4]))
        >>>
        >>> # Write to multi-attribute 2D array
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     dom = tiledb.Domain(ctx,
        ...         tiledb.Dim(ctx, domain=(0, 1), tile=2, dtype=np.uint64),
        ...         tiledb.Dim(ctx, domain=(0, 1), tile=2, dtype=np.uint64))
        ...     schema = tiledb.ArraySchema(ctx, domain=dom,
        ...         attrs=(tiledb.Attr(ctx, name="a1", dtype=np.int64),
        ...                tiledb.Attr(ctx, name="a2", dtype=np.int64)))
        ...     tiledb.DenseArray.create(tmp + "/array", schema)
        ...     with tiledb.DenseArray(ctx, tmp + "/array", mode='w') as A:
        ...         # Write to each attribute
        ...         A[0:2, 0:2] = {"a1": np.array(([-3, -4], [-5, -6])),
        ...                        "a2": np.array(([1, 2], [3, 4]))}

        """
        if not self.isopen or self.mode != 'w':
            raise TileDBError("DenseArray is not opened for writing")
        cdef Domain domain = self.domain
        cdef tuple idx = replace_ellipsis(domain, index_as_tuple(selection))
        cdef np.ndarray subarray = index_domain_subarray(domain, idx)
        cdef Attr attr
        cdef list attributes = list()
        cdef list values = list()
        if isinstance(val, dict):
            for (k, v) in val.items():
                attr = self.schema.attr(k)
                attributes.append(attr.name)
                values.append(
                    np.ascontiguousarray(v, dtype=attr.dtype))
        elif np.isscalar(val):
            for i in range(self.schema.nattr):
                attr = self.schema.attr(i)
                subarray_shape = tuple(int(subarray[r, 1] - subarray[r, 0]) + 1
                                       for r in range(subarray.shape[0]))
                attributes.append(attr.name)
                A = np.empty(subarray_shape, dtype=attr.dtype)
                A[:] = val
                values.append(A)
        elif self.schema.nattr == 1:
            attr = self.schema.attr(0)
            attributes.append(attr.name)
            values.append(
                np.ascontiguousarray(val, dtype=attr.dtype))
        else:
            raise ValueError("ambiguous attribute assignment, "
                             "more than one array attribute")
        # Check value layouts
        nattr = len(attributes)
        value = values[0]
        isfortran = value.ndim > 1 and value.flags.f_contiguous
        if nattr > 1:
            for i in range(1, nattr):
                value = values[i]
                if value.ndim > 1 and value.flags.f_contiguous and not isfortran:
                    raise ValueError("mixed C and Fortran array layouts")
        self._write_dense_subarray(subarray, attributes, values, isfortran)
        return

    cdef _write_dense_subarray(self, np.ndarray subarray, list attributes, list values, int isfortran):
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_array_t* array_ptr = self.ptr

        cdef Py_ssize_t nattr = len(attributes)
        cdef np.ndarray buffer_sizes = np.zeros((nattr,),  dtype=np.uint64)
        for i in range(nattr):
            buffer_sizes[i] = values[i].nbytes

        cdef tiledb_query_t* query_ptr = NULL
        cdef int rc = TILEDB_OK
        rc = tiledb_query_alloc(ctx_ptr, array_ptr, TILEDB_WRITE, &query_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)

        cdef tiledb_layout_t layout = TILEDB_COL_MAJOR if isfortran else TILEDB_ROW_MAJOR
        rc = tiledb_query_set_layout(ctx_ptr, query_ptr, layout)
        if rc != TILEDB_OK:
            tiledb_query_free(&query_ptr)
            _raise_ctx_err(ctx_ptr, rc)

        cdef void* subarray_ptr = np.PyArray_DATA(subarray)
        rc = tiledb_query_set_subarray(ctx_ptr, query_ptr, subarray_ptr)
        if rc != TILEDB_OK:
            tiledb_query_free(&query_ptr)
            _raise_ctx_err(ctx_ptr, rc)

        cdef bytes battr_name
        cdef void* buffer_ptr = NULL
        cdef uint64_t* buffer_sizes_ptr = <uint64_t*> np.PyArray_DATA(buffer_sizes)
        try:
            for i in range(nattr):
                battr_name = attributes[i].encode('UTF-8')
                buffer_ptr = np.PyArray_DATA(values[i])
                rc = tiledb_query_set_buffer(ctx_ptr, query_ptr, battr_name,
                                             buffer_ptr, &(buffer_sizes_ptr[i]))
                if rc != TILEDB_OK:
                    _raise_ctx_err(ctx_ptr, rc)
        except:
            tiledb_query_free(&query_ptr)
            raise

        with nogil:
            rc = tiledb_query_submit(ctx_ptr, query_ptr)
        tiledb_query_free(&query_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return

    def __array__(self, dtype=None, **kw):
        if self.schema.nattr > 1:
            raise ValueError("cannot create numpy array from TileDB array with more than one attribute")
        array = self.read_direct(name = self.schema.attr(0).name)
        if dtype and array.dtype != dtype:
            return array.astype(dtype)
        return array

    def write_direct(self, np.ndarray array not None):
        """
        Write directly to given array attribute with minimal checks,
        assumes that the numpy array is the same shape as the array's domain

        :param np.ndarray array: Numpy contigous dense array of the same dtype \
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
        cdef bytes battr_name = attr.name.encode('UTF-8')
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
        rc = tiledb_query_set_layout(ctx_ptr, query_ptr, layout)
        if rc != TILEDB_OK:
            tiledb_query_free(&query_ptr)
            _raise_ctx_err(ctx_ptr, rc)
        rc = tiledb_query_set_buffer(ctx_ptr, query_ptr, attr_name_ptr, buff_ptr, &buff_size)
        if rc != TILEDB_OK:
            tiledb_query_free(&query_ptr)
            _raise_ctx_err(ctx_ptr, rc)
        with nogil:
            rc = tiledb_query_submit(ctx_ptr, query_ptr)
        if rc != TILEDB_OK:
            tiledb_query_free(&query_ptr)
            _raise_ctx_err(ctx_ptr, rc)
        with nogil:
            rc = tiledb_query_finalize(ctx_ptr, query_ptr)
        tiledb_query_free(&query_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
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
        cdef bytes battr_name
        if name is None and self.schema.nattr != 1:
            raise ValueError(
                "read_direct with no provided attribute is ambiguous for multi-attribute arrays")
        elif name is None:
            attr = self.schema.attr(0)
            battr_name = attr.name.encode('UTF-8')
        else:
            attr = self.schema.attr(name)
            battr_name = attr.name.encode('UTF-8')
        cdef const char* attr_name_ptr = PyBytes_AS_STRING(battr_name)

        order = 'C'
        cdef tiledb_layout_t cell_layout = TILEDB_ROW_MAJOR
        if self.schema.cell_order == 'col-major' and self.schema.tile_order == 'col-major':
            order = 'F'
            cell_layout = TILEDB_COL_MAJOR

        out = np.empty(self.schema.domain.shape, dtype=attr.dtype, order=order)

        cdef void* buff_ptr = np.PyArray_DATA(out)
        cdef uint64_t buff_size = out.nbytes

        cdef tiledb_query_t* query_ptr = NULL
        cdef int rc = TILEDB_OK
        rc = tiledb_query_alloc(ctx_ptr, array_ptr, TILEDB_READ, &query_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        rc = tiledb_query_set_layout(ctx_ptr, query_ptr, cell_layout)
        if rc != TILEDB_OK:
            tiledb_query_free(&query_ptr)
            _raise_ctx_err(ctx_ptr, rc)
        rc = tiledb_query_set_buffer(ctx_ptr, query_ptr, attr_name_ptr, buff_ptr, &buff_size)
        if rc != TILEDB_OK:
            tiledb_query_free(&query_ptr)
            _raise_ctx_err(ctx_ptr, rc)
        with nogil:
            rc = tiledb_query_submit(ctx_ptr, query_ptr)
        tiledb_query_free(&query_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return out


# point query index a tiledb array (zips) columnar index vectors
def index_domain_coords(Domain dom, tuple idx):
    """
    Returns a (zipped) coordinate array representation
    given coordinate indices in numpy's point indexing format
    """
    ndim = len(idx)
    if ndim != dom.ndim:
        raise IndexError("sparse index ndim must match "
                         "domain ndim: {0!r} != {1!r}".format(ndim, dom.ndim))
    idx = tuple(np.asarray(idx[i], dtype=dom.dim(i).dtype)
                for i in range(ndim))
    # check that all sparse coordinates are the same size and dtype
    len0, dtype0 = len(idx[0]), idx[0].dtype
    for i in range(2, ndim):
        if len(idx[i]) != len0:
            raise IndexError("sparse index dimension length mismatch")
        if idx[i].dtype != dtype0:
            raise IndexError("sparse index dimension dtype mismatch")
    # zip coordinates
    return np.column_stack(idx)


cdef class SparseArray(Array):
    """Class representing a sparse TileDB array.

    Inherits properties and methods of :py:class:`tiledb.Array`.
    """

    def __init__(self, *args, **kw):
        super().__init__(*args, **kw)
        if not self.schema.sparse:
            raise ValueError("Array at {:r} is not a sparse array".format(self.uri))
        return

    def __len__(self):
        raise TypeError("SparseArray length is ambiguous; use shape[0]")

    def __setitem__(self, selection, val):
        """Set / update sparse data cells

        :param tuple selection: N coordinate value arrays (dim0, dim1, ...) where N in the ndim of the SparseArray,
            The format follows numpy sparse (point) indexing semantics.
        :param value: a dictionary of nonempty array attribute values, values must able to be converted to 1-d numpy arrays.\
            if the number of attributes is one, than a 1-d numpy array is accepted.
        :type value: dict or :py:class:`numpy.ndarray`
        :raises IndexError: invalid or unsupported index selection
        :raises ValueError: value / coordinate length mismatch
        :raises: :py:exc:`tiledb.TileDBError`

        **Example:**

        >>> import tiledb, numpy as np, tempfile
        >>> ctx = tiledb.Ctx()
        >>> # Write to multi-attribute 2D array
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     dom = tiledb.Domain(ctx,
        ...         tiledb.Dim(ctx, domain=(0, 1), tile=2, dtype=np.uint64),
        ...         tiledb.Dim(ctx, domain=(0, 1), tile=2, dtype=np.uint64))
        ...     schema = tiledb.ArraySchema(ctx, domain=dom, sparse=True,
        ...         attrs=(tiledb.Attr(ctx, name="a1", dtype=np.int64),
        ...                tiledb.Attr(ctx, name="a2", dtype=np.int64)))
        ...     tiledb.SparseArray.create(tmp + "/array", schema)
        ...     with tiledb.SparseArray(ctx, tmp + "/array", mode='w') as A:
        ...         # Write in the corner cells (0,0) and (1,1) only.
        ...         I, J = [0, 1], [0, 1]
        ...         # Write to each attribute
        ...         A[I, J] = {"a1": np.array([1, 2]),
        ...                    "a2": np.array([3, 4])}

        """
        if not self.isopen or self.mode != 'w':
            raise TileDBError("SparseArray is not opened for writing")
        idx = index_as_tuple(selection)
        sparse_coords = index_domain_coords(self.schema.domain, idx)
        ncells = sparse_coords.shape[0]
        if self.schema.nattr == 1 and not isinstance(val, dict):
            attr = self.attr(0)
            name = attr.name
            value = np.asarray(val, dtype=attr.dtype)
            if len(value) != ncells:
                raise ValueError("value length does not match coordinate length")
            sparse_values = dict(((name, value),))
        else:
            sparse_values = dict()
            for (k, v) in dict(val).items():
                attr = self.attr(k)
                name = attr.name
                value = np.asarray(v, dtype=attr.dtype)
                if len(value) != ncells:
                    raise ValueError("value length does not match coordinate length")
                sparse_values[name] = value
        self._write_sparse(sparse_coords, sparse_values)
        return

    cdef _write_sparse(self, np.ndarray coords, dict values):
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_array_t* array_ptr = self.ptr

        # attr names
        cdef Py_ssize_t nattr = len(values) + 1
        cdef np.ndarray buffer_sizes = np.zeros((nattr,),  dtype=np.uint64)
        for (i, buffer) in enumerate(values.values()):
            buffer_sizes[i] = buffer.nbytes
        buffer_sizes[nattr - 1] = coords.nbytes

        cdef tiledb_query_t* query_ptr = NULL
        cdef int rc = TILEDB_OK
        rc = tiledb_query_alloc(ctx_ptr, array_ptr, TILEDB_WRITE, &query_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        rc = tiledb_query_set_layout(ctx_ptr, query_ptr, TILEDB_UNORDERED)
        if rc != TILEDB_OK:
            tiledb_query_free(&query_ptr)
            _raise_ctx_err(ctx_ptr, rc)
        cdef bytes battr_name
        cdef void* buffer_ptr = NULL
        cdef uint64_t* buffer_sizes_ptr = <uint64_t*> np.PyArray_DATA(buffer_sizes)
        try:
            for i, (name, buffer) in enumerate(values.items()):
                battr_name = name.encode('UTF-8')
                buffer_ptr = np.PyArray_DATA(buffer)
                rc = tiledb_query_set_buffer(ctx_ptr, query_ptr, battr_name,
                                             buffer_ptr, &(buffer_sizes_ptr[i]))
                if rc != TILEDB_OK:
                    _raise_ctx_err(ctx_ptr, rc)
        except:
            tiledb_query_free(&query_ptr)
            raise
        buffer_ptr = np.PyArray_DATA(coords)
        rc = tiledb_query_set_buffer(ctx_ptr, query_ptr, tiledb_coords(), buffer_ptr, &(buffer_sizes_ptr[nattr - 1]))
        if rc != TILEDB_OK:
            tiledb_query_free(&query_ptr)
            _raise_ctx_err(ctx_ptr, rc)
        with nogil:
            rc = tiledb_query_submit(ctx_ptr, query_ptr)
        with nogil:
            rc = tiledb_query_finalize(ctx_ptr, query_ptr)
        tiledb_query_free(&query_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return

    def __getitem__(self, object selection):
        """Retrieve nonempty cell data for an item or region of the array

        :param tuple selection: An int index, slice or tuple of integer/slice objects,
            specifying the selected subarray region for each dimension of the SparseArray.
        :rtype: :py:class:`collections.OrderedDict`
        :returns: An OrderedDict is returned with "coords" coordinate values being the first key. \
            "coords" is a Numpy record array representation of the coordinate values of non-empty attribute cells. \
            Nonempty attribute values are returned as Numpy 1-d arrays.
        :raises IndexError: invalid or unsupported index selection
        :raises: :py:exc:`tiledb.TileDBError`

        **Example:**

        >>> import tiledb, numpy as np, tempfile
        >>> ctx = tiledb.Ctx()
        >>> # Write to multi-attribute 2D array
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     dom = tiledb.Domain(ctx,
        ...         tiledb.Dim(ctx, name="y", domain=(0, 9), tile=2, dtype=np.uint64),
        ...         tiledb.Dim(ctx, name="x", domain=(0, 9), tile=2, dtype=np.uint64))
        ...     schema = tiledb.ArraySchema(ctx, domain=dom, sparse=True,
        ...         attrs=(tiledb.Attr(ctx, name="a1", dtype=np.int64),
        ...                tiledb.Attr(ctx, name="a2", dtype=np.int64)))
        ...     tiledb.SparseArray.create(tmp + "/array", schema)
        ...     with tiledb.SparseArray(ctx, tmp + "/array", mode='w') as A:
        ...         # Write in the twp cells (0,0) and (2,3) only.
        ...         I, J = [0, 2], [0, 3]
        ...         # Write to each attribute
        ...         A[I, J] = {"a1": np.array([1, 2]),
        ...                    "a2": np.array([3, 4])}
        ...     with tiledb.SparseArray(ctx, tmp + "/array", mode='r') as A:
        ...         # Return an OrderedDict with cell coordinates
        ...         A[0:3, 0:10]
        ...         # Return the NumpyRecord array of TileDB cell coordinates
        ...         A[0:3, 0:10]["coords"]
        ...         # Return just the "x" coordinates values
        ...         A[0:3, 0:10]["coords"]["x"]
        OrderedDict([('coords', array([(0, 0), (2, 3)],
              dtype=[('y', '<u8'), ('x', '<u8')])), ('a1', array([1, 2])), ('a2', array([3, 4]))])
        array([(0, 0), (2, 3)],
              dtype=[('y', '<u8'), ('x', '<u8')])
        array([0, 3], dtype=uint64)

        With a floating-point array domain, index bounds are inclusive, e.g.:

        >>> # Return nonempty cells within a floating point array domain (fp index bounds are inclusive):
        >>> # A[5.0:579.9]

        """
        return self.subarray(selection)

    def query(self, attrs=None, coords=True, order='C'):
        """
        Construct a proxy Query object for easy subarray queries of cells
        for an item or region of the array across one or more attributes.

        Optionally subselect over attributes, return dense result coordinate values,
        and specify a layout a result layout / cell-order.

        :param attrs: the SparseArray attributes to subselect over.
            If attrs is None (default) all array attributes will be returned.
            Array attributes can be defined by name or by positional index.
        :param coords: if True, return array of coodinate value (default False).
        :param order: 'C', 'F', or 'G' (row-major, col-major, tiledb global order)
        :return: A proxy Query object that can be used for indexing into the SparseArray
            over the defined attributes, in the given result layout (order).

        **Example:**

        >>> import tiledb, numpy as np, tempfile
        >>> ctx = tiledb.Ctx()
        >>> # Write to multi-attribute 2D array
        >>> with tempfile.TemporaryDirectory() as tmp:
        ...     dom = tiledb.Domain(ctx,
        ...         tiledb.Dim(ctx, name="y", domain=(0, 9), tile=2, dtype=np.uint64),
        ...         tiledb.Dim(ctx, name="x", domain=(0, 9), tile=2, dtype=np.uint64))
        ...     schema = tiledb.ArraySchema(ctx, domain=dom, sparse=True,
        ...         attrs=(tiledb.Attr(ctx, name="a1", dtype=np.int64),
        ...                tiledb.Attr(ctx, name="a2", dtype=np.int64)))
        ...     tiledb.SparseArray.create(tmp + "/array", schema)
        ...     with tiledb.SparseArray(ctx, tmp + "/array", mode='w') as A:
        ...         # Write in the twp cells (0,0) and (2,3) only.
        ...         I, J = [0, 2], [0, 3]
        ...         # Write to each attribute
        ...         A[I, J] = {"a1": np.array([1, 2]),
        ...                    "a2": np.array([3, 4])}
        ...     with tiledb.SparseArray(ctx, tmp + "/array", mode='r') as A:
        ...         A.query(attrs=("a1",), coords=False, order='G')[0:3, 0:10]
        OrderedDict([('a1', array([1, 2]))])
        
        """
        if not self.isopen:
            raise TileDBError("SparseArray is not opened")
        return Query(self, attrs=attrs, coords=coords, order=order)

    def subarray(self, selection, coords=True, attrs=None, order=None):
        """
        Retrieve coordinate and data cells for an item or region of the array.

        Optionally subselect over attributes, return sparse result coordinate values,
        and specify a layout a result layout / cell-order.

        :param selection: tuple of scalar and/or slice objects
        :param coords: if True, return array of coordinate value (default True).
        :param attrs: the SparseArray attributes to subselect over.
            If attrs is None (default) all array attributes will be returned.
            Array attributes can be defined by name or by positional index.
        :param order: 'C', 'F', or 'G' (row-major, col-major, tiledb global order)
        :returns: An OrderedDict is returned with "coords" coordinate values being the first key. \
            "coords" is a Numpy record array representation of the coordinate values of non-empty attribute cells. \
            Nonempty attribute values are returned as Numpy 1-d arrays.


        """
        if not self.isopen or self.mode != 'r':
            raise TileDBError("SparseArray is not opened for reading")
        cdef tiledb_layout_t layout = TILEDB_UNORDERED
        if order is None or order == 'C':
            layout = TILEDB_ROW_MAJOR
        elif order == 'F':
            layout = TILEDB_COL_MAJOR
        elif order == 'G':
            layout = TILEDB_GLOBAL_ORDER
        else:
            raise ValueError("order must be 'C' (TILEDB_ROW_MAJOR), 'F' (TILEDB_COL_MAJOR), or 'G' (TILEDB_GLOBAL_ORDER)")
        attr_names = list()
        if coords:
            attr_names.append("coords")
        if attrs is None:
            attr_names.extend(self.schema.attr(i).name for i in range(self.schema.nattr))
        else:
            attr_names.extend(self.schema.attr(a).name for a in attrs)
        dom = self.schema.domain
        idx = index_as_tuple(selection)
        idx = replace_ellipsis(dom, idx)
        idx, drop_axes = replace_scalars_slice(dom, idx)
        subarray = index_domain_subarray(dom, idx)
        return self._read_sparse_subarray(subarray, attr_names, layout)

    cdef _read_sparse_subarray(self, np.ndarray subarray, list attr_names, tiledb_layout_t layout):
        # ctx references
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef tiledb_array_t* array_ptr = self.ptr

        # set subarray / layout
        cdef void* subarray_ptr = np.PyArray_DATA(subarray)
        cdef tiledb_query_t* query_ptr = NULL
        cdef int rc = TILEDB_OK
        rc = tiledb_query_alloc(ctx_ptr, array_ptr, TILEDB_READ, &query_ptr)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        rc = tiledb_query_set_layout(ctx_ptr, query_ptr, layout)
        if rc != TILEDB_OK:
            tiledb_query_free(&query_ptr)
            _raise_ctx_err(ctx_ptr, rc)
        rc = tiledb_query_set_subarray(ctx_ptr, query_ptr, <void*> subarray_ptr)
        if rc != TILEDB_OK:
            tiledb_query_free(&query_ptr)
            _raise_ctx_err(ctx_ptr, rc)

        # check the max read buffer size
        cdef Py_ssize_t nattr = len(attr_names)
        cdef uint64_t* buffer_sizes_ptr = <uint64_t*> PyMem_Malloc(nattr * sizeof(uint64_t))
        if buffer_sizes_ptr == NULL:
            tiledb_query_free(&query_ptr)
            raise MemoryError()

        cdef bytes battr_name
        try:
            for i in range(nattr):
                name = attr_names[i]
                if name == "coords":
                    rc = tiledb_array_max_buffer_size(ctx_ptr, array_ptr, tiledb_coords(),
                                                      subarray_ptr, &(buffer_sizes_ptr[0]))
                else:
                    battr_name = name.encode('UTF-8')
                    rc = tiledb_array_max_buffer_size(ctx_ptr, array_ptr, battr_name,
                                                      subarray_ptr, &(buffer_sizes_ptr[i]))
                if rc != TILEDB_OK:
                    _raise_ctx_err(ctx_ptr, rc)
        except:
            PyMem_Free(buffer_sizes_ptr)
            tiledb_query_free(&query_ptr)
            raise

        # allocate the max read buffer size and set query buffers
        cdef void** buffers_ptr = <void**> PyMem_Malloc(nattr * sizeof(uintptr_t))
        if buffers_ptr == NULL:
            PyMem_Free(buffer_sizes_ptr)
            tiledb_query_free(&query_ptr)
            raise MemoryError()

        # initalize the buffer ptrs
        for i in range(nattr):
            buffers_ptr[i] = <void*> PyMem_Malloc(<size_t>(buffer_sizes_ptr[i]))
            if buffers_ptr[i] == NULL:
                PyMem_Free(buffer_sizes_ptr)
                for j in range(i):
                    PyMem_Free(buffers_ptr[j])
                PyMem_Free(buffers_ptr)
                tiledb_query_free(&query_ptr)
                raise MemoryError()
        try:
            for i in range(nattr):
                name = attr_names[i]
                if name == "coords":
                    rc = tiledb_query_set_buffer(ctx_ptr, query_ptr, tiledb_coords(),
                                                  buffers_ptr[0], &(buffer_sizes_ptr[0]))
                else:
                    battr_name = name.encode('UTF-8')
                    rc = tiledb_query_set_buffer(ctx_ptr, query_ptr, battr_name,
                                                 buffers_ptr[i], &(buffer_sizes_ptr[i]))
                if rc != TILEDB_OK:
                    _raise_ctx_err(ctx_ptr, rc)
        except:
            PyMem_Free(buffer_sizes_ptr)
            for i in range(nattr):
                PyMem_Free(buffers_ptr[i])
            PyMem_Free(buffers_ptr)
            tiledb_query_free(&query_ptr)
            raise

        with nogil:
            rc = tiledb_query_submit(ctx_ptr, query_ptr)
        tiledb_query_free(&query_ptr)
        if rc != TILEDB_OK:
            PyMem_Free(buffer_sizes_ptr)
            for i in range(nattr):
                PyMem_Free(buffers_ptr[i])
            PyMem_Free(buffers_ptr)
            _raise_ctx_err(ctx_ptr, rc)

        # collect a list of dtypes for resulting to construct array
        dtypes = list()
        try:
            for i in range(nattr):
                name = attr_names[i]
                if name == "coords":
                    dtypes.append(self.coords_dtype)
                else:
                    dtypes.append(self.attr(name).dtype)
            # we need to increase the reference count of all dtype objects
            # because PyArray_NewFromDescr steals a reference
            for i in range(nattr):
                Py_INCREF(dtypes[i])
        except:
            PyMem_Free(buffer_sizes_ptr)
            for i in range(nattr):
                PyMem_Free(buffers_ptr[i])
            PyMem_Free(buffers_ptr)
            raise

        cdef object out = OrderedDict()
        # all results are 1-d vectors
        cdef np.npy_intp dims[1]
        for i in range(nattr):
            try:
                name = attr_names[i]
                dtype = dtypes[i]
                dims[0] = buffer_sizes_ptr[i] / dtypes[i].itemsize
                out[name] = \
                    PyArray_NewFromDescr(
                        <PyTypeObject*> np.ndarray,
                        dtype, 1, dims, NULL,
                        PyMem_Realloc(buffers_ptr[i], <size_t>(buffer_sizes_ptr[i])),
                        np.NPY_OWNDATA, <object> NULL)
            except:
                PyMem_Free(buffer_sizes_ptr)
                # the previous buffers are now "owned"
                # by the constructed numpy arrays
                for j in range(i, nattr):
                    PyMem_Free(buffers_ptr[i])
                PyMem_Free(buffers_ptr)
                raise
        PyMem_Free(buffer_sizes_ptr)
        PyMem_Free(buffers_ptr)
        return out


def consolidate(Ctx ctx, uri=None, key=None):
    """Consolidates a TileDB Array updates for improved read performance

    :param tiledb.Ctx ctx: The TileDB Context
    :param str uri: URI to the TileDB Array
    :param str: (default None) Key to decrypt array if the array is encrypted
    :rtype: str or bytes
    :return: path (URI) to the consolidated TileDB Array
    :raises TypeError: cannot convert path to unicode string
    :raises: :py:exc:`tiledb.TileDBError`

    """
    cdef tiledb_ctx_t* ctx_ptr = ctx.ptr
    cdef bytes buri = unicode_path(uri)
    cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
    # encyrption key
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
        rc = tiledb_array_consolidate_with_key(ctx_ptr, uri_ptr, key_type, key_ptr, key_len)
    if rc != TILEDB_OK:
        _raise_ctx_err(ctx_ptr, rc)
    return uri


def group_create(Ctx ctx, uri):
    """Create a TileDB Group object at the specified path (URI)

    :param tiledb.Ctx ctx: The TileDB Context
    :param str uri: URI of the TileDB Group to be created
    :rtype: str
    :return: The URI of the created TileDB Group
    :raises TypeError: cannot convert path to unicode string
    :raises: :py:exc:`tiledb.TileDBError`

    """
    cdef int rc = TILEDB_OK
    cdef tiledb_ctx_t* ctx_ptr = ctx.ptr
    cdef bytes buri = unicode_path(uri)
    cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
    with nogil:
        rc = tiledb_group_create(ctx_ptr, uri_ptr)
    if rc != TILEDB_OK:
        check_error(ctx, rc)
    return uri


def object_type(Ctx ctx, uri):
    """Returns the TileDB object type at the specified path (URI)

    :param tiledb.Ctx ctx: The TileDB Context
    :param str path: path (URI) of the TileDB resource
    :rtype: str
    :return: object type string
    :raises TypeError: cannot convert path to unicode string

    """
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
    elif obj == TILEDB_KEY_VALUE:
        objtype = "kv"
    elif obj == TILEDB_GROUP:
        objtype = "group"
    return objtype


def remove(Ctx ctx, uri):
    """Removes (deletes) the TileDB object at the specified path (URI)

    :param tiledb.Ctx ctx: The TileDB Context
    :param str uri: URI of the TileDB resource
    :raises TypeError: uri cannot be converted to a unicode string
    :raises: :py:exc:`tiledb.TileDBError`

    """
    cdef int rc = TILEDB_OK
    cdef tiledb_ctx_t* ctx_ptr = ctx.ptr
    cdef bytes buri = unicode_path(uri)
    cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
    with nogil:
        rc = tiledb_object_remove(ctx_ptr, uri_ptr)
    if rc != TILEDB_OK:
        check_error(ctx, rc)
    return


def move(Ctx ctx, old_uri, new_uri):
    """Moves a TileDB resource (group, array, key-value).

    :param tiledb.Ctx ctx: The TileDB Context
    :param str old_uri: path (URI) of the TileDB resource to move
    :param str new_uri: path (URI) of the destination
    :raises TypeError: uri cannot be converted to a unicode string
    :raises: :py:exc:`TileDBError`
    """
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


cdef int walk_callback(const char* path_ptr, tiledb_object_t obj, void* pyfunc):
    objtype = None
    if obj == TILEDB_GROUP:
        objtype = "group"
    if obj == TILEDB_ARRAY:
        objtype = "array"
    elif obj == TILEDB_KEY_VALUE:
        objtype = "kv"
    try:
        (<object> pyfunc)(path_ptr.decode('UTF-8'), objtype)
    except StopIteration:
        return 0
    return 1


def ls(Ctx ctx, path, func):
    """Lists TileDB resources and applies a callback that have a prefix of ``path`` (one level deep).

    :param tiledb.Ctx ctx: TileDB context
    :param str path: URI of TileDB group object
    :param function func: callback to execute on every listed TileDB resource,\
            URI resource path and object type label are passed as arguments to the callback
    :raises TypeError: cannot convert path to unicode string
    :raises: :py:exc:`tiledb.TileDBError`

    """
    cdef bytes bpath = unicode_path(path)
    check_error(ctx,
                tiledb_object_ls(ctx.ptr, bpath, walk_callback, <void*> func))
    return


def walk(Ctx ctx, path, func, order="preorder"):
    """Recursively visits TileDB resources and applies a callback that have a prefix of ``path``

    :param tiledb.Ctx ctx: The TileDB context
    :param str path: URI of TileDB group object
    :param function func: callback to execute on every listed TileDB resource,\
            URI resource path and object type label are passed as arguments to the callback
    :param str order: 'preorder' (default) or 'postorder' tree traversal
    :raises TypeError: cannot convert path to unicode string
    :raises ValueError: unknown order
    :raises: :py:exc:`tiledb.TileDBError`

    """
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

    cdef VFS vfs
    cdef unicode uri
    cdef tiledb_vfs_fh_t* ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        cdef Ctx ctx = self.vfs.ctx
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

    cdef Ctx ctx
    cdef tiledb_vfs_t* ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr != NULL:
            tiledb_vfs_free(&self.ptr)

    def __init__(self, Ctx ctx, config=None):
        cdef Config _config = Config()
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

    def open(self, uri, mode=None):
        """Opens a VFS file resource for reading / writing / appends at URI

        If the file did not exist upon opening, a new file is created.

        :param str uri: URI of VFS file resource
        :param mode str: 'r' for opening the file to read, 'w' to write, 'a' to append
        :rtype: FileHandle
        :return: VFS FileHandle
        :raises TypeError: cannot convert `uri` to unicode string
        :raises ValueError: invalid mode
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef tiledb_vfs_mode_t vfs_mode
        if mode == "r":
            vfs_mode = TILEDB_VFS_READ
        elif mode == "w":
            vfs_mode = TILEDB_VFS_WRITE
        elif mode == "a":
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

    def readinto(self, FileHandle fh, bytes buffer, offset, nbytes):
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
        cdef char* buffer_ptr = PyBytes_AS_STRING(buffer)
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_vfs_read(ctx_ptr, fh_ptr, _offset, <void*> buffer_ptr, _nbytes)
        if rc != TILEDB_OK:
            _raise_ctx_err(ctx_ptr, rc)
        return buffer

    def read(self, FileHandle fh, offset, nbytes):
        """Read nbytes from an opened VFS FileHandle at a given offset

        :param FileHandle fh: An opened VFS FileHandle in 'r' mode
        :param int offset: offset position in bytes to read from
        :param int nbytes: number of bytes to read
        :rtype: :py:func:`bytes`
        :return: read bytes
        :raises: :py:exc:`tiledb.TileDBError`

        """
        cdef Py_ssize_t _nbytes = nbytes
        cdef bytes buffer = PyBytes_FromStringAndSize(NULL, _nbytes)
        return self.readinto(fh, buffer, offset, nbytes)

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
        elif scheme == "hdfs":
            check_error(self.ctx,
                        tiledb_ctx_is_supported_fs(self.ctx.ptr, TILEDB_HDFS, &supports))
            return bool(supports)
        else:
            raise ValueError("unsupported vfs scheme '{0!s}://'".format(scheme))

    def config(self):
        """Returns the Config instance associated with the VFS
        """
        cdef tiledb_config_t* config_ptr = NULL
        check_error(self.ctx,
                    tiledb_vfs_get_config(self.ctx.ptr, self.ptr, &config_ptr))
        return Config.from_ptr(config_ptr)


class FileIO(object):

    def __init__(self, VFS vfs, uri, mode="r"):
        self.fh = vfs.open(uri, mode=mode)
        self.vfs = vfs
        self._offset = 0
        self._closed = False
        self._readonly = True
        if mode == "r":
            try:
                self._nbytes = vfs.file_size(uri)
            except:
                raise IOError("URI {0!r} is not a valid file")
            self._read_only = True
        elif mode == "w":
            self._readonly = False
            self._nbytes = 0
        else:
            raise ValueError("invalid mode {0!r}".format(mode))
        self._mode = mode
        return

    def __enter__(self):
        pass

    def __exit__(self):
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

    def seek(self, offset, whence=0):
        if not isinstance(offset, int):
            raise TypeError("offset must be an integer")
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

    def tell(self):
        return self._offset

    def writeable(self):
        return not self._readonly

    def read(self, size=-1):
        if not isinstance(size, int):
            raise TypeError("offset must be an integer")
        if self._mode == "w":
            raise IOError("cannot read from write-only FileIO handle")
        if self.closed:
            raise IOError("cannot read from closed FileIO handle")
        nbytes_remaining = self._nbytes - self._offset
        cdef Py_ssize_t nbytes
        if size < 0:
            nbytes = nbytes_remaining
        elif size > nbytes_remaining:
            nbytes = nbytes_remaining
        else:
            nbytes = size
        cdef bytes buff = PyBytes_FromStringAndSize(NULL, nbytes)
        self.vfs.readinto(self.fh, buff, self._offset, nbytes)
        self._offset += nbytes
        return buff

    def readall(self):
        if self._mode == "w":
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
        if self._mode == "w":
            raise IOError("cannot read from a write-only FileIO handle")
        if self.closed:
            raise IOError("cannot read from closed FileIO handle")
        nbytes = self._nbytes - self._offset
        if nbytes == 0:
            return
        self.vfs.readinto(self.fh, buff, self._offset, nbytes)
        self._offset += nbytes
        return

    def write(self, buff):
        if not self.writeable():
            raise IOError("cannot write to read-only FileIO handle")
        nbytes = len(buff)
        self.vfs.write(self.fh, buff)
        self._nbytes += nbytes
        self._offset += nbytes
        return nbytes
