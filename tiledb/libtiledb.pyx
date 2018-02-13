from cpython.version cimport PY_MAJOR_VERSION

from cpython.bytes cimport (PyBytes_GET_SIZE,
                            PyBytes_AS_STRING,
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


class TileDBError(Exception):
    pass


cdef _raise_tiledb_error(tiledb_error_t* err_ptr):
    cdef const char* err_msg_ptr = NULL
    ret = tiledb_error_message(err_ptr, &err_msg_ptr)
    if ret != TILEDB_OK:
        tiledb_error_free(err_ptr)
        if ret == TILEDB_OOM:
            return MemoryError()
        raise TileDBError("error retrieving error message")
    cdef unicode message_string = err_msg_ptr.decode('UTF-8', 'strict')
    tiledb_error_free(err_ptr)
    raise TileDBError(message_string)


cdef _raise_ctx_err(tiledb_ctx_t* ctx_ptr, int rc):
    if rc == TILEDB_OK:
        return
    if rc == TILEDB_OOM:
        raise MemoryError()
    cdef tiledb_error_t* err_ptr = NULL
    cdef int ret = tiledb_ctx_get_last_error(ctx_ptr, &err_ptr)
    if ret != TILEDB_OK:
        tiledb_error_free(err_ptr)
        if ret == TILEDB_OOM:
            raise MemoryError()
        raise TileDBError("error retrieving error object from ctx")
    _raise_tiledb_error(err_ptr)


cpdef check_error(Ctx ctx, int rc):
    _raise_ctx_err(ctx.ptr, rc)


def version():
    cdef:
        int major = 0
        int minor = 0
        int rev = 0
    tiledb_version(&major, &minor, &rev)
    return major, minor, rev


cdef unicode ustring(s):
    if type(s) is unicode:
        return <unicode> s
    elif PY_MAJOR_VERSION < 3 and isinstance(s, bytes):
        return (<bytes> s).decode('ascii')
    elif isinstance(s, unicode):
        return unicode(s)
    raise TypeError(
        "ustring() must be a string or a bytes-like object"
        ", not {0!r}".format(type(s)))


cdef bytes unicode_path(path):
    return ustring(path).encode('UTF-8')


cdef class Config(object):

    cdef tiledb_config_t* ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr is not NULL:
            tiledb_config_free(self.ptr)

    def __init__(self, params=None, path=None):
        cdef tiledb_config_t* config_ptr = NULL
        cdef tiledb_error_t* err_ptr = NULL
        cdef int rc = tiledb_config_create(&config_ptr, &err_ptr)
        if rc == TILEDB_OOM:
            raise MemoryError()
        elif rc == TILEDB_ERR:
            _raise_tiledb_error(err_ptr)
        assert (config_ptr != NULL)
        self.ptr = config_ptr
        if path is not None:
            self.load(path)
        if params is not None:
            self.update(params)

    @staticmethod
    cdef from_ptr(tiledb_config_t*ptr):
        cdef Config config = Config.__new__(Config)
        config.ptr = ptr
        return config

    @staticmethod
    def from_file(object filename):
        cdef bytes bfilename = unicode_path(filename)
        cdef Config config = Config.__new__(Config)
        cdef tiledb_config_t* config_ptr = NULL
        cdef tiledb_error_t* err_ptr = NULL
        cdef int rc = tiledb_config_create(&config_ptr, &err_ptr)
        if rc == TILEDB_OOM:
            raise MemoryError()
        if rc == TILEDB_ERR:
            _raise_tiledb_error(err_ptr)
        rc = tiledb_config_load_from_file(config_ptr, bfilename, &err_ptr)
        if rc == TILEDB_OOM:
            tiledb_config_free(config_ptr)
            raise MemoryError()
        if rc == TILEDB_ERR:
            tiledb_config_free(config_ptr)
            _raise_tiledb_error(err_ptr)
        config.ptr = config_ptr
        return config

    def __setitem__(self, object key, object value):
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
        return ConfigKeys(self)

    def __len__(self):
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

    def items(self, prefix=""):
        return ConfigItems(self, prefix=prefix)

    def keys(self, prefix=""):
        return ConfigKeys(self, prefix=prefix)

    def values(self, prefix=""):
        return ConfigValues(self, prefix=prefix)

    def dict(self, prefix=""):
        return dict(ConfigItems(self, prefix=prefix))

    def clear(self):
        for k in self.keys():
            del self[k]

    def get(self, key, *args):
        nargs = len(args)
        if nargs > 1:
            raise TypeError("get expected at most 2 arguments, got {}".format(nargs))
        try:
            return self[key]
        except KeyError:
            return args[0] if nargs == 1 else None

    def update(self, object odict):
        for (key, value) in odict.items():
            self[key] = value
        return

    def load(self, path):
        config = Config.from_file(path)
        self.update(config)

    def save(self, path):
        cdef bytes bpath = unicode_path(path)
        cdef tiledb_error_t* err_ptr = NULL
        cdef int rc = tiledb_config_save_to_file(self.ptr, bpath, &err_ptr)
        if rc == TILEDB_OOM:
            raise MemoryError()
        elif rc == TILEDB_ERR:
            _raise_tiledb_error(err_ptr)
        return


cdef class ConfigKeys(object):

    cdef ConfigItems config_items

    def __init__(self, Config config, unicode prefix=u""):
        self.config_items = ConfigItems(config, prefix=prefix)

    def __iter__(self):
        return self

    def __next__(self):
        (k, _) = self.config_items.__next__()
        return k


cdef class ConfigValues(object):

    cdef ConfigItems config_items

    def __init__(self, Config config, unicode prefix=u""):
        self.config_items = ConfigItems(config, prefix=prefix)

    def __iter__(self):
        return self

    def __next__(self):
        (_, v) = self.config_items.__next__()
        return v


cdef class ConfigItems(object):

    cdef Config config
    cdef tiledb_config_iter_t* ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr is not NULL:
            tiledb_config_iter_free(self.ptr)

    def __init__(self, Config config, unicode prefix=u""):
        cdef bytes bprefix = prefix.encode("UTF-8")
        cdef tiledb_config_iter_t* config_iter_ptr = NULL
        cdef tiledb_error_t* err_ptr = NULL
        cdef rc = tiledb_config_iter_create(
            config.ptr, &config_iter_ptr, bprefix, &err_ptr)
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

    cdef tiledb_ctx_t*ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr is not NULL:
            tiledb_ctx_free(self.ptr)

    def __init__(self, config=None, config_file=None):
        cdef Config _config = Config()
        if config_file is not None:
            _config = Config.from_file(config_file)
        if config is not None:
            if isinstance(config, Config):
                _config = config
            else:
                _config.update(config)
        cdef tiledb_ctx_t* ctx_ptr = NULL
        cdef int rc = tiledb_ctx_create(&ctx_ptr, _config.ptr)
        if rc == TILEDB_OOM:
            raise MemoryError()
        elif rc == TILEDB_ERR:
            # we assume that the ctx pointer is valid if not OOM
            # the ctx object will be free'd when it goes out of scope
            # after the exception is raised
            _raise_ctx_err(ctx_ptr, rc)
        self.ptr = ctx_ptr

    def config(self):
        cdef tiledb_config_t* config_ptr = NULL
        check_error(self,
                    tiledb_ctx_get_config(self.ptr, &config_ptr))
        return Config.from_ptr(config_ptr)


cdef tiledb_datatype_t _tiledb_dtype(np.dtype dtype) except? TILEDB_CHAR:
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


cdef int _numpy_type_num(tiledb_datatype_t tiledb_dtype):
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


cdef _numpy_scalar(tiledb_datatype_t typ, void*data, uint64_t nbytes):
    if typ == TILEDB_CHAR:
        # bytes type, ensure a full copy
        return PyBytes_FromStringAndSize(<char*> data, nbytes)
    # fixed size numeric type
    cdef int type_num = _numpy_type_num(typ)
    return PyArray_Scalar(data, np.PyArray_DescrFromType(type_num), None)


cdef tiledb_compressor_t _tiledb_compressor(object c) except TILEDB_NO_COMPRESSION:
    if c is None:
        return TILEDB_NO_COMPRESSION
    elif c == "gzip":
        return TILEDB_GZIP
    elif c == "zstd":
        return TILEDB_ZSTD
    elif c == "lz4":
        return TILEDB_LZ4
    elif c == "blosc-lz":
        return TILEDB_BLOSC
    elif c == "blosc-lz4":
        return TILEDB_BLOSC_LZ4
    elif c == "blosc-lz4hc":
        return TILEDB_BLOSC_LZ4HC
    elif c == "blosc-snappy":
        return TILEDB_BLOSC_SNAPPY
    elif c == "blosc-zstd":
        return TILEDB_BLOSC_ZSTD
    elif c == "rle":
        return TILEDB_RLE
    elif c == "bzip2":
        return TILEDB_BZIP2
    elif c == "double-delta":
        return TILEDB_DOUBLE_DELTA
    raise AttributeError("unknown compressor: {0!r}".format(c))


cdef unicode _tiledb_compressor_string(tiledb_compressor_t c):
    if c == TILEDB_NO_COMPRESSION:
        return u"none"
    elif c == TILEDB_GZIP:
        return u"gzip"
    elif c == TILEDB_ZSTD:
        return u"zstd"
    elif c == TILEDB_LZ4:
        return u"lz4"
    elif c == TILEDB_BLOSC:
        return u"blosc-lz"
    elif c == TILEDB_BLOSC_LZ4:
        return u"blosc-lz4"
    elif c == TILEDB_BLOSC_LZ4HC:
        return u"blosc-lz4hc"
    elif c == TILEDB_BLOSC_SNAPPY:
        return u"blosc-snappy"
    elif c == TILEDB_BLOSC_ZSTD:
        return u"blosc-zstd"
    elif c == TILEDB_RLE:
        return u"rle"
    elif c == TILEDB_BZIP2:
        return u"bzip2"
    elif c == TILEDB_DOUBLE_DELTA:
        return u"double-delta"


cdef class Attr(object):

    cdef Ctx ctx
    cdef tiledb_attribute_t* ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr is not NULL:
            tiledb_attribute_free(self.ctx.ptr, self.ptr)

    @staticmethod
    cdef from_ptr(Ctx ctx, const tiledb_attribute_t*ptr):
        cdef Attr attr = Attr.__new__(Attr)
        attr.ctx = ctx
        attr.ptr = <tiledb_attribute_t*> ptr
        return attr

    def __init__(self,
                 Ctx ctx,
                 name=u"",
                 dtype=np.float64,
                 compressor=None,
                 level=-1):
        cdef bytes bname = ustring(name).encode('UTF-8')
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
            ncells = ntypes
        else:
            tiledb_dtype = _tiledb_dtype(_dtype)
            ncells = 1
        cdef tiledb_attribute_t* attr_ptr = NULL
        check_error(ctx,
                    tiledb_attribute_create(ctx.ptr, &attr_ptr, bname, tiledb_dtype))
        check_error(ctx,
                    tiledb_attribute_set_cell_val_num(ctx.ptr, attr_ptr, ncells))
        cdef tiledb_compressor_t compr = TILEDB_NO_COMPRESSION
        if compressor is not None:
            compr = _tiledb_compressor(compressor)
            check_error(ctx,
                        tiledb_attribute_set_compressor(ctx.ptr, attr_ptr, compr, level))
        self.ctx = ctx
        self.ptr = attr_ptr

    def dump(self):
        check_error(self.ctx,
                    tiledb_attribute_dump(self.ctx.ptr, self.ptr, stdout))
        print('\n')
        return

    @property
    def dtype(self):
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
        return self._get_name()

    @property
    def isanon(self):
        cdef unicode name = self._get_name()
        return name == u"" or name.startswith(u"__attr")

    @property
    def compressor(self):
        cdef int level = -1
        cdef tiledb_compressor_t compr = TILEDB_NO_COMPRESSION
        check_error(self.ctx,
                    tiledb_attribute_get_compressor(self.ctx.ptr, self.ptr, &compr, &level))
        if compr == TILEDB_NO_COMPRESSION:
            return (None, -1)
        return (_tiledb_compressor_string(compr), level)

    cdef unsigned int _cell_val_num(Attr self) except? 0:
        cdef unsigned int ncells = 0
        check_error(self.ctx,
                    tiledb_attribute_get_cell_val_num(self.ctx.ptr, self.ptr, &ncells))
        return ncells

    @property
    def isvar(self):
        cdef unsigned int ncells = self._cell_val_num()
        return ncells == TILEDB_VAR_NUM

    @property
    def ncells(self):
        cdef unsigned int ncells = self._cell_val_num()
        assert (ncells != 0)
        return int(ncells)


cdef class Dim(object):

    cdef Ctx ctx
    cdef tiledb_dimension_t*ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr is not NULL:
            tiledb_dimension_free(self.ctx.ptr, self.ptr)

    @staticmethod
    cdef from_ptr(Ctx ctx, const tiledb_dimension_t*ptr):
        cdef Dim dim = Dim.__new__(Dim)
        dim.ctx = ctx
        dim.ptr = <tiledb_dimension_t*> ptr
        return dim

    def __init__(self, Ctx ctx, name=u"", domain=None, tile=0, dtype=np.uint64):
        cdef bytes bname = ustring(name).encode('UTF-8')
        if len(domain) != 2:
            raise AttributeError('invalid domain extent, must be a pair')
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
                raise AttributeError(
                    "invalid domain extent, domain cannot be safely cast to dtype {0!r}".format(dtype))
        domain_array = np.asarray(domain, dtype=dtype)
        domain_dtype = domain_array.dtype
        if (not np.issubdtype(domain_dtype, np.integer) and
                not np.issubdtype(domain_dtype, np.floating)):
            raise TypeError("invalid Dim dtype {0!r}".format(domain_dtype))
        cdef void* tile_size_ptr = NULL
        if tile > 0:
            tile_size_array = np.array(tile, dtype=domain_dtype)
            if tile_size_array.size != 1:
                raise ValueError("tile extent must be a scalar")
            tile_size_ptr = np.PyArray_DATA(tile_size_array)
        cdef tiledb_dimension_t* dim_ptr = NULL
        check_error(ctx,
                    tiledb_dimension_create(ctx.ptr,
                                            &dim_ptr,
                                            bname,
                                            _tiledb_dtype(domain_dtype),
                                            np.PyArray_DATA(domain_array),
                                            tile_size_ptr))
        assert(dim_ptr != NULL)
        self.ctx = ctx
        self.ptr = dim_ptr

    def __repr__(self):
        return 'Dim(name={0!r}, domain={1!s}, tile={2!s}, dtype={3!s})' \
            .format(self.name, self.domain, self.tile, self.dtype)

    cdef tiledb_datatype_t _get_type(Dim self):
        cdef tiledb_datatype_t typ
        check_error(self.ctx,
                    tiledb_dimension_get_type(self.ctx.ptr, self.ptr, &typ))
        return typ

    @property
    def dtype(self):
        return np.dtype(_numpy_type(self._get_type()))

    @property
    def name(self):
        cdef const char* name_ptr = NULL
        check_error(self.ctx,
                    tiledb_dimension_get_name(self.ctx.ptr, self.ptr, &name_ptr))
        return name_ptr.decode('UTF-8', 'strict')

    @property
    def shape(self):
        #TODO: this will not work for floating point domains / dimensions
        domain = self.domain
        return ((np.asscalar(domain[1]) -
                 np.asscalar(domain[0]) + 1),)

    @property
    def tile(self):
        cdef void* tile_ptr = NULL
        check_error(self.ctx,
                    tiledb_dimension_get_tile_extent(self.ctx.ptr, self.ptr, &tile_ptr))
        if tile_ptr == NULL:
            return None
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> 1
        cdef int type_num = _numpy_type_num(self._get_type())
        assert(type_num != np.NPY_NOTYPE)
        cdef np.ndarray tile_array = \
            np.PyArray_SimpleNewFromData(1, shape, type_num, tile_ptr)
        if tile_array[0] == 0:
            # undefined tiles should span the whole dimension domain
            return self.shape[0]
        return tile_array[0]

    @property
    def domain(self):
        cdef void* domain_ptr = NULL
        check_error(self.ctx,
                    tiledb_dimension_get_domain(self.ctx.ptr,
                                                self.ptr,
                                                &domain_ptr))
        assert(domain_ptr != NULL)
        cdef np.npy_intp shape[1]
        shape[0] = <np.npy_intp> 2
        cdef int typeid = _numpy_type_num(self._get_type())
        assert (typeid != np.NPY_NOTYPE)
        cdef np.ndarray domain_array = \
            np.PyArray_SimpleNewFromData(1, shape, typeid, domain_ptr)
        return (domain_array[0], domain_array[1])


cdef class Domain(object):

    cdef Ctx ctx
    cdef tiledb_domain_t* ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr is not NULL:
            tiledb_domain_free(self.ctx.ptr, self.ptr)

    @staticmethod
    cdef from_ptr(Ctx ctx, const tiledb_domain_t*ptr):
        cdef Domain dom = Domain.__new__(Domain)
        dom.ctx = ctx
        dom.ptr = <tiledb_domain_t*> ptr
        return dom

    def __init__(self, Ctx ctx, *dims):
        cdef unsigned int rank = len(dims)
        if rank == 0:
            raise TileDBError("Domain must have rank >= 1")
        cdef Dim dimension = dims[0]
        cdef tiledb_datatype_t domain_type = dimension._get_type()
        for i in range(1, rank):
            dimension = dims[i]
            if dimension._get_type() != domain_type:
                raise AttributeError("all dimensions must have the same dtype")
        cdef tiledb_domain_t* domain_ptr = NULL
        cdef int rc = tiledb_domain_create(ctx.ptr, &domain_ptr)
        if rc != TILEDB_OK:
            check_error(ctx, rc)
        assert(domain_ptr != NULL)
        for i in range(rank):
            dimension = dims[i]
            rc = tiledb_domain_add_dimension(
                ctx.ptr, domain_ptr, dimension.ptr)
            if rc != TILEDB_OK:
                tiledb_domain_free(ctx.ptr, domain_ptr)
                check_error(ctx, rc)
        self.ctx = ctx
        self.ptr = domain_ptr

    def __repr__(self):
        dims = ",\n       ".join(
            [repr(self.dim(i)) for i in range(self.rank)])
        return "Domain({0!s})".format(dims)

    def __len__(self):
        return self.rank

    def __iter__(self):
        return (self.dim(i) for i in range(self.rank))

    @property
    def rank(self):
        cdef unsigned int rank = 0
        check_error(self.ctx,
                    tiledb_domain_get_rank(self.ctx.ptr, self.ptr, &rank))
        return rank

    @property
    def ndim(self):
        return self.rank

    @property
    def dtype(self):
        cdef tiledb_datatype_t typ
        check_error(self.ctx,
                    tiledb_domain_get_type(self.ctx.ptr, self.ptr, &typ))
        return np.dtype(_numpy_type(typ))

    @property
    def shape(self):
        return tuple(self.dim(i).shape[0] for i in range(self.rank))

    def dim(self, int idx):
        cdef tiledb_dimension_t* dim_ptr = NULL
        check_error(self.ctx,
                    tiledb_domain_get_dimension_from_index(
                        self.ctx.ptr, self.ptr, idx, &dim_ptr))
        assert(dim_ptr != NULL)
        return Dim.from_ptr(self.ctx, dim_ptr)

    def dim(self, unicode name):
        cdef bytes uname = ustring(name).encode('UTF-8')
        cdef const char* name_ptr = uname
        cdef tiledb_dimension_t* dim_ptr = NULL
        check_error(self.ctx,
                    tiledb_domain_get_dimension_from_name(
                        self.ctx.ptr, self.ptr, name_ptr, &dim_ptr))
        assert(dim_ptr != NULL)
        return Dim.from_ptr(self.ctx, dim_ptr)

    def dump(self):
        check_error(self.ctx,
                    tiledb_domain_dump(self.ctx.ptr, self.ptr, stdout))
        print("\n")
        return


cdef tiledb_layout_t _tiledb_layout(order) except TILEDB_UNORDERED:
    if order == "row-major":
        return TILEDB_ROW_MAJOR
    elif order == "col-major":
        return TILEDB_COL_MAJOR
    elif order == "global":
        return TILEDB_GLOBAL_ORDER
    elif order == None or order == "unordered":
        return TILEDB_UNORDERED
    raise AttributeError("unknown tiledb layout: {0!r}".format(order))


cdef unicode _tiledb_layout_string(tiledb_layout_t order):
    if order == TILEDB_ROW_MAJOR:
        return u"row-major"
    elif order == TILEDB_COL_MAJOR:
        return u"col-major"
    elif order == TILEDB_GLOBAL_ORDER:
        return u"global"
    elif order == TILEDB_UNORDERED:
        return u"unordered"


cdef class KVIter(object):

    cdef Ctx ctx
    cdef tiledb_kv_iter_t*ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr is not NULL:
            tiledb_kv_iter_free(self.ctx.ptr, self.ptr)

    def __init__(self, Ctx ctx, uri):
        cdef bytes buri = unicode_path(uri)
        cdef tiledb_kv_iter_t* kv_iter_ptr = NULL
        check_error(ctx,
                    tiledb_kv_iter_create(ctx.ptr, &kv_iter_ptr, buri, NULL, 0))
        assert(kv_iter_ptr != NULL)
        self.ctx = ctx
        self.ptr = kv_iter_ptr

    def __iter__(self):
        return self

    def __next__(self):
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef int done = 0
        check_error(self.ctx,
                    tiledb_kv_iter_done(ctx_ptr, self.ptr, &done))
        if done > 0:
            raise StopIteration()
        cdef int rc
        cdef tiledb_kv_item_t* kv_item_ptr = NULL
        check_error(self.ctx,
                    tiledb_kv_iter_here(ctx_ptr, self.ptr, &kv_item_ptr))
        cdef tiledb_datatype_t dtype
        cdef const char* key_ptr = NULL
        cdef uint64_t key_size = 0
        check_error(self.ctx,
                    tiledb_kv_item_get_key(ctx_ptr, kv_item_ptr,
                                           <const void**> (&key_ptr), &dtype, &key_size))
        cdef bytes bkey = PyBytes_FromStringAndSize(key_ptr, key_size)
        cdef const char* val_ptr = NULL
        cdef uint64_t val_size = 0
        cdef bytes battr = b"value"
        check_error(self.ctx,
                    tiledb_kv_item_get_value(ctx_ptr, kv_item_ptr, battr,
                                             <const void**> (&val_ptr), &dtype, &val_size))
        cdef bytes bval = PyBytes_FromStringAndSize(val_ptr, val_size)
        check_error(self.ctx,
                    tiledb_kv_iter_next(ctx_ptr, self.ptr))
        return (bkey.decode('UTF-8'), bval.decode('UTF-8'))


cdef class Assoc(object):

    cdef Ctx ctx
    cdef unicode uri
    cdef tiledb_kv_schema_t* ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr is not NULL:
            tiledb_kv_schema_free(self.ctx.ptr, self.ptr)

    @staticmethod
    cdef from_ptr(Ctx ctx, unicode uri, const tiledb_kv_schema_t* ptr):
        cdef Assoc arr = Assoc.__new__(Assoc)
        arr.ctx = ctx
        arr.uri = uri
        arr.ptr = <tiledb_kv_schema_t*> ptr
        return arr

    @staticmethod
    def load(Ctx ctx, uri):
        cdef bytes buri = unicode_path(uri)
        cdef const char* uri_ptr = buri
        cdef tiledb_kv_schema_t*schema_ptr = NULL
        cdef int rc
        with nogil:
            rc = tiledb_kv_schema_load(ctx.ptr, &schema_ptr, uri_ptr)
        if rc != TILEDB_OK:
            check_error(ctx, rc)
        return Assoc.from_ptr(ctx, uri, schema_ptr)

    def __init__(self, Ctx ctx, uri, attrs=()):
        cdef bytes buri = unicode_path(uri)
        cdef tiledb_kv_schema_t* schema_ptr = NULL
        check_error(ctx,
                    tiledb_kv_schema_create(ctx.ptr, &schema_ptr))
        cdef int rc = TILEDB_OK
        cdef tiledb_attribute_t* attr_ptr = NULL
        for attr in attrs:
            if not isinstance(attr, Attr):
                tiledb_kv_schema_free(ctx.ptr, schema_ptr)
                raise TypeError("invalid attribute type {0!r}".format(type(attr)))
            attr_ptr = (<Attr> attr).ptr
            rc = tiledb_kv_schema_add_attribute(ctx.ptr, schema_ptr, attr_ptr)
            if rc != TILEDB_OK:
                tiledb_kv_schema_free(ctx.ptr, schema_ptr)
                check_error(ctx, rc)
        rc = tiledb_kv_schema_check(ctx.ptr, schema_ptr)
        if rc != TILEDB_OK:
            tiledb_kv_schema_free(ctx.ptr, schema_ptr)
            check_error(ctx, rc)
        rc = tiledb_kv_create(ctx.ptr, buri, schema_ptr)
        if rc != TILEDB_OK:
            tiledb_kv_schema_free(ctx.ptr, schema_ptr)
            check_error(ctx, rc)
        self.ctx = ctx
        self.uri = uri
        self.ptr = schema_ptr

    @property
    def nattr(self):
        cdef unsigned int nattr = 0
        check_error(self.ctx,
                    tiledb_kv_schema_get_attribute_num(self.ctx.ptr, self.ptr, &nattr))
        return nattr

    cdef Attr _attr_name(self, unicode name):
        cdef bytes bname = ustring(name).encode('UTF-8')
        cdef tiledb_attribute_t* attr_ptr = NULL
        check_error(self.ctx,
                    tiledb_kv_schema_get_attribute_from_name(
                        self.ctx.ptr, self.ptr, bname, &attr_ptr))
        return Attr.from_ptr(self.ctx, attr_ptr)

    cdef Attr _attr_idx(self, int idx):
        cdef tiledb_attribute_t* attr_ptr = NULL
        check_error(self.ctx,
                    tiledb_kv_schema_get_attribute_from_index(
                        self.ctx.ptr, self.ptr, idx, &attr_ptr))
        return Attr.from_ptr(self.ctx, attr_ptr)

    def attr(self, object key not None):
        if isinstance(key, str):
            return self._attr_name(key)
        elif isinstance(key, _inttypes):
            return self._attr_idx(int(key))
        raise AttributeError("attr() key must be a string name, "
                             "or an integer index, not {0!r}".format(type(key)))

    def consolidate(self):
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef bytes buri = self.uri.encode('UTF-8')
        cdef const char* uri_ptr = PyBytes_AS_STRING(buri)
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_kv_consolidate(ctx_ptr, uri_ptr)
        if rc != TILEDB_OK:
            check_error(self.ctx, rc)
        return

    def dict(self):
        return dict(self)

    def dump(self):
        check_error(self.ctx,
                    tiledb_kv_schema_dump(self.ctx.ptr, self.ptr, stdout))
        print("\n")
        return

    def __iter__(self):
        return KVIter(self.ctx, self.uri)

    def __setitem__(self, object key, object value):
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef bytes buri = unicode_path(self.uri)

        cdef Attr attr = self.attr(0)
        cdef bytes battr = attr.name.encode('UTF-8')
        cdef const char* battr_ptr = PyBytes_AS_STRING(battr)

        # Create KV item object
        cdef int rc
        cdef tiledb_kv_item_t* kv_item_ptr = NULL
        rc = tiledb_kv_item_create(ctx_ptr, &kv_item_ptr)
        if rc != TILEDB_OK:
            check_error(self.ctx, rc)

        # add key
        # TODO: specialized for strings
        cdef bytes bkey = key.encode('UTF-8')
        cdef const void* bkey_ptr = PyBytes_AS_STRING(bkey)
        cdef uint64_t bkey_size = PyBytes_GET_SIZE(bkey)

        rc = tiledb_kv_item_set_key(ctx_ptr, kv_item_ptr,
                                    bkey_ptr, TILEDB_CHAR, bkey_size)
        if rc != TILEDB_OK:
            tiledb_kv_item_free(ctx_ptr, kv_item_ptr)
            check_error(self.ctx, rc)

        # add value
        # TODO: specialized for strings
        cdef bytes bvalue = value.encode('UTF-8')
        cdef const void* bvalue_ptr = PyBytes_AS_STRING(bvalue)
        cdef uint64_t bvalue_size = PyBytes_GET_SIZE(bvalue)

        rc = tiledb_kv_item_set_value(ctx_ptr, kv_item_ptr, battr_ptr,
                                      bvalue_ptr, TILEDB_CHAR, bvalue_size)
        if rc != TILEDB_OK:
            tiledb_kv_item_free(ctx_ptr, kv_item_ptr)
            check_error(self.ctx, rc)

        # save items
        cdef tiledb_kv_t* kv_ptr = NULL
        rc = tiledb_kv_open(ctx_ptr, &kv_ptr, buri, NULL, 1)
        if rc != TILEDB_OK:
            tiledb_kv_item_free(ctx_ptr, kv_item_ptr)
            check_error(self.ctx, rc)

        rc = tiledb_kv_add_item(ctx_ptr, kv_ptr, kv_item_ptr)
        if rc != TILEDB_OK:
            tiledb_kv_item_free(ctx_ptr, kv_item_ptr)
            check_error(self.ctx, rc)

        rc = tiledb_kv_flush(ctx_ptr, kv_ptr)
        if rc != TILEDB_OK:
            tiledb_kv_item_free(ctx_ptr, kv_item_ptr)
            check_error(self.ctx, rc)

        rc = tiledb_kv_close(ctx_ptr, kv_ptr)
        tiledb_kv_item_free(ctx_ptr, kv_item_ptr)
        if rc != TILEDB_OK:
            check_error(self.ctx, rc)
        return

    def __getitem__(self, object key):
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr
        cdef bytes buri = unicode_path(self.uri)

        cdef Attr attr = self.attr(0)
        cdef bytes battr = attr.name.encode('UTF-8')
        cdef const char* battr_ptr = PyBytes_AS_STRING(battr)

        # Create KV item object
        cdef tiledb_kv_t* kv_ptr = NULL
        cdef int rc = tiledb_kv_open(ctx_ptr, &kv_ptr, buri, &battr_ptr, 1)
        if rc != TILEDB_OK:
            check_error(self.ctx, rc)

        # add key
        # TODO: specialized for strings
        cdef bytes bkey = key.encode('UTF-8')
        cdef const void* bkey_ptr = PyBytes_AS_STRING(bkey)
        cdef uint64_t bkey_size = PyBytes_GET_SIZE(bkey)

        cdef tiledb_kv_item_t* kv_item_ptr = NULL
        rc = tiledb_kv_get_item(ctx_ptr, kv_ptr, &kv_item_ptr,
                                bkey_ptr, TILEDB_CHAR, bkey_size)
        if rc != TILEDB_OK:
            tiledb_kv_close(ctx_ptr, kv_ptr)
            check_error(self.ctx, rc)

        if kv_item_ptr == NULL:
            tiledb_kv_close(ctx_ptr, kv_ptr)
            raise KeyError(key)

        cdef const void* value_ptr = NULL
        cdef tiledb_datatype_t value_type = TILEDB_CHAR
        cdef uint64_t value_size = 0
        rc = tiledb_kv_item_get_value(ctx_ptr, kv_item_ptr, battr_ptr,
                                      &value_ptr, &value_type, &value_size)
        if rc != TILEDB_OK:
            tiledb_kv_item_free(ctx_ptr, kv_item_ptr)
            tiledb_kv_close(ctx_ptr, kv_ptr)
            check_error(self.ctx, rc)

        assert(value_ptr != NULL)
        cdef bytes val
        try:
            val = PyBytes_FromStringAndSize(<char*> value_ptr, value_size)
        finally:
            tiledb_kv_item_free(ctx_ptr, kv_item_ptr)
            tiledb_kv_close(ctx_ptr, kv_ptr)
        return val.decode('UTF-8')

    def __contains__(self, unicode key):
        try:
            self[key]
            return True
        except KeyError:
            return False
        except:
            raise

    def update(self, *args, **kwargs):
        cdef dict update = {}
        if len(args) == 0:
            update.update(**kwargs)
        elif len(args) == 1:
            update.update(args[0])
        else:
            raise AttributeError(
                "must provide a dictionary, iterable of key/value pairs "
                "or explict keyword arguments")
        cdef np.ndarray key_array = np.array(update.keys())
        cdef np.ndarray value_array = np.array(update.values())
        cdef uint64_t nkeys = len(key_array)
        for i in range(nkeys):
            pass
        raise NotImplementedError()


def index_as_tuple(idx):
    if isinstance(idx, tuple):
        return idx
    return (idx,)


def replace_ellipsis(Domain dom, tuple idx):
    rank = dom.rank
    # count number of ellipsis
    n_ellip = sum(1 for i in idx if i is Ellipsis)
    if n_ellip > 1:
        raise IndexError("an index can only have a single ellipsis ('...')")
    elif n_ellip == 1:
        n = len(idx)
        if (n - 1) >= rank:
            # does nothing, strip it out
            idx = tuple(i for i in idx if i is not Ellipsis)
        else:
            # locate where the ellipse is, count the number of items to left and right
            # fill in whole dim slices up to th rank of the array
            left = idx.index(Ellipsis)
            right = n - (left + 1)
            new_idx = idx[:left] + ((slice(None),) * (rank - (n - 1)))
            if right:
                new_idx += idx[-right:]
            idx = new_idx
    idx_rank = len(idx)
    if idx_rank < rank:
        idx += (slice(None),) * (rank - idx_rank)
    if len(idx) > rank:
        raise IndexError("too many indices for array")
    return idx


def replace_scalars_slice(Domain dom, tuple idx):
    new_idx, drop_axes = [], []
    for i in range(dom.rank):
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
    rank = dom.rank
    if len(idx) != rank:
        raise IndexError("number of indices does not match domain raank: "
                         "({:!r} expected {:!r]".format(len(idx), rank))
    # populate a subarray array / buffer to pass to tiledb
    subarray = np.zeros(shape=(rank, 2), dtype=dom.dtype)
    for r in range(rank):
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

    cdef Ctx ctx
    cdef unicode uri
    cdef tiledb_array_schema_t* ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr is not NULL:
            tiledb_array_schema_free(self.ctx.ptr, self.ptr)

    def __init__(self, Ctx ctx, unicode uri,
                 domain=None,
                 attrs=(),
                 cell_order='row-major',
                 tile_order='row-major',
                 capacity=0,
                 sparse=False):
        cdef bytes buri = ustring(uri).encode('UTF-8')
        cdef tiledb_array_type_t array_type = TILEDB_SPARSE if sparse else TILEDB_DENSE
        cdef tiledb_array_schema_t* schema_ptr = NULL
        check_error(ctx,
                    tiledb_array_schema_create(ctx.ptr, &schema_ptr, array_type))
        cdef tiledb_layout_t cell_layout = _tiledb_layout(cell_order)
        cdef tiledb_layout_t tile_layout = _tiledb_layout(tile_order)
        cdef int rc = TILEDB_OK
        rc = tiledb_array_schema_set_cell_order(ctx.ptr, schema_ptr, cell_layout)
        if rc != TILEDB_OK:
            tiledb_array_schema_free(ctx.ptr, schema_ptr)
            check_error(ctx, rc)
        rc = tiledb_array_schema_set_tile_order(ctx.ptr, schema_ptr, tile_layout)
        if rc != TILEDB_OK:
            tiledb_array_schema_free(ctx.ptr, schema_ptr)
            check_error(ctx, rc)
        cdef uint64_t c_capacity = 0
        if sparse and capacity > 0:
            c_capacity = <uint64_t> capacity
            rc = tiledb_array_schema_set_capacity(ctx.ptr, schema_ptr, c_capacity)
            if rc != TILEDB_OK:
                tiledb_array_schema_free(ctx.ptr, schema_ptr)
                check_error(ctx, rc)
        cdef tiledb_domain_t* domain_ptr = (<Domain> domain).ptr
        rc = tiledb_array_schema_set_domain(ctx.ptr, schema_ptr, domain_ptr)
        if rc != TILEDB_OK:
            tiledb_array_schema_free(ctx.ptr, schema_ptr)
            check_error(ctx, rc)
        cdef tiledb_attribute_t* attr_ptr = NULL
        for attr in attrs:
            attr_ptr = (<Attr> attr).ptr
            rc = tiledb_array_schema_add_attribute(ctx.ptr, schema_ptr, attr_ptr)
            if rc != TILEDB_OK:
                tiledb_array_schema_free(ctx.ptr, schema_ptr)
                check_error(ctx, rc)
        rc = tiledb_array_schema_check(ctx.ptr, schema_ptr)
        if rc != TILEDB_OK:
            tiledb_array_schema_free(ctx.ptr, schema_ptr)
            check_error(ctx, rc)
        rc = tiledb_array_create(ctx.ptr, buri, schema_ptr)
        if rc != TILEDB_OK:
            check_error(ctx, rc)
        self.ctx = ctx
        self.uri = uri
        self.ptr = schema_ptr

    @property
    def name(self):
        return self.uri

    @property
    def sparse(self):
        cdef tiledb_array_type_t typ = TILEDB_DENSE
        check_error(self.ctx,
                    tiledb_array_schema_get_array_type(self.ctx.ptr, self.ptr, &typ))
        return typ == TILEDB_SPARSE

    @property
    def capacity(self):
        cdef uint64_t cap = 0
        check_error(self.ctx,
                    tiledb_array_schema_get_capacity(self.ctx.ptr, self.ptr, &cap))
        return int(cap)

    @property
    def cell_order(self):
        cdef tiledb_layout_t order = TILEDB_UNORDERED
        check_error(self.ctx,
                    tiledb_array_schema_get_cell_order(self.ctx.ptr, self.ptr, &order))
        return _tiledb_layout_string(order)

    @property
    def tile_order(self):
        cdef tiledb_layout_t order = TILEDB_UNORDERED
        check_error(self.ctx,
                    tiledb_array_schema_get_tile_order(self.ctx.ptr, self.ptr, &order))
        return _tiledb_layout_string(order)

    @property
    def coord_compressor(self):
        cdef tiledb_compressor_t comp = TILEDB_NO_COMPRESSION
        cdef int level = -1
        check_error(self.ctx,
                    tiledb_array_schema_get_coords_compressor(
                        self.ctx.ptr, self.ptr, &comp, &level))
        return (_tiledb_compressor_string(comp), int(level))

    @property
    def rank(self):
        return self.domain.rank

    @property
    def domain(self):
        cdef tiledb_domain_t* dom = NULL
        check_error(self.ctx,
                    tiledb_array_schema_get_domain(self.ctx.ptr, self.ptr, &dom))
        return Domain.from_ptr(self.ctx, dom)

    @property
    def nattr(self):
        cdef unsigned int nattr = 0
        check_error(self.ctx,
                    tiledb_array_schema_get_attribute_num(self.ctx.ptr, self.ptr, &nattr))
        return int(nattr)

    cdef _attr_name(self, unicode name):
        cdef bytes bname = name.encode('UTF-8')
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
        if isinstance(key, str):
            return self._attr_name(key)
        elif isinstance(key, _inttypes):
            return self._attr_idx(int(key))
        raise AttributeError("attr indices must be a string name, "
                             "or an integer index, not {0!r}".format(type(key)))
    @property
    def ndim(self):
        return self.domain.ndim

    @property
    def shape(self):
        return self.domain.shape

    def dump(self):
        check_error(self.ctx,
                    tiledb_array_schema_dump(self.ctx.ptr, self.ptr, stdout))
        print("\n")
        return

    def consolidate(self):
        return array_consolidate(self.ctx, self.uri)

    def nonempty_domain(self):
        cdef Domain dom = self.domain
        cdef bytes buri = self.uri.encode('UTF-8')
        cdef np.ndarray extents = np.zeros(shape=(dom.rank, 2), dtype=dom.dtype)
        cdef void* extents_ptr = np.PyArray_DATA(extents)
        cdef int empty = 0
        check_error(self.ctx,
                    tiledb_array_get_non_empty_domain(self.ctx.ptr, buri, extents_ptr, &empty))
        if empty > 0:
            return None
        return tuple((extents[i, 0].item(), extents[i, 1].item())
                     for i in range(dom.rank))


cdef class DenseArray(ArraySchema):

    @staticmethod
    cdef from_ptr(Ctx ctx, unicode uri, const tiledb_array_schema_t* schema_ptr):
        cdef DenseArray arr = DenseArray.__new__(DenseArray)
        arr.ctx = ctx
        arr.uri = uri
        arr.ptr = <tiledb_array_schema_t*> schema_ptr
        return arr

    @staticmethod
    def load(Ctx ctx, unicode uri):
        cdef tiledb_ctx_t* ctx_ptr = ctx.ptr

        cdef bytes buri = ustring(uri).encode('UTF-8')
        cdef const char* buri_ptr = PyBytes_AS_STRING(buri)

        cdef tiledb_array_schema_t* schema_ptr = NULL
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_array_schema_load(ctx_ptr, &schema_ptr, buri_ptr)
        if rc != TILEDB_OK:
            check_error(ctx, rc)
        return DenseArray.from_ptr(ctx, uri, schema_ptr)

    @staticmethod
    def from_numpy(Ctx ctx, unicode path, np.ndarray array, **kw):
        dims = []
        for d in range(array.ndim):
            extent = array.shape[d]
            domain = (0, extent - 1)
            dims.append(Dim(ctx, "", domain, extent, np.uint64))
        dom = Domain(ctx, *dims)
        att = Attr(ctx, "", dtype=array.dtype)
        arr = DenseArray(ctx, path, domain=dom, attrs=[att], **kw)
        arr.write_direct(array)
        return arr

    def __init__(self, *args, **kw):
        kw['sparse'] = False
        super().__init__(*args, **kw)

    def __len__(self):
        return self.domain.shape[0]

    def __getitem__(self, object key):
        key = index_as_tuple(key)
        idx = replace_ellipsis(self.domain, key)
        idx, drop_axes = replace_scalars_slice(self.domain, idx)
        subarray = index_domain_subarray(self.domain, idx)
        attr_names = [self.attr(i).name for i in range(self.nattr)]
        out = self._read_dense_subarray(subarray, attr_names)
        if any(s.step for s in idx):
            steps = tuple(slice(None, None, s.step) for s in idx)
            for (k, v) in out.items():
                out[k] = v.__getitem__(steps)
        if drop_axes:
            for (k, v) in out.items():
                out[k] = v.squeeze(axis=drop_axes)
        # attribute is anonymous, just return the result
        if self.nattr == 1 and self.attr(0).isanon:
            return out[self.attr(0).name]
        return out

    cdef _read_dense_subarray(self, np.ndarray subarray, list attr_names):
        cdef Ctx ctx = self.ctx
        cdef tiledb_ctx_t* ctx_ptr = ctx.ptr

        # array name TODO: fix name
        cdef bytes array_name = self.uri.encode('UTF-8')
        cdef const char* c_aname = array_name

        cdef tuple shape = \
            tuple(int(subarray[r, 1]) - int(subarray[r, 0]) + 1
                  for r in range(self.rank))

        out = OrderedDict()
        for n in attr_names:
            out[n] = np.empty(shape=shape, dtype=self.attr(n).dtype)

        # attr names
        battr_names = list(bytes(k, 'UTF-8') for k in out.keys())
        cdef size_t nattr = len(out.keys())
        cdef const char** c_attr_names = <const char**> calloc(nattr, sizeof(uintptr_t))
        for i in range(nattr):
            c_attr_names[i] = PyBytes_AS_STRING(battr_names[i])

        attr_values = list(out.values())
        cdef void** buffers = <void**> calloc(nattr, sizeof(uintptr_t))
        cdef uint64_t* buffer_sizes = <uint64_t*> calloc(nattr, sizeof(uint64_t))
        cdef np.ndarray array_value
        for (i, array_value) in enumerate(out.values()):
            buffers[i] = np.PyArray_DATA(array_value)
            buffer_sizes[i] = <uint64_t> array_value.nbytes

        cdef int rc = TILEDB_OK
        cdef tiledb_query_t* query = NULL
        rc = tiledb_query_create(ctx_ptr, &query, c_aname, TILEDB_READ)
        if rc != TILEDB_OK:
            check_error(ctx, rc)

        cdef void* subarray_ptr = np.PyArray_DATA(subarray)
        rc = tiledb_query_set_subarray(ctx_ptr, query, subarray_ptr)
        if rc != TILEDB_OK:
            tiledb_query_free(ctx_ptr, query)
            check_error(ctx, rc)

        rc = tiledb_query_set_buffers(ctx_ptr, query, c_attr_names, nattr,
                                      buffers, buffer_sizes)
        if rc != TILEDB_OK:
            tiledb_query_free(ctx_ptr, query)
            check_error(ctx, rc)

        rc = tiledb_query_set_layout(ctx_ptr, query, TILEDB_ROW_MAJOR)
        if rc != TILEDB_OK:
            tiledb_query_free(ctx_ptr, query)
            check_error(ctx, rc)

        with nogil:
            rc = tiledb_query_submit(ctx_ptr, query)

        tiledb_query_free(ctx_ptr, query)
        if rc != TILEDB_OK:
            check_error(ctx, rc)
        return out

    def __setitem__(self, object key, object val):
        cdef Domain domain = self.domain
        cdef tuple idx = replace_ellipsis(domain, index_as_tuple(key))
        cdef np.ndarray subarray = index_domain_subarray(domain, idx)
        cdef dict dense_values = dict()
        cdef Attr attr
        if isinstance(val, dict):
            for (k, v) in val.items():
                attr = self.attr(k)
                dense_values[attr.name] = \
                    np.ascontiguousarray(v, dtype=attr.dtype)
        elif np.isscalar(val):
            for i in range(self.nattr):
                attr = self.attr(i)
                subarray_shape = tuple(int(subarray[r, 1] - subarray[r, 0]) + 1
                                       for r in range(subarray.shape[0]))
                dense_values[attr.name] = \
                    np.full(subarray_shape, val, dtype=attr.dtype)
        elif self.nattr == 1:
            attr = self.attr(0)
            dense_values[attr.name] = \
                np.ascontiguousarray(val, dtype=attr.dtype)
        else:
            raise ValueError("ambiguous attribute assignment, "
                             "more than one array attribue")
        self._write_dense_subarray(subarray, dense_values)
        return

    cdef _write_dense_subarray(self, np.ndarray subarray, dict values):
        cdef Ctx ctx = self.ctx
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr

        # array name #TODO: name
        cdef bytes barray_name = self.uri.encode('UTF-8')
        cdef const char* c_array_name = barray_name

        # attribute names / values
        battr_names = list(bytes(k, 'UTF-8') for k in values.keys())
        cdef size_t nattr = len(values.keys())
        cdef const char** c_attr_names = <const char**> calloc(nattr, sizeof(uintptr_t))
        for i in range(nattr):
            c_attr_names[i] = PyBytes_AS_STRING(battr_names[i])

        attr_values = list(values.values())
        cdef void** buffers = <void**> calloc(nattr, sizeof(uintptr_t))
        cdef uint64_t* buffer_sizes = <uint64_t*> calloc(nattr, sizeof(uint64_t))
        cdef np.ndarray array_value
        for (i, array_value) in enumerate(values.values()):
            buffers[i] = np.PyArray_DATA(array_value)
            buffer_sizes[i] = <uint64_t> array_value.nbytes

        # subarray
        cdef void* subarray_ptr = np.PyArray_DATA(subarray)

        # TODO: check if all the layouts are the same
        cdef tiledb_layout_t layout = TILEDB_ROW_MAJOR

        cdef tiledb_query_t* query_ptr = NULL
        check_error(ctx,
                    tiledb_query_create(ctx_ptr, &query_ptr, c_array_name, TILEDB_WRITE))
        check_error(ctx,
                    tiledb_query_set_layout(ctx_ptr, query_ptr, layout))
        check_error(ctx,
                    tiledb_query_set_subarray(ctx_ptr, query_ptr, subarray_ptr))
        check_error(ctx,
                    tiledb_query_set_buffers(ctx_ptr, query_ptr, c_attr_names, nattr,
                                             buffers, buffer_sizes))
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_query_submit(ctx_ptr, query_ptr)
        tiledb_query_free(ctx_ptr, query_ptr)
        if rc != TILEDB_OK:
            check_error(ctx, rc)
        return

    def __array__(self, dtype=None, **kw):
        array = self.read_direct("")
        if dtype and array.dtype != dtype:
            return array.astype(dtype)
        return array

    def write_direct(self, np.ndarray array not None, unicode attr=u""):
        cdef Ctx ctx = self.ctx
        cdef tiledb_ctx_t* ctx_ptr = ctx.ptr

        # array name #TODO: name
        cdef bytes barray_name = self.uri.encode('UTF-8')
        cdef const char* c_array_name = barray_name

        # attr name
        cdef bytes battr_name = attr.encode('UTF-8')
        cdef const char* c_attr_name = battr_name

        cdef void* buff_ptr = np.PyArray_DATA(array)
        cdef uint64_t buff_size = array.nbytes

        cdef tiledb_layout_t layout = TILEDB_ROW_MAJOR
        if np.isfortran(array):
            layout = TILEDB_COL_MAJOR

        cdef tiledb_query_t* query_ptr = NULL
        check_error(ctx,
                    tiledb_query_create(ctx_ptr, &query_ptr, c_array_name, TILEDB_WRITE))
        check_error(ctx,
                    tiledb_query_set_layout(ctx_ptr, query_ptr, layout))
        check_error(ctx,
                    tiledb_query_set_buffers(ctx_ptr, query_ptr, &c_attr_name, 1, &buff_ptr, &buff_size))

        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_query_submit(ctx_ptr, query_ptr)
        tiledb_query_free(ctx_ptr, query_ptr)
        if rc != TILEDB_OK:
            check_error(ctx, rc)
        return

    def read_direct(self, unicode attribute_name=u""):
        cdef Ctx ctx = self.ctx
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr

        # array name
        cdef bytes barray_name = self.uri.encode('UTF-8')

        # attr name
        cdef bytes battribute_name = attribute_name.encode('UTF-8')
        cdef const char* c_attribute_name = battribute_name

        cdef Attr attr = self.attr(attribute_name)

        order = 'C'
        cell_order = self.cell_order
        cdef tiledb_layout_t cell_layout = TILEDB_ROW_MAJOR
        if cell_order == 'row-major':
            order = 'C'
            cell_layout = TILEDB_ROW_MAJOR
        elif cell_order == 'col-major':
            order = 'F'
            cell_layout = TILEDB_COL_MAJOR

        out = np.empty(self.domain.shape, dtype=attr.dtype, order=order)

        cdef void* buff_ptr = np.PyArray_DATA(out)
        cdef uint64_t buff_size = out.nbytes

        cdef tiledb_query_t* query_ptr = NULL
        check_error(ctx,
                    tiledb_query_create(ctx_ptr, &query_ptr, barray_name, TILEDB_READ))
        check_error(ctx,
                    tiledb_query_set_layout(ctx_ptr, query_ptr, cell_layout))
        check_error(ctx,
                    tiledb_query_set_buffers(ctx_ptr, query_ptr,
                                             &c_attribute_name, 1, &buff_ptr, &buff_size))
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_query_submit(ctx_ptr, query_ptr)
        tiledb_query_free(ctx_ptr, query_ptr)
        if rc != TILEDB_OK:
            check_error(ctx, rc)
        return out


# point query index a tiledb array (zips) columnar index vectors
def index_domain_coords(Domain dom, tuple idx):
    rank = len(idx)
    if rank != dom.rank:
        raise IndexError("sparse index rank must match "
                         "domain rank: {0!r} != {1!r}".format(rank, dom.rank))
    idx = tuple(np.asarray(idx[i], dtype=dom.dim(i).dtype)
                for i in range(rank))
    # check that all sparse coordinates are the same size and dtype
    len0, dtype0 = len(idx[0]), idx[0].dtype
    for i in range(2, rank):
        if len(idx[i]) != len0:
            raise IndexError("sparse index dimension length mismatch")
        if idx[i].dtype != dtype0:
            raise IndexError("sparse index dimension dtype mismatch")
    # zip coordinates
    return np.column_stack(idx)


cdef class Query(object):

    cdef Ctx ctx
    cdef tiledb_query_t*ptr

    cdef const char** buffer_name_ptrs
    cdef void** buffer_ptrs
    cdef uint64_t* buffer_size_ptr

    def __cinit__(self):
        self.ptr = NULL
        self.buffer_name_ptrs = NULL
        self.buffer_ptrs = NULL
        self.buffer_size_ptr = NULL

    def __dealloc__(self):
        if self.buffer_name_ptrs != NULL:
            free(self.buffer_name_ptrs)
        if self.buffer_ptrs != NULL:
            free(self.buffer_ptrs)
        if self.buffer_size_ptr != NULL:
            free(self.buffer_size_ptr)
        if self.ptr != NULL:
            tiledb_query_free(self.ctx.ptr, self.ptr)

    def __init__(self, Ctx ctx, unicode uri, ):
        self.ctx = ctx


cdef class SparseArray(ArraySchema):

    @staticmethod
    cdef from_ptr(Ctx ctx, unicode uri, const tiledb_array_schema_t* schema_ptr):
        cdef SparseArray arr = SparseArray.__new__(SparseArray)
        arr.ctx = ctx
        arr.uri = uri
        arr.ptr = <tiledb_array_schema_t*> schema_ptr
        return arr

    @staticmethod
    def load(Ctx ctx, unicode uri):
        cdef tiledb_ctx_t* ctx_ptr = ctx.ptr

        cdef bytes buri = ustring(uri).encode('UTF-8')
        cdef const char* buri_ptr = PyBytes_AS_STRING(buri)

        cdef tiledb_array_schema_t* schema_ptr = NULL
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_array_schema_load(ctx_ptr, &schema_ptr, buri_ptr)
        if rc != TILEDB_OK:
            check_error(ctx, rc)
        return SparseArray.from_ptr(ctx, uri, schema_ptr)

    def __init__(self, *args, **kw):
        kw['sparse'] = True
        super().__init__(*args, **kw)

    def __len__(self):
        raise TypeError("SparseArray length is ambiguous; use shape[0]")

    def __setitem__(self, object key, object val):
        idx = index_as_tuple(key)
        sparse_coords = index_domain_coords(self.domain, idx)
        ncells = sparse_coords.shape[0]
        if self.nattr == 1:
            attr = self.attr(0)
            name = attr.name
            value = np.asarray(val, dtype=attr.dtype)
            if len(value) != ncells:
                raise AttributeError("value length does not match coordinate length")
            sparse_values = dict(((name, value),))
        else:
            sparse_values = dict()
            for (k, v) in dict(val).items():
                attr = self.attr(k)
                name = attr.name
                value = np.asarray(v, dtype=attr.dtype)
                if len(value) != ncells:
                    raise AttributeError("value length does not match coordinate length")
                sparse_values[name] = value
        self._write_sparse(sparse_coords, sparse_values)
        return

    cdef _write_sparse(self, np.ndarray coords, dict values):
        cdef Ctx ctx = self.ctx
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr

        # array name
        cdef bytes barray_name = self.uri.encode('UTF-8')

        # attr names
        battr_names = list(bytes(k, 'UTF-8') for k in values.keys())
        cdef size_t nattr = self.nattr
        cdef const char** c_attr_names = <const char**> calloc(nattr + 1, sizeof(uintptr_t))
        for i in range(nattr):
            c_attr_names[i] = PyBytes_AS_STRING(battr_names[i])
        c_attr_names[nattr] = tiledb_coords()
        attr_values = list(values.values())
        cdef void** buffers = <void**> calloc(nattr + 1, sizeof(uintptr_t))
        cdef uint64_t* buffer_sizes = <uint64_t*> calloc(nattr + 1, sizeof(uint64_t))
        cdef np.ndarray array_value
        for i in range(nattr):
            array_value = attr_values[i]
            buffers[i] = np.PyArray_DATA(array_value)
            buffer_sizes[i] = <uint64_t> array_value.nbytes
        buffers[nattr] = np.PyArray_DATA(coords)
        buffer_sizes[nattr] = <uint64_t> coords.nbytes
        cdef tiledb_query_t* query_ptr = NULL
        check_error(ctx,
                    tiledb_query_create(ctx_ptr, &query_ptr, barray_name, TILEDB_WRITE))
        check_error(ctx,
                    tiledb_query_set_layout(ctx_ptr, query_ptr, TILEDB_UNORDERED))
        check_error(ctx,
                    tiledb_query_set_buffers(ctx_ptr, query_ptr, c_attr_names, nattr + 1, buffers, buffer_sizes))
        cdef int rc = TILEDB_OK
        with nogil:
            rc = tiledb_query_submit(ctx_ptr, query_ptr)
        tiledb_query_free(ctx_ptr, query_ptr)
        if rc != TILEDB_OK:
            check_error(ctx, rc)
        return

    def __getitem__(self, object key):
        idx = index_as_tuple(key)
        idx = replace_ellipsis(self.domain, idx)
        idx, drop_axes = replace_scalars_slice(self.domain, idx)
        subarray = index_domain_subarray(self.domain, idx)
        attr_names = [self.attr(i).name for i in range(self.nattr)]
        return self._read_sparse_subarray(subarray, attr_names)

    def _sparse_read_query(self, object idx):
        sparse_coords = index_domain_coords(self.domain, idx)
        ncells = sparse_coords.shape[0]
        sparse_values = dict()
        for i in range(self.nattr):
            attr = self.attr(i)
            name = attr.name
            value = np.zeros(shape=ncells, dtype=attr.dtype)
            sparse_values[name] = value
        self._read_sparse_multiple(sparse_coords, sparse_values)
        # attribute is anonymous, just return the result
        if self.nattr == 1 and self.attr(0).isanon:
            return sparse_values[self.attr(0).name]
        return sparse_values

    cpdef np.dtype coords_dtype(self):
        # TODO: should be a method of domain?
        # returns the record array dtype of the coordinate array
        return np.dtype([(dim.name, dim.dtype) for dim in self.domain])

    cdef _read_sparse_subarray(self, np.ndarray subarray, list attr_names):
        # ctx references
        cdef Ctx ctx = self.ctx
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr

        # array name
        cdef bytes barray_uri = self.uri.encode('UTF-8')

        # attr name
        cdef const char* attr_ptr = tiledb_coords()
        cdef void* subarray_ptr = np.PyArray_DATA(subarray)

        cdef tiledb_query_t* query_ptr = NULL
        check_error(ctx,
                    tiledb_query_create(ctx_ptr, &query_ptr, barray_uri, TILEDB_READ))
        check_error(ctx,
                    tiledb_query_set_subarray(ctx_ptr, query_ptr, <void*> subarray_ptr))
        check_error(ctx,
                    tiledb_query_set_layout(ctx_ptr, query_ptr, TILEDB_ROW_MAJOR))

        # attr names
        battr_names = list(bytes(k, 'UTF-8') for k in attr_names)

        cdef size_t nattr = len(battr_names) + 1  # coordinates
        cdef const char** attr_names_ptr = <const char**> PyMem_Malloc(nattr * sizeof(uintptr_t))
        if attr_names_ptr == NULL:
            raise MemoryError()
        try:
            attr_names_ptr[0] = tiledb_coords()
            for i in range(1, nattr):
                attr_names_ptr[i] = PyBytes_AS_STRING(battr_names[i - 1])
        except:
            PyMem_Free(attr_names_ptr)
            raise

        cdef uint64_t* buffer_sizes_ptr = <uint64_t*> PyMem_Malloc(nattr * sizeof(uint64_t))
        if buffer_sizes_ptr == NULL:
            PyMem_Free(attr_names_ptr)
            raise MemoryError()

        # check the max read buffer size
        cdef int rc = tiledb_array_compute_max_read_buffer_sizes(
            ctx_ptr, barray_uri, subarray_ptr, attr_names_ptr, nattr, buffer_sizes_ptr)
        if rc != TILEDB_OK:
            PyMem_Free(attr_names_ptr)
            PyMem_Free(buffer_sizes_ptr)
            check_error(ctx, rc)

        # allocate the max read buffer size
        cdef void** buffers_ptr = <void**> PyMem_Malloc(nattr * sizeof(uintptr_t))
        if buffers_ptr == NULL:
            PyMem_Free(attr_names_ptr)
            PyMem_Free(buffer_sizes_ptr)
            raise MemoryError()

        # initalize the buffer ptrs
        for i in range(nattr):
            buffers_ptr[i] = <void*> PyMem_Malloc(buffer_sizes_ptr[i])
            if buffers_ptr[i] == NULL:
                PyMem_Free(attr_names_ptr)
                PyMem_Free(buffer_sizes_ptr)
                for j in range(i):
                    PyMem_Free(buffers_ptr[j])
                PyMem_Free(buffers_ptr)
                raise MemoryError()

        rc = tiledb_query_set_buffers(ctx_ptr, query_ptr, &attr_ptr, nattr,
                                      buffers_ptr, buffer_sizes_ptr)
        if rc != TILEDB_OK:
            PyMem_Free(attr_names_ptr)
            PyMem_Free(buffer_sizes_ptr)
            for i in range(nattr):
                PyMem_Free(buffers_ptr[i])
            PyMem_Free(buffers_ptr)
            check_error(ctx, rc)

        with nogil:
            rc = tiledb_query_submit(ctx_ptr, query_ptr)
        tiledb_query_free(ctx_ptr, query_ptr)

        if rc != TILEDB_OK:
            PyMem_Free(attr_names_ptr)
            PyMem_Free(buffer_sizes_ptr)
            for i in range(nattr):
                PyMem_Free(buffers_ptr[i])
            PyMem_Free(buffers_ptr)
            check_error(ctx, rc)

        # collect a list of dtypes for resulting to construct array
        dtypes = list()
        try:
            dtypes.append(self.coords_dtype())
            for i in range(1, nattr):
                dtypes.append(self.attr(i - 1).dtype)
            # we need to increase the reference count of all dtype objects
            # because PyArray_NewFromDescr steals a reference
            for i in range(nattr):
                Py_INCREF(dtypes[i])
        except:
            PyMem_Free(attr_names_ptr)
            PyMem_Free(buffer_sizes_ptr)
            for i in range(nattr):
                PyMem_Free(buffers_ptr[i])
            PyMem_Free(buffers_ptr)
            raise

        cdef object out = OrderedDict()
        # all results are 1-d vectors
        cdef np.npy_intp dims[1]
        # coordinates
        try:
            dtype = dtypes[0]
            dims[0] = buffer_sizes_ptr[0] / dtype.itemsize
            out["coords"] = PyArray_NewFromDescr(
                <PyTypeObject*> np.ndarray,
                dtype, 1, dims, NULL,
                PyMem_Realloc(buffers_ptr[0], buffer_sizes_ptr[0]),
                np.NPY_OWNDATA, <object> NULL)
        except:
            PyMem_Free(attr_names_ptr)
            PyMem_Free(buffer_sizes_ptr)
            for i in range(nattr):
                PyMem_Free(buffers_ptr[i])
            PyMem_Free(buffers_ptr)
            raise

        # attributes
        for i in range(1, nattr):
            try:
                dtype = dtypes[i]
                dims[0] = buffer_sizes_ptr[i] / dtypes[i].itemsize
                out[attr_names[i - 1]] = \
                    PyArray_NewFromDescr(
                        <PyTypeObject*> np.ndarray,
                        dtype, 1, dims, NULL,
                        PyMem_Realloc(buffers_ptr[i], buffer_sizes_ptr[i]),
                        np.NPY_OWNDATA, <object> NULL)
            except:
                PyMem_Free(attr_names_ptr)
                PyMem_Free(buffer_sizes_ptr)
                # the previous buffers are now "owned"
                # by the constructed numpy arrays
                for j in range(i, nattr):
                    PyMem_Free(buffers_ptr[i])
                PyMem_Free(buffers_ptr)
                raise
        return out

    cdef _read_sparse_multiple(self, np.ndarray coords, dict values):
        # ctx references
        cdef Ctx ctx = self.ctx
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr

        # array name
        cdef bytes barray_name = self.uri.encode('UTF-8')

        # attr name
        battr_names = list(bytes(k, 'UTF-8') for k in values.keys())
        cdef size_t nattr = self.nattr
        cdef const char** c_attr_names = <const char**> calloc(nattr + 1, sizeof(uintptr_t))
        try:
            for i in range(nattr):
                c_attr_names[i] = PyBytes_AS_STRING(battr_names[i])
            c_attr_names[nattr] = tiledb_coords()
        except:
            free(c_attr_names)
            raise
        attr_values = list(values.values())
        cdef size_t ncoords = coords.shape[0]
        cdef char** buffers = <char**> calloc(nattr + 1, sizeof(uintptr_t))
        cdef uint64_t* buffer_sizes = <uint64_t*> calloc(nattr + 1, sizeof(uint64_t))
        cdef uint64_t* buffer_item_sizes = <uint64_t*> calloc(nattr + 1, sizeof(uint64_t))
        cdef np.ndarray array_value
        try:
            for i in range(nattr):
                array_value = attr_values[i]
                buffers[i] = <char*> np.PyArray_DATA(array_value)
                buffer_sizes[i] = <uint64_t> array_value.nbytes
                buffer_item_sizes[i] = <uint64_t> array_value.dtype.itemsize
            buffers[nattr] = <char*> np.PyArray_DATA(coords)
            buffer_sizes[nattr] = <uint64_t> coords.nbytes
            buffer_item_sizes[nattr] = <uint64_t> (coords.dtype.itemsize * self.domain.rank)
        except:
            free(c_attr_names)
            free(buffers)
            free(buffer_sizes)
            raise

        cdef int rc = TILEDB_OK
        cdef tiledb_query_t* query_ptr = NULL
        rc = tiledb_query_create(ctx_ptr, &query_ptr, barray_name, TILEDB_READ)
        if rc != TILEDB_OK:
            free(c_attr_names)
            free(buffers)
            free(buffer_sizes)
            check_error(ctx, rc)

        rc = tiledb_query_set_layout(ctx_ptr, query_ptr, TILEDB_GLOBAL_ORDER)
        if rc != TILEDB_OK:
            free(c_attr_names)
            free(buffers)
            free(buffer_sizes)
            check_error(ctx, rc)

        for i in range(ncoords):
            if i == 0:
                rc = tiledb_query_set_buffers(ctx_ptr, query_ptr, c_attr_names,
                                              nattr + 1, <void**> buffers, buffer_item_sizes)
            else:
                for aidx in range(nattr):
                    buffers[aidx] += buffer_item_sizes[aidx]
                buffers[nattr] += buffer_item_sizes[nattr]
                rc = tiledb_query_reset_buffers(ctx_ptr, query_ptr,
                                                <void**> buffers, buffer_item_sizes)
            if rc != TILEDB_OK:
                break
            with nogil:
                rc = tiledb_query_submit(ctx_ptr, query_ptr)
            if rc != TILEDB_OK:
                break

        free(c_attr_names)
        free(buffers)
        free(buffer_sizes)
        tiledb_query_free(ctx_ptr, query_ptr)
        if rc != TILEDB_OK:
            check_error(ctx, rc)
        return


def array_consolidate(Ctx ctx, path):
    cdef bytes bpath = unicode_path(path)
    check_error(ctx,
                tiledb_array_consolidate(ctx.ptr, bpath))
    return path


def group_create(Ctx ctx, path):
    cdef bytes bpath = unicode_path(path)
    check_error(ctx,
                tiledb_group_create(ctx.ptr, bpath))
    return path


def object_type(Ctx ctx, path):
    cdef bytes bpath = unicode_path(path)
    cdef tiledb_object_t obj = TILEDB_INVALID
    check_error(ctx,
                tiledb_object_type(ctx.ptr, bpath, &obj))
    return obj


def delete(Ctx ctx, path):
    cdef bytes bpath = unicode_path(path)
    cdef const char* path_ptr = PyBytes_AS_STRING(bpath)
    check_error(ctx,
                tiledb_object_remove(ctx.ptr, path_ptr))
    return


def move(Ctx ctx, oldpath, newpath, force=False):
    cdef bytes boldpath = unicode_path(oldpath)
    cdef bytes bnewpath = unicode_path(newpath)
    cdef int force_move = 1 if force else 0
    check_error(ctx,
                tiledb_object_move(ctx.ptr, boldpath, bnewpath, force_move))
    return


cdef int walk_callback(const char* c_path, tiledb_object_t obj, void* pyfunc):
    objtype = None
    if obj == TILEDB_ARRAY:
        objtype = "array"
    elif obj == TILEDB_KEY_VALUE:
        objtype = "kv "
    elif obj == TILEDB_GROUP:
        objtype = "group"
    try:
        (<object> pyfunc)(c_path.decode('UTF-8'), objtype)
    except StopIteration:
        return 0
    return 1


def walk(Ctx ctx, path, func, order="preorder"):
    cdef bytes bpath = unicode_path(path)
    cdef tiledb_walk_order_t walk_order
    if order == "postorder":
        walk_order = TILEDB_POSTORDER
    elif order == "preorder":
        walk_order = TILEDB_PREORDER
    else:
        raise AttributeError("unknown walk order {}".format(order))
    check_error(ctx,
                tiledb_object_walk(ctx.ptr, bpath, walk_order, walk_callback, <void*> func))
    return


cdef class FileHandle(object):

    cdef VFS vfs
    cdef unicode uri
    cdef tiledb_vfs_fh_t*ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        cdef Ctx ctx = self.vfs.ctx
        cdef tiledb_ctx_t* ctx_ptr = ctx.ptr
        if self.ptr != NULL:
            tiledb_vfs_fh_free(ctx_ptr, self.ptr)

    @staticmethod
    cdef from_ptr(VFS vfs, unicode uri, tiledb_vfs_fh_t* fh_ptr):
        cdef FileHandle fh = FileHandle.__new__(FileHandle)
        fh.vfs = vfs
        fh.uri = uri
        fh.ptr = fh_ptr
        return fh

    cpdef closed(self):
        cdef Ctx ctx = self.vfs.ctx
        cdef int isclosed = 0
        check_error(ctx,
                    tiledb_vfs_fh_is_closed(ctx.ptr, self.ptr, &isclosed))
        return bool(isclosed)

# VFS
cdef class VFS(object):

    cdef Ctx ctx
    cdef tiledb_vfs_t* ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr is not NULL:
            tiledb_vfs_free(self.ctx.ptr, self.ptr)

    def __init__(self, Ctx ctx, config=None):
        cdef Config _config = Config()
        if config is not None:
            if isinstance(config, Config):
                _config = config
            else:
                _config.update(config)
        cdef tiledb_vfs_t* vfs_ptr = NULL
        check_error(ctx,
                    tiledb_vfs_create(ctx.ptr, &vfs_ptr, _config.ptr))
        self.ctx = ctx
        self.ptr = vfs_ptr

    def create_bucket(self, uri):
        cdef bytes buri = unicode_path(uri)
        check_error(self.ctx,
                    tiledb_vfs_create_bucket(self.ctx.ptr, self.ptr, buri))
        return uri

    def remove_bucket(self, uri):
        cdef bytes buri = unicode_path(uri)
        check_error(self.ctx,
                    tiledb_vfs_remove_bucket(self.ctx.ptr, self.ptr, buri))
        return

    def empty_bucket(self, uri):
        cdef bytes buri = unicode_path(uri)
        check_error(self.ctx,
                    tiledb_vfs_empty_bucket(self.ctx.ptr, self.ptr, buri))
        return

    def is_empty_bucket(self, uri):
        cdef bytes buri = unicode_path(uri)
        cdef int isempty = 0
        check_error(self.ctx,
                    tiledb_vfs_is_empty_bucket(self.ctx.ptr, self.ptr, buri, &isempty))
        return bool(isempty)

    def is_bucket(self, uri):
        cdef bytes buri = unicode_path(uri)
        cdef int is_bucket = 0
        check_error(self.ctx,
                    tiledb_vfs_is_bucket(self.ctx.ptr, self.ptr, buri, &is_bucket))
        return bool(is_bucket)

    def create_dir(self, uri):
        cdef bytes buri = unicode_path(uri)
        check_error(self.ctx,
                    tiledb_vfs_create_dir(self.ctx.ptr, self.ptr, buri))
        return uri

    def is_dir(self, uri):
        cdef bytes buri = unicode_path(uri)
        cdef int is_dir = 0
        check_error(self.ctx,
                    tiledb_vfs_is_dir(self.ctx.ptr, self.ptr, buri, &is_dir))
        return bool(is_dir)

    def remove_dir(self, uri):
        cdef bytes buri = unicode_path(uri)
        check_error(self.ctx,
                    tiledb_vfs_remove_dir(self.ctx.ptr, self.ptr, buri))
        return

    def is_file(self, uri):
        cdef bytes buri = unicode_path(uri)
        cdef int is_file = 0
        check_error(self.ctx,
                    tiledb_vfs_is_file(self.ctx.ptr, self.ptr, buri, &is_file))
        return bool(is_file)

    def remove_file(self, uri):
        cdef bytes buri = unicode_path(uri)
        check_error(self.ctx,
                    tiledb_vfs_remove_file(self.ctx.ptr, self.ptr, buri))
        return

    def file_size(self, uri):
        cdef bytes buri = unicode_path(uri)
        cdef uint64_t nbytes = 0
        check_error(self.ctx,
                    tiledb_vfs_file_size(self.ctx.ptr, self.ptr, buri, &nbytes))
        return int(nbytes)

    def move(self, old_uri, new_uri, force=False):
        cdef bytes bold_uri = unicode_path(old_uri)
        cdef bytes bnew_uri = unicode_path(new_uri)
        cdef int force_move = 0
        if force:
            force_move = 1
        check_error(self.ctx,
                    tiledb_vfs_move(self.ctx.ptr, self.ptr, bold_uri, bnew_uri, force_move))
        return

    def open(self, uri, mode=None):
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
        cdef tiledb_vfs_fh_t* fh_ptr = NULL
        check_error(self.ctx,
                    tiledb_vfs_open(self.ctx.ptr, self.ptr, buri, vfs_mode, &fh_ptr))
        cdef FileHandle fh = FileHandle.from_ptr(self, uri, fh_ptr)
        return fh

    def close(self, FileHandle fh):
        check_error(self.ctx,
                    tiledb_vfs_close(self.ctx.ptr, fh.ptr))
        return fh

    def readinto(self, FileHandle fh, bytes buffer, offset, nbytes):
        if offset < 0:
            raise AttributeError("read offset must be >= 0")
        if nbytes < 0:
            raise AttributeError("read nbytes but be >= 0")
        if nbytes > len(buffer):
            raise AttributeError("read buffer is smaller than nbytes")
        cdef Py_ssize_t _offset = offset
        cdef Py_ssize_t _nbytes = nbytes
        cdef char* buffer_ptr = PyBytes_AS_STRING(buffer)
        check_error(self.ctx,
                    tiledb_vfs_read(self.ctx.ptr,
                                    fh.ptr,
                                    <uint64_t> _offset,
                                    <void*> buffer_ptr,
                                    <uint64_t> _nbytes))
        return buffer

    def read(self, FileHandle fh, offset, nbytes):
        cdef Py_ssize_t _nbytes = nbytes
        cdef bytes buffer = PyBytes_FromStringAndSize(NULL, _nbytes)
        return self.readinto(fh, buffer, offset, nbytes)

    def write(self, FileHandle fh, offset, buff):
        if offset < 0:
            raise AttributeError("read offset must be >= 0")
        cdef bytes buffer = bytes(buff)
        cdef const char* buffer_ptr = PyBytes_AS_STRING(buffer)
        cdef Py_ssize_t _nbytes = PyBytes_GET_SIZE(buffer)
        check_error(self.ctx,
                    tiledb_vfs_write(self.ctx.ptr,
                                     fh.ptr,
                                     <const void*> buffer_ptr,
                                     <uint64_t> _nbytes))
        return

    def sync(self, FileHandle fh):
        check_error(self.ctx,
                    tiledb_vfs_sync(self.ctx.ptr, fh.ptr))
        return fh

    def touch(self, uri):
        cdef bytes buri = unicode_path(uri)
        check_error(self.ctx,
                    tiledb_vfs_touch(self.ctx.ptr, self.ptr, buri))
        return uri

    def supports(self, scheme):
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
            raise TileDBError("unsupported vfs scheme '{0!s}://'".format(scheme))


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
            raise AttributeError("invalid mode {0!r}".format(mode))
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
                raise AttributeError("offset must be a positive or zero value when SEEK_SET")
            self._offset = offset
        elif whence == 1:
            self._offset += offset
        elif whence == 2:
            self._offset = self._nbytes + offset
        else:
            raise AttributeError('whence must be equal to SEEK_SET, SEEK_START, SEEK_END')
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
        if size < 0:
            nbytes = nbytes_remaining
        elif size > nbytes_remaining:
            nbytes = nbytes_remaining
        else:
            nbytes = size
        buff = bytes(nbytes)
        self.vfs.readinto(self.fh, buff, self._offset, nbytes)
        self._offset += nbytes
        return buff

    def readall(self):
        if self._mode == "w":
            raise IOError("cannot read from a write-only FileIO handle")
        if self.closed:
            raise IOError("cannot read from closed FileIO handle")
        nbytes = self._nbytes - self._offset
        if nbytes == 0:
            return bytes(0)
        buff = bytes(nbytes)
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
        self.vfs.write(self.fh, 0, buff)
        self._nbytes += nbytes
        self._offset += nbytes
        return nbytes