from cpython.version cimport PY_MAJOR_VERSION

from cpython cimport PyBytes_GET_SIZE

from cython.operator cimport dereference as deref
from libc.stdio cimport stdout
from libc.stdint cimport uint64_t, int64_t
from libc.stdlib cimport calloc

cdef extern from "Python.h":
    char* PyBytes_AS_STRING(object obj)

# Numpy imports
import numpy as np
cimport numpy as np

import sys
from os.path import abspath

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


class TileDBError(Exception):
    pass


cdef check_error(Ctx ctx, int rc):
    if rc == TILEDB_OK:
        return
    if rc == TILEDB_OOM:
        raise MemoryError()
    ctx_ptr = ctx.ptr
    cdef tiledb_error_t* err = NULL
    cdef int ret = tiledb_error_last(ctx_ptr, &err)
    if ret != TILEDB_OK:
        tiledb_error_free(ctx_ptr, err)
        if ret == TILEDB_OOM:
            raise MemoryError()
        raise TileDBError("error retrieving error object from ctx")
    cdef const char* err_msg = NULL
    ret = tiledb_error_message(ctx_ptr, err, &err_msg)
    if ret != TILEDB_OK:
        tiledb_error_free(ctx_ptr, err)
        if ret == TILEDB_OOM:
            return MemoryError()
        raise TileDBError("error retrieving error message from ctx")
    message_string = err_msg.decode('UTF-8', 'strict')
    tiledb_error_free(ctx_ptr, err)
    raise TileDBError(message_string)


cdef class Ctx(object):

    cdef tiledb_ctx_t* ptr

    def __cinit__(self):
        cdef int rc = tiledb_ctx_create(&self.ptr)
        if rc == TILEDB_OOM:
            raise MemoryError()
        if rc == TILEDB_ERR:
            raise TileDBError("unknown error creating tiledb.Ctx")

    def __dealloc__(self):
        if self.ptr is not NULL:
            tiledb_ctx_free(self.ptr)



cdef tiledb_datatype_t _tiledb_dtype(typ) except TILEDB_CHAR:
    dtype = np.dtype(typ)
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
    raise TypeError("data type {0!r} not understood".format(dtype))


cdef _numpy_type(tiledb_datatype_t dtype):
    if dtype == TILEDB_INT32:
        return np.int32
    elif dtype == TILEDB_UINT32:
        return np.uint32
    elif dtype == TILEDB_INT64:
        return np.int64
    elif dtype == TILEDB_UINT64:
        return np.uint64
    elif dtype == TILEDB_FLOAT32:
        return np.float32
    elif dtype == TILEDB_FLOAT64:
        return np.float64
    elif dtype == TILEDB_INT8:
        return np.int8
    elif dtype == TILEDB_UINT8:
        return np.uint8
    elif dtype == TILEDB_INT16:
        return np.int16
    elif dtype == TILEDB_UINT16:
        return np.uint16
    raise TypeError("tiledb datatype not understood")


cdef tiledb_compressor_t _tiledb_compressor(c) except TILEDB_NO_COMPRESSION:
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
    cdef from_ptr(Ctx ctx, const tiledb_attribute_t* ptr):
        cdef Attr attr = Attr.__new__(Attr)
        attr.ctx = ctx
        attr.ptr = <tiledb_attribute_t*> ptr
        return attr

    # TODO: use numpy compund dtypes to choose number of cells
    def __init__(self, Ctx ctx,  name="", dtype='f8', compressor=None, level=-1):
        uname = ustring(name).encode('UTF-8')
        cdef tiledb_attribute_t* attr_ptr = NULL
        cdef tiledb_compressor_t compr = TILEDB_NO_COMPRESSION
        cdef tiledb_datatype_t tiledb_dtype = _tiledb_dtype(dtype)
        check_error(ctx,
            tiledb_attribute_create(ctx.ptr, &attr_ptr, uname, tiledb_dtype))
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
        return np.dtype(_numpy_type(typ))

    @property
    def name(self):
        cdef const char* c_name = NULL
        check_error(self.ctx,
            tiledb_attribute_get_name(self.ctx.ptr, self.ptr, &c_name))
        return c_name.decode('UTF-8', 'strict')

    @property
    def compressor(self):
        cdef int c_level = -1
        cdef tiledb_compressor_t compr = TILEDB_NO_COMPRESSION
        check_error(self.ctx,
            tiledb_attribute_get_compressor(self.ctx.ptr, self.ptr, &compr, &c_level))
        if compr == TILEDB_NO_COMPRESSION:
            return (None, int(c_level))
        return (_tiledb_compressor_string(compr), int(c_level))


cdef _tiledb_tagged_val(tiledb_datatype_t dtype, const void* val):
    if dtype == TILEDB_UINT64:
        return deref(<uint64_t*> val)
    else:
        return 2

cdef tuple _dimension_domain():
    return ()


cdef tuple _dimension_tile_extent():
    return ()


cdef class Dim(object):

    cdef Ctx ctx
    cdef tiledb_dimension_t* ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr is not NULL:
            tiledb_dimension_free(self.ctx.ptr, self.ptr)

    @staticmethod
    cdef from_ptr(Ctx ctx, const tiledb_dimension_t* ptr):
        cdef Dim dim = Dim.__new__(Dim)
        dim.ctx = ctx
        dim.ptr = <tiledb_dimension_t*> ptr
        return dim

    def __init__(self, Ctx ctx, name=None, domain=None, tile=0):
        if len(domain) != 2:
            raise AttributeError("invalid domain extent")
        if name is None:
            name = u""
        cdef bytes bname = ustring(name).encode('UTF-8')
        # TODO: specialized for uint64 datatype
        cdef uint64_t* domain_ptr = <uint64_t*> calloc(2, sizeof(uint64_t))
        domain_ptr[0] = <uint64_t>(domain[0])
        domain_ptr[1] = <uint64_t>(domain[1])
        cdef uint64_t tile_extent = <uint64_t>(tile)
        cdef tiledb_dimension_t* dim_ptr = NULL
        check_error(ctx,
            tiledb_dimension_create(ctx.ptr,
                                    &dim_ptr,
                                    bname,
                                    TILEDB_UINT64,
                                    domain_ptr,
                                    &tile_extent))
        self.ctx = ctx
        self.ptr = dim_ptr

    @property
    def name(self):
        cdef const char* c_name = NULL
        check_error(self.ctx,
                    tiledb_dimension_get_name(self.ctx.ptr, self.ptr, &c_name))
        return c_name.decode('UTF-8', 'strict')

    @property
    def shape(self):
        # TODO: specialized for uint64 datatype
        cdef tuple domain = self.domain
        return ((domain[1] - domain[0] + 1),)

    @property
    def tile(self):
        #TODO: specialized for uint64 datatype
        cdef uint64_t* tile_ptr = NULL
        check_error(self.ctx,
            tiledb_dimension_get_tile_extent(self.ctx.ptr, self.ptr, <void**>(&tile_ptr)))
        if tile_ptr[0] == 0:
            # undefined tiles should span the whole dimension domain
            return self.shape[0]
        return tile_ptr[0]


    @property
    def domain(self):
        # TODO: specialized for uint64 datatype
        cdef uint64_t* domain_ptr = NULL
        check_error(self.ctx,
                    tiledb_dimension_get_domain(self.ctx.ptr,
                                                self.ptr,
                                                <void**>(&domain_ptr)))
        assert(domain_ptr != NULL)
        return (domain_ptr[0], domain_ptr[1])


cdef class Domain(object):

    cdef Ctx ctx
    cdef tiledb_domain_t* ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr is not NULL:
            tiledb_domain_free(self.ctx.ptr, self.ptr)

    @staticmethod
    cdef from_ptr(Ctx ctx, const tiledb_domain_t* ptr):
        cdef Domain dom = Domain.__new__(Domain)
        dom.ctx = ctx
        dom.ptr = <tiledb_domain_t*> ptr
        return dom

    def __init__(self, Ctx ctx, *dims, dtype='u8'):
        cdef tiledb_datatype_t domain_type = _tiledb_dtype(dtype)
        cdef tiledb_domain_t* domain_ptr = NULL
        check_error(ctx,
                    tiledb_domain_create(ctx.ptr, &domain_ptr, domain_type))

        cdef int rc = TILEDB_OK
        cdef Dim dimension
        cdef tiledb_dimension_t* dimension_ptr = NULL
        for dim in dims:
            if not isinstance(dim, Dim):
                tiledb_domain_free(ctx.ptr, domain_ptr)
                raise TypeError("unknown dimension type {0!r}".format(dim))
            dimension = dim
            rc = tiledb_domain_add_dimension(
                ctx.ptr, domain_ptr, dimension.ptr)
            if rc != TILEDB_OK:
                tiledb_domain_free(ctx.ptr, domain_ptr)
                check_error(ctx, rc)
        self.ctx = ctx
        self.ptr = domain_ptr

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
        check_error(self.ctx, tiledb_domain_get_type(self.ctx.ptr, self.ptr, &typ))
        return np.dtype(_numpy_type(typ))

    @property
    def shape(self):
        return tuple(self.dim(i).shape[0] for i in range(self.rank))

    def dim(self, int idx):
        cdef tiledb_dimension_t* dim_ptr = NULL
        check_error(self.ctx,
                    tiledb_dimension_from_index(self.ctx.ptr, self.ptr, idx, &dim_ptr))
        assert(dim_ptr != NULL)
        return Dim.from_ptr(self.ctx, dim_ptr)

    def dim(self, unicode name):
        cdef bytes uname = ustring(name).encode('UTF-8')
        cdef const char* c_name = uname
        cdef tiledb_dimension_t* dim_ptr = NULL
        check_error(self.ctx,
                    tiledb_dimension_from_name(self.ctx.ptr, self.ptr, c_name, &dim_ptr))
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


cdef class Query(object):

    cdef Ctx ctx
    cdef tiledb_query_t* ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr is not NULL:
            tiledb_query_free(self.ctx.ptr, self.ptr)

    def __init__(self, Ctx ctx):
        self.ctx = ctx


cdef class Assoc(object):

    cdef Ctx ctx
    cdef unicode name
    cdef tiledb_array_metadata_t* ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr is not NULL:
            tiledb_array_metadata_free(self.ctx.ptr, self.ptr)

    @staticmethod
    cdef from_ptr(Ctx ctx, unicode name, const tiledb_array_metadata_t* ptr):
        cdef Assoc arr = Assoc.__new__(Assoc)
        arr.ctx = ctx
        arr.name = name
        arr.ptr = <tiledb_array_metadata_t*> ptr
        return arr

    @staticmethod
    def load(Ctx ctx, unicode uri):
        cdef bytes buri = ustring(uri).encode('UTF-8')
        cdef tiledb_array_metadata_t* metadata_ptr = NULL
        cdef int rc = tiledb_array_metadata_load(ctx.ptr, &metadata_ptr, buri)
        if rc != TILEDB_OK:
            check_error(ctx, rc)
        cdef int is_kv = 0;
        rc = tiledb_array_metadata_get_as_kv(ctx.ptr, metadata_ptr, &is_kv)
        if rc != TILEDB_OK:
            tiledb_array_metadata_free(ctx.ptr, metadata_ptr)
            check_error(ctx, rc)
        if not is_kv:
            tiledb_array_metadata_free(ctx.ptr, metadata_ptr)
            raise TileDBError("TileDB Array {0!r} is not an Assoc array".format(uri))
        return Assoc.from_ptr(ctx, uri, metadata_ptr)

    def __init__(self, Ctx ctx, unicode name, *attrs, int capacity=0):

        #TODO: key types other than strings
        uname = ustring(name).encode('UTF-8')

        cdef int rc = TILEDB_OK
        cdef tiledb_array_metadata_t* metadata_ptr = NULL
        check_error(ctx,
            tiledb_array_metadata_create(ctx.ptr, &metadata_ptr, uname))

        rc = tiledb_array_metadata_set_as_kv(ctx.ptr, metadata_ptr)
        if rc != TILEDB_OK:
            tiledb_array_metadata_free(ctx.ptr, metadata_ptr)
            check_error(ctx, rc)

        cdef uint64_t c_capacity = capacity
        if capacity > 0:
            rc = tiledb_array_metadata_set_capacity(ctx.ptr, metadata_ptr, c_capacity)
            if rc != TILEDB_OK:
                tiledb_array_metadata_free(ctx.ptr, metadata_ptr)
                check_error(ctx, rc)

        cdef tiledb_attribute_t* attr_ptr = NULL
        for attr in attrs:
            attr_ptr = (<Attr> attr).ptr
            rc = tiledb_array_metadata_add_attribute(ctx.ptr, metadata_ptr, attr_ptr)
            if rc != TILEDB_OK:
                tiledb_array_metadata_free(ctx.ptr, metadata_ptr)
                check_error(ctx, rc)

        rc = tiledb_array_metadata_check(ctx.ptr, metadata_ptr)
        if rc != TILEDB_OK:
            tiledb_array_metadata_free(ctx.ptr, metadata_ptr)
            check_error(ctx, rc)

        rc = tiledb_array_create(ctx.ptr, metadata_ptr)
        if rc != TILEDB_OK:
            tiledb_array_metadata_free(ctx.ptr, metadata_ptr)
            check_error(ctx, rc)

        self.ctx = ctx
        self.name = name
        self.ptr = metadata_ptr

    @property
    def nattr(self):
        cdef unsigned int nattr = 0
        check_error(self.ctx,
            tiledb_array_metadata_get_num_attributes(self.ctx.ptr, self.ptr, &nattr))
        return int(nattr)

    def attr(self, int idx):
        cdef tiledb_attribute_t* attr_ptr = NULL
        check_error(self.ctx,
                    tiledb_attribute_from_index(self.ctx.ptr, self.ptr, idx, &attr_ptr))
        return Attr.from_ptr(self.ctx, attr_ptr)

    def __setitem__(self, unicode key, int value):
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr

        cdef bytes bkey = ustring(key).encode('UTF-8')
        cdef const void* bkey_ptr = PyBytes_AS_STRING(bkey)
        cdef uint64_t bkey_size = PyBytes_GET_SIZE(bkey)

        # create kv object
        cdef bytes battr = b"value"
        cdef const char* battr_ptr = PyBytes_AS_STRING(battr)

        cdef tiledb_datatype_t typ = TILEDB_INT64
        cdef unsigned int nitems = 1

        cdef tiledb_kv_t* kv_ptr = NULL
        check_error(self.ctx,
                tiledb_kv_create(ctx_ptr, &kv_ptr, 1, &battr_ptr, &typ, &nitems))

        # add key
        cdef int rc
        rc = tiledb_kv_add_key(ctx_ptr, kv_ptr, bkey_ptr, TILEDB_CHAR, bkey_size)
        if rc != TILEDB_OK:
            tiledb_kv_free(ctx_ptr, kv_ptr)
            check_error(self.ctx, rc)

        # add value
        cdef int64_t val = value
        rc = tiledb_kv_add_value(ctx_ptr, kv_ptr, 0, <void*>(&val))
        if rc != TILEDB_OK:
            tiledb_kv_free(ctx_ptr, kv_ptr)
            check_error(self.ctx, rc)

        # Create query
        cdef bytes bname = self.name.encode('UTF-8')
        cdef const char* bname_ptr = PyBytes_AS_STRING(bname)

        cdef tiledb_query_t* query_ptr;
        rc = tiledb_query_create(ctx_ptr, &query_ptr, bname_ptr, TILEDB_WRITE)
        if rc != TILEDB_OK:
            tiledb_kv_free(ctx_ptr, kv_ptr)
            check_error(self.ctx, rc)

        rc = tiledb_query_set_kv(ctx_ptr, query_ptr, kv_ptr);
        if rc != TILEDB_OK:
            tiledb_query_free(ctx_ptr, query_ptr)
            tiledb_kv_free(ctx_ptr, kv_ptr)
            check_error(self.ctx, rc)

        # Submit query
        rc = tiledb_query_submit(ctx_ptr, query_ptr);
        tiledb_query_free(ctx_ptr, query_ptr)
        tiledb_kv_free(ctx_ptr, kv_ptr)
        if rc != TILEDB_OK:
            check_error(self.ctx, rc)
        return

    def __getitem__(self, unicode key):
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr

        cdef bytes bkey = ustring(key).encode('UTF-8')
        cdef const void* bkey_ptr = PyBytes_AS_STRING(bkey)
        cdef uint64_t bkey_size = PyBytes_GET_SIZE(bkey)

        # create kv object
        cdef bytes battr = b"value"
        cdef const char* battr_ptr = PyBytes_AS_STRING(battr)

        cdef tiledb_datatype_t typ = TILEDB_INT64
        cdef unsigned int nitems = 1

        cdef tiledb_kv_t* kv_ptr = NULL
        check_error(self.ctx,
                tiledb_kv_create(ctx_ptr, &kv_ptr, 1, &battr_ptr, &typ, &nitems))

        # Create query
        cdef bytes bname = self.name.encode('UTF-8')
        cdef const char* bname_ptr = PyBytes_AS_STRING(bname)

        cdef tiledb_query_t* query_ptr;
        rc = tiledb_query_create(ctx_ptr, &query_ptr, bname_ptr, TILEDB_READ)
        if rc != TILEDB_OK:
            tiledb_kv_free(ctx_ptr, kv_ptr)
            check_error(self.ctx, rc)

        rc = tiledb_query_set_kv_key(ctx_ptr, query_ptr, bkey_ptr, TILEDB_CHAR, bkey_size)
        if rc != TILEDB_OK:
            tiledb_query_free(ctx_ptr, query_ptr)
            tiledb_kv_free(ctx_ptr, kv_ptr)
            check_error(self.ctx, rc)

        rc = tiledb_query_set_kv(ctx_ptr, query_ptr, kv_ptr);
        if rc != TILEDB_OK:
            tiledb_query_free(ctx_ptr, query_ptr)
            tiledb_kv_free(ctx_ptr, kv_ptr)
            check_error(self.ctx, rc)

        # Submit query
        rc = tiledb_query_submit(ctx_ptr, query_ptr)
        if rc != TILEDB_OK:
            tiledb_query_free(ctx_ptr, query_ptr)
            tiledb_kv_free(ctx_ptr, kv_ptr)
            check_error(self.ctx, rc)

        cdef int64_t* value_ptr = NULL;
        rc = tiledb_kv_get_value(ctx_ptr, kv_ptr, 0, 0, <void**>(&value_ptr))
        if value_ptr == NULL:
            if value_ptr == NULL:
                raise TileDBError("KV get value is NULL")
        cdef int64_t val = value_ptr[0]
        tiledb_query_free(ctx_ptr, query_ptr)
        tiledb_kv_free(ctx_ptr, kv_ptr)
        return val


cdef class Array(object):

    cdef Ctx ctx
    cdef unicode name
    cdef tiledb_array_metadata_t* ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr is not NULL:
            tiledb_array_metadata_free(self.ctx.ptr, self.ptr)

    @staticmethod
    def create(Ctx ctx,
               unicode name,
               domain=None,
               attrs=[],
               cell_order='row-major',
               tile_order='row-major',
               capacity=0,
               sparse=False):
        uname = ustring(name).encode('UTF-8')
        cdef tiledb_array_metadata_t* metadata_ptr = NULL
        check_error(ctx,
            tiledb_array_metadata_create(ctx.ptr, &metadata_ptr, uname))
        cdef tiledb_layout_t cell_layout = _tiledb_layout(cell_order)
        cdef tiledb_layout_t tile_layout = _tiledb_layout(tile_order)
        cdef tiledb_array_type_t array_type = TILEDB_SPARSE if sparse else TILEDB_DENSE
        tiledb_array_metadata_set_array_type(
            ctx.ptr, metadata_ptr, array_type)
        tiledb_array_metadata_set_cell_order(
            ctx.ptr, metadata_ptr, cell_layout)
        tiledb_array_metadata_set_tile_order(
            ctx.ptr, metadata_ptr, tile_layout)
        cdef uint64_t c_capacity = 0
        if sparse and capacity > 0:
            c_capacity = <uint64_t>capacity
            tiledb_array_metadata_set_capacity(ctx.ptr, metadata_ptr, c_capacity)
        cdef tiledb_domain_t* domain_ptr = (<Domain>domain).ptr
        tiledb_array_metadata_set_domain(
            ctx.ptr, metadata_ptr, domain_ptr)
        cdef tiledb_attribute_t* attr_ptr = NULL
        for attr in attrs:
            attr_ptr = (<Attr> attr).ptr
            tiledb_array_metadata_add_attribute(
                ctx.ptr, metadata_ptr, attr_ptr)
        cdef int rc = TILEDB_OK
        rc = tiledb_array_metadata_check(ctx.ptr, metadata_ptr)
        if rc != TILEDB_OK:
            tiledb_array_metadata_free(ctx.ptr, metadata_ptr)
            check_error(ctx, rc)
        rc = tiledb_array_create(ctx.ptr, metadata_ptr)
        if rc != TILEDB_OK:
            check_error(ctx, rc)
        return Array.from_ptr(ctx, name, metadata_ptr)

    @staticmethod
    cdef from_ptr(Ctx ctx, unicode name, const tiledb_array_metadata_t* ptr):
        cdef Array arr = Array.__new__(Array)
        arr.ctx = ctx
        arr.name = name
        arr.ptr = <tiledb_array_metadata_t*> ptr
        return arr

    @staticmethod
    def from_numpy(Ctx ctx, unicode path, np.ndarray array, **kw):
        shape = array.shape
        ndims = array.ndim
        dtype = array.dtype
        dims  = []
        for d in range(ndims):
            extent = shape[d]
            domain = (0, extent - 1)
            dims.append(Dim(ctx, "", domain, extent))
        dom = Domain(ctx, *dims)
        att = Attr(ctx, "", dtype=dtype)
        arr = Array.create(ctx, path, domain=dom, attrs=[att], **kw)
        arr.write_direct("", array)
        return arr

    def __init__(self, Ctx ctx, unicode name):
        cdef bytes uname = ustring(name).encode('UTF-8')
        cdef const char* c_name = uname
        cdef tiledb_array_metadata_t* metadata_ptr = NULL
        check_error(ctx,
            tiledb_array_metadata_load(ctx.ptr, &metadata_ptr, c_name))
        self.ctx = ctx
        self.name = name
        self.ptr = metadata_ptr

    @property
    def name(self):
        return self.name

    @property
    def sparse(self):
        cdef tiledb_array_type_t typ = TILEDB_DENSE
        check_error(self.ctx,
            tiledb_array_metadata_get_array_type(self.ctx.ptr, self.ptr, &typ))
        return typ == TILEDB_SPARSE

    @property
    def capacity(self):
        cdef uint64_t cap = 0
        check_error(self.ctx,
            tiledb_array_metadata_get_capacity(self.ctx.ptr, self.ptr, &cap))
        return int(cap)

    @property
    def cell_order(self):
        cdef tiledb_layout_t order = TILEDB_UNORDERED
        check_error(self.ctx,
            tiledb_array_metadata_get_cell_order(self.ctx.ptr, self.ptr, &order))
        return _tiledb_layout_string(order)

    @property
    def tile_order(self):
        cdef tiledb_layout_t order = TILEDB_UNORDERED
        check_error(self.ctx,
            tiledb_array_metadata_get_tile_order(self.ctx.ptr, self.ptr, &order))
        return _tiledb_layout_string(order)

    @property
    def coord_compressor(self):
        cdef tiledb_compressor_t comp = TILEDB_NO_COMPRESSION
        cdef int level = -1
        check_error(self.ctx,
            tiledb_array_metadata_get_coords_compressor(
                self.ctx.ptr, self.ptr, &comp, &level))
        return (_tiledb_compressor_string(comp), int(level))

    @property
    def domain(self):
        cdef tiledb_domain_t* dom = NULL
        check_error(self.ctx,
            tiledb_array_metadata_get_domain(self.ctx.ptr, self.ptr, &dom))
        return Domain.from_ptr(self.ctx, dom)

    @property
    def nattr(self):
        cdef unsigned int nattr = 0
        check_error(self.ctx,
            tiledb_array_metadata_get_num_attributes(self.ctx.ptr, self.ptr, &nattr))
        return int(nattr)

    def attr(self, unicode name):
        cdef bytes bname = ustring(name).encode('UTF-8')
        cdef tiledb_attribute_t* attr_ptr = NULL
        check_error(self.ctx,
                    tiledb_attribute_from_name(self.ctx.ptr, self.ptr, bname, &attr_ptr))
        return Attr.from_ptr(self.ctx, attr_ptr)

    def attr(self, int idx):
        cdef tiledb_attribute_t* attr_ptr = NULL
        check_error(self.ctx,
                    tiledb_attribute_from_index(self.ctx.ptr, self.ptr, idx, &attr_ptr))
        return Attr.from_ptr(self.ctx, attr_ptr)

    def dump(self):
        check_error(self.ctx,
            tiledb_array_metadata_dump(self.ctx.ptr, self.ptr, stdout))
        print("\n")
        return

    def consolidate(self):
        return array_consolidate(self.ctx, self.name)

    cdef void _getrange(self, uint64_t start, uint64_t stop,
                        void* buff_ptr, uint64_t buff_size):
        # array name
        cdef bytes array_name = self.name.encode('UTF-8')
        cdef const char* c_aname = array_name

        # attr name
        cdef bytes attr_name = u"".encode('UTF-8')
        cdef const char* c_attr = attr_name

        cdef int rc = TILEDB_OK
        cdef tiledb_query_t* query = NULL
        cdef tiledb_ctx_t* ctx_ptr = self.ctx.ptr

        rc = tiledb_query_create(ctx_ptr, &query, c_aname, TILEDB_READ)
        if rc != TILEDB_OK:
            check_error(self.ctx, rc)
        cdef uint64_t[2] subarray
        subarray[0] = start
        subarray[1] = stop - 1
        rc = tiledb_query_set_subarray(ctx_ptr, query, <void*>(subarray), TILEDB_UINT64)
        if rc != TILEDB_OK:
            tiledb_query_free(ctx_ptr, query)
            check_error(self.ctx, rc)
        rc = tiledb_query_set_buffers(ctx_ptr, query, &c_attr, 1, &buff_ptr, &buff_size)
        if rc != TILEDB_OK:
            tiledb_query_free(ctx_ptr, query)
            check_error(self.ctx, rc)
        rc = tiledb_query_set_layout(ctx_ptr, query, TILEDB_ROW_MAJOR)
        if rc != TILEDB_OK:
            tiledb_query_free(ctx_ptr, query)
            check_error(self.ctx, rc)

        rc = tiledb_query_submit(ctx_ptr, query)
        tiledb_query_free(ctx_ptr, query)
        if rc != TILEDB_OK:
            check_error(self.ctx, rc)
        return

    def __getitem__(self, object key):
        cdef np.dtype _dtype
        cdef np.ndarray array
        cdef object start, stop, step

        if isinstance(key, _inttypes):
            raise IndexError("key not suitable:", key)
        elif isinstance(key, slice):
            (start, stop, step) = key.start, key.stop, key.step
            print(start, stop, step)
        else:
            raise IndexError("key not suitable:", key)

        cdef tuple domain_shape = self.domain.shape
        cdef Attr attr = self.attr("")
        cdef np.dtype attr_dtype = attr.dtype

        # clamp to domain
        stop = domain_shape[0] if stop > domain_shape[0] else stop
        array = np.zeros(shape=((stop - start),), dtype=attr_dtype)
        cdef void* buff_ptr = np.PyArray_DATA(array)
        cdef uint64_t buff_size = <uint64_t>(array.nbytes)
        self._getrange(start, stop, buff_ptr, buff_size)
        if step:
            return array[::step]
        return array

    def __array__(self, dtype=None, **kw):
        array = self.read_direct("")
        if dtype and array.dtype != dtype:
            return array.astype(dtype)
        return array

    def write_direct(self, unicode attr, np.ndarray array not None):
        # array name
        cdef bytes barray_name = self.name.encode('UTF-8')
        cdef const char* c_array_name = barray_name

        # attr name
        cdef bytes battr_name = attr.encode('UTF-8')
        cdef const char* c_attr_name = battr_name

        cdef void* buff = np.PyArray_DATA(array)
        cdef uint64_t buff_size = array.nbytes

        cdef tiledb_query_t* query = NULL
        check_error(self.ctx,
            tiledb_query_create(self.ctx.ptr, &query, c_array_name, TILEDB_WRITE))
        check_error(self.ctx,
            tiledb_query_set_layout(self.ctx.ptr, query, TILEDB_ROW_MAJOR))
        check_error(self.ctx,
            tiledb_query_set_buffers(self.ctx.ptr, query, &c_attr_name, 1, &buff, &buff_size))

        cdef int rc = tiledb_query_submit(self.ctx.ptr, query)
        tiledb_query_free(self.ctx.ptr, query)

        if rc != TILEDB_OK:
            check_error(self.ctx, rc)
        return

    def read_direct(self, unicode attribute_name not None):
        # array name
        cdef bytes barray_name = self.name.encode('UTF-8')

        # attr name
        cdef bytes battribute_name = attribute_name.encode('UTF-8')
        cdef const char* c_attribute_name = battribute_name

        cdef tuple domain_shape = self.domain.shape
        cdef Attr attr = self.attr(attribute_name)
        cdef np.dtype attr_dtype = attr.dtype

        out = np.empty(domain_shape, dtype=attr_dtype)

        cdef void* buff = np.PyArray_DATA(out)
        cdef uint64_t buff_size = out.nbytes

        cdef tiledb_query_t* query = NULL
        check_error(self.ctx,
            tiledb_query_create(self.ctx.ptr, &query, barray_name, TILEDB_READ))

        check_error(self.ctx,
            tiledb_query_set_layout(self.ctx.ptr, query, TILEDB_ROW_MAJOR))
        check_error(self.ctx,
            tiledb_query_set_buffers(self.ctx.ptr, query, &c_attribute_name, 1, &buff, &buff_size))

        cdef int rc = tiledb_query_submit(self.ctx.ptr, query)
        tiledb_query_free(self.ctx.ptr, query)
        if rc != TILEDB_OK:
            check_error(self.ctx, rc)
        return out

cdef bytes unicode_path(path):
    return ustring(abspath(path)).encode('UTF-8')


def array_consolidate(Ctx ctx, path):
    upath = unicode_path(path)
    cdef const char* c_path = upath
    check_error(ctx,
        tiledb_array_consolidate(ctx.ptr, c_path))
    return upath


def group_create(Ctx ctx, path):
    upath = unicode_path(path)
    cdef const char* c_path = upath
    check_error(ctx,
       tiledb_group_create(ctx.ptr, c_path))
    return upath


def object_type(Ctx ctx, path):
    upath = unicode_path(path)
    cdef const char* c_path = upath
    cdef tiledb_object_t obj = TILEDB_INVALID
    check_error(ctx,
       tiledb_object_type(ctx.ptr, c_path, &obj))
    return obj


def delete(Ctx ctx, path):
    upath = unicode_path(path)
    cdef const char* c_path = upath
    check_error(ctx,
       tiledb_delete(ctx.ptr, c_path))
    return


def move(Ctx ctx, oldpath, newpath, force=False):
    uoldpath = unicode_path(oldpath)
    unewpath = unicode_path(newpath)
    cdef const char* c_oldpath = uoldpath
    cdef const char* c_newpath = unewpath
    cdef int c_force = 0
    if force:
       c_force = True
    check_error(ctx,
        tiledb_move(ctx.ptr, c_oldpath, c_newpath, c_force))
    return


cdef int walk_callback(const char* c_path,
                       tiledb_object_t obj,
                       void* pyfunc):
    objtype = None
    if obj == TILEDB_ARRAY:
        objtype = "array"
    elif obj == TILEDB_GROUP:
        objtype = "group"
    try:
        (<object> pyfunc)(c_path.decode('UTF-8'), objtype)
    except StopIteration:
        return 0
    return 1


def walk(Ctx ctx, path, func, order="preorder"):
    upath = unicode_path(path)
    cdef const char* c_path = upath
    cdef tiledb_walk_order_t c_order
    if order == "postorder":
        c_order = TILEDB_POSTORDER
    elif order == "preorder":
        c_order = TILEDB_PREORDER
    else:
        raise AttributeError("unknown walk order {}".format(order))
    check_error(ctx,
        tiledb_walk(ctx.ptr, c_path, c_order, walk_callback, <void*> func))
    return
