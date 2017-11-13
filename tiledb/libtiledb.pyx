from cpython.version cimport PY_MAJOR_VERSION
from libc.stdio cimport stdout
from libc.stdint cimport uintptr_t, int64_t
from libc.stdlib cimport malloc, free

cimport numpy as np

import numpy as np
from os.path import abspath

def version():
    cdef:
        int major = 0
        int minor = 0
        int rev = 0
    tiledb_version(&major, &minor, &rev)
    return major, minor, rev

cdef unicode ustring(s):
    if type(s) is unicode:
        return <unicode>s
    elif PY_MAJOR_VERSION < 3 and isinstance(s, bytes):
        return (<bytes>s).decode('ascii')
    elif isinstance(s, unicode):
        return unicode(s)
    raise TypeError(
        "ustring() must be a string or a bytes-like object"
        ", not {0!r}".format(type(s)))


class TileDBError(Exception):
    pass


cdef check_error(Ctx ctx, int rc):
    ctx_ptr = ctx.ptr
    if rc == TILEDB_OK:
        return
    if rc == TILEDB_OOM:
        raise MemoryError()
    cdef int ret = TILEDB_OK
    cdef tiledb_error_t* err = NULL
    ret = tiledb_error_last(ctx_ptr, &err)
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


cdef tiledb_datatype_t _tiledb_dtype(dtype) except TILEDB_CHAR:
    if dtype == "i4":
        return TILEDB_INT32
    elif dtype == "u4":
        return TILEDB_UINT32
    elif dtype == "i8":
        return TILEDB_INT64
    elif dtype == "u8":
        return TILEDB_UINT64
    elif dtype == "f4":
        return TILEDB_FLOAT32
    elif dtype == "f8":
        return TILEDB_FLOAT64
    elif dtype == "i1":
        return TILEDB_INT8
    elif dtype == "u1":
        return TILEDB_UINT8
    elif dtype == "i2":
        return TILEDB_INT16
    elif dtype == "u2":
        return TILEDB_UINT16
    raise TypeError("data type {0!r} not understood".format(dtype))

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

    @property
    def name(self):
        cdef const char* c_name = NULL
        check_error(self.ctx,
            tiledb_attribute_get_name(self.ctx.ptr, self.ptr, &c_name))
        return c_name.decode('UTF-8')

    @property
    def compressor(self):
        cdef int c_level = -1
        cdef tiledb_compressor_t compr = TILEDB_NO_COMPRESSION
        check_error(self.ctx,
            tiledb_attribute_get_compressor(self.ctx.ptr, self.ptr, &compr, &c_level))
        return (_tiledb_compressor_string(compr), int(c_level))

cdef class Dim(object):

    cdef unicode label
    cdef tuple dim
    cdef object tile

    def __init__(self, label=None, dim=None, tile=None):
        self.label = label
        if len(dim) != 2:
            raise AttributeError("invalid extent")
        self.dim = (dim[0], dim[1])
        self.tile = tile

    @property
    def label(self):
        return self.label

    @property
    def dim(self):
        return self.dim

    @property
    def tile(self):
        return self.tile

cdef class Domain(object):

    cdef Ctx ctx
    cdef tiledb_domain_t* ptr
    cdef object dims

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

    def __init__(self, Ctx ctx, *dims, dtype='i8'):
        for d in dims:
            if not isinstance(d, Dim):
                raise TypeError("unknown dimension type {0!r}".format(d))
        cdef tiledb_datatype_t domain_type = _tiledb_dtype(dtype)
        cdef tiledb_domain_t* domain_ptr = NULL
        check_error(ctx,
                    tiledb_domain_create(ctx.ptr, &domain_ptr, domain_type))
        cdef int rc
        cdef uint64_t tile_extent
        cdef uint64_t[2] dim_range
        for d in dims:
            ulabel = ustring(d.label).encode('UTF-8')
            dim_range[0] = d.dim[0]
            dim_range[1] = d.dim[1]
            tile_extent = d.tile
            rc = tiledb_domain_add_dimension(
                ctx.ptr, domain_ptr, ulabel, &dim_range, &tile_extent)
            if rc != TILEDB_OK:
                tiledb_domain_free(ctx.ptr, domain_ptr)
                check_error(ctx, rc)
        self.dims = dims
        self.ctx = ctx
        self.ptr = domain_ptr

    @property
    def dims(self):
        return self.dims

    @property
    def ndim(self):
        return len(self.dims)

    @property
    def dtype(self):
        cdef tiledb_datatype_t typ
        pass

    def dim(self, unicode idx):
        for dim in self.dims:
            if dim.label == idx:
                return dim
        raise TileDBError("unknown dimension: {0!r}".format(idx))

    def dump(self):
        check_error(self.ctx,
                    tiledb_domain_dump(self.ctx.ptr, self.ptr, stdout))


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
    cdef from_ptr(Ctx ctx, unicode name, const tiledb_array_metadata_t* ptr):
        cdef Array arr = Array.__new__(Array)
        arr.ctx = ctx
        arr.name = name
        arr.ptr = <tiledb_array_metadata_t*> ptr
        return arr

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
            attr_ptr = (<Attr>attr).ptr
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

    def attr(self, unicode idx):
        cdef:
            Attr attr
            tiledb_ctx_t* ctx_ptr = self.ctx.ptr
            tiledb_attribute_iter_t* it_ptr = NULL
            const tiledb_attribute_t* attr_ptr = NULL
            const char* attr_name = NULL

        check_error(self.ctx,
            tiledb_attribute_iter_create(ctx_ptr, self.ptr, &it_ptr))

        cdef int rc = TILEDB_OK
        cdef int done = 1

        while True:
            rc = tiledb_attribute_iter_done(ctx_ptr, it_ptr, &done)
            if rc != TILEDB_OK:
                tiledb_attribute_iter_free(ctx_ptr, it_ptr)
                check_error(self.ctx, rc)
            if done:
                break

            rc = tiledb_attribute_iter_here(ctx_ptr, it_ptr, &attr_ptr)
            if rc != TILEDB_OK:
                tiledb_attribute_iter_free(ctx_ptr, it_ptr)
                check_error(self.ctx, rc)

            rc = tiledb_attribute_get_name(ctx_ptr, attr_ptr, &attr_name)
            if rc != TILEDB_OK:
                tiledb_attribute_iter_free(ctx_ptr, it_ptr)
                check_error(self.ctx, rc)

            if attr_name.decode('UTF-8') == idx:
                tiledb_attribute_iter_free(ctx_ptr, it_ptr)
                return Attr.from_ptr(self.ctx, attr_ptr)

            rc = tiledb_attribute_iter_next(ctx_ptr, it_ptr)
            if rc != TILEDB_OK:
                tiledb_attribute_iter_free(ctx_ptr, it_ptr)
                check_error(self.ctx, rc)

        tiledb_attribute_iter_free(ctx_ptr, it_ptr)
        raise TileDBError("unknown array attribute: {0!r}".format(idx))

    def dump(self):
        check_error(self.ctx,
            tiledb_array_metadata_dump(self.ctx.ptr, self.ptr, stdout))

    def write_direct(self, unicode attr, np.ndarray array):
        # array name
        cdef bytes array_name = self.name.encode('UTF-8')
        cdef const char* c_aname = array_name
        # attr name
        cdef bytes attr_name = attr.encode('UTF-8')
        cdef const char* c_attr = attr_name

        cdef void* buff = <void*>(np.PyArray_DATA(array))
        cdef uint64_t buffsize = <uint64_t>(array.nbytes)

        cdef tiledb_query_t* query
        cdef int rc
        rc = tiledb_query_create(self.ctx.ptr, &query,
                c_aname, TILEDB_WRITE, TILEDB_ROW_MAJOR,
                NULL, &c_attr, 1, &buff, &buffsize)
        if rc != TILEDB_OK:
            tiledb_query_free(self.ctx.ptr, query)
            check_error(self.ctx, rc)
        rc = tiledb_query_submit(self.ctx.ptr, query)
        tiledb_query_free(self.ctx.ptr, query)
        if rc != TILEDB_OK:
            check_error(self.ctx, rc)
        return

    def read_direct(self, unicode attr):
        # array name
        cdef bytes b_aname = self.name.encode('UTF-8')
        cdef const char* c_aname = b_aname
        # attr name
        cdef bytes b_attr = attr.encode('UTF-8')
        cdef const char* c_attr = b_attr

        out = np.empty((3,), dtype='int64')
        cdef void* buff = <void*>(np.PyArray_DATA(out))
        cdef uint64_t buff_size = <uint64_t>(out.nbytes)

        cdef tiledb_query_t* query
        rc = tiledb_query_create(self.ctx.ptr, &query,
                c_aname, TILEDB_READ, TILEDB_ROW_MAJOR,
                NULL, &c_attr, 1, &buff, &buff_size)
        if rc != TILEDB_OK:
            tiledb_query_free(self.ctx.ptr, query)
            check_error(self.ctx, rc)
        rc = tiledb_query_submit(self.ctx.ptr, query)
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
