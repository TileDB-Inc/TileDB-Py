from cpython.version cimport PY_MAJOR_VERSION
from libc.stdio cimport stdout

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
        return (<bytes> s).decode('ascii')
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


cdef tiledb_datatype_t dtype_to_tiledb(dtype):
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
    raise TypeError("data type '{0!r}' not understood".format(dtype))


cdef class Attr(object):

    cdef Ctx ctx
    cdef tiledb_attribute_t* ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr is not NULL:
            tiledb_attribute_free(self.ctx.ptr, self.ptr)

    # TODO: use numpy compund dtypes to choose number of cells
    def __init__(self, Ctx ctx,  name=None, dtype='f8', compressor=None, level=-1):
        uname = ustring(name).encode('UTF-8')
        cdef tiledb_attribute_t* attr_ptr
        cdef tiledb_datatype_t tiledb_dtype = dtype_to_tiledb(dtype)
        check_error(ctx,
            tiledb_attribute_create(ctx.ptr, &attr_ptr, uname, tiledb_dtype))
        if compressor is not None:
            pass
        if level > -1:
            check_error(ctx,
                tiledb_attribute())
        self.ctx = ctx
        self.ptr = attr_ptr


    def dump(self):
        check_error(self.ctx,
            tiledb_attribute_dump(self.ctx.ptr, self.ptr, stdout))


cdef class Domain(object):

    cdef Ctx ctx
    cdef tiledb_domain_t* ptr

    def __cinit__(self):
        self.ptr = NULL

    def __dealloc__(self):
        if self.ptr is not NULL:
            tiledb_domain_free(self.ctx.ptr, self.ptr)

    def __init__(self, Ctx ctx, *dims, dtype='i8'):
        for d in dims:
            if not isinstance(d, Dim):
                raise TypeError("unknown dimension type {0!r}".format(d))
        cdef tiledb_datatype_t domain_type = dtype_to_tiledb(dtype)
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
        self.ctx = ctx
        self.ptr = domain_ptr

    def dump(self):
        check_error(self.ctx,
                    tiledb_domain_dump(self.ctx.ptr, self.ptr, stdout))



cdef class Dim(object):

    cdef unicode label
    cdef tuple dim
    cdef object tile

    def __init__(self, label=None, dim=None, tile=None):
        self.label = label
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


cdef unicode unicode_path(path):
    return ustring(abspath(path)).encode('UTF-8')

def group_create(Ctx ctx, path):
    upath = unicode_path(path)
    cdef const char* c_path = upath
    check_error(ctx,
       tiledb_group_create(ctx.ptr, c_path))
    return upath.decode('UTF-8')

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
