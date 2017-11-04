from os.path import abspath
from cpython.version cimport PY_MAJOR_VERSION

ctypedef unsigned char char_type


cdef unicode ustring(s):
    if type(s) is unicode:
        return <unicode>s
    elif PY_MAJOR_VERSION < 2 and isinstance(s, bytes):
        return (<bytes> s).decode('ascii')
    elif isinstance(s, unicode):
        return unicode(s)
    raise TypeError(
        "ustring() must be a string or a bytes-like object"
        ", not {}".format(s))


class Group(object):

    def __init__(self, path=None):
        self._path = path

    @property
    def path(self):
        self._path

    @property
    def name(self):
        if self._path:
            name = self._path
            if name[0] != '/':
                name = '/' + name
            return name
        return '/'

    def __eq__(self, other):
        return isinstance(other, Group) and self._path == other.path

    def __repr__(self):
        return "<tiledb.Group '%s'>" % self.name


class TileDBError(Exception):
    pass


cdef check_error(tiledb_ctx_t* ctx, int rc):
    if rc == TILEDB_OK:
        return
    if rc == TILEDB_OOM:
        raise MemoryError()
    cdef int ret = TILEDB_OK
    cdef tiledb_error_t* err = NULL
    ret = tiledb_error_last(ctx, &err)
    if ret != TILEDB_OK:
        tiledb_error_free(ctx, err)
        if ret == TILEDB_OOM:
            raise MemoryError()
        raise TileDBError("error retrieving error from ctx")
    cdef const char* err_msg = NULL
    ret = tiledb_error_message(ctx, err, &err_msg)
    if ret != TILEDB_OK:
        tiledb_error_free(ctx, err)
        if ret == TILEDB_OOM:
            return MemoryError()
        raise TileDBError("error retrieving error message from ctx")
    message_string = err_msg.decode('UTF-8', 'strict')
    tiledb_error_free(ctx, err)
    raise TileDBError(message_string)


cdef class Ctx(object):

    cdef tiledb_ctx_t* ptr

    def __cinit__(self):
        cdef int rc = tiledb_ctx_create(&self.ptr)
        if rc == TILEDB_OOM:
            raise MemoryError()
        if rc == TILEDB_ERR:
            raise Exception("better error here")

    def __dealloc__(self):
        if self.ptr is not NULL:
            tiledb_ctx_free(self.ptr)

    def __repr__(self):
        return "<tiledb.Ctx>"

    def group_create(self, path=None, force=False):
        if path is None:
            raise AttributeError("invalid path, path is None")
        upath = ustring(abspath(path)).encode('UTF-8')
        cdef const char* c_path = upath
        check_error(self.ptr,
                    tiledb_group_create(self.ptr, c_path))
        return Group(upath.decode('UTF-8'))

def ctx():
    return Ctx()


def version():
    cdef:
        int major = 0
        int minor = 0
        int rev = 0
    tiledb_version(&major, &minor, &rev)
    return major, minor, rev


def object_type():
    cdef:
        int rc = 0
        tiledb_ctx_t* ctx = NULL
        tiledb_object_t obj = TILEDB_INVALID
    rc = tiledb_ctx_create(&ctx)
    if rc != TILEDB_OK:
        tiledb_ctx_free(ctx)
        return Exception("could not create context")
    rc = tiledb_object_type(ctx, "/", &obj)
    if rc != TILEDB_OK:
        tiledb_ctx_free(ctx)
        return Exception("error retrieving object type")
    objtype = None
    if obj == TILEDB_GROUP:
        objtype =  "group"
    elif obj == TILEDB_ARRAY:
        objtype = "array"
    tiledb_ctx_free(ctx)
    if obj != TILEDB_INVALID:
        raise Exception("error retrieving object type")
    return objtype

