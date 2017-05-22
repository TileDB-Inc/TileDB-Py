import ctypes

def _version():
    libtiledb = ctypes.cdll.LoadLibrary("libtiledb.dylib")
    major, minor, rev = ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0)
    libtiledb.tiledb_version(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(rev))
    return (major.value, minor.value, rev.value)

__version__ = "{:d}.{:d}.{:d}".format(*_version())
