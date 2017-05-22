import ctypes

def _version():
    libtiledb = ctypes.cdll.LoadLibrary("libtiledb.dylib")
    major, minor, rev = ctypes.c_int(0), ctypes.c_int(0), ctypes.c_int(0)
    libtiledb.tiledb_version(ctypes.byref(major), ctypes.byref(minor), ctypes.byref(rev))
    return (int(major), int(minor), int(rev))

__version__ = "{:d}.{:d}.{:d}".format(*_version())
