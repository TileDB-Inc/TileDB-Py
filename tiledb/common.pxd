from cpython.version cimport PY_MAJOR_VERSION
from cpython.bytes cimport (PyBytes_GET_SIZE,
                            PyBytes_AS_STRING,
                            PyBytes_Size,
                            PyBytes_FromString,
                            PyBytes_FromStringAndSize)
from cpython.ref cimport (Py_INCREF, Py_DECREF, PyTypeObject)
from libc.stdio cimport (FILE, stdout)
from libc.stdio cimport stdout
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memcpy
from libc.stdint cimport (uint8_t, uint64_t, int64_t, uintptr_t)
from libc cimport limits
from libcpp.vector cimport vector

cimport numpy as np
