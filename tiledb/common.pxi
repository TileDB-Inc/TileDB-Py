from cpython.bytes cimport (PyBytes_GET_SIZE,
                            PyBytes_AS_STRING,
                            PyBytes_Size,
                            PyBytes_FromString,
                            PyBytes_FromStringAndSize)
from cpython.float cimport PyFloat_FromDouble
from cpython.long cimport PyLong_FromLong

from cpython.ref cimport (Py_INCREF, Py_DECREF, PyTypeObject)

from libc.stdio cimport (FILE, stdout)
from libc.stdio cimport stdout
from libc.stdlib cimport malloc, calloc, free
from libc.string cimport memcpy
from libc.stdint cimport (uint8_t, int8_t,
                          uint16_t, int16_t,
                          uint32_t, int32_t,
                          uint64_t, int64_t,
                          uintptr_t)
from libc.stddef cimport ptrdiff_t

from libc cimport limits
from libcpp.vector cimport vector

cdef extern from "Python.h":
    object PyUnicode_FromStringAndSize(const char *u, Py_ssize_t size)
    object PyUnicode_FromString(const char *u)

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
    void PyArray_ENABLEFLAGS(np.ndarray arr, int flags)
    void* PyDataMem_NEW(size_t nbytes)
    void* PyDataMem_RENEW(void* data, size_t nbytes)
    void PyDataMem_FREE(void* data)
