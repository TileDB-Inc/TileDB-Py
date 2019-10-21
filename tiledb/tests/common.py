from __future__ import absolute_import

import glob
import os
import sys
import random
import shutil
import tempfile
import traceback
from unittest import TestCase

import numpy as np
from numpy.testing import assert_equal, assert_array_equal


class DiskTestCase(TestCase):
    pathmap = dict()

    def setUp(self):
        prefix = 'tiledb-' + self.__class__.__name__
        self.rootdir = tempfile.mkdtemp(prefix=prefix)

    def tearDown(self):
        # Remove every directory starting with rootdir
        for dirpath in glob.glob(self.rootdir + "*"):
            try:
                shutil.rmtree(dirpath)
            except OSError as exc:
                print("test '{}' error deleting '{}'".format(self.__class__.__name__,
                                                             dirpath))
                print("registered paths and originating functions:")
                for path,frame in self.pathmap.items():
                    print("  '{}' <- '{}'".format(path,frame))
                raise exc

    def path(self, path):
        out = os.path.abspath(os.path.join(self.rootdir, path))
        frame = traceback.extract_stack(limit=2)[-2][2]
        self.pathmap[out] = frame
        return out


def assert_subarrays_equal(a, b):
    assert_equal(a.shape, b.shape)

    for a_el, b_el in zip(a.flat, b.flat):
        assert_array_equal(a_el, b_el)

def assert_all_arrays_equal(*arrays):
    # TODO this should display raise in the calling location if possible
    assert len(arrays) % 2 == 0, \
           "Expected even number of arrays"

    for a1,a2 in zip(arrays[0::2], arrays[1::2]):
        assert_array_equal(a1, a2)

# python 2 vs 3 compatibility
if sys.hexversion >= 0x3000000:
    getchr = chr
else:
    getchr = unichr

def gen_chr(max):
    while True:
        s = getchr(random.randrange(max))
        if len(s) > 0: break
    return s

def rand_utf8(size=5):
    return u''.join([gen_chr(0xD7FF) for _ in range(0, size)])

def rand_ascii(size=5):
    return u''.join([gen_chr(127) for _ in range(0,size)])

def rand_ascii_bytes(size=5):
    return b''.join([gen_chr(127).encode('utf-8') for _ in range(0,size)])

def dtype_max(dtype):
    if not np.issubdtype(dtype, np.generic):
        raise TypeError("expected numpy dtype!")
    iinfo = np.iinfo(dtype)
    return iinfo.max

def dtype_min(dtype):
    if not np.issubdtype(dtype, np.generic):
        raise TypeError("expected numpy dtype!")
    iinfo = np.iinfo(dtype)
    return iinfo.min

def rand_int_sequential(size, dtype=np.uint64):
    arr = np.random.randint(dtype_max(dtype), size=size, dtype=dtype)
    return np.sort(arr)

def intspace(start, stop, num=50, dtype=np.int64):
    """
    Return evenly spaced values over range ensuring that stop is
    always the maximum (will not overflow with int dtype as linspace)
    :param start:
    :param stop:
    :param num:
    :param dtype:
    :return:
    """
    rval = np.zeros(num, dtype=dtype)
    step = (stop-start)//num
    nextval = start

    for i in range(num):
        rval[i] = nextval
        nextval += step

    rval[-1] = stop
    return rval
