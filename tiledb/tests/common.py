from __future__ import absolute_import

import glob
import os
import sys
import random
import shutil
import tempfile
import datetime
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

# exclude whitespace: if we generate unquoted newline then pandas will be confused
_ws_set = set("\n\t\r")
def gen_chr(max, printable=False):
    while True:
        # TODO we exclude 0x0 here because the key API does not embedded NULL
        s = getchr(random.randrange(1, max))
        if printable and (not s.isprintable()) or (s in _ws_set):
            continue
        if len(s) > 0:
            break
    return s

def rand_utf8(size=5):
    return u''.join([gen_chr(0xD7FF) for _ in range(0, size)])

def rand_ascii(size=5, printable=False):
    return u''.join([gen_chr(127, printable) for _ in range(0,size)])

def rand_ascii_bytes(size=5, printable=False):
    return b''.join([gen_chr(127, printable).encode('utf-8') for _ in range(0,size)])

def dtype_max(dtype):
    if not np.issubdtype(dtype, np.generic):
        raise TypeError("expected numpy dtype!")

    if np.issubdtype(dtype, np.floating):
        finfo = np.finfo(dtype)
        return finfo.max

    elif np.issubdtype(dtype, np.integer):
        iinfo = np.iinfo(dtype)
        return int(iinfo.max)

    elif np.issubdtype(dtype, np.datetime64):
        return np.datetime64(datetime.datetime.max)

    raise "Unknown dtype for dtype_max '{}'".format(str(dtype))

def dtype_min(dtype):
    if not np.issubdtype(dtype, np.generic):
        raise TypeError("expected numpy dtype!")

    if np.issubdtype(dtype, np.floating):
        finfo = np.finfo(dtype)
        return finfo.min

    elif np.issubdtype(dtype, np.integer):
        iinfo = np.iinfo(dtype)
        return int(iinfo.min)

    elif np.issubdtype(dtype, np.datetime64):
        return np.datetime64(datetime.datetime.min)

    raise "Unknown dtype for dtype_min '{dtype}'".format(str(dtype))

def rand_int_sequential(size, dtype=np.uint64):
    arr = np.random.randint(
        dtype_min(dtype), high=dtype_max(dtype), size=size, dtype=dtype
    )
    return np.sort(arr)

def rand_datetime64_array(size, start=None, stop=None, dtype=None):
    if not dtype:
        dtype = np.dtype('M8[ns]')

    # generate randint inbounds on the range of the dtype
    units = np.datetime_data(dtype)[0]
    intmin, intmax = np.iinfo(np.int64).min, np.iinfo(np.int64).max

    if start is None:
        start = np.datetime64(intmin + 1, units)
    else:
        start = np.datetime64(start)
    if stop is None:
        stop = np.datetime64(intmax, units)
    else:
        stop = np.datetime64(stop)

    arr = np.random.randint(
        start.astype(dtype).astype(np.int64), stop.astype(dtype).astype(np.int64),
        size=size, dtype=np.int64
    )
    arr.sort()

    return arr.astype(dtype)

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
    step = (stop-start) // num
    nextval = start

    if np.issubdtype(dtype, np.integer) and step < 1:
      raise ValueError("Cannot use non-integral step value '{}' for integer dtype!".format(
                      step))

    for i in range(num):
        rval[i] = nextval
        nextval += step

    rval[-1] = stop
    return rval

import pprint as _pprint
pp = _pprint.PrettyPrinter(indent=4)
def xprint(*x):
    for xp in x:
        pp.pprint(xp)