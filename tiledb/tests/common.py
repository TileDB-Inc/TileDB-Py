import contextlib
import datetime
import glob
import importlib
import os
import random
import shutil
import subprocess
import sys
import tempfile
import traceback
import urllib
import uuid

import numpy as np
import pytest
from numpy.testing import assert_almost_equal, assert_array_equal, assert_equal

import tiledb

SUPPORTED_DATETIME64_DTYPES = tuple(
    np.dtype(f"M8[{res}]") for res in "Y M W D h m s ms us ns".split()
)


def has_pandas():
    return importlib.util.find_spec("pandas") is not None


def has_pyarrow():
    return importlib.util.find_spec("pyarrow") is not None


def assert_tail_equal(a, *rest, **kwargs):
    """Assert that all arrays in target equal first array"""
    for target in rest:
        assert_array_equal(a, target, **kwargs)


def create_vfs_dir(path):
    """Create a directory at the given scheme-prefixed path,
    first creating the base bucket if needed."""
    split_uri = urllib.parse.urlsplit(path)
    scheme = split_uri.scheme
    bucket = split_uri.netloc
    bucket_uri = scheme + "://" + bucket

    vfs = tiledb.VFS()
    if not vfs.is_bucket(bucket_uri):
        vfs.create_bucket(bucket_uri)
    vfs.create_dir(path)


class DiskTestCase:
    """Helper class to store paths and associated allocation frames. This is both
    a cleanup step and a test of resource management. Some platforms will
    refuse to delete an open file, indicating a potentially leaked resource.
    """

    @classmethod
    def setup_method(self):
        # .lower: because bucket name must be all lowercase
        prefix = "tiledb-" + self.__name__.lower()
        if hasattr(pytest, "tiledb_vfs") and pytest.tiledb_vfs == "s3":
            self.path_scheme = pytest.tiledb_vfs + "://"
            self.rootdir = self.path_scheme + prefix + str(random.randint(0, 10e10))
            create_vfs_dir(self.rootdir)
        else:
            self.path_scheme = ""
            self.rootdir = tempfile.mkdtemp(prefix=prefix)

        self.vfs = tiledb.VFS()
        self.pathmap = dict()

    @classmethod
    def teardown_method(self):
        # Remove every directory starting with rootdir
        # This is both a clean-up step and an implicit test
        # of proper resource deallocation (see notes below)
        for dirpath in glob.glob(self.rootdir + "*"):
            try:
                shutil.rmtree(dirpath)
            except OSError as exc:
                print(
                    "test '{}' error deleting '{}'".format(
                        self.__class__.__name__, dirpath
                    )
                )
                print("registered paths and originating functions:")
                for path, frame in self.pathmap.items():
                    print("  '{}' <- '{}'".format(path, frame))
                raise exc

    def path(self, basename=None, shared=False):
        if self.path_scheme:
            basename = basename if basename else str(uuid.uuid4())
            out = os.path.join(self.rootdir, basename)
            self.vfs.create_dir(out)
        else:
            if basename is not None:
                # Note: this must be `is not None` because we need to match empty string
                out = os.path.abspath(os.path.join(self.rootdir, basename))
            else:
                out = tempfile.mkdtemp(dir=self.rootdir)

        if os.name == "nt" and shared:
            subprocess.run(
                f'cmd //c "net share tiledb-shared={out}"', shell=True, check=True
            )

        # We have had issues in both py and libtiledb in the past
        # where files were not released (eg: destructor not called)
        # Often this is invisible on POSIX platforms, but will
        # cause errors on Windows because two processes cannot access
        # the same file at once.
        # In order to debug this issue, we save the caller where
        # this path was allocated so that we can determine what
        # test created an unreleased file
        frame = traceback.extract_stack(limit=2)[-2][2]
        self.pathmap[out] = frame

        return out

    def assertRaises(self, *args):
        return pytest.raises(*args)

    def assertRaisesRegex(self, e, m):
        return pytest.raises(e, match=m)

    @contextlib.contextmanager
    def assertEqual(self, *args):
        if not len(args) == 2:
            raise Exception("Unexpected input len > 2 to assertEquals")
        assert args[0] == args[1]

    @contextlib.contextmanager
    def assertNotEqual(self, *args):
        if not len(args) == 2:
            raise Exception("Unexpected input len > 2 to assertEquals")
        assert args[0] != args[1]

    @contextlib.contextmanager
    def assertTrue(self, a, msg=None):
        if msg:
            assert a, msg
        else:
            assert a

    @contextlib.contextmanager
    def assertFalse(self, a):
        assert a is False

    @contextlib.contextmanager
    def assertIsInstance(self, v, t):
        assert isinstance(v, t)

    @contextlib.contextmanager
    def assertSetEqual(self, s1, s2):
        assert all(isinstance(x, set) for x in (s1, s2))
        assert s1 == s2

    @contextlib.contextmanager
    def assertIsNone(self, a1):
        assert a1 is None

    @contextlib.contextmanager
    def assertTupleEqual(self, a1, a2):
        assert a1 == a2

    @contextlib.contextmanager
    def assertAlmostEqual(self, a1, a2):
        assert_almost_equal(a1, a2)


# exclude whitespace: if we generate unquoted newline then pandas will be confused
_ws_set = set("\n\t\r")


def gen_chr(max, printable=False):
    while True:
        # TODO we exclude 0x0 here because the key API does not embedded NULL
        s = chr(random.randrange(1, max))
        if printable and (not s.isprintable()) or (s in _ws_set):
            continue
        if len(s) > 0:
            break
    return s


def rand_utf8(size=5, printable=False):
    # This is a hack to ensure that all UTF-8 is parseable. It would be nicer to
    # exclude invalid surrogate pairs, but this will do.
    while True:
        try:
            v = "".join([gen_chr(0xD007F, printable=printable) for _ in range(0, size)])
            return v.encode("UTF-8").decode()
        except UnicodeError:
            continue


def rand_ascii(size=5, printable=False):
    return "".join([gen_chr(127, printable) for _ in range(0, size)])


def rand_ascii_bytes(size=5, printable=False):
    return b"".join([gen_chr(127, printable).encode("utf-8") for _ in range(0, size)])


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

    raise f"Unknown dtype for dtype_max '{dtype}'"


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

    raise f"Unknown dtype for dtype_min '{dtype}'"


def rand_datetime64_array(
    size, start=None, stop=None, include_extremes=True, dtype=None
):
    if not dtype:
        dtype = np.dtype("M8[ns]")

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
        start.astype(dtype).astype(np.int64),
        stop.astype(dtype).astype(np.int64),
        size=size,
        dtype=np.int64,
    )

    arr.sort()
    arr = arr.astype(dtype)

    # enable after fix for core issue: ch 7192
    if include_extremes:
        arr[0] = start
        arr[-1] = stop

    return arr


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
    step = (stop - start) // num
    nextval = start

    if np.issubdtype(dtype, np.integer) and step < 1:
        raise ValueError(
            "Cannot use non-integral step value '{}' for integer dtype!".format(step)
        )

    for i in range(num):
        rval[i] = nextval
        nextval += step

    rval[-1] = stop
    return rval


def assert_unordered_equal(a1, a2, unordered=True):
    """Assert that arrays are equal after sorting if
    `unordered==True`"""
    if unordered:
        a1 = np.sort(a1)
        a2 = np.sort(a2)
    assert_array_equal(a1, a2)


def assert_subarrays_equal(a, b, ordered=True):
    assert_equal(a.shape, b.shape)

    if not ordered:
        a = np.sort(a)
        b = np.sort(b)

    for a_el, b_el in zip(a.flat, b.flat):
        assert_array_equal(a_el, b_el)


def assert_dict_arrays_equal(d1, d2, ordered=True):
    assert d1.keys() == d2.keys(), "Keys not equal"

    if ordered:
        for k in d1.keys():
            assert_array_equal(d1[k], d2[k])
    else:
        d1_dtypes = [tuple((name, value.dtype)) for name, value in d1.items()]
        d1_records = [tuple(values) for values in zip(*d1.values())]
        array1 = np.array(d1_records, dtype=d1_dtypes)

        d2_dtypes = [tuple((name, value.dtype)) for name, value in d2.items()]
        d2_records = [tuple(values) for values in zip(*d2.values())]
        array2 = np.array(d2_records, dtype=d2_dtypes)

        assert_unordered_equal(array1, array2, True)


def assert_captured(cap, expected):
    if sys.platform != "win32":
        import ctypes

        libc = ctypes.CDLL(None)
        libc.fflush(None)

        out, err = cap.readouterr()
        assert not err
        assert expected in out


@pytest.fixture(scope="module", params=["hilbert", "row-major"])
def fx_sparse_cell_order(request):
    yield request.param
