from __future__ import absolute_import

try:
    import pandas as pd
    import pandas._testing as tm

    import_failed = False
except ImportError:
    import_failed = True

import unittest, os
import string, random
import numpy as np
from numpy.testing import assert_array_equal

import tiledb
from tiledb.tests.common import *

def make_dataframe_basic1(col_size=10):
    data_dict = {
        'time': rand_int_sequential(col_size, dtype=np.uint64),
        'x': np.array([rand_ascii(4) for _ in range(col_size)]),
        'chars': np.array([rand_ascii_bytes(2) for _ in range(col_size)]),
        'cccc': np.arange(0, col_size),

        'q': np.array([rand_utf8(np.random.randint(1, 100)) for _ in range(col_size)]),
        'r': np.array([rand_ascii_bytes(np.random.randint(1, 100)) for _ in range(col_size)]),
        's': np.array([rand_ascii() for _ in range(col_size)]),
        'vals_int64': np.random.randint(dtype_max(np.int64), size=col_size, dtype=np.int64),
        'vals_float64': np.random.rand(col_size),
        't': np.array([rand_utf8(4) for _ in range(col_size)]),
        'u': np.array([rand_ascii_bytes() for _ in range(col_size)]),
        'v': np.array([rand_ascii_bytes() for _ in range(col_size)]),
    }

    # TODO: dump this dataframe to pickle/base64 so that it can be reconstructed if
    #       there are weird failures on CI?

    df = pd.DataFrame.from_dict(data_dict)
    return df

def make_dataframe_basic2():
    # This code is from Pandas feather i/o tests "test_basic" function:
    #   https://github.com/pandas-dev/pandas/blob/master/pandas/tests/io/test_feather.py
    # (available under BSD 3-clause license
    #   https://github.com/pandas-dev/pandas/blob/master/LICENSE

    import pandas as pd

    df = pd.DataFrame(
        {
            "string": list("abc"),
            "int": list(range(1, 4)),
            "uint": np.arange(3, 6).astype("u1"),
            "float": np.arange(4.0, 7.0, dtype="float64"),
            # TODO "float_with_null": [1.0, np.nan, 3],
            "bool": [True, False, True],
            # TODO "bool_with_null": [True, np.nan, False],
            #"cat": pd.Categorical(list("abc")),
            "dt": pd.date_range("20130101", periods=3),
            #"dttz": pd.date_range("20130101", periods=3, tz="US/Eastern"),
            #"dt_with_null": [
            #    pd.Timestamp("20130101"),
            #    pd.NaT,
            #    pd.Timestamp("20130103"),
            #],
            "dtns": pd.date_range("20130101", periods=3, freq="ns"),
        }
    )

    return df


class PandasDataFrameRoundtrip(DiskTestCase):
    def setUp(self):
        if import_failed:
            self.skipTest("Pandas not available")
        else:
            super(PandasDataFrameRoundtrip, self).setUp()

    def test_dataframe_basic_rt1_manual(self):

        uri = self.path("dataframe_basic_rt1_manual")

        ctx = tiledb.Ctx()
        dom = tiledb.Domain(tiledb.Dim(name="i_chars",
                                       domain=(0, 10000),
                                       tile=10,
                                       dtype=np.uint64),
                            tiledb.Dim(name="datetime",
                                       domain=(0, np.iinfo(np.uint64).max - 3600 * 1000000000),
                                       tile=3600 * 1000000000,
                                       dtype=np.uint64),
                            tiledb.Dim(name="cccc",
                                       domain=(0, dtype_max(np.uint64) - 1),
                                       tile=dtype_max(np.uint64),
                                       dtype=np.uint64),
                            ctx=ctx)

        compression = tiledb.FilterList([tiledb.ZstdFilter(level=-1)])
        attrs = [
            tiledb.Attr(name="x", dtype='U', filters=compression, ctx=ctx),
            tiledb.Attr(name="chars", dtype='|S2', filters=compression, ctx=ctx),

            tiledb.Attr(name="q", dtype='U', filters=compression, ctx=ctx),
            tiledb.Attr(name="r", dtype='S', filters=compression, ctx=ctx),
            tiledb.Attr(name="s", dtype='U', filters=compression, ctx=ctx),
            tiledb.Attr(name="vals_int64", dtype=np.int64, filters=compression, ctx=ctx),
            tiledb.Attr(name="vals_float64", dtype=np.float64, filters=compression, ctx=ctx),
            tiledb.Attr(name="t", dtype='U', filters=compression, ctx=ctx),
            tiledb.Attr(name="u", dtype='S', filters=compression, ctx=ctx),
            tiledb.Attr(name="v", dtype='S', filters=compression, ctx=ctx),
        ]
        schema = tiledb.ArraySchema(domain=dom, sparse=True,
                                    attrs=attrs,
                                    ctx=ctx)
        tiledb.SparseArray.create(uri, schema)

        df = make_dataframe_basic1()
        incr = 0
        with tiledb.SparseArray(uri, 'w') as A:
            s_ichars = []
            for s in df['chars']:
                s_ichars.append(incr)
                incr += 1

            times = df['time']
            cccc = df['cccc']

            df = df.drop(columns=['time', 'cccc'], axis=1)
            A[s_ichars, times, cccc] = df.to_dict(orient='series')

        with tiledb.SparseArray(uri) as A:
            df1 = pd.DataFrame.from_dict(A[:])
            for col in df.columns:
                assert_array_equal(df[col], df1[col])

    def test_dataframe_basic1(self):
        uri = self.path("dataframe_basic_rt1")
        df = make_dataframe_basic1()

        tiledb.from_dataframe(uri, df)

        # TODO tiledb.read_dataframe
        with tiledb.open(uri) as B:
            df_readback = pd.DataFrame.from_dict(B[:])
            tm.assert_frame_equal(df, df_readback)

    def test_dataframe_basic2(self):
        uri = self.path("dataframe_basic_rt2")

        df = make_dataframe_basic2()

        tiledb.from_dataframe(uri, df)

        with tiledb.open(uri) as B:
            df_readback = tiledb.open_dataframe(uri)
            tm.assert_frame_equal(df, df_readback)

    def test_dataframe_csv_rt1(self):
        def rand_dtype(dtype, size):
            import os
            nbytes = size * np.dtype(dtype).itemsize

            randbytes = os.urandom(nbytes)
            return np.frombuffer(randbytes, dtype=dtype)

        uri = self.path("dataframe_csv_rt1")
        os.mkdir(uri)
        col_size=15
        data_dict = {
            'dates': np.array(
                rand_dtype(np.uint64, col_size), dtype=np.dtype('datetime64[ns]')
            ),
            'float64s': rand_dtype(np.float64, col_size),
            'ints': rand_dtype(np.int64, col_size),
            'strings': [rand_utf8(5) for _ in range(col_size)],
        }

        df_orig = pd.DataFrame.from_dict(data_dict)

        csv_uri = os.path.join(uri, "test.csv")
        # note: encoding must be specified to avoid printing the b'' bytes
        #       prefix, see https://github.com/pandas-dev/pandas/issues/9712
        df_orig.to_csv(csv_uri, mode='w')

        csv_array_uri = os.path.join(uri, "tiledb_csv")
        tiledb.from_csv(csv_array_uri, csv_uri, index_col = 0, parse_dates=[1])

        df_from_array = tiledb.open_dataframe(csv_array_uri)
        tm.assert_frame_equal(df_orig, df_from_array)

        # Test reading via TileDB VFS. The main goal is to support reading
        # from a remote VFS, using local with `file://` prefix as a test for now.
        with tiledb.FileIO(tiledb.VFS(), csv_uri, 'rb') as fio:
            csv_uri_unc = "file:///" + csv_uri
            csv_array_uri2 = "file:///" + os.path.join(csv_array_uri+"_2")
            tiledb.from_csv(csv_array_uri2, csv_uri_unc, index_col=0, parse_dates=[1])

            df_from_array2 = tiledb.open_dataframe(csv_array_uri2)
            tm.assert_frame_equal(df_orig, df_from_array2)

