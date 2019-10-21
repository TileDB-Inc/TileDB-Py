from __future__ import absolute_import

try:
    import pandas as pd
    import_failed = False
except ImportError:
    import_failed = True

import unittest, os
import string, random
import numpy as np
from numpy.testing import assert_array_equal

import tiledb
from tiledb.tests.common import *

def make_dataframe_ex1(col_size=10):
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

class PandasDataFrameRoundtrip(DiskTestCase):
    def setUp(self):
        if import_failed:
            self.skipTest("Pandas not available")
        else:
            super().setUp()

    def test_manual_dataframe_rt(self):

        uri = self.path("df_roundtrip")

        print(uri)
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
            tiledb.Attr(name="chars", dtype='S', filters=compression, ctx=ctx),

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

        df = make_dataframe_ex1()
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
