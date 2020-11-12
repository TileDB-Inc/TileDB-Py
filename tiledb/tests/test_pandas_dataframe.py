from __future__ import absolute_import

try:
    import pandas as pd
    import pandas._testing as tm

    import_failed = False
except ImportError:
    import_failed = True

import unittest, os
import warnings
import string, random, copy
import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal

import tiledb
from tiledb.tests.common import *

if (sys.version_info > (3,0)):
    str_type = str
else:
    str_type = unicode

def make_dataframe_basic1(col_size=10):
    # ensure no duplicates when using as string dim
    chars = list()
    for _ in range(col_size):
        next = rand_ascii_bytes(2)
        while next in chars:
            next = rand_ascii_bytes(2)
        chars.append(next)

    data_dict = {
        'time': rand_datetime64_array(col_size),
        'x': np.array([rand_ascii(4).encode('UTF-8') for _ in range(col_size)]),
        'chars': np.array(chars),
        'cccc': np.arange(0, col_size),

        'q': np.array([rand_utf8(np.random.randint(1, 100)) for _ in range(col_size)]),
        't': np.array([rand_utf8(4) for _ in range(col_size)]),

        'r': np.array([rand_ascii_bytes(np.random.randint(1, 100)) for _ in range(col_size)]),
        's': np.array([rand_ascii() for _ in range(col_size)]),
        'u': np.array([rand_ascii_bytes() for _ in range(col_size)]),
        'v': np.array([rand_ascii_bytes() for _ in range(col_size)]),

        'vals_int64': np.random.randint(dtype_max(np.int64), size=col_size, dtype=np.int64),
        'vals_float64': np.random.rand(col_size),
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

def make_dataframe_basic3(col_size=10, time_range=(None,None)):
    df_dict = {
        'time': rand_datetime64_array(col_size, start=time_range[0], stop=time_range[1]),
        'double_range': np.linspace(-1000, 1000, col_size),
        'int_vals': np.random.randint(dtype_max(np.int64), size=col_size, dtype=np.int64)
        }
    df = pd.DataFrame(df_dict)
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
            tiledb.Attr(name="x", dtype='S', filters=compression, ctx=ctx),
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

        ctx = tiledb.Ctx()
        tiledb.from_dataframe(uri, df, sparse=False, ctx=ctx)

        df_readback = tiledb.open_dataframe(uri)
        tm.assert_frame_equal(df, df_readback)

        uri = self.path("dataframe_basic_rt1_unlimited")
        tiledb.from_dataframe(uri, df, full_domain=True, sparse=False, ctx=ctx)
        df_readback = tiledb.open_dataframe(uri)
        tm.assert_frame_equal(df, df_readback)

    def test_dataframe_basic2(self):
        uri = self.path("dataframe_basic_rt2")

        df = make_dataframe_basic2()

        tiledb.from_dataframe(uri, df, sparse=False)

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
        tiledb.from_csv(csv_array_uri, csv_uri, index_col = 0, parse_dates=[1], sparse=False)

        ctx = tiledb.default_ctx()
        df_from_array = tiledb.open_dataframe(csv_array_uri, ctx=ctx)
        tm.assert_frame_equal(df_orig, df_from_array)

        # Test reading via TileDB VFS. The main goal is to support reading
        # from a remote VFS, using local with `file://` prefix as a test for now.
        with tiledb.FileIO(tiledb.VFS(), csv_uri, 'rb') as fio:
            csv_uri_unc = "file:///" + csv_uri
            csv_array_uri2 = "file:///" + os.path.join(csv_array_uri+"_2")
            tiledb.from_csv(csv_array_uri2, csv_uri_unc, index_col=0, parse_dates=[1], sparse=False)

            df_from_array2 = tiledb.open_dataframe(csv_array_uri2)
            tm.assert_frame_equal(df_orig, df_from_array2)

    def test_dataframe_index_to_sparse_dims(self):
        # This test
        # - loops over all of the columns from make_basic_dataframe,
        # - sets the index to the current column
        # - creates a dataframe
        # - check that indexing the nonempty_domain of the resulting
        #   dimension matches the input

        # TODO should find a way to dump the whole dataframe dict to a
        #      (print-safe) bytestring in order to debug generated output
        df = make_dataframe_basic1(100)

        for col in df.columns:
            uri = self.path("df_indx_dim+{}".format(str(col)))

            # ensure that all column which will be used as string dim index
            # is sorted, because that is how it will be returned
            if df.dtypes[col] == 'O':
                df.sort_values(col, inplace=True)

                # also ensure that string columns are converted to bytes
                # b/c only TILEDB_ASCII supported for string dimension
                if type(df[col][0]) == str_type:
                    df[col] = [x.encode('UTF-8') for x in df[col]]

            new_df = df.drop_duplicates(subset=col)
            new_df.set_index(col, inplace=True)

            tiledb.from_dataframe(uri, new_df, sparse=True)

            with tiledb.open(uri) as A:
                self.assertEqual(A.domain.dim(0).name, col)

                nonempty = A.nonempty_domain()[0]
                res = A.multi_index[nonempty[0]:nonempty[1]]

                res_df = pd.DataFrame(res, index=res.pop(col))
                tm.assert_frame_equal(new_df, res_df, check_like=True)

    def test_dataframe_multiindex_dims(self):
        uri = self.path("df_multiindex_dims")

        col_size = 10
        df = make_dataframe_basic3(col_size)
        df_dict = df.to_dict(orient='series')
        df.set_index(['time', 'double_range'], inplace=True)

        tiledb.from_dataframe(uri, df)

        with tiledb.open(uri) as A:
            ned_time = A.nonempty_domain()[0]
            ned_dbl = A.nonempty_domain()[1]

            res = A.multi_index[slice(*ned_time), :]
            assert_array_equal(
                res['time'], df_dict['time']
            )
            assert_array_equal(
                res['double_range'], df_dict['double_range']
            )
            assert_array_equal(
                res['int_vals'], df.int_vals.values
            )

            # test .df[] indexing
            df_idx_res = A.df[slice(*ned_time), :]
            tm.assert_frame_equal(df_idx_res, df)

            # test .df[] indexing with query
            df_idx_res = A.query(attrs=['int_vals']).df[slice(*ned_time), :]
            tm.assert_frame_equal(df_idx_res, df)

    def test_csv_dense(self):
        col_size = 10
        df_data = {
            'index': np.arange(0,col_size),
            'chars': np.array([rand_ascii(4).encode('UTF-8') for _ in range(col_size)]),
            'vals_float64': np.random.rand(col_size),
        }
        df = pd.DataFrame(df_data).set_index('index')

        # Test 1: basic round-trip
        tmp_dir = self.path("csv_dense")
        os.mkdir(tmp_dir)
        tmp_csv = os.path.join(tmp_dir, "generated.csv")

        df.to_csv(tmp_csv)

        tmp_array = os.path.join(tmp_dir, "array")
        tiledb.from_csv(tmp_array,
                        tmp_csv,
                        index_col=['index'],
                        dtype={'index': np.uint64},
                        sparse=False)

        tmp_array2 = os.path.join(tmp_dir, "array2")
        tiledb.from_csv(tmp_array2, tmp_csv,
                        sparse=False)

    def test_csv_col_to_sparse_dims(self):
        df = make_dataframe_basic3(20)

        # Test 1: basic round-trip
        tmp_dir = self.path("csv_col_to_sparse_dims")
        os.mkdir(tmp_dir)
        tmp_csv = os.path.join(tmp_dir, "generated.csv")

        df.sort_values('time', inplace=True)
        df.to_csv(tmp_csv, index=False)
        df.set_index(['time', 'double_range'], inplace=True)

        tmp_array = os.path.join(tmp_dir, "array")
        tiledb.from_csv(tmp_array, tmp_csv, index_col=['time', 'double_range'], parse_dates=['time'])

        df_bk = tiledb.open_dataframe(tmp_array)

        tm.assert_frame_equal(df, df_bk)

        # Test 2: check from_csv `sparse` and `allows_duplicates` keyword args
        df = make_dataframe_basic3(20)
        tmp_csv2 = os.path.join(tmp_dir, "generated2.csv")
        tmp_array2a = os.path.join(tmp_dir, "array2a")
        tmp_array2b = os.path.join(tmp_dir, "array2b")

        # create a duplicate value
        df.loc[0, 'int_vals'] = df.int_vals[1]
        df.sort_values('int_vals', inplace=True)

        df.to_csv(tmp_csv2, index=False)

        # try once and make sure error is raised because of duplicate value
        with self.assertRaisesRegex(tiledb.TileDBError, "Duplicate coordinates \\(.*\\) are not allowed"):
            tiledb.from_csv(tmp_array2a, tmp_csv2, index_col=['int_vals'], sparse=True, allows_duplicates=False)

        # try again, check from_csv(allows_duplicates=True, sparse=True)
        tiledb.from_csv(tmp_array2b, tmp_csv2, index_col=['int_vals'],
                        sparse=True, allows_duplicates=True, float_precision='round-trip')

        with tiledb.open(tmp_array2b) as A:
            #self.assertTrue(A.schema.sparse)
            res = A[:]
            assert_array_equal(res['int_vals'], df.int_vals.values)
            assert_array_almost_equal(res['double_range'], df.double_range.values)

    def test_dataframe_csv_schema_only(self):
        col_size = 10
        df = make_dataframe_basic3(col_size)

        tmp_dir = self.path("csv_schema_only")
        os.mkdir(tmp_dir)
        tmp_csv = os.path.join(tmp_dir, "generated.csv")

        df.sort_values('time', inplace=True)
        df.to_csv(tmp_csv, index=False)

        attrs_filters = tiledb.FilterList([tiledb.ZstdFilter(1)])
        # from_dataframe default is 1, so use 7 here to check
        #   the arg is correctly parsed/passed
        coords_filters = tiledb.FilterList([tiledb.ZstdFilter(7)])

        tmp_assert_dir = os.path.join(tmp_dir, "array")
        # this should raise an error
        with self.assertRaises(ValueError):
            tiledb.from_csv(tmp_assert_dir, tmp_csv, tile='abc')

        with self.assertRaises(ValueError):
            tiledb.from_csv(tmp_assert_dir, tmp_csv, tile=(3,1.0))

        tmp_array = os.path.join(tmp_dir, "array")
        tiledb.from_csv(tmp_array, tmp_csv,
                        index_col=['time', 'double_range'],
                        parse_dates=['time'],
                        mode='schema_only',
                        capacity=1001,
                        tile={'time': 5},
                        coords_filters=coords_filters)

        t0, t1 = df.time.min(), df.time.max()

        import numpy
        ref_schema = tiledb.ArraySchema(
                        domain=tiledb.Domain(*[
                          tiledb.Dim(name='time', domain=(t0.to_datetime64(), t1.to_datetime64()),
                                     tile=5, dtype='datetime64[ns]'),
                          tiledb.Dim(name='double_range', domain=(-1000.0, 1000.0), tile=1000, dtype='float64'),
                        ]),
                        attrs=[
                          tiledb.Attr(name='int_vals', dtype='int64', filters=attrs_filters),
                        ],
                        coords_filters=coords_filters,
                        cell_order='row-major',
                        tile_order='row-major',
                        capacity=1001,
                        sparse=True,
                        allows_duplicates=False)
                        # note: filters omitted

        array_nfiles = len(tiledb.VFS().ls(tmp_array))
        self.assertEqual(array_nfiles, 3)

        with tiledb.open(tmp_array) as A:
            self.assertEqual(A.schema, ref_schema)

            # TODO currently no equality check for filters
            self.assertEqual(
                A.schema.coords_filters[0].level, coords_filters[0].level
            )
            self.assertEqual(
                A.schema.attr(0).filters[0].level, attrs_filters[0].level
            )

        # Test mode='append'
        tiledb.from_csv(tmp_array, tmp_csv,
                        index_col=['time', 'double_range'], mode='append')
        df2 = make_dataframe_basic3(10, time_range=(t0, t1))
        df2.sort_values('time', inplace=True)
        df2.set_index(['time', 'double_range'], inplace=True)
        tiledb.from_dataframe(tmp_array, df2, mode='append')

        with tiledb.open(tmp_array) as A:
            res = A[:]
            df_bk = pd.DataFrame(res)
            df_bk.set_index(['time','double_range'], inplace=True)

            df.set_index(['time','double_range'], inplace=True)
            df_combined = pd.concat([df, df2])
            df_combined.sort_index(level='time', inplace=True)
            tm.assert_frame_equal(df_bk, df_combined)

    def test_dataframe_csv_chunked(self):
        col_size = 200
        df = make_dataframe_basic3(col_size)

        tmp_dir = self.path("csv_chunked")
        os.mkdir(tmp_dir)
        tmp_csv = os.path.join(tmp_dir, "generated.csv")

        df.sort_values('time', inplace=True)
        df.to_csv(tmp_csv, index=False)

        # Test sparse chunked
        tmp_array = os.path.join(tmp_dir, "array")
        tiledb.from_csv(tmp_array, tmp_csv,
                        index_col=['double_range'],
                        parse_dates=['time'],
                        date_spec={'time': "%Y-%m-%dT%H:%M:%S.%f"},
                        chunksize=10)

        with tiledb.open(tmp_array) as A:
            res = A[:]
            df_bk = pd.DataFrame(res)
            df_bk.set_index(['double_range'], inplace=True)

            df_ck = df.set_index(['double_range'])
            tm.assert_frame_equal(df_bk, df_ck)

        # Test dense chunked
        tmp_array_dense = os.path.join(tmp_dir, "array_dense")
        tiledb.from_csv(tmp_array_dense, tmp_csv,
                        parse_dates=['time'],
                        sparse=False,
                        chunksize=25)

        with tiledb.open(tmp_array_dense) as A:
            # with sparse=False and no index column, we expect to have unlimited domain
            self.assertEqual(A.schema.domain.dim(0).domain[1], 18446744073709541615)

            # chunked writes go to unlimited domain, so we must only read nonempty
            ned = A.nonempty_domain()[0]
            # TODO should support numpy scalar here
            res = A.multi_index[int(ned[0]):int(ned[1])]
            df_bk = pd.DataFrame(res)

            tm.assert_frame_equal(df_bk, df)

            # test .df[] indexing
            df_idx_res = A.df[int(ned[0]):int(ned[1])]
            tm.assert_frame_equal(df_idx_res, df)

            # test .df[] indexing with query
            df_idx_res = A.query(attrs=['time']).df[int(ned[0]):int(ned[1])]
            tm.assert_frame_equal(df_idx_res, df[['time']])

            df_idx_res = A.query(attrs=['double_range']).df[int(ned[0]):int(ned[1])]
            tm.assert_frame_equal(df_idx_res, df[['double_range']])

            # disable coordinate dimension/index
            df_idx_res = A.query(coords=False).df[int(ned[0]):int(ned[1])]
            tm.assert_frame_equal(df_idx_res, df.reset_index(drop=True))

    def test_csv_fillna(self):
        col_size = 10
        data = np.random.rand(10) * 100 # make some integers for the 2nd test
        data[4] = np.nan
        df = pd.DataFrame({'v': data})

        tmp_dir = self.path("csv_fillna")
        os.mkdir(tmp_dir)
        tmp_csv = os.path.join(tmp_dir, "generated.csv")

        df.to_csv(tmp_csv, index=False, na_rep="NaN")

        tmp_array = os.path.join(tmp_dir, "array")
        tiledb.from_csv(tmp_array, tmp_csv, fillna={'v': 0})

        def check_array(path, df):
            # update the value in the original dataframe to match what we expect on read-back
            df['v'][4] = 0

            with tiledb.open(path) as A:
                df_bk = A.df[:]
                tm.assert_frame_equal(df_bk, df)

        check_array(tmp_array, copy.deepcopy(df))

        if hasattr(pd, 'StringDtype'):
            tmp_array2 = os.path.join(tmp_dir, "array2")
            tiledb.from_csv(tmp_array2,
                            tmp_csv,
                            fillna={'v': 0},
                            column_types={'v': pd.Int64Dtype})
            df_to_check = copy.deepcopy(df)
            df_to_check['v'][4] = 0
            df_to_check = df_to_check.astype({'v': np.int64})
            check_array(tmp_array2, df_to_check)

    def test_csv_multi_file(self):
        col_size = 10

        csv_dir = self.path("csv_multi_dir")
        os.mkdir(csv_dir)

        # Write a set of CSVs with 10 rows each
        input_dfs = list()
        for i in range(20):
            df = make_dataframe_basic3(col_size)
            output_path = os.path.join(csv_dir, "csv_{}.csv".format(i))
            df.to_csv(output_path, index=False)
            input_dfs.append(df)

        tmp_dir = self.path("csv_multi_array_dir")
        os.mkdir(tmp_dir)

        # Create TileDB array with flush every 25 rows
        csv_paths = glob.glob(csv_dir + "/*.csv")
        tmp_array = os.path.join(tmp_dir, "array")
        tiledb.from_csv(tmp_array, csv_paths,
                        index_col=['time'],
                        parse_dates=['time'],
                        chunksize=25,
                        sparse=True)

        # Check number of fragments
        # * should equal 8 based on chunksize=25
        # * 20 files, 10 rows each, 200 rows == 8 writes:
        fragments = glob.glob(tmp_array + "/*.ok")
        self.assertEqual(len(fragments), 8)

        # Check the returned data
        # note: tiledb returns sorted values
        df_orig = pd.concat(input_dfs, axis=0).set_index(["time"]).sort_values('time')

        with tiledb.open(tmp_array) as A:
            res = A[:]
            df_bk = pd.DataFrame(res)
            df_bk = df_bk.set_index(['time'])

            tm.assert_frame_equal(df_bk, df_orig)
