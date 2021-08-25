import pytest

pd = pytest.importorskip("pandas")
tm = pd._testing

import copy
import glob
import os
import random
from pathlib import Path

import numpy as np
from numpy.testing import assert_array_equal

import tiledb
from tiledb.dataframe_ import ColumnInfo
from tiledb.tests.common import (
    DiskTestCase,
    checked_path,
    dtype_max,
    rand_ascii,
    rand_ascii_bytes,
    rand_datetime64_array,
    rand_utf8,
)


def make_dataframe_basic1(col_size=10):
    # ensure no duplicates when using as string dim
    chars = list()
    for _ in range(col_size):
        next = rand_ascii_bytes(2)
        while next in chars:
            next = rand_ascii_bytes(2)
        chars.append(next)

    data_dict = {
        "time": rand_datetime64_array(col_size),
        "x": np.array([rand_ascii(4).encode("UTF-8") for _ in range(col_size)]),
        "chars": np.array(chars),
        "cccc": np.arange(0, col_size),
        "q": np.array([rand_utf8(np.random.randint(1, 100)) for _ in range(col_size)]),
        "t": np.array([rand_utf8(4) for _ in range(col_size)]),
        "r": np.array(
            [rand_ascii_bytes(np.random.randint(1, 100)) for _ in range(col_size)]
        ),
        "s": np.array([rand_ascii() for _ in range(col_size)]),
        "u": np.array([rand_ascii_bytes().decode() for _ in range(col_size)]),
        "v": np.array([rand_ascii_bytes() for _ in range(col_size)]),
        "vals_int64": np.random.randint(
            dtype_max(np.int64), size=col_size, dtype=np.int64
        ),
        "vals_float64": np.random.rand(col_size),
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
    df = pd.DataFrame(
        {
            "string": list("abc"),
            "int": list(range(1, 4)),
            "uint": np.arange(3, 6).astype("u1"),
            "float": np.arange(4.0, 7.0, dtype="float64"),
            # TODO "float_with_null": [1.0, np.nan, 3],
            "bool": [True, False, True],
            # TODO "bool_with_null": [True, np.nan, False],
            # "cat": pd.Categorical(list("abc")),
            "dt": pd.date_range("20130101", periods=3),
            # "dttz": pd.date_range("20130101", periods=3, tz="US/Eastern"),
            # "dt_with_null": [
            #    pd.Timestamp("20130101"),
            #    pd.NaT,
            #    pd.Timestamp("20130103"),
            # ],
            "dtns": pd.date_range("20130101", periods=3, freq="ns"),
        }
    )

    return df


def make_dataframe_basic3(col_size=10, time_range=(None, None)):
    df_dict = {
        "time": rand_datetime64_array(
            col_size, start=time_range[0], stop=time_range[1]
        ),
        "double_range": np.linspace(-1000, 1000, col_size),
        "int_vals": np.random.randint(
            dtype_max(np.int64), size=col_size, dtype=np.int64
        ),
    }
    df = pd.DataFrame(df_dict)
    return df


class TestColumnInfo:
    def assertColumnInfo(self, info, info_dtype, info_repr=None, info_nullable=False):
        assert isinstance(info.dtype, np.dtype)
        assert info.dtype == info_dtype

        assert info.repr is None or isinstance(info.repr, str)
        assert info.repr == info_repr

        assert isinstance(info.nullable, bool)
        assert info.nullable == info_nullable

    @pytest.mark.parametrize(
        "type_specs, info_dtype, info_repr, info_nullable",
        [
            # bool types
            ([bool, "b1"], np.dtype("uint8"), "bool", False),
            ([pd.BooleanDtype()], np.dtype("uint8"), "boolean", True),
            # numeric types
            ([np.uint8, "u1"], np.dtype("uint8"), None, False),
            ([np.uint16, "u2"], np.dtype("uint16"), None, False),
            ([np.uint32, "u4"], np.dtype("uint32"), None, False),
            ([np.uint64, "u8"], np.dtype("uint64"), None, False),
            ([np.int8, "i1"], np.dtype("int8"), None, False),
            ([np.int16, "i2"], np.dtype("int16"), None, False),
            ([np.int32, "i4"], np.dtype("int32"), None, False),
            ([np.int64, "i8"], np.dtype("int64"), None, False),
            ([np.float32, "f4"], np.dtype("float32"), None, False),
            ([np.float64, "f8", float], np.dtype("float64"), None, False),
            # nullable int types
            ([pd.UInt8Dtype(), "UInt8"], np.dtype("uint8"), "UInt8", True),
            ([pd.UInt16Dtype(), "UInt16"], np.dtype("uint16"), "UInt16", True),
            ([pd.UInt32Dtype(), "UInt32"], np.dtype("uint32"), "UInt32", True),
            ([pd.UInt64Dtype(), "UInt64"], np.dtype("uint64"), "UInt64", True),
            ([pd.Int8Dtype(), "Int8"], np.dtype("int8"), "Int8", True),
            ([pd.Int16Dtype(), "Int16"], np.dtype("int16"), "Int16", True),
            ([pd.Int32Dtype(), "Int32"], np.dtype("int32"), "Int32", True),
            ([pd.Int64Dtype(), "Int64"], np.dtype("int64"), "Int64", True),
            # datetime types
            (["datetime64[ns]"], np.dtype("<M8[ns]"), None, False),
            # string types
            ([np.str_, str], np.dtype("<U"), None, False),
            ([np.bytes_, bytes], np.dtype("S"), None, False),
            ([pd.StringDtype()], np.dtype("<U"), "string", True),
        ],
    )
    def test_implemented(self, type_specs, info_dtype, info_repr, info_nullable):
        assert isinstance(info_dtype, np.dtype)
        assert info_repr is None or isinstance(info_repr, str)
        assert isinstance(info_nullable, bool)
        for type_spec in type_specs:
            self.assertColumnInfo(
                ColumnInfo.from_dtype(type_spec), info_dtype, info_repr, info_nullable
            )

            series = pd.Series([], dtype=type_spec)
            if series.dtype == type_spec:
                self.assertColumnInfo(
                    ColumnInfo.from_values(series), info_dtype, info_repr, info_nullable
                )

    def test_object_dtype(self):
        self.assertColumnInfo(
            ColumnInfo.from_values(pd.Series(["hello", "world"])), np.dtype("<U")
        )
        self.assertColumnInfo(
            ColumnInfo.from_values(pd.Series([b"hello", b"world"])), np.dtype("S")
        )
        for s in ["hello", b"world"], ["hello", 1], [b"hello", 1]:
            pytest.raises(NotImplementedError, ColumnInfo.from_values, pd.Series(s))

    unsupported_type_specs = [
        [np.float16, "f2"],
        [np.complex64, "c8"],
        [np.complex128, "c16"],
        [np.datetime64, "<M8", "datetime64"],
        [
            "<M8[Y]",
            "<M8[M]",
            "<M8[W]",
            "<M8[D]",
            "<M8[h]",
            "<M8[m]",
            "<M8[s]",
            "<M8[ms]",
            "<M8[us]",
            "<M8[ps]",
            "<M8[fs]",
            "<M8[as]",
        ],
    ]
    if hasattr(np, "float128"):
        unsupported_type_specs.append([np.float128, "f16"])
    if hasattr(np, "complex256"):
        unsupported_type_specs.append([np.complex256, "c32"])

    @pytest.mark.parametrize("type_specs", unsupported_type_specs)
    def test_not_implemented(self, type_specs):
        for type_spec in type_specs:
            pytest.raises(NotImplementedError, ColumnInfo.from_dtype, type_spec)
            try:
                series = pd.Series([], dtype=type_spec)
            except (ValueError, TypeError):
                pass
            else:
                if series.dtype == type_spec:
                    pytest.raises(NotImplementedError, ColumnInfo.from_values, series)


class TestDimType:
    @pytest.mark.parametrize(
        ["df", "expected", "kwargs"],
        [
            (
                pd.DataFrame(
                    {"data": np.array(["a"], dtype=np.str_)},
                    index=pd.Index(np.array(["b"], dtype=np.str_), name="str_dim"),
                ),
                np.bytes_,
                {},
            )
        ],
    )
    def test_schema_dim(self, checked_path, df, expected, kwargs):
        assert isinstance(df, pd.DataFrame)

        uri = checked_path.path()
        tiledb.from_pandas(uri, df, **kwargs)

        with tiledb.open(uri) as A:
            assert A.schema.domain.dim(0).dtype == expected
            assert A.schema.domain.dim(0).name == df.index.name


class TestPandasDataFrameRoundtrip(DiskTestCase):
    def test_dataframe_basic_rt1_manual(self):
        uri = self.path("dataframe_basic_rt1_manual")
        dom = tiledb.Domain(
            tiledb.Dim(name="i_chars", domain=(0, 10000), tile=10, dtype=np.uint64),
            tiledb.Dim(
                name="datetime",
                domain=(0, np.iinfo(np.uint64).max - 3600 * 1000000000),
                tile=3600 * 1000000000,
                dtype=np.uint64,
            ),
            tiledb.Dim(
                name="cccc",
                domain=(0, dtype_max(np.uint64) - 1),
                tile=dtype_max(np.uint64),
                dtype=np.uint64,
            ),
        )

        compression = tiledb.FilterList([tiledb.ZstdFilter(level=-1)])
        attrs = [
            tiledb.Attr(name="x", dtype="S", filters=compression),
            tiledb.Attr(name="chars", dtype="|S2", filters=compression),
            tiledb.Attr(name="q", dtype="U", filters=compression),
            tiledb.Attr(name="r", dtype="S", filters=compression),
            tiledb.Attr(name="s", dtype="U", filters=compression),
            tiledb.Attr(name="vals_int64", dtype=np.int64, filters=compression),
            tiledb.Attr(name="vals_float64", dtype=np.float64, filters=compression),
            tiledb.Attr(name="t", dtype="U", filters=compression),
            tiledb.Attr(name="u", dtype="U", filters=compression),
            tiledb.Attr(name="v", dtype="S", filters=compression),
        ]
        schema = tiledb.ArraySchema(domain=dom, sparse=True, attrs=attrs)
        tiledb.SparseArray.create(uri, schema)

        df = make_dataframe_basic1()
        incr = 0
        with tiledb.SparseArray(uri, "w") as A:
            s_ichars = []
            for s in df["chars"]:
                s_ichars.append(incr)
                incr += 1

            times = df["time"]
            cccc = df["cccc"]

            df = df.drop(columns=["time", "cccc"], axis=1)
            A[s_ichars, times, cccc] = df.to_dict(orient="series")

        with tiledb.SparseArray(uri) as A:
            df_res = pd.DataFrame.from_dict(A[:])
            for col in df.columns:
                # TileDB default return is unordered, so must sort to compare
                assert_array_equal(df[col].sort_values(), df_res[col].sort_values())

    def test_dataframe_basic1(self):
        uri = self.path("dataframe_basic_rt1")
        df = make_dataframe_basic1()

        tiledb.from_pandas(uri, df, sparse=False)

        df_readback = tiledb.open_dataframe(uri)
        tm.assert_frame_equal(df, df_readback)

        uri = self.path("dataframe_basic_rt1_unlimited")
        tiledb.from_pandas(uri, df, full_domain=True, sparse=False)

        with tiledb.open(uri) as A:
            dim = A.schema.domain.dim(0)
            assert dim.domain[0] == 0
            assert dim.domain[1] == dtype_max(np.uint64) - dim.tile

        for use_arrow in None, False, True:
            df_readback = tiledb.open_dataframe(uri, use_arrow=use_arrow)
            tm.assert_frame_equal(df, df_readback)

            attrs = ["s", "q", "t"]
            df_readback = tiledb.open_dataframe(uri, attrs=attrs, use_arrow=use_arrow)
            tm.assert_frame_equal(df[attrs], df_readback)

            df_readback = tiledb.open_dataframe(
                uri, idx=slice(2, 4), use_arrow=use_arrow
            )
            tm.assert_frame_equal(df[2:5], df_readback)

    def test_dataframe_basic2(self):
        uri = self.path("dataframe_basic_rt2")

        df = make_dataframe_basic2()

        tiledb.from_pandas(uri, df, sparse=False)

        df_readback = tiledb.open_dataframe(uri)
        tm.assert_frame_equal(df, df_readback)

        with tiledb.open(uri) as B:
            tm.assert_frame_equal(df, B.df[:])

    def test_dataframe_csv_rt1(self):
        def rand_dtype(dtype, size):
            import os

            nbytes = size * np.dtype(dtype).itemsize

            randbytes = os.urandom(nbytes)
            return np.frombuffer(randbytes, dtype=dtype)

        uri = self.path("dataframe_csv_rt1")
        self.vfs.create_dir(uri)
        col_size = 15
        data_dict = {
            "dates": np.array(
                rand_dtype(np.uint64, col_size), dtype=np.dtype("datetime64[ns]")
            ),
            "float64s": rand_dtype(np.float64, col_size),
            "ints": rand_dtype(np.int64, col_size),
            "strings": [rand_utf8(5) for _ in range(col_size)],
        }

        df_orig = pd.DataFrame.from_dict(data_dict)
        csv_uri = os.path.join(uri, "test.csv")

        # note: encoding must be specified to avoid printing the b'' bytes
        #       prefix, see https://github.com/pandas-dev/pandas/issues/9712
        with tiledb.FileIO(self.vfs, csv_uri, "wb") as fio:
            df_orig.to_csv(fio, mode="w")

        csv_array_uri = os.path.join(uri, "tiledb_csv")
        tiledb.from_csv(
            csv_array_uri, csv_uri, index_col=0, parse_dates=[1], sparse=False
        )

        df_from_array = tiledb.open_dataframe(csv_array_uri)
        tm.assert_frame_equal(df_orig, df_from_array)

        # Test reading via TileDB VFS. The main goal is to support reading
        # from a remote VFS, using local with `file://` prefix as a test for now.
        with tiledb.FileIO(tiledb.VFS(), csv_uri, "rb") as fio:
            csv_array_uri2 = os.path.join(csv_array_uri + "_2")
            tiledb.from_csv(
                csv_array_uri2, csv_uri, index_col=0, parse_dates=[1], sparse=False
            )

            df_from_array2 = tiledb.open_dataframe(csv_array_uri2)
            tm.assert_frame_equal(df_orig, df_from_array2)

        # test timestamp write
        uri2 = self.path("dataframe_csv_timestamp")
        timestamp = random.randint(0, np.iinfo(np.int64).max)
        tiledb.from_csv(uri2, csv_uri, timestamp=0, index_col=0)
        tiledb.from_pandas(
            uri2,
            df_orig,
            timestamp=timestamp,
            mode="append",
            row_start_idx=0,
            index_col=0,
        )

        with tiledb.open(uri2, timestamp=0) as A:
            assert A.timestamp_range == (0, 0)
        with tiledb.open(uri2, timestamp=timestamp) as A:
            assert A.timestamp_range == (0, timestamp)

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
            if df.dtypes[col] == "O":
                df.sort_values(col, inplace=True)

                # also ensure that string columns are converted to bytes
                # b/c only TILEDB_ASCII supported for string dimension
                if type(df[col][0]) is str:
                    df[col] = [x.encode("UTF-8") for x in df[col]]

            new_df = df.drop_duplicates(subset=col)
            new_df.set_index(col, inplace=True)

            tiledb.from_pandas(uri, new_df, sparse=True)

            with tiledb.open(uri) as A:
                self.assertEqual(A.domain.dim(0).name, col)

                nonempty = A.nonempty_domain()[0]
                res = A.multi_index[nonempty[0] : nonempty[1]]

                index = pd.Index(res.pop(col), name=col)
                res_df = pd.DataFrame(res, index=index)
                tm.assert_frame_equal(new_df, res_df, check_like=True)

    def test_dataframe_set_index_dims(self):
        uri = self.path("df_set_index_dims")

        df = make_dataframe_basic3()
        tiledb.from_pandas(uri, df, sparse=True, index_dims=["time"])

        df_readback = tiledb.open_dataframe(uri)
        df_reindexed = df.set_index("time")
        tm.assert_frame_equal(df_reindexed, df_readback)

        with tiledb.open(uri) as B:
            tm.assert_frame_equal(df_reindexed, B.df[:])

    def test_dataframe_append_index_dims(self):
        uri = self.path("df_append_index_dims")

        df = make_dataframe_basic3()
        tiledb.from_pandas(uri, df, sparse=True, index_dims=[None, "time"])

        df_readback = tiledb.open_dataframe(uri)
        df_reindexed = df.set_index("time", append=True)
        tm.assert_frame_equal(df_reindexed, df_readback)

        with tiledb.open(uri) as B:
            tm.assert_frame_equal(df_reindexed, B.df[:])

    def test_dataframe_multiindex_dims(self):
        uri = self.path("df_multiindex_dims")

        col_size = 10
        df = make_dataframe_basic3(col_size)
        df_dict = df.to_dict(orient="series")
        df.set_index(["time", "double_range"], inplace=True)

        tiledb.from_pandas(uri, df, sparse=True)

        with tiledb.open(uri) as A:
            ned_time = A.nonempty_domain()[0]
            ned_dbl = A.nonempty_domain()[1]

            res = A.multi_index[slice(*ned_time), :]
            assert_array_equal(res["time"], df_dict["time"])
            assert_array_equal(res["double_range"], df_dict["double_range"])
            assert_array_equal(res["int_vals"], df.int_vals.values)

            # test .df[] indexing
            df_idx_res = A.df[slice(*ned_time), :]
            tm.assert_frame_equal(df_idx_res, df)

            # test .df[] indexing with query
            df_idx_res = A.query(attrs=["int_vals"]).df[slice(*ned_time), :]
            tm.assert_frame_equal(df_idx_res, df)
            # test .df[] with Arrow
            df_idx_res = A.query(use_arrow=True).df[slice(*ned_time), :]
            tm.assert_frame_equal(df_idx_res, df)
            df_idx_res = A.query(use_arrow=False).df[slice(*ned_time), :]
            tm.assert_frame_equal(df_idx_res, df)

    def test_dataframe_index_name(self):
        uri = self.path("df_index_name")

        df = make_dataframe_basic1(10)
        df.index.name = "range_idx"

        tiledb.from_pandas(uri, df)

        df_readback = tiledb.open_dataframe(uri, use_arrow=False)
        tm.assert_frame_equal(df, df_readback)

        # TODO: fix index name when using pyarrow
        # df_readback = tiledb.open_dataframe(uri, use_arrow=True)
        # tm.assert_frame_equal(df, df_readback)

    def test_dataframe_fillna(self):
        data = pd.Series(np.arange(10), dtype=pd.Int64Dtype())
        data[4] = None
        df = pd.DataFrame({"v": data})
        df_copy = df.copy()

        uri = self.path("df_fillna")
        tiledb.from_pandas(uri, df, fillna={"v": -1})

        # check that original df has not changed
        tm.assert_frame_equal(df, df_copy)

        # update the value in the original dataframe to match what we expect on read-back
        df["v"][4] = -1
        df_bk = tiledb.open_dataframe(uri)
        tm.assert_frame_equal(df_bk, df)

    def test_dataframe_date_spec(self):
        date_fmt = "%d/%m/%Y"
        df = make_dataframe_basic3()
        df["date"] = df["time"].dt.strftime(date_fmt)

        df_copy = df.copy()

        uri = self.path("df_date_spec")
        tiledb.from_pandas(uri, df, date_spec={"date": date_fmt})

        # check that original df has not changed
        tm.assert_frame_equal(df, df_copy)

        # update the column in the original dataframe to match what we expect on read-back
        df["date"] = pd.to_datetime(df["date"], format=date_fmt)
        df_bk = tiledb.open_dataframe(uri)
        tm.assert_frame_equal(df_bk, df)

    def test_dataframe_empty(self):
        dfs = [
            make_dataframe_basic1(),
            make_dataframe_basic2(),
            make_dataframe_basic3(),
        ]
        for i, df in enumerate(dfs, start=1):
            for sparse in False, True:
                uri = self.path(f"dataframe_empty_{i}_{sparse}")
                tiledb.from_pandas(uri, df, sparse=sparse)
                with tiledb.open(uri) as A:
                    tm.assert_frame_equal(df.iloc[:0], A.df[tiledb.EmptyRange])

    def test_csv_dense(self):
        col_size = 10
        df_data = {
            "index": np.arange(0, col_size),
            "chars": np.array([rand_ascii(4).encode("UTF-8") for _ in range(col_size)]),
            "vals_float64": np.random.rand(col_size),
        }
        df = pd.DataFrame(df_data).set_index("index")

        # Test 1: basic round-trip
        tmp_dir = self.path("csv_dense")
        self.vfs.create_dir(tmp_dir)
        tmp_csv = os.path.join(tmp_dir, "generated.csv")

        with tiledb.FileIO(self.vfs, tmp_csv, "wb") as fio:
            df.to_csv(fio)

        tmp_array = os.path.join(tmp_dir, "array")
        tiledb.from_csv(
            tmp_array,
            tmp_csv,
            index_col=["index"],
            dtype={"index": np.uint64},
            sparse=False,
        )

        tmp_array2 = os.path.join(tmp_dir, "array2")
        tiledb.from_csv(tmp_array2, tmp_csv, sparse=False)

    def test_csv_col_to_sparse_dims(self):
        df = make_dataframe_basic3(20)

        # Test 1: basic round-trip
        tmp_dir = self.path("csv_col_to_sparse_dims")
        self.vfs.create_dir(tmp_dir)
        tmp_csv = os.path.join(tmp_dir, "generated.csv")

        df.sort_values("time", inplace=True)
        with tiledb.FileIO(self.vfs, tmp_csv, "wb") as fio:
            df.to_csv(fio, index=False)
        df.set_index(["time", "double_range"], inplace=True)

        tmp_array = os.path.join(tmp_dir, "array")
        tiledb.from_csv(
            tmp_array,
            tmp_csv,
            sparse=True,
            index_col=["time", "double_range"],
            parse_dates=["time"],
        )

        df_bk = tiledb.open_dataframe(tmp_array)

        tm.assert_frame_equal(df, df_bk)

        # Test 2: check from_csv `sparse` and `allows_duplicates` keyword args
        df = make_dataframe_basic3(20)
        tmp_csv2 = os.path.join(tmp_dir, "generated2.csv")
        tmp_array2a = os.path.join(tmp_dir, "array2a")
        tmp_array2b = os.path.join(tmp_dir, "array2b")

        # create a duplicate value
        df.loc[0, "int_vals"] = df.int_vals[1]
        df.sort_values("int_vals", inplace=True)

        with tiledb.FileIO(self.vfs, tmp_csv2, "wb") as fio:
            df.to_csv(fio, index=False)

        # try once and make sure error is raised because of duplicate value
        with self.assertRaisesRegex(
            tiledb.TileDBError, "Duplicate coordinates \\(.*\\) are not allowed"
        ):
            tiledb.from_csv(
                tmp_array2a,
                tmp_csv2,
                index_col=["int_vals"],
                sparse=True,
                allows_duplicates=False,
            )

        # try again, check from_csv(allows_duplicates=True, sparse=True)
        tiledb.from_csv(
            tmp_array2b,
            tmp_csv2,
            index_col=["int_vals"],
            parse_dates=["time"],
            sparse=True,
            allows_duplicates=True,
            float_precision="round_trip",
        )

        with tiledb.open(tmp_array2b) as A:
            self.assertTrue(A.schema.sparse)
            res_df = A.df[:]
            # the duplicate value is on the dimension and can be retrieved in arbitrary
            # order. we need to re-sort in order to compare, to avoid spurious failures.
            res_df.sort_values("time", inplace=True)
            cmp_df = df.set_index("int_vals").sort_values(by="time")
            tm.assert_frame_equal(res_df, cmp_df)

    def test_dataframe_csv_schema_only(self):
        col_size = 10
        df = make_dataframe_basic3(col_size)

        tmp_dir = self.path("csv_schema_only")
        self.vfs.create_dir(tmp_dir)
        tmp_csv = os.path.join(tmp_dir, "generated.csv")

        df.sort_values("time", inplace=True)

        with tiledb.FileIO(self.vfs, tmp_csv, "wb") as fio:
            df.to_csv(fio, index=False)

        attrs_filters = tiledb.FilterList([tiledb.ZstdFilter(3)])
        # from_pandas default is 1, so use 7 here to check
        #   the arg is correctly parsed/passed
        coords_filters = tiledb.FilterList([tiledb.ZstdFilter(7)])

        tmp_assert_dir = os.path.join(tmp_dir, "array")
        # this should raise an error
        with self.assertRaises(ValueError):
            tiledb.from_csv(tmp_assert_dir, tmp_csv, tile="abc")

        with self.assertRaises(ValueError):
            tiledb.from_csv(tmp_assert_dir, tmp_csv, tile=(3, 1.0))

        tmp_array = os.path.join(tmp_dir, "array")
        tiledb.from_csv(
            tmp_array,
            tmp_csv,
            index_col=["time", "double_range"],
            parse_dates=["time"],
            mode="schema_only",
            capacity=1001,
            sparse=True,
            tile={"time": 5},
            coords_filters=coords_filters,
            attr_filters=attrs_filters,
        )

        t0, t1 = df.time.min(), df.time.max()

        import numpy

        ref_schema = tiledb.ArraySchema(
            domain=tiledb.Domain(
                *[
                    tiledb.Dim(
                        name="time",
                        domain=(t0.to_datetime64(), t1.to_datetime64()),
                        tile=5,
                        dtype="datetime64[ns]",
                    ),
                    tiledb.Dim(
                        name="double_range",
                        domain=(-1000.0, 1000.0),
                        tile=1000,
                        dtype="float64",
                    ),
                ]
            ),
            attrs=[tiledb.Attr(name="int_vals", dtype="int64", filters=attrs_filters)],
            coords_filters=coords_filters,
            cell_order="row-major",
            tile_order="row-major",
            capacity=1001,
            sparse=True,
            allows_duplicates=False,
        )
        # note: filters omitted

        array_nfiles = len(tiledb.VFS().ls(tmp_array))
        self.assertEqual(array_nfiles, 3)

        with tiledb.open(tmp_array) as A:
            self.assertEqual(A.schema, ref_schema)

            # TODO currently no equality check for filters
            self.assertEqual(A.schema.coords_filters[0].level, coords_filters[0].level)
            self.assertEqual(A.schema.attr(0).filters[0].level, attrs_filters[0].level)

        # Test mode='append' for from_csv
        tiledb.from_csv(tmp_array, tmp_csv, mode="append", row_start_idx=0)
        df2 = make_dataframe_basic3(10, time_range=(t0, t1))
        df2.sort_values("time", inplace=True)
        df2.set_index(["time", "double_range"], inplace=True)

        # Test mode='append' for from_pandas
        tiledb.from_pandas(tmp_array, df2, row_start_idx=len(df2), mode="append")

        with tiledb.open(tmp_array) as A:
            df_bk = A.df[:]

            df.set_index(["time", "double_range"], inplace=True)
            df_combined = pd.concat([df, df2])
            df_combined.sort_index(level="time", inplace=True)
            df_bk.sort_index(level="time", inplace=True)
            tm.assert_frame_equal(df_bk, df_combined)

    def test_dataframe_csv_chunked(self):
        col_size = 200
        df = make_dataframe_basic3(col_size)

        tmp_dir = self.path("csv_chunked")
        self.vfs.create_dir(tmp_dir)
        tmp_csv = os.path.join(tmp_dir, "generated.csv")

        df.sort_values("time", inplace=True)

        with tiledb.FileIO(self.vfs, tmp_csv, "wb") as fio:
            df.to_csv(fio, index=False)

        # Test sparse chunked
        tmp_array = os.path.join(tmp_dir, "array")
        tiledb.from_csv(
            tmp_array,
            tmp_csv,
            index_col=["double_range"],
            parse_dates=["time"],
            date_spec={"time": "%Y-%m-%dT%H:%M:%S.%f"},
            chunksize=10,
            sparse=True,
        )

        with tiledb.open(tmp_array) as A:
            res = A[:]
            df_bk = pd.DataFrame(res)
            df_bk.set_index(["double_range"], inplace=True)

            df_ck = df.set_index(["double_range"])
            tm.assert_frame_equal(df_bk, df_ck)

        # Test dense chunked
        tmp_array_dense = os.path.join(tmp_dir, "array_dense")
        tiledb.from_csv(
            tmp_array_dense, tmp_csv, parse_dates=["time"], sparse=False, chunksize=25
        )

        with tiledb.open(tmp_array_dense) as A:
            # with sparse=False and no index column, we expect to have unlimited domain
            self.assertEqual(A.schema.domain.dim(0).domain[1], 18446744073709541615)

            # chunked writes go to unlimited domain, so we must only read nonempty
            ned = A.nonempty_domain()[0]
            # TODO should support numpy scalar here
            res = A.multi_index[int(ned[0]) : int(ned[1])]
            df_bk = pd.DataFrame(res)

            tm.assert_frame_equal(df_bk, df)

            # test .df[] indexing
            df_idx_res = A.df[int(ned[0]) : int(ned[1])]
            tm.assert_frame_equal(df_idx_res, df)

            # test .df[] indexing with query
            df_idx_res = A.query(attrs=["time"]).df[int(ned[0]) : int(ned[1])]
            tm.assert_frame_equal(df_idx_res, df[["time"]])

            df_idx_res = A.query(attrs=["double_range"]).df[int(ned[0]) : int(ned[1])]
            tm.assert_frame_equal(df_idx_res, df[["double_range"]])

            # test .df[] indexing with arrow
            df_idx_res = A.query(use_arrow=True, attrs=["time"]).df[
                int(ned[0]) : int(ned[1])
            ]
            tm.assert_frame_equal(df_idx_res, df[["time"]])
            df_idx_res = A.query(use_arrow=True, attrs=["double_range"]).df[
                int(ned[0]) : int(ned[1])
            ]
            tm.assert_frame_equal(df_idx_res, df[["double_range"]])

            # disable coordinate dimension/index
            df_idx_res = A.query(coords=False).df[int(ned[0]) : int(ned[1])]
            tm.assert_frame_equal(df_idx_res, df.reset_index(drop=True))

    def test_csv_fillna(self):
        if pytest.tiledb_vfs == "s3":
            pytest.skip(
                "TODO need more plumbing to make pandas use TileDB VFS to read CSV files"
            )

        def check_array(path, df):
            # update the value in the original dataframe to match what we expect on read-back
            df["v"][4] = 0

            df_bk = tiledb.open_dataframe(path)
            tm.assert_frame_equal(df_bk, df)

        ### Test 1
        col_size = 10
        data = np.random.rand(col_size) * 100  # make some integers for the 2nd test
        data[4] = np.nan
        df = pd.DataFrame({"v": data})

        tmp_dir = self.path("csv_fillna")
        self.vfs.create_dir(tmp_dir)
        tmp_csv = os.path.join(tmp_dir, "generated.csv")

        with tiledb.FileIO(self.vfs, tmp_csv, "wb") as fio:
            df.to_csv(fio, index=False, na_rep="NaN")

        tmp_array = os.path.join(tmp_dir, "array")
        # TODO: test Dense too
        tiledb.from_csv(tmp_array, tmp_csv, fillna={"v": 0}, sparse=True)
        check_array(tmp_array, copy.deepcopy(df))

        ### Test 2
        # Test roundtrip a Int64Dtype in newer pandas versions
        tmp_csv2 = os.path.join(tmp_dir, "generated.csv")
        df2 = pd.DataFrame({"v": pd.Series(np.int64(df["v"]), dtype=pd.Int64Dtype())})
        df2["v"][4] = None

        with tiledb.FileIO(self.vfs, tmp_csv2, "wb") as fio:
            df2.to_csv(fio, index=False)

        if hasattr(pd, "Int64Dtype"):
            tmp_array2 = os.path.join(tmp_dir, "array2")
            tiledb.from_csv(
                tmp_array2,
                tmp_csv2,
                fillna={"v": 0},
                column_types={"v": pd.Int64Dtype()},
                sparse=True,
            )
            df_to_check = copy.deepcopy(df2)
            check_array(tmp_array2, df_to_check)

        col_size = 10

        csv_dir = self.path("csv_multi_dir")
        self.vfs.create_dir(csv_dir)

        # Write a set of CSVs with 10 rows each
        input_dfs = list()
        for i in range(20):
            df = make_dataframe_basic3(col_size)
            output_path = os.path.join(csv_dir, "csv_{}.csv".format(i))

            with tiledb.FileIO(self.vfs, output_path, "wb") as fio:
                df.to_csv(fio, index=False)
            input_dfs.append(df)

        tmp_dir = self.path("csv_multi_array_dir")
        self.vfs.create_dir(tmp_dir)

        # Create TileDB array with flush every 25 rows
        csv_paths = list(filter(lambda x: x.endswith(".csv"), self.vfs.ls(csv_dir)))
        tmp_array = os.path.join(tmp_dir, "array")
        # Note: this test is skipped when running on S3 because the from_csv call
        #       below uses pandas to read the CSV, but we do not currently have a
        #       hook in place for pandas to be able to read via TileDB VFS.
        tiledb.from_csv(
            tmp_array,
            csv_paths,
            index_col=["time"],
            parse_dates=["time"],
            chunksize=25,
            sparse=True,
        )

        # Check number of fragments
        # * should equal 8 based on chunksize=25
        # * 20 files, 10 rows each, 200 rows == 8 writes:
        fragments = glob.glob(tmp_array + "/*.ok")
        self.assertEqual(len(fragments), 8)

        # Check the returned data
        # note: tiledb returns sorted values
        df_orig = pd.concat(input_dfs, axis=0).set_index(["time"]).sort_values("time")

        with tiledb.open(tmp_array) as A:
            df_bk = A.df[:]
            # TileDB default return is unordered, so sort to compare
            df_bk = df_bk.sort_index()

            tm.assert_frame_equal(df_bk, df_orig)

    def test_dataframe_misc(self):
        uri = self.path("test_small_domain_range")
        df = pd.DataFrame({"data": [2]}, index=[0])
        tiledb.from_pandas(uri, df)

        data = {
            "data": np.array([1, 2, 3]),
            "raw": np.array([4, 5, 6]),
            "index": np.array(["a", "b", "c"], dtype=np.dtype("|S")),
            "indey": np.array([0.0, 0.5, 0.9]),
        }
        df = pd.DataFrame.from_dict(data)
        df = df.set_index(["index", "indey"])
        uri = self.path("test_string_index_infer")
        tiledb.from_pandas(uri, df)
        with tiledb.open(uri) as A:
            self.assertTrue(A.schema.domain.dim(0).dtype == np.dtype("|S"))

        # test setting Attr and Dim filter list by override
        uri = self.path("test_df_attrs_filters1")
        bz_filter = [tiledb.Bzip2Filter(4)]
        def_filter = [tiledb.GzipFilter(-1)]
        tiledb.from_pandas(uri, df, attr_filters=bz_filter, dim_filters=bz_filter)
        with tiledb.open(uri) as A:
            self.assertTrue(A.schema.attr("data").filters == bz_filter)
            self.assertTrue(A.schema.attr("raw").filters == bz_filter)
            self.assertTrue(A.schema.domain.dim("index").filters == bz_filter)
            self.assertTrue(A.schema.domain.dim("indey").filters == bz_filter)

        # test setting Attr and Dim filter list by dict
        uri = self.path("test_df_attrs_filters2")
        tiledb.from_pandas(
            uri, df, attr_filters={"data": bz_filter}, dim_filters={"index": bz_filter}
        )
        with tiledb.open(uri) as A:
            self.assertTrue(A.schema.attr("data").filters == bz_filter)
            self.assertTrue(A.schema.attr("raw").filters == tiledb.FilterList())
            self.assertTrue(A.schema.domain.dim("index").filters == bz_filter)
            self.assertTrue(A.schema.domain.dim("indey").filters == tiledb.FilterList())

    def test_dataframe_query(self):
        uri = self.path("df_query")

        col_size = 10
        df = make_dataframe_basic3(col_size)
        df.set_index(["time"], inplace=True)

        tiledb.from_pandas(uri, df, sparse=True)

        with tiledb.open(uri) as A:
            with self.assertRaises(tiledb.TileDBError):
                A.query(dims=["nodimnodim"])
            with self.assertRaises(tiledb.TileDBError):
                A.query(attrs=["noattrnoattr"])

            res_df = A.query(dims=["time"], attrs=["int_vals"]).df[:]
            self.assertTrue("time" == res_df.index.name)
            self.assertTrue("int_vals" in res_df)
            self.assertTrue("double_range" not in res_df)

            # try index_col alone: should have *only* the default RangeIndex column
            res_df2 = A.query(index_col=None).df[:]
            self.assertTrue(isinstance(res_df2.index, pd.RangeIndex))

            # try no dims, index_col None: should only value cols and default index
            res_df3 = A.query(dims=False, index_col=None).df[:]
            self.assertTrue("time" not in res_df3)
            self.assertTrue("int_vals" in res_df3)
            self.assertTrue("double_range" in res_df3)
            self.assertTrue(isinstance(res_df3.index, pd.RangeIndex))

            # try attr as index_col:
            res_df4 = A.query(dims=False, index_col=["int_vals"]).df[:]
            self.assertTrue("time" not in res_df4)
            self.assertTrue("double_range" in res_df4)
            self.assertTrue("int_vals" == res_df4.index.name)

    def test_read_parquet(self):
        # skip: because to_parquet is erroring out with FileIO object
        if pytest.tiledb_vfs == "s3":
            pytest.skip(
                "TODO need more plumbing to make pandas use TileDB VFS to read CSV files"
            )

        uri = self.path("test_read_parquet")
        self.vfs.create_dir(uri)

        def try_rt(name, df, pq_args={}):
            tdb_uri = os.path.join(uri, f"{name}.tdb")
            pq_uri = os.path.join(uri, f"{name}.pq")

            df.to_parquet(
                pq_uri,
                # this is required to losslessly serialize timestamps
                # until Parquet 2.0 is default.
                use_deprecated_int96_timestamps=True,
                **pq_args,
            )

            tiledb.from_parquet(str(tdb_uri), str(pq_uri))
            df_bk = tiledb.open_dataframe(tdb_uri)
            tm.assert_frame_equal(df, df_bk)

        basic1 = make_dataframe_basic1()
        try_rt("basic1", basic1)

        try_rt("basic2", make_dataframe_basic2())

        basic3 = make_dataframe_basic3()
        try_rt("basic3", basic3)

    def test_nullable_integers(self):
        nullable_int_dtypes = (
            pd.Int64Dtype(),
            pd.Int32Dtype(),
            pd.Int16Dtype(),
            pd.Int8Dtype(),
            pd.UInt64Dtype(),
            pd.UInt32Dtype(),
            pd.UInt16Dtype(),
            pd.UInt8Dtype(),
        )

        col_size = 100
        null_count = 20
        for pdtype in nullable_int_dtypes:
            uri = self.path(f"test_nullable_{str(pdtype)}")
            nptype = pdtype.numpy_dtype

            data = np.random.randint(
                dtype_max(nptype), size=col_size, dtype=nptype
            ).astype("O")
            null_idxs = np.random.randint(col_size, size=null_count)
            data[null_idxs] = None

            series = pd.Series(data, dtype=pdtype)
            df = pd.DataFrame({"data": series})

            tiledb.from_pandas(uri, df)

            df_bk = tiledb.open_dataframe(uri)
            tm.assert_frame_equal(df, df_bk)

    def test_nullable_bool(self):
        uri = self.path("test_nullable_bool")
        col_size = 100
        null_count = 20

        data = np.random.randint(2, size=col_size, dtype=np.uint8).astype("O")
        null_idxs = np.random.randint(col_size, size=null_count)
        data[null_idxs] = None

        series = pd.Series(data, dtype="boolean")
        df = pd.DataFrame({"data": series})

        tiledb.from_pandas(uri, df)

        df_bk = tiledb.open_dataframe(uri)
        tm.assert_frame_equal(df, df_bk)

    def test_var_length(self):
        from tiledb.tests.datatypes import RaggedDtype

        dtype = np.dtype("uint16")
        data = np.empty(100, dtype="O")
        data[:] = [
            np.random.randint(1000, size=np.random.randint(1, 10), dtype=dtype)
            for _ in range(len(data))
        ]

        df = pd.DataFrame({"data": data})

        uri = self.path("test_var_length")
        data_dtype = RaggedDtype(dtype)
        tiledb.from_pandas(
            uri, df, column_types={"data": data_dtype}, varlen_types={data_dtype}
        )

        with tiledb.open(uri) as A:
            # TODO: update the test when we support Arrow lists
            import pyarrow

            with pytest.raises(pyarrow.lib.ArrowInvalid):
                A.df[:]

            df2 = A.query(use_arrow=False).df[:]
            tm.assert_frame_equal(df, df2, check_dtype=False)
            for array1, array2 in zip(df["data"].values, df2["data"].values):
                self.assertEqual(array1.dtype, array2.dtype)
                np.testing.assert_array_equal(array1, array2)

    def test_incomplete_df(self):
        ncells = 1000
        null_count = round(0.56 * ncells)

        path = self.path("incomplete_sparse_varlen")

        validity_idx = np.random.randint(ncells, size=null_count)
        data = np.array(
            np.random.randint(0, 10e10, size=ncells, dtype=np.int64), dtype="O"
        )
        data[validity_idx] = None

        # TODO - not supported
        # str_data = np.array([rand_utf8(random.randint(0, n)) for n in range(ncells)],
        #                dtype=np.unicode_)
        # str_data[validity_idx] = None

        df = pd.DataFrame({"int64": pd.Series(data, dtype=pd.Int64Dtype())})

        tiledb.from_pandas(path, df, sparse=True)

        init_buffer_bytes = 512
        config = tiledb.Config(
            {
                "py.init_buffer_bytes": init_buffer_bytes,
                "py.exact_init_buffer_bytes": "true",
            }
        )
        self.assertEqual(config["py.init_buffer_bytes"], str(init_buffer_bytes))

        with tiledb.SparseArray(path, mode="r", ctx=tiledb.Ctx(config)) as T2:
            tm.assert_frame_equal(df, T2.df[:])

    def test_int_column_names(self):
        uri = self.path("test_int_column_names")
        data = np.random.rand(10_000, 100)
        df = pd.DataFrame(data)
        tiledb.from_pandas(uri, df)


class TestFromPandasOptions(DiskTestCase):
    def test_filters_options(self):
        def assert_filters_eq(left, right):
            # helper to check equality:
            # - None should produce empty FilterList
            # - a dict should match the first and only key/None
            if isinstance(left, dict):
                assert len(left) == 1
                left = list(left.values())[0]
            if isinstance(right, dict):
                assert len(right) == 1
                right = list(right.values())[0]

            left = (
                tiledb.FilterList([left]) if isinstance(left, tiledb.Filter) else left
            )
            right = (
                tiledb.FilterList([right]) if isinstance(right, tiledb.Filter) else left
            )

            if left is None:
                left = tiledb.FilterList()
            if right is None:
                right = tiledb.FilterList([right])

            assert left == right

        df_orig = pd.DataFrame({"x": pd.Series([1, 2, 3])})
        df_orig.index.name = "d"

        filters_to_check = [
            tiledb.ZstdFilter(2),
            [tiledb.ZstdFilter(2)],
            None,
            tiledb.FilterList(),
            tiledb.FilterList([tiledb.ZstdFilter(2), tiledb.ZstdFilter(4)]),
            {"d": None},
            {"d": tiledb.FilterList([tiledb.ZstdFilter(3), tiledb.ZstdFilter(5)])},
        ]

        # mapping of options to getters for comparison on read-back
        checks = [
            ("dim_filters", lambda A: A.schema.domain.dim(0).filters),
            ("attr_filters", lambda A: A.schema.attr(0).filters),
            ("coords_filters", lambda A: A.schema.coords_filters),
            ("offsets_filters", lambda A: A.schema.offsets_filters),
        ]

        for opt, getter in checks:
            df = df_orig

            # change the names for attr test expectation
            if opt == "attr_filters":
                df.index.name = ""
                df = df.rename(columns={"x": "d"})

            for f in filters_to_check:
                if opt in ("coords_filters", "offsets_filters") and isinstance(f, dict):
                    continue

                uri = self.path()
                tiledb.from_pandas(uri, df, **{opt: f})
                with tiledb.open(uri) as A:
                    assert_filters_eq(getter(A), f)
