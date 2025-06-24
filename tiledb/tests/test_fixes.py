import concurrent
import concurrent.futures
import json
import os
import subprocess
import sys

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import tiledb

from .common import DiskTestCase, has_pandas, has_pyarrow

pd = pytest.importorskip("pandas")
tm = pd._testing


class FixesTest(DiskTestCase):
    def test_sc50378_overflowerror_python_int_too_large_to_convert_to_c_long(self):
        uri = self.path(
            "test_sc50378_overflowerror_python_int_too_large_to_convert_to_c_long"
        )
        MAX_UINT64 = np.iinfo(np.uint64).max
        dim = tiledb.Dim(
            name="id",
            domain=(0, MAX_UINT64 - 1),
            dtype=np.dtype(np.uint64),
        )
        dom = tiledb.Domain(dim)
        text_attr = tiledb.Attr(name="text", dtype=np.dtype("U1"), var=True)
        attrs = [text_attr]
        schema = tiledb.ArraySchema(
            domain=dom,
            sparse=True,
            allows_duplicates=False,
            attrs=attrs,
        )
        tiledb.Array.create(uri, schema)

        with tiledb.open(uri, "w") as A:
            external_ids = np.array([0, 100, MAX_UINT64 - 1], dtype=np.dtype(np.uint64))
            data = {"text": np.array(["foo", "bar", "baz"], dtype="<U3")}
            A[external_ids] = data

        array = tiledb.open(uri, "r", timestamp=None, config=None)
        array[0]["text"][0]
        array[100]["text"][0]
        # This used to fail
        array[MAX_UINT64 - 1]["text"][0]

    def test_ch7727_float32_dim_estimate_incorrect(self):
        # set max allocation: because windows won't overallocate
        with tiledb.scope_ctx({"py.alloc_max_bytes": 1024**2 * 100}):
            uri = self.path()
            dom = tiledb.Domain(tiledb.Dim("x", domain=(1, 100), dtype=np.float32))
            att = tiledb.Attr("", dtype=np.bytes_)
            schema = tiledb.ArraySchema(domain=dom, attrs=(att,), sparse=True)
            tiledb.Array.create(uri, schema)

            with tiledb.open(uri, mode="w") as T:
                T[50.4] = b"hello"

            with tiledb.open(uri, mode="r") as T:
                assert T[:][""] == b"hello"
                assert T[50.4][""] == b"hello"

    def test_ch8292(self):
        # test fix for ch8292
        # We need to ensure that py.alloc_max_bytes is *not* applied to
        # dense arrays. Dense arrays should have exact estimates based
        # on the ranges, so there should be no risk of over-estimates.
        # This test sets py.alloc_max_bytes to 1 less than the expected
        # result array size, and asserts that the allocated buffers match
        # the expected result size rather than py.alloc_max_bytes.
        uri = self.path()
        max_val = np.iinfo(np.uint8).max
        with tiledb.from_numpy(uri, np.uint8(range(max_val))):
            pass
        with tiledb.scope_ctx(
            {"py.init_buffer_bytes": 2 * 1024**2, "py.alloc_max_bytes": 1024**2}
        ) as ctx3:
            with tiledb.open(uri) as b:
                q = tiledb.main.PyQuery(ctx3, b, ("",), (), 0, False)
                q._return_incomplete = True
                subarray = tiledb.Subarray(b, ctx3)
                subarray.add_ranges([[(0, max_val - 1)]])
                q.set_subarray(subarray)
                q._allocate_buffers()
                buffers = list(*q._get_buffers().values())
                assert buffers[0].nbytes == max_val

    @pytest.mark.skipif(not has_pandas(), reason="pandas>=1.0,<3.0 not installed")
    def test_ch10282_concurrent_multi_index(self):
        """Test concurrent access to a single tiledb.Array using
        Array.multi_index and Array.df. We pass an array and slice
        into a function run by a set of futures, along with expected
        result; then assert that the result from TileDB matches the
        expectation.
        """

        def slice_array(a: tiledb.Array, indexer, selection, expected):
            """Helper function to slice a given tiledb.Array with an indexer
            and assert that the selection matches the expected result."""
            res = getattr(a, indexer)[selection][""]
            if indexer == "df":
                res = res.values

            assert_array_equal(res, expected)

        uri = self.path()

        data = np.random.rand(100)
        with tiledb.from_numpy(uri, data):
            pass

        futures = []
        with tiledb.open(uri) as A:
            with concurrent.futures.ThreadPoolExecutor(10) as executor:
                for indexer in ["multi_index", "df"]:  #
                    for end_idx in range(1, 100, 5):
                        sel = slice(0, end_idx)
                        expected = data[sel.start : sel.stop + 1]
                        futures.append(
                            executor.submit(slice_array, A, indexer, sel, expected)
                        )

                concurrent.futures.wait(futures)

            # Important: must get each result here or else assertion
            # failures or exceptions will disappear.
            list(map(lambda x: x.result(), futures))

    # skip, does not currently work, because we cannot force use
    # of the memory estimate
    @pytest.mark.skip
    def test_sc16301_arrow_extra_estimate_dense(self):
        """
        Test that dense query of array with var-length attribute completes
        in one try. We are currently adding an extra element to the offset
        estimate from libtiledb, in order to avoid an unnecessary pair of
        query resubmits when the offsets won't fit in the estimated buffer.
        """

        uri = self.path("test_sc16301_arrow_extra_estimate_dense")

        dim1 = tiledb.Dim(name="d1", dtype="int64", domain=(1, 3))
        att = tiledb.Attr(name="a1", dtype="<U0", var=True)

        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(dim1),
            attrs=(att,),
            sparse=False,
            allows_duplicates=False,
        )
        tiledb.Array.create(uri, schema)

        with tiledb.open(uri, "w") as A:
            A[:] = np.array(["aaa", "bb", "c"])

        with tiledb.open(uri) as A:
            tiledb.stats_enable()
            A[:]

            stats_dump_str = tiledb.stats_dump(print_out=False)
            if tiledb.libtiledb.version() >= (2, 27):
                assert """"Context.Query.Reader.loop_num": 1""" in stats_dump_str
            else:
                assert (
                    """"Context.StorageManager.Query.Reader.loop_num": 1"""
                    in stats_dump_str
                )
            tiledb.stats_disable()

    def test_sc58286_fix_stats_dump_return_value_broken(self):
        uri = self.path("test_sc58286_fix_stats_dump_return_value_broken")
        dim1 = tiledb.Dim(name="d1", dtype="int64", domain=(1, 3))
        att = tiledb.Attr(name="a1", dtype="<U0", var=True)

        schema = tiledb.ArraySchema(
            domain=tiledb.Domain(dim1),
            attrs=(att,),
            sparse=False,
            allows_duplicates=False,
        )
        tiledb.Array.create(uri, schema)

        with tiledb.open(uri, "w") as A:
            A[:] = np.array(["aaa", "bb", "c"])

        with tiledb.open(uri) as A:
            tiledb.stats_enable()
            A[:]

            # check that the stats cannot be parsed as json
            stats = tiledb.stats_dump(print_out=False, json=False)
            assert isinstance(stats, str)
            with pytest.raises(json.decoder.JSONDecodeError):
                json.loads(stats)

            stats = tiledb.stats_dump(print_out=False, json=False, include_python=True)
            assert isinstance(stats, str)
            with pytest.raises(json.decoder.JSONDecodeError):
                json.loads(stats)

            # check that the stats can be parsed as json
            stats = tiledb.stats_dump(print_out=False, json=True)
            assert isinstance(stats, str)
            json.loads(stats)

            stats = tiledb.stats_dump(print_out=False, json=True, include_python=True)
            assert isinstance(stats, str)
            res = json.loads(stats)

            tiledb.stats_disable()

            # check that some fields are present in the json output and are of the correct type
            assert "counters" in res and isinstance(res["counters"], dict)
            assert "timers" in res and isinstance(res["timers"], dict)
            assert "python" in res and isinstance(res["python"], dict)

    def test_fix_stats_error_messages(self):
        # Test that stats_dump prints a user-friendly error message when stats are not enabled
        with pytest.raises(tiledb.TileDBError) as exc:
            tiledb.stats_dump()
        assert "Statistics are not enabled. Call tiledb.stats_enable() first." in str(
            exc.value
        )

    @pytest.mark.skipif(
        not has_pandas() and has_pyarrow(),
        reason="pandas>=1.0,<3.0 or pyarrow>=1.0 not installed",
    )
    def test_py1078_df_all_empty_strings(self):
        uri = self.path()
        df = pd.DataFrame(
            index=np.arange(10).astype(str),
            data={
                "A": list(str(i) for i in range(10)),
                "B": list("" for i in range(10)),
            },
        )

        tiledb.from_pandas(uri, df)
        with tiledb.open(uri) as arr:
            tm.assert_frame_equal(arr.df[:], df)

    @pytest.mark.skipif(
        tiledb.libtiledb.version() < (2, 14, 0),
        reason="SC-23287 fix not implemented until libtiledb 2.14",
    )
    @pytest.mark.skipif(
        sys.platform == "win32",
        reason="TODO does not run on windows due to env passthrough",
    )
    def test_sc23827_aws_region(self):
        # Test for SC-23287
        # The expected behavior here for `vfs.s3.region` is:
        # - default to '' if no environment variables are set
        # - empty if AWS_REGION or AWS_DEFAULT_REGION is set (to any value)

        def get_config_with_env(env, key):
            python_exe = sys.executable
            cmd = """import tiledb; print(tiledb.Config()[\"{}\"])""".format(key)
            test_path = os.path.dirname(os.path.abspath(__file__))

            sp_output = subprocess.check_output(
                [python_exe, "-c", cmd], cwd=test_path, env=env
            )
            return sp_output.decode("UTF-8").strip()

        if tiledb.libtiledb.version() >= (2, 27, 0):
            assert get_config_with_env({}, "vfs.s3.region") == ""
        else:
            assert get_config_with_env({}, "vfs.s3.region") == "us-east-1"
        assert get_config_with_env({"AWS_DEFAULT_REGION": ""}, "vfs.s3.region") == ""
        assert get_config_with_env({"AWS_REGION": ""}, "vfs.s3.region") == ""

    @pytest.mark.skipif(not has_pandas(), reason="pandas>=1.0,<3.0 not installed")
    @pytest.mark.parametrize("is_sparse", [True, False])
    def test_sc1430_nonexisting_timestamp(self, is_sparse):
        path = self.path("nonexisting_timestamp")

        if is_sparse:
            tiledb.from_pandas(
                path, pd.DataFrame({"a": np.random.rand(4)}), sparse=True
            )

            with tiledb.open(path, timestamp=1) as A:
                assert pd.DataFrame.equals(
                    A.df[:]["a"], pd.Series([], dtype=np.float64)
                )
        else:
            with tiledb.from_numpy(path, np.random.rand(4)) as A:
                pass

            with tiledb.open(path, timestamp=1) as A:
                assert_array_equal(A[:], np.ones(4) * np.nan)

    def test_sc27374_hilbert_default_tile_order(self):
        import os
        import shutil

        import tiledb

        uri = "repro"
        if os.path.exists(uri):
            shutil.rmtree(uri)

        dom = tiledb.Domain(
            tiledb.Dim(
                name="var_id",
                domain=(None, None),
                dtype="ascii",
                filters=[tiledb.ZstdFilter(level=1)],
            ),
        )

        attrs = []

        sch = tiledb.ArraySchema(
            domain=dom,
            attrs=attrs,
            sparse=True,
            allows_duplicates=False,
            offsets_filters=[
                tiledb.DoubleDeltaFilter(),
                tiledb.BitWidthReductionFilter(),
                tiledb.ZstdFilter(),
            ],
            capacity=1000,
            cell_order="hilbert",
            tile_order=None,  # <-------------------- note
        )

        tiledb.Array.create(uri, sch)

        with tiledb.open(uri) as A:
            assert A.schema.cell_order == "hilbert"
            assert A.schema.tile_order is None

    def test_sc43221(self):
        # GroupMeta object did not have a representation test; repr failed due to non-existent attribute access in check.
        tiledb.Group.create("mem://tmp1")
        a = tiledb.Group("mem://tmp1")
        repr(a.meta)

    def test_sc56611(self):
        # test from_numpy with sparse argument set to True
        uri = self.path("test_sc56611")
        data = np.random.rand(10, 10)
        with pytest.raises(tiledb.libtiledb.TileDBError) as exc_info:
            tiledb.from_numpy(uri, data, sparse=True)
        assert str(exc_info.value) == "from_numpy only supports dense arrays"

    @pytest.mark.parametrize(
        "array_data",
        [
            np.array([b"", b"testing", b"", b"with empty", b"bytes"], dtype="S"),
            np.array([b"and", b"\0\0", b"again"], dtype="S"),
            np.array(
                [b"", b"and with", b"the last one", b"", b"emtpy", b""], dtype="S"
            ),
        ],
    )
    def test_sc62594_buffer_resize(self, array_data):
        uri = self.path("test_agis")
        dom = tiledb.Domain(
            tiledb.Dim(name="dim", domain=(0, len(array_data) - 1), dtype=np.int64)
        )

        schema = tiledb.ArraySchema(
            domain=dom, sparse=False, attrs=[tiledb.Attr(name="a", dtype="S", var=True)]
        )

        tiledb.DenseArray.create(uri, schema)

        with tiledb.DenseArray(uri, mode="w") as T:
            T[...] = array_data

        with tiledb.DenseArray(uri) as T:
            assert_array_equal(array_data, T)

    def test_sc_64885_ctx_reference_lost(self):
        uri = self.path("test_sc_64885_ctx_reference_lost")
        config = tiledb.Config()

        enmr = tiledb.Enumeration("e", True, dtype="int")
        attrs = [tiledb.Attr(name="a", dtype=int, enum_label="e")]
        domain = tiledb.Domain(tiledb.Dim(domain=(0, 3), dtype=np.uint64))
        schema = tiledb.ArraySchema(domain=domain, attrs=attrs, enums=[enmr])
        tiledb.Array.create(uri, schema, ctx=tiledb.Ctx(config=config.dict()))

        se = tiledb.ArraySchemaEvolution(ctx=tiledb.Ctx(config=config.dict()))
        data = [1, 2, 3, 4]
        updated_enmr = enmr.extend(data)
        # this used to fail with a "tiledb.libtiledb.TileDBError: error retrieving error object from ctx"
        se.extend_enumeration(updated_enmr)


class SOMA919Test(DiskTestCase):
    """
    ORIGINAL CONTEXT:
    https://github.com/single-cell-data/TileDB-SOMA/issues/919
    https://gist.github.com/atolopko-czi/26683305258a9f77a57ccc364916338f

    We've distilled @atolopko-czi's gist example using the TileDB-Py API directly.
    """

    def run_test(self):
        import tempfile

        import numpy as np

        import tiledb

        root_uri = tempfile.mkdtemp()
        group_ctx100 = tiledb.Ctx()
        timestamp = None

        # create the group and add a dummy subgroup "causes_bug"
        tiledb.Group.create(root_uri, ctx=group_ctx100)
        with tiledb.Group(root_uri, "w", ctx=group_ctx100) as expt:
            tiledb.Group.create(root_uri + "/causes_bug", ctx=group_ctx100)
            expt.add(name="causes_bug", uri=root_uri + "/causes_bug")

        # add an array to the group (in a separate write operation)
        with tiledb.Group(root_uri, mode="w", ctx=group_ctx100) as expt:
            df_path = os.path.join(root_uri, "df")
            tiledb.from_numpy(df_path, np.ones((100, 100)), timestamp=timestamp)
            expt.add(name="df", uri=df_path)

        # check our view of the group at current time;
        # (previously, "df" is sometimes missing (non-deterministic)
        with tiledb.Group(root_uri) as expt:
            assert "df" in expt

        # IMPORTANT: commenting out either line 29 or 32 (individually) makes df always visible.
        # That is, to invite the bug we must BOTH add the causes_bug sibling element AND then reopen
        # the group write handle to add df. The separate reopen (line 32) simulates
        # tiledbsoma.tdb_handles.Wrapper._flush_hack().

    @pytest.mark.skipif(
        tiledb.libtiledb.version() < (2, 15, 0),
        reason="SOMA919 fix implemented in libtiledb 2.15",
    )
    def test_soma919(self):
        N = 100
        fails = 0
        for i in range(N):
            try:
                self.run_test()
            except AssertionError:
                fails += 1
        if fails > 0:
            pytest.fail(f"SOMA919 test, failure rate {100*fails/N}%")
