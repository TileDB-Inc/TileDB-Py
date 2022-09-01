from math import hypot
import tiledb
import time
import numpy as np
import pytest

pd = pytest.importorskip("pandas")
tm = pd._testing

import hypothesis
import hypothesis.strategies as st
from hypothesis import given, reproduce_failure
from numpy.testing import assert_array_equal

from tiledb.tests.common import DiskTestCase, has_pandas


def pytest_configure(config):
    # Try to work around `https://github.com/HypothesisWorks/hypothesis/issues/2108` and the like
    # Setting up to generate the strategy can be slow.
    @given(st.binary())
    def foo(x):
        pass

    foo()
    return


class AttrDataTest(DiskTestCase):
    @hypothesis.settings(deadline=1000)
    @given(st.binary())
    def test_bytes_numpy(self, data):
        start = time.time()
        # TODO this test is slow. might be nice to run with in-memory
        #      VFS (if faster) but need to figure out correct setup
        # uri = "mem://" + str(uri_int)

        uri = "mem://" + self.path()

        array = np.array([data], dtype="S0")

        start_fnp = time.time()
        with tiledb.from_numpy(uri, array) as A:
            pass
        fnp_time = time.time() - start_fnp
        hypothesis.note(f"from_numpy time: {fnp_time}")

        # DEBUG
        tiledb.stats_enable()
        tiledb.stats_reset()
        # END DEBUG

        with tiledb.open(uri) as A:
            assert_array_equal(A.multi_index[:][""], array)

        hypothesis.note(tiledb.stats_dump(print_out=False))

        # DEBUG
        tiledb.stats_disable()
        hypothesis.note(f"test_bytes_numpy time: {start - time.time()}")

    @pytest.mark.skipif(not has_pandas(), reason="pandas not installed")
    @hypothesis.settings(deadline=1000)
    @given(st.binary())
    def test_bytes_df(self, data):
        start = time.time()

        # TODO this test is slow. might be nice to run with in-memory
        #      VFS (if faster) but need to figure out correct setup
        # uri = "mem://" + str(uri_int)

        uri_df = self.path()

        array = np.array([data], dtype="S0")

        series = pd.Series(array)
        df = pd.DataFrame({"": series})

        start_fpd = time.time()
        tiledb.from_pandas(uri_df, df, sparse=False)
        fpd_time = time.time() - start_fpd
        hypothesis.note(f"from_pandas time: {fpd_time}")

        # DEBUG
        tiledb.stats_enable()
        tiledb.stats_reset()
        # END DEBUG

        with tiledb.open(uri_df) as A:
            tm.assert_frame_equal(A.df[:], df)

        hypothesis.note(tiledb.stats_dump(print_out=False))

        # DEBUG
        tiledb.stats_disable()
        hypothesis.note(f"test_bytes_df time: {start - time.time()}")
