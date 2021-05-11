import tiledb
import numpy as np
import pandas as pd
import pandas._testing as tm

import hypothesis
import hypothesis.strategies as st
from hypothesis import given
from numpy.testing import assert_array_equal

from tiledb.tests.common import DiskTestCase


class AttrDataTest(DiskTestCase):
    @hypothesis.settings(deadline=1000)
    @given(st.binary())
    def test_bytes(self, data):
        # TODO this test is slow. might be nice to run with in-memory
        #      VFS (if faster) but need to figure out correct setup
        # uri = "mem://" + str(uri_int)

        uri = self.path()
        uri_df = self.path()

        if data == b"" or data.count(b"\x00") == len(data):
            # single-cell empty writes are not supported; TileDB PR 1646
            array = np.array([data, b"1"], dtype="S0")
        else:
            array = np.array([data], dtype="S0")

        with tiledb.from_numpy(uri, array) as A:
            pass

        with tiledb.open(uri) as A:
            assert_array_equal(A.multi_index[:][""], array)

        series = pd.Series(array)
        df = pd.DataFrame({"": series})
        tiledb.from_pandas(uri_df, df, sparse=False)

        with tiledb.open(uri_df) as A:
            tm.assert_frame_equal(A.df[:], df)
