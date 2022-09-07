from math import hypot
import tiledb
import time
import numpy as np
import pytest

pd = pytest.importorskip("pandas")
tm = pd._testing

import hypothesis as hp
import hypothesis.strategies as st
from hypothesis import given, reproduce_failure
from numpy.testing import assert_array_equal

from tiledb.tests.common import DiskTestCase, has_pandas


class AttrDataTest(DiskTestCase):
    @hp.settings(deadline=None, verbosity=hp.Verbosity.verbose)
    @given(st.binary())
    @pytest.mark.parametrize("mode", ["np", "df"])
    def test_bytes_npdf(self, mode, data):
        start = time.time()

        uri = "mem://" + self.path()
        hp.note(f"!!! self.path() '{uri}' time: {time.time() - start}")

        array = np.array([data], dtype="S0")

        start_ingest = time.time()
        if mode == "np":
            with tiledb.from_numpy(uri, array) as A:
                pass
        else:
            series = pd.Series(array)
            df = pd.DataFrame({"": series})
            # NOTE: ctx required here for mem://
            tiledb.from_pandas(uri, df, sparse=False, ctx=tiledb.default_ctx())

        hp.note(f"{mode} ingest time: {time.time() - start_ingest}")

        # DEBUG
        tiledb.stats_enable()
        tiledb.stats_reset()
        # END DEBUG

        with tiledb.open(uri) as A:
            if mode == "np":
                assert_array_equal(A.multi_index[:][""], array)
            else:
                tm.assert_frame_equal(A.df[:], df)

        hp.note(tiledb.stats_dump(print_out=False))

        # DEBUG
        tiledb.stats_disable()

        duration = time.time() - start
        hp.note(f"!!! test_bytes_{mode} duration: {duration}")
        if duration > 2:
            # Hypothesis setup is (maybe) causing deadline exceeded errors
            # https://github.com/TileDB-Inc/TileDB-Py/issues/1194
            # Set deadline=None and use internal timing instead.
            pytest.fail(f"!!! {mode} function body duration exceeded 2s: {duration}")
