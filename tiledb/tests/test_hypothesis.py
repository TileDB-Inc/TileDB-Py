import hypothesis as hp
import hypothesis.strategies as st
import numpy as np
import pytest

import tiledb

from .common import has_pandas

pd = pytest.importorskip("pandas")
tm = pd._testing


@pytest.mark.skipif(not has_pandas(), reason="pandas>=1.0,<3.0 not installed")
@pytest.mark.parametrize("mode", ["np", "df"])
@hp.settings(deadline=None, verbosity=hp.Verbosity.verbose)
@hp.given(st.binary())
def test_bytes_npdf(checked_path, mode, data):
    uri = "mem://" + checked_path.path()
    array = np.array([data], dtype="S0")

    if mode == "np":
        with tiledb.from_numpy(uri, array) as A:
            pass
    else:
        series = pd.Series(array)
        df = pd.DataFrame({"": series})
        # NOTE: ctx required here for mem://
        tiledb.from_pandas(uri, df, sparse=False, ctx=tiledb.default_ctx())

    with tiledb.open(uri) as A:
        if mode == "np":
            np.testing.assert_array_equal(A.multi_index[:][""], array)
        else:
            tm.assert_frame_equal(A.df[:], df)
