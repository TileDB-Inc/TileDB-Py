import unittest, os

import tiledb
from tiledb import TileDBError, core
from tiledb.tests.common import DiskTestCase
import sys
import numpy as np
from numpy.testing import assert_array_equal

class CoreCCTest(DiskTestCase):

    def test_pyquery_basic(self):
        ctx = tiledb.default_ctx()
        uri = self.path("test_pyquery_basic")
        with tiledb.from_numpy(uri, np.random.rand(4)) as A:
            pass

        with tiledb.open(uri) as a:
            q = core.PyQuery(ctx, a, ("",), False, 0)

            try:
                q._test_err("bad foo happened")
            except Exception as exc:
                assert isinstance(exc, tiledb.TileDBError)
                assert exc.message == "bad foo happened"

            q.set_ranges([[(0, 3)]])

            with self.assertRaises(TileDBError):
                q.set_ranges([[(0, 3.0)]])

            q.set_ranges([[(0, np.int32(3))]])

            with self.assertRaises(TileDBError):
                q.set_ranges([[(3, "a")]])

            if sys.hexversion >= 0x3000000:
                # assertRaisesRegex is not available in 2.7
                with self.assertRaisesRegex(
                    TileDBError,
                    "Failed to cast dim range '\\(1.2344, 5.6789\\)' to dim type UINT64.*$",
                ):
                    q.set_ranges([[(1.2344, 5.6789)]])

                with self.assertRaisesRegex(
                    TileDBError,
                    "Failed to cast dim range '\\('aa', 'bbbb'\\)' to dim type UINT64.*$",
                ):
                    q.set_ranges([[("aa", "bbbb")]])

        with tiledb.open(uri) as a:
            q2 = core.PyQuery(ctx, a, ("",), False, 0)

            q2.set_ranges([[(0,3)]])
            q2.submit()
            res = q2.results()[''][0]
            res.dtype = np.double
            assert_array_equal(res, a[:])

    def test_pyquery_init(self):
        uri = self.path("test_pyquery_basic")
        intmax = np.iinfo(np.int64).max
        config_dict = {
            "sm.tile_cache_size": '100',
            "py.init_buffer_bytes": str(intmax)
            }
        ctx = tiledb.Ctx(config=config_dict)

        with tiledb.from_numpy(uri, np.random.rand(4)) as A:
            pass

        with tiledb.open(uri) as a:
            q = core.PyQuery(ctx, a, ("",), False, 0)
            self.assertEqual(q._test_init_buffer_bytes, intmax)