import unittest, os

import tiledb
import numpy as np
from tiledb import TileDBError, core

uri = "/tmp/axxxa1"

ctx = tiledb.default_ctx()

if not os.path.isdir(uri):
    tiledb.from_numpy(uri, np.random.rand(4))

class BasicTest(unittest.TestCase):
    def __init__(self):
        pass

    def test(self):
        a = tiledb.open(uri)
        self.r = core.PyQuery(ctx, a, ("",), False)

        try:
            self.r.test_err("foobar")
        except Exception as exc:
            assert isinstance(exc, tiledb.TileDBError)
            assert exc.message == "foobar"

        self.r.set_ranges([[(0, 3)]])

        with self.assertRaises(TileDBError):
            self.r.set_ranges([[(0, 3.0)]])

        self.r.set_ranges([[(0, np.int32(3))]])

        with self.assertRaises(TileDBError):
            self.r.set_ranges([[(3, "a")]])

        with self.assertRaisesRegex(
            TileDBError,
            "Failed to cast dim range '\(1.2344, 5.6789\)' to dim type UINT64.*$",
        ):
            self.r.set_ranges([[(1.2344, 5.6789)]])

        with self.assertRaisesRegex(
            TileDBError,
            "Failed to cast dim range '\('aa', 'bbbb'\)' to dim type UINT64.*$",
        ):
            self.r.set_ranges([[("aa", "bbbb")]])

        print("done")

    def test_read(self):
        a = tiledb.open(uri)
        q = core.PyQuery(ctx, a, ("",), False)

        q.set_ranges([[(0,3)]])
        q.submit()
        self.assertEqual(q.results()[''], a[:])


BasicTest().test()

a = tiledb.open(uri)
r = core.PyQuery(ctx, a, ("",), False)
