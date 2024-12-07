import copy
import random

import numpy as np
from numpy.testing import assert_array_equal

import tiledb
import tiledb.main as core

from .common import DiskTestCase, rand_ascii


class CoreCCTest(DiskTestCase):
    def test_pyquery_basic(self):
        ctx = tiledb.Ctx()
        uri = self.path("test_pyquery_basic")
        with tiledb.from_numpy(uri, np.random.rand(4)):
            pass

        with tiledb.open(uri) as a:
            with tiledb.scope_ctx({"py.init_buffer_bytes": "abcd"}) as testctx:
                with self.assertRaises(ValueError):
                    core.PyQuery(testctx, a, ("",), (), 0, False)

            q = core.PyQuery(ctx, a, ("",), (), 0, False)

            try:
                q._test_err("bad foo happened")
            except Exception as exc:
                assert isinstance(exc, tiledb.TileDBError)
                assert str(exc) == "bad foo happened"

        with tiledb.open(uri) as a:
            q2 = core.PyQuery(ctx, a, ("",), (), 0, False)
            subarray = tiledb.Subarray(a)
            subarray.add_ranges([[(0, 3)]])
            q2.set_subarray(subarray)
            q2.submit()
            res = q2.results()[""][0]
            res.dtype = np.double
            assert_array_equal(res, a[:])

    def test_pyquery_init(self):
        uri = self.path("test_pyquery_init")
        intmax = np.iinfo(np.int64).max
        config_dict = {
            "sm.tile_cache_size": "100",
            "py.init_buffer_bytes": str(intmax),
            "py.alloc_max_bytes": str(intmax),
        }
        with tiledb.scope_ctx(config_dict) as ctx:
            with tiledb.from_numpy(uri, np.random.rand(4)):
                pass

            with tiledb.open(uri) as a:
                q = core.PyQuery(ctx, a, ("",), (), 0, False)
                self.assertEqual(q._test_init_buffer_bytes, intmax)
                self.assertEqual(q._test_alloc_max_bytes, intmax)

                with self.assertRaisesRegex(
                    ValueError,
                    "Invalid parameter: 'py.alloc_max_bytes' must be >= 1 MB ",
                ), tiledb.scope_ctx({"py.alloc_max_bytes": 10}) as ctx2:
                    q = core.PyQuery(ctx2, a, ("",), (), 0, False)

    def test_import_buffer(self):
        uri = self.path("test_import_buffer")

        def_tile = 1
        if tiledb.libtiledb.version() < (2, 2):
            def_tile = 2

        dom = tiledb.Domain(
            tiledb.Dim(domain=(0, 3), tile=def_tile, dtype=np.int64),
            tiledb.Dim(domain=(0, 3), tile=def_tile, dtype=np.int64),
        )
        attrs = [
            tiledb.Attr(name="", dtype=np.float64),
            tiledb.Attr(name="foo", dtype=np.int32),
            tiledb.Attr(name="str", dtype=str),
        ]
        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False)
        tiledb.DenseArray.create(uri, schema)

        data_orig = {
            "": 2.5 * np.identity(4, dtype=np.float64),
            "foo": 8 * np.identity(4, dtype=np.int32),
            "str": np.array(
                [rand_ascii(random.randint(0, 5)) for _ in range(16)], dtype="U0"
            ).reshape(4, 4),
        }

        with tiledb.open(uri, "w") as A:
            A[:] = data_orig

        with tiledb.open(uri) as B:
            assert_array_equal(B[:][""], data_orig[""]),
            assert_array_equal(B[:]["foo"], data_orig["foo"])

        data_mod = {
            "": 5 * np.identity(4, dtype=np.float64),
            "foo": 32 * np.identity(4, dtype=np.int32),
            "str": np.array(
                [rand_ascii(random.randint(1, 7)) for _ in range(16)], dtype="U0"
            ).reshape(4, 4),
        }

        str_offsets = np.array(
            [0] + [len(x) for x in data_mod["str"].flatten()[:-1]], dtype=np.uint64
        )
        str_offsets = np.cumsum(str_offsets)

        str_raw = np.array(
            [ord(c) for c in "".join([x for x in data_mod["str"].flatten()])],
            dtype=np.uint8,
        )

        data_mod_bfr = {
            "": (data_mod[""].flatten().view(np.uint8), np.array([], dtype=np.uint64)),
            "foo": (
                data_mod["foo"].flatten().view(np.uint8),
                np.array([], dtype=np.uint64),
            ),
            "str": (str_raw.flatten().view(np.uint8), str_offsets),
        }

        with tiledb.open(uri) as C:
            res = C.multi_index[0:3, 0:3]
            assert_array_equal(res[""], data_orig[""])
            assert_array_equal(res["foo"], data_orig["foo"])
            assert_array_equal(res["str"], data_orig["str"])

            C._set_buffers(copy.deepcopy(data_mod_bfr))
            res = C.multi_index[0:3, 0:3]
            assert_array_equal(res[""], data_mod[""])
            assert_array_equal(res["foo"], data_mod["foo"])
            assert_array_equal(res["str"], data_mod["str"])

        with tiledb.open(uri) as D:
            D._set_buffers(copy.deepcopy(data_mod_bfr))
            res = D[:, :]
            assert_array_equal(res[""], data_mod[""])
            assert_array_equal(res["foo"], data_mod["foo"])
            assert_array_equal(res["str"], data_mod["str"])

        with tiledb.DenseArray(uri, mode="r") as E, tiledb.scope_ctx() as ctx:
            # Ensure that query only returns specified attributes
            q = core.PyQuery(ctx, E, ("foo",), (), 0, False)
            subarray = tiledb.Subarray(E, ctx)
            subarray.add_ranges([[(0, 1)]])
            q.set_subarray(subarray)
            q.submit()
            r = q.results()
            self.assertTrue("foo" in r)
            self.assertTrue("str" not in r)
            del q
