import unittest, os, sys, copy, random

import tiledb
from tiledb import TileDBError, core
from tiledb.tests.common import DiskTestCase, rand_ascii

import numpy as np
from numpy.testing import assert_array_equal

class CoreCCTest(DiskTestCase):

    def test_pyquery_basic(self):
        ctx = tiledb.default_ctx()
        uri = self.path("test_pyquery_basic")
        with tiledb.from_numpy(uri, np.random.rand(4)) as A:
            pass

        with tiledb.open(uri) as a:
            with self.assertRaises(ValueError):
                testctx = tiledb.Ctx(config={'py.init_buffer_bytes': 'abcd'})
                core.PyQuery(testctx, a, ("",), (), 0, False)

            q = core.PyQuery(ctx, a, ("",), (), 0, False)

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
            q2 = core.PyQuery(ctx, a, ("",), (), 0, False)

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
            q = core.PyQuery(ctx, a, ("",), (), 0, False)
            self.assertEqual(q._test_init_buffer_bytes, intmax)

    def test_import_buffer(self):
        uri = self.path("test_import_buffer")

        ctx = tiledb.Ctx()

        def_tile = 1
        if tiledb.libtiledb.version() < (2,2):
          def_tile = 2

        dom = tiledb.Domain(tiledb.Dim(domain=(0, 3), tile=def_tile,
                                       dtype=np.int64, ctx=ctx),
                            tiledb.Dim(domain=(0, 3), tile=def_tile,
                                       dtype=np.int64, ctx=ctx),
                            ctx=ctx)
        attrs = [
            tiledb.Attr(name='', dtype=np.float64, ctx=ctx),
            tiledb.Attr(name='foo', dtype=np.int32, ctx=ctx),
            tiledb.Attr(name='str', dtype=np.str, ctx=ctx)
        ]
        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False, ctx=ctx)
        tiledb.DenseArray.create(uri, schema)

        data_orig = {
            '': 2.5 * np.identity(4, dtype=np.float64),
            'foo': 8 * np.identity(4, dtype=np.int32),
            'str': np.array([rand_ascii(random.randint(0,5)) for _ in range(16)]).reshape(4,4)
        }

        with tiledb.open(uri, 'w') as A:
           A[:] = data_orig

        with tiledb.open(uri) as B:
            assert_array_equal(B[:][''], data_orig['']),
            assert_array_equal(B[:]['foo'], data_orig['foo'])

        data_mod = {
            '': 5 * np.identity(4, dtype=np.float64),
            'foo': 32 * np.identity(4, dtype=np.int32),
            'str': np.array([rand_ascii(random.randint(1,7))
                             for _ in range(16)], dtype='U0').reshape(4,4)
        }

        str_offsets = np.array([0] + [len(x) for x in
                                      data_mod['str'].flatten()[:-1]],
                               dtype=np.uint64)
        str_offsets = np.cumsum(str_offsets)

        str_raw = np.array([ord(c) for c in
                            ''.join([x for x in data_mod['str'].flatten()])],
                           dtype=np.uint8)

        data_mod_bfr = {
            '': (data_mod[''].flatten().view(np.uint8),
                 np.array([], dtype=np.uint64)),
            'foo': (data_mod['foo'].flatten().view(np.uint8),
                    np.array([], dtype=np.uint64)),
            'str': (str_raw.flatten().view(np.uint8), str_offsets)
        }

        with tiledb.open(uri) as C:
            res = C.multi_index[0:3,0:3]
            assert_array_equal(res[''], data_orig[''])
            assert_array_equal(res['foo'], data_orig['foo'])
            assert_array_equal(res['str'], data_orig['str'])

            C._set_buffers(copy.deepcopy(data_mod_bfr))
            res = C.multi_index[0:3,0:3]
            assert_array_equal(res[''], data_mod[''])
            assert_array_equal(res['foo'], data_mod['foo'])
            assert_array_equal(res['str'], data_mod['str'])

        with tiledb.open(uri) as D:
            D._set_buffers(copy.deepcopy(data_mod_bfr))
            res = D[:,:]
            assert_array_equal(res[''], data_mod[''])
            assert_array_equal(res['foo'], data_mod['foo'])
            assert_array_equal(res['str'], data_mod['str'])

        with tiledb.DenseArray(uri, mode='r') as E:
             # Ensure that query only returns specified attributes
            q = core.PyQuery(ctx, E, ("foo",), (), 0, False)
            q.set_ranges([[(0,1)]])
            q.submit()
            r = q.results()
            self.assertTrue("foo" in r)
            self.assertTrue("str" not in r)
            del q
