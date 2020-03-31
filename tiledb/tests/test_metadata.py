#%%

import tiledb
from tiledb.libtiledb import ustring
import numpy as np

import unittest, os, time
from tiledb.tests.common import *

class MetadataTest(DiskTestCase):

    def test_metadata_basic(self):
        path = self.path("test_md_basic")

        with tiledb.from_numpy(path, np.ones((5,), np.float64)) as A:
            pass

        # sanity checks
        A = tiledb.open(path)
        A.close()

        # can't read from a closed array
        with self.assertRaises(tiledb.TileDBError):
            A.meta['x']

        with tiledb.Array(path) as A:
            # can't write to a mode='r' array
            with self.assertRaises(tiledb.TileDBError):
                A.meta['invalid_write'] = 1

            # missing key raises KeyError
            with self.assertRaises(KeyError):
                 A.meta['xyz123nokey']

        # test invalid input
        with tiledb.Array(path, 'w') as A:
            # keys must be strings
            with self.assertRaises(ValueError):
                A.meta[123] = 1

            # can't write an int > typemax(Int64)
            with self.assertRaises(OverflowError):
                A.meta['bigint'] = np.iinfo(np.int64).max + 1

            # can't write mixed-type list
            with self.assertRaises(TypeError):
                A.meta['mixed_list'] = [1, 2.1]

            # can't write mixed-type tuple
            with self.assertRaises(TypeError):
                A.meta['mixed_list'] = (0, 3.1)

            # can't write objects
            with self.assertRaises(ValueError):
                A.meta['object'] = object

        test_vals = {
            'int': 10,
            'double': 1.000001212,
            'bytes': b"0123456789abcdeF0123456789abcdeF",
            'str': "abcdefghijklmnopqrstuvwxyz",
            'emptystr': "",
            'tuple_int': (1,2,3,2,1, int(np.random.randint(0,10000,1)[0]) ),
            'list_int': [1,2,3,2,1, int(np.random.randint(0,10000,1)[0]) ],
            'tuple_float': (10.0, 11.0, float(np.random.rand(1)[0]) ),
            'list_float': [10.0, 11.0, float(np.random.rand(1)[0]) ]
        }

        def tupleize(v):
            if isinstance(v, list):
                v = tuple(v)
            return v

        with tiledb.Array(path, mode='w') as A:
            for k,v in test_vals.items():
                A.meta[k] = v

        with tiledb.Array(path) as A:
            for k,v in test_vals.items():
                # metadata only has one iterable type: tuple, so we cannot
                # perfectly round-trip the input type.

                self.assertEqual(A.meta[k], tupleize(v))

        # test dict-like functionality
        with tiledb.Array(path) as A:
            self.assertSetEqual(set(A.meta.keys()), set(test_vals.keys()))
            self.assertFalse('gnokey' in A.meta)
            self.assertEqual(len(A.meta), len(test_vals))

            for k,v in A.meta.items():
                self.assertTrue(k in test_vals.keys())
                self.assertEqual(tupleize(v), tupleize(test_vals[k]),)

        # test a 1 MB blob
        blob = np.random.rand(int((1024**2)/8)).tostring()
        with tiledb.Array(path, 'w') as A:
            A.meta['bigblob'] = blob

        with tiledb.Array(path) as A:
            self.assertEqual(A.meta['bigblob'], blob)
            self.assertEqual(len(A.meta), len(test_vals)+1)

        # test del key
        with tiledb.Array(path, 'w') as A:
            del A.meta['bigblob']

        with tiledb.Array(path) as A:
            self.assertTrue('bigblob' not in A.meta)
            self.assertEqual(len(A.meta), len(test_vals))
            with self.assertRaises(KeyError):
                A.meta['bigblob']

        # test pop NotImplementedError
        with tiledb.Array(path, 'w') as A:
            with self.assertRaises(NotImplementedError):
                A.meta.pop('nokey', 'hello!')

        # Note: this requires a work-around to check all keys
        # test empty value
        with tiledb.Array(path, 'w') as A:
            A.meta['empty_val'] = ()

        with tiledb.Array(path) as A:
            self.assertTrue('empty_val' in A.meta)
            self.assertEqual(A.meta['empty_val'], ())


    def test_metadata_consecutive(self):
        ctx = tiledb.Ctx({
            'sm.vacuum.mode': 'array_meta',
            'sm.consolidation.mode': 'array_meta'
        })
        vfs = tiledb.VFS(ctx=ctx)
        path = self.path("test_md_consecutive")

        write_count = 100

        with tiledb.from_numpy(path, np.ones((5,), np.float64), ctx=ctx) as A:
            pass

        randints = np.random.randint(0,int(np.iinfo(np.int64).max) - 1,
                                     size=write_count, dtype=np.int64)
        randutf8s = [rand_utf8(i) for i in np.random.randint(1,30,size=write_count)]

        # write 100 times, then consolidate
        for i in range(write_count):
            with tiledb.Array(path, mode='w', ctx=ctx) as A:
                A.meta['randint'] = int(randints[i])
                A.meta['randutf8'] = randutf8s[i]
                time.sleep(0.001)

        with tiledb.Array(path) as A:
            self.assertEqual(A.meta['randint'], randints[-1])
            self.assertEqual(A.meta['randutf8'], randutf8s[-1])

        with tiledb.Array(path, mode='w', ctx=ctx) as aw:
            aw.meta.consolidate()

        with tiledb.Array(path) as A:
            self.assertEqual(A.meta['randint'], randints[-1])
            self.assertEqual(A.meta['randutf8'], randutf8s[-1])

        # use randutf8s as keys, then consolidate
        for _ in range(2):
            for i in range(write_count):
                with tiledb.Array(path, mode='w') as A:
                    A.meta[randutf8s[i] + u'{}'.format(randints[i])] = int(randints[i])
                    A.meta[randutf8s[i]] = randutf8s[i]
                    time.sleep(0.001)

        # test data
        with tiledb.Array(path, ctx=ctx) as A:
            for i in range(write_count):
                key_int = randutf8s[i] + u'{}'.format(randints[i])
                self.assertEqual(A.meta[key_int], randints[i])
                self.assertEqual(A.meta[randutf8s[i]], randutf8s[i])

        # test expected number of fragments before consolidating
        self.assertEqual(
            len( vfs.ls(os.path.join(path, "__meta")) ),
            302
            )

        with tiledb.Array(path, mode='w', ctx=ctx) as A:
            A.meta.consolidate()

        # test expected number of fragments before vacuuming
        self.assertEqual(
            len( vfs.ls(os.path.join(path, "__meta")) ),
            304
            )

        tiledb.vacuum(path, ctx=ctx)

        # should only have one fragment+'.ok' after vacuuming
        self.assertEqual(
            len( vfs.ls(os.path.join(path, "__meta")) ),
            1
            )

        # test data again after consolidation
        with tiledb.Array(path, ctx=ctx) as A:
            for i in range(write_count):
                key_int = randutf8s[i] + u'{}'.format(randints[i])
                self.assertEqual(A.meta[key_int], randints[i])
                self.assertEqual(A.meta[randutf8s[i]], randutf8s[i])

    def test_metadata_small_dtypes(self):
        path = self.path("test_md_small_dtypes")

        with tiledb.from_numpy(path, np.arange(1)) as A:
            pass

        test_vals = {
            "np.int8": np.array((-1,), dtype=np.int8),
            "np.uint8": np.array((2,), dtype=np.uint8),
            "np.int16": np.array((-3,), dtype=np.int16),
            "np.uint16": np.array((4,), dtype=np.uint16),
            "np.int32": np.array((-5,), dtype=np.int32),
            "np.uint32": np.array((6,), dtype=np.uint32),
            "np.float32": np.array((-7.0,), dtype=np.float32)
        }

        with tiledb.Array(path, 'w') as A:
            for k,v in test_vals.items():
                A.meta._set_numpy(k, v)

        # note: the goal here is to test read-back of these datatypes,
        #       which is currently done as int for all types
        with tiledb.Array(path) as A:
            for k,v in test_vals.items():
                self.assertEqual(A.meta[k], int(v))
