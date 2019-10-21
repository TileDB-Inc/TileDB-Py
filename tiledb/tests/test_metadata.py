#%%

import tiledb
from tiledb.libtiledb import ustring
import numpy as np

import unittest, os
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

            # can't write objects
            with self.assertRaises(ValueError):
                A.meta['object'] = object

        test_vals = {
            'int': 10,
            'double': 1.000001212,
            'bytes': b"0123456789abcdeF0123456789abcdeF",
            'str': "abcdefghijklmnopqrstuvwxyz",
        }

        with tiledb.Array(path, mode='w') as A:
            for k,v in test_vals.items():
                A.meta[k] = v

        with tiledb.Array(path) as A:
            for k,v in test_vals.items():
                self.assertEqual(A.meta[k], v)

        # test dict-like functionality
        with tiledb.Array(path) as A:
            self.assertSetEqual(set(A.meta.keys()), set(test_vals.keys()))
            self.assertFalse('gnokey' in A.meta)
            self.assertEqual(len(A.meta), 4)

            for k,v in A.meta.items():
                self.assertTrue(k in test_vals.keys())
                self.assertEqual(v, test_vals[k])

        # test a 1 MB blob
        blob = np.random.rand(int((1024**2)/8)).tostring()
        with tiledb.Array(path, 'w') as A:
            A.meta['bigblob'] = blob

        with tiledb.Array(path) as A:
            self.assertEqual(A.meta['bigblob'], blob)
            self.assertEqual(len(A.meta), 5)

        # test del key
        with tiledb.Array(path, 'w') as A:
            del A.meta['bigblob']

        with tiledb.Array(path) as A:
            self.assertEqual(len(A.meta), 4)
            with self.assertRaises(KeyError):
                A.meta['bigblob']

        # test pop NotImplementedError
        with tiledb.Array(path, 'w') as A:
            with self.assertRaises(NotImplementedError):
                A.meta.pop('nokey', 'hello!')

    def test_metadata_consecutive(self):
        path = self.path("test_md_consecutive")

        write_count = 100

        with tiledb.from_numpy(path, np.ones((5,), np.float64)) as A:
            pass

        randints = np.random.randint(0,np.iinfo(np.int64).max - 1,
                                     size=write_count, dtype=np.int64)
        randutf8s = [rand_utf8(i) for i in np.random.randint(1,30,size=write_count)]

        # write 100 times, then consolidate
        for i in range(write_count):
            with tiledb.Array(path, mode='w') as A:
                A.meta['randint'] = int(randints[i])
                A.meta['randutf8'] = randutf8s[i]

        with tiledb.Array(path) as A:
            self.assertEqual(A.meta['randint'], randints[-1])
            self.assertEqual(A.meta['randutf8'], randutf8s[-1])

        with tiledb.Array(path, mode='w') as aw:
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

        # test data
        with tiledb.Array(path) as A:
            for i in range(write_count):
                key_int = randutf8s[i] + u'{}'.format(randints[i])
                self.assertEqual(A.meta[key_int], randints[i])
                self.assertEqual(A.meta[randutf8s[i]], randutf8s[i])

        # test expected number of fragments
        self.assertEqual(
            len( os.listdir(os.path.join(path, "__meta")) ),
            201
            )

        with tiledb.Array(path, mode='w') as A:
            A.meta.consolidate()

        # should only have one fragment after consolidation
        self.assertEqual(
            len( os.listdir(os.path.join(path, "__meta")) ),
            1
            )

        # test data again after consolidation
        with tiledb.Array(path) as A:
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
