from __future__ import absolute_import

import sys, os, io, re, platform, unittest, random, warnings

import numpy as np
from numpy.testing import assert_array_equal

import tiledb
from tiledb.tests.common import DiskTestCase, assert_subarrays_equal, rand_utf8, rand_ascii, rand_ascii_bytes

class BackwardCompatibilityTests(DiskTestCase):
    def test_compat_tiledb_py_0_5_anon_attr_dense(self):
        # Test that anonymous attributes internally stored as "__attr" are presented as ""
        # Normally, we can't actually write an attribute named "__attr" anymore, so we
        # restore a schema written by a patched libtiledb, and rename the attr file.

        schema_data = b'\x05\x00\x00\x00]\x00\x00\x00\x00\x00\x00\x00q\x00\x00\x00\x00\x00\x00\x00\x04\x01\x00\x00\x00\x00\x00\x00\x00\x00\x12\x00\x00\x00\x00\x00\x01\x00\x01\x00\x00\x00\x01\x05\x00\x00\x00\x01\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00q\x00\x00\x009\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00q\x00\x00\x009\x00\x00\x00x\x01ce\x80\x00\x01u(\x83\x81\x11\x08\x19\x18\x98XA\xc4\x7f `\xc0\x10\x01\xc9\x83p\n\x1b\x88\x84\xb0\x81\x8a\xc1l\x88\x00H\x9c\r\x88\xe3\xe3\x13KJ\x8aP\x94\x01\x00\xa2c\x0bD'

        path = self.path("tiledb_py_0_6_anon_attr")
        ctx = tiledb.default_ctx()
        dom = tiledb.Domain(tiledb.Dim(name="d", domain=(0, 0), tile=1, dtype=np.uint8))
        attrs = (tiledb.Attr(name="_attr_", dtype=np.uint8, ctx=ctx),)

        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False, ctx=ctx)
        tiledb.DenseArray.create(path, schema, ctx=ctx)

        with tiledb.open(path, 'w') as A:
            A[0] = 1

        fragment_name = os.path.split(list(A.last_write_info.keys())[0])[-1]
        fragment_path = os.path.join(path, fragment_name)

        # fix up the array the override schema
        with open(os.path.join(path, "__array_schema.tdb"), 'wb') as f:
            f.write(schema_data)
        import shutil
        shutil.move(
            os.path.join(fragment_path, "_attr_.tdb"),
            os.path.join(fragment_path, "__attr.tdb")
        )
        with tiledb.open(path) as A:
            self.assertEqual(A.schema.attr(0).name, "")
            self.assertEqual(A.schema.attr(0)._internal_name, "__attr")
            self.assertEqual(A[0], 1)
            mres = A.multi_index[0]
            self.assertEqual(mres[''], 1)

            qres = A.query(coords=True).multi_index[0]
            self.assertEqual(qres['d'], 0)


    def test_compat_py_0_5_anon_attr_sparse(self):
        import tarfile, base64
        from io import BytesIO

        # This array was written with TileDB-Py 0.5.9:
        # - using the invocation below, followed by
        """
        tiledb.Array.create("path", tiledb.ArraySchema(
                      domain=tiledb.Domain(*[
                      tiledb.Dim(name='d', domain=(0, 2), tile=2, dtype='uint64'),]),
                      attrs=[tiledb.Attr(name='', dtype='int64'),], sparse=True,))
        with tiledb.open("path", 'w') as A:
            A[[0,1,2]] = np.array([1.0,2.0,5.0])
        """
        # - followed by `tar czf array.tgz -C path`
        # - followed by `base64.encodebytes(open("sp6.tgz", 'rb').read())`
        test_array = b"""H4sIANDnmV8AA+2Xz2vUQBTHJ6mLlnpYBGkRD0EQBGV3ZpLJdBFk9bBnj3pKJpvESrsbmo2otyoI
                         Pe/JSy9ePXnwruJBPPYv0P4VRRDNhAxm07o/dBN6eJ9lMpmXSd6Eb96bt602qhyMMcfYyHqbZT3O
                         xwqDmNyyzfRnWwYmFDOCDFb90hB6MkpEnC7l8TCKk2j413lPt4JgZ8pzJl/KWPo6K6LVdpxBkIgq
                         P4OF9Gck1d+kHPSvBan/TtTfbiW+V5WPf9CfM44MXNWCioD+johj8dwZ9beCgajiO5ilP6V2SX9m
                         cdC/Fs6lTQm+q2yaunopO2pIGrSGPGRnhfl30tbMx1rB9kzrC9d1fbd5//yh++HCEcXvXu7/6qJx
                         7/J3fffuZmP/497qgTYOVo6Ojz+Px9d6zfU3r15o6O322r0q3xgoIuOf2NjsULJppVHHSiOPh6Hn
                         9ZnAWFicsk4YspCEOOAd7jFO56kbFq7/KCXEhv2/Dv5bf8cJY/FoEAyTrI70RXJiD5mhPyEWKelv
                         M0Yh/9eBzP+38/PryjZn/pfz19Fk/le2NP/7rvtNFz1D+/Rlb/WrhvQf6Ip0p1KGum1ed3L+Wsmd
                         skl33fQOA+ngYgEXf9ALkyUreX8r77vodKK8P8x7lj/gtXbabOCMsYT8L5Iknvq3Yeb+z6xS/rew
                         bUL+rwMVpRt5K9pUSmjUuiKgTpYQ//0oiv3RlAwwK/7JifrfMjnUf7VQjP+raLJmULYb79s/jY0D
                         hB6kdpUUdHTz4cWspAAAAAAAAAAAAAAA4IzzG7vsp0oAKAAA"""

        path = self.path("test_tiledb_py_0_5_anon_attr_sparse")
        with tarfile.open(fileobj=BytesIO(base64.b64decode(test_array))) as tf:
            tf.extractall(path)
        with tiledb.open(path) as A:
            assert_array_equal(A[:][''], np.array([1.0,2.0,5.0]))
