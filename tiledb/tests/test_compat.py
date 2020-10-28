from __future__ import absolute_import

import sys, os, io, re, platform, unittest, random, warnings

import numpy as np
from numpy.testing import assert_array_equal

import tiledb
from tiledb.tests.common import DiskTestCase, assert_subarrays_equal, rand_utf8, rand_ascii, rand_ascii_bytes

class BackwardCompatibilityTests(DiskTestCase):
    def test_tiledb_py_0_6_anon_attr_dense(self):
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