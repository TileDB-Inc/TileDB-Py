from __future__ import absolute_import

import sys, os, io, re, platform, unittest, random, warnings

import numpy as np
import tarfile, base64
from io import BytesIO

from numpy.testing import assert_array_equal

import tiledb
from tiledb.tests.common import (
    DiskTestCase,
    assert_subarrays_equal,
    rand_utf8,
    rand_ascii,
    rand_ascii_bytes,
)


class BackwardCompatibilityTests(DiskTestCase):
    def test_compat_tiledb_py_0_5_anon_attr_dense(self):
        # array written with the following script:
        """
        import tiledb, numpy as np
        ctx = tiledb.default_ctx()
        dom = tiledb.Domain(tiledb.Dim(name="d", domain=(0, 0), tile=1, dtype=np.uint8))
        attrs = (tiledb.Attr(name="_attr_", dtype=np.uint8, ctx=ctx),)
        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False, ctx=ctx)
        path = "py0.5.9-test"
        tiledb.DenseArray.create(path, schema, ctx=ctx)
        with tiledb.open(path, "w") as A:
            A[0] = 1
        """
        # save and print tgz of array directory:
        # f = open("/tmp/py0.5.9-testa2.tgz",'rb').read()
        # s = base64.encodebytes(f)
        # print(f"{s.decode():>32}")

        array_tgz = b"""H4sIADjvS2AAA+2YzW4TMRCA7fIX0SJVFdz9AAg8XtubvbR9AF6gEpLjJg4FmgRttwJuReKAuFFe
                        oUcO9A165NJ7jxWPwBOwXq3RZgnNtmkiBPNJ2bEnY89uRjMZrzGgQal2ArFUXNZm0sa8D7GL2tpJ
                        SKIk6XIFTiVxlIg4UY9JEzjnMeeskFoVkpfzAAPJhYh1LLVmXIDgQJhqtPuM7O9lNs1v5flwlGaj
                        4R/tXu84t3vBPuMPxa79PueEmS3+xvRT+2zghpkZuMz2bGYfZb3tcR9T4g8AuhZ/paOYML6IH+A/
                        j//N/KPL8b2go+HbteJKiVfQW/5SjCr23mK1nNOK7g3t9jqd86Vtzfr59JCseU+hXoQVTT15++Wa
                        p6DznjbzFYwsoYtLuPi1Y2X8gFzMi1KelpKXCz/TSdbI38/M9d9mWfp7yR9j6v+/ULX6H4GUWP8X
                        Aa1IWtMh/z55AqepfWv2ujtuMKF3uw6m5b+AWv6DiiTH/F8EvhPYKsdPg65hs+Ht/Rmt2mwEXd5s
                        WHKD7rdOT05a71dWnnxh3zdWOx+/vrt/8Oruh9twdtBeXz8+Omo9vPPJdQj58W15Y47PiUzGmN1R
                        9+V88j5w6fM/RFoIzP9FYIpze7P3OFflCvGHSOL7HwRBEARBEARBEARBEARBkFn4CRFQSoEAKAAA"""

        path = self.path("tiledb_py_0_6_anon_attr")
        with tarfile.open(fileobj=BytesIO(base64.b64decode(array_tgz))) as tf:
            tf.extractall(path)

        with tiledb.open(path) as A:
            self.assertEqual(A.schema.attr(0).name, "")
            self.assertEqual(A.schema.attr(0)._internal_name, "__attr")
            self.assertEqual(A[0], 1)
            mres = A.multi_index[0]
            self.assertEqual(mres[""], 1)

            qres = A.query(coords=True).multi_index[0]
            self.assertEqual(qres["d"], 0)

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
            assert_array_equal(A[:][""], np.array([1.0, 2.0, 5.0]))
