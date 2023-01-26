import base64
import io
import tarfile

import numpy as np
import pytest
from numpy.testing import assert_array_equal

import tiledb

from .common import DiskTestCase


# This test writes to local filesystem, skip
#   TODO: unskip if we support transparent file ops on a VFS
@pytest.mark.skipif(
    pytest.tiledb_vfs != "file", reason="Do not run compat test against non-file VFS"
)
class TestBackwardCompatibility(DiskTestCase):
    def test_compat_tiledb_py_0_5_anon_attr_dense(self):
        # array written with the following script:
        """
        import tiledb, numpy as np
        dom = tiledb.Domain(tiledb.Dim(name="d", domain=(0, 0), tile=1, dtype=np.uint8))
        attrs = (tiledb.Attr(name="_attr_", dtype=np.uint8),)
        schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False)
        path = "py0.5.9-test"
        tiledb.DenseArray.create(path, schema)
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
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(array_tgz))) as tf:
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
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(test_array))) as tf:
            tf.extractall(path)
        with tiledb.open(path) as A:
            assert_array_equal(A[:][""], np.array([1.0, 2.0, 5.0]))

    def test_tiledb_py_0_6_anon_attr(self):
        # same creation steps as above for 0.5
        tgz_sparse = b"""H4sIAJKNpWAAA+2aPW/TQBjHz2nTFlGJClUoAxIuA0ICpXf2vdhbGWBiYEIgihI7MRT1JVKairKh
                         qgNfgA2kDnwFVga+ABtfgE8AEwsS5/ROTUzBjWpbKv3/JPexLxc/l/59zz3PJc1lUjqUUiWEO7Ty
                         0GqsPbxgnArmUymk71LmUc6JK8ofGiE724Oor4fyYqvXH/S2/trv5VqSbPzjPuMfyi18nCXRXG61
                         NpNBVOZjMIH+XEip9fc9xaB/FaT6b/Q6681BNy7Lh/5/SM4n0l8JPf9pWQMaBfq3on4/etXa7qwl
                         m1EZz0Ge/p6X1V9wKaF/FdT1sWrOXxs77dhXLw//OiRtcNKuzvBspH+gjwVz7Yy07TqdhNTuzcw4
                         OwtT0407qzM3Hi58vzZH7678cN99rl9f2ji40JZ77T0Wzb+JD/rdp8SZnfta2gcFx5LOfyY9xqXn
                         ByoIVeYqDJMu44GOyGHCeRIGKuHCF1HsRRGLaacl8jOHifM/z2M+8r9KOL3+zd56jo8J1n+rPxcC
                         8b8KjvRnvlSh8rJXcRJ2Euor7gne8XgsJdVPhAoSFXZFogrWX6//aqg/p9C/Ck6vf6Hx3+rPmEL8
                         r4IC9G+1nvWj55vJ1mC4k9CNBpkqImf+a7VFRn8phI/5XwVpUh+Yc9fYk+b/af9FMp7/27Zd51vc
                         brf3Y7c+e//BFeJ8IJfSG9hoYd9zUl9p/4sZX7ZN1xrdlXrquwYXcAEXx7s4ojbSOWXK2NtknBVy
                         Mmxc/GKsZ2781tifxj4xjj8Zu2Qc79sBgKopYP3v5u0Z5uX/7I/8z6ce9n8rwYaAhj6ukvE4Yttu
                         flz+5TbeE/JIt9vYUSO3Hs8Pwww4wxQw/3O/Msit/wXP1n9Sof6vhNH538i02ak+njyA/4kC9v+L
                         rP/N/q8UmP/VgPofLuDiXLg4AvU/MBSw/hdZ/5v1XxcCDOt/FaD+P98UMP+LrP/t7z8Uxe8/KgH1
                         PwAAAAAAAAAAAAAAAAAAAAAAAHD2+Q18oX51AFAAAA=="""

        path = self.path("0_6_anon_sparse")
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(tgz_sparse))) as tf:
            tf.extractall(path)
        with tiledb.open(path) as A:
            if A.schema.sparse:
                assert_array_equal(A[:][""], np.array([1.0, 2.0, 5.0]))

        ###########################################################################################
        # This test checks that anonymous attributes internally stored as "__attr" are presented
        # as "".
        # The following steps were run under TileDB-Py 0.6
        # Normally, we can't actually write an attribute named "__attr" anymore, so
        # restored a schema written by a patched libtiledb, and rename the attr file.

        # schema_data = b"\x05\x00\x00\x00]\x00\x00\x00\x00\x00\x00\x00q\x00\x00\x00\x00\x00\x00\x00\x04\x01\x00\x00\x00\x00\x00\x00\x00\x00\x12\x00\x00\x00\x00\x00\x01\x00\x01\x00\x00\x00\x01\x05\x00\x00\x00\x01\x01\x00\x00\x00\x01\x00\x00\x00\x00\x00\x00\x00q\x00\x00\x009\x00\x00\x00\x10\x00\x00\x00\x00\x00\x00\x00\x01\x00\x00\x00q\x00\x00\x009\x00\x00\x00x\x01ce\x80\x00\x01u(\x83\x81\x11\x08\x19\x18\x98XA\xc4\x7f `\xc0\x10\x01\xc9\x83p\n\x1b\x88\x84\xb0\x81\x8a\xc1l\x88\x00H\x9c\r\x88\xe3\xe3\x13KJ\x8aP\x94\x01\x00\xa2c\x0bD"

        # path = self.path("tiledb_py_0_6_anon_attr")
        # ctx = tiledb.default_ctx()
        # dom = tiledb.Domain(tiledb.Dim(name="d", domain=(0, 0), tile=1, dtype=np.uint8))
        # attrs = (tiledb.Attr(name="_attr_", dtype=np.uint8, ctx=ctx),)

        # schema = tiledb.ArraySchema(domain=dom, attrs=attrs, sparse=False, ctx=ctx)
        # tiledb.DenseArray.create(path, schema, ctx=ctx)

        # with tiledb.open(path, "w") as A:
        #    A[0] = 1

        # fragment_name = os.path.split(list(A.last_write_info.keys())[0])[-1]
        # fragment_path = os.path.join(path, fragment_name)

        ## fix up the array the override schema
        # with open(os.path.join(path, "__array_schema.tdb"), "wb") as f:
        #    f.write(schema_data)

        # shutil.move(
        #    os.path.join(fragment_path, "_attr_.tdb"),
        #    os.path.join(fragment_path, "__attr.tdb"),
        # )

        tgz_dense = b"""H4sIAL6RpWAAA+2YPW/TQBjH71qQKiKkAEIqYvEIS3p3uRd5A4kB0QUxdUHm/AJFzQu4rlrUoa3K
                        EFWMDB2Y+AQs7CAhJD5HPgBfgXNyRq4pdVNyHtDzk5z/3fni55y/L8+TdFaQcwghSghvonKqhkKn
                        HcqJoF1KpOx6hDLCCfKE+6UhtLWZ6dQs5eVgmGbDwV/nba8nSe+M65y8KW/u63REZyUI+kmmXT4G
                        s/vfZZKD/02Q+98bRhudLA5dxTCfh+R8Jv8VV8gjrhZUBvwPdJrqN8FmtJ70tYvnoM5/xmjFf8El
                        A/+b4LI5ntr2a6uXcHH2+uQVo3wA51PxpFWa75ujbfu4NLaDo2Qf4a07hwfXlm4tH6/d/7bnPfvS
                        xj8OX125PXr76eDoa2+EHn64OhqPb6w+Onr8HqOPUeuBy5sF/iDf/1QyymWXK6GYqvS4r3gcR2Gi
                        lc9JSLTvKxVqbRK6r0jsB6Iz3KiJMfP3P2OCwf5vhH/3v75ynLn+Y4wRCvVfE8zB/yB4nuoX/WSQ
                        TX5JxDqrVBE1+59RKSv+S8lh/zdCntSLHbxk9bz5P5/fQifzfzG2g8fhvtE11CqHKKaeN0T7lBDF
                        mCkx4nvmHR5agBAQAkKcHuL3FUvtm+hiRFa/W71rL/jO6k+rTxam+tnq8uJUdxcvGBhwxFzyv86y
                        9Iw/DmrrfyYq+Z9TTiH/NwEuKa6MAQAAAAAAAAAAAADwf/ALzPk2VwAoAAA="""

        path = self.path("0_6_anon_dense")
        with tarfile.open(fileobj=io.BytesIO(base64.b64decode(tgz_dense))) as tf:
            tf.extractall(path)
        with tiledb.open(path) as A:
            self.assertEqual(A.schema.attr(0).name, "")
            self.assertEqual(A.schema.attr(0)._internal_name, "__attr")
            self.assertEqual(A[0], 1)
            mres = A.multi_index[0]
            self.assertEqual(mres[""], 1)

            qres = A.query(coords=True).multi_index[0]
            self.assertEqual(qres["d"], 0)
