from __future__ import absolute_import

import glob
import os
import shutil
import tempfile
import traceback
from unittest import TestCase
from numpy.testing import assert_equal, assert_array_equal


def assert_subarrays_equal(a, b):
    assert_equal(a.shape, b.shape)

    for a_el, b_el in zip(a.flat, b.flat):
        assert_array_equal(a_el, b_el)


class DiskTestCase(TestCase):
    pathmap = dict()

    def setUp(self):
        prefix = 'tiledb-' + self.__class__.__name__
        self.rootdir = tempfile.mkdtemp(prefix=prefix)

    def tearDown(self):
        # Remove every directory starting with rootdir
        for dirpath in glob.glob(self.rootdir + "*"):
            try:
                shutil.rmtree(dirpath)
            except OSError as exc:
                print("test '{}' error deleting '{}'".format(self.__class__.__name__,
                                                             dirpath))
                print("registered paths and originating functions:")
                for path,frame in self.pathmap.items():
                    print("  '{}' <- '{}'".format(path,frame))
                raise exc

    def path(self, path):
        out = os.path.abspath(os.path.join(self.rootdir, path))
        frame = traceback.extract_stack(limit=2)[-2][2]
        self.pathmap[out] = frame
        return out

