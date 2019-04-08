from __future__ import absolute_import

import glob
import os
import shutil
import tempfile
from unittest import TestCase

class DiskTestCase(TestCase):

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
                raise

    def path(self, path):
        return os.path.abspath(os.path.join(self.rootdir, path))

