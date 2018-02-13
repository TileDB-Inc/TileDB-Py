from __future__ import absolute_import

import glob
import os
import shutil
import tempfile
from unittest import TestCase


def remove_tree(rootdir):
    # Remove every directory starting with rootdir
    for dirpath in glob.glob(rootdir + "*"):
        shutil.rmtree(dirpath)


class DiskTestCase(TestCase):

    def setUp(self):
        prefix = 'tiledb-' + self.__class__.__name__
        self.rootdir = tempfile.mkdtemp(prefix=prefix)

    def tearDown(self):
        remove_tree(self.rootdir)

    def path(self, path):
        return os.path.abspath(os.path.join(self.rootdir, path))

