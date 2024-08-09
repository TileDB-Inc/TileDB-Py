import shutil
import tempfile

import numpy as np

import tiledb


class Basic:
    def setup(self, *shape):
        self.path = tempfile.mkdtemp()
        self.array = np.random.rand(4)
        tiledb.from_numpy(self.path, self.array)

    def time_open(self):
        for i in range(5_000):
            with tiledb.open(self.path):
                pass


class DenseRead:
    # parameterize over different array shapes
    # the functions below will be called with permutations
    # of these tuples
    params = [
        (100, 500),
        (1000, 100000),
    ]

    def setup(self, *shape):
        self.path = tempfile.mkdtemp()
        self.array = np.random.rand(*shape)
        tiledb.from_numpy(self.path, self.array)

    def time_read(self, *shape):
        with tiledb.open(self.path) as A:
            A[:]

    def teardown(self, *shape):
        shutil.rmtree(self.path)


class DenseWrite:
    params = [
        (100, 500),
        (1000, 100000),
    ]
    paths = []

    def setup(self, *shape):
        self.array = np.random.rand(*shape)

    def time_write(self, *shape):
        path = tempfile.mkdtemp()
        tiledb.from_numpy(path, self.array)
