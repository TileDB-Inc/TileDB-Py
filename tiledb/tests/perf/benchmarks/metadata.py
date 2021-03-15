import tiledb
import numpy as np
import tempfile, shutil
import time


class MetadataTest:
    def setup(self):
        self.path = tempfile.mkdtemp()
        print(self.path)
        self.array = np.random.rand(4)
        tiledb.from_numpy(self.path, self.array)


class MetadataWrite(MetadataTest):
    def setup(self):
        super().setup()

    def time_write(self):
        with tiledb.open(self.path, "w") as A:
            for i in range(1_000_000):
                A.meta["x"] = "xyz"


class MetadataRead(MetadataTest):
    def setup(self):
        super().setup()

        with tiledb.open(self.path, "w") as A:
            A.meta["x"] = "xyz"

    def time_read(self):
        with tiledb.open(self.path) as A:
            for i in range(1_000_000):
                A.meta["x"]
