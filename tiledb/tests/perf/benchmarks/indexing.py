import tiledb
import numpy as np
import tempfile, shutil


class MultiIndex:
    params = [10, 100, 1000, 10_000, 100_000]

    def setup(self, _):
        self.uri = tempfile.mkdtemp()

        self.dmin = -10_000_000
        self.dmax = 10_000_000
        self.ncoords = 3_000_000

        schema = tiledb.ArraySchema(
            tiledb.Domain([tiledb.Dim(dtype=np.int64, domain=(self.dmin, self.dmax))]),
            attrs=[
                tiledb.Attr(name="", dtype="float64", var=False, nullable=False),
            ],
            cell_order="row-major",
            tile_order="row-major",
            capacity=1000,
            sparse=True,
        )

        tiledb.Array.create(self.uri, schema)

        # use `choice` here because randint doesn't support non-replacement
        self.coords = np.random.choice(
            np.arange(self.dmin, self.dmax + 1), size=self.ncoords, replace=False
        )

        with tiledb.open(self.uri, "w") as A:
            A[self.coords] = np.random.rand(self.ncoords)

    def time_multiindex_read(self, coords_count):
        coords = np.random.choice(self.coords, size=coords_count, replace=False)

        with tiledb.open(self.uri) as A:
            A.multi_index[list(coords)]
