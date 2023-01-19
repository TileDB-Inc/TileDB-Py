import tiledb
from tiledb import SparseArray
import numpy as np


def create_array(uri):
  schema = tiledb.ArraySchema(
      tiledb.Domain(
          [tiledb.Dim(dtype=np.int64, domain=(-100, 100))]
      ),
      attrs=[
          tiledb.Attr(name="a", dtype="float64", var=False, nullable=False)
      ],
      cell_order="row-major",
      tile_order="row-major",
      capacity=10000,
      sparse=True,
  )

  tiledb.Array.create(uri, schema)

def write_sparse(uri):
  data = np.arange(-100, 100, dtype=np.int64)
  with tiledb.open(uri, "w") as A:
      A[data] = data

def run():
  uri = "some_path"
  create_array(uri)
  write_sparse(uri)

  input("wait for lldb")

  with tiledb.open(uri) as A:
    r1 = A.multi_index[9223372036854775808]

if __name__ == "__main__":
  run()
