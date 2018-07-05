import numpy as np
import sys
import tiledb

array_name = "quickstart_dense"


def create_array():
    # Create a TileDB context
    ctx = tiledb.Ctx()

    # Check if the array already exists.
    if tiledb.object_type(ctx, array_name) == "array":
        print("Array already exists.")
        sys.exit(0)

    # The array will be 4x4 with dimensions "rows" and "cols", with domain [1,4].
    dom = tiledb.Domain(ctx,
                        tiledb.Dim(ctx, name="rows", domain=(1, 4), tile=4, dtype=np.int32),
                        tiledb.Dim(ctx, name="cols", domain=(1, 4), tile=4, dtype=np.int32))

    # The array will be dense with a single attribute "a" so each (i,j) cell can store an integer.
    schema = tiledb.ArraySchema(ctx, domain=dom, sparse=False,
                                attrs=[tiledb.Attr(ctx, name="a", dtype=np.int32)])

    # Create the (empty) array on disk.
    tiledb.DenseArray.create(array_name, schema)


def write_array():
    ctx = tiledb.Ctx()
    # Open the array and write to it.
    with tiledb.DenseArray(ctx, array_name, mode='w') as A:
        data = np.array(([1, 2, 3, 4],
                         [5, 6, 7, 8],
                         [9, 10, 11, 12],
                         [13, 14, 15, 16]))
        A[:] = data


def read_array():
    ctx = tiledb.Ctx()
    # Open the array and read from it.
    with tiledb.DenseArray(ctx, array_name, mode='r') as A:
        # Slice only rows 1, 2 and cols 2, 3, 4.
        data = A[1:3, 2:5]
        print(data["a"])


create_array()
write_array()
read_array()
