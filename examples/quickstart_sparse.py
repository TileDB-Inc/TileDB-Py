import numpy as np
import sys
import tiledb

array_name = "quickstart_sparse"


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

    # The array will be sparse with a single attribute "a" so each (i,j) cell can store an integer.
    schema = tiledb.ArraySchema(ctx, domain=dom, sparse=True,
                                attrs=[tiledb.Attr(ctx, name="a", dtype=np.int32)])

    # Create the (empty) array on disk.
    tiledb.DenseArray.create(array_name, schema)


def write_array():
    ctx = tiledb.Ctx()
    # Open the array and write to it.
    with tiledb.SparseArray(ctx, array_name, mode='w') as A:
        # Write some simple data to cells (1, 1), (2, 4) and (2, 3).
        I, J = [1, 2, 2], [1, 4, 3]
        data = np.array(([1, 2, 3]));
        A[I, J] = data


def read_array():
    ctx = tiledb.Ctx()
    # Open the array and read from it.
    with tiledb.SparseArray(ctx, array_name, mode='r') as A:
        # Slice only rows 1, 2 and cols 2, 3, 4.
        data = A[1:3, 2:5]
        a_vals = data["a"]
        for i, coord in enumerate(data["coords"]):
            print("Cell (%d,%d) has data %d" % (coord[0], coord[1], a_vals[i]))


create_array()
write_array()
read_array()
