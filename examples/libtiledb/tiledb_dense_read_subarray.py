#!/usr/bin/python

"""
This example shows how to do a global read (entire domain) of a TileDB
multiple attribute dense array.

Simply run:

    $ python tiledb_dense_create.py
    $ python tiledb_dense_write.py
    $ python tiledb_dense_read.py

"""

import tiledb


def main():
    # Create TileDB context
    ctx = tiledb.Ctx()

    # Load the dense array schema
    dense_example = tiledb.DenseArray.load(ctx, "my_dense_array")

    # Retrieve and print the non-empty domain
    nonempty = dense_example.nonempty_domain()
    print("Non-empty domain:")
    for i in range(dense_example.ndim):
        print("{0!s}: {1!r}".format(dense_example.domain.dim(i).name, nonempty[i]))

    # Read an array slice. `result` is a dict of numpy arrays
    result = dense_example[3:5, 3:5]

    # Print the results
    result_num = result["a1"].size
    print("\nResult num: ", result_num)
    print()
    print("{:<5s}{:<10s}{:<10s}{:<10s}".format("a1", "a2", "a3[0]", "a3[1]"))
    print("------------------------------")

    for i in range(2):
        for j in range(2):
            print("{:<5d}{:<10s}{:<10.1f}{:<10.1f}"
                  .format(result["a1"][i, j],
                          result["a2"][i, j],
                          result["a3"][i, j][0],
                          result["a3"][i, j][1]))
    print()


if __name__ == '__main__':
    main()