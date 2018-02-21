#!/usr/bin/python

"""
This example shows how to do a global read (entire domain) of a TileDB
multiple attribute sparse array.

Simply run:

    $ python tiledb_sparse_create.py
    $ python tiledb_sparse_write.py
    $ python tiledb_sparse_read.py

"""

import tiledb


def main():
    ctx = tiledb.Ctx()
    sparse_example = tiledb.SparseArray.load(ctx, "my_sparse_array")
    nonempty = sparse_example.nonempty_domain()

    # print non-empty domain
    print("Non-empty domain:")
    for i in range(sparse_example.ndim):
        print("{0!s}: {1!r}".format(sparse_example.domain.dim(i).name, nonempty[i]))

    result = sparse_example[:]
    result_num = len(result["coords"])
    print("\nResult num: ", result_num)

    print("{:<10s}{:<5s}{:<10s}{:<10s}{:<10s}".format("coords", "a1", "a2", "a3[0]", "a3[1]"))
    print("----------------------------------------")
    for i in range(result_num):
        print("{:<10s}{:<5d}{:<10s}{:<10.1f}{:<10.1f}"
              .format(result["coords"][i],
                      result["a1"][i],
                      result["a2"][i],
                      result["a3"][i][0],
                      result["a3"][i][1]))
    print()


if __name__ == '__main__':
    main()