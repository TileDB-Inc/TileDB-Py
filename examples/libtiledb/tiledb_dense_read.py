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
    ctx = tiledb.Ctx()
    dense_example = tiledb.DenseArray.load(ctx, "my_dense_array")
    nonempty = dense_example.nonempty_domain()

    # print non-empty domain
    print("Non-empty domain:")
    for i in range(dense_example.ndim):
        print("{0!s}: {1!r}".format(dense_example.domain.dim(i).name, nonempty[i]))

    result = dense_example[:]
    result_num = result["a1"].size
    print("\nResult num: ", result_num)

    attributes = tuple(dense_example.attr(i).name for i in range(dense_example.nattr))
    for attr in attributes:
        print("{0!r}: {1!r}\n".format(attr, result[attr]))


if __name__ == '__main__':
    main()