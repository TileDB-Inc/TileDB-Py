#!/usr/bin/python

"""
This example shows how to do a global write (entire domain) of a TileDB
multiple attribute dense array.

Simply run:

    $ python tiledb_dense_create.py
    $ python tiledb_dense_write.py

"""

import tiledb
import numpy as np


def main():

    array_ints = np.array([[0, 1, 4, 5],
                           [2, 3, 6, 7],
                           [8, 9, 12, 13],
                           [10, 11, 14, 15]],
                          dtype="int32")

    array_strs = np.array([[b"a", b"bb", b"e", b"ff"],
                           [b"ccc", b"dddd", b"ggg", b"hhhh"],
                           [b"i", b"jj", b"m", b"nn"],
                           [b"kkk", b"llll", b"ooo", b"pppp"]],
                           dtype="S4")

    array_pairs = np.array([[(0.1, 0.2), (1.1, 1.2), (4.1, 4.2), (5.1, 5.2)],
                            [(1.1, 1.2), (3.1, 3.2), (6.1, 6.2), (7.1, 7.2)],
                            [(8.1, 8.2), (9.1, 9.2), (12.1, 12.2), (13.1, 13.2)],
                            [(10.1, 10.2), (11.1, 11.2), (14.1, 14.2), (15.1, 15.2)]],
                           dtype="float32,float32")

    ctx = tiledb.Ctx()
    dense_example = tiledb.DenseArray.load(ctx, "my_dense_array")
    dense_example[:] = {"a1": array_ints,
                        "a2": array_strs,
                        "a3": array_pairs}


if __name__ == '__main__':
    main()