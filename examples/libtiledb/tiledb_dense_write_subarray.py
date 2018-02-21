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
    array_ints = np.array([[111, 112],
                           [113, 114]],
                          dtype="int32")

    array_strs = np.array([[b"M", b"NN"],
                           [b"OOO", b"PPPP"]],
                           dtype="S4")

    array_pairs = np.array([[(111.1, 111.2), (112.1, 112.2)],
                            [(113.1, 113.2), (114.1, 114.2)]],
                           dtype="float32,float32")

    ctx = tiledb.Ctx()
    dense_example = tiledb.DenseArray.load(ctx, "my_dense_array")
    dense_example[3:5, 3:5] = {"a1": array_ints,
                        "a2": array_strs,
                        "a3": array_pairs}


if __name__ == '__main__':
    main()