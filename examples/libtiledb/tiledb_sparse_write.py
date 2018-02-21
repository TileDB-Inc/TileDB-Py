#!/usr/bin/python


"""
This example shows how to do a write of a TileDB multiple attribute sparse array.

Simply run:

    $ python tiledb_sparse_create.py
    $ python tiledb_sparse_write.py
"""

import tiledb
import numpy as np


def main():

    # coordinates
    I = [1, 1, 1, 2, 3, 4, 3, 3]
    J = [1, 2, 4, 3, 1, 2, 3, 4]

    # attribute data
    array_ints = np.array([0, 1, 2, 3, 4, 5, 6, 7],
                           dtype="int32")

    array_strs = np.array([b"a", b"bb", b"ccc", b"dddd", b"e", b"ff", b"ggg", b"hhhh"],
                            dtype="S4")

    array_pairs = np.array([(0.1, 0.2), (1.1, 1.2), (2.1, 2.2), (3.1, 3.2),
                            (4.1, 4.2), (5.1, 5.2), (6.1, 6.2), (7.1, 7.2)],
                            dtype="float32, float32")

    ctx = tiledb.Ctx()
    sparse_example = tiledb.SparseArray.load(ctx, "my_sparse_array")
    sparse_example[I, J] = {"a1": array_ints,
                            "a2": array_strs,
                            "a3": array_pairs}


if __name__ == '__main__':
    main()


