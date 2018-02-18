#!/usr/bin/python

"""
This example shows how to consolidate arrays

One way to make this work:

    $ python tiledb_dense_create.py
    $ python tiledb_dense_write_global_1.py
    $ python tiledb_dense_write_global_subarray.py
    $ python tiledb_array_consolidate.py
"""

import tiledb


def main():
    ctx = tiledb.Ctx()
    tiledb.array_consolidate(ctx, "my_dense_array")


if __name__ == '__main__':
    main()