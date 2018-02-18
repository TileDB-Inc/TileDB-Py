#!/usr/bin/python

"""
It creates a hierarchical directory structure with three groups:
     my_group
        |_ dense_arrays
        |_ sparse_arrays

Make sure that no folder with name `my_group` exists in the working
directory before running this example.

Simply run:

    $ python tiledb_group_create.py
"""

import tiledb


def main():
    ctx = tiledb.Ctx()
    tiledb.group_create(ctx, "my_group")
    tiledb.group_create(ctx, "my_group/dense_arrays")
    tiledb.group_create(ctx, "my_group/sparse_arrays")


if __name__ == '__main__':
    main()
