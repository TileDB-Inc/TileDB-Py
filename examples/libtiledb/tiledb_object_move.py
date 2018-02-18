#!/usr/bin/python

"""
Example shows how to move / rename a TileDB resource

Simply run:

    $ python tiledb_group_create.py
    $ python tiledb_dense_create.py
    $ python tiledb_object_move.py
"""

import tiledb


def main():
    ctx = tiledb.Ctx()

    tiledb.move(ctx, "my_group", "my_group_2")
    tiledb.move(ctx, "my_dense_array", "my_group_2/dense_arrays/my_dense_array")

    try:
        tiledb.move(ctx, "invalid_path", "path", force=False)
    except tiledb.TileDBError:
        print("Failed moving invalid path")


if __name__ == '__main__':
    main()