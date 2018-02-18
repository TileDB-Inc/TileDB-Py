#!/usr/bin/python

"""
Example shows how to remove a TileDB resource

Simply run:

    $ python tiledb_group_create.py
    $ python tiledb_dense_create.py
    $ python tiledb_object_remove.py
"""

import tiledb


def main():
    ctx = tiledb.Ctx()

    # Delete
    tiledb.remove(ctx, "my_group")
    tiledb.remove(ctx, "my_dense_array")
    try:
        tiledb.remove(ctx, "invalid_path")
    except tiledb.TileDBError:
        print("Failed to delete invalid path")


if __name__ == '__main__':
    main()