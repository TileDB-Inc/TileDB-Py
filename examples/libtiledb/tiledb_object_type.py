#!/usr/bin/python

"""
Example shows how to get the type of a TileDB object (resource).

 You need to run the following to make this work:

    $ python tiledb_group_create.py
    $ python tiledb_dense_create.py
    $ python tiledb_kv.py
    $ python tiledb_object_type.py
"""

import tiledb


def main():
    ctx = tiledb.Ctx()
    print("{!r}".format(tiledb.object_type(ctx, "my_group")))
    print("{!r}".format(tiledb.object_type(ctx, "my_dense_array")))
    print("{!r}".format(tiledb.object_type(ctx, "my_kv")))
    print("{!r}".format(tiledb.object_type(ctx, "invalid_path")))


if __name__ == '__main__':
    main()